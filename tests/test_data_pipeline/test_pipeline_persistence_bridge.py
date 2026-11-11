"""TDD unit and integration tests for the Module 3/4 -> Module 2 persistence bridge."""

from __future__ import annotations

import ast
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pytest
import sqlalchemy as sa
from astropy.io import fits
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from LUMIA.core.schemas import CandidateDetection
from LUMIA.data_pipeline.bridge import (
    EpochPersistenceContext,
    OLAPCandidateDetectionRecord,
    OLAPCutoutStampRecord,
    OLAPEpochPersistResult,
    PipelineJobCreateSpec,
    PipelineJobRecord,
    PipelineTaskCreateSpec,
    PipelineTaskLogRecord,
    PipelineTaskRecord,
    TelescopeProfileRecord,
    ZOGYEpochPersistenceBundle,
    build_zogy_persistence_bundle,
    fetch_olap_cutout_stamps,
    fetch_olap_detections_for_job,
    get_pipeline_job,
    get_telescope_profile,
    list_pipeline_task_logs,
    map_astrometry_photometry_to_detection,
    persist_olap_cutout_stamps,
    persist_olap_detections,
    persist_zogy_bundle_to_olap,
    register_pipeline_job,
    register_pipeline_task,
    transition_pipeline_job,
    transition_pipeline_task,
    upsert_telescope_profile,
)
from LUMIA.data_pipeline.database import (
    CatalogDatabase,
    CatalogQueryParams,
    DuckDBConfig,
    StellarCatalogRecord,
)
from LUMIA.data_pipeline.repositories import CandidateDetectionRepository
from LUMIA.data_pipeline.schema_olap import OLAPSchemaConfig, initialize_olap_schema
from LUMIA.data_pipeline.schema_oltp import (
    OLTPDatabaseConfig,
    PipelineJobORM,
    PipelineTaskORM,
    create_oltp_engine,
    initialize_oltp_schema,
)
from LUMIA.physics.astrometry import (
    AstrometricCalibrationConfig,
    AstrometricMeasurement,
    BarycentricTimeResult,
    compute_bjd_tdb,
    propagate_astrometric_uncertainty,
    sky_to_pixel,
)
from LUMIA.physics.config import ObservatoryLocation, TelescopeConfig
from LUMIA.physics.cutout_extraction import (
    CutoutConfig,
    PeakExtractionConfig,
    extract_zogy_cutout_batch,
)
from LUMIA.physics.gpu_processing import (
    GPUProcessingConfig,
    ZOGYConfig,
    ZOGYResult,
    zogy_subtract,
)
from LUMIA.physics.photometry import (
    AperturePhotometryMeasurement,
    PhotometryConfig,
    compute_zero_point,
    perform_aperture_photometry,
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _make_tan_wcs_header(
    *,
    ra0_deg: float = 180.0,
    dec0_deg: float = -15.0,
    scale_arcsec_px: float = 1.0,
    width: int = 128,
    height: int = 128,
) -> fits.Header:
    """Create a deterministic TAN-projected celestial WCS FITS header."""
    scale_deg = scale_arcsec_px / 3600.0
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = width
    header["NAXIS2"] = height
    header["CRPIX1"] = width / 2.0
    header["CRPIX2"] = height / 2.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = ra0_deg
    header["CRVAL2"] = dec0_deg
    header["CD1_1"] = -scale_deg
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = scale_deg
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["RADESYS"] = "ICRS"
    return header


def _make_gaussian_psf(size: int = 15, sigma_px: float = 1.6) -> np.ndarray:
    """Build a normalized 2D circular Gaussian PSF array."""
    center = size // 2
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    r2 = (x_grid - center) ** 2 + (y_grid - center) ** 2
    psf = np.exp(-0.5 * r2 / (sigma_px**2))
    psf /= np.sum(psf)
    return psf.astype(np.float32)


def _inject_gaussian_source(
    image: np.ndarray,
    x0: float,
    y0: float,
    total_flux_adu: float,
    sigma_px: float = 1.6,
) -> None:
    """Inject an analytical 2D circular Gaussian point source into ``image``."""
    ny, nx = image.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    norm = total_flux_adu / (2.0 * math.pi * sigma_px**2)
    r2 = (x_grid - x0) ** 2 + (y_grid - y0) ** 2
    image += (norm * np.exp(-0.5 * r2 / (sigma_px**2))).astype(image.dtype)


def _sample_telescope_config() -> TelescopeConfig:
    return TelescopeConfig(
        name="Palomar 48-inch ZTF Bridge Test",
        pixel_scale=1.01,
        gain=6.2,
        read_noise=10.3,
        saturation_limit=55000.0,
        dark_current=0.02,
        aperture_diameter_m=1.22,
        focal_length_m=3.07,
        observatory=ObservatoryLocation(
            latitude_deg=33.3563,
            longitude_deg=-116.8648,
            elevation_m=1712.0,
            mpc_code="I41",
        ),
    )


@pytest.fixture
def sqlite_oltp_engine() -> sa.Engine:
    """Provide an isolated in-memory SQLite engine initialized with all 3NF OLTP tables."""
    engine = create_oltp_engine(
        OLTPDatabaseConfig(url="sqlite+pysqlite:///:memory:", echo=False)
    )
    initialize_oltp_schema(engine)
    return engine


# ---------------------------------------------------------------------------
# 1. Strict ADR-0002 Contracts & Pure Domain-to-Persistence Mappers
# ---------------------------------------------------------------------------


class TestBridgePydanticContractsAndPureMappers:
    """Verify strict Pydantic v2 bridge contracts and pure mapping functions."""

    BRIDGE_MODELS: tuple[type[BaseModel], ...] = (
        EpochPersistenceContext,
        OLAPCandidateDetectionRecord,
        OLAPCutoutStampRecord,
        ZOGYEpochPersistenceBundle,
        OLAPEpochPersistResult,
        TelescopeProfileRecord,
        PipelineJobCreateSpec,
        PipelineJobRecord,
        PipelineTaskCreateSpec,
        PipelineTaskRecord,
        PipelineTaskLogRecord,
    )

    @pytest.mark.parametrize("model_cls", BRIDGE_MODELS, ids=lambda c: c.__name__)
    def test_all_bridge_models_enforce_strict_frozen_forbid_finite_policy(
        self,
        model_cls: type[BaseModel],
    ) -> None:
        cfg = model_cls.model_config
        assert cfg.get("strict") is True
        assert cfg.get("extra") == "forbid"
        assert cfg.get("frozen") is True
        assert cfg.get("allow_inf_nan") is False

    def test_epoch_context_rejects_path_traversal_and_non_strict_types(self) -> None:
        with pytest.raises(ValidationError):
            EpochPersistenceContext(
                job_id="../bad",
                field_id="ZTF_0412",
                observation_night=date(2026, 11, 7),
            )
        with pytest.raises(ValidationError):
            EpochPersistenceContext(
                job_id="job-01",
                field_id="ZTF..0412",
                observation_night=date(2026, 11, 7),
            )
        with pytest.raises(ValidationError):
            EpochPersistenceContext(
                job_id="job-01",
                field_id="ZTF_0412",
                observation_night=date(2026, 11, 7),
                epoch_index="0",  # type: ignore[arg-type]
            )

    def test_pure_map_astrometry_photometry_to_detection_and_core_projection(
        self,
    ) -> None:
        ctx = EpochPersistenceContext(
            job_id="job-m3-001",
            field_id="ZTF_0412",
            observation_night=date(2026, 11, 7),
            epoch_index=1,
        )
        astro = AstrometricMeasurement(
            ra_deg=180.25,
            dec_deg=-14.75,
            sigma_ra_cosdec_arcsec=0.06,
            sigma_dec_arcsec=0.08,
        )
        bjd = BarycentricTimeResult(
            jd_utc=2460755.5000,
            bjd_tdb=2460755.5032,
            light_travel_time_s=276.48,
        )
        phot = AperturePhotometryMeasurement(
            x_px=63.5,
            y_px=64.25,
            aperture_sum_adu=1800.0,
            sky_background_per_px_adu=10.0,
            net_flux_adu=1300.0,
            flux_err_adu=26.0,
            instrumental_mag=-7.78,
            instrumental_mag_err=0.0217,
            calibrated_mag=19.72,
            calibrated_mag_err=0.031,
        )

        rec = map_astrometry_photometry_to_detection(
            candidate_id="job-m3-001_e1_c0",
            context=ctx,
            astrometry=astro,
            barycentric_time=bjd,
            photometry=phot,
            cnn_score=0.94,
            model_version="lumia-rb-1.0",
        )

        assert rec.astrometric_uncertainty_arcsec == pytest.approx(0.10)
        assert rec.mag == pytest.approx(19.72)
        assert rec.mag_err == pytest.approx(0.031)
        assert rec.snr == pytest.approx(50.0)
        assert rec.x_centroid_px == pytest.approx(63.5)
        assert rec.y_centroid_px == pytest.approx(64.25)

        core_det: CandidateDetection = rec.to_candidate_detection()
        assert core_det.candidate_id == "job-m3-001_e1_c0"
        assert core_det.field_id == "ZTF_0412"
        assert core_det.ra_deg == pytest.approx(180.25)
        assert core_det.dec_deg == pytest.approx(-14.75)
        assert core_det.bjd_tdb == pytest.approx(2460755.5032)
        assert core_det.mag == pytest.approx(19.72)
        assert core_det.snr == pytest.approx(50.0)
        assert core_det.cnn_score == pytest.approx(0.94)
        assert core_det.model_version == "lumia-rb-1.0"
        assert core_det.astrometric_uncertainty_arcsec == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 2. Module 3 (Astrometry + Photometry) -> Module 2 Persistence Bridge
# ---------------------------------------------------------------------------


class TestModule3AstrometryAndPhotometryPersistenceBridge:
    """Verify real Module 3 WCS, BJD, and Photutils outputs persist into Module 2 OLAP."""

    def test_end_to_end_astrometry_and_calibrated_photometry_persist_to_duckdb(
        self,
    ) -> None:
        header = _make_tan_wcs_header(ra0_deg=180.0, dec0_deg=-15.0, scale_arcsec_px=1.0)
        tel_cfg = _sample_telescope_config()
        phot_cfg = PhotometryConfig.from_telescope_config(
            tel_cfg,
            aperture_radius_px=4.0,
            annulus_r_in_px=7.0,
            annulus_r_out_px=11.0,
            exposure_time_s=30.0,
        )

        # Synthetic 128x128 science image with 2 catalog calibration stars + 1 moving object
        rng = np.random.default_rng(101)
        science_img = rng.normal(50.0, 1.5, size=(128, 128)).astype(np.float64)

        with CatalogDatabase(DuckDBConfig(database=":memory:")) as cat_db:
            cat_db.create_stellar_catalog_table("ref_stars")
            ref_records = [
                StellarCatalogRecord(
                    source_id=1001,
                    ra_deg=180.005,
                    dec_deg=-15.005,
                    mag_g=16.5,
                ),
                StellarCatalogRecord(
                    source_id=1002,
                    ra_deg=179.995,
                    dec_deg=-14.994,
                    mag_g=17.2,
                ),
            ]
            cat_db.insert_stellar_records(ref_records, table_name="ref_stars")
            queried_stars = cat_db.query_stellar_catalog(
                CatalogQueryParams(
                    ra_min_deg=179.9,
                    ra_max_deg=180.1,
                    dec_min_deg=-15.1,
                    dec_max_deg=-14.9,
                    max_mag=18.0,
                ),
                table_name="ref_stars",
            )
            assert len(queried_stars) == 2

            star_x, star_y = sky_to_pixel(
                header,
                queried_stars["ra_deg"],
                queried_stars["dec_deg"],
            )
            # Inject reference stars with fluxes consistent with ZP = 25.0 at t_exp = 30s
            for sx, sy, smag in zip(
                star_x,
                star_y,
                queried_stars["mag_g"],
                strict=True,
            ):
                flux_adu = 30.0 * (10.0 ** (-0.4 * (float(smag) - 25.0)))
                _inject_gaussian_source(science_img, float(sx), float(sy), flux_adu)

            # Inject moving-object transient at (82.25, 44.75)
            target_x, target_y = 82.25, 44.75
            _inject_gaussian_source(science_img, target_x, target_y, 2400.0)

            ref_phot = perform_aperture_photometry(
                science_img,
                np.column_stack((star_x, star_y)),
                phot_cfg,
            )
            zp_res = compute_zero_point(
                flux_adu=[m.net_flux_adu for m in ref_phot],
                flux_err_adu=[m.flux_err_adu for m in ref_phot],
                catalog_mag=queried_stars["mag_g"],
                exposure_time_s=phot_cfg.exposure_time_s,
            )

            astro_meas = propagate_astrometric_uncertainty(
                header,
                x_px=target_x,
                y_px=target_y,
                sigma_x_px=0.08,
                sigma_y_px=0.09,
                calibration=AstrometricCalibrationConfig(
                    wcs_rms_ra_arcsec=0.05,
                    wcs_rms_dec_arcsec=0.05,
                ),
            )
            bjd_res = compute_bjd_tdb(
                "2026-11-07T06:30:00",
                astro_meas.ra_deg,
                astro_meas.dec_deg,
                tel_cfg.observatory,
                exposure_time_s=phot_cfg.exposure_time_s,
                timestamp_is_start=True,
            )
            cand_phot = perform_aperture_photometry(
                science_img,
                [(target_x, target_y)],
                phot_cfg,
                zero_point=zp_res,
            )[0]

            initialize_olap_schema(cat_db)
            ctx = EpochPersistenceContext(
                job_id="job-astro-phot-01",
                field_id="ZTF_0412",
                observation_night=date(2026, 11, 7),
                epoch_index=0,
            )
            olap_rec = map_astrometry_photometry_to_detection(
                candidate_id="job-astro-phot-01_e0_c0",
                context=ctx,
                astrometry=astro_meas,
                barycentric_time=bjd_res,
                photometry=cand_phot,
            )
            inserted = persist_olap_detections(cat_db, [olap_rec])
            assert inserted == 1

            fetched_list = fetch_olap_detections_for_job(cat_db, "job-astro-phot-01")
            assert len(fetched_list) == 1
            fetched = fetched_list[0]
            assert fetched == olap_rec
            assert fetched.mag is not None and 17.0 < fetched.mag < 21.0
            assert fetched.net_flux_adu == pytest.approx(cand_phot.net_flux_adu)

            # Also verify seamless read compatibility via CandidateDetectionRepository
            legacy_repo = CandidateDetectionRepository(cat_db)
            via_repo = legacy_repo.get_by_id("job-astro-phot-01_e0_c0")
            assert via_repo == olap_rec.to_candidate_detection()


# ---------------------------------------------------------------------------
# 3. Module 4 (GPU ZOGY + Cutouts) -> DuckDB & Hive Parquet Persistence
# ---------------------------------------------------------------------------


class TestModule4ZOGYAndCutoutPersistenceBridge:
    """Verify GPU-optimized ZOGY outputs and 3-channel stamps persist to DuckDB/Parquet."""

    def test_zogy_subtract_and_cutouts_persist_to_duckdb_and_hive_parquet(
        self,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(2026)
        science = rng.normal(100.0, 1.2, size=(128, 128)).astype(np.float32)
        reference = rng.normal(100.0, 1.2, size=(128, 128)).astype(np.float32)

        # Stationary star cancels in ZOGY; two moving transients appear only in science
        _inject_gaussian_source(science, 64.0, 64.0, total_flux_adu=900.0)
        _inject_gaussian_source(reference, 64.0, 64.0, total_flux_adu=900.0)
        _inject_gaussian_source(science, 42.0, 54.0, total_flux_adu=260.0)
        _inject_gaussian_source(science, 90.0, 82.0, total_flux_adu=170.0)

        psf = _make_gaussian_psf(size=15, sigma_px=1.6)
        zogy_res: ZOGYResult = zogy_subtract(
            science,
            reference,
            psf,
            psf,
            ZOGYConfig(
                flux_scale_science=1.0,
                flux_scale_reference=1.0,
                noise_std_science=1.2,
                noise_std_reference=1.2,
            ),
            GPUProcessingConfig(device="cpu", deterministic=True, seed=42),
        )
        cutout_batch = extract_zogy_cutout_batch(
            science_frame=science,
            reference_frame=reference,
            zogy_result=zogy_res,
            peak_config=PeakExtractionConfig(
                snr_threshold=6.0,
                nms_radius_px=3,
                id_prefix="job-zogy-01_e0",
            ),
            cutout_config=CutoutConfig(stamp_size=31, boundary_mode="pad"),
        )
        assert len(cutout_batch.candidates) == 2

        header = _make_tan_wcs_header(ra0_deg=210.0, dec0_deg=12.0, scale_arcsec_px=1.0)
        tel_cfg = _sample_telescope_config()
        phot_cfg = PhotometryConfig.from_telescope_config(
            tel_cfg,
            aperture_radius_px=4.0,
            annulus_r_in_px=6.5,
            annulus_r_out_px=10.0,
            exposure_time_s=30.0,
        )
        zp = compute_zero_point(
            flux_adu=[1500.0, 3000.0],
            flux_err_adu=[25.0, 35.0],
            catalog_mag=[19.0, 18.247],
            exposure_time_s=30.0,
        )

        positions = [(p.x_centroid_px, p.y_centroid_px) for p in cutout_batch.candidates]
        phot_list = perform_aperture_photometry(
            science,
            positions,
            phot_cfg,
            zero_point=zp,
        )
        astro_list = [
            propagate_astrometric_uncertainty(
                header,
                x_px=p.x_centroid_px,
                y_px=p.y_centroid_px,
                sigma_x_px=0.10,
                sigma_y_px=0.10,
            )
            for p in cutout_batch.candidates
        ]
        bjd_epoch = compute_bjd_tdb(
            "2026-11-07T08:15:00",
            210.0,
            12.0,
            tel_cfg.observatory,
            exposure_time_s=30.0,
        )

        ctx = EpochPersistenceContext(
            job_id="job-zogy-01",
            field_id="ZTF_0888",
            observation_night=date(2026, 11, 7),
            epoch_index=0,
        )
        bundle = build_zogy_persistence_bundle(
            context=ctx,
            zogy_result=zogy_res,
            cutout_batch=cutout_batch,
            astrometry_measurements=astro_list,
            barycentric_times=bjd_epoch,
            photometry_measurements=phot_list,
            zero_point=zp,
        )
        assert len(bundle.detections) == 2
        assert len(bundle.cutout_stamps) == 2
        assert bundle.theoretical_noise_std == pytest.approx(
            zogy_res.theoretical_noise_std
        )

        # Length mismatch in build_zogy_persistence_bundle must fail fast
        with pytest.raises(ValueError, match="length"):
            build_zogy_persistence_bundle(
                context=ctx,
                zogy_result=zogy_res,
                cutout_batch=cutout_batch,
                astrometry_measurements=astro_list[:1],
                barycentric_times=bjd_epoch,
            )

        olap_cfg = OLAPSchemaConfig(
            parquet_root_dir=tmp_path / "olap_store",
            compression="ZSTD",
            row_group_size=100_000,
        )
        with CatalogDatabase(DuckDBConfig(database=":memory:")) as cat_db:
            persist_res = persist_zogy_bundle_to_olap(
                cat_db,
                bundle,
                olap_config=olap_cfg,
                export_parquet=True,
            )
            assert persist_res.persisted_detection_count == 2
            assert persist_res.persisted_cutout_count == 2
            assert persist_res.detections_parquet_path is not None
            assert persist_res.cutouts_parquet_path is not None
            assert persist_res.detections_parquet_path.is_file()
            assert persist_res.cutouts_parquet_path.is_file()

            # Round-trip stamps from DuckDB FLOAT[][] columns
            first_cid = bundle.detections[0].candidate_id
            stamps = fetch_olap_cutout_stamps(cat_db, first_cid)
            assert len(stamps) == 1
            assert stamps[0].cutout_size == 31
            assert np.allclose(
                np.asarray(stamps[0].science_pixels),
                cutout_batch.triplets[0, 0].numpy(),
                atol=1e-4,
            )
            assert stamps[0].difference_pixels is not None
            assert np.allclose(
                np.asarray(stamps[0].difference_pixels),
                cutout_batch.triplets[0, 2].numpy(),
                atol=1e-4,
            )

            # Convert to repository stamp contract cleanly
            repo_stamp = stamps[0].to_stamp_record()
            assert repo_stamp.candidate_id == first_cid
            assert repo_stamp.cutout_size == 31

        # Verify Parquet files hydrate cleanly in an independent DuckDB reader
        reader = duckdb.connect(":memory:")
        try:
            det_rows = reader.execute(
                "SELECT candidate_id, snr FROM read_parquet(?, hive_partitioning=true)",
                [persist_res.detections_parquet_path.as_posix()],
            ).fetchall()
            assert len(det_rows) == 2
            assert float(det_rows[0][1]) >= float(det_rows[1][1])

            cut_rows = reader.execute(
                "SELECT candidate_id, cutout_size, len(science_pixels) "
                "FROM read_parquet(?, hive_partitioning=true)",
                [persist_res.cutouts_parquet_path.as_posix()],
            ).fetchall()
            assert len(cut_rows) == 2
            assert int(cut_rows[0][1]) == 31
            assert int(cut_rows[0][2]) == 31
        finally:
            reader.close()

    def test_persist_empty_detections_and_stamps_are_noops(self) -> None:
        with CatalogDatabase(DuckDBConfig(database=":memory:")) as cat_db:
            initialize_olap_schema(cat_db)
            assert persist_olap_detections(cat_db, []) == 0
            assert persist_olap_cutout_stamps(cat_db, []) == 0


# ---------------------------------------------------------------------------
# 4. PostgreSQL / SQLite 3NF OLTP Job & Task Lifecycle Persistence
# ---------------------------------------------------------------------------


class TestOLTPJobAndTaskLifecyclePersistence:
    """Verify PipelineJobORM, PipelineTaskORM, and TelescopeProfileORM lifecycle updates."""

    def test_telescope_profile_3nf_upsert_and_reconstruction(
        self,
        sqlite_oltp_engine: sa.Engine,
    ) -> None:
        tel_cfg = _sample_telescope_config()
        saved = upsert_telescope_profile(sqlite_oltp_engine, "ZTF_P48", tel_cfg)
        assert saved.profile_id == "ZTF_P48"
        assert saved.telescope == tel_cfg

        fetched = get_telescope_profile(sqlite_oltp_engine, "ZTF_P48")
        assert fetched is not None
        assert fetched.telescope == tel_cfg
        assert get_telescope_profile(sqlite_oltp_engine, "MISSING_TEL") is None

    def test_pipeline_job_and_task_lifecycle_transitions_in_oltp_tables(
        self,
        sqlite_oltp_engine: sa.Engine,
    ) -> None:
        tel_cfg = _sample_telescope_config()
        upsert_telescope_profile(sqlite_oltp_engine, "ZTF_P48", tel_cfg)

        job_rec = register_pipeline_job(
            sqlite_oltp_engine,
            PipelineJobCreateSpec(
                job_id="job-lifecycle-001",
                profile_id="ZTF_P48",
                target_field_id="ZTF_0412",
                observation_night_utc=date(2026, 11, 7),
                ra_min_deg=180.0,
                ra_max_deg=181.5,
                dec_min_deg=-15.5,
                dec_max_deg=-14.0,
                max_mag=21.5,
            ),
        )
        assert job_rec.status == "queued"
        assert job_rec.started_at_utc is None
        assert job_rec.completed_at_utc is None

        # Start job -> status='running'
        running_job = transition_pipeline_job(
            sqlite_oltp_engine,
            "job-lifecycle-001",
            new_status="running",
            matched_catalog_sources=142,
            brightest_catalog_mag_g=13.4,
        )
        assert running_job.status == "running"
        assert running_job.started_at_utc is not None
        assert running_job.matched_catalog_sources == 142
        assert running_job.brightest_catalog_mag_g == pytest.approx(13.4)

        # Register Module 3 and Module 4 tasks
        m3_task = register_pipeline_task(
            sqlite_oltp_engine,
            PipelineTaskCreateSpec(
                job_id="job-lifecycle-001",
                task_name="astrometry-and-photometry",
                module_number=3,
                max_retries=2,
            ),
        )
        m4_task = register_pipeline_task(
            sqlite_oltp_engine,
            PipelineTaskCreateSpec(
                job_id="job-lifecycle-001",
                task_name="gpu-zogy-and-cutouts",
                module_number=4,
                max_retries=2,
            ),
        )
        assert m3_task.status == "pending"
        assert m3_task.attempt_count == 1

        # Module 3 task: pending -> running -> completed
        m3_running = transition_pipeline_task(
            sqlite_oltp_engine,
            m3_task.task_run_id,
            new_status="running",
            log_severity="INFO",
            log_message="Starting WCS plate-solve and Photutils zero-point.",
            event_code="M3_START",
        )
        assert m3_running.status == "running"
        assert m3_running.started_at_utc is not None

        m3_done = transition_pipeline_task(
            sqlite_oltp_engine,
            m3_task.task_run_id,
            new_status="completed",
            log_severity="INFO",
            log_message="Calibrated 2 transient candidates.",
            event_code="M3_DONE",
        )
        assert m3_done.status == "completed"
        assert m3_done.finished_at_utc is not None

        # Module 4 task: pending -> running -> retrying -> running (attempt 2) -> completed
        transition_pipeline_task(
            sqlite_oltp_engine,
            m4_task.task_run_id,
            new_status="running",
        )
        retry_at = datetime.now(UTC) + timedelta(seconds=5)
        m4_retrying = transition_pipeline_task(
            sqlite_oltp_engine,
            m4_task.task_run_id,
            new_status="retrying",
            next_retry_at_utc=retry_at,
            log_severity="WARNING",
            log_message="Transient CUDA contention; scheduling retry.",
            event_code="M4_RETRY",
        )
        assert m4_retrying.status == "retrying"
        assert m4_retrying.next_retry_at_utc is not None

        m4_attempt2 = transition_pipeline_task(
            sqlite_oltp_engine,
            m4_task.task_run_id,
            new_status="running",
            log_severity="INFO",
            log_message="Attempt 2 running ZOGY FFT.",
        )
        assert m4_attempt2.attempt_count == 2
        assert m4_attempt2.next_retry_at_utc is None

        m4_done = transition_pipeline_task(
            sqlite_oltp_engine,
            m4_task.task_run_id,
            new_status="completed",
            log_severity="INFO",
            log_message="ZOGY subtraction and 31x31 cutouts persisted.",
        )
        assert m4_done.status == "completed"
        assert m4_done.finished_at_utc is not None

        # Complete job with Parquet partition URI
        parquet_uri = (
            "data/olap/candidate_detections/"
            "observation_night=2026-11-07/field_id=ZTF_0412/job_id=job-lifecycle-001.parquet"
        )
        completed_job = transition_pipeline_job(
            sqlite_oltp_engine,
            "job-lifecycle-001",
            new_status="completed",
            parquet_partition_uri=parquet_uri,
        )
        assert completed_job.status == "completed"
        assert completed_job.completed_at_utc is not None
        assert completed_job.parquet_partition_uri == parquet_uri

        # Verify underlying SQLAlchemy ORM rows in PipelineJobORM and PipelineTaskORM
        with Session(sqlite_oltp_engine) as session:
            orm_job = session.get(PipelineJobORM, "job-lifecycle-001")
            assert orm_job is not None
            assert orm_job.status == "completed"
            assert orm_job.parquet_partition_uri == parquet_uri

            orm_tasks = session.scalars(
                sa.select(PipelineTaskORM)
                .where(PipelineTaskORM.job_id == "job-lifecycle-001")
                .order_by(PipelineTaskORM.module_number.asc())
            ).all()
            assert [t.module_number for t in orm_tasks] == [3, 4]
            assert [t.status for t in orm_tasks] == ["completed", "completed"]
            assert [t.attempt_count for t in orm_tasks] == [1, 2]

        # Verify structured task logs were persisted in order
        m4_logs = list_pipeline_task_logs(sqlite_oltp_engine, m4_task.task_run_id)
        assert len(m4_logs) == 3
        assert [lg.severity for lg in m4_logs] == ["WARNING", "INFO", "INFO"]
        assert [lg.attempt_number for lg in m4_logs] == [1, 2, 2]

        # Terminal job & task states must reject illegal re-transitions
        with pytest.raises(ValueError, match="transition"):
            transition_pipeline_job(
                sqlite_oltp_engine,
                "job-lifecycle-001",
                new_status="running",
            )
        with pytest.raises(ValueError, match="transition"):
            transition_pipeline_task(
                sqlite_oltp_engine,
                m4_task.task_run_id,
                new_status="running",
            )

    def test_failed_job_records_error_summary(
        self,
        sqlite_oltp_engine: sa.Engine,
    ) -> None:
        register_pipeline_job(
            sqlite_oltp_engine,
            PipelineJobCreateSpec(
                job_id="job-fail-001",
                target_field_id="ZTF_FAIL",
                observation_night_utc=date(2026, 11, 7),
                ra_min_deg=10.0,
                ra_max_deg=11.0,
                dec_min_deg=5.0,
                dec_max_deg=6.0,
                max_mag=20.5,
            ),
        )
        transition_pipeline_job(
            sqlite_oltp_engine,
            "job-fail-001",
            new_status="running",
        )
        failed = transition_pipeline_job(
            sqlite_oltp_engine,
            "job-fail-001",
            new_status="failed",
            error_summary="Singular WCS matrix in exposure 2.",
        )
        assert failed.status == "failed"
        assert failed.error_summary == "Singular WCS matrix in exposure 2."
        assert failed.completed_at_utc is not None
        assert get_pipeline_job(sqlite_oltp_engine, "job-fail-001") == failed


# ---------------------------------------------------------------------------
# 5. Unidirectional Dependency & Database Driver Isolation
# ---------------------------------------------------------------------------


class TestUnidirectionalDependencyAndDriverIsolation:
    """Verify physics remains pure and no database drivers leak outside data_pipeline."""

    def test_physics_never_imports_data_pipeline_or_database_drivers(self) -> None:
        physics_dir = _REPO_ROOT / "src" / "LUMIA" / "physics"
        forbidden = ("duckdb", "sqlalchemy", "LUMIA.data_pipeline", "LUMIA.api")
        violations: list[str] = []

        for py_file in sorted(physics_dir.rglob("*.py")):
            rel = py_file.relative_to(_REPO_ROOT)
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(
                            alias.name == p or alias.name.startswith(f"{p}.")
                            for p in forbidden
                        ):
                            violations.append(f"{rel}:{node.lineno} -> {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if any(mod == p or mod.startswith(f"{p}.") for p in forbidden):
                        violations.append(f"{rel}:{node.lineno} -> {mod}")

        assert violations == []

    def test_no_database_drivers_leak_outside_data_pipeline(self) -> None:
        lumia_dir = _REPO_ROOT / "src" / "LUMIA"
        dp_dir = (lumia_dir / "data_pipeline").resolve()
        violations: list[str] = []

        for py_file in sorted(lumia_dir.rglob("*.py")):
            if py_file.resolve().is_relative_to(dp_dir):
                continue
            rel = py_file.relative_to(_REPO_ROOT)
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in {"duckdb", "sqlalchemy"} or alias.name.startswith(
                            ("duckdb.", "sqlalchemy.")
                        ):
                            violations.append(f"{rel}:{node.lineno} -> {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod in {"duckdb", "sqlalchemy"} or mod.startswith(
                        ("duckdb.", "sqlalchemy.")
                    ):
                        violations.append(f"{rel}:{node.lineno} -> {mod}")

        assert violations == []
