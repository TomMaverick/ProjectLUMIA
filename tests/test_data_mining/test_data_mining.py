"""Unit tests for the Project LUMIA data-mining and augmentation subsystem."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003

import numpy as np

from LUMIA.data_mining.archive_crawler import (
    CutoutTriplet,
    WCSMetadata,
    compute_fixed_sky_midpoint,
    partition_tracklet_sliding_window,
    slice_subpixel_cutout,
    verify_wcs_containment_and_quality,
)
from LUMIA.data_mining.artifact_filter import (
    ARTIFACT_TAXONOMY_MAP,
    ArtifactFilterConfig,
    DeepStreaksArtifactFilter,
)
from LUMIA.data_mining.augmentation import (
    FaintSyntheticConfig,
    FaintSyntheticInjector,
    apply_d4_transform,
    apply_d4_transform_with_coords,
    apply_spatial_crop_jitter,
    transform_coordinates_d4,
)
from LUMIA.data_mining.pipeline import (
    DataMiningPipelineConfig,
    SolarSystemDataMiningPipeline,
)
from LUMIA.data_mining.storage import (
    HDF5TrainingStorageManager,
    TrackletSample,
)
from LUMIA.data_mining.taxonomy import (
    EphemerisPoint,
    HorizonsClient,
    HorizonsQueryConfig,
    TargetObjectSpec,
    compute_optimal_cadence_window,
)


def test_target_taxonomy_and_cadence_window() -> None:
    """Verify adaptive cadence calculation across solar system taxonomy classes."""
    # Fast NEO (8 arcsec/min) requires short window ~10-20 min
    neo_dt = compute_optimal_cadence_window("NEO", angular_speed_arcsec_min=8.0)
    assert 10.0 <= neo_dt <= 20.0

    # Main Belt Asteroid (0.5 arcsec/min) requires ~25-50 min
    mba_dt = compute_optimal_cadence_window("MBA", angular_speed_arcsec_min=0.5)
    assert 25.0 <= mba_dt <= 50.0

    # Distant TNO (0.04 arcsec/min) requires 180 to 1440 min (3 to 24 hours)
    tno_dt = compute_optimal_cadence_window("TNO", angular_speed_arcsec_min=0.04)
    assert 180.0 <= tno_dt <= 1440.0

    # Comet requires 20 to 45 min
    comet_dt = compute_optimal_cadence_window("COMET", angular_speed_arcsec_min=1.2)
    assert 20.0 <= comet_dt <= 45.0


def test_horizons_client_fallback_and_bjd() -> None:
    """Verify ephemeris generation and BJD_TDB computation."""
    client = HorizonsClient()
    target = TargetObjectSpec(
        target_id="99942",
        target_name="Apophis",
        taxonomy_class="NEO",
    )
    now = datetime.now(UTC)
    config = HorizonsQueryConfig(
        start_time=now,
        end_time=now + timedelta(hours=1),
        step_minutes=15,
    )
    points = client.query_ephemeris(target, config)
    assert len(points) >= 4
    for pt in points:
        assert pt.target_id == "99942"
        assert 0.0 <= pt.ra_deg < 360.0
        assert -90.0 <= pt.dec_deg <= 90.0
        assert pt.bjd_tdb > 2450000.0  # Plausible Julian Date
        assert pt.total_speed_arcsec_min > 0.0


def test_subpixel_cutout_slicing() -> None:
    """Verify in-memory 2D sub-pixel cutout slicing and padding."""
    full_image = np.ones((500, 500), dtype=np.float32) * 50.0
    full_image[250, 250] = 999.0

    cutout_128 = slice_subpixel_cutout(full_image, center_x=250.0, center_y=250.0, stamp_size=128)
    assert cutout_128.shape == (128, 128)
    assert cutout_128[64, 64] == 999.0

    # Edge test (near boundary)
    edge_cutout = slice_subpixel_cutout(full_image, center_x=5.0, center_y=5.0, stamp_size=128)
    assert edge_cutout.shape == (128, 128)
    assert np.all(np.isfinite(edge_cutout))


def test_wcs_containment_and_quality_verification() -> None:
    """Verify strict 3-stage WCS containment (12 <= x, y < 116) and NaN quality check."""
    clean_image = np.ones((128, 128), dtype=np.float32) * 100.0

    # 1. Clean center point passes
    is_valid, reason = verify_wcs_containment_and_quality(
        clean_image, target_x=64.0, target_y=64.0, stamp_size=128, margin_px=12
    )
    assert is_valid
    assert reason == "OK"

    # 2. Border margin violation (x=8 < 12) fails
    is_valid_x, reason_x = verify_wcs_containment_and_quality(
        clean_image, target_x=8.0, target_y=64.0, stamp_size=128, margin_px=12
    )
    assert not is_valid_x
    assert "OUT_OF_BOUNDS" in reason_x

    # 3. NaN in 5x5 neighborhood fails
    nan_image = clean_image.copy()
    nan_image[63, 65] = np.nan
    is_valid_nan, reason_nan = verify_wcs_containment_and_quality(
        nan_image, target_x=64.0, target_y=64.0, stamp_size=128, margin_px=12
    )
    assert not is_valid_nan
    assert "NAN_OR_MASKED_CCD" in reason_nan


def test_fixed_sky_midpoint_and_sliding_window_partition() -> None:
    """Verify trajectory midpoint calculation and sub-tracklet sliding window partitioning."""
    now = datetime.now(UTC)
    points: list[EphemerisPoint] = []
    # Create 12 points spanning 150 pixels of motion (> safe interior of a single 128x128 stamp)
    scale_deg_per_px = 1.012 / 3600.0
    for i in range(12):
        points.append(
            EphemerisPoint(
                target_id="NEO_FAST",
                timestamp_utc=now + timedelta(minutes=10 * i),
                bjd_tdb=2459000.5 + (i * 0.007),
                ra_deg=10.0 + (i * 12.0 * scale_deg_per_px),
                dec_deg=20.0,
                apparent_mag=19.5,
                ra_rate_arcsec_min=2.0,
                dec_rate_arcsec_min=0.0,
                total_speed_arcsec_min=2.0,
                delta_au=1.0,
                r_au=1.5,
                phase_angle_deg=10.0,
            )
        )

    # Centroid of whole trajectory
    mid_ra, mid_dec = compute_fixed_sky_midpoint(points)
    assert 10.0 < mid_ra < 10.0 + (12 * 12.0 * scale_deg_per_px)
    assert np.isclose(mid_dec, 20.0, atol=1e-4)

    # Sliding-window partitioning should split 12 points into multiple sub-tracklets
    sub_tracklets = partition_tracklet_sliding_window(
        points, stamp_size=128, margin_px=12, min_length=2
    )
    assert len(sub_tracklets) >= 2
    for st in sub_tracklets:
        assert len(st) >= 2
        st_mid_ra, st_mid_dec = compute_fixed_sky_midpoint(st)
        for pt in st:
            dx_px = abs(pt.ra_deg - st_mid_ra) * np.cos(np.radians(st_mid_dec)) / scale_deg_per_px
            dy_px = abs(pt.dec_deg - st_mid_dec) / scale_deg_per_px
            assert dx_px <= 52.0  # safe interior: (128/2) - 12 = 52 px
            assert dy_px <= 52.0


def test_deepstreaks_bounded_streak_vs_unbounded_satellite() -> None:
    """Verify bounded fast-NEO streak is preserved while unbounded satellite trail is rejected."""
    filter_engine = DeepStreaksArtifactFilter(ArtifactFilterConfig())
    h = w = 128
    clean_noise = np.random.default_rng(42).normal(0.0, 5.0, (h, w)).astype(np.float32)

    # 1. Clean background noise
    clean_assessment = filter_engine.evaluate_cutout(clean_noise)
    assert not clean_assessment.is_artifact
    assert clean_assessment.hard_negative_taxonomy == "REAL_SOURCE"

    # 2. Unbounded satellite streak (traverses from left border col 0 to right border col 127)
    unbounded_streak = clean_noise.copy()
    for col in range(w):
        unbounded_streak[64, col] += 150.0
    sat_assessment = filter_engine.evaluate_cutout(unbounded_streak)
    assert sat_assessment.is_artifact
    assert sat_assessment.artifact_type == "SATELLITE_TRAIL"
    assert sat_assessment.hard_negative_taxonomy == ARTIFACT_TAXONOMY_MAP["SATELLITE_TRAIL"]

    # 3. Bounded fast-NEO streak (length 20 px, PSF width in y, entirely inside interior [50, 70])
    bounded_streak = clean_noise.copy()
    for col in range(54, 74):
        bounded_streak[63, col] += 15.0
        bounded_streak[64, col] += 30.0
        bounded_streak[65, col] += 15.0
    neo_assessment = filter_engine.evaluate_cutout(bounded_streak)
    assert not neo_assessment.is_artifact
    assert neo_assessment.artifact_type == "NONE"

    # 4. Cosmic ray (sharp isolated single-pixel spike)
    cr_image = clean_noise.copy()
    cr_image[64, 64] += 600.0
    cr_assessment = filter_engine.evaluate_cutout(cr_image)
    assert cr_assessment.is_artifact
    assert cr_assessment.artifact_type == "COSMIC_RAY"
    assert cr_assessment.hard_negative_taxonomy == ARTIFACT_TAXONOMY_MAP["COSMIC_RAY"]


def test_d4_transform_and_coordinate_tracking() -> None:
    """Verify exact D4 symmetry operations and coordinate mapping."""
    triplet = np.zeros((3, 128, 128), dtype=np.float32)
    tx, ty = 40.0, 30.0
    triplet[:, int(ty), int(tx)] = 25.0

    # Horizontal flip (op=4) maps x -> w - 1 - x
    tx_trans, ty_trans = transform_coordinates_d4(tx, ty, op=4, stamp_size=128)
    assert tx_trans == 128 - 1 - 40.0
    assert ty_trans == 30.0

    aug_triplet, (new_x, new_y) = apply_d4_transform_with_coords(triplet, tx, ty, op=4)
    raw_aug = apply_d4_transform(triplet, op=4)
    np.testing.assert_array_equal(raw_aug, aug_triplet)
    assert aug_triplet.shape == (3, 128, 128)
    assert new_x == tx_trans
    assert new_y == ty_trans
    assert aug_triplet[0, int(new_y), int(new_x)] == 25.0


def test_spatial_crop_jitter() -> None:
    """Verify spatial crop jitter translates pixel data and target coordinates identically."""
    wcs = WCSMetadata(crpix=(64.0, 64.0), crval=(180.0, 10.0), cd=((-0.00028, 0.0), (0.0, 0.00028)))
    data = np.zeros((3, 128, 128), dtype=np.float32)
    data[:, 64, 64] = 50.0

    frame = CutoutTriplet(
        target_id="AST_1",
        tracklet_id="TRK_1",
        epoch_index=0,
        timestamp_utc=datetime.now(UTC),
        bjd_tdb=2459000.5,
        ra_deg=180.0,
        dec_deg=10.0,
        target_x=64.0,
        target_y=64.0,
        wcs_params=wcs,
        stamp_size=128,
        filter_band="r",
        survey="ZTF",
        snr=10.0,
        mag_apparent=20.0,
        is_synthetic=False,
        is_artifact=False,
        artifact_type="NONE",
        data=data,
    )

    rng = np.random.default_rng(42)
    jittered = apply_spatial_crop_jitter([frame], max_offset_px=3, rng=rng)
    assert len(jittered) == 1
    jf = jittered[0]
    dx = jf.target_x - 64.0
    dy = jf.target_y - 64.0
    assert abs(dx) <= 3
    assert abs(dy) <= 3
    assert jf.data[0, int(round(64.0 + dy)), int(round(64.0 + dx))] == 50.0


def test_faint_synthetic_tracklet_injection() -> None:
    """Verify multi-epoch coherent synthetic tracklet injection."""
    injector = FaintSyntheticInjector(
        FaintSyntheticConfig(mag_min=19.5, mag_max=21.5, zero_point=26.3, seed=123)
    )
    clean_bgs = [
        np.random.default_rng(i).normal(0.0, 5.0, (3, 128, 128)).astype(np.float32)
        for i in range(4)
    ]

    triplets, metas = injector.inject_synthetic_tracklet(
        clean_bgs,
        target_id="SYNTH_TRACK_01",
        speed_px_per_frame=4.0,
        position_angle_deg=0.0,
        mag=20.5,
    )
    assert len(triplets) == 4
    assert len(metas) == 4

    # Object moves along x axis by 4 px per frame
    for idx in range(3):
        dx = metas[idx + 1].centroid_x_px - metas[idx].centroid_x_px
        assert np.isclose(dx, 4.0, atol=0.2)
        # Science and difference channels received injected flux
        assert np.max(triplets[idx][0]) > np.max(clean_bgs[idx][0])
        assert np.max(triplets[idx][2]) > np.max(clean_bgs[idx][2])


def test_variable_length_tracklet_csr_hdf5_storage(tmp_path: Path) -> None:
    """Verify CSR offset indexing for variable-length tracklets and zero-leakage routing."""
    h5_path = tmp_path / "test_csr_asteroids.h5"
    manager = HDF5TrainingStorageManager(
        file_path=h5_path, stamp_size=128, seed=42, splits_manifest_path=None
    )

    wcs = WCSMetadata(crpix=(64.0, 64.0), crval=(180.0, 10.0), cd=((-0.00028, 0.0), (0.0, 0.00028)))

    def make_frame(obj_id: str, trk_id: str, ep: int) -> CutoutTriplet:
        d = np.ones((3, 128, 128), dtype=np.float32) * float(ep + 1)
        return CutoutTriplet(
            target_id=obj_id,
            tracklet_id=trk_id,
            epoch_index=ep,
            timestamp_utc=datetime.now(UTC) + timedelta(minutes=15 * ep),
            bjd_tdb=2459000.5 + (ep * 0.01),
            ra_deg=180.0 + (ep * 0.001),
            dec_deg=10.0,
            target_x=64.0,
            target_y=64.0,
            wcs_params=wcs,
            stamp_size=128,
            filter_band="r",
            survey="ZTF",
            snr=8.0,
            mag_apparent=20.0,
            is_synthetic=False,
            is_artifact=False,
            artifact_type="NONE",
            data=d,
        )

    # Tracklet A: 3 frames
    sample_a = TrackletSample(
        tracklet_id="TRK_A",
        object_id="AST_A",
        taxonomy_class="MBA",
        label=1,
        frames=[make_frame("AST_A", "TRK_A", i) for i in range(3)],
    )

    # Tracklet B: 5 frames
    sample_b = TrackletSample(
        tracklet_id="TRK_B",
        object_id="AST_B",
        taxonomy_class="NEO",
        label=2,
        frames=[make_frame("AST_B", "TRK_B", i) for i in range(5)],
    )

    # Route verification (same object must always go to the same split)
    split_a = manager.route_target_to_split("AST_A")
    split_a_repeat = manager.route_target_to_split("AST_A")
    assert split_a == split_a_repeat

    manager.append_tracklets([sample_a, sample_b])

    target_split_a = manager.route_target_to_split("AST_A")
    count_a = manager.get_tracklet_count(target_split_a)
    assert count_a >= 1

    # O(1) Slicing verification
    triplets, meta = manager.get_tracklet(target_split_a, 0)
    assert triplets.shape[1:] == (3, 128, 128)
    assert meta["length"] == triplets.shape[0]
    assert len(meta["bjd_tdb"]) == meta["length"]
    assert meta["tracklet_id"] in ("TRK_A", "TRK_B")


def test_pipeline_campaign_orchestration(tmp_path: Path) -> None:
    """Verify end-to-end data mining campaign with fixed-sky tracklets and hard negatives."""
    h5_path = tmp_path / "campaign_tracklets.h5"
    config = DataMiningPipelineConfig(
        output_hdf5_path=h5_path,
        stamp_size=128,
        enable_crop_jitter=True,
        crop_jitter_count=1,
        enable_synthetic_injection=True,
        synthetic_tracklets_per_target=2,
        synthetic_tracklet_length=3,
        max_hard_negative_ratio=0.35,
        splits_manifest_path=None,
        dry_run=True,
    )
    pipeline = SolarSystemDataMiningPipeline(config)

    targets = [
        TargetObjectSpec(target_id="TEST_NEO", target_name="Test NEO", taxonomy_class="NEO"),
        TargetObjectSpec(target_id="TEST_MBA", target_name="Test MBA", taxonomy_class="MBA"),
    ]

    report = pipeline.run_mining_campaign(targets)
    assert report.total_targets_processed == 2
    assert report.total_samples_written > 0
    assert h5_path.exists()

    summary = pipeline.storage.get_summary()
    total_frames = sum(s["total_frames"] for s in summary.values())
    total_tracklets = sum(s["total_tracklets"] for s in summary.values())
    assert total_tracklets >= 2
    assert total_frames > total_tracklets  # Variable length N >= 2
