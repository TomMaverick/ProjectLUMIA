"""End-to-end pilot pipeline benchmark verifying Crawler, SQLite Ledger, and HDF5 CSR integration."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from astropy.io import fits
import numpy as np
import pytest
import responses
import torch

from LUMIA.data_mining.crawler_ledger import SQLiteCrawlerLedger
from LUMIA.data_mining.dataset import create_ztf_tracklet_dataloader
from LUMIA.data_mining.pipeline import (
    DataMiningPipelineConfig,
    SolarSystemDataMiningPipeline,
)
from LUMIA.data_mining.storage import (
    DEFAULT_SPLITS_MANIFEST_PATH,
    HDF5TrainingStorageManager,
)
from LUMIA.data_mining.taxonomy import (
    EphemerisPoint,
    HorizonsQueryConfig,
    TargetObjectSpec,
)


def _generate_valid_fits_cutout_bytes(size: int = 144, value: float = 100.0) -> bytes:
    """Generate in-memory byte buffer of a valid FITS file with header SIMPLE = T."""
    data = np.full((size, size), fill_value=value, dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["NAXIS1"] = size
    hdu.header["NAXIS2"] = size
    hdu.header["CRPIX1"] = float(size / 2.0)
    hdu.header["CRPIX2"] = float(size / 2.0)
    buf = io.BytesIO()
    hdu.writeto(buf)
    return buf.getvalue()


@responses.activate
def test_pilot_pipeline_production_e2e(tmp_path: Path) -> None:
    """Execute end-to-end pilot run across crawler, ledger, and storage."""
    h5_path = tmp_path / "pilot_tracklets_144x144.h5"
    db_path = tmp_path / "pilot_crawler_ledger.db"

    # Mock IPAC ZTF IBE responses for science, reference, and difference cutouts
    fits_sci = _generate_valid_fits_cutout_bytes(144, 120.0)
    fits_ref = _generate_valid_fits_cutout_bytes(144, 100.0)
    fits_diff = _generate_valid_fits_cutout_bytes(144, 20.0)

    responses.add(
        responses.GET,
        "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/ztf_test_sci.fits",
        body=fits_sci,
        status=200,
        content_type="image/fits",
    )
    responses.add(
        responses.GET,
        "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/ztf_test_ref.fits",
        body=fits_ref,
        status=200,
        content_type="image/fits",
    )
    responses.add(
        responses.GET,
        "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/diff/ztf_test_diff.fits",
        body=fits_diff,
        status=200,
        content_type="image/fits",
    )

    # 1. Setup targets from dataset splits manifest
    # Object 99942 (Apophis, NEO) and Object 1 (Ceres, MBA) are mapped in configs/dataset_splits.json
    targets = [
        TargetObjectSpec(
            target_id="99942",
            target_name="99942 Apophis",
            taxonomy_class="NEO",
        ),
        TargetObjectSpec(
            target_id="1",
            target_name="1 Ceres",
            taxonomy_class="MBA",
        ),
    ]

    # Generate synthetic Horizons ephemerides to mock JPL network calls
    def mock_query_ephemeris(
        self: object, target: TargetObjectSpec, config: HorizonsQueryConfig
    ) -> list[EphemerisPoint]:
        base_utc = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
        points = []
        for i in range(4):
            t = base_utc + timedelta(minutes=15 * i)
            points.append(
                EphemerisPoint(
                    target_id=target.target_id,
                    timestamp_utc=t,
                    bjd_tdb=2460462.5 + (i * 0.01),
                    ra_deg=180.0 + (i * 0.005),
                    dec_deg=20.0 + (i * 0.005),
                    ra_rate_arcsec_min=0.3,
                    dec_rate_arcsec_min=0.3,
                    total_speed_arcsec_min=0.424,
                    apparent_mag=18.5,
                    delta_au=1.2,
                    r_au=1.8,
                    phase_angle_deg=15.0,
                )
            )
        return points

    # Mock crawler query_overlapping_exposures to return URLs pointing to our mock responses
    from LUMIA.data_mining.archive_crawler import ExposureMetadata

    def mock_query_overlapping_exposures(
        self: object, points: list[EphemerisPoint]
    ) -> list[ExposureMetadata]:
        return [
            ExposureMetadata(
                exposure_id="EXP_PILOT_01",
                survey="ZTF",
                filter_band="r",
                timestamp_utc=datetime(2024, 6, 1, 0, 0, tzinfo=UTC),
                ra_center_deg=180.0,
                dec_center_deg=20.0,
                science_url="https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/ztf_test_sci.fits",
                reference_url="https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/ztf_test_ref.fits",
                difference_url="https://irsa.ipac.caltech.edu/ibe/data/ztf/products/diff/ztf_test_diff.fits",
            )
        ]

    config = DataMiningPipelineConfig(
        output_hdf5_path=h5_path,
        stamp_size=144,
        splits_manifest_path=DEFAULT_SPLITS_MANIFEST_PATH,
        ledger_db_path=db_path,
        dry_run=False,  # Full production mode
        enable_crop_jitter=False,
        enable_synthetic_injection=True,
        synthetic_tracklets_per_target=1,
        synthetic_tracklet_length=3,
    )

    with (
        patch("LUMIA.data_mining.taxonomy.HorizonsClient.query_ephemeris", mock_query_ephemeris),
        patch("LUMIA.data_mining.archive_crawler.ArchiveCrawler.query_overlapping_exposures", mock_query_overlapping_exposures),
    ):
        pipeline = SolarSystemDataMiningPipeline(config)

        # First Pilot Execution
        report1 = pipeline.run_mining_campaign(targets)

        assert report1.total_targets_processed == 2
        assert report1.total_samples_written > 0
        assert h5_path.exists()
        assert db_path.exists()

        # Check SQLite ledger
        ledger = SQLiteCrawlerLedger(db_path=db_path)
        completed = ledger.get_completed_ids()
        assert len(completed) >= 2  # Real and synthetic tracklets recorded

        # Check HDF5 storage
        storage = HDF5TrainingStorageManager(
            file_path=h5_path,
            stamp_size=144,
            splits_manifest_path=DEFAULT_SPLITS_MANIFEST_PATH,
        )
        repairs = storage.repair_hdf5_csr_integrity()
        assert sum(repairs.values()) == 0  # 100% clean CSR integrity

        summary = storage.get_summary()
        total_tracklets = sum(s["total_tracklets"] for s in summary.values())
        assert total_tracklets == report1.total_samples_written

        # Second Pilot Execution: Idempotent resumption
        # Since ledger recorded completed tracklets, running again must skip all completed tracklets cleanly
        report2 = pipeline.run_mining_campaign(targets)
        assert report2.total_targets_processed == 2
        # Zero new tracklets created on resumption
        assert sum(r.tracklets_created for r in report2.target_reports) == 0
        # Cumulative storage count remains identical
        assert report2.total_samples_written == report1.total_samples_written

        # Verify DataLoader loads from newly created pilot archive seamlessly
        # Find which split received tracklets
        for sp in ("train", "val", "test", "ood"):
            if summary[sp]["total_tracklets"] > 0:
                loader = create_ztf_tracklet_dataloader(
                    h5_path=h5_path,
                    split=sp,  # type: ignore[arg-type]
                    batch_size=2,
                    shuffle=False,
                    num_workers=0,
                )
                batch = next(iter(loader))
                assert "tracklets" in batch
                assert batch["tracklets"].shape[-2:] == (128, 128)  # Jitter cropped to 128
                break
