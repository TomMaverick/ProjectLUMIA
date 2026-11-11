"""Hermetic unit tests for GroundTruthPipeline orchestrator."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from LUMIA.core.schemas import KnownObjectEphemeris
from LUMIA.data_pipeline.ground_truth_orchestrator import (
    GroundTruthEpochResult,
    GroundTruthPipeline,
)
from LUMIA.data_pipeline.hdf5_storage import HDF5GroundTruthWriter
from LUMIA.physics.mpc_client import MinorPlanetCenterClient

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def test_wcs() -> WCS:
    """Create a calibrated reference WCS for 1000x1000 test exposures."""
    w = WCS(naxis=2)
    w.wcs.crpix = [500.0, 500.0]
    w.wcs.cdelt = [-0.001, 0.001]
    w.wcs.crval = [180.0, 10.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


# ---------------------------------------------------------------------------
# 1. Pipeline End-to-End Orchestration Tests
# ---------------------------------------------------------------------------


class TestGroundTruthPipelineOrchestration:
    """Verify end-to-end integration of MPC querying, cutout slicing, and HDF5 persistence."""

    def test_process_epoch_success_with_real_and_bogus(
        self, tmp_path: Path, test_wcs: WCS
    ) -> None:
        """Verify process_epoch extracts Real and Bogus stamps and archives them to HDF5."""
        archive_path = tmp_path / "ground_truth_dataset.h5"
        writer = HDF5GroundTruthWriter(archive_path, chunk_size=32)

        # 1. Setup Mock MPC client returning 3 objects:
        # Object 1: at pixel (500, 500) -> Center of image (Valid Real)
        # Object 2: at pixel (300, 700) -> Valid Real
        # Object 3: at pixel (5, 5) -> Near edge (Boundary hit, must be discarded)
        c1 = test_wcs.pixel_to_world(500.0, 500.0)
        c2 = test_wcs.pixel_to_world(300.0, 700.0)
        c3 = test_wcs.pixel_to_world(5.0, 5.0)

        fake_asteroids = [
            KnownObjectEphemeris(
                designation="1999 RQ36",
                ra_deg=float(c1.ra.deg),
                dec_deg=float(c1.dec.deg),
                vmag=19.2,
                ra_rate_arcsec_hr=1.5,
                dec_rate_arcsec_hr=-0.8,
            ),
            KnownObjectEphemeris(
                designation="433_Eros",
                ra_deg=float(c2.ra.deg),
                dec_deg=float(c2.dec.deg),
                vmag=14.5,
                ra_rate_arcsec_hr=2.1,
                dec_rate_arcsec_hr=0.4,
            ),
            KnownObjectEphemeris(
                designation="Boundary_Asteroid",
                ra_deg=float(c3.ra.deg),
                dec_deg=float(c3.dec.deg),
                vmag=20.0,
                ra_rate_arcsec_hr=0.1,
                dec_rate_arcsec_hr=0.1,
            ),
        ]

        mock_mpc = MagicMock(spec=MinorPlanetCenterClient)
        mock_mpc.query_field_ephemerides.return_value = fake_asteroids

        # 2. Setup 1000x1000 mock tensors with markers at asteroid centers
        h, w = 1000, 1000
        science = torch.zeros((h, w), dtype=torch.float32)
        reference = torch.zeros((h, w), dtype=torch.float32)
        difference = torch.zeros((h, w), dtype=torch.float32)

        science[500, 500] = 42.0
        reference[500, 500] = 5.0
        difference[500, 500] = 37.0

        science[700, 300] = 99.0
        difference[700, 300] = 90.0

        # 3. Instantiate and run pipeline
        pipeline = GroundTruthPipeline(
            mpc_client=mock_mpc,
            hdf5_writer=writer,
            bogus_count=6,
            min_bogus_distance_px=50.0,
            stamp_size=31,
            random_seed=123,
        )

        result: GroundTruthEpochResult = pipeline.process_epoch(
            science_tensor=science,
            reference_tensor=reference,
            difference_tensor=difference,
            wcs_header=test_wcs,
            bjd_tdb=2460755.75,
            field_id="FIELD_ZTF_01",
            bogus_count=6,
            split="train",
        )

        # 4. Verify returned summary
        assert isinstance(result, GroundTruthEpochResult)
        assert result.field_id == "FIELD_ZTF_01"
        assert result.real_count == 2, "Object 3 at (5, 5) must be discarded as a boundary hit."
        assert result.bogus_count == 6
        assert result.total_appended == 8
        assert result.archive_total_samples == 8

        # 5. Verify physical HDF5 storage content
        with h5py.File(archive_path, "r") as h5:
            train_grp = h5["train"]
            cutouts = train_grp["cutouts"][:]
            labels = train_grp["labels"][:]
            cand_ids = [s.decode("ascii") for s in train_grp["candidate_id"][:]]

            assert cutouts.shape == (8, 3, 31, 31)
            assert labels.shape == (8,)

            # Real samples are first 2
            assert list(labels[:2]) == [1, 1]
            # Bogus samples are next 6
            assert list(labels[2:]) == [0, 0, 0, 0, 0, 0]

            # Verify Real cutout marker content at (15, 15)
            # Object 1: science channel = 42.0, reference = 5.0, difference = 37.0
            assert cutouts[0, 0, 15, 15] == pytest.approx(42.0)
            assert cutouts[0, 1, 15, 15] == pytest.approx(5.0)
            assert cutouts[0, 2, 15, 15] == pytest.approx(37.0)

            # Object 2: science channel = 99.0
            assert cutouts[1, 0, 15, 15] == pytest.approx(99.0)

            # Check candidate IDs format
            assert "real_FIELD_ZTF_01" in cand_ids[0]
            assert "bogus_FIELD_ZTF_01" in cand_ids[2]

    def test_bogus_cutouts_satisfy_minimum_distance_constraint(
        self, tmp_path: Path, test_wcs: WCS
    ) -> None:
        """Verify all sampled bogus cutout centroids are at least 50px from known objects."""
        archive_path = tmp_path / "distance_test.h5"
        writer = HDF5GroundTruthWriter(archive_path)

        # Place known asteroid at center (500, 500)
        c = test_wcs.pixel_to_world(500.0, 500.0)
        fake_asteroid = KnownObjectEphemeris(
            designation="Asteroid_Center",
            ra_deg=float(c.ra.deg),
            dec_deg=float(c.dec.deg),
            vmag=18.0,
            ra_rate_arcsec_hr=0.0,
            dec_rate_arcsec_hr=0.0,
        )

        mock_mpc = MagicMock(spec=MinorPlanetCenterClient)
        mock_mpc.query_field_ephemerides.return_value = [fake_asteroid]

        science = np.zeros((1000, 1000), dtype=np.float32)
        reference = np.zeros((1000, 1000), dtype=np.float32)
        difference = np.zeros((1000, 1000), dtype=np.float32)

        min_dist = 50.0
        pipeline = GroundTruthPipeline(
            mpc_client=mock_mpc,
            hdf5_writer=writer,
            bogus_count=15,
            min_bogus_distance_px=min_dist,
            stamp_size=31,
            random_seed=999,
        )

        pipeline.process_epoch(
            science_tensor=science,
            reference_tensor=reference,
            difference_tensor=difference,
            wcs_header=test_wcs,
            bjd_tdb=2460755.0,
            field_id="FIELD_DIST",
            bogus_count=15,
        )

        with h5py.File(archive_path, "r") as h5:
            ra_arr = h5["train/ra_deg"][:]
            dec_arr = h5["train/dec_deg"][:]

            # Samples 1 to 15 are bogus
            for i in range(1, 16):
                b_sky = SkyCoord(ra=ra_arr[i], dec=dec_arr[i], unit="deg")
                bx, by = test_wcs.world_to_pixel(b_sky)
                dist_to_center = math.hypot(float(bx) - 500.0, float(by) - 500.0)
                assert dist_to_center >= min_dist - 0.5, (
                    f"Bogus cutout {i} at ({bx:.1f}, {by:.1f}) is within {min_dist}px "
                    f"of known object (actual distance: {dist_to_center:.2f}px)"
                )

    def test_zero_known_objects_still_samples_bogus_cutouts(
        self, tmp_path: Path, test_wcs: WCS
    ) -> None:
        """Verify that when MPC finds zero known objects, bogus sampling still proceeds."""
        archive_path = tmp_path / "zero_objects.h5"
        writer = HDF5GroundTruthWriter(archive_path)

        mock_mpc = MagicMock(spec=MinorPlanetCenterClient)
        mock_mpc.query_field_ephemerides.return_value = []

        science = np.zeros((500, 500), dtype=np.float32)
        reference = np.zeros((500, 500), dtype=np.float32)
        difference = np.zeros((500, 500), dtype=np.float32)

        pipeline = GroundTruthPipeline(
            mpc_client=mock_mpc,
            hdf5_writer=writer,
            bogus_count=5,
        )

        result = pipeline.process_epoch(
            science_tensor=science,
            reference_tensor=reference,
            difference_tensor=difference,
            wcs_header=test_wcs,
            bjd_tdb=2460755.0,
            field_id="EMPTY_FIELD",
            bogus_count=5,
        )

        assert result.real_count == 0
        assert result.bogus_count == 5
        assert result.total_appended == 5

        with h5py.File(archive_path, "r") as h5:
            labels = h5["train/labels"][:]
            assert list(labels) == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# 2. Input Validation and Defensive Bounds Tests
# ---------------------------------------------------------------------------


class TestGroundTruthPipelineValidation:
    """Verify defensive checks on array shapes, stamp sizes, and pipeline options."""

    def test_mismatched_image_shapes_rejected(self, test_wcs: WCS, tmp_path: Path) -> None:
        """Verify ValueError when science, reference, and difference shapes differ."""
        writer = HDF5GroundTruthWriter(tmp_path / "mismatch.h5")
        mock_mpc = MagicMock(spec=MinorPlanetCenterClient)
        pipeline = GroundTruthPipeline(mock_mpc, writer)

        sci = torch.zeros((500, 500))
        ref = torch.zeros((400, 400))
        diff = torch.zeros((500, 500))

        with pytest.raises(ValueError, match="Image shapes must match"):
            pipeline.process_epoch(sci, ref, diff, test_wcs, 2460755.0, "FIELD_ERR")

    def test_image_smaller_than_stamp_size_rejected(self, test_wcs: WCS, tmp_path: Path) -> None:
        """Verify ValueError when images are smaller than the cutout stamp size."""
        writer = HDF5GroundTruthWriter(tmp_path / "small.h5")
        mock_mpc = MagicMock(spec=MinorPlanetCenterClient)
        pipeline = GroundTruthPipeline(mock_mpc, writer, stamp_size=31)

        sci = torch.zeros((20, 20))
        ref = torch.zeros((20, 20))
        diff = torch.zeros((20, 20))

        with pytest.raises(ValueError, match="smaller than stamp_size"):
            pipeline.process_epoch(sci, ref, diff, test_wcs, 2460755.0, "FIELD_SMALL")

    @pytest.mark.parametrize("bad_stamp_size", [30, 14, 0, -31])
    def test_invalid_stamp_size_raises_value_error(self, tmp_path: Path, bad_stamp_size: int) -> None:
        """Verify stamp_size must be an odd integer >= 15."""
        writer = HDF5GroundTruthWriter(tmp_path / "bad_stamp.h5")
        mock_mpc = MagicMock(spec=MinorPlanetCenterClient)
        with pytest.raises(ValueError, match="stamp_size must be an odd integer"):
            GroundTruthPipeline(mock_mpc, writer, stamp_size=bad_stamp_size)

    def test_invalid_tensor_type_raises_type_error(self, test_wcs: WCS, tmp_path: Path) -> None:
        """Verify passing non-tensor/non-array inputs raises TypeError."""
        writer = HDF5GroundTruthWriter(tmp_path / "type_err.h5")
        mock_mpc = MagicMock(spec=MinorPlanetCenterClient)
        pipeline = GroundTruthPipeline(mock_mpc, writer)

        with pytest.raises(TypeError, match="must be a torch.Tensor or numpy.ndarray"):
            pipeline.process_epoch([[1.0]], [[1.0]], [[1.0]], test_wcs, 2460755.0, "FIELD_TYPE")  # type: ignore[arg-type]
