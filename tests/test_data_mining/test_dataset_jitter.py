"""Unit tests for sequence-synchronized spatial crop jitter and astrometric invariance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from LUMIA.data_mining.archive_crawler import CutoutTriplet, WCSMetadata
from LUMIA.data_mining.dataset import (
    ZTFTrackletDataset,
    create_ztf_tracklet_dataloader,
)
from LUMIA.data_mining.storage import (
    HDF5TrainingStorageManager,
    TrackletSample,
)


def _make_144_triplet(
    target_id: str,
    tracklet_id: str,
    epoch: int,
    tx: float,
    ty: float,
    crpix: tuple[float, float] = (72.0, 72.0),
) -> CutoutTriplet:
    """Create a 144x144 CutoutTriplet with known coordinates and WCS metadata."""
    wcs = WCSMetadata(
        crpix=crpix,
        crval=(180.0, 20.0),
        cd=((-0.000281, 0.0), (0.0, 0.000281)),
    )
    # Put a distinct marker value at the pixel corresponding to integer target position
    data = np.zeros((3, 144, 144), dtype=np.float32)
    ix, iy = int(round(tx)), int(round(ty))
    if 0 <= ix < 144 and 0 <= iy < 144:
        data[:, iy, ix] = 999.0

    return CutoutTriplet(
        target_id=target_id,
        tracklet_id=tracklet_id,
        epoch_index=epoch,
        timestamp_utc=datetime.now(UTC),
        bjd_tdb=2459000.5 + (epoch * 0.01),
        ra_deg=180.0 + (epoch * 0.001),
        dec_deg=20.0 + (epoch * 0.001),
        target_x=tx,
        target_y=ty,
        wcs_params=wcs,
        stamp_size=144,
        filter_band="r",
        survey="ZTF",
        snr=15.0,
        mag_apparent=19.0,
        is_synthetic=False,
        is_artifact=False,
        artifact_type="NONE",
        data=data,
    )


@pytest.fixture
def h5_archive_144(tmp_path: Path) -> Path:
    """Create a 144x144 HDF5 storage archive with tracklets across all 4 splits."""
    h5_file = tmp_path / "test_144x144.h5"
    storage = HDF5TrainingStorageManager(
        file_path=h5_file, stamp_size=144, splits_manifest_path=None
    )

    for split in ("train", "val", "test", "ood"):
        samples = []
        for trk_idx in range(2):
            t_id = f"TRK_{split.upper()}_{trk_idx:02d}"
            obj_id = f"OBJ_{split.upper()}_{trk_idx:02d}"
            frames = [
                _make_144_triplet(
                    target_id=obj_id,
                    tracklet_id=t_id,
                    epoch=ep,
                    tx=60.0 + (ep * 4.0),
                    ty=65.0 + (ep * 3.0),
                    crpix=(72.0, 72.0),
                )
                for ep in range(3 + trk_idx)
            ]
            samples.append(
                TrackletSample(
                    tracklet_id=t_id,
                    object_id=obj_id,
                    taxonomy_class="MBA" if split != "ood" else "ISO",
                    label=1 if split != "ood" else 3,
                    frames=frames,
                )
            )
        storage.append_tracklets(samples, force_split=split)  # type: ignore[arg-type]

    return h5_file


def test_jitter_sampling_range_and_synchronization(h5_archive_144: Path) -> None:
    """Verify discrete jitter sampling in [-8, +8] and sequence-synchronized cropping."""
    dataset = ZTFTrackletDataset(
        h5_path=h5_archive_144,
        split="train",
        augment=False,
        is_train=True,
        jitter=True,
    )
    assert len(dataset) == 2

    offsets_seen: set[tuple[int, int]] = set()

    for _ in range(50):
        item = dataset[0]
        tracklet = item["tracklet"]
        assert tracklet.shape == (3, 3, 128, 128)
        assert tracklet.dtype == torch.float32

        dx, dy = item["jitter_offset"].tolist()
        assert -8 <= dx <= 8
        assert -8 <= dy <= 8
        offsets_seen.add((dx, dy))

        # Check coordinate shift synchronization
        # Original: tx=60.0, ty=65.0 for ep=0
        expected_tx = 60.0 - (8 + dx)
        expected_ty = 65.0 - (8 + dy)
        actual_tx, actual_ty = item["target_coords"][0].tolist()
        assert pytest.approx(actual_tx, abs=1e-5) == expected_tx
        assert pytest.approx(actual_ty, abs=1e-5) == expected_ty

        # Check WCS CRPIX shift synchronization
        if "wcs_crpix" in item:
            actual_crpix_x, actual_crpix_y = item["wcs_crpix"][0].tolist()
            assert pytest.approx(actual_crpix_x, abs=1e-5) == 72.0 - (8 + dx)
            assert pytest.approx(actual_crpix_y, abs=1e-5) == 72.0 - (8 + dy)

    # Random sampling across 50 iterations must explore multiple distinct offsets
    assert len(offsets_seen) > 5


def test_centered_crop_on_eval_and_ood_splits(h5_archive_144: Path) -> None:
    """Verify validation, test, and ood splits strictly apply centered crops (dx=0, dy=0)."""
    for split in ("val", "test", "ood"):
        dataset = ZTFTrackletDataset(
            h5_path=h5_archive_144,
            split=split,  # type: ignore[arg-type]
            augment=False,
        )
        assert len(dataset) == 2

        for i in range(len(dataset)):
            item = dataset[i]
            assert item["tracklet"].shape[1:] == (3, 128, 128)
            dx, dy = item["jitter_offset"].tolist()
            assert dx == 0
            assert dy == 0

            # Ep 0 original coords: tx=60.0, ty=65.0 -> shifted by -(8+0) = -8
            assert pytest.approx(item["target_coords"][0, 0].item(), abs=1e-5) == 60.0 - 8.0
            assert pytest.approx(item["target_coords"][0, 1].item(), abs=1e-5) == 65.0 - 8.0

            if "wcs_crpix" in item:
                assert pytest.approx(item["wcs_crpix"][0, 0].item(), abs=1e-5) == 72.0 - 8.0
                assert pytest.approx(item["wcs_crpix"][0, 1].item(), abs=1e-5) == 72.0 - 8.0


def test_coordinate_and_wcs_transformation_math(h5_archive_144: Path) -> None:
    """Verify coordinate transformation and WCS CRPIX shift formulas for explicit offsets."""
    dataset = ZTFTrackletDataset(h5_path=h5_archive_144, split="train", augment=False)

    for dx, dy in [(-8, -8), (0, 0), (8, 8), (-3, 5), (7, -4)]:
        dataset.jitter_offset = (dx, dy)
        item = dataset[0]

        # Frame 0: orig tx=60, ty=65
        expected_tx = 60.0 - (8 + dx)
        expected_ty = 65.0 - (8 + dy)
        assert pytest.approx(item["target_coords"][0, 0].item(), abs=1e-5) == expected_tx
        assert pytest.approx(item["target_coords"][0, 1].item(), abs=1e-5) == expected_ty

        if "wcs_crpix" in item:
            assert pytest.approx(item["wcs_crpix"][0, 0].item(), abs=1e-5) == 72.0 - (8 + dx)
            assert pytest.approx(item["wcs_crpix"][0, 1].item(), abs=1e-5) == 72.0 - (8 + dy)


def test_apparent_velocity_vector_preservation(h5_archive_144: Path) -> None:
    """Verify that inter-frame apparent velocity vectors are strictly invariant under jitter."""
    dataset = ZTFTrackletDataset(h5_path=h5_archive_144, split="train", augment=False)

    # In our fixture:
    # ep 0: (60, 65), t0 = 0.0
    # ep 1: (64, 68), t1 = 0.01 days
    # ep 2: (68, 71), t2 = 0.02 days
    # vx = (64 - 60) / 0.01 = 400.0, vy = (68 - 65) / 0.01 = 300.0
    expected_vx = (64.0 - 60.0) / 0.01
    expected_vy = (68.0 - 65.0) / 0.01

    for _ in range(20):
        item = dataset[0]
        coords = item["target_coords"].numpy()
        bjd = item["bjd_tdb"].numpy()

        dt = bjd[1] - bjd[0]
        vx = (coords[1, 0] - coords[0, 0]) / dt
        vy = (coords[1, 1] - coords[0, 1]) / dt

        assert pytest.approx(vx, abs=1e-4) == expected_vx
        assert pytest.approx(vy, abs=1e-4) == expected_vy


def test_dataset_and_dataloader_all_four_splits(h5_archive_144: Path) -> None:
    """Verify PyTorch DataLoader creates padded batches across train, val, test, and ood."""
    for split in ("train", "val", "test", "ood"):
        loader = create_ztf_tracklet_dataloader(
            h5_path=h5_archive_144,
            split=split,  # type: ignore[arg-type]
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))

        assert "tracklets" in batch
        assert "labels" in batch
        assert "lengths" in batch
        assert "padding_mask" in batch
        assert "target_coords" in batch
        assert "wcs_crpix" in batch

        b, n_max, c, h, w = batch["tracklets"].shape
        assert b == 2
        assert c == 3
        assert h == 128
        assert w == 128
        assert batch["labels"].shape == (2,)
        assert batch["lengths"].shape == (2,)
        assert batch["padding_mask"].shape == (2, n_max)
        assert batch["target_coords"].shape == (2, n_max, 2)
        assert batch["wcs_crpix"].shape == (2, n_max, 2)
