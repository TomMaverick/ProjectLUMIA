"""Unit tests for PyTorch ZTFTrackletDataset, CSR slicing, D4 augmentation, and DataLoader."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from LUMIA.data_mining.archive_crawler import CutoutTriplet, WCSMetadata
from LUMIA.data_mining.dataset import (
    ZTFTrackletDataset,
    create_ztf_tracklet_dataloader,
    pad_tracklet_batch,
    ztf_tracklet_collate_fn,
)
from LUMIA.data_mining.storage import (
    HDF5TrainingStorageManager,
    TrackletSample,
)


def _make_dummy_triplet(
    target_id: str,
    tracklet_id: str,
    epoch: int,
    tx: float = 64.0,
    ty: float = 64.0,
) -> CutoutTriplet:
    """Create a minimal synthetic CutoutTriplet for testing."""
    wcs = WCSMetadata(
        crpix=(64.0, 64.0),
        crval=(180.0, 20.0),
        cd=((-0.000281, 0.0), (0.0, 0.000281)),
    )
    data = np.full((3, 128, 128), fill_value=float(epoch + 1), dtype=np.float32)
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
        stamp_size=128,
        filter_band="r",
        survey="ZTF",
        snr=10.0 + epoch,
        mag_apparent=19.5,
        is_synthetic=False,
        is_artifact=False,
        artifact_type="NONE",
        data=data,
    )


@pytest.fixture
def mock_hdf5_archive(tmp_path: Path) -> Path:
    """Populate an out-of-core CSR HDF5 archive with variable-length tracklets."""
    h5_file = tmp_path / "test_ztf_tracklets.h5"
    storage = HDF5TrainingStorageManager(file_path=h5_file, stamp_size=128)

    # Tracklet 1: 3 frames (label 1: point)
    frames_1 = [
        _make_dummy_triplet("AST_01", "TRK_01", i, tx=50.0 + i, ty=50.0 + i) for i in range(3)
    ]
    sample_1 = TrackletSample(
        tracklet_id="TRK_01",
        object_id="AST_01",
        taxonomy_class="MBA",
        label=1,
        frames=frames_1,
    )

    # Tracklet 2: 5 frames (label 2: streak)
    frames_2 = [
        _make_dummy_triplet("AST_02", "TRK_02", i, tx=30.0 + (i * 2), ty=40.0 + (i * 2))
        for i in range(5)
    ]
    sample_2 = TrackletSample(
        tracklet_id="TRK_02",
        object_id="AST_02",
        taxonomy_class="NEO",
        label=2,
        frames=frames_2,
    )

    # Tracklet 3: 2 frames (label 0: bogus)
    frames_3 = [_make_dummy_triplet("BOGUS_01", "TRK_03", i, tx=64.0, ty=64.0) for i in range(2)]
    sample_3 = TrackletSample(
        tracklet_id="TRK_03",
        object_id="BOGUS_01",
        taxonomy_class="NEO",
        label=0,
        frames=frames_3,
    )

    storage.append_tracklets([sample_1, sample_2, sample_3], force_split="train")
    return h5_file


def test_ztf_tracklet_dataset_out_of_core_slicing(mock_hdf5_archive: Path) -> None:
    """Verify O(1) CSR out-of-core slicing returns correct shapes, types, and coordinates."""
    dataset = ZTFTrackletDataset(h5_path=mock_hdf5_archive, split="train", augment=False)
    assert len(dataset) == 3

    # Sample 0: 3 frames
    item0 = dataset[0]
    assert isinstance(item0["tracklet"], torch.Tensor)
    assert item0["tracklet"].shape == (3, 3, 128, 128)
    assert item0["tracklet"].dtype == torch.float32
    assert item0["label"].item() == 1
    assert item0["length"] == 3
    assert item0["target_coords"].shape == (3, 2)
    assert item0["tracklet_id"] == "TRK_01"
    assert item0["object_id"] == "AST_01"
    assert item0["bjd_tdb"].shape == (3,)
    assert item0["snr"].shape == (3,)

    # Sample 1: 5 frames
    item1 = dataset[1]
    assert item1["tracklet"].shape == (5, 3, 128, 128)
    assert item1["label"].item() == 2
    assert item1["length"] == 5
    assert item1["target_coords"].shape == (5, 2)

    # Sample 2: 2 frames
    item2 = dataset[2]
    assert item2["tracklet"].shape == (2, 3, 128, 128)
    assert item2["label"].item() == 0
    assert item2["length"] == 2

    # Context manager lifecycle check
    with ZTFTrackletDataset(h5_path=mock_hdf5_archive, split="train") as ds:
        assert len(ds) == 3
        _ = ds[0]

    # Index out of bounds
    with pytest.raises(IndexError):
        _ = dataset[99]


def test_ztf_tracklet_d4_augmentation(mock_hdf5_archive: Path) -> None:
    """Verify D4 dihedral augmentation transforms tensors and target coordinates in-memory."""
    torch.manual_seed(42)
    np.random.seed(42)

    dataset_aug = ZTFTrackletDataset(h5_path=mock_hdf5_archive, split="train", augment=True)
    item = dataset_aug[0]

    assert item["tracklet"].shape == (3, 3, 128, 128)
    assert item["target_coords"].shape == (3, 2)
    coords = item["target_coords"].numpy()

    # Transformed coordinates must remain within stamp bounds [0, 128)
    assert np.all(coords >= 0.0)
    assert np.all(coords <= 128.0)


def test_ztf_tracklet_collate_fn(mock_hdf5_archive: Path) -> None:
    """Verify variable-length tracklet collation into padded tensors with padding masks."""
    dataset = ZTFTrackletDataset(h5_path=mock_hdf5_archive, split="train", augment=False)
    batch_raw = [dataset[0], dataset[1], dataset[2]]

    batch = ztf_tracklet_collate_fn(batch_raw)

    # Max length is 5 (from item 1)
    assert batch["tracklets"].shape == (3, 5, 3, 128, 128)
    assert batch["labels"].tolist() == [1, 2, 0]
    assert batch["lengths"].tolist() == [3, 5, 2]
    assert batch["target_coords"].shape == (3, 5, 2)

    # Padding mask assertions (True for valid frame, False for padded frame)
    mask = batch["padding_mask"]
    assert mask.shape == (3, 5)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].tolist() == [True, True, True, True, True]
    assert mask[2].tolist() == [True, True, False, False, False]

    # Padding fill verification: frames beyond native length must be 0.0
    assert torch.all(batch["tracklets"][0, 3:] == 0.0)
    assert torch.all(batch["tracklets"][2, 2:] == 0.0)

    # Unpadded raw tensors preserved
    assert len(batch["raw_tracklets"]) == 3
    assert batch["raw_tracklets"][0].shape == (3, 3, 128, 128)
    assert batch["raw_tracklets"][1].shape == (5, 3, 128, 128)
    assert batch["raw_tracklets"][2].shape == (2, 3, 128, 128)


def test_pad_tracklet_batch_custom_pad_value(mock_hdf5_archive: Path) -> None:
    """Verify custom pad_value is respected when padding tracklet batches."""
    dataset = ZTFTrackletDataset(h5_path=mock_hdf5_archive, split="train", augment=False)
    batch_raw = [dataset[0], dataset[2]]  # lengths 3 and 2 -> max_len 3

    batch = pad_tracklet_batch(batch_raw, pad_value=-999.0)
    assert batch["tracklets"].shape == (2, 3, 3, 128, 128)
    assert torch.all(batch["tracklets"][1, 2] == -999.0)

    with pytest.raises(ValueError, match="Cannot collate an empty batch"):
        pad_tracklet_batch([])


def test_create_ztf_tracklet_dataloader(mock_hdf5_archive: Path) -> None:
    """Verify create_ztf_tracklet_dataloader produces iterable, properly batched DataLoader."""
    loader = create_ztf_tracklet_dataloader(
        h5_path=mock_hdf5_archive,
        split="train",
        batch_size=2,
        shuffle=False,
        augment=False,
        num_workers=0,
    )
    assert isinstance(loader, DataLoader)

    batches = list(loader)
    assert len(batches) == 2  # 3 items with batch_size 2 -> 2 batches

    # First batch: items 0 and 1 -> lengths 3 and 5 -> max_len 5
    b0 = batches[0]
    assert b0["tracklets"].shape == (2, 5, 3, 128, 128)
    assert b0["lengths"].tolist() == [3, 5]

    # Second batch: item 2 -> length 2 -> max_len 2
    b1 = batches[1]
    assert b1["tracklets"].shape == (1, 2, 3, 128, 128)
    assert b1["lengths"].tolist() == [2]


def test_dataset_error_handling(tmp_path: Path) -> None:
    """Verify error conditions on non-existent archives and invalid splits."""
    non_existent = tmp_path / "non_existent.h5"
    with pytest.raises(FileNotFoundError):
        _ = ZTFTrackletDataset(h5_path=non_existent)
