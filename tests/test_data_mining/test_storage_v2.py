"""Unit tests for Storage V2 (schema 2.1.0, 144x144, 4 splits, CSR repair, and ledger)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from datetime import UTC, datetime
from pathlib import Path

from LUMIA.data_mining.archive_crawler import CutoutTriplet, WCSMetadata
from LUMIA.data_mining.crawler_ledger import CrawlerLedgerEntry, SQLiteCrawlerLedger
from LUMIA.data_mining.storage import (
    HDF5TrainingStorageManager,
    TrackletSample,
)


def _make_dummy_frame(
    object_id: str,
    tracklet_id: str,
    epoch_index: int,
    stamp_size: int = 144,
) -> CutoutTriplet:
    wcs = WCSMetadata(
        crpix=(stamp_size / 2.0, stamp_size / 2.0),
        crval=(180.0, 15.0),
        cd=((-0.000281, 0.0), (0.0, 0.000281)),
    )
    data = np.full((3, stamp_size, stamp_size), float(epoch_index + 1), dtype=np.float32)
    return CutoutTriplet(
        target_id=object_id,
        tracklet_id=tracklet_id,
        epoch_index=epoch_index,
        timestamp_utc=datetime.now(UTC),
        bjd_tdb=2459000.5 + (epoch_index * 0.01),
        ra_deg=180.0 + (epoch_index * 0.001),
        dec_deg=15.0,
        target_x=stamp_size / 2.0,
        target_y=stamp_size / 2.0,
        wcs_params=wcs,
        stamp_size=stamp_size,
        filter_band="r",
        survey="ZTF",
        snr=10.0,
        mag_apparent=20.0,
        is_synthetic=False,
        is_artifact=False,
        artifact_type="NONE",
        data=data,
    )


def test_storage_v2_initialization(tmp_path: Path) -> None:
    """Verify HDF5 archive initializes with schema 2.1.0, 144x144 stamps, and 4 splits."""
    h5_path = tmp_path / "test_v2.h5"
    manager = HDF5TrainingStorageManager(file_path=h5_path, stamp_size=144)

    assert h5_path.exists()
    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["schema_version"] == "2.1.0"
        assert h5.attrs["stamp_size"] == 144
        splits_grp = h5["splits"]
        for split in ("train", "val", "test", "ood"):
            assert split in splits_grp
            grp = splits_grp[split]
            assert grp["triplets"].shape == (0, 3, 144, 144)
            assert grp["triplets"].chunks == (16, 3, 144, 144)
            assert grp["tracklet_offsets"].shape == (0,)


def test_manifest_based_routing_and_keyerror(tmp_path: Path) -> None:
    """Verify route_target_to_split resolves splits via manifest and raises KeyError on unlisted targets."""
    h5_path = tmp_path / "test_routing.h5"
    manager = HDF5TrainingStorageManager(file_path=h5_path, stamp_size=144)

    # Observable ISO targets must strictly route to ood
    assert manager.route_target_to_split("2I") == "ood"
    assert manager.route_target_to_split("3I") == "ood"

    # Known catalog targets from configs/dataset_splits.json
    assert manager.route_target_to_split("99942") == "test"
    assert manager.route_target_to_split("101955") == "train"

    # Prefix support for hard negatives
    assert manager.route_target_to_split("BOGUS_99942") == "test"

    # Strict KeyError policy on unknown targets
    with pytest.raises(KeyError, match="not mapped"):
        manager.route_target_to_split("NON_EXISTENT_ASTEROID_999")


def test_atomic_append_with_crawler_ledger(tmp_path: Path) -> None:
    """Verify two-phase commit: HDF5 CSR atomic append + SQLiteCrawlerLedger transaction."""
    h5_path = tmp_path / "test_atomic.h5"
    db_path = tmp_path / "test_ledger.db"

    ledger = SQLiteCrawlerLedger(db_path=db_path)
    manager = HDF5TrainingStorageManager(file_path=h5_path, stamp_size=144, ledger=ledger)

    # Sample 1: 101955 (routes to train) with 3 frames
    sample_train = TrackletSample(
        tracklet_id="TRK_TRAIN_01",
        object_id="101955",
        taxonomy_class="NEO",
        label=1,
        frames=[_make_dummy_frame("101955", "TRK_TRAIN_01", i, stamp_size=144) for i in range(3)],
    )

    # Sample 2: 2I (routes to ood) with 2 frames
    sample_ood = TrackletSample(
        tracklet_id="TRK_OOD_01",
        object_id="2I",
        taxonomy_class="ISO",
        label=3,
        frames=[_make_dummy_frame("2I", "TRK_OOD_01", i, stamp_size=144) for i in range(2)],
    )

    counts = manager.append_tracklets([sample_train, sample_ood])
    assert counts["train"] == 1
    assert counts["ood"] == 1
    assert counts["val"] == 0
    assert counts["test"] == 0

    # Verify HDF5 contents
    assert manager.get_tracklet_count("train") == 1
    assert manager.get_tracklet_count("ood") == 1

    triplets_train, meta_train = manager.get_tracklet("train", 0)
    assert triplets_train.shape == (3, 3, 144, 144)
    assert meta_train["tracklet_id"] == "TRK_TRAIN_01"

    triplets_ood, meta_ood = manager.get_tracklet("ood", 0)
    assert triplets_ood.shape == (2, 3, 144, 144)
    assert meta_ood["tracklet_id"] == "TRK_OOD_01"

    # Verify SQLite ledger state transitions
    assert ledger.is_tracklet_completed("101955", "TRK_TRAIN_01")
    assert ledger.is_tracklet_completed("2I", "TRK_OOD_01")
    assert ("101955", "TRK_TRAIN_01") in ledger.get_completed_ids()
    assert ("2I", "TRK_OOD_01") in ledger.get_completed_ids()


def test_repair_hdf5_csr_integrity_orphan_truncation(tmp_path: Path) -> None:
    """Verify repair_hdf5_csr_integrity detects uncommitted orphan frames and cleanly truncates."""
    h5_path = tmp_path / "test_repair.h5"
    manager = HDF5TrainingStorageManager(file_path=h5_path, stamp_size=144)

    # Append 1 valid tracklet of 3 frames to train
    sample = TrackletSample(
        tracklet_id="TRK_VALID_01",
        object_id="101955",
        taxonomy_class="NEO",
        label=1,
        frames=[_make_dummy_frame("101955", "TRK_VALID_01", i, stamp_size=144) for i in range(3)],
    )
    manager.append_tracklets([sample])

    # Simulate crash mid-append: inject 2 orphan frames into triplets & frame datasets without committing offset
    with h5py.File(h5_path, "a") as h5:
        grp = h5["splits"]["train"]
        curr_n = grp["triplets"].shape[0]
        assert curr_n == 3
        # Resize to 5 (2 orphan frames)
        for ds_name in manager.FRAME_DATASETS:
            ds = grp[ds_name]
            if ds.ndim == 4:
                ds.resize((5, 3, 144, 144))
            elif ds.ndim == 3:
                ds.resize((5, 2, 2))
            elif ds.ndim == 2:
                ds.resize((5, 2))
            else:
                ds.resize((5,))

    # Verify file now has orphan frames
    with h5py.File(h5_path, "r") as h5:
        assert h5["splits"]["train"]["triplets"].shape[0] == 5
        assert h5["splits"]["train"]["tracklet_offsets"].shape[0] == 1  # only 1 tracklet committed

    # Run repair routine
    repairs = manager.repair_hdf5_csr_integrity()
    assert repairs["train"] == 2  # 2 orphan frames truncated

    # Verify cleanly restored to 3 frames
    with h5py.File(h5_path, "r") as h5:
        assert h5["splits"]["train"]["triplets"].shape[0] == 3
        assert h5["splits"]["train"]["bjd_tdb"].shape[0] == 3

    # Verify slicing still works cleanly
    triplets, meta = manager.get_tracklet("train", 0)
    assert triplets.shape == (3, 3, 144, 144)
    assert meta["length"] == 3
