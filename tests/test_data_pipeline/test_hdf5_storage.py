"""Tests for HDF5 ground truth storage container and GroundTruthBatch schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from LUMIA.core.schemas import GroundTruthBatch
from LUMIA.data_pipeline.hdf5_storage import HDF5GroundTruthWriter

if TYPE_CHECKING:
    from pathlib import Path


def _make_sample_batch(
    n: int = 5,
    *,
    stamp_size: int = 31,
    cutout_dtype: np.dtype[Any] = np.dtype(np.float32),
    label_dtype: np.dtype[Any] = np.dtype(np.int8),
    cutout_channels: int = 3,
    label_count: int | None = None,
    meta_count: int | None = None,
    inject_nan: bool = False,
    missing_meta_key: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to generate dictionary arguments for GroundTruthBatch."""
    m_count = n if meta_count is None else meta_count
    l_count = n if label_count is None else label_count

    cutouts = np.zeros((n, cutout_channels, stamp_size, stamp_size), dtype=cutout_dtype)
    if n > 0:
        cutouts[0, 0, 0, 0] = 1.0
        if inject_nan:
            cutouts[0, 0, 0, 0] = np.nan

    labels = np.ones(l_count, dtype=label_dtype)

    metadata: dict[str, Any] = {
        "bjd_tdb": [2460755.5 + float(i) * 0.01 for i in range(m_count)],
        "ra_deg": [180.0 + float(i) * 0.1 for i in range(m_count)],
        "dec_deg": [15.0 + float(i) * 0.1 for i in range(m_count)],
        "candidate_id": [f"cand_{i:04d}" for i in range(m_count)],
    }

    if missing_meta_key:
        metadata.pop(missing_meta_key, None)

    if extra_meta:
        metadata.update(extra_meta)

    return {
        "cutouts": cutouts,
        "labels": labels,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# 1. GroundTruthBatch Pydantic Contract Validation Tests
# ---------------------------------------------------------------------------


class TestGroundTruthBatchContract:
    """Verify strict validation, dimension alignment, and typing for GroundTruthBatch."""

    def test_valid_batch_initialization(self) -> None:
        """Verify successful instantiation and property access with valid data."""
        data = _make_sample_batch(n=10)
        batch = GroundTruthBatch(**data)

        assert batch.batch_size == 10
        assert batch.cutouts.shape == (10, 3, 31, 31)
        assert batch.labels.shape == (10,)
        assert len(batch.metadata["candidate_id"]) == 10
        assert batch.metadata["candidate_id"][0] == "cand_0000"

    def test_empty_batch_is_allowed(self) -> None:
        """Verify empty batch (N=0) initializes cleanly."""
        data = _make_sample_batch(n=0)
        batch = GroundTruthBatch(**data)

        assert batch.batch_size == 0
        assert batch.cutouts.shape == (0, 3, 31, 31)
        assert batch.labels.shape == (0,)

    @pytest.mark.parametrize(
        ("n_cutouts", "n_labels"),
        [
            (10, 9),
            (5, 6),
            (1, 0),
        ],
    )
    def test_labels_dimension_mismatch_raises_validation_error(
        self, n_cutouts: int, n_labels: int
    ) -> None:
        """Verify ValidationError when labels length does not match cutouts batch size."""
        data = _make_sample_batch(n=n_cutouts, label_count=n_labels)
        with pytest.raises(ValidationError, match="Batch dimension mismatch"):
            GroundTruthBatch(**data)

    @pytest.mark.parametrize(
        ("n_cutouts", "n_meta"),
        [
            (10, 8),
            (5, 7),
            (4, 0),
        ],
    )
    def test_metadata_dimension_mismatch_raises_validation_error(
        self, n_cutouts: int, n_meta: int
    ) -> None:
        """Verify ValidationError when metadata arrays length does not match cutouts batch size."""
        data = _make_sample_batch(n=n_cutouts, meta_count=n_meta)
        with pytest.raises(ValidationError, match="Batch dimension mismatch"):
            GroundTruthBatch(**data)

    @pytest.mark.parametrize(
        "missing_key",
        ["bjd_tdb", "ra_deg", "dec_deg", "candidate_id"],
    )
    def test_missing_required_metadata_key_raises_validation_error(
        self, missing_key: str
    ) -> None:
        """Verify ValidationError when any required metadata column is absent."""
        data = _make_sample_batch(n=5, missing_meta_key=missing_key)
        with pytest.raises(ValidationError, match=f"missing required key '{missing_key}'"):
            GroundTruthBatch(**data)

    def test_wrong_spatial_dimensions_rejected(self) -> None:
        """Verify stamps not sized exactly 31x31 fail validation."""
        data = _make_sample_batch(n=5, stamp_size=51)
        with pytest.raises(ValidationError, match="spatial dimensions .* must be \\(31, 31\\)"):
            GroundTruthBatch(**data)

    def test_wrong_channels_count_rejected(self) -> None:
        """Verify cutouts with channel count != 3 fail validation."""
        data = _make_sample_batch(n=5, cutout_channels=1)
        with pytest.raises(ValidationError, match="must have exactly 3 channels"):
            GroundTruthBatch(**data)

    def test_non_float32_cutouts_rejected(self) -> None:
        """Verify cutouts must strictly be float32."""
        data = _make_sample_batch(n=5, cutout_dtype=np.dtype(np.float64))
        with pytest.raises(ValidationError, match="cutouts dtype must be float32"):
            GroundTruthBatch(**data)

    def test_non_int8_labels_rejected(self) -> None:
        """Verify labels must strictly be int8."""
        data = _make_sample_batch(n=5, label_dtype=np.dtype(np.int64))
        with pytest.raises(ValidationError, match="labels dtype must be int8"):
            GroundTruthBatch(**data)

    def test_non_finite_cutouts_rejected(self) -> None:
        """Verify NaN or Inf in cutouts tensor triggers ValidationError."""
        data = _make_sample_batch(n=5, inject_nan=True)
        with pytest.raises(ValidationError, match="non-finite"):
            GroundTruthBatch(**data)

    def test_empty_candidate_id_rejected(self) -> None:
        """Verify blank candidate_id string fails validation."""
        data = _make_sample_batch(n=3)
        data["metadata"]["candidate_id"][1] = ""
        with pytest.raises(ValidationError, match="candidate_id values must be non-empty"):
            GroundTruthBatch(**data)


# ---------------------------------------------------------------------------
# 2. HDF5 Storage Writer Unit Tests (Isolated via tmp_path)
# ---------------------------------------------------------------------------


class TestHDF5GroundTruthWriter:
    """Verify out-of-core file creation, dataset schema, chunking, and incremental appending."""

    def test_initial_batch_append_creates_file_and_datasets(self, tmp_path: Path) -> None:
        """Verify append_batch creates groups, chunked datasets, and softlinks."""
        archive_path = tmp_path / "dataset_v1.h5"
        writer = HDF5GroundTruthWriter(archive_path, chunk_size=64)

        batch = GroundTruthBatch(**_make_sample_batch(n=8))
        total_samples = writer.append_batch(batch, split="train")

        assert total_samples == 8
        assert archive_path.is_file()

        with h5py.File(archive_path, "r") as h5:
            # File root attributes
            assert h5.attrs["schema_version"] == "1.0.0"
            assert h5.attrs["stamp_size"] == 31
            assert list(h5.attrs["channel_order"]) == ["science", "reference", "difference_zogy"]

            # Train split group
            train_grp = h5["train"]
            assert "cutouts" in train_grp
            assert "labels" in train_grp
            assert "candidate_id" in train_grp
            assert "bjd_tdb" in train_grp
            assert "ra_deg" in train_grp
            assert "dec_deg" in train_grp

            # Dataset properties & chunking
            cutouts_ds = train_grp["cutouts"]
            assert cutouts_ds.shape == (8, 3, 31, 31)
            assert cutouts_ds.chunks == (64, 3, 31, 31)
            assert cutouts_ds.compression == "lzf"
            assert cutouts_ds.dtype == np.float32

            labels_ds = train_grp["labels"]
            assert labels_ds.shape == (8,)
            assert labels_ds.dtype == np.int8

            # Fixed-length ASCII string dtype check
            cand_ds = train_grp["candidate_id"]
            assert cand_ds.shape == (8,)
            assert h5py.check_string_dtype(cand_ds.dtype).encoding == "ascii"
            cand_values = [s.decode("ascii") for s in cand_ds[:]]
            assert cand_values == [f"cand_{i:04d}" for i in range(8)]

            # Softlink compatibility
            assert "triplets" in train_grp
            assert np.array_equal(train_grp["triplets"][:], cutouts_ds[:])
            assert "candidate_ids" in train_grp
            assert np.array_equal(train_grp["candidate_ids"][:], cand_ds[:])

    def test_multiple_appends_resize_and_concatenate(self, tmp_path: Path) -> None:
        """Verify sequential appends dynamically resize datasets without data corruption."""
        archive_path = tmp_path / "dataset_multi.h5"
        writer = HDF5GroundTruthWriter(archive_path, chunk_size=32)

        batch1 = GroundTruthBatch(**_make_sample_batch(n=10))
        batch2 = GroundTruthBatch(
            **_make_sample_batch(
                n=15,
                extra_meta={"snr": [10.5 + float(i) for i in range(15)]},
            )
        )

        count1 = writer.append_batch(batch1, split="train")
        assert count1 == 10

        count2 = writer.append_batch(batch2, split="train")
        assert count2 == 25

        assert writer.get_sample_count("train") == 25
        assert writer.get_dataset_shape("train", "cutouts") == (25, 3, 31, 31)

        with h5py.File(archive_path, "r") as h5:
            cand_ids = [s.decode("ascii") for s in h5["train/candidate_id"][:]]
            assert len(cand_ids) == 25
            assert cand_ids[:10] == [f"cand_{i:04d}" for i in range(10)]
            assert cand_ids[10:] == [f"cand_{i:04d}" for i in range(15)]

    def test_multi_split_independence(self, tmp_path: Path) -> None:
        """Verify train, val, and test splits remain completely separate in storage."""
        archive_path = tmp_path / "dataset_splits.h5"
        writer = HDF5GroundTruthWriter(archive_path)

        train_batch = GroundTruthBatch(**_make_sample_batch(n=20))
        val_batch = GroundTruthBatch(**_make_sample_batch(n=5))
        test_batch = GroundTruthBatch(**_make_sample_batch(n=3))

        writer.append_batch(train_batch, split="train")
        writer.append_batch(val_batch, split="val")
        writer.append_batch(test_batch, split="test")

        assert writer.get_sample_count("train") == 20
        assert writer.get_sample_count("val") == 5
        assert writer.get_sample_count("test") == 3

        with h5py.File(archive_path, "r") as h5:
            assert h5["train/cutouts"].shape == (20, 3, 31, 31)
            assert h5["val/cutouts"].shape == (5, 3, 31, 31)
            assert h5["test/cutouts"].shape == (3, 3, 31, 31)

    def test_nested_directory_creation(self, tmp_path: Path) -> None:
        """Verify writer automatically creates nested parent directories on initialization."""
        nested_file = tmp_path / "deep" / "nested" / "ground_truth" / "dataset.h5"
        assert not nested_file.parent.exists()

        writer = HDF5GroundTruthWriter(nested_file)
        assert nested_file.parent.is_dir()

        batch = GroundTruthBatch(**_make_sample_batch(n=4))
        writer.append_batch(batch, split="train")
        assert nested_file.is_file()

    def test_roundtrip_read_batch(self, tmp_path: Path) -> None:
        """Verify read_batch re-hydrates valid GroundTruthBatch instances from HDF5."""
        archive_path = tmp_path / "dataset_roundtrip.h5"
        writer = HDF5GroundTruthWriter(archive_path)

        original_batch = GroundTruthBatch(**_make_sample_batch(n=12))
        writer.append_batch(original_batch, split="train")

        loaded_batch = writer.read_batch("train", start=2, stop=8)
        assert isinstance(loaded_batch, GroundTruthBatch)
        assert loaded_batch.batch_size == 6
        assert loaded_batch.cutouts.shape == (6, 3, 31, 31)
        assert loaded_batch.labels.shape == (6,)
        assert loaded_batch.metadata["candidate_id"][0] == "cand_0002"
        assert loaded_batch.metadata["candidate_id"][-1] == "cand_0007"
        assert np.isclose(
            loaded_batch.metadata["bjd_tdb"][0],
            original_batch.metadata["bjd_tdb"][2],
        )

    def test_empty_batch_append_is_noop(self, tmp_path: Path) -> None:
        """Verify appending an empty batch does not raise errors or corrupt existing data."""
        archive_path = tmp_path / "dataset_empty.h5"
        writer = HDF5GroundTruthWriter(archive_path)

        # 1. Append empty batch into fresh file
        empty_batch = GroundTruthBatch(**_make_sample_batch(n=0))
        res0 = writer.append_batch(empty_batch, split="train")
        assert res0 == 0
        assert writer.get_sample_count("train") == 0

        # 2. Append real data
        real_batch = GroundTruthBatch(**_make_sample_batch(n=5))
        res1 = writer.append_batch(real_batch, split="train")
        assert res1 == 5

        # 3. Append empty data to existing file
        res2 = writer.append_batch(empty_batch, split="train")
        assert res2 == 5
        assert writer.get_sample_count("train") == 5

    def test_empty_split_name_raises_value_error(self, tmp_path: Path) -> None:
        """Verify empty split name raises ValueError."""
        archive_path = tmp_path / "dataset_split_err.h5"
        writer = HDF5GroundTruthWriter(archive_path)
        batch = GroundTruthBatch(**_make_sample_batch(n=2))

        with pytest.raises(ValueError, match="split name cannot be empty"):
            writer.append_batch(batch, split="   ")
