"""PyTorch Dataset and DataLoader manager for out-of-core CSR-indexed ZTF tracklets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from LUMIA.data_mining.augmentation import apply_d4_transform_with_coords
from LUMIA.data_mining.storage import (
    DEFAULT_HDF5_STORAGE_PATH,
    SplitName,
)

if TYPE_CHECKING:
    from LUMIA.data_mining.storage import HDF5TrainingStorageManager

logger = logging.getLogger(__name__)

DEFAULT_ZTF_DATASET_PATH = DEFAULT_HDF5_STORAGE_PATH


class ZTFTrackletDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset interfacing with variable-length CSR-indexed HDF5 tracklet storage.

    Features:
    - Out-of-core streaming: slices arbitrary N-frame sequences in O(1) time via CSR index.
    - Zero padding at dataset level: each item yields (N, 3, 128, 128) where N is native length.
    - Sequence-synchronized on-the-fly random spatial cropping (144x144 -> 128x128) for training.
    - Deterministic centered cropping (144x144 -> 128x128) for validation, test, and OOD splits.
    - Exact astrometric coordinate translation and WCS CRPIX shifts preserving apparent velocities.
    - On-the-fly D4 dihedral augmentation: transforms all frames and target coordinates
      consistently in-memory during batch generation without disk expansion.
    """

    def __init__(
        self,
        h5_path: Path | str | None = None,
        storage_manager: HDF5TrainingStorageManager | None = None,
        split: SplitName = "train",
        augment: bool = False,
        is_train: bool | None = None,
        jitter: bool = True,
        jitter_offset: tuple[int, int] | None = None,
    ) -> None:
        """Initialize out-of-core ZTF tracklet dataset.

        Args:
            h5_path: Path to the CSR HDF5 archive file.
            storage_manager: Optional existing storage manager instance.
            split: Target dataset partition ("train", "val", "test", or "ood").
            augment: Whether to apply randomized in-memory D4 dihedral symmetries.
            is_train: Whether dataset runs in training mode (enables randomized jitter crop).
                Defaults to True if split == "train", False otherwise.
            jitter: Whether to apply spatial jitter cropping on 144x144 inputs.
            jitter_offset: Optional deterministic (dx, dy) offset for testing.
        """
        super().__init__()
        self._h5_file: h5py.File | None = None

        if storage_manager is not None:
            self.h5_path = Path(storage_manager.file_path)
            self._storage_manager = storage_manager
        elif h5_path is not None:
            self.h5_path = Path(h5_path)
            self._storage_manager = None
        else:
            self.h5_path = DEFAULT_ZTF_DATASET_PATH
            self._storage_manager = None

        if not self.h5_path.exists():
            raise FileNotFoundError(f"ZTF tracklet HDF5 archive not found at: {self.h5_path}")

        self.split: SplitName = split
        self.augment: bool = augment
        self.is_train: bool = (split == "train") if is_train is None else is_train
        self.jitter: bool = jitter
        self.jitter_offset: tuple[int, int] | None = jitter_offset

        # Read dataset length and schema metadata once at initialization
        with h5py.File(self.h5_path, "r") as h5:
            if "splits" not in h5 or self.split not in h5["splits"]:
                raise KeyError(f"Split '{self.split}' not found in HDF5 archive: {self.h5_path}")
            grp = h5["splits"][self.split]
            if "tracklet_offsets" not in grp:
                raise KeyError(f"'tracklet_offsets' dataset missing from split '{self.split}'")
            self._length = int(grp["tracklet_offsets"].shape[0])
            self.stamp_size = int(h5.attrs.get("stamp_size", 144))
            self.schema_version = str(h5.attrs.get("schema_version", "1.0.0"))

        self._h5_file: h5py.File | None = None

    def __len__(self) -> int:
        """Return the number of variable-length tracklets in this partition."""
        return self._length

    def _get_h5_handle(self) -> h5py.File:
        """Get or lazily open the HDF5 file handle for worker process safety."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Slice an arbitrary variable-length tracklet in O(1) time via CSR offsets.

        Returns:
            Dictionary containing:
            - "tracklet": torch.FloatTensor of shape (N, 3, 128, 128)
            - "label": torch.LongTensor scalar with target classification label
            - "target_coords": torch.FloatTensor of shape (N, 2) with (x, y) coordinates
            - "tracklet_id": str unique tracklet identifier
            - "object_id": str target body identifier
            - "taxonomy_class": str dynamical population name
            - "length": int number of frames N in tracklet
            - "bjd_tdb": torch.DoubleTensor of shape (N,) with mid-exposure epochs
            - "snr": torch.FloatTensor of shape (N,) with signal-to-noise ratios
            - "jitter_offset": torch.IntTensor of shape (2,) with [dx, dy] shift
            - "wcs_crpix": torch.FloatTensor of shape (N, 2) with WCS reference pixels (if present)
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Tracklet index {idx} out of range [0, {self._length})")

        h5 = self._get_h5_handle()
        grp = h5["splits"][self.split]

        offset = int(grp["tracklet_offsets"][idx])
        length = int(grp["tracklet_lengths"][idx])
        label = int(grp["tracklet_labels"][idx])
        obj_id = grp["tracklet_object_ids"][idx].decode("ascii")
        taxonomy = grp["tracklet_taxonomy"][idx].decode("ascii")
        t_id = grp["tracklet_ids"][idx].decode("ascii")

        f_slice = slice(offset, offset + length)
        triplets = np.array(grp["triplets"][f_slice], dtype=np.float32)
        txs = np.array(grp["target_x"][f_slice], dtype=np.float32)
        tys = np.array(grp["target_y"][f_slice], dtype=np.float32)
        bjds = np.array(grp["bjd_tdb"][f_slice], dtype=np.float64)
        snrs = np.array(grp["snr"][f_slice], dtype=np.float32)
        has_wcs_crpix = "wcs_crpix" in grp
        crpix = (
            np.array(grp["wcs_crpix"][f_slice], dtype=np.float32)
            if has_wcs_crpix
            else None
        )

        # Apply sequence-synchronized spatial crop jitter (144x144 -> 128x128)
        _, _, h, w = triplets.shape
        dx = 0
        dy = 0
        if h == 144 and w == 144:
            if self.jitter_offset is not None:
                dx, dy = self.jitter_offset
            elif self.is_train and self.jitter:
                dx = int(np.random.randint(-8, 9))
                dy = int(np.random.randint(-8, 9))
            else:
                dx = 0
                dy = 0

            x_start = 8 + dx
            y_start = 8 + dy
            triplets = triplets[:, :, y_start : y_start + 128, x_start : x_start + 128].copy()
            txs = txs - x_start
            tys = tys - y_start
            if crpix is not None:
                crpix = crpix.copy()
                crpix[:, 0] -= x_start
                crpix[:, 1] -= y_start

        # Apply on-the-fly D4 dihedral augmentation uniformly across all frames
        if self.augment:
            op = int(np.random.randint(0, 8))
            if op > 0:
                aug_triplets = np.empty_like(triplets)
                aug_txs = np.empty_like(txs)
                aug_tys = np.empty_like(tys)
                for i in range(length):
                    t_img, (cx, cy) = apply_d4_transform_with_coords(
                        triplets[i],
                        float(txs[i]),
                        float(tys[i]),
                        op=op,  # type: ignore[arg-type]
                    )
                    aug_triplets[i] = t_img
                    aug_txs[i] = cx
                    aug_tys[i] = cy
                triplets = aug_triplets
                txs = aug_txs
                tys = aug_tys

        coords = np.stack([txs, tys], axis=-1).astype(np.float32)

        result: dict[str, Any] = {
            "tracklet": torch.from_numpy(triplets).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "target_coords": torch.from_numpy(coords).float(),
            "tracklet_id": t_id,
            "object_id": obj_id,
            "taxonomy_class": taxonomy,
            "length": length,
            "bjd_tdb": torch.from_numpy(bjds).double(),
            "snr": torch.from_numpy(snrs).float(),
            "jitter_offset": torch.tensor([dx, dy], dtype=torch.int32),
        }
        if crpix is not None:
            result["wcs_crpix"] = torch.from_numpy(crpix).float()

        return result

    def close(self) -> None:
        """Close the underlying HDF5 file handle."""
        h5 = getattr(self, "_h5_file", None)
        if h5 is not None:
            h5.close()
            self._h5_file = None

    def __enter__(self) -> ZTFTrackletDataset:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def pad_tracklet_batch(
    batch: list[dict[str, Any]],
    pad_value: float = 0.0,
) -> dict[str, Any]:
    """Collate variable-length tracklet dictionaries into padded batch tensors.

    Args:
        batch: List of sample dictionaries returned by ZTFTrackletDataset.__getitem__.
        pad_value: Fill value for padded frames beyond native sequence length.

    Returns:
        Batched dictionary containing:
        - "tracklets": FloatTensor of shape (B, N_max, 3, 128, 128)
        - "labels": LongTensor of shape (B,)
        - "lengths": LongTensor of shape (B,) with native frame counts
        - "padding_mask": BoolTensor of shape (B, N_max) where True indicates valid frames
        - "target_coords": FloatTensor of shape (B, N_max, 2)
        - "tracklet_ids": list[str] of length B
        - "object_ids": list[str] of length B
        - "taxonomy_classes": list[str] of length B
        - "raw_tracklets": list of unpadded FloatTensors [(N_i, 3, 128, 128), ...]
        - "jitter_offsets": IntTensor of shape (B, 2)
        - "wcs_crpix": FloatTensor of shape (B, N_max, 2) (if present in batch)
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch of tracklets.")

    batch_size = len(batch)
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)

    first_tracklet: torch.Tensor = batch[0]["tracklet"]
    c, h, w = first_tracklet.shape[1:]

    padded_tracklets = torch.full(
        (batch_size, max_len, c, h, w),
        fill_value=pad_value,
        dtype=first_tracklet.dtype,
    )
    padded_coords = torch.zeros(
        (batch_size, max_len, 2),
        dtype=torch.float32,
    )
    has_crpix = "wcs_crpix" in batch[0]
    padded_crpix = (
        torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
        if has_crpix
        else None
    )

    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    jitter_offsets = (
        torch.stack([item["jitter_offset"] for item in batch], dim=0)
        if "jitter_offset" in batch[0]
        else None
    )

    for i, item in enumerate(batch):
        seq_len = item["length"]
        padded_tracklets[i, :seq_len] = item["tracklet"]
        padded_coords[i, :seq_len] = item["target_coords"]
        padding_mask[i, :seq_len] = True
        if padded_crpix is not None and "wcs_crpix" in item:
            padded_crpix[i, :seq_len] = item["wcs_crpix"]

    result: dict[str, Any] = {
        "tracklets": padded_tracklets,
        "labels": labels,
        "lengths": lengths_tensor,
        "padding_mask": padding_mask,
        "target_coords": padded_coords,
        "tracklet_ids": [item["tracklet_id"] for item in batch],
        "object_ids": [item["object_id"] for item in batch],
        "taxonomy_classes": [item["taxonomy_class"] for item in batch],
        "raw_tracklets": [item["tracklet"] for item in batch],
    }
    if jitter_offsets is not None:
        result["jitter_offsets"] = jitter_offsets
    if padded_crpix is not None:
        result["wcs_crpix"] = padded_crpix

    return result


def ztf_tracklet_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Standard collate function for PyTorch DataLoader batching of ZTF tracklets."""
    return pad_tracklet_batch(batch, pad_value=0.0)


def create_ztf_tracklet_dataloader(
    h5_path: Path | str = DEFAULT_ZTF_DATASET_PATH,
    split: SplitName = "train",
    batch_size: int = 16,
    shuffle: bool | None = None,
    augment: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    is_train: bool | None = None,
    jitter: bool = True,
) -> DataLoader[dict[str, Any]]:
    """Factory creating an optimized PyTorch DataLoader for variable-length ZTF tracklets.

    Args:
        h5_path: Path to the out-of-core CSR HDF5 archive.
        split: Dataset split ("train", "val", "test", or "ood").
        batch_size: Number of tracklets per batch.
        shuffle: Whether to shuffle tracklets (defaults to True for "train", False otherwise).
        augment: Whether to apply D4 dihedral augmentations (defaults to True for "train").
        num_workers: Number of worker subprocesses for data loading.
        pin_memory: If True, copies tensors into CUDA pinned memory before returning.
        is_train: Whether running in training mode for jitter cropping.
        jitter: Whether to apply spatial jitter cropping on 144x144 cutouts.

    Returns:
        Configured PyTorch DataLoader instance.
    """
    train_mode = (split == "train") if is_train is None else is_train
    do_shuffle = train_mode if shuffle is None else shuffle
    do_augment = train_mode if augment is None else augment

    dataset = ZTFTrackletDataset(
        h5_path=h5_path,
        split=split,
        augment=do_augment,
        is_train=train_mode,
        jitter=jitter,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        collate_fn=ztf_tracklet_collate_fn,
        pin_memory=pin_memory,
    )
