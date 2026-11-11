"""HDF5 ground truth storage container for out-of-core CNN training and triage feedback."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from LUMIA.core.schemas import GroundTruthBatch

if TYPE_CHECKING:
    pass

__all__ = [
    "HDF5GroundTruthWriter",
]


class HDF5GroundTruthWriter:
    """Out-of-core HDF5 storage manager for supervised Real/Bogus candidate cutouts.

    Appends chunked, compressed 3-channel stamps (Science, Reference, Difference),
    binary targets, and astronomical metadata into deterministic splits (train/val/test).
    """

    def __init__(self, file_path: Path | str, chunk_size: int = 64) -> None:
        """Initialize writer with target archive path.

        Parameters
        ----------
        file_path : Path | str
            Absolute or relative path to the target HDF5 archive (e.g. data/ground_truth/dataset_v1.h5).
        chunk_size : int, optional
            Primary chunk length along the sample axis (default 64).
        """
        self._file_path: Path = Path(file_path).resolve()
        self._chunk_size: int = chunk_size
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def file_path(self) -> Path:
        """Return canonical path to target HDF5 file."""
        return self._file_path

    @property
    def chunk_size(self) -> int:
        """Return chunk sample size."""
        return self._chunk_size

    def append_batch(self, batch: GroundTruthBatch, split: str = "train") -> int:
        """Append a validated GroundTruthBatch into the specified split group.

        Parameters
        ----------
        batch : GroundTruthBatch
            Validated Pydantic batch with shape (N, 3, 31, 31).
        split : str, optional
            Target split name, e.g. 'train', 'val', or 'test' (default 'train').

        Returns
        -------
        int
            Updated total sample count in the target split.
        """
        split_name = split.strip().strip("/")
        if not split_name:
            msg = "split name cannot be empty."
            raise ValueError(msg)

        n_samples = batch.batch_size
        ascii_dt = h5py.string_dtype(encoding="ascii", length=128)

        with h5py.File(self._file_path, "a") as h5_file:
            # Initialize root metadata attributes if newly created
            if "schema_version" not in h5_file.attrs:
                h5_file.attrs["schema_version"] = "1.0.0"
                h5_file.attrs["stamp_size"] = 31
                h5_file.attrs["channel_order"] = ["science", "reference", "difference_zogy"]

            split_group = h5_file.require_group(split_name)

            # 1. Primary Cutouts Tensor (N, 3, 31, 31)
            if "cutouts" not in split_group:
                if n_samples > 0:
                    split_group.create_dataset(
                        "cutouts",
                        data=batch.cutouts,
                        maxshape=(None, 3, 31, 31),
                        chunks=(self._chunk_size, 3, 31, 31),
                        compression="lzf",
                        shuffle=True,
                        dtype=np.float32,
                    )
                else:
                    split_group.create_dataset(
                        "cutouts",
                        shape=(0, 3, 31, 31),
                        maxshape=(None, 3, 31, 31),
                        chunks=(self._chunk_size, 3, 31, 31),
                        compression="lzf",
                        shuffle=True,
                        dtype=np.float32,
                    )
                if "triplets" not in split_group:
                    split_group["triplets"] = h5py.SoftLink(f"/{split_name}/cutouts")
            elif n_samples > 0:
                cutouts_ds = split_group["cutouts"]
                old_len = cutouts_ds.shape[0]
                cutouts_ds.resize((old_len + n_samples, 3, 31, 31))
                cutouts_ds[old_len : old_len + n_samples] = batch.cutouts

            # 2. Binary Labels (N,)
            if "labels" not in split_group:
                if n_samples > 0:
                    split_group.create_dataset(
                        "labels",
                        data=batch.labels,
                        maxshape=(None,),
                        chunks=(self._chunk_size,),
                        compression="lzf",
                        dtype=np.int8,
                    )
                else:
                    split_group.create_dataset(
                        "labels",
                        shape=(0,),
                        maxshape=(None,),
                        chunks=(self._chunk_size,),
                        compression="lzf",
                        dtype=np.int8,
                    )
            elif n_samples > 0:
                labels_ds = split_group["labels"]
                old_len = labels_ds.shape[0]
                labels_ds.resize((old_len + n_samples,))
                labels_ds[old_len : old_len + n_samples] = batch.labels

            # 3. Metadata: Candidate IDs (Fixed-Length ASCII)
            raw_cand_ids = batch.metadata["candidate_id"]
            cand_id_bytes = np.array(
                [
                    s.encode("ascii", errors="replace") if isinstance(s, str) else s
                    for s in raw_cand_ids
                ],
                dtype="S128",
            )
            if "candidate_id" not in split_group:
                if n_samples > 0:
                    split_group.create_dataset(
                        "candidate_id",
                        data=cand_id_bytes,
                        maxshape=(None,),
                        chunks=(self._chunk_size,),
                        compression="lzf",
                        dtype=ascii_dt,
                    )
                else:
                    split_group.create_dataset(
                        "candidate_id",
                        shape=(0,),
                        maxshape=(None,),
                        chunks=(self._chunk_size,),
                        compression="lzf",
                        dtype=ascii_dt,
                    )
                if "candidate_ids" not in split_group:
                    split_group["candidate_ids"] = h5py.SoftLink(f"/{split_name}/candidate_id")
            elif n_samples > 0:
                cand_ds = split_group["candidate_id"]
                old_len = cand_ds.shape[0]
                cand_ds.resize((old_len + n_samples,))
                cand_ds[old_len : old_len + n_samples] = cand_id_bytes

            # 4. Metadata: Astrometric & Photometric Scalar Arrays
            for key in ("bjd_tdb", "ra_deg", "dec_deg"):
                arr = np.asarray(batch.metadata[key], dtype=np.float64)
                if key not in split_group:
                    if n_samples > 0:
                        split_group.create_dataset(
                            key,
                            data=arr,
                            maxshape=(None,),
                            chunks=(self._chunk_size,),
                            compression="lzf",
                            dtype=np.float64,
                        )
                    else:
                        split_group.create_dataset(
                            key,
                            shape=(0,),
                            maxshape=(None,),
                            chunks=(self._chunk_size,),
                            compression="lzf",
                            dtype=np.float64,
                        )
                elif n_samples > 0:
                    ds = split_group[key]
                    old_len = ds.shape[0]
                    ds.resize((old_len + n_samples,))
                    ds[old_len : old_len + n_samples] = arr

            # 5. Optional metadata fields (e.g. mag, snr)
            for key, val in batch.metadata.items():
                if key in ("candidate_id", "bjd_tdb", "ra_deg", "dec_deg"):
                    continue
                if isinstance(val, (list, tuple, np.ndarray)):
                    # Determine appropriate dtype
                    first_val = val[0] if len(val) > 0 else None
                    if isinstance(first_val, str) or isinstance(first_val, (bytes, np.bytes_)):
                        str_arr = np.array(
                            [
                                s.encode("ascii", errors="replace") if isinstance(s, str) else s
                                for s in val
                            ],
                            dtype="S128",
                        )
                        if key not in split_group:
                            split_group.create_dataset(
                                key,
                                data=str_arr if n_samples > 0 else np.empty(0, dtype="S128"),
                                maxshape=(None,),
                                chunks=(self._chunk_size,),
                                compression="lzf",
                                dtype=ascii_dt,
                            )
                        elif n_samples > 0:
                            ds = split_group[key]
                            old_len = ds.shape[0]
                            ds.resize((old_len + n_samples,))
                            ds[old_len : old_len + n_samples] = str_arr
                    else:
                        num_arr = np.asarray(val, dtype=np.float32)
                        if key not in split_group:
                            split_group.create_dataset(
                                key,
                                data=num_arr if n_samples > 0 else np.empty(0, dtype=np.float32),
                                maxshape=(None,),
                                chunks=(self._chunk_size,),
                                compression="lzf",
                                dtype=np.float32,
                            )
                        elif n_samples > 0:
                            ds = split_group[key]
                            old_len = ds.shape[0]
                            ds.resize((old_len + n_samples,))
                            ds[old_len : old_len + n_samples] = num_arr

            h5_file.flush()
            return int(split_group["cutouts"].shape[0])

    def get_sample_count(self, split: str = "train") -> int:
        """Return current sample count in specified split, or 0 if non-existent."""
        split_name = split.strip().strip("/")
        if not self._file_path.is_file():
            return 0
        with h5py.File(self._file_path, "r") as h5_file:
            if split_name not in h5_file or "cutouts" not in h5_file[split_name]:
                return 0
            return int(h5_file[split_name]["cutouts"].shape[0])

    def get_dataset_shape(
        self, split: str = "train", dataset_name: str = "cutouts"
    ) -> tuple[int, ...] | None:
        """Return shape of specified dataset, or None if it does not exist."""
        split_name = split.strip().strip("/")
        if not self._file_path.is_file():
            return None
        with h5py.File(self._file_path, "r") as h5_file:
            if split_name not in h5_file or dataset_name not in h5_file[split_name]:
                return None
            return tuple(int(d) for d in h5_file[split_name][dataset_name].shape)

    def read_batch(
        self, split: str = "train", start: int = 0, stop: int | None = None
    ) -> GroundTruthBatch:
        """Read a contiguous slice from HDF5 and return as a validated GroundTruthBatch."""
        split_name = split.strip().strip("/")
        if not split_name:
            msg = "split name cannot be empty."
            raise ValueError(msg)
        if not self._file_path.is_file():
            msg = f"Archive file does not exist: {self._file_path}"
            raise FileNotFoundError(msg)

        with h5py.File(self._file_path, "r") as h5_file:
            if split_name not in h5_file:
                msg = f"Split '{split_name}' does not exist in {self._file_path}"
                raise KeyError(msg)
            grp = h5_file[split_name]
            total = int(grp["cutouts"].shape[0])
            end = total if stop is None else min(stop, total)

            cutouts = grp["cutouts"][start:end].astype(np.float32)
            labels = grp["labels"][start:end].astype(np.int8)

            metadata: dict[str, Any] = {
                "bjd_tdb": grp["bjd_tdb"][start:end].astype(np.float64).tolist(),
                "ra_deg": grp["ra_deg"][start:end].astype(np.float64).tolist(),
                "dec_deg": grp["dec_deg"][start:end].astype(np.float64).tolist(),
                "candidate_id": [
                    item.decode("ascii") if isinstance(item, bytes) else str(item)
                    for item in grp["candidate_id"][start:end]
                ],
            }

            for key in grp.keys():
                if key in ("cutouts", "labels", "triplets", "candidate_ids") or key in metadata:
                    continue
                val = grp[key][start:end]
                if h5py.check_string_dtype(grp[key].dtype):
                    metadata[key] = [
                        item.decode("ascii") if isinstance(item, bytes) else str(item)
                        for item in val
                    ]
                else:
                    metadata[key] = val.tolist()

        return GroundTruthBatch(
            cutouts=cutouts,
            labels=labels,
            metadata=metadata,
        )
