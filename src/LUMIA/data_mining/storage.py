"""Out-of-core HDF5 storage manager for variable-length solar system tracklets."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
import zarr
from filelock import FileLock
from pydantic import ConfigDict, Field

from LUMIA.core.schemas import LumiaBaseModel
from LUMIA.data_mining.archive_crawler import CutoutTriplet  # noqa: TC001
from LUMIA.data_mining.crawler_ledger import CrawlerLedgerEntry, SQLiteCrawlerLedger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

SplitName = Literal["train", "val", "test", "ood"]

DEFAULT_HDF5_STORAGE_PATH = Path("data/ground_truth/ztf_sso_tracklets_144x144.h5")
DEFAULT_SPLITS_MANIFEST_PATH = Path("configs/dataset_splits.json")


class TrackletSample(LumiaBaseModel):
    """A variable-length time-ordered tracklet sequence of N >= 2 frames."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    tracklet_id: str = Field(..., min_length=1)
    object_id: str = Field(..., min_length=1)
    taxonomy_class: str = Field(default="MBA")
    label: int = Field(
        default=1,
        ge=0,
        le=3,
        description="0: Bogus/Hard Negative, 1: Point, 2: Streak, 3: Coma",
    )
    frames: list[CutoutTriplet] = Field(..., min_length=2)


TrackletSample.model_rebuild()

# Backward compatibility alias
AsteroidTrainingSample = TrackletSample


class HDF5TrainingStorageManager:
    """Manages creation, chunked append, and O(1) CSR tracklet slicing of HDF5 archives."""

    FRAME_DATASETS = (
        "triplets",
        "bjd_tdb",
        "ra_deg",
        "dec_deg",
        "target_x",
        "target_y",
        "wcs_crpix",
        "wcs_crval",
        "wcs_cd",
        "snr",
        "mag_apparent",
        "epoch_index",
        "tracklet_id",
        "is_synthetic",
        "provenance_survey",
    )

    TRACKLET_DATASETS = (
        "tracklet_offsets",
        "tracklet_lengths",
        "tracklet_labels",
        "tracklet_taxonomy",
        "tracklet_object_ids",
        "tracklet_ids",
    )

    def __init__(
        self,
        file_path: Path | str = DEFAULT_HDF5_STORAGE_PATH,
        stamp_size: int = 144,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        splits_manifest_path: Path | str | None = DEFAULT_SPLITS_MANIFEST_PATH,
        ledger: SQLiteCrawlerLedger | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.stamp_size = stamp_size
        self.split_ratios = split_ratios
        self.seed = seed
        self.splits_manifest_path = (
            Path(splits_manifest_path) if splits_manifest_path is not None else None
        )
        self.ledger = ledger
        self._lock_path = self.file_path.with_suffix(".h5.lock")
        self._manifest_cache: dict[str, SplitName] | None = None
        self._ensure_file_initialized()

    def route_target_to_split(self, object_id: str) -> SplitName:
        """Route target to partition based on configs/dataset_splits.json."""
        if self.splits_manifest_path is not None and self.splits_manifest_path.exists():
            if self._manifest_cache is None:
                with self.splits_manifest_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                splits_dict = data.get("splits", {})
                self._manifest_cache = {}
                for sp, targets in splits_dict.items():
                    for t in targets:
                        self._manifest_cache[t] = sp  # type: ignore[assignment]

            target_lookup = object_id
            if target_lookup in self._manifest_cache:
                return self._manifest_cache[target_lookup]

            # Strip prefixes if present
            if target_lookup.startswith("BOGUS_"):
                base_id = target_lookup[6:]
                if base_id in self._manifest_cache:
                    return self._manifest_cache[base_id]
            if target_lookup.startswith("SYNTH_"):
                base_id = target_lookup[6:]
                if base_id in self._manifest_cache:
                    return self._manifest_cache[base_id]

            raise KeyError(f"Target '{object_id}' is not mapped in dataset splits manifest.")

        # Deterministic cryptographic fallback if no manifest is configured
        h = hashlib.blake2b(f"{object_id}:{self.seed}".encode(), digest_size=4).digest()
        val = int.from_bytes(h, byteorder="big") % 10000 / 10000.0

        train_cutoff = self.split_ratios[0]
        val_cutoff = train_cutoff + self.split_ratios[1]

        if val < train_cutoff:
            return "train"
        if val < val_cutoff:
            return "val"
        return "test"

    def repair_hdf5_csr_integrity(self) -> dict[SplitName, int]:
        """Verify CSR tracklet offsets against triplets frame count and truncate orphan frames."""
        repairs: dict[SplitName, int] = {"train": 0, "val": 0, "test": 0, "ood": 0}
        if not self.file_path.exists():
            return repairs

        with FileLock(self._lock_path), h5py.File(self.file_path, "a") as h5:
            splits_grp = h5.get("splits")
            if splits_grp is None:
                return repairs

            for split in ("train", "val", "test", "ood"):
                if split not in splits_grp:
                    continue
                grp = splits_grp[split]
                n_offsets = grp["tracklet_offsets"].shape[0]
                n_triplets = grp["triplets"].shape[0]

                if n_offsets == 0:
                    expected_frames = 0
                else:
                    last_offset = int(grp["tracklet_offsets"][-1])
                    last_length = int(grp["tracklet_lengths"][-1])
                    expected_frames = last_offset + last_length

                if n_triplets > expected_frames:
                    orphan_count = n_triplets - expected_frames
                    repairs[split] = orphan_count  # type: ignore[index]
                    logger.warning(
                        "Repairing CSR integrity in split '%s': truncating %d orphan frames to %d",
                        split,
                        orphan_count,
                        expected_frames,
                    )
                    for ds_name in self.FRAME_DATASETS:
                        ds = grp[ds_name]
                        if ds.ndim == 4:
                            ds.resize((expected_frames, 3, self.stamp_size, self.stamp_size))
                        elif ds.ndim == 3:
                            ds.resize((expected_frames, 2, 2))
                        elif ds.ndim == 2:
                            ds.resize((expected_frames, 2))
                        else:
                            ds.resize((expected_frames,))

                elif n_triplets < expected_frames:
                    valid_n_trk = 0
                    for i in range(n_offsets):
                        off = int(grp["tracklet_offsets"][i])
                        lng = int(grp["tracklet_lengths"][i])
                        if off + lng <= n_triplets:
                            valid_n_trk = i + 1
                        else:
                            break
                    for ds_name in self.TRACKLET_DATASETS:
                        grp[ds_name].resize((valid_n_trk,))

        return repairs

    def _ensure_file_initialized(self) -> None:
        """Initialize the HDF5 archive layout if it does not already exist."""
        if self.file_path.exists():
            self.repair_hdf5_csr_integrity()
            return

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        s = self.stamp_size

        with FileLock(self._lock_path):
            if self.file_path.exists():
                self.repair_hdf5_csr_integrity()
                return

            with h5py.File(self.file_path, "w") as h5:
                h5.attrs["schema_version"] = "2.1.0"
                h5.attrs["stamp_size"] = s
                h5.attrs["channel_order"] = ["science", "reference", "difference"]
                h5.attrs["created_at_utc"] = datetime.now(UTC).isoformat()
                h5.attrs["split_seed"] = self.seed

                splits_grp = h5.create_group("splits")

                dt_str64 = h5py.string_dtype(encoding="ascii", length=64)
                dt_str32 = h5py.string_dtype(encoding="ascii", length=32)

                for split in ("train", "val", "test", "ood"):
                    grp = splits_grp.create_group(split)

                    # 1. Frame-Level Datasets
                    grp.create_dataset(
                        "triplets",
                        shape=(0, 3, s, s),
                        maxshape=(None, 3, s, s),
                        dtype=np.float32,
                        chunks=(16, 3, s, s),
                        compression="lzf",
                        shuffle=True,
                    )
                    grp.create_dataset(
                        "bjd_tdb",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float64,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "ra_deg",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float64,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "dec_deg",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float64,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "target_x",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float32,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "target_y",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float32,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "wcs_crpix",
                        shape=(0, 2),
                        maxshape=(None, 2),
                        dtype=np.float32,
                        chunks=(4096, 2),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "wcs_crval",
                        shape=(0, 2),
                        maxshape=(None, 2),
                        dtype=np.float64,
                        chunks=(4096, 2),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "wcs_cd",
                        shape=(0, 2, 2),
                        maxshape=(None, 2, 2),
                        dtype=np.float64,
                        chunks=(4096, 2, 2),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "snr",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float32,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "mag_apparent",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float32,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "epoch_index",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.int32,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "tracklet_id",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=dt_str64,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "is_synthetic",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=bool,
                        chunks=(4096,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "provenance_survey",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=dt_str32,
                        chunks=(4096,),
                        compression="lzf",
                    )

                    # 2. Tracklet Index CSR Datasets
                    grp.create_dataset(
                        "tracklet_offsets",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.int64,
                        chunks=(1024,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "tracklet_lengths",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.int32,
                        chunks=(1024,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "tracklet_labels",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.int32,
                        chunks=(1024,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "tracklet_taxonomy",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=dt_str32,
                        chunks=(1024,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "tracklet_object_ids",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=dt_str64,
                        chunks=(1024,),
                        compression="lzf",
                    )
                    grp.create_dataset(
                        "tracklet_ids",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=dt_str64,
                        chunks=(1024,),
                        compression="lzf",
                    )

    def append_tracklet(
        self,
        sample: TrackletSample,
        force_split: SplitName | None = None,
    ) -> SplitName:
        """Append a single variable-length tracklet to the archive."""
        split = (
            force_split if force_split is not None else self.route_target_to_split(sample.object_id)
        )
        self.append_tracklets([sample], force_split=split)
        return split

    def append_tracklets(
        self,
        samples: Sequence[TrackletSample],
        force_split: SplitName | None = None,
        ledger: SQLiteCrawlerLedger | None = None,
    ) -> dict[SplitName, int]:
        """Append a collection of variable-length tracklets indexed via CSR offsets."""
        if not samples:
            return {"train": 0, "val": 0, "test": 0, "ood": 0}

        partitioned: dict[SplitName, list[TrackletSample]] = {
            "train": [],
            "val": [],
            "test": [],
            "ood": [],
        }

        for smp in samples:
            target_split = (
                force_split
                if force_split is not None
                else self.route_target_to_split(smp.object_id)
            )
            partitioned[target_split].append(smp)

        counts = {k: len(v) for k, v in partitioned.items()}

        with FileLock(self._lock_path), h5py.File(self.file_path, "a") as h5:
            splits_grp = h5["splits"]

            for split, trk_list in partitioned.items():
                if not trk_list:
                    continue

                grp = splits_grp[split]

                # Current CSR tracklet and frame counts
                curr_n_tracklets = grp["tracklet_offsets"].shape[0]
                curr_n_frames = grp["triplets"].shape[0]

                new_n_tracklets = curr_n_tracklets + len(trk_list)
                total_new_frames = sum(len(t.frames) for t in trk_list)
                new_n_frames = curr_n_frames + total_new_frames

                # Resize tracklet index datasets
                grp["tracklet_offsets"].resize((new_n_tracklets,))
                grp["tracklet_lengths"].resize((new_n_tracklets,))
                grp["tracklet_labels"].resize((new_n_tracklets,))
                grp["tracklet_taxonomy"].resize((new_n_tracklets,))
                grp["tracklet_object_ids"].resize((new_n_tracklets,))
                grp["tracklet_ids"].resize((new_n_tracklets,))

                # Resize frame datasets
                grp["triplets"].resize((new_n_frames, 3, self.stamp_size, self.stamp_size))
                grp["bjd_tdb"].resize((new_n_frames,))
                grp["ra_deg"].resize((new_n_frames,))
                grp["dec_deg"].resize((new_n_frames,))
                grp["target_x"].resize((new_n_frames,))
                grp["target_y"].resize((new_n_frames,))
                grp["wcs_crpix"].resize((new_n_frames, 2))
                grp["wcs_crval"].resize((new_n_frames, 2))
                grp["wcs_cd"].resize((new_n_frames, 2, 2))
                grp["snr"].resize((new_n_frames,))
                grp["mag_apparent"].resize((new_n_frames,))
                grp["epoch_index"].resize((new_n_frames,))
                grp["tracklet_id"].resize((new_n_frames,))
                grp["is_synthetic"].resize((new_n_frames,))
                grp["provenance_survey"].resize((new_n_frames,))

                # Prepare tracklet-level arrays
                offsets: list[int] = []
                lengths: list[int] = []
                labels: list[int] = []
                taxonomies: list[bytes] = []
                obj_ids: list[bytes] = []
                trk_ids: list[bytes] = []

                # Prepare frame-level lists
                frame_triplets: list[np.ndarray] = []
                frame_bjd: list[float] = []
                frame_ra: list[float] = []
                frame_dec: list[float] = []
                frame_tx: list[float] = []
                frame_ty: list[float] = []
                frame_crpix: list[tuple[float, float]] = []
                frame_crval: list[tuple[float, float]] = []
                frame_cd: list[tuple[tuple[float, float], tuple[float, float]]] = []
                frame_snr: list[float] = []
                frame_mag: list[float] = []
                frame_epoch: list[int] = []
                frame_trk_id: list[bytes] = []
                frame_is_syn: list[bool] = []
                frame_survey: list[bytes] = []

                running_offset = curr_n_frames
                for t in trk_list:
                    n_fr = len(t.frames)
                    offsets.append(running_offset)
                    lengths.append(n_fr)
                    labels.append(t.label)
                    taxonomies.append(t.taxonomy_class.encode("ascii"))
                    obj_ids.append(t.object_id.encode("ascii"))
                    trk_ids.append(t.tracklet_id.encode("ascii"))
                    running_offset += n_fr

                    for fr in t.frames:
                        frame_triplets.append(fr.data)
                        frame_bjd.append(fr.bjd_tdb)
                        frame_ra.append(fr.ra_deg)
                        frame_dec.append(fr.dec_deg)
                        frame_tx.append(fr.target_x)
                        frame_ty.append(fr.target_y)
                        frame_crpix.append(fr.wcs_params.crpix)
                        frame_crval.append(fr.wcs_params.crval)
                        frame_cd.append(fr.wcs_params.cd)
                        frame_snr.append(fr.snr)
                        frame_mag.append(fr.mag_apparent)
                        frame_epoch.append(fr.epoch_index)
                        frame_trk_id.append(t.tracklet_id.encode("ascii"))
                        frame_is_syn.append(fr.is_synthetic)
                        frame_survey.append(fr.survey.encode("ascii"))

                # Assign tracklet index datasets
                t_slice = slice(curr_n_tracklets, new_n_tracklets)
                grp["tracklet_offsets"][t_slice] = np.array(offsets, dtype=np.int64)
                grp["tracklet_lengths"][t_slice] = np.array(lengths, dtype=np.int32)
                grp["tracklet_labels"][t_slice] = np.array(labels, dtype=np.int32)
                grp["tracklet_taxonomy"][t_slice] = np.array(taxonomies)
                grp["tracklet_object_ids"][t_slice] = np.array(obj_ids)
                grp["tracklet_ids"][t_slice] = np.array(trk_ids)

                # Assign frame datasets
                f_slice = slice(curr_n_frames, new_n_frames)
                grp["triplets"][f_slice] = np.stack(frame_triplets, axis=0)
                grp["bjd_tdb"][f_slice] = np.array(frame_bjd, dtype=np.float64)
                grp["ra_deg"][f_slice] = np.array(frame_ra, dtype=np.float64)
                grp["dec_deg"][f_slice] = np.array(frame_dec, dtype=np.float64)
                grp["target_x"][f_slice] = np.array(frame_tx, dtype=np.float32)
                grp["target_y"][f_slice] = np.array(frame_ty, dtype=np.float32)
                grp["wcs_crpix"][f_slice] = np.array(frame_crpix, dtype=np.float32)
                grp["wcs_crval"][f_slice] = np.array(frame_crval, dtype=np.float64)
                grp["wcs_cd"][f_slice] = np.array(frame_cd, dtype=np.float64)
                grp["snr"][f_slice] = np.array(frame_snr, dtype=np.float32)
                grp["mag_apparent"][f_slice] = np.array(frame_mag, dtype=np.float32)
                grp["epoch_index"][f_slice] = np.array(frame_epoch, dtype=np.int32)
                grp["tracklet_id"][f_slice] = np.array(frame_trk_id)
                grp["is_synthetic"][f_slice] = np.array(frame_is_syn, dtype=bool)
                grp["provenance_survey"][f_slice] = np.array(frame_survey)

        # Two-phase commit: record completed tracklets in the SQLite ledger
        active_ledger = ledger if ledger is not None else self.ledger
        if active_ledger is not None:
            now_iso = datetime.now(UTC).isoformat()
            for split, trk_list in partitioned.items():
                for t in trk_list:
                    active_ledger.commit_tracklet(
                        CrawlerLedgerEntry(
                            object_id=t.object_id,
                            tracklet_id=t.tracklet_id,
                            n_frames=len(t.frames),
                            split=split,
                            status="COMPLETED",
                            created_at_utc=now_iso,
                        )
                    )

        return counts

    def get_tracklet(
        self,
        split: SplitName,
        tracklet_idx: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Slice any variable-length N-frame tracklet in O(1) time via CSR index."""
        with h5py.File(self.file_path, "r") as h5:
            grp = h5["splits"][split]
            offset = int(grp["tracklet_offsets"][tracklet_idx])
            length = int(grp["tracklet_lengths"][tracklet_idx])
            label = int(grp["tracklet_labels"][tracklet_idx])
            obj_id = grp["tracklet_object_ids"][tracklet_idx].decode("ascii")
            taxonomy = grp["tracklet_taxonomy"][tracklet_idx].decode("ascii")
            t_id = grp["tracklet_ids"][tracklet_idx].decode("ascii")

            f_slice = slice(offset, offset + length)
            triplets = grp["triplets"][f_slice]  # shape (length, 3, stamp_size, stamp_size)
            bjds = grp["bjd_tdb"][f_slice]
            ras = grp["ra_deg"][f_slice]
            decs = grp["dec_deg"][f_slice]
            txs = grp["target_x"][f_slice]
            tys = grp["target_y"][f_slice]
            crpixs = grp["wcs_crpix"][f_slice]
            crvals = grp["wcs_crval"][f_slice]
            cds = grp["wcs_cd"][f_slice]
            snrs = grp["snr"][f_slice]

        meta: dict[str, Any] = {
            "tracklet_id": t_id,
            "object_id": obj_id,
            "taxonomy_class": taxonomy,
            "label": label,
            "length": length,
            "bjd_tdb": bjds,
            "ra_deg": ras,
            "dec_deg": decs,
            "target_x": txs,
            "target_y": tys,
            "wcs_crpix": crpixs,
            "wcs_crval": crvals,
            "wcs_cd": cds,
            "snr": snrs,
        }
        return triplets, meta

    def get_tracklet_count(self, split: SplitName) -> int:
        """Return the total number of variable-length tracklets in the split."""
        with h5py.File(self.file_path, "r") as h5:
            return int(h5["splits"][split]["tracklet_offsets"].shape[0])

    def get_summary(self) -> dict[str, dict[str, int]]:
        """Return tracklet and frame count summaries across splits."""
        summary: dict[str, dict[str, int]] = {}
        with h5py.File(self.file_path, "r") as h5:
            splits_grp = h5["splits"]
            for split in ("train", "val", "test", "ood"):
                if split not in splits_grp:
                    continue
                grp = splits_grp[split]
                n_trk = int(grp["tracklet_offsets"].shape[0])
                n_frames = int(grp["triplets"].shape[0])
                labels = grp["tracklet_labels"][:] if n_trk > 0 else np.array([], dtype=np.int32)
                summary[split] = {
                    "total_tracklets": n_trk,
                    "total_frames": n_frames,
                    "bogus": int(np.sum(labels == 0)),
                    "point": int(np.sum(labels == 1)),
                    "streak": int(np.sum(labels == 2)),
                    "coma": int(np.sum(labels == 3)),
                }
        return summary


class WorkerPartitionedZarrStore:
    """Isolated per-worker Zarr v3 store avoiding Windows NTFS lock collisions.

    Each worker process writes exclusively to its dedicated partition path:
    <root_dir>/worker_<worker_id>.zarr
    """

    def __init__(
        self,
        root_dir: Path | str,
        worker_id: int | str,
        stamp_size: int = 48,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.worker_id = str(worker_id)
        self.stamp_size = stamp_size
        self.store_path = self.root_dir / f"worker_{self.worker_id}.zarr"
        self._ensure_store()

    def _ensure_store(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            root = zarr.open_group(store=str(self.store_path), mode="w")
            root.attrs["worker_id"] = self.worker_id
            root.attrs["stamp_size"] = self.stamp_size
            root.attrs["record_count"] = 0

    def append_cutout(
        self,
        candidate_id: str,
        triplet: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Appends a single (3, stamp_size, stamp_size) cutout and metadata to worker store."""
        triplet_arr = np.asarray(triplet, dtype=np.float32)
        if triplet_arr.shape != (3, self.stamp_size, self.stamp_size):
            msg = (
                f"Expected triplet shape (3, {self.stamp_size}, {self.stamp_size}), "
                f"got {triplet_arr.shape}"
            )
            raise ValueError(msg)

        root = zarr.open_group(store=str(self.store_path), mode="a")
        curr_count = int(root.attrs.get("record_count", 0))

        if "triplets" not in root:
            root.create_array(
                "triplets",
                shape=(0, 3, self.stamp_size, self.stamp_size),
                chunks=(32, 3, self.stamp_size, self.stamp_size),
                dtype="float32",
            )

        triplets_arr = root["triplets"]
        new_shape = (curr_count + 1, 3, self.stamp_size, self.stamp_size)
        triplets_arr.resize(new_shape)
        triplets_arr[curr_count] = triplet_arr

        meta = metadata.copy() if metadata is not None else {}
        meta["candidate_id"] = candidate_id

        meta_list = list(root.attrs.get("metadata", []))
        meta_list.append(meta)
        root.attrs["metadata"] = meta_list
        root.attrs["record_count"] = curr_count + 1
        return curr_count

    def get_record_count(self) -> int:
        """Returns total records stored in this worker partition."""
        root = zarr.open_group(store=str(self.store_path), mode="r")
        return int(root.attrs.get("record_count", 0))

    def read_records(self) -> list[dict[str, Any]]:
        """Reads all records and stamps from this worker partition."""
        root = zarr.open_group(store=str(self.store_path), mode="r")
        count = int(root.attrs.get("record_count", 0))
        if count == 0:
            return []
        triplets = root["triplets"][:]
        meta_list = root.attrs.get("metadata", [])
        records = []
        for i in range(count):
            m = meta_list[i].copy() if i < len(meta_list) else {}
            m["triplet"] = triplets[i]
            records.append(m)
        return records


def compact_worker_stores(
    partition_paths: Sequence[Path | str],
    destination_store_path: Path | str,
    deduplicate_by: str = "candidate_id",
    stamp_size: int = 48,
) -> int:
    """Offline single-process compaction and de-duplication of worker stores.

    Guarantees 0 file locking collisions on Windows NTFS by consolidating sequentially.
    """
    dest_path = Path(destination_store_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    dest_root = zarr.open_group(store=str(dest_path), mode="w")
    dest_root.attrs["stamp_size"] = stamp_size

    seen_keys: set[str] = set()
    total_written = 0

    dest_triplets = dest_root.create_array(
        "triplets",
        shape=(0, 3, stamp_size, stamp_size),
        chunks=(64, 3, stamp_size, stamp_size),
        dtype="float32",
    )
    all_metadata: list[dict[str, Any]] = []

    for p in partition_paths:
        p_path = Path(p)
        if not p_path.exists():
            continue
        part_root = zarr.open_group(store=str(p_path), mode="r")
        count = int(part_root.attrs.get("record_count", 0))
        if count == 0:
            continue
        p_triplets = part_root["triplets"][:]
        p_meta = part_root.attrs.get("metadata", [])

        for i in range(count):
            m = p_meta[i] if i < len(p_meta) else {}
            dedup_key = str(m.get(deduplicate_by, f"idx_{total_written}"))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            dest_triplets.resize((total_written + 1, 3, stamp_size, stamp_size))
            dest_triplets[total_written] = p_triplets[i]
            all_metadata.append(m)
            total_written += 1

    dest_root.attrs["metadata"] = all_metadata
    dest_root.attrs["record_count"] = total_written
    return total_written

