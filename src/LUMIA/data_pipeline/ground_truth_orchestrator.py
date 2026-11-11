"""Orchestrator pipeline connecting MPC client, cutout extraction, and HDF5 storage."""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from pydantic import ConfigDict, Field

from LUMIA.core.schemas import GroundTruthBatch, LumiaBaseModel
from LUMIA.data_pipeline.hdf5_storage import HDF5GroundTruthWriter
from LUMIA.physics.mpc_client import MinorPlanetCenterClient

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "GroundTruthEpochResult",
    "GroundTruthPipeline",
]

_CORE_MODEL_CONFIG: ConfigDict = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)


class GroundTruthEpochResult(LumiaBaseModel):
    """Result summary of an ingested ground-truth observation epoch."""

    model_config = _CORE_MODEL_CONFIG

    field_id: str = Field(..., min_length=1, description="Survey field or pointing identifier.")
    bjd_tdb: float = Field(..., gt=0.0, description="Observation epoch in Barycentric Julian Date.")
    split: str = Field(..., min_length=1, description="HDF5 archive split (train/val/test).")
    real_count: int = Field(..., ge=0, description="Number of confirmed real cutouts extracted.")
    bogus_count: int = Field(..., ge=0, description="Number of background bogus cutouts extracted.")
    total_appended: int = Field(..., ge=0, description="Total samples appended in this epoch.")
    archive_total_samples: int = Field(..., ge=0, description="Updated total samples in target split.")


class GroundTruthPipeline:
    """Orchestrates MPC ephemeris querying, cutout slicing, and out-of-core HDF5 archiving."""

    def __init__(
        self,
        mpc_client: MinorPlanetCenterClient,
        hdf5_writer: HDF5GroundTruthWriter,
        *,
        bogus_count: int = 10,
        min_bogus_distance_px: float = 50.0,
        stamp_size: int = 31,
        random_seed: int = 42,
    ) -> None:
        """Initialize pipeline with MPC service client and HDF5 archive writer.

        Parameters
        ----------
        mpc_client : MinorPlanetCenterClient
            Client for querying minor planet ephemerides.
        hdf5_writer : HDF5GroundTruthWriter
            Writer managing append-only HDF5 storage.
        bogus_count : int, optional
            Default number of non-object background cutouts per epoch (default 10).
        min_bogus_distance_px : float, optional
            Minimum Euclidean distance in pixels between bogus samples and known bodies (default 50.0).
        stamp_size : int, optional
            Spatial dimension of extracted square cutouts (default 31).
        random_seed : int, optional
            Seed for deterministic background rejection sampling (default 42).
        """
        if stamp_size % 2 == 0 or stamp_size < 15:
            msg = f"stamp_size must be an odd integer >= 15, got {stamp_size}."
            raise ValueError(msg)
        if min_bogus_distance_px <= 0.0:
            msg = f"min_bogus_distance_px must be positive, got {min_bogus_distance_px}."
            raise ValueError(msg)
        if bogus_count < 0:
            msg = f"bogus_count must be non-negative, got {bogus_count}."
            raise ValueError(msg)

        self._mpc_client: MinorPlanetCenterClient = mpc_client
        self._hdf5_writer: HDF5GroundTruthWriter = hdf5_writer
        self._default_bogus_count: int = bogus_count
        self._min_bogus_distance_px: float = float(min_bogus_distance_px)
        self._stamp_size: int = stamp_size
        self._rng: np.random.Generator = np.random.default_rng(random_seed)

    @property
    def mpc_client(self) -> MinorPlanetCenterClient:
        """Return the configured MinorPlanetCenterClient."""
        return self._mpc_client

    @property
    def hdf5_writer(self) -> HDF5GroundTruthWriter:
        """Return the configured HDF5GroundTruthWriter."""
        return self._hdf5_writer

    @property
    def stamp_size(self) -> int:
        """Return cutout stamp spatial size."""
        return self._stamp_size

    def process_epoch(
        self,
        science_tensor: torch.Tensor | np.ndarray,
        reference_tensor: torch.Tensor | np.ndarray,
        difference_tensor: torch.Tensor | np.ndarray,
        wcs_header: Any,
        bjd_tdb: float,
        field_id: str,
        *,
        bogus_count: int | None = None,
        split: str = "train",
    ) -> GroundTruthEpochResult:
        """Process a single calibrated observation epoch into ground-truth HDF5 cutouts.

        Parameters
        ----------
        science_tensor : torch.Tensor | np.ndarray
            Calibrated 2D science image array of shape (H, W).
        reference_tensor : torch.Tensor | np.ndarray
            Calibrated 2D co-added reference image array of shape (H, W).
        difference_tensor : torch.Tensor | np.ndarray
            Subtracted 2D ZOGY difference image array of shape (H, W).
        wcs_header : Any
            Astropy WCS object, FITS Header, or header dict defining pixel-to-sky mapping.
        bjd_tdb : float
            Observation epoch mid-point in Barycentric Julian Date.
        field_id : str
            Unique sky field or pointing identifier.
        bogus_count : int | None, optional
            Number of bogus cutouts to sample (overrides instance default if specified).
        split : str, optional
            Target HDF5 split group ('train', 'val', or 'test', default 'train').

        Returns
        -------
        GroundTruthEpochResult
            Execution summary detailing real/bogus cutout counts and archive state.
        """
        # Convert input arrays to float32 NumPy arrays
        sci_arr = self._to_numpy_2d(science_tensor, "science_tensor")
        ref_arr = self._to_numpy_2d(reference_tensor, "reference_tensor")
        diff_arr = self._to_numpy_2d(difference_tensor, "difference_tensor")

        if sci_arr.shape != ref_arr.shape or sci_arr.shape != diff_arr.shape:
            msg = (
                f"Image shapes must match: science {sci_arr.shape}, "
                f"reference {ref_arr.shape}, difference {diff_arr.shape}."
            )
            raise ValueError(msg)

        height, width = sci_arr.shape
        half_stamp = self._stamp_size // 2

        if height < self._stamp_size or width < self._stamp_size:
            msg = f"Images ({height}, {width}) are smaller than stamp_size {self._stamp_size}."
            raise ValueError(msg)

        # Resolve WCS
        wcs = wcs_header if isinstance(wcs_header, WCS) else WCS(wcs_header)

        # 1. Compute Field Center and Search Radius from WCS
        xc = (width - 1) / 2.0
        yc = (height - 1) / 2.0
        center_sky = wcs.pixel_to_world(xc, yc)
        ra_center = float(center_sky.ra.deg)
        dec_center = float(center_sky.dec.deg)

        corners = [(0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0), (width - 1.0, height - 1.0)]
        max_corner_sep = max(center_sky.separation(wcs.pixel_to_world(cx, cy)).deg for cx, cy in corners)
        radius_deg = float(max_corner_sep) * 1.05

        # 2. Query Known Objects from Minor Planet Center
        known_objects = self._mpc_client.query_field_ephemerides(
            ra_center=ra_center,
            dec_center=dec_center,
            radius_deg=radius_deg,
            bjd_tdb=bjd_tdb,
        )

        # 3. Extract Real Cutouts (Discard Boundary Hits)
        real_cutouts: list[np.ndarray] = []
        real_bjd: list[float] = []
        real_ra: list[float] = []
        real_dec: list[float] = []
        real_ids: list[str] = []
        all_known_px: list[tuple[float, float]] = []

        safe_field = re.sub(r"[^A-Za-z0-9_.-]", "_", field_id)
        bjd_tag = f"{bjd_tdb:.4f}".replace(".", "_")

        for idx, obj in enumerate(known_objects):
            try:
                sky_coord = SkyCoord(ra=obj.ra_deg, dec=obj.dec_deg, unit="deg")
                px, py = wcs.world_to_pixel(sky_coord)
                px_f, py_f = float(px), float(py)
            except Exception as exc:
                logger.debug("Failed to project sky coordinate to pixel for %s: %s", obj.designation, exc)
                continue

            if not (math.isfinite(px_f) and math.isfinite(py_f)):
                continue

            all_known_px.append((px_f, py_f))

            ix = int(round(px_f))
            iy = int(round(py_f))

            # Discard boundary hits (stamp must fit fully inside frame)
            if (
                ix - half_stamp < 0
                or ix + half_stamp + 1 > width
                or iy - half_stamp < 0
                or iy + half_stamp + 1 > height
            ):
                logger.debug("Discarding edge hit for %s at pixel (%d, %d)", obj.designation, ix, iy)
                continue

            # Extract 3-channel slice [Science, Reference, Difference]
            sci_cut = sci_arr[iy - half_stamp : iy + half_stamp + 1, ix - half_stamp : ix + half_stamp + 1]
            ref_cut = ref_arr[iy - half_stamp : iy + half_stamp + 1, ix - half_stamp : ix + half_stamp + 1]
            diff_cut = diff_arr[iy - half_stamp : iy + half_stamp + 1, ix - half_stamp : ix + half_stamp + 1]

            triplet = np.stack([sci_cut, ref_cut, diff_cut], axis=0).astype(np.float32)
            real_cutouts.append(triplet)
            real_bjd.append(bjd_tdb)
            real_ra.append(obj.ra_deg)
            real_dec.append(obj.dec_deg)

            safe_desig = re.sub(r"[^A-Za-z0-9_.-]", "_", obj.designation)
            real_ids.append(f"real_{safe_field}_{safe_desig}_{bjd_tag}_{idx:03d}")

        # 4. Extract Bogus Cutouts (Random Non-Object Background Coordinates)
        target_bogus = self._default_bogus_count if bogus_count is None else max(0, bogus_count)
        bogus_cutouts: list[np.ndarray] = []
        bogus_bjd: list[float] = []
        bogus_ra: list[float] = []
        bogus_dec: list[float] = []
        bogus_ids: list[str] = []

        selected_bogus_px: list[tuple[int, int]] = []
        max_attempts = 2000 * max(1, target_bogus)
        attempts = 0

        while len(selected_bogus_px) < target_bogus and attempts < max_attempts:
            attempts += 1
            xb = int(self._rng.integers(half_stamp, width - half_stamp))
            yb = int(self._rng.integers(half_stamp, height - half_stamp))

            # Must be at least min_bogus_distance_px away from any known object
            if any(
                math.hypot(xb - xr, yb - yr) < self._min_bogus_distance_px
                for xr, yr in all_known_px
            ):
                continue

            # Avoid overlapping previously chosen bogus cutouts
            if any(
                math.hypot(xb - pxb, yb - pyb) < float(self._stamp_size)
                for pxb, pyb in selected_bogus_px
            ):
                continue

            selected_bogus_px.append((xb, yb))

        for b_idx, (xb, yb) in enumerate(selected_bogus_px):
            sci_cut = sci_arr[yb - half_stamp : yb + half_stamp + 1, xb - half_stamp : xb + half_stamp + 1]
            ref_cut = ref_arr[yb - half_stamp : yb + half_stamp + 1, xb - half_stamp : xb + half_stamp + 1]
            diff_cut = diff_arr[yb - half_stamp : yb + half_stamp + 1, xb - half_stamp : xb + half_stamp + 1]

            triplet = np.stack([sci_cut, ref_cut, diff_cut], axis=0).astype(np.float32)
            bogus_cutouts.append(triplet)
            bogus_bjd.append(bjd_tdb)

            try:
                b_coord = wcs.pixel_to_world(xb, yb)
                b_ra = float(b_coord.ra.deg) % 360.0
                b_dec = float(b_coord.dec.deg)
            except Exception:
                b_ra = ra_center
                b_dec = dec_center

            bogus_ra.append(b_ra)
            bogus_dec.append(b_dec)
            bogus_ids.append(f"bogus_{safe_field}_{bjd_tag}_{b_idx:03d}")

        # 5. Build and Save GroundTruthBatch
        n_real = len(real_cutouts)
        n_bogus = len(bogus_cutouts)
        total_samples = n_real + n_bogus

        if total_samples > 0:
            all_cutouts = np.concatenate([real_cutouts, bogus_cutouts], axis=0) if (n_real and n_bogus) else (
                np.array(real_cutouts, dtype=np.float32) if n_real else np.array(bogus_cutouts, dtype=np.float32)
            )
            all_labels = np.concatenate(
                [np.ones(n_real, dtype=np.int8), np.zeros(n_bogus, dtype=np.int8)],
                axis=0,
            )
            metadata = {
                "bjd_tdb": real_bjd + bogus_bjd,
                "ra_deg": real_ra + bogus_ra,
                "dec_deg": real_dec + bogus_dec,
                "candidate_id": real_ids + bogus_ids,
            }

            batch = GroundTruthBatch(
                cutouts=all_cutouts,
                labels=all_labels,
                metadata=metadata,
            )

            archive_total = self._hdf5_writer.append_batch(batch, split=split)
        else:
            archive_total = self._hdf5_writer.get_sample_count(split=split)

        logger.info(
            "Processed epoch field '%s' (BJD %.4f): appended %d samples (%d real, %d bogus) to '%s'.",
            field_id,
            bjd_tdb,
            total_samples,
            n_real,
            n_bogus,
            split,
        )

        return GroundTruthEpochResult(
            field_id=field_id,
            bjd_tdb=bjd_tdb,
            split=split,
            real_count=n_real,
            bogus_count=n_bogus,
            total_appended=total_samples,
            archive_total_samples=archive_total,
        )

    @staticmethod
    def _to_numpy_2d(tensor_or_array: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
        """Convert input array or tensor into float32 2D NumPy array."""
        if isinstance(tensor_or_array, torch.Tensor):
            arr = tensor_or_array.detach().cpu().numpy()
        elif isinstance(tensor_or_array, np.ndarray):
            arr = tensor_or_array
        else:
            msg = f"{name} must be a torch.Tensor or numpy.ndarray, got {type(tensor_or_array)}."
            raise TypeError(msg)

        if arr.ndim != 2:
            msg = f"{name} must be 2-dimensional (H, W), got shape {arr.shape}."
            raise ValueError(msg)

        return arr.astype(np.float32)
