"""Canonical inter-module Pydantic schemas shared by physics, ML, data and API layers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CUTOUT_TRIPLET_CHANNELS",
    "CandidateDetection",
    "ClassificationResult",
    "CutoutBatchResult",
    "CutoutStampSize",
    "GroundObservatoryProfile",
    "GroundTruthBatch",
    "KnownObjectEphemeris",
    "LumiaBaseModel",
    "ObservatoryProfile",
    "SNRPeakCandidate",
    "SpacecraftObservatoryProfile",
]

CutoutStampSize = Literal[31, 51]
CUTOUT_TRIPLET_CHANNELS: int = 3

_CORE_MODEL_CONFIG: ConfigDict = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)


class LumiaBaseModel(BaseModel):
    """Shared strict, immutable Pydantic contract for module boundaries."""

    model_config = _CORE_MODEL_CONFIG


class GroundObservatoryProfile(LumiaBaseModel):
    """Ground-based observatory parameterized by geodetic coordinates and MPC code."""

    platform_type: Literal["ground"] = "ground"
    mpc_code: str = Field(..., min_length=3, max_length=4, pattern=r"^[A-Z0-9]{3,4}$")
    name: str = Field(..., min_length=1, max_length=128)
    latitude_deg: float = Field(..., ge=-90.0, le=90.0)
    longitude_deg: float = Field(..., ge=-180.0, le=180.0)
    elevation_meters: float = Field(..., ge=-500.0, le=9000.0)
    pixel_scale_arcsec_per_px: float = Field(..., gt=0.01, le=10.0)
    default_filter: str = Field(..., min_length=1, max_length=16)


class SpacecraftObservatoryProfile(LumiaBaseModel):
    """Spacecraft observatory parameterized by NAIF SPICE identity."""

    platform_type: Literal["spacecraft"] = "spacecraft"
    naif_id: int = Field(..., description="NAIF ID (e.g. -170 for JWST, -48 for HST, -236 for Roman)")
    name: str = Field(..., min_length=1, max_length=128)
    kernel_bundle: tuple[str, ...] = Field(
        ...,
        description="Ordered list of required SPICE kernels: LSK, planetary SPK (de440), and spacecraft SPK",
    )
    barycentric_frame: Literal["J2000", "ICRF"] = "J2000"
    center_naif_id: int = Field(default=0, description="Solar System Barycenter (NAIF ID 0)")
    pixel_scale_arcsec_per_px: float = Field(..., gt=0.001, le=10.0)
    primary_bandpass: str = Field(..., min_length=1, max_length=32)


ObservatoryProfile = Annotated[
    GroundObservatoryProfile | SpacecraftObservatoryProfile,
    Field(discriminator="platform_type"),
]


class CandidateDetection(LumiaBaseModel):
    """Authoritative single-epoch moving-object candidate detection record.

    This is the only ``CandidateDetection`` definition in the code base. It is consumed
    by tracklet linking (Module 5), populated with classification provenance by the CNN
    classifier (Module 7) and serialized unchanged by the web API (Module 10).

    The two astrometric error fields have distinct meanings and must not be conflated:

    * ``astrometric_uncertainty_arcsec``: single-epoch 1-sigma measurement error of this
      detection's position (propagated from centroiding and the WCS fit, Module 3).
    * ``astrometric_residual_arcsec``: residual of this detection against a fitted
      multi-epoch tracklet motion model (Module 5). ``None`` until it has been linked.
    """

    model_config = _CORE_MODEL_CONFIG

    candidate_id: str = Field(..., min_length=1, max_length=128)
    job_id: str | None = Field(default=None, min_length=1, max_length=128)
    field_id: str = Field(default="default", min_length=1, max_length=128)
    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="ICRS Right Ascension [deg].")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="ICRS Declination [deg].")
    bjd_tdb: float = Field(
        ...,
        description="Light-time corrected Barycentric Julian Date (TDB).",
    )
    mag: float | None = Field(default=None, description="Calibrated apparent magnitude.")
    snr: float | None = Field(default=None, ge=0.0, description="Detection SNR.")
    cnn_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Real/Bogus CNN probability in [0, 1] (Module 7).",
    )
    model_version: str | None = Field(
        default=None,
        min_length=1,
        description="Version of the classifier that produced cnn_score (provenance).",
    )
    astrometric_uncertainty_arcsec: float = Field(
        default=0.15,
        ge=0.0,
        description="Single-epoch 1-sigma positional measurement error [arcsec].",
    )
    astrometric_residual_arcsec: float | None = Field(
        default=None,
        ge=0.0,
        description="Residual against the fitted tracklet motion model [arcsec].",
    )

    # Backward-Compatible Multi-Survey Extensions
    survey_id: str = Field(default="ztf", min_length=1, max_length=64)
    filter_band: str = Field(default="r", min_length=1, max_length=16)
    observer_pos_au: tuple[float, float, float] | None = Field(
        default=None,
        description="Barycentric (X, Y, Z) observer position in AU",
    )
    observer_vel_au_per_day: tuple[float, float, float] | None = Field(
        default=None,
        description="Barycentric (Vx, Vy, Vz) observer velocity in AU/day",
    )
    light_travel_time_sec: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _model_version_requires_score(self) -> CandidateDetection:
        if self.model_version is not None and self.cnn_score is None:
            msg = "model_version may only be set together with a cnn_score."
            raise ValueError(msg)
        return self


class SNRPeakCandidate(LumiaBaseModel):
    """Single deterministic SNR peak candidate extracted from a significance map."""

    model_config = _CORE_MODEL_CONFIG

    candidate_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    x_px: int = Field(..., ge=0, description="0-indexed integer column coordinate.")
    y_px: int = Field(..., ge=0, description="0-indexed integer row coordinate.")
    x_centroid_px: float = Field(
        ...,
        ge=0.0,
        description="Positive significance-weighted 3x3 sub-pixel column centroid.",
    )
    y_centroid_px: float = Field(
        ...,
        ge=0.0,
        description="Positive significance-weighted 3x3 sub-pixel row centroid.",
    )
    snr: float = Field(..., gt=0.0, description="Peak significance value in sigma.")


class CutoutBatchResult(LumiaBaseModel):
    """Validated batch of aligned (Science, Reference, Difference) tensor triplets.

    Shared inter-module contract bridging Module 4 (``LUMIA.physics.cutout_extraction``)
    to Module 7 (``LUMIA.ml.classifier``) with zero cross-domain model dependencies.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    triplets: torch.Tensor
    candidates: list[SNRPeakCandidate]
    stamp_size: CutoutStampSize
    rejected_boundary_count: int = Field(default=0, ge=0)

    @property
    def candidate_ids(self) -> list[str]:
        """Return candidate identifiers in batch order for CNN inference."""
        return [cand.candidate_id for cand in self.candidates]

    @model_validator(mode="after")
    def _validate_triplet_batch(self) -> CutoutBatchResult:
        if not self.triplets.is_floating_point():
            msg = f"triplets must be a floating-point tensor, got {self.triplets.dtype}."
            raise ValueError(msg)
        if self.triplets.ndim != 4:
            msg = (
                "triplets must be a 4D tensor of shape (B, 3, H, W), "
                f"got {tuple(self.triplets.shape)}."
            )
            raise ValueError(msg)
        batch_n, channels, height, width = (int(d) for d in self.triplets.shape)
        if channels != CUTOUT_TRIPLET_CHANNELS:
            msg = (
                f"triplets must have {CUTOUT_TRIPLET_CHANNELS} channels, "
                f"got {channels}."
            )
            raise ValueError(msg)
        if height != self.stamp_size or width != self.stamp_size or height % 2 == 0:
            msg = (
                f"triplets spatial dimensions ({height}, {width}) must match odd "
                f"stamp_size={self.stamp_size}."
            )
            raise ValueError(msg)
        if batch_n != len(self.candidates):
            msg = (
                f"triplets batch size ({batch_n}) must equal candidates length "
                f"({len(self.candidates)})."
            )
            raise ValueError(msg)
        ids = [cand.candidate_id for cand in self.candidates]
        if len(set(ids)) != len(ids):
            msg = "Duplicate candidate_id values are prohibited in CutoutBatchResult."
            raise ValueError(msg)
        if batch_n > 0 and not bool(torch.all(torch.isfinite(self.triplets)).item()):
            msg = "triplets tensor contains non-finite (NaN or Inf) values."
            raise ValueError(msg)
        return self


class ClassificationResult(LumiaBaseModel):
    """Validated Real/Bogus CNN prediction for a single candidate."""

    model_config = _CORE_MODEL_CONFIG

    candidate_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0, description="Real probability in [0, 1].")
    is_real: bool = Field(..., description="True if score >= decision threshold.")
    model_version: str = Field(..., min_length=1)


class KnownObjectEphemeris(LumiaBaseModel):
    """Canonical ephemeris state of a known solar system object from MPC or JPL Horizons."""

    model_config = _CORE_MODEL_CONFIG

    designation: str = Field(
        ..., min_length=1, max_length=128, description="Target designation or name."
    )
    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="ICRS Right Ascension [deg].")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="ICRS Declination [deg].")
    vmag: float | None = Field(default=None, description="Predicted apparent visual magnitude V.")
    ra_rate_arcsec_hr: float = Field(
        default=0.0,
        description="Sky-plane motion rate along RA (dRA*cos(Dec)) [arcsec/hr].",
    )
    dec_rate_arcsec_hr: float = Field(
        default=0.0, description="Sky-plane motion rate along Dec (dDec) [arcsec/hr]."
    )


class GroundTruthBatch(LumiaBaseModel):
    """Validated batch of 3-channel cutouts, binary labels, and aligned metadata.

    Shared contract between data pipeline, HDF5 storage container, and CNN training.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    cutouts: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any]

    @property
    def batch_size(self) -> int:
        """Return the batch dimension N."""
        return int(self.cutouts.shape[0])

    @model_validator(mode="after")
    def _validate_ground_truth_batch(self) -> GroundTruthBatch:
        if not isinstance(self.cutouts, np.ndarray):
            msg = f"cutouts must be a numpy.ndarray, got {type(self.cutouts)}."
            raise ValueError(msg)
        if self.cutouts.ndim != 4:
            msg = (
                f"cutouts must be a 4D array of shape (N, 3, 31, 31), "
                f"got ndim={self.cutouts.ndim} with shape {self.cutouts.shape}."
            )
            raise ValueError(msg)
        n_cutouts, channels, height, width = (int(d) for d in self.cutouts.shape)
        if channels != 3:
            msg = (
                f"cutouts must have exactly 3 channels (Science, Reference, Difference), "
                f"got {channels}."
            )
            raise ValueError(msg)
        if height != 31 or width != 31:
            msg = f"cutouts spatial dimensions ({height}, {width}) must be (31, 31)."
            raise ValueError(msg)
        if self.cutouts.dtype != np.float32:
            msg = f"cutouts dtype must be float32, got {self.cutouts.dtype}."
            raise ValueError(msg)
        if n_cutouts > 0 and not bool(np.isfinite(self.cutouts).all()):
            msg = "cutouts array contains non-finite (NaN or Inf) values."
            raise ValueError(msg)

        if not isinstance(self.labels, np.ndarray):
            msg = f"labels must be a numpy.ndarray, got {type(self.labels)}."
            raise ValueError(msg)
        if self.labels.ndim != 1:
            msg = (
                f"labels must be a 1D array of shape (N,), "
                f"got ndim={self.labels.ndim} with shape {self.labels.shape}."
            )
            raise ValueError(msg)
        if self.labels.dtype != np.int8:
            msg = f"labels dtype must be int8, got {self.labels.dtype}."
            raise ValueError(msg)
        if self.labels.shape[0] != n_cutouts:
            msg = (
                f"Batch dimension mismatch: cutouts has {n_cutouts} samples "
                f"but labels has {self.labels.shape[0]} samples."
            )
            raise ValueError(msg)

        if not isinstance(self.metadata, dict):
            msg = f"metadata must be a dictionary, got {type(self.metadata)}."
            raise ValueError(msg)

        required_meta_keys = ("bjd_tdb", "ra_deg", "dec_deg", "candidate_id")
        for key in required_meta_keys:
            if key not in self.metadata:
                msg = f"metadata missing required key '{key}'."
                raise ValueError(msg)

        for key, val in self.metadata.items():
            if isinstance(val, (list, tuple, np.ndarray)):
                val_len = len(val)
            else:
                msg = f"metadata field '{key}' must be a list, tuple, or numpy array, got {type(val)}."
                raise ValueError(msg)
            if val_len != n_cutouts:
                msg = (
                    f"Batch dimension mismatch: metadata['{key}'] has length {val_len} "
                    f"which does not match cutouts batch size {n_cutouts}."
                )
                raise ValueError(msg)

            if key == "candidate_id":
                for item in val:
                    item_str = (
                        item.decode("ascii")
                        if isinstance(item, (bytes, np.bytes_))
                        else str(item)
                    )
                    if not item_str:
                        msg = "candidate_id values must be non-empty strings."
                        raise ValueError(msg)
            elif key in ("bjd_tdb", "ra_deg", "dec_deg"):
                if isinstance(val, np.ndarray):
                    if not bool(np.isfinite(val).all()):
                        msg = f"metadata['{key}'] contains non-finite values."
                        raise ValueError(msg)
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        if not isinstance(item, (int, float)) or not bool(np.isfinite(item)):
                            msg = f"metadata['{key}'] contains invalid or non-finite value: {item}."
                            raise ValueError(msg)

        return self

