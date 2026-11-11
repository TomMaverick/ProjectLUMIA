"""Persistence bridge connecting Physics Modules 3 & 4 to the Module 2 Polyglot Data Plane.

Implements pure Pydantic v2 mapping functions and repository/ORM adapters in strict
compliance with ADR-0001, ADR-0002, and ADR-0003:
- Maps calibrated astrometry and aperture photometry from ``LUMIA.physics.astrometry`` and
  ``LUMIA.physics.photometry`` into validated ``candidate_detections`` records.
- Maps GPU-accelerated ZOGY outputs and vectorized 3-channel cutout batches from
  ``LUMIA.physics.gpu_processing`` and ``LUMIA.physics.cutout_extraction`` into DuckDB
  ``candidate_detections`` / ``candidate_cutout_stamps`` tables and Hive-partitioned Parquet.
- Updates 3NF transactional pipeline lifecycle records (``TelescopeProfileORM``,
  ``PipelineJobORM``, ``PipelineTaskORM``, ``PipelineTaskLogORM``) via SQLAlchemy 2.x.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, date, datetime  # noqa: TC003 - needed at runtime by Pydantic
from pathlib import Path  # noqa: TC003 - needed at runtime by Pydantic
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID  # noqa: TC003 - needed at runtime by Pydantic

import numpy as np
import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from LUMIA.core.schemas import CandidateDetection
from LUMIA.data_pipeline.repositories import (
    CandidateCutoutStampRecord,
    CutoutStampSize,
    _validate_square_matrix,
)
from LUMIA.data_pipeline.schema_olap import (
    OLAPSchemaConfig,
    _unwrap_duckdb_connection,
    _validate_safe_linkage_id,
    export_job_table_to_parquet,
    initialize_olap_schema,
)
from LUMIA.data_pipeline.schema_oltp import (
    PipelineJobORM,
    PipelineTaskLogORM,
    PipelineTaskORM,
    TelescopeProfileORM,
)
from LUMIA.physics.config import ObservatoryLocation, TelescopeConfig
from LUMIA.physics.photometry import (
    AperturePhotometryMeasurement,
    ZeroPointResult,  # noqa: TC001 - needed at runtime by Pydantic
)

if TYPE_CHECKING:
    import duckdb

    from LUMIA.data_pipeline.database import CatalogDatabase
    from LUMIA.physics.astrometry import AstrometricMeasurement, BarycentricTimeResult
    from LUMIA.physics.cutout_extraction import CutoutBatchResult
    from LUMIA.physics.gpu_processing import ZOGYResult

__all__ = [
    "EpochPersistenceContext",
    "OLAPCandidateDetectionRecord",
    "OLAPCutoutStampRecord",
    "OLAPEpochPersistResult",
    "PipelineJobCreateSpec",
    "PipelineJobRecord",
    "PipelineJobStatus",
    "PipelineTaskCreateSpec",
    "PipelineTaskLogRecord",
    "PipelineTaskRecord",
    "PipelineTaskStatus",
    "TaskLogSeverity",
    "TelescopeProfileRecord",
    "ZOGYEpochPersistenceBundle",
    "build_zogy_persistence_bundle",
    "fetch_olap_cutout_stamps",
    "fetch_olap_detections_for_job",
    "get_pipeline_job",
    "get_telescope_profile",
    "list_pipeline_task_logs",
    "map_astrometry_photometry_to_detection",
    "persist_olap_cutout_stamps",
    "persist_olap_detections",
    "persist_zogy_bundle_to_olap",
    "register_pipeline_job",
    "register_pipeline_task",
    "transition_pipeline_job",
    "transition_pipeline_task",
    "upsert_telescope_profile",
]

PipelineJobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
PipelineTaskStatus = Literal[
    "pending",
    "running",
    "retrying",
    "completed",
    "failed",
    "skipped",
]
TaskLogSeverity = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_BRIDGE_MODEL_CONFIG: ConfigDict = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)

_ALLOWED_JOB_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled", "failed"}),
    "running": frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

_ALLOWED_TASK_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running", "skipped", "failed"}),
    "running": frozenset({"retrying", "completed", "failed"}),
    "retrying": frozenset({"running", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "skipped": frozenset(),
}

_DETECTION_COLUMNS_SQL: str = (
    "candidate_id, job_id, observation_night, field_id, ra_deg, dec_deg, bjd_tdb, "
    "x_centroid_px, y_centroid_px, net_flux_adu, flux_err_adu, mag, mag_err, snr, "
    "cnn_score, model_version, astrometric_uncertainty_arcsec, "
    "astrometric_residual_arcsec, tracklet_id"
)

_CUTOUT_COLUMNS_SQL: str = (
    "candidate_id, epoch_index, observation_night, field_id, cutout_size, bjd_tdb, "
    "mag, mag_err, target_x_px, target_y_px, science_pixels, reference_pixels, "
    "difference_pixels"
)


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def _coerce_pixel_grid(raw_grid: Any) -> list[list[float]]:
    """Convert a DuckDB nested FLOAT[][] sequence into a strict Python float matrix."""
    return [[float(val) for val in row] for row in raw_grid]


# ---------------------------------------------------------------------------
# Strict ADR-0002 Pydantic v2 Bridge Contracts (OLAP & OLTP)
# ---------------------------------------------------------------------------


class EpochPersistenceContext(BaseModel):
    """Immutable Hive-partition and job-linkage context for a single science epoch."""

    model_config = _BRIDGE_MODEL_CONFIG

    job_id: str = Field(..., min_length=1, max_length=128)
    field_id: str = Field(..., min_length=1, max_length=128)
    observation_night: date
    epoch_index: int = Field(default=0, ge=0)

    @field_validator("job_id", "field_id")
    @classmethod
    def _check_safe_ids(cls, value: str, info: Any) -> str:
        label = str(info.field_name or "identifier")
        return _validate_safe_linkage_id(label, value)


class OLAPCandidateDetectionRecord(BaseModel):
    """Full 19-column analytical candidate detection contract for DuckDB/Parquet OLAP."""

    model_config = _BRIDGE_MODEL_CONFIG

    candidate_id: str = Field(..., min_length=1, max_length=128)
    job_id: str = Field(..., min_length=1, max_length=128)
    observation_night: date
    field_id: str = Field(..., min_length=1, max_length=128)
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    bjd_tdb: float
    x_centroid_px: float | None = Field(default=None)
    y_centroid_px: float | None = Field(default=None)
    net_flux_adu: float | None = Field(default=None)
    flux_err_adu: float | None = Field(default=None, ge=0.0)
    mag: float | None = Field(default=None)
    mag_err: float | None = Field(default=None, ge=0.0)
    snr: float | None = Field(default=None, ge=0.0)
    cnn_score: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str | None = Field(default=None, min_length=1)
    astrometric_uncertainty_arcsec: float = Field(..., ge=0.0)
    astrometric_residual_arcsec: float | None = Field(default=None, ge=0.0)
    tracklet_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("candidate_id", "job_id", "field_id")
    @classmethod
    def _check_required_linkage_ids(cls, value: str, info: Any) -> str:
        label = str(info.field_name or "identifier")
        return _validate_safe_linkage_id(label, value)

    @field_validator("tracklet_id")
    @classmethod
    def _check_optional_tracklet_id(cls, value: str | None) -> str | None:
        if value is not None:
            return _validate_safe_linkage_id("tracklet_id", value)
        return None

    def to_candidate_detection(self) -> CandidateDetection:
        """Project this analytical row onto the canonical core ``CandidateDetection`` DTO."""
        return CandidateDetection(
            candidate_id=self.candidate_id,
            job_id=self.job_id,
            field_id=self.field_id,
            ra_deg=self.ra_deg,
            dec_deg=self.dec_deg,
            bjd_tdb=self.bjd_tdb,
            mag=self.mag,
            snr=self.snr,
            cnn_score=self.cnn_score,
            model_version=self.model_version,
            astrometric_uncertainty_arcsec=self.astrometric_uncertainty_arcsec,
            astrometric_residual_arcsec=self.astrometric_residual_arcsec,
        )


class OLAPCutoutStampRecord(BaseModel):
    """Full 13-column analytical 3-channel cutout stamp contract for DuckDB/Parquet OLAP."""

    model_config = _BRIDGE_MODEL_CONFIG

    candidate_id: str = Field(..., min_length=1, max_length=128)
    epoch_index: int = Field(..., ge=0)
    observation_night: date
    field_id: str = Field(..., min_length=1, max_length=128)
    cutout_size: CutoutStampSize
    bjd_tdb: float
    mag: float
    mag_err: float = Field(..., ge=0.0)
    target_x_px: float = Field(..., ge=0.0)
    target_y_px: float = Field(..., ge=0.0)
    science_pixels: list[list[float]]
    reference_pixels: list[list[float]] | None = Field(default=None)
    difference_pixels: list[list[float]] | None = Field(default=None)

    @field_validator("candidate_id", "field_id")
    @classmethod
    def _check_safe_ids(cls, value: str, info: Any) -> str:
        label = str(info.field_name or "identifier")
        return _validate_safe_linkage_id(label, value)

    @model_validator(mode="after")
    def _validate_pixel_grids(self) -> OLAPCutoutStampRecord:
        _validate_square_matrix("science_pixels", self.science_pixels, self.cutout_size)
        if self.reference_pixels is not None:
            _validate_square_matrix(
                "reference_pixels",
                self.reference_pixels,
                self.cutout_size,
            )
        if self.difference_pixels is not None:
            _validate_square_matrix(
                "difference_pixels",
                self.difference_pixels,
                self.cutout_size,
            )
        return self

    def to_stamp_record(self) -> CandidateCutoutStampRecord:
        """Convert into the repository ``CandidateCutoutStampRecord`` DTO."""
        return CandidateCutoutStampRecord(
            candidate_id=self.candidate_id,
            epoch_index=self.epoch_index,
            cutout_size=self.cutout_size,
            bjd_tdb=self.bjd_tdb,
            mag=self.mag,
            mag_err=self.mag_err,
            target_x_px=self.target_x_px,
            target_y_px=self.target_y_px,
            science_pixels=self.science_pixels,
            reference_pixels=self.reference_pixels,
            difference_pixels=self.difference_pixels,
        )


class ZOGYEpochPersistenceBundle(BaseModel):
    """Immutable persistence bundle uniting ZOGY detections, stamps, and calibration."""

    model_config = _BRIDGE_MODEL_CONFIG

    context: EpochPersistenceContext
    detections: tuple[OLAPCandidateDetectionRecord, ...]
    cutout_stamps: tuple[OLAPCutoutStampRecord, ...]
    theoretical_noise_std: float = Field(..., gt=0.0)
    flux_scale_diff: float = Field(..., gt=0.0)
    zero_point: ZeroPointResult | None = Field(default=None)


class OLAPEpochPersistResult(BaseModel):
    """Immutable summary returned after persisting an epoch bundle to DuckDB/Parquet."""

    model_config = _BRIDGE_MODEL_CONFIG

    job_id: str = Field(..., min_length=1, max_length=128)
    field_id: str = Field(..., min_length=1, max_length=128)
    observation_night: date
    persisted_detection_count: int = Field(..., ge=0)
    persisted_cutout_count: int = Field(..., ge=0)
    detections_parquet_path: Path | None = Field(default=None)
    cutouts_parquet_path: Path | None = Field(default=None)


class TelescopeProfileRecord(BaseModel):
    """Immutable snapshot of a normalized 3NF ``TelescopeProfileORM`` row."""

    model_config = _BRIDGE_MODEL_CONFIG

    profile_id: str = Field(..., min_length=1, max_length=64)
    telescope: TelescopeConfig
    created_at_utc: datetime


class PipelineJobCreateSpec(BaseModel):
    """Validated input contract for registering a new ``PipelineJobORM`` record."""

    model_config = _BRIDGE_MODEL_CONFIG

    job_id: str = Field(..., min_length=1, max_length=128)
    target_field_id: str = Field(..., min_length=1, max_length=128)
    observation_night_utc: date
    ra_min_deg: float = Field(..., ge=0.0, lt=360.0)
    ra_max_deg: float = Field(..., ge=0.0, le=360.0)
    dec_min_deg: float = Field(..., ge=-90.0, le=90.0)
    dec_max_deg: float = Field(..., ge=-90.0, le=90.0)
    max_mag: float
    profile_id: str | None = Field(default=None, min_length=1, max_length=64)
    submitted_by_operator_id: UUID | None = Field(default=None)
    prefect_flow_run_id: UUID | None = Field(default=None)

    @field_validator("job_id", "target_field_id")
    @classmethod
    def _check_required_ids(cls, value: str, info: Any) -> str:
        label = str(info.field_name or "identifier")
        return _validate_safe_linkage_id(label, value)

    @field_validator("profile_id")
    @classmethod
    def _check_optional_profile_id(cls, value: str | None) -> str | None:
        if value is not None:
            return _validate_safe_linkage_id("profile_id", value)
        return None

    @model_validator(mode="after")
    def _check_declination_order(self) -> PipelineJobCreateSpec:
        if self.dec_min_deg > self.dec_max_deg:
            msg = "dec_min_deg must be less than or equal to dec_max_deg."
            raise ValueError(msg)
        return self


class PipelineJobRecord(BaseModel):
    """Immutable snapshot of a persisted ``PipelineJobORM`` row."""

    model_config = _BRIDGE_MODEL_CONFIG

    job_id: str
    submitted_by_operator_id: UUID | None = None
    profile_id: str | None = None
    target_field_id: str
    observation_night_utc: date
    ra_min_deg: float
    ra_max_deg: float
    dec_min_deg: float
    dec_max_deg: float
    max_mag: float
    prefect_flow_run_id: UUID | None = None
    status: PipelineJobStatus
    matched_catalog_sources: int | None = None
    brightest_catalog_mag_g: float | None = None
    parquet_partition_uri: str | None = None
    error_summary: str | None = None
    submitted_at_utc: datetime
    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None


class PipelineTaskCreateSpec(BaseModel):
    """Validated input contract for registering a new ``PipelineTaskORM`` record."""

    model_config = _BRIDGE_MODEL_CONFIG

    job_id: str = Field(..., min_length=1, max_length=128)
    task_name: str = Field(..., min_length=1, max_length=128)
    module_number: int = Field(..., ge=1, le=10)
    max_retries: int = Field(default=3, ge=0)
    backoff_factor_s: float = Field(default=1.0, ge=0.0)
    prefect_task_run_id: UUID | None = Field(default=None)

    @field_validator("job_id")
    @classmethod
    def _check_job_id(cls, value: str) -> str:
        return _validate_safe_linkage_id("job_id", value)


class PipelineTaskRecord(BaseModel):
    """Immutable snapshot of a persisted ``PipelineTaskORM`` row."""

    model_config = _BRIDGE_MODEL_CONFIG

    task_run_id: UUID
    job_id: str
    task_name: str
    module_number: int
    prefect_task_run_id: UUID | None = None
    status: PipelineTaskStatus
    attempt_count: int
    max_retries: int
    backoff_factor_s: float
    next_retry_at_utc: datetime | None = None
    started_at_utc: datetime | None = None
    finished_at_utc: datetime | None = None


class PipelineTaskLogRecord(BaseModel):
    """Immutable snapshot of a structured ``PipelineTaskLogORM`` diagnostic entry."""

    model_config = _BRIDGE_MODEL_CONFIG

    log_id: int
    task_run_id: UUID
    attempt_number: int
    severity: TaskLogSeverity
    event_code: str | None = None
    message: str
    logged_at_utc: datetime


# ---------------------------------------------------------------------------
# Pure Domain-to-Persistence Mappers (Modules 3 & 4 -> Module 2)
# ---------------------------------------------------------------------------


def map_astrometry_photometry_to_detection(
    *,
    candidate_id: str,
    context: EpochPersistenceContext,
    astrometry: AstrometricMeasurement,
    barycentric_time: BarycentricTimeResult,
    photometry: AperturePhotometryMeasurement | None = None,
    x_centroid_px: float | None = None,
    y_centroid_px: float | None = None,
    snr_override: float | None = None,
    cnn_score: float | None = None,
    model_version: str | None = None,
) -> OLAPCandidateDetectionRecord:
    """Purely map Module 3 astrometry, BJD_TDB, and photometry into an OLAP detection row."""
    astro_unc_arcsec = math.hypot(
        astrometry.sigma_ra_cosdec_arcsec,
        astrometry.sigma_dec_arcsec,
    )

    resolved_x = (
        x_centroid_px
        if x_centroid_px is not None
        else (photometry.x_px if photometry is not None else None)
    )
    resolved_y = (
        y_centroid_px
        if y_centroid_px is not None
        else (photometry.y_px if photometry is not None else None)
    )

    net_flux: float | None = None
    flux_err: float | None = None
    mag: float | None = None
    mag_err: float | None = None
    derived_snr: float | None = None

    if photometry is not None:
        net_flux = photometry.net_flux_adu
        flux_err = photometry.flux_err_adu
        if photometry.calibrated_mag is not None:
            mag = photometry.calibrated_mag
            mag_err = photometry.calibrated_mag_err
        else:
            mag = photometry.instrumental_mag
            mag_err = photometry.instrumental_mag_err

        if photometry.net_flux_adu > 0.0 and photometry.flux_err_adu > 0.0:
            derived_snr = photometry.net_flux_adu / photometry.flux_err_adu
        elif photometry.net_flux_adu <= 0.0:
            derived_snr = 0.0

    resolved_snr = snr_override if snr_override is not None else derived_snr

    return OLAPCandidateDetectionRecord(
        candidate_id=candidate_id,
        job_id=context.job_id,
        observation_night=context.observation_night,
        field_id=context.field_id,
        ra_deg=astrometry.ra_deg,
        dec_deg=astrometry.dec_deg,
        bjd_tdb=barycentric_time.bjd_tdb,
        x_centroid_px=resolved_x,
        y_centroid_px=resolved_y,
        net_flux_adu=net_flux,
        flux_err_adu=flux_err,
        mag=mag,
        mag_err=mag_err,
        snr=resolved_snr,
        cnn_score=cnn_score,
        model_version=model_version,
        astrometric_uncertainty_arcsec=astro_unc_arcsec,
        astrometric_residual_arcsec=None,
        tracklet_id=None,
    )


def build_zogy_persistence_bundle(
    *,
    context: EpochPersistenceContext,
    zogy_result: ZOGYResult,
    cutout_batch: CutoutBatchResult,
    astrometry_measurements: Sequence[AstrometricMeasurement],
    barycentric_times: Sequence[BarycentricTimeResult] | BarycentricTimeResult,
    photometry_measurements: Sequence[AperturePhotometryMeasurement] | None = None,
    zero_point: ZeroPointResult | None = None,
    fallback_mag: float = 99.0,
    fallback_mag_err: float = 0.0,
    pixel_decimals: int = 4,
) -> ZOGYEpochPersistenceBundle:
    """Purely assemble an immutable OLAP epoch bundle from ZOGY, cutouts, and calibrations."""
    n_candidates = len(cutout_batch.candidates)
    if len(astrometry_measurements) != n_candidates:
        msg = (
            f"astrometry_measurements length ({len(astrometry_measurements)}) must match "
            f"cutout_batch.candidates length ({n_candidates})."
        )
        raise ValueError(msg)

    if isinstance(barycentric_times, Sequence):
        if len(barycentric_times) != n_candidates:
            msg = (
                f"barycentric_times length ({len(barycentric_times)}) must match "
                f"cutout_batch.candidates length ({n_candidates})."
            )
            raise ValueError(msg)
        bjd_list: list[BarycentricTimeResult] = list(barycentric_times)
    else:
        bjd_list = [barycentric_times] * n_candidates

    if photometry_measurements is not None and len(photometry_measurements) != n_candidates:
        msg = (
            f"photometry_measurements length ({len(photometry_measurements)}) must match "
            f"cutout_batch.candidates length ({n_candidates})."
        )
        raise ValueError(msg)

    triplets_np = np.asarray(
        cutout_batch.triplets.detach().cpu().numpy(),
        dtype=np.float64,
    )
    rounded_triplets = np.round(triplets_np, decimals=pixel_decimals)
    stamp_size: CutoutStampSize = cutout_batch.stamp_size
    half_stamp = float(stamp_size // 2)
    max_coord = float(stamp_size - 1)

    detections: list[OLAPCandidateDetectionRecord] = []
    cutouts: list[OLAPCutoutStampRecord] = []

    for idx, peak in enumerate(cutout_batch.candidates):
        phot_item = (
            photometry_measurements[idx] if photometry_measurements is not None else None
        )
        det_rec = map_astrometry_photometry_to_detection(
            candidate_id=peak.candidate_id,
            context=context,
            astrometry=astrometry_measurements[idx],
            barycentric_time=bjd_list[idx],
            photometry=phot_item,
            x_centroid_px=peak.x_centroid_px,
            y_centroid_px=peak.y_centroid_px,
            snr_override=peak.snr,
        )
        detections.append(det_rec)

        stamp_mag = det_rec.mag if det_rec.mag is not None else fallback_mag
        stamp_mag_err = (
            det_rec.mag_err if det_rec.mag_err is not None else fallback_mag_err
        )
        local_x = min(
            max(half_stamp + (peak.x_centroid_px - float(peak.x_px)), 0.0),
            max_coord,
        )
        local_y = min(
            max(half_stamp + (peak.y_centroid_px - float(peak.y_px)), 0.0),
            max_coord,
        )

        triplet_arr = rounded_triplets[idx]
        cutout_rec = OLAPCutoutStampRecord(
            candidate_id=peak.candidate_id,
            epoch_index=context.epoch_index,
            observation_night=context.observation_night,
            field_id=context.field_id,
            cutout_size=stamp_size,
            bjd_tdb=bjd_list[idx].bjd_tdb,
            mag=stamp_mag,
            mag_err=stamp_mag_err,
            target_x_px=local_x,
            target_y_px=local_y,
            science_pixels=triplet_arr[0].tolist(),
            reference_pixels=triplet_arr[1].tolist(),
            difference_pixels=triplet_arr[2].tolist(),
        )
        cutouts.append(cutout_rec)

    return ZOGYEpochPersistenceBundle(
        context=context,
        detections=tuple(detections),
        cutout_stamps=tuple(cutouts),
        theoretical_noise_std=zogy_result.theoretical_noise_std,
        flux_scale_diff=zogy_result.flux_scale_diff,
        zero_point=zero_point,
    )


# ---------------------------------------------------------------------------
# DuckDB & Hive-Partitioned Parquet OLAP Persistence Adapters
# ---------------------------------------------------------------------------


def persist_olap_detections(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
    detections: Sequence[OLAPCandidateDetectionRecord],
) -> int:
    """Batch-insert validated ``OLAPCandidateDetectionRecord`` rows into DuckDB."""
    if not detections:
        return 0
    conn = _unwrap_duckdb_connection(conn_or_db)
    rows: list[tuple[Any, ...]] = [
        (
            det.candidate_id,
            det.job_id,
            det.observation_night,
            det.field_id,
            det.ra_deg,
            det.dec_deg,
            det.bjd_tdb,
            det.x_centroid_px,
            det.y_centroid_px,
            det.net_flux_adu,
            det.flux_err_adu,
            det.mag,
            det.mag_err,
            det.snr,
            det.cnn_score,
            det.model_version,
            det.astrometric_uncertainty_arcsec,
            det.astrometric_residual_arcsec,
            det.tracklet_id,
        )
        for det in detections
    ]
    placeholders = ", ".join(["?"] * 19)
    conn.executemany(
        f"INSERT INTO candidate_detections ({_DETECTION_COLUMNS_SQL}) "
        f"VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def fetch_olap_detections_for_job(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
    job_id: str,
) -> list[OLAPCandidateDetectionRecord]:
    """Fetch all ``candidate_detections`` rows for ``job_id`` ordered by triage priority."""
    safe_job_id = _validate_safe_linkage_id("job_id", job_id)
    conn = _unwrap_duckdb_connection(conn_or_db)
    raw_rows = conn.execute(
        f"SELECT {_DETECTION_COLUMNS_SQL} FROM candidate_detections "
        "WHERE job_id = ? "
        "ORDER BY cnn_score DESC NULLS LAST, snr DESC NULLS LAST, candidate_id ASC",
        [safe_job_id],
    ).fetchall()

    return [
        OLAPCandidateDetectionRecord(
            candidate_id=str(row[0]),
            job_id=str(row[1]),
            observation_night=row[2],
            field_id=str(row[3]),
            ra_deg=float(row[4]),
            dec_deg=float(row[5]),
            bjd_tdb=float(row[6]),
            x_centroid_px=float(row[7]) if row[7] is not None else None,
            y_centroid_px=float(row[8]) if row[8] is not None else None,
            net_flux_adu=float(row[9]) if row[9] is not None else None,
            flux_err_adu=float(row[10]) if row[10] is not None else None,
            mag=float(row[11]) if row[11] is not None else None,
            mag_err=float(row[12]) if row[12] is not None else None,
            snr=float(row[13]) if row[13] is not None else None,
            cnn_score=float(row[14]) if row[14] is not None else None,
            model_version=str(row[15]) if row[15] is not None else None,
            astrometric_uncertainty_arcsec=float(row[16]),
            astrometric_residual_arcsec=float(row[17]) if row[17] is not None else None,
            tracklet_id=str(row[18]) if row[18] is not None else None,
        )
        for row in raw_rows
    ]


def persist_olap_cutout_stamps(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
    stamps: Sequence[OLAPCutoutStampRecord],
) -> int:
    """Batch-insert validated ``OLAPCutoutStampRecord`` rows into DuckDB."""
    if not stamps:
        return 0
    conn = _unwrap_duckdb_connection(conn_or_db)
    rows: list[tuple[Any, ...]] = [
        (
            stamp.candidate_id,
            stamp.epoch_index,
            stamp.observation_night,
            stamp.field_id,
            stamp.cutout_size,
            stamp.bjd_tdb,
            stamp.mag,
            stamp.mag_err,
            stamp.target_x_px,
            stamp.target_y_px,
            stamp.science_pixels,
            stamp.reference_pixels,
            stamp.difference_pixels,
        )
        for stamp in stamps
    ]
    placeholders = ", ".join(["?"] * 13)
    conn.executemany(
        f"INSERT INTO candidate_cutout_stamps ({_CUTOUT_COLUMNS_SQL}) "
        f"VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def fetch_olap_cutout_stamps(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
    candidate_id: str,
) -> list[OLAPCutoutStampRecord]:
    """Fetch all ``candidate_cutout_stamps`` rows for ``candidate_id`` ordered by epoch."""
    safe_cid = _validate_safe_linkage_id("candidate_id", candidate_id)
    conn = _unwrap_duckdb_connection(conn_or_db)
    raw_rows = conn.execute(
        f"SELECT {_CUTOUT_COLUMNS_SQL} FROM candidate_cutout_stamps "
        "WHERE candidate_id = ? ORDER BY epoch_index ASC",
        [safe_cid],
    ).fetchall()

    results: list[OLAPCutoutStampRecord] = []
    for row in raw_rows:
        raw_size = int(row[4])
        cutout_size_lit: CutoutStampSize = 31 if raw_size == 31 else 51
        results.append(
            OLAPCutoutStampRecord(
                candidate_id=str(row[0]),
                epoch_index=int(row[1]),
                observation_night=row[2],
                field_id=str(row[3]),
                cutout_size=cutout_size_lit,
                bjd_tdb=float(row[5]),
                mag=float(row[6]),
                mag_err=float(row[7]),
                target_x_px=float(row[8]),
                target_y_px=float(row[9]),
                science_pixels=_coerce_pixel_grid(row[10]),
                reference_pixels=(
                    _coerce_pixel_grid(row[11]) if row[11] is not None else None
                ),
                difference_pixels=(
                    _coerce_pixel_grid(row[12]) if row[12] is not None else None
                ),
            )
        )
    return results


def persist_zogy_bundle_to_olap(
    conn_or_db: duckdb.DuckDBPyConnection | CatalogDatabase,
    bundle: ZOGYEpochPersistenceBundle,
    *,
    olap_config: OLAPSchemaConfig | None = None,
    export_parquet: bool = False,
) -> OLAPEpochPersistResult:
    """Persist a ``ZOGYEpochPersistenceBundle`` into DuckDB and optionally Hive Parquet."""
    initialize_olap_schema(conn_or_db)
    det_count = persist_olap_detections(conn_or_db, bundle.detections)
    cut_count = persist_olap_cutout_stamps(conn_or_db, bundle.cutout_stamps)

    det_parquet_path: Path | None = None
    cut_parquet_path: Path | None = None

    if export_parquet:
        effective_cfg = olap_config if olap_config is not None else OLAPSchemaConfig()
        det_parquet_path = export_job_table_to_parquet(
            conn_or_db,
            effective_cfg,
            dataset_name="candidate_detections",
            observation_night=bundle.context.observation_night,
            field_id=bundle.context.field_id,
            job_id=bundle.context.job_id,
        )
        cut_parquet_path = export_job_table_to_parquet(
            conn_or_db,
            effective_cfg,
            dataset_name="candidate_cutout_stamps",
            observation_night=bundle.context.observation_night,
            field_id=bundle.context.field_id,
            job_id=bundle.context.job_id,
        )

    return OLAPEpochPersistResult(
        job_id=bundle.context.job_id,
        field_id=bundle.context.field_id,
        observation_night=bundle.context.observation_night,
        persisted_detection_count=det_count,
        persisted_cutout_count=cut_count,
        detections_parquet_path=det_parquet_path,
        cutouts_parquet_path=cut_parquet_path,
    )


# ---------------------------------------------------------------------------
# PostgreSQL / SQLite 3NF OLTP Lifecycle Persistence Adapters
# ---------------------------------------------------------------------------


def _orm_to_telescope_profile_record(orm_row: TelescopeProfileORM) -> TelescopeProfileRecord:
    """Convert a ``TelescopeProfileORM`` entity into an immutable ``TelescopeProfileRecord``."""
    return TelescopeProfileRecord(
        profile_id=orm_row.profile_id,
        telescope=TelescopeConfig(
            name=orm_row.name,
            pixel_scale=orm_row.pixel_scale_arcsec,
            gain=orm_row.gain_e_per_adu,
            read_noise=orm_row.read_noise_e,
            saturation_limit=orm_row.saturation_limit_adu,
            dark_current=orm_row.dark_current_e_per_s,
            aperture_diameter_m=orm_row.aperture_diameter_m,
            focal_length_m=orm_row.focal_length_m,
            observatory=ObservatoryLocation(
                latitude_deg=orm_row.latitude_deg,
                longitude_deg=orm_row.longitude_deg,
                elevation_m=orm_row.elevation_m,
                mpc_code=orm_row.mpc_code,
            ),
        ),
        created_at_utc=orm_row.created_at_utc,
    )


def _orm_to_pipeline_job_record(orm_job: PipelineJobORM) -> PipelineJobRecord:
    """Convert a ``PipelineJobORM`` entity into an immutable ``PipelineJobRecord``."""
    status_val: Any = orm_job.status
    return PipelineJobRecord(
        job_id=orm_job.job_id,
        submitted_by_operator_id=orm_job.submitted_by_operator_id,
        profile_id=orm_job.profile_id,
        target_field_id=orm_job.target_field_id,
        observation_night_utc=orm_job.observation_night_utc,
        ra_min_deg=orm_job.ra_min_deg,
        ra_max_deg=orm_job.ra_max_deg,
        dec_min_deg=orm_job.dec_min_deg,
        dec_max_deg=orm_job.dec_max_deg,
        max_mag=orm_job.max_mag,
        prefect_flow_run_id=orm_job.prefect_flow_run_id,
        status=status_val,
        matched_catalog_sources=orm_job.matched_catalog_sources,
        brightest_catalog_mag_g=orm_job.brightest_catalog_mag_g,
        parquet_partition_uri=orm_job.parquet_partition_uri,
        error_summary=orm_job.error_summary,
        submitted_at_utc=orm_job.submitted_at_utc,
        started_at_utc=orm_job.started_at_utc,
        completed_at_utc=orm_job.completed_at_utc,
    )


def _orm_to_pipeline_task_record(orm_task: PipelineTaskORM) -> PipelineTaskRecord:
    """Convert a ``PipelineTaskORM`` entity into an immutable ``PipelineTaskRecord``."""
    status_val: Any = orm_task.status
    return PipelineTaskRecord(
        task_run_id=orm_task.task_run_id,
        job_id=orm_task.job_id,
        task_name=orm_task.task_name,
        module_number=orm_task.module_number,
        prefect_task_run_id=orm_task.prefect_task_run_id,
        status=status_val,
        attempt_count=orm_task.attempt_count,
        max_retries=orm_task.max_retries,
        backoff_factor_s=orm_task.backoff_factor_s,
        next_retry_at_utc=orm_task.next_retry_at_utc,
        started_at_utc=orm_task.started_at_utc,
        finished_at_utc=orm_task.finished_at_utc,
    )


def _orm_to_pipeline_task_log_record(orm_log: PipelineTaskLogORM) -> PipelineTaskLogRecord:
    """Convert a ``PipelineTaskLogORM`` entity into an immutable ``PipelineTaskLogRecord``."""
    severity_val: Any = orm_log.severity
    return PipelineTaskLogRecord(
        log_id=orm_log.log_id,
        task_run_id=orm_log.task_run_id,
        attempt_number=orm_log.attempt_number,
        severity=severity_val,
        event_code=orm_log.event_code,
        message=orm_log.message,
        logged_at_utc=orm_log.logged_at_utc,
    )


def upsert_telescope_profile(
    engine: sa.Engine,
    profile_id: str,
    telescope: TelescopeConfig,
) -> TelescopeProfileRecord:
    """Insert or update a normalized 3NF ``TelescopeProfileORM`` from ``TelescopeConfig``."""
    safe_pid = _validate_safe_linkage_id("profile_id", profile_id)
    with Session(engine) as session:
        existing = session.get(TelescopeProfileORM, safe_pid)
        if existing is None:
            existing = TelescopeProfileORM(
                profile_id=safe_pid,
                name=telescope.name,
                mpc_code=telescope.observatory.mpc_code,
                pixel_scale_arcsec=telescope.pixel_scale,
                gain_e_per_adu=telescope.gain,
                read_noise_e=telescope.read_noise,
                saturation_limit_adu=telescope.saturation_limit,
                dark_current_e_per_s=telescope.dark_current,
                latitude_deg=telescope.observatory.latitude_deg,
                longitude_deg=telescope.observatory.longitude_deg,
                elevation_m=telescope.observatory.elevation_m,
                aperture_diameter_m=telescope.aperture_diameter_m,
                focal_length_m=telescope.focal_length_m,
            )
            session.add(existing)
        else:
            existing.name = telescope.name
            existing.mpc_code = telescope.observatory.mpc_code
            existing.pixel_scale_arcsec = telescope.pixel_scale
            existing.gain_e_per_adu = telescope.gain
            existing.read_noise_e = telescope.read_noise
            existing.saturation_limit_adu = telescope.saturation_limit
            existing.dark_current_e_per_s = telescope.dark_current
            existing.latitude_deg = telescope.observatory.latitude_deg
            existing.longitude_deg = telescope.observatory.longitude_deg
            existing.elevation_m = telescope.observatory.elevation_m
            existing.aperture_diameter_m = telescope.aperture_diameter_m
            existing.focal_length_m = telescope.focal_length_m

        session.commit()
        session.refresh(existing)
        return _orm_to_telescope_profile_record(existing)


def get_telescope_profile(
    engine: sa.Engine,
    profile_id: str,
) -> TelescopeProfileRecord | None:
    """Fetch a ``TelescopeProfileORM`` by primary key and reconstruct ``TelescopeConfig``."""
    safe_pid = _validate_safe_linkage_id("profile_id", profile_id)
    with Session(engine) as session:
        orm_row = session.get(TelescopeProfileORM, safe_pid)
        if orm_row is None:
            return None
        return _orm_to_telescope_profile_record(orm_row)


def register_pipeline_job(
    engine: sa.Engine,
    spec: PipelineJobCreateSpec,
) -> PipelineJobRecord:
    """Create a new ``PipelineJobORM`` row in ``queued`` state inside the OLTP database."""
    with Session(engine) as session:
        orm_job = PipelineJobORM(
            job_id=spec.job_id,
            submitted_by_operator_id=spec.submitted_by_operator_id,
            profile_id=spec.profile_id,
            target_field_id=spec.target_field_id,
            observation_night_utc=spec.observation_night_utc,
            ra_min_deg=spec.ra_min_deg,
            ra_max_deg=spec.ra_max_deg,
            dec_min_deg=spec.dec_min_deg,
            dec_max_deg=spec.dec_max_deg,
            max_mag=spec.max_mag,
            prefect_flow_run_id=spec.prefect_flow_run_id,
            status="queued",
        )
        session.add(orm_job)
        session.commit()
        session.refresh(orm_job)
        return _orm_to_pipeline_job_record(orm_job)


def transition_pipeline_job(
    engine: sa.Engine,
    job_id: str,
    *,
    new_status: PipelineJobStatus,
    matched_catalog_sources: int | None = None,
    brightest_catalog_mag_g: float | None = None,
    parquet_partition_uri: str | None = None,
    error_summary: str | None = None,
) -> PipelineJobRecord:
    """Atomically validate and apply a lifecycle state transition on ``PipelineJobORM``."""
    safe_job_id = _validate_safe_linkage_id("job_id", job_id)
    now_utc = _utc_now()
    with Session(engine) as session:
        orm_job = session.get(PipelineJobORM, safe_job_id)
        if orm_job is None:
            msg = f"PipelineJobORM {safe_job_id!r} not found."
            raise KeyError(msg)

        allowed = _ALLOWED_JOB_TRANSITIONS.get(orm_job.status, frozenset())
        if new_status not in allowed:
            msg = (
                f"Illegal pipeline job status transition from {orm_job.status!r} "
                f"to {new_status!r} for job {safe_job_id!r}."
            )
            raise ValueError(msg)

        orm_job.status = new_status
        if new_status == "running" and orm_job.started_at_utc is None:
            orm_job.started_at_utc = now_utc
        elif new_status in {"completed", "failed", "cancelled"}:
            orm_job.completed_at_utc = now_utc

        if matched_catalog_sources is not None:
            orm_job.matched_catalog_sources = matched_catalog_sources
        if brightest_catalog_mag_g is not None:
            orm_job.brightest_catalog_mag_g = brightest_catalog_mag_g
        if parquet_partition_uri is not None:
            orm_job.parquet_partition_uri = parquet_partition_uri
        if error_summary is not None:
            orm_job.error_summary = error_summary

        session.commit()
        session.refresh(orm_job)
        return _orm_to_pipeline_job_record(orm_job)


def get_pipeline_job(
    engine: sa.Engine,
    job_id: str,
) -> PipelineJobRecord | None:
    """Fetch an immutable ``PipelineJobRecord`` snapshot by ``job_id``, or ``None``."""
    safe_job_id = _validate_safe_linkage_id("job_id", job_id)
    with Session(engine) as session:
        orm_job = session.get(PipelineJobORM, safe_job_id)
        if orm_job is None:
            return None
        return _orm_to_pipeline_job_record(orm_job)


def register_pipeline_task(
    engine: sa.Engine,
    spec: PipelineTaskCreateSpec,
) -> PipelineTaskRecord:
    """Create a new ``PipelineTaskORM`` row in ``pending`` state for a registered job."""
    with Session(engine) as session:
        if session.get(PipelineJobORM, spec.job_id) is None:
            msg = f"Parent PipelineJobORM {spec.job_id!r} does not exist."
            raise KeyError(msg)

        orm_task = PipelineTaskORM(
            job_id=spec.job_id,
            task_name=spec.task_name,
            module_number=spec.module_number,
            prefect_task_run_id=spec.prefect_task_run_id,
            status="pending",
            attempt_count=1,
            max_retries=spec.max_retries,
            backoff_factor_s=spec.backoff_factor_s,
        )
        session.add(orm_task)
        session.commit()
        session.refresh(orm_task)
        return _orm_to_pipeline_task_record(orm_task)


def transition_pipeline_task(
    engine: sa.Engine,
    task_run_id: UUID,
    *,
    new_status: PipelineTaskStatus,
    next_retry_at_utc: datetime | None = None,
    log_severity: TaskLogSeverity = "INFO",
    log_message: str | None = None,
    event_code: str | None = None,
) -> PipelineTaskRecord:
    """Atomically transition ``PipelineTaskORM`` state and optionally emit a task log."""
    now_utc = _utc_now()
    with Session(engine) as session:
        orm_task = session.get(PipelineTaskORM, task_run_id)
        if orm_task is None:
            msg = f"PipelineTaskORM {task_run_id!s} not found."
            raise KeyError(msg)

        prev_status = orm_task.status
        allowed = _ALLOWED_TASK_TRANSITIONS.get(prev_status, frozenset())
        if new_status not in allowed:
            msg = (
                f"Illegal pipeline task status transition from {prev_status!r} "
                f"to {new_status!r} for task {task_run_id!s}."
            )
            raise ValueError(msg)

        if prev_status == "retrying" and new_status == "running":
            next_attempt = orm_task.attempt_count + 1
            if next_attempt > orm_task.max_retries + 1:
                msg = (
                    f"Task {task_run_id!s} exceeded max_retries ({orm_task.max_retries}) "
                    "during retry transition."
                )
                raise ValueError(msg)
            orm_task.attempt_count = next_attempt
            orm_task.next_retry_at_utc = None
            if orm_task.started_at_utc is None:
                orm_task.started_at_utc = now_utc
        elif new_status == "running":
            if orm_task.started_at_utc is None:
                orm_task.started_at_utc = now_utc
            orm_task.next_retry_at_utc = None
        elif new_status == "retrying":
            orm_task.next_retry_at_utc = next_retry_at_utc
        elif new_status in {"completed", "failed", "skipped"}:
            orm_task.finished_at_utc = now_utc
            orm_task.next_retry_at_utc = None

        orm_task.status = new_status

        if log_message is not None:
            log_entry = PipelineTaskLogORM(
                task_run_id=orm_task.task_run_id,
                attempt_number=orm_task.attempt_count,
                severity=log_severity,
                event_code=event_code,
                message=log_message,
                logged_at_utc=now_utc,
            )
            session.add(log_entry)

        session.commit()
        session.refresh(orm_task)
        return _orm_to_pipeline_task_record(orm_task)


def list_pipeline_task_logs(
    engine: sa.Engine,
    task_run_id: UUID,
) -> list[PipelineTaskLogRecord]:
    """Return ordered ``PipelineTaskLogRecord`` entries for ``task_run_id``."""
    with Session(engine) as session:
        rows = session.scalars(
            sa.select(PipelineTaskLogORM)
            .where(PipelineTaskLogORM.task_run_id == task_run_id)
            .order_by(
                PipelineTaskLogORM.logged_at_utc.asc(),
                PipelineTaskLogORM.log_id.asc(),
            )
        ).all()
        return [_orm_to_pipeline_task_log_record(row) for row in rows]
