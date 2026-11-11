"""Strict Pydantic v2 request, response, and WebSocket contracts for Module 10."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

from LUMIA.api.auth import (
    AccessTokenResponse,
    LoginRequest,
    OperatorProfileResponse,
    WebSocketTicketResponse,
)
from LUMIA.core.schemas import CandidateDetection, LumiaBaseModel
from LUMIA.data_pipeline.database import CatalogQueryParams  # noqa: TC001
from LUMIA.data_pipeline.oltp_repositories import (  # noqa: TC001
    JobLifecycle,
    TriageDisposition,
)

CutoutStampSize = Literal[31, 51]
SafeIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]{1,128}$"),
]


class PipelineStatusResponse(LumiaBaseModel):
    status: Literal["operational", "degraded"]
    duckdb_connected: bool
    postgres_connected: bool = True
    total_jobs: int = Field(ge=0)
    theme_mode: Literal["dark"] = "dark"
    accent_color_hex: str = Field(default="#581c87", pattern=r"^#[0-9a-fA-F]{6}$")


APIErrorCode = Literal[
    "BAD_REQUEST",
    "AUTH_REQUIRED",
    "INSUFFICIENT_ROLE",
    "RESOURCE_NOT_FOUND",
    "STATE_CONFLICT",
    "RATE_LIMITED",
    "STORAGE_UNAVAILABLE",
]


class APIErrorResponse(LumiaBaseModel):
    error_code: APIErrorCode
    detail: str


class DetectionSummaryResponse(LumiaBaseModel):
    total_candidates: int = Field(ge=0)
    high_confidence_count: int = Field(ge=0)
    distinct_fields: int = Field(ge=0)
    mean_snr: float | None = None
    brightest_mag: float | None = None


class CandidateListResponse(LumiaBaseModel):
    total_returned: int = Field(ge=0)
    candidates: list[CandidateDetection]


class CandidateDetailResponse(CandidateDetection):
    disposition: TriageDisposition = "unreviewed"
    reviewed_by_operator_id: UUID | None = None
    updated_at_utc: datetime | None = None


class JobSubmissionRequest(LumiaBaseModel):
    job_id: SafeIdentifier
    target_field: SafeIdentifier
    observation_night_utc: date = Field(default_factory=lambda: datetime.now(UTC).date())
    profile_id: SafeIdentifier | None = None
    query_params: CatalogQueryParams

    @field_validator("observation_night_utc", mode="before")
    @classmethod
    def _parse_iso_date(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("observation_night_utc must use YYYY-MM-DD.") from exc
            if parsed.isoformat() != value:
                raise ValueError("observation_night_utc must use YYYY-MM-DD.")
            return parsed
        return value


class JobStatusResponse(LumiaBaseModel):
    job_id: SafeIdentifier
    target_field: SafeIdentifier
    observation_night_utc: date
    matched_catalog_sources: int | None = Field(default=None, ge=0)
    brightest_catalog_mag_g: float | None = None
    status: JobLifecycle
    submitted_at_utc: datetime
    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None
    profile_id: SafeIdentifier | None = None


class JobListResponse(LumiaBaseModel):
    total_returned: int = Field(ge=0)
    jobs: list[JobStatusResponse]


class CandidateTriageRequest(LumiaBaseModel):
    disposition: TriageDisposition
    notes: str | None = Field(default=None, max_length=2000)


class CandidateTriageResponse(LumiaBaseModel):
    triage_event_id: int = Field(ge=1)
    candidate_id: SafeIdentifier
    job_id: SafeIdentifier
    field_id: SafeIdentifier
    previous_disposition: TriageDisposition
    disposition: TriageDisposition
    reviewed_by_operator_id: UUID
    cnn_score_at_review: float | None = None
    model_version_at_review: str | None = None
    queued_for_hdf5_export: bool
    decided_at_utc: datetime


class CatalogStarItem(LumiaBaseModel):
    source_id: int = Field(ge=0)
    ra_deg: float = Field(ge=0.0, lt=360.0)
    dec_deg: float = Field(ge=-90.0, le=90.0)
    mag_g: float


class CatalogQueryResponse(LumiaBaseModel):
    query: CatalogQueryParams
    matched_count: int = Field(ge=0)
    items: list[CatalogStarItem]


class CandidateCutoutEpoch(LumiaBaseModel):
    epoch_index: int = Field(ge=0)
    bjd_tdb: float
    mag: float
    mag_err: float = Field(ge=0.0)
    target_x_px: float = Field(ge=0.0)
    target_y_px: float = Field(ge=0.0)
    pixels: list[list[float]]
    reference_pixels: list[list[float]] | None = None
    difference_pixels: list[list[float]] | None = None
    survey_id: str = Field(
        default="ZTF",
        description="Originating survey or observatory identifier (e.g., 'ZTF', 'Pan-STARRS', 'Rubin', 'JWST').",
    )
    filter_band: str = Field(
        default="r",
        description="Optical/NIR photometric filter bandpass (e.g., 'g', 'r', 'i', 'z').",
    )
    pixel_scale_arcsec: float = Field(
        default=1.012,
        gt=0.0,
        description="Angular pixel scale in arcseconds per pixel.",
    )
    fov_arcsec: float | None = Field(
        default=None,
        description="Field of view width/height of the cutout in arcseconds.",
    )


class CandidateCutoutSequenceResponse(LumiaBaseModel):
    candidate_id: SafeIdentifier
    field_id: SafeIdentifier
    cutout_size: CutoutStampSize
    epochs: list[CandidateCutoutEpoch]
    survey_id: str = Field(
        default="ZTF",
        description="Primary or reference survey ID.",
    )

    @model_validator(mode="after")
    def _validate_epoch_stamp_dimensions(self) -> CandidateCutoutSequenceResponse:
        for epoch in self.epochs:
            for name in ("pixels", "reference_pixels", "difference_pixels"):
                grid = getattr(epoch, name)
                if grid is not None and (
                    len(grid) != self.cutout_size
                    or any(len(row) != self.cutout_size for row in grid)
                ):
                    raise ValueError(
                        f"Epoch {epoch.epoch_index} {name} must be "
                        f"{self.cutout_size}x{self.cutout_size}."
                    )
        return self


class TriageQueueItem(LumiaBaseModel):
    candidate: CandidateDetection
    cutouts: CandidateCutoutSequenceResponse | None = None
    ground_truth_label: int | None = Field(
        default=None,
        description="Ground-truth label if sourced from labeled dataset (1=real, 0=bogus).",
    )


class TriageQueueResponse(LumiaBaseModel):
    total_returned: int = Field(ge=0)
    source: Literal["hdf5_tomoe", "database"]
    split: str = Field(default="test")
    candidates: list[CandidateDetection]
    items: list[TriageQueueItem] = Field(default_factory=list)


class JobSnapshotItem(LumiaBaseModel):
    job_id: SafeIdentifier
    target_field_id: SafeIdentifier
    observation_night_utc: date
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    matched_catalog_sources: int | None = Field(default=None, ge=0)
    brightest_catalog_mag_g: float | None = None
    completed_tasks: int = Field(default=0, ge=0)
    total_tasks: int = Field(default=0, ge=0)
    submitted_at_utc: datetime
    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None


class SnapshotPayload(LumiaBaseModel):
    subscribed_job_id: SafeIdentifier | None = None
    active_jobs: list[JobSnapshotItem]


class JobStateChangedPayload(LumiaBaseModel):
    job_id: SafeIdentifier
    target_field_id: SafeIdentifier
    status: JobLifecycle
    previous_status: JobLifecycle | None = None


class TaskStateChangedPayload(LumiaBaseModel):
    task_run_id: UUID
    job_id: SafeIdentifier
    task_name: str
    module_number: int = Field(ge=1, le=10)
    status: Literal["pending", "running", "retrying", "completed", "failed", "skipped"]
    attempt_count: int = Field(ge=1)
    max_retries: int = Field(ge=0)
    next_retry_at_utc: datetime | None = None
    started_at_utc: datetime | None = None
    finished_at_utc: datetime | None = None


class JobMetricsUpdatedPayload(LumiaBaseModel):
    job_id: SafeIdentifier
    field_id: SafeIdentifier
    extracted_candidates: int = Field(ge=0)
    cnn_passed_candidates: int = Field(ge=0)
    linked_tracklets: int = Field(ge=0)
    brightest_candidate_mag: float | None = None
    mean_candidate_snr: float | None = None


class TaskLogAppendedPayload(LumiaBaseModel):
    log_id: int = Field(ge=1)
    job_id: SafeIdentifier
    task_run_id: UUID
    attempt_number: int = Field(ge=1)
    severity: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    event_code: str | None = Field(default=None, max_length=64)
    message: str
    logged_at_utc: datetime


class HeartbeatPayload(LumiaBaseModel):
    server_status: Literal["operational", "degraded"] = "operational"
    dropped_log_frames: int = Field(default=0, ge=0)


class ErrorPayload(LumiaBaseModel):
    error_code: str
    message: str


class SnapshotEnvelope(LumiaBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    event_type: Literal["snapshot"] = "snapshot"
    sequence_no: int = Field(ge=1)
    emitted_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: SnapshotPayload


class JobStateChangedEnvelope(LumiaBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    event_type: Literal["job.state_changed"] = "job.state_changed"
    sequence_no: int = Field(ge=1)
    emitted_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: JobStateChangedPayload


class TaskStateChangedEnvelope(LumiaBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    event_type: Literal["task.state_changed"] = "task.state_changed"
    sequence_no: int = Field(ge=1)
    emitted_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: TaskStateChangedPayload


class JobMetricsUpdatedEnvelope(LumiaBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    event_type: Literal["job.metrics_updated"] = "job.metrics_updated"
    sequence_no: int = Field(ge=1)
    emitted_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: JobMetricsUpdatedPayload


class TaskLogAppendedEnvelope(LumiaBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    event_type: Literal["task.log_appended"] = "task.log_appended"
    sequence_no: int = Field(ge=1)
    emitted_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: TaskLogAppendedPayload


class HeartbeatEnvelope(LumiaBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    event_type: Literal["heartbeat"] = "heartbeat"
    sequence_no: int = Field(ge=1)
    emitted_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: HeartbeatPayload


class ErrorEnvelope(LumiaBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    event_type: Literal["error"] = "error"
    sequence_no: int = Field(ge=1)
    emitted_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: ErrorPayload


WebSocketEnvelope = Annotated[
    SnapshotEnvelope
    | JobStateChangedEnvelope
    | TaskStateChangedEnvelope
    | JobMetricsUpdatedEnvelope
    | TaskLogAppendedEnvelope
    | HeartbeatEnvelope
    | ErrorEnvelope,
    Field(discriminator="event_type"),
]
WEBSOCKET_ENVELOPE_ADAPTER: TypeAdapter[WebSocketEnvelope] = TypeAdapter(WebSocketEnvelope)

class SettingsResponse(LumiaBaseModel):
    """Current runtime settings including archive storage and compute budgets."""

    archive_storage_path: str = Field(..., description="Telescope FITS archive storage path.")
    num_cpu_threads: int = Field(..., description="Configured CPU worker threads.")
    max_gpu_memory_gb: float = Field(..., description="Configured maximum GPU memory in GB.")
    enable_openmp_threading_control: bool = Field(..., description="OpenMP thread control status.")


class SettingsUpdateRequest(LumiaBaseModel):
    """Payload to update runtime settings."""

    archive_storage_path: str | None = Field(
        default=None,
        min_length=1,
        description="Telescope FITS archive storage path.",
    )
    num_cpu_threads: int | None = Field(
        default=None,
        ge=1,
        description="Optional CPU worker thread count.",
    )
    max_gpu_memory_gb: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional maximum GPU memory in GB.",
    )
    enable_openmp_threading_control: bool | None = Field(
        default=None,
        description="Optional OpenMP thread control flag.",
    )


__all__ = [
    "AccessTokenResponse",
    "APIErrorCode",
    "APIErrorResponse",
    "CandidateCutoutEpoch",
    "CandidateCutoutSequenceResponse",
    "CandidateDetailResponse",
    "CandidateListResponse",
    "CandidateTriageRequest",
    "CandidateTriageResponse",
    "CatalogQueryResponse",
    "CatalogStarItem",
    "DetectionSummaryResponse",
    "ErrorEnvelope",
    "ErrorPayload",
    "HeartbeatEnvelope",
    "HeartbeatPayload",
    "JobListResponse",
    "JobMetricsUpdatedEnvelope",
    "JobMetricsUpdatedPayload",
    "JobSnapshotItem",
    "JobStateChangedEnvelope",
    "JobStateChangedPayload",
    "JobStatusResponse",
    "JobSubmissionRequest",
    "LoginRequest",
    "OperatorProfileResponse",
    "PipelineStatusResponse",
    "SettingsResponse",
    "SettingsUpdateRequest",
    "SnapshotEnvelope",
    "SnapshotPayload",
    "TaskLogAppendedEnvelope",
    "TaskLogAppendedPayload",
    "TaskStateChangedEnvelope",
    "TaskStateChangedPayload",
    "WebSocketEnvelope",
    "WebSocketTicketResponse",
]
