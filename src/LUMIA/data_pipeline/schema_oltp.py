"""SQLAlchemy 2.x 3NF relational schema for the PostgreSQL OLTP plane (Module 2, ADR-0003).

Encapsulates transactional lifecycle state, operator authentication, human Active Learning
triage dispositions, and curated Minor Planet Center (MPC) discovery archives strictly inside
``src/LUMIA/data_pipeline/``. Supports both PostgreSQL 17 production deployments and isolated
in-memory SQLite engines for deterministic unit testing.
"""

from __future__ import annotations

from datetime import UTC, date, datetime  # noqa: TC003 - needed at runtime by SQLAlchemy Mapped
from typing import Any
from uuid import UUID, uuid4  # noqa: TC003 - needed at runtime by SQLAlchemy Mapped

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Double,
    Engine,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    TypeDecorator,
    Uuid,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.elements import ColumnElement

__all__ = [
    "AuthRefreshSessionORM",
    "CandidateTriageEventORM",
    "CandidateTriageStateORM",
    "CuratedDiscoveryORM",
    "MPCSubmissionItemORM",
    "MPCSubmissionORM",
    "OLTPBase",
    "OLTPDatabaseConfig",
    "OperatorORM",
    "OrbitalSolutionORM",
    "PipelineJobORM",
    "PipelineTaskLogORM",
    "PipelineTaskORM",
    "TelescopeProfileORM",
    "UTCDateTime",
    "configure_sqlite_engine",
    "create_oltp_engine",
    "initialize_oltp_schema",
]

_SAFE_ID_REGEX: str = r"^[A-Za-z0-9_.-]+$"
_MPC_CODE_REGEX: str = r"^[A-Za-z0-9]{3}$"


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class OLTPDatabaseConfig(BaseModel):
    """Strict ADR-0002 configuration contract for SQLAlchemy 2.x OLTP engines."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    url: str = Field(
        default="sqlite+pysqlite:///:memory:",
        min_length=1,
        description=(
            "SQLAlchemy 2.x database URL (postgresql+psycopg://... or sqlite+pysqlite://...)."
        ),
    )
    echo: bool = Field(
        default=False,
        description="Whether to emit generated SQL statements to the SQLAlchemy logger.",
    )
    pool_pre_ping: bool = Field(
        default=True,
        description="Verify pooled connections with a lightweight ping before checkout.",
    )


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-preserving UTC timestamp compatible with PostgreSQL TIMESTAMPTZ and SQLite."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: sa.Dialect,
    ) -> datetime | None:
        if value is None:
            return None
        utc_val = (
            value.replace(tzinfo=UTC)
            if value.tzinfo is None
            else value.astimezone(UTC)
        )
        if dialect.name == "sqlite":
            return utc_val.replace(tzinfo=None)
        return utc_val

    def process_result_value(
        self,
        value: datetime | None,
        dialect: sa.Dialect,
    ) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class _SafeAsciiIdentifierMatch(ColumnElement[bool]):
    """Cross-dialect CHECK predicate enforcing ``^[A-Za-z0-9_.-]+$``."""

    inherit_cache = True

    def __init__(self, column_expr: ColumnElement[Any]) -> None:
        super().__init__()
        self.column_expr: ColumnElement[Any] = column_expr


@compiles(_SafeAsciiIdentifierMatch, "postgresql")
def _compile_safe_id_pg(
    element: _SafeAsciiIdentifierMatch,
    compiler: Any,
    **kw: Any,
) -> str:
    col_sql: str = compiler.process(element.column_expr, **kw)
    return f"{col_sql} ~ '{_SAFE_ID_REGEX}'"


@compiles(_SafeAsciiIdentifierMatch)
def _compile_safe_id_default(
    element: _SafeAsciiIdentifierMatch,
    compiler: Any,
    **kw: Any,
) -> str:
    col_sql: str = compiler.process(element.column_expr, **kw)
    return f"(length({col_sql}) >= 1 AND {col_sql} NOT GLOB '*[^A-Za-z0-9_.-]*')"


class _MPCCodeMatch(ColumnElement[bool]):
    """Cross-dialect CHECK predicate enforcing nullable 3-char ``^[A-Za-z0-9]{3}$``."""

    inherit_cache = True

    def __init__(self, column_expr: ColumnElement[Any]) -> None:
        super().__init__()
        self.column_expr: ColumnElement[Any] = column_expr


@compiles(_MPCCodeMatch, "postgresql")
def _compile_mpc_code_pg(
    element: _MPCCodeMatch,
    compiler: Any,
    **kw: Any,
) -> str:
    col_sql: str = compiler.process(element.column_expr, **kw)
    return f"{col_sql} ~ '{_MPC_CODE_REGEX}'"


@compiles(_MPCCodeMatch)
def _compile_mpc_code_default(
    element: _MPCCodeMatch,
    compiler: Any,
    **kw: Any,
) -> str:
    col_sql: str = compiler.process(element.column_expr, **kw)
    return (
        f"({col_sql} IS NULL OR "
        f"(length({col_sql}) = 3 AND {col_sql} NOT GLOB '*[^A-Za-z0-9]*'))"
    )


_BIGSERIAL_PK_TYPE = BigInteger().with_variant(Integer, "sqlite")
_CLIENT_IP_TYPE = String(45).with_variant(INET(), "postgresql")


class OLTPBase(DeclarativeBase):
    """SQLAlchemy 2.x DeclarativeBase for all 3NF PostgreSQL OLTP tables."""


# ---------------------------------------------------------------------------
# A. Operator & JWT Authentication Tables
# ---------------------------------------------------------------------------


class OperatorORM(OLTPBase):
    """Authenticated human astronomer, reviewer, administrator, or service account."""

    __tablename__ = "operators"

    operator_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=sa.true(),
    )
    created_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )
    updated_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
        server_default=func.current_timestamp(),
    )

    refresh_sessions: Mapped[list[AuthRefreshSessionORM]] = relationship(
        back_populates="operator",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    submitted_jobs: Mapped[list[PipelineJobORM]] = relationship(
        back_populates="submitted_by_operator",
        passive_deletes=True,
    )
    reviewed_candidates: Mapped[list[CandidateTriageStateORM]] = relationship(
        back_populates="reviewed_by_operator",
        passive_deletes=True,
    )
    triage_events: Mapped[list[CandidateTriageEventORM]] = relationship(
        back_populates="operator",
        passive_deletes=True,
    )
    confirmed_discoveries: Mapped[list[CuratedDiscoveryORM]] = relationship(
        back_populates="confirmed_by_operator",
        passive_deletes=True,
    )
    mpc_submissions: Mapped[list[MPCSubmissionORM]] = relationship(
        back_populates="submitted_by_operator",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('admin', 'astronomer', 'reviewer', 'service')",
            name="ck_operators_role",
        ),
    )


class AuthRefreshSessionORM(OLTPBase):
    """Revocable JWT refresh session bound to an OperatorORM account."""

    __tablename__ = "auth_refresh_sessions"

    session_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    operator_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("operators.operator_id", ondelete="CASCADE"),
        nullable=False,
    )
    token_jti_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False, unique=True)
    issued_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )
    expires_at_utc: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    revoked_at_utc: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )
    client_ip: Mapped[str | None] = mapped_column(
        _CLIENT_IP_TYPE,
        nullable=True,
        default=None,
    )
    user_agent: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        default=None,
    )

    operator: Mapped[OperatorORM] = relationship(back_populates="refresh_sessions")

    __table_args__ = (
        CheckConstraint(
            "expires_at_utc > issued_at_utc",
            name="ck_auth_refresh_sessions_expiry_after_issue",
        ),
        Index(
            "idx_auth_sessions_operator_active",
            "operator_id",
            "expires_at_utc",
            postgresql_where=sa.text("revoked_at_utc IS NULL"),
            sqlite_where=sa.text("revoked_at_utc IS NULL"),
        ),
    )


# ---------------------------------------------------------------------------
# B. Telescope Profiles & Pipeline Job Lifecycle Tables
# ---------------------------------------------------------------------------


class TelescopeProfileORM(OLTPBase):
    """Normalized 3NF optical, detector, and geodetic site profile for an instrument."""

    __tablename__ = "telescope_profiles"

    profile_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    mpc_code: Mapped[str | None] = mapped_column(CHAR(3), nullable=True, default=None)
    pixel_scale_arcsec: Mapped[float] = mapped_column(Double, nullable=False)
    gain_e_per_adu: Mapped[float] = mapped_column(Double, nullable=False)
    read_noise_e: Mapped[float] = mapped_column(Double, nullable=False)
    saturation_limit_adu: Mapped[float] = mapped_column(Double, nullable=False)
    dark_current_e_per_s: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        default=0.0,
        server_default=sa.text("0.0"),
    )
    latitude_deg: Mapped[float] = mapped_column(Double, nullable=False)
    longitude_deg: Mapped[float] = mapped_column(Double, nullable=False)
    elevation_m: Mapped[float] = mapped_column(Double, nullable=False)
    aperture_diameter_m: Mapped[float | None] = mapped_column(
        Double,
        nullable=True,
        default=None,
    )
    focal_length_m: Mapped[float | None] = mapped_column(
        Double,
        nullable=True,
        default=None,
    )
    created_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )

    jobs: Mapped[list[PipelineJobORM]] = relationship(
        back_populates="telescope_profile",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            _SafeAsciiIdentifierMatch(sa.column("profile_id")),
            name="ck_telescope_profiles_profile_id_format",
        ),
        CheckConstraint(
            _MPCCodeMatch(sa.column("mpc_code")),
            name="ck_telescope_profiles_mpc_code_format",
        ),
        CheckConstraint(
            "pixel_scale_arcsec > 0.0",
            name="ck_telescope_profiles_pixel_scale_pos",
        ),
        CheckConstraint(
            "gain_e_per_adu > 0.0",
            name="ck_telescope_profiles_gain_pos",
        ),
        CheckConstraint(
            "read_noise_e >= 0.0",
            name="ck_telescope_profiles_read_noise_nonneg",
        ),
        CheckConstraint(
            "saturation_limit_adu > 0.0",
            name="ck_telescope_profiles_saturation_pos",
        ),
        CheckConstraint(
            "dark_current_e_per_s >= 0.0",
            name="ck_telescope_profiles_dark_current_nonneg",
        ),
        CheckConstraint(
            "latitude_deg BETWEEN -90.0 AND 90.0",
            name="ck_telescope_profiles_latitude_bounds",
        ),
        CheckConstraint(
            "longitude_deg BETWEEN -180.0 AND 180.0",
            name="ck_telescope_profiles_longitude_bounds",
        ),
        CheckConstraint(
            "elevation_m BETWEEN -500.0 AND 10000.0",
            name="ck_telescope_profiles_elevation_bounds",
        ),
        CheckConstraint(
            "aperture_diameter_m > 0.0",
            name="ck_telescope_profiles_aperture_pos",
        ),
        CheckConstraint(
            "focal_length_m > 0.0",
            name="ck_telescope_profiles_focal_length_pos",
        ),
    )


class PipelineJobORM(OLTPBase):
    """Transactional Prefect/API pipeline job lifecycle and sky-box record."""

    __tablename__ = "pipeline_jobs"

    job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    submitted_by_operator_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("operators.operator_id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )
    profile_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("telescope_profiles.profile_id", ondelete="RESTRICT"),
        nullable=True,
        default=None,
    )
    target_field_id: Mapped[str] = mapped_column(String(128), nullable=False)
    observation_night_utc: Mapped[date] = mapped_column(Date, nullable=False)
    ra_min_deg: Mapped[float] = mapped_column(Double, nullable=False)
    ra_max_deg: Mapped[float] = mapped_column(Double, nullable=False)
    dec_min_deg: Mapped[float] = mapped_column(Double, nullable=False)
    dec_max_deg: Mapped[float] = mapped_column(Double, nullable=False)
    max_mag: Mapped[float] = mapped_column(Double, nullable=False)
    prefect_flow_run_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        unique=True,
        default=None,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        server_default=sa.text("'queued'"),
    )
    matched_catalog_sources: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )
    brightest_catalog_mag_g: Mapped[float | None] = mapped_column(
        Double,
        nullable=True,
        default=None,
    )
    parquet_partition_uri: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        default=None,
    )
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    submitted_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )
    started_at_utc: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )
    completed_at_utc: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )

    submitted_by_operator: Mapped[OperatorORM | None] = relationship(
        back_populates="submitted_jobs"
    )
    telescope_profile: Mapped[TelescopeProfileORM | None] = relationship(
        back_populates="jobs"
    )
    tasks: Mapped[list[PipelineTaskORM]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    triage_states: Mapped[list[CandidateTriageStateORM]] = relationship(
        back_populates="job",
        passive_deletes=True,
    )
    discoveries: Mapped[list[CuratedDiscoveryORM]] = relationship(
        back_populates="discovery_job",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            _SafeAsciiIdentifierMatch(sa.column("job_id")),
            name="ck_pipeline_jobs_job_id_format",
        ),
        CheckConstraint(
            _SafeAsciiIdentifierMatch(sa.column("target_field_id")),
            name="ck_pipeline_jobs_target_field_id_format",
        ),
        CheckConstraint(
            "ra_min_deg >= 0.0 AND ra_min_deg < 360.0",
            name="ck_pipeline_jobs_ra_min_bounds",
        ),
        CheckConstraint(
            "ra_max_deg >= 0.0 AND ra_max_deg <= 360.0",
            name="ck_pipeline_jobs_ra_max_bounds",
        ),
        CheckConstraint(
            "dec_min_deg BETWEEN -90.0 AND 90.0",
            name="ck_pipeline_jobs_dec_min_bounds",
        ),
        CheckConstraint(
            "dec_max_deg BETWEEN -90.0 AND 90.0",
            name="ck_pipeline_jobs_dec_max_bounds",
        ),
        CheckConstraint(
            "dec_min_deg <= dec_max_deg",
            name="ck_pipeline_jobs_dec_order",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_pipeline_jobs_status",
        ),
        CheckConstraint(
            "matched_catalog_sources >= 0",
            name="ck_pipeline_jobs_matched_sources_nonneg",
        ),
        Index(
            "idx_pipeline_jobs_status_submitted",
            "status",
            sa.desc("submitted_at_utc"),
        ),
        Index(
            "idx_pipeline_jobs_night_field",
            "observation_night_utc",
            "target_field_id",
        ),
    )


class PipelineTaskORM(OLTPBase):
    """Individual Prefect task execution state and retry metadata within a pipeline job."""

    __tablename__ = "pipeline_tasks"

    task_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    job_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("pipeline_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    task_name: Mapped[str] = mapped_column(String(128), nullable=False)
    module_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    prefect_task_run_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        unique=True,
        default=None,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default=sa.text("'pending'"),
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=sa.text("1"),
    )
    max_retries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        server_default=sa.text("3"),
    )
    backoff_factor_s: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        default=1.0,
        server_default=sa.text("1.0"),
    )
    next_retry_at_utc: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )
    started_at_utc: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )
    finished_at_utc: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )

    job: Mapped[PipelineJobORM] = relationship(back_populates="tasks")
    logs: Mapped[list[PipelineTaskLogORM]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "module_number BETWEEN 1 AND 10",
            name="ck_pipeline_tasks_module_number",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'retrying', 'completed', 'failed', 'skipped')",
            name="ck_pipeline_tasks_status",
        ),
        CheckConstraint(
            "attempt_count >= 1",
            name="ck_pipeline_tasks_attempt_count_min",
        ),
        CheckConstraint(
            "max_retries >= 0",
            name="ck_pipeline_tasks_max_retries_nonneg",
        ),
        CheckConstraint(
            "backoff_factor_s >= 0.0",
            name="ck_pipeline_tasks_backoff_nonneg",
        ),
        Index("idx_pipeline_tasks_job_status", "job_id", "status"),
    )


class PipelineTaskLogORM(OLTPBase):
    """Structured diagnostic event log emitted during a PipelineTaskORM attempt."""

    __tablename__ = "pipeline_task_logs"

    log_id: Mapped[int] = mapped_column(
        _BIGSERIAL_PK_TYPE,
        primary_key=True,
        autoincrement=True,
    )
    task_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("pipeline_tasks.task_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    event_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default=None,
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    logged_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )

    task: Mapped[PipelineTaskORM] = relationship(back_populates="logs")

    __table_args__ = (
        CheckConstraint(
            "attempt_number >= 1",
            name="ck_pipeline_task_logs_attempt_min",
        ),
        CheckConstraint(
            "severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')",
            name="ck_pipeline_task_logs_severity",
        ),
        Index("idx_task_logs_run_logged", "task_run_id", sa.asc("logged_at_utc")),
    )


# ---------------------------------------------------------------------------
# C. Human Active Learning Triage Disposition Tables
# ---------------------------------------------------------------------------


class CandidateTriageStateORM(OLTPBase):
    """Mutable current human triage disposition and HDF5 export state for a candidate."""

    __tablename__ = "candidate_triage_state"

    candidate_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("pipeline_jobs.job_id", ondelete="RESTRICT"),
        nullable=False,
    )
    field_id: Mapped[str] = mapped_column(String(128), nullable=False)
    disposition: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="unreviewed",
        server_default=sa.text("'unreviewed'"),
    )
    reviewed_by_operator_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("operators.operator_id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )
    cnn_score_at_review: Mapped[float | None] = mapped_column(
        Double,
        nullable=True,
        default=None,
    )
    model_version_at_review: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        default=None,
    )
    hdf5_export_batch_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        default=None,
    )
    updated_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
        server_default=func.current_timestamp(),
    )

    job: Mapped[PipelineJobORM] = relationship(back_populates="triage_states")
    reviewed_by_operator: Mapped[OperatorORM | None] = relationship(
        back_populates="reviewed_candidates"
    )
    events: Mapped[list[CandidateTriageEventORM]] = relationship(
        back_populates="triage_state",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            _SafeAsciiIdentifierMatch(sa.column("candidate_id")),
            name="ck_candidate_triage_state_candidate_id_format",
        ),
        CheckConstraint(
            _SafeAsciiIdentifierMatch(sa.column("field_id")),
            name="ck_candidate_triage_state_field_id_format",
        ),
        CheckConstraint(
            "disposition IN ('unreviewed', 'confirmed_real', 'follow_up', 'flagged_bogus')",
            name="ck_candidate_triage_state_disposition",
        ),
        CheckConstraint(
            "cnn_score_at_review BETWEEN 0.0 AND 1.0",
            name="ck_candidate_triage_state_cnn_score_bounds",
        ),
        Index(
            "idx_triage_state_disposition_unexported",
            "disposition",
            "updated_at_utc",
            postgresql_where=sa.text(
                "hdf5_export_batch_id IS NULL "
                "AND disposition IN ('confirmed_real', 'flagged_bogus')"
            ),
            sqlite_where=sa.text(
                "hdf5_export_batch_id IS NULL "
                "AND disposition IN ('confirmed_real', 'flagged_bogus')"
            ),
        ),
    )


class CandidateTriageEventORM(OLTPBase):
    """Immutable audit log of human Active Learning disposition transitions."""

    __tablename__ = "candidate_triage_events"

    triage_event_id: Mapped[int] = mapped_column(
        _BIGSERIAL_PK_TYPE,
        primary_key=True,
        autoincrement=True,
    )
    candidate_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("candidate_triage_state.candidate_id", ondelete="CASCADE"),
        nullable=False,
    )
    operator_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("operators.operator_id", ondelete="RESTRICT"),
        nullable=False,
    )
    previous_disposition: Mapped[str] = mapped_column(String(32), nullable=False)
    new_disposition: Mapped[str] = mapped_column(String(32), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    decided_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )

    triage_state: Mapped[CandidateTriageStateORM] = relationship(
        back_populates="events"
    )
    operator: Mapped[OperatorORM] = relationship(back_populates="triage_events")

    __table_args__ = (
        CheckConstraint(
            "previous_disposition IN "
            "('unreviewed', 'confirmed_real', 'follow_up', 'flagged_bogus')",
            name="ck_candidate_triage_events_prev_disposition",
        ),
        CheckConstraint(
            "new_disposition IN ('unreviewed', 'confirmed_real', 'follow_up', 'flagged_bogus')",
            name="ck_candidate_triage_events_new_disposition",
        ),
        Index(
            "idx_triage_events_candidate_time",
            "candidate_id",
            sa.desc("decided_at_utc"),
        ),
    )


# ---------------------------------------------------------------------------
# D. Curated Asteroid Discoveries, Orbital Elements & MPC Submissions
# ---------------------------------------------------------------------------


class CuratedDiscoveryORM(OLTPBase):
    """Long-term verified moving-object discovery linked to an OLAP Tracklet."""

    __tablename__ = "curated_discoveries"

    discovery_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    tracklet_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    internal_designation: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )
    mpc_provisional_desig: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        unique=True,
        default=None,
    )
    discovery_job_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("pipeline_jobs.job_id", ondelete="RESTRICT"),
        nullable=False,
    )
    confirmed_by_operator_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("operators.operator_id", ondelete="RESTRICT"),
        nullable=False,
    )
    dynamical_class: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="Unknown",
        server_default=sa.text("'Unknown'"),
    )
    first_obs_bjd_tdb: Mapped[float] = mapped_column(Double, nullable=False)
    last_obs_bjd_tdb: Mapped[float] = mapped_column(Double, nullable=False)
    discovered_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )

    discovery_job: Mapped[PipelineJobORM] = relationship(back_populates="discoveries")
    confirmed_by_operator: Mapped[OperatorORM] = relationship(
        back_populates="confirmed_discoveries"
    )
    orbital_solutions: Mapped[list[OrbitalSolutionORM]] = relationship(
        back_populates="discovery",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    submission_items: Mapped[list[MPCSubmissionItemORM]] = relationship(
        back_populates="discovery",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            _SafeAsciiIdentifierMatch(sa.column("tracklet_id")),
            name="ck_curated_discoveries_tracklet_id_format",
        ),
        CheckConstraint(
            "dynamical_class IN ('NEA', 'MBA', 'Trojan', 'Centaur', 'TNO', 'Comet', 'Unknown')",
            name="ck_curated_discoveries_dynamical_class",
        ),
        CheckConstraint(
            "last_obs_bjd_tdb >= first_obs_bjd_tdb",
            name="ck_curated_discoveries_bjd_order",
        ),
    )


class OrbitalSolutionORM(OLTPBase):
    """Keplerian heliocentric orbital solution and physical parameters for a discovery."""

    __tablename__ = "orbital_solutions"

    orbit_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    discovery_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("curated_discoveries.discovery_id", ondelete="CASCADE"),
        nullable=False,
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=sa.true(),
    )
    epoch_tdb_jd: Mapped[float] = mapped_column(Double, nullable=False)
    semi_major_axis_au: Mapped[float] = mapped_column(Double, nullable=False)
    eccentricity: Mapped[float] = mapped_column(Double, nullable=False)
    inclination_deg: Mapped[float] = mapped_column(Double, nullable=False)
    lon_asc_node_deg: Mapped[float] = mapped_column(Double, nullable=False)
    arg_perihelion_deg: Mapped[float] = mapped_column(Double, nullable=False)
    mean_anomaly_deg: Mapped[float] = mapped_column(Double, nullable=False)
    earth_moid_au: Mapped[float | None] = mapped_column(
        Double,
        nullable=True,
        default=None,
    )
    abs_mag_h: Mapped[float | None] = mapped_column(Double, nullable=True, default=None)
    albedo_pv: Mapped[float | None] = mapped_column(Double, nullable=True, default=None)
    rms_residual_arcsec: Mapped[float] = mapped_column(Double, nullable=False)
    n_observations_used: Mapped[int] = mapped_column(Integer, nullable=False)
    orbit_computer: Mapped[str] = mapped_column(String(64), nullable=False)
    computed_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )

    discovery: Mapped[CuratedDiscoveryORM] = relationship(
        back_populates="orbital_solutions"
    )

    __table_args__ = (
        CheckConstraint(
            "semi_major_axis_au > 0.0",
            name="ck_orbital_solutions_sma_pos",
        ),
        CheckConstraint(
            "eccentricity >= 0.0",
            name="ck_orbital_solutions_ecc_nonneg",
        ),
        CheckConstraint(
            "inclination_deg BETWEEN 0.0 AND 180.0",
            name="ck_orbital_solutions_inclination_bounds",
        ),
        CheckConstraint(
            "lon_asc_node_deg >= 0.0 AND lon_asc_node_deg < 360.0",
            name="ck_orbital_solutions_node_bounds",
        ),
        CheckConstraint(
            "arg_perihelion_deg >= 0.0 AND arg_perihelion_deg < 360.0",
            name="ck_orbital_solutions_argperi_bounds",
        ),
        CheckConstraint(
            "mean_anomaly_deg >= 0.0 AND mean_anomaly_deg < 360.0",
            name="ck_orbital_solutions_mean_anom_bounds",
        ),
        CheckConstraint(
            "earth_moid_au >= 0.0",
            name="ck_orbital_solutions_moid_nonneg",
        ),
        CheckConstraint(
            "albedo_pv BETWEEN 0.0 AND 1.0",
            name="ck_orbital_solutions_albedo_bounds",
        ),
        CheckConstraint(
            "rms_residual_arcsec >= 0.0",
            name="ck_orbital_solutions_rms_nonneg",
        ),
        CheckConstraint(
            "n_observations_used >= 3",
            name="ck_orbital_solutions_min_obs",
        ),
        Index(
            "uq_orbital_solutions_current",
            "discovery_id",
            unique=True,
            postgresql_where=sa.column("is_current", Boolean()).is_(True),
            sqlite_where=sa.column("is_current", Boolean()).is_(True),
        ),
    )


class MPCSubmissionORM(OLTPBase):
    """Minor Planet Center (MPC) ADES/OBS80 submission batch metadata."""

    __tablename__ = "mpc_submissions"

    submission_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    submitted_by_operator_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("operators.operator_id", ondelete="RESTRICT"),
        nullable=False,
    )
    submission_format: Mapped[str] = mapped_column(String(16), nullable=False)
    mpc_ack_reference: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        unique=True,
        default=None,
    )
    mpec_citation: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default=None,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="draft",
        server_default=sa.text("'draft'"),
    )
    payload_sha256: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=_utc_now,
        server_default=func.current_timestamp(),
    )
    transmitted_at_utc: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )
    acknowledged_at_utc: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )

    submitted_by_operator: Mapped[OperatorORM] = relationship(
        back_populates="mpc_submissions"
    )
    items: Mapped[list[MPCSubmissionItemORM]] = relationship(
        back_populates="submission",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "submission_format IN ('ADES_PSV', 'ADES_XML', 'OBS80')",
            name="ck_mpc_submissions_format",
        ),
        CheckConstraint(
            "status IN ('draft', 'transmitted', 'acknowledged', 'published_mpec', 'rejected')",
            name="ck_mpc_submissions_status",
        ),
    )


class MPCSubmissionItemORM(OLTPBase):
    """2NF composite-key junction item associating a discovery with an MPC submission batch."""

    __tablename__ = "mpc_submission_items"

    submission_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("mpc_submissions.submission_id", ondelete="CASCADE"),
        primary_key=True,
    )
    discovery_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("curated_discoveries.discovery_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    tracklet_id_submitted: Mapped[str] = mapped_column(String(128), nullable=False)
    n_astrometric_measurements: Mapped[int] = mapped_column(Integer, nullable=False)

    submission: Mapped[MPCSubmissionORM] = relationship(back_populates="items")
    discovery: Mapped[CuratedDiscoveryORM] = relationship(
        back_populates="submission_items"
    )

    __table_args__ = (
        CheckConstraint(
            "n_astrometric_measurements >= 1",
            name="ck_mpc_submission_items_min_measurements",
        ),
    )


# ---------------------------------------------------------------------------
# Engine Configuration & Schema Lifecycle Helpers
# ---------------------------------------------------------------------------


def configure_sqlite_engine(engine: Engine) -> Engine:
    """Attach a connection listener enabling ``PRAGMA foreign_keys=ON`` for SQLite engines."""
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            del connection_record
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

    return engine


def create_oltp_engine(config: OLTPDatabaseConfig | None = None) -> Engine:
    """Create a configured SQLAlchemy 2.x Engine for PostgreSQL or isolated SQLite testing."""
    resolved_cfg = config if config is not None else OLTPDatabaseConfig()
    options: dict[str, Any] = {
        "echo": resolved_cfg.echo,
        "pool_pre_ping": resolved_cfg.pool_pre_ping,
    }
    if resolved_cfg.url.endswith(":memory:"):
        options["connect_args"] = {"check_same_thread": False}
        options["poolclass"] = StaticPool
    engine = sa.create_engine(
        resolved_cfg.url,
        **options,
    )
    return configure_sqlite_engine(engine)


def initialize_oltp_schema(engine: Engine) -> None:
    """Materialize all 12 3NF OLTP tables and indexes on ``engine`` if absent."""
    OLTPBase.metadata.create_all(engine)
