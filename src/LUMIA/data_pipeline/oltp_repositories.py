"""Module 2 repositories for transactional authentication, jobs, and candidate triage."""

from __future__ import annotations

from datetime import UTC, date, datetime
from hashlib import sha256
from typing import Literal, cast
from uuid import UUID  # noqa: TC003

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from pydantic import Field
from sqlalchemy import Engine, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from LUMIA.core.schemas import LumiaBaseModel
from LUMIA.data_pipeline.schema_oltp import (
    AuthRefreshSessionORM,
    CandidateTriageEventORM,
    CandidateTriageStateORM,
    OLTPDatabaseConfig,
    OperatorORM,
    PipelineJobORM,
    create_oltp_engine,
    initialize_oltp_schema,
)

OperatorRole = Literal["admin", "astronomer", "reviewer", "service"]
JobLifecycle = Literal["queued", "running", "completed", "failed", "cancelled"]
TriageDisposition = Literal["unreviewed", "confirmed_real", "follow_up", "flagged_bogus"]

_PASSWORD_HASHER = PasswordHasher()


class RepositoryConflictError(ValueError):
    """A requested transactional state change violates a persistence constraint."""



class OperatorProfile(LumiaBaseModel):
    operator_id: UUID
    username: str = Field(min_length=1, max_length=64)
    email: str = Field(min_length=1, max_length=255)
    role: OperatorRole
    is_active: bool


class PipelineJobRecord(LumiaBaseModel):
    job_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    target_field_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]{1,128}$"
    )
    observation_night_utc: date
    status: JobLifecycle
    matched_catalog_sources: int | None = Field(default=None, ge=0)
    brightest_catalog_mag_g: float | None = None
    submitted_at_utc: datetime
    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None
    profile_id: str | None = None


class TriageRecord(LumiaBaseModel):
    triage_event_id: int = Field(ge=1)
    candidate_id: str
    job_id: str
    field_id: str
    previous_disposition: TriageDisposition
    disposition: TriageDisposition
    reviewed_by_operator_id: UUID
    cnn_score_at_review: float | None = None
    model_version_at_review: str | None = None
    queued_for_hdf5_export: bool
    decided_at_utc: datetime


class CandidateTriageStateRecord(LumiaBaseModel):
    candidate_id: str
    job_id: str
    field_id: str
    disposition: TriageDisposition
    reviewed_by_operator_id: UUID | None = None
    updated_at_utc: datetime


def _profile(operator: OperatorORM) -> OperatorProfile:
    return OperatorProfile(
        operator_id=operator.operator_id,
        username=operator.username,
        email=operator.email,
        role=cast("OperatorRole", operator.role),
        is_active=operator.is_active,
    )


def _job(job: PipelineJobORM) -> PipelineJobRecord:
    return PipelineJobRecord(
        job_id=job.job_id,
        target_field_id=job.target_field_id,
        observation_night_utc=job.observation_night_utc,
        status=cast("JobLifecycle", job.status),
        matched_catalog_sources=job.matched_catalog_sources,
        brightest_catalog_mag_g=job.brightest_catalog_mag_g,
        submitted_at_utc=job.submitted_at_utc,
        started_at_utc=job.started_at_utc,
        completed_at_utc=job.completed_at_utc,
        profile_id=job.profile_id,
    )


def refresh_token_hash(jti: str) -> str:
    """Return the persisted SHA-256 digest for a refresh-token jti."""
    return sha256(jti.encode("utf-8")).hexdigest()


class OperatorAuthRepository:
    """Persist operator accounts and atomically rotate/revoke refresh sessions."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def register_operator(
        self,
        *,
        username: str,
        email: str,
        password: str,
        role: OperatorRole,
        is_active: bool = True,
    ) -> OperatorProfile:
        password_hash = _PASSWORD_HASHER.hash(password)
        with Session(self._engine) as session, session.begin():
            operator = OperatorORM(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                is_active=is_active,
            )
            session.add(operator)
            session.flush()
            return _profile(operator)

    def verify_credentials(self, username: str, password: str) -> OperatorProfile | None:
        with Session(self._engine) as session:
            operator = session.scalar(select(OperatorORM).where(OperatorORM.username == username))
            if operator is None or not operator.is_active:
                return None
            try:
                valid = _PASSWORD_HASHER.verify(operator.password_hash, password)
            except (InvalidHashError, VerifyMismatchError):
                return None
            return _profile(operator) if valid else None

    def get_active_operator(self, operator_id: UUID) -> OperatorProfile | None:
        with Session(self._engine) as session:
            operator = session.get(OperatorORM, operator_id)
            if operator is None or not operator.is_active:
                return None
            return _profile(operator)

    def create_refresh_session(
        self,
        operator_id: UUID,
        token_jti_hash: str,
        expires_at_utc: datetime,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> bool:
        with Session(self._engine) as session, session.begin():
            if session.get(OperatorORM, operator_id) is None:
                return False
            session.add(
                AuthRefreshSessionORM(
                    operator_id=operator_id,
                    token_jti_hash=token_jti_hash,
                    expires_at_utc=expires_at_utc,
                    client_ip=client_ip,
                    user_agent=user_agent,
                )
            )
        return True

    def rotate_refresh_session(
        self,
        old_jti_hash: str,
        new_jti_hash: str,
        expires_at_utc: datetime,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> OperatorProfile | None:
        now = datetime.now(UTC)
        with Session(self._engine) as session, session.begin():
            old = session.scalar(
                select(AuthRefreshSessionORM)
                .where(AuthRefreshSessionORM.token_jti_hash == old_jti_hash)
                .with_for_update()
            )
            if old is None:
                return None
            if old.revoked_at_utc is not None:
                session.execute(
                    update(AuthRefreshSessionORM)
                    .where(
                        AuthRefreshSessionORM.operator_id == old.operator_id,
                        AuthRefreshSessionORM.revoked_at_utc.is_(None),
                    )
                    .values(revoked_at_utc=now)
                )
                return None
            if old.expires_at_utc <= now:
                return None
            operator = session.get(OperatorORM, old.operator_id)
            if operator is None or not operator.is_active:
                old.revoked_at_utc = now
                return None
            old.revoked_at_utc = now
            session.add(
                AuthRefreshSessionORM(
                    operator_id=old.operator_id,
                    token_jti_hash=new_jti_hash,
                    expires_at_utc=expires_at_utc,
                    client_ip=client_ip,
                    user_agent=user_agent,
                )
            )
            session.flush()
            return _profile(operator)

    def revoke_refresh_session(self, token_jti_hash: str) -> bool:
        now = datetime.now(UTC)
        with Session(self._engine) as session, session.begin():
            row = session.scalar(
                select(AuthRefreshSessionORM).where(
                    AuthRefreshSessionORM.token_jti_hash == token_jti_hash
                )
            )
            if row is None:
                return False
            if row.revoked_at_utc is None:
                row.revoked_at_utc = now
            return True


class PipelineJobRepository:
    """ACID job lifecycle repository backed by Module 2's OLTP schema."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create_job(
        self,
        *,
        job_id: str,
        operator_id: UUID,
        target_field_id: str,
        observation_night_utc: date,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
        max_mag: float,
        profile_id: str | None = None,
        matched_catalog_sources: int | None = None,
        brightest_catalog_mag_g: float | None = None,
    ) -> PipelineJobRecord:
        try:
            with Session(self._engine) as session, session.begin():
                entity = PipelineJobORM(
                    job_id=job_id,
                    submitted_by_operator_id=operator_id,
                    target_field_id=target_field_id,
                    observation_night_utc=observation_night_utc,
                    ra_min_deg=ra_min_deg,
                    ra_max_deg=ra_max_deg,
                    dec_min_deg=dec_min_deg,
                    dec_max_deg=dec_max_deg,
                    max_mag=max_mag,
                    profile_id=profile_id,
                    matched_catalog_sources=matched_catalog_sources,
                    brightest_catalog_mag_g=brightest_catalog_mag_g,
                    status="queued",
                )
                session.add(entity)
                session.flush()
                return _job(entity)
        except IntegrityError as exc:
            raise RepositoryConflictError("Job could not be created.") from exc

    def get_job(self, job_id: str) -> PipelineJobRecord | None:
        with Session(self._engine) as session:
            entity = session.get(PipelineJobORM, job_id)
            return _job(entity) if entity is not None else None

    def list_jobs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status_filter: str | None = None,
        observation_night_utc: date | None = None,
    ) -> list[PipelineJobRecord]:
        statement = select(PipelineJobORM).order_by(PipelineJobORM.submitted_at_utc.desc())
        if status_filter is not None:
            statement = statement.where(PipelineJobORM.status == status_filter)
        if observation_night_utc is not None:
            statement = statement.where(
                PipelineJobORM.observation_night_utc == observation_night_utc
            )
        statement = statement.limit(limit).offset(offset)
        with Session(self._engine) as session:
            return [_job(entity) for entity in session.scalars(statement)]

    def count_jobs(self) -> int:
        with Session(self._engine) as session:
            return len(session.scalars(select(PipelineJobORM.job_id)).all())

    def cancel_job(self, job_id: str) -> PipelineJobRecord | None:
        with Session(self._engine) as session, session.begin():
            entity = session.scalar(
                select(PipelineJobORM).where(PipelineJobORM.job_id == job_id).with_for_update()
            )
            if entity is None:
                return None
            if entity.status in {"queued", "running"}:
                entity.status = "cancelled"
                entity.completed_at_utc = datetime.now(UTC)
            return _job(entity)

    def active_jobs(self, job_id: str | None = None) -> list[PipelineJobRecord]:
        statement = select(PipelineJobORM).where(
            PipelineJobORM.status.in_(("queued", "running"))
        )
        if job_id is not None:
            statement = statement.where(PipelineJobORM.job_id == job_id)
        statement = statement.order_by(PipelineJobORM.submitted_at_utc.asc())
        with Session(self._engine) as session:
            return [_job(entity) for entity in session.scalars(statement)]


class CandidateTriageRepository:
    """Atomic current-disposition and immutable audit-event repository."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record_disposition(
        self,
        *,
        candidate_id: str,
        job_id: str,
        field_id: str,
        operator_id: UUID,
        disposition: TriageDisposition,
        notes: str | None,
        cnn_score: float | None,
        model_version: str | None,
    ) -> TriageRecord:
        now = datetime.now(UTC)
        try:
            with Session(self._engine) as session, session.begin():
                state = session.scalar(
                    select(CandidateTriageStateORM)
                    .where(CandidateTriageStateORM.candidate_id == candidate_id)
                    .with_for_update()
                )
                previous: TriageDisposition = (
                    cast("TriageDisposition", state.disposition)
                    if state is not None
                    else "unreviewed"
                )
                if state is None:
                    state = CandidateTriageStateORM(
                        candidate_id=candidate_id,
                        job_id=job_id,
                        field_id=field_id,
                        disposition=disposition,
                        reviewed_by_operator_id=operator_id,
                        cnn_score_at_review=cnn_score,
                        model_version_at_review=model_version,
                        hdf5_export_batch_id=None,
                        updated_at_utc=now,
                    )
                    session.add(state)
                else:
                    state.disposition = disposition
                    state.reviewed_by_operator_id = operator_id
                    state.cnn_score_at_review = cnn_score
                    state.model_version_at_review = model_version
                    if previous != disposition:
                        state.hdf5_export_batch_id = None
                    state.updated_at_utc = now
                event = CandidateTriageEventORM(
                    candidate_id=candidate_id,
                    operator_id=operator_id,
                    previous_disposition=previous,
                    new_disposition=disposition,
                    notes=notes,
                    decided_at_utc=now,
                )
                session.add(event)
                session.flush()
                return TriageRecord(
                    triage_event_id=event.triage_event_id,
                    candidate_id=candidate_id,
                    job_id=job_id,
                    field_id=field_id,
                    previous_disposition=previous,
                    disposition=disposition,
                    reviewed_by_operator_id=operator_id,
                    cnn_score_at_review=cnn_score,
                    model_version_at_review=model_version,
                    queued_for_hdf5_export=disposition in {"confirmed_real", "flagged_bogus"},
                    decided_at_utc=now,
                )
        except IntegrityError as exc:
            raise RepositoryConflictError("Triage transaction conflicted.") from exc

    def get_dispositions(self, candidate_ids: list[str]) -> dict[str, TriageDisposition]:
        if not candidate_ids:
            return {}
        with Session(self._engine) as session:
            rows = session.execute(
                select(CandidateTriageStateORM.candidate_id, CandidateTriageStateORM.disposition)
                .where(CandidateTriageStateORM.candidate_id.in_(candidate_ids))
            )
            return {
                str(candidate_id): cast("TriageDisposition", disposition)
                for candidate_id, disposition in rows
            }

    def get_state(self, candidate_id: str) -> CandidateTriageStateRecord | None:
        with Session(self._engine) as session:
            row = session.get(CandidateTriageStateORM, candidate_id)
            if row is None:
                return None
            return CandidateTriageStateRecord(
                candidate_id=row.candidate_id,
                job_id=row.job_id,
                field_id=row.field_id,
                disposition=cast("TriageDisposition", row.disposition),
                reviewed_by_operator_id=row.reviewed_by_operator_id,
                updated_at_utc=row.updated_at_utc,
            )


class OLTPRepositorySet:
    """Module 2-owned OLTP engine lifecycle and repository collection."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.auth = OperatorAuthRepository(engine)
        self.jobs = PipelineJobRepository(engine)
        self.triage = CandidateTriageRepository(engine)

    def close(self) -> None:
        self.engine.dispose()

    def check_connection(self) -> bool:
        """Return whether the configured OLTP engine can execute a scalar health query."""
        try:
            with self.engine.connect() as connection:
                connection.execute(select(1))
        except SQLAlchemyError:
            return False
        return True


def create_oltp_repositories(
    config: OLTPDatabaseConfig | None = None,
) -> OLTPRepositorySet:
    """Create and initialize Module 2 OLTP repositories for an API lifespan."""
    engine = create_oltp_engine(config)
    initialize_oltp_schema(engine)
    return OLTPRepositorySet(engine)


__all__ = [
    "CandidateTriageRepository",
    "CandidateTriageStateRecord",
    "JobLifecycle",
    "OperatorAuthRepository",
    "OperatorProfile",
    "OLTPRepositorySet",
    "PipelineJobRecord",
    "PipelineJobRepository",
    "TriageDisposition",
    "TriageRecord",
    "RepositoryConflictError",
    "create_oltp_repositories",
    "refresh_token_hash",
]
