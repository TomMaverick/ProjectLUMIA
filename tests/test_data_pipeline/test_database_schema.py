"""TDD unit tests for Module 2 PostgreSQL OLTP (SQLAlchemy 2.x) and DuckDB/Parquet OLAP schemas."""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pytest
import sqlalchemy as sa
from pydantic import BaseModel, ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex, CreateTable

from LUMIA.data_pipeline.schema_olap import (
    OLAP_TABLE_DEFINITIONS,
    OLAPColumnSpec,
    OLAPSchemaConfig,
    OLAPTableDefinition,
    export_job_table_to_parquet,
    export_stellar_catalog_zone_to_parquet,
    initialize_olap_schema,
    initialize_parquet_directory_layout,
    job_dataset_partition_path,
    stellar_catalog_partition_path,
)
from LUMIA.data_pipeline.schema_oltp import (
    AuthRefreshSessionORM,
    CandidateTriageEventORM,
    CandidateTriageStateORM,
    CuratedDiscoveryORM,
    MPCSubmissionItemORM,
    MPCSubmissionORM,
    OLTPBase,
    OLTPDatabaseConfig,
    OperatorORM,
    OrbitalSolutionORM,
    PipelineJobORM,
    PipelineTaskLogORM,
    PipelineTaskORM,
    TelescopeProfileORM,
    configure_sqlite_engine,
    create_oltp_engine,
    initialize_oltp_schema,
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture
def sqlite_engine() -> sa.Engine:
    """Create an isolated in-memory SQLite engine with foreign key enforcement enabled."""
    engine = create_oltp_engine(
        OLTPDatabaseConfig(url="sqlite+pysqlite:///:memory:", echo=False)
    )
    initialize_oltp_schema(engine)
    return engine


def _seed_operator(
    session: Session,
    *,
    username: str = "astro_alice",
    email: str = "alice@lumia.org",
    role: str = "astronomer",
) -> OperatorORM:
    operator = OperatorORM(
        operator_id=uuid4(),
        username=username,
        email=email,
        password_hash="$argon2id$v=19$m=65536,t=3,p=4$dummyhashvalue",
        role=role,
        is_active=True,
    )
    session.add(operator)
    session.flush()
    return operator


def _seed_telescope_profile(
    session: Session,
    *,
    profile_id: str = "ZTF_P48",
    name: str = "Palomar 48-inch Oschin Schmidt",
) -> TelescopeProfileORM:
    profile = TelescopeProfileORM(
        profile_id=profile_id,
        name=name,
        mpc_code="I41",
        pixel_scale_arcsec=1.01,
        gain_e_per_adu=6.2,
        read_noise_e=10.3,
        saturation_limit_adu=55000.0,
        dark_current_e_per_s=0.02,
        latitude_deg=33.3563,
        longitude_deg=-116.8648,
        elevation_m=1712.0,
        aperture_diameter_m=1.22,
        focal_length_m=3.07,
    )
    session.add(profile)
    session.flush()
    return profile


def _seed_pipeline_job(
    session: Session,
    *,
    job_id: str = "job-20261107-001",
    operator_id: Any = None,
    profile_id: str | None = "ZTF_P48",
    target_field_id: str = "ZTF_0412",
) -> PipelineJobORM:
    job = PipelineJobORM(
        job_id=job_id,
        submitted_by_operator_id=operator_id,
        profile_id=profile_id,
        target_field_id=target_field_id,
        observation_night_utc=date(2026, 11, 7),
        ra_min_deg=180.0,
        ra_max_deg=182.0,
        dec_min_deg=-15.5,
        dec_max_deg=-13.5,
        max_mag=21.5,
        status="queued",
    )
    session.add(job)
    session.flush()
    return job


# ---------------------------------------------------------------------------
# 1. OLTP Metadata, 3NF Normalization & Dialect Compilation
# ---------------------------------------------------------------------------


class TestOLTPSchemaMetadataAnd3NFStructure:
    """Verify all 12 PostgreSQL 3NF OLTP tables, indexes, and dialect compilations."""

    EXPECTED_TABLES: frozenset[str] = frozenset(
        {
            "operators",
            "auth_refresh_sessions",
            "telescope_profiles",
            "pipeline_jobs",
            "pipeline_tasks",
            "pipeline_task_logs",
            "candidate_triage_state",
            "candidate_triage_events",
            "curated_discoveries",
            "orbital_solutions",
            "mpc_submissions",
            "mpc_submission_items",
        }
    )

    def test_all_twelve_3nf_tables_and_indexes_exist_in_sqlite(
        self,
        sqlite_engine: sa.Engine,
    ) -> None:
        """Verify create_all materializes all 12 blueprint tables, composite PKs, and indexes."""
        inspector = sa.inspect(sqlite_engine)
        actual_tables = frozenset(inspector.get_table_names())
        assert actual_tables == self.EXPECTED_TABLES

        # 2NF composite primary key on mpc_submission_items
        pk_info = inspector.get_pk_constraint("mpc_submission_items")
        assert pk_info["constrained_columns"] == ["submission_id", "discovery_id"]

        # Verify key blueprint indexes exist
        expected_indexes: dict[str, str] = {
            "auth_refresh_sessions": "idx_auth_sessions_operator_active",
            "pipeline_jobs": "idx_pipeline_jobs_status_submitted",
            "pipeline_tasks": "idx_pipeline_tasks_job_status",
            "pipeline_task_logs": "idx_task_logs_run_logged",
            "candidate_triage_state": "idx_triage_state_disposition_unexported",
            "candidate_triage_events": "idx_triage_events_candidate_time",
            "orbital_solutions": "uq_orbital_solutions_current",
        }
        for table_name, index_name in expected_indexes.items():
            idx_names = {idx["name"] for idx in inspector.get_indexes(table_name)}
            assert index_name in idx_names, (
                f"Missing index '{index_name}' on '{table_name}' (found {idx_names})."
            )

        night_field_idx = {
            idx["name"] for idx in inspector.get_indexes("pipeline_jobs")
        }
        assert "idx_pipeline_jobs_night_field" in night_field_idx

    def test_postgresql_dialect_compiles_native_types_regex_and_partial_indexes(
        self,
    ) -> None:
        """Verify PostgreSQL 17 DDL emits UUID, INET, BIGSERIAL, ~ regex, and WHERE indexes."""
        pg_dialect = postgresql.dialect()

        sessions_ddl = str(
            CreateTable(OLTPBase.metadata.tables["auth_refresh_sessions"]).compile(
                dialect=pg_dialect
            )
        )
        assert "UUID" in sessions_ddl
        assert "INET" in sessions_ddl
        assert "TIMESTAMP WITH TIME ZONE" in sessions_ddl

        profiles_ddl = str(
            CreateTable(OLTPBase.metadata.tables["telescope_profiles"]).compile(
                dialect=pg_dialect
            )
        )
        assert "~ '^[A-Za-z0-9_.-]+$'" in profiles_ddl
        assert "~ '^[A-Za-z0-9]{3}$'" in profiles_ddl

        logs_ddl = str(
            CreateTable(OLTPBase.metadata.tables["pipeline_task_logs"]).compile(
                dialect=pg_dialect
            )
        )
        assert "BIGSERIAL" in logs_ddl

        orbit_indexes = {
            idx.name: str(CreateIndex(idx).compile(dialect=pg_dialect))
            for idx in OLTPBase.metadata.tables["orbital_solutions"].indexes
        }
        assert "uq_orbital_solutions_current" in orbit_indexes
        assert "UNIQUE INDEX" in orbit_indexes["uq_orbital_solutions_current"]
        assert "WHERE" in orbit_indexes["uq_orbital_solutions_current"]

    def test_raw_sqlite_engine_helper_enables_foreign_keys(self) -> None:
        """Verify configure_sqlite_engine activates PRAGMA foreign_keys=ON on raw SQLite engines."""
        raw_engine = sa.create_engine("sqlite+pysqlite:///:memory:")
        configure_sqlite_engine(raw_engine)
        with raw_engine.connect() as conn:
            pragma_val = conn.execute(sa.text("PRAGMA foreign_keys")).scalar_one()
            assert int(pragma_val) == 1
        raw_engine.dispose()

    def test_oltp_config_enforces_strict_pydantic_contract(self) -> None:
        """Verify OLTPDatabaseConfig enforces ADR-0002 strict/frozen/forbid/allow_inf_nan=False."""
        cfg = OLTPDatabaseConfig.model_config
        assert cfg.get("strict") is True
        assert cfg.get("extra") == "forbid"
        assert cfg.get("frozen") is True
        assert cfg.get("allow_inf_nan") is False

        with pytest.raises(ValidationError):
            OLTPDatabaseConfig(url="")
        with pytest.raises(ValidationError):
            OLTPDatabaseConfig(echo="true")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            OLTPDatabaseConfig(unknown_option=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 2. OLTP Foreign Key Cascades, SET NULL, and RESTRICT Enforcement
# ---------------------------------------------------------------------------


class TestOLTPForeignKeyCascadesAndRestrictions:
    """Verify ON DELETE CASCADE, ON DELETE SET NULL, and ON DELETE RESTRICT on SQLite."""

    def test_rejects_orphan_foreign_keys_across_planes(
        self,
        sqlite_engine: sa.Engine,
    ) -> None:
        """Inserting child rows referencing nonexistent parent keys raises IntegrityError."""
        now = datetime.now(UTC)
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                AuthRefreshSessionORM(
                    operator_id=uuid4(),
                    token_jti_hash="a" * 64,
                    issued_at_utc=now,
                    expires_at_utc=now + timedelta(hours=1),
                )
            )
            session.commit()

        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                PipelineTaskORM(
                    job_id="nonexistent-job",
                    task_name="wcs-solve",
                    module_number=3,
                )
            )
            session.commit()

        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                CandidateTriageStateORM(
                    candidate_id="cand-orphan-01",
                    job_id="nonexistent-job",
                    field_id="ZTF_0412",
                )
            )
            session.commit()

    def test_on_delete_cascade_propagates_cleanly(
        self,
        sqlite_engine: sa.Engine,
    ) -> None:
        """Verify CASCADE deletes child sessions, tasks/logs, triage events, orbits, and items."""
        now = datetime.now(UTC)
        with Session(sqlite_engine) as session:
            # 1. Operator -> AuthRefreshSession CASCADE
            temp_op = _seed_operator(
                session, username="temp_user", email="temp@lumia.org"
            )
            temp_op_id = temp_op.operator_id
            session.add(
                AuthRefreshSessionORM(
                    operator_id=temp_op_id,
                    token_jti_hash="b" * 64,
                    issued_at_utc=now,
                    expires_at_utc=now + timedelta(days=7),
                    client_ip="192.168.10.5",
                )
            )
            session.commit()

        with Session(sqlite_engine) as session:
            session.execute(
                sa.delete(OperatorORM).where(OperatorORM.operator_id == temp_op_id)
            )
            session.commit()
            remaining_sessions = session.scalars(
                sa.select(AuthRefreshSessionORM).where(
                    AuthRefreshSessionORM.operator_id == temp_op_id
                )
            ).all()
            assert remaining_sessions == []

        # 2. PipelineJob -> PipelineTask -> PipelineTaskLog CASCADE
        with Session(sqlite_engine) as session:
            profile = _seed_telescope_profile(session)
            job = _seed_pipeline_job(
                session, job_id="job-cascade-1", profile_id=profile.profile_id
            )
            task = PipelineTaskORM(
                job_id=job.job_id,
                task_name="zogy-subtract",
                module_number=4,
                status="running",
            )
            session.add(task)
            session.flush()
            task_id = task.task_run_id
            session.add(
                PipelineTaskLogORM(
                    task_run_id=task_id,
                    attempt_number=1,
                    severity="INFO",
                    event_code="ZOGY_START",
                    message="Started ZOGY image subtraction.",
                )
            )
            session.commit()

        with Session(sqlite_engine) as session:
            session.execute(
                sa.delete(PipelineJobORM).where(PipelineJobORM.job_id == "job-cascade-1")
            )
            session.commit()
            assert session.get(PipelineTaskORM, task_id) is None
            remaining_logs = session.scalars(
                sa.select(PipelineTaskLogORM).where(
                    PipelineTaskLogORM.task_run_id == task_id
                )
            ).all()
            assert remaining_logs == []

        # 3. CandidateTriageState -> CandidateTriageEvent CASCADE
        # 4. CuratedDiscovery -> OrbitalSolution CASCADE
        # 5. MPCSubmission -> MPCSubmissionItem CASCADE
        with Session(sqlite_engine) as session:
            reviewer = _seed_operator(
                session, username="reviewer_bob", email="bob@lumia.org", role="reviewer"
            )
            job2 = _seed_pipeline_job(
                session,
                job_id="job-cascade-2",
                operator_id=reviewer.operator_id,
                profile_id="ZTF_P48",
            )
            triage_state = CandidateTriageStateORM(
                candidate_id="job-cascade-2_e0_c1",
                job_id=job2.job_id,
                field_id="ZTF_0412",
                disposition="confirmed_real",
                reviewed_by_operator_id=reviewer.operator_id,
                cnn_score_at_review=0.95,
                model_version_at_review="rb-1.0",
            )
            session.add(triage_state)
            session.flush()
            session.add(
                CandidateTriageEventORM(
                    candidate_id=triage_state.candidate_id,
                    operator_id=reviewer.operator_id,
                    previous_disposition="unreviewed",
                    new_disposition="confirmed_real",
                    notes="Clean PSF point source moving linearly.",
                )
            )

            discovery = CuratedDiscoveryORM(
                tracklet_id="job-cascade-2_trk_01",
                internal_designation="LUMIA-2026-00001",
                discovery_job_id=job2.job_id,
                confirmed_by_operator_id=reviewer.operator_id,
                dynamical_class="NEA",
                first_obs_bjd_tdb=2460755.5,
                last_obs_bjd_tdb=2460755.7,
            )
            session.add(discovery)
            session.flush()
            disc_id = discovery.discovery_id

            session.add(
                OrbitalSolutionORM(
                    discovery_id=disc_id,
                    is_current=True,
                    epoch_tdb_jd=2460755.6,
                    semi_major_axis_au=1.42,
                    eccentricity=0.31,
                    inclination_deg=12.4,
                    lon_asc_node_deg=88.1,
                    arg_perihelion_deg=210.5,
                    mean_anomaly_deg=45.2,
                    earth_moid_au=0.042,
                    abs_mag_h=21.3,
                    albedo_pv=0.14,
                    rms_residual_arcsec=0.11,
                    n_observations_used=4,
                    orbit_computer="OpenOrb-5.x",
                )
            )

            submission = MPCSubmissionORM(
                submitted_by_operator_id=reviewer.operator_id,
                submission_format="ADES_PSV",
                status="draft",
                payload_sha256="c" * 64,
            )
            session.add(submission)
            session.flush()
            sub_id = submission.submission_id
            session.add(
                MPCSubmissionItemORM(
                    submission_id=sub_id,
                    discovery_id=disc_id,
                    tracklet_id_submitted="job-cascade-2_trk_01",
                    n_astrometric_measurements=4,
                )
            )
            session.commit()

        with Session(sqlite_engine) as session:
            # Delete triage_state -> cascades triage_events
            session.execute(
                sa.delete(CandidateTriageStateORM).where(
                    CandidateTriageStateORM.candidate_id == "job-cascade-2_e0_c1"
                )
            )
            # Delete mpc_submission -> cascades mpc_submission_items
            session.execute(
                sa.delete(MPCSubmissionORM).where(
                    MPCSubmissionORM.submission_id == sub_id
                )
            )
            session.commit()

            assert (
                session.scalars(
                    sa.select(CandidateTriageEventORM).where(
                        CandidateTriageEventORM.candidate_id == "job-cascade-2_e0_c1"
                    )
                ).all()
                == []
            )
            assert (
                session.scalars(
                    sa.select(MPCSubmissionItemORM).where(
                        MPCSubmissionItemORM.submission_id == sub_id
                    )
                ).all()
                == []
            )

            # Now discovery has no mpc_submission_items -> deleting it cascades orbital_solutions
            session.execute(
                sa.delete(CuratedDiscoveryORM).where(
                    CuratedDiscoveryORM.discovery_id == disc_id
                )
            )
            session.commit()
            assert (
                session.scalars(
                    sa.select(OrbitalSolutionORM).where(
                        OrbitalSolutionORM.discovery_id == disc_id
                    )
                ).all()
                == []
            )

    def test_on_delete_set_null_and_restrict_constraints(
        self,
        sqlite_engine: sa.Engine,
    ) -> None:
        """Verify SET NULL preserves jobs/triage on operator deletion and RESTRICT blocks."""
        with Session(sqlite_engine) as session:
            op_submitter = _seed_operator(
                session, username="submitter_op", email="sub@lumia.org"
            )
            op_confirmer = _seed_operator(
                session, username="confirmer_op", email="conf@lumia.org"
            )
            profile = _seed_telescope_profile(session, profile_id="LSST_SIM")
            job = _seed_pipeline_job(
                session,
                job_id="job-restrict-01",
                operator_id=op_submitter.operator_id,
                profile_id=profile.profile_id,
            )
            triage = CandidateTriageStateORM(
                candidate_id="job-restrict-01_e0_c0",
                job_id=job.job_id,
                field_id="ZTF_0412",
                disposition="follow_up",
                reviewed_by_operator_id=op_submitter.operator_id,
            )
            discovery = CuratedDiscoveryORM(
                tracklet_id="job-restrict-01_trk_0",
                internal_designation="LUMIA-2026-00099",
                discovery_job_id=job.job_id,
                confirmed_by_operator_id=op_confirmer.operator_id,
                dynamical_class="MBA",
                first_obs_bjd_tdb=2460755.1,
                last_obs_bjd_tdb=2460755.3,
            )
            session.add_all([triage, discovery])
            session.flush()

            submission = MPCSubmissionORM(
                submitted_by_operator_id=op_confirmer.operator_id,
                submission_format="OBS80",
                status="transmitted",
                payload_sha256="d" * 64,
            )
            session.add(submission)
            session.flush()
            session.add(
                MPCSubmissionItemORM(
                    submission_id=submission.submission_id,
                    discovery_id=discovery.discovery_id,
                    tracklet_id_submitted=discovery.tracklet_id,
                    n_astrometric_measurements=3,
                )
            )
            submitter_id = op_submitter.operator_id
            confirmer_id = op_confirmer.operator_id
            disc_id = discovery.discovery_id
            session.commit()

        # 1. Deleting op_submitter triggers ON DELETE SET NULL on jobs and triage_state
        with Session(sqlite_engine) as session:
            session.execute(
                sa.delete(OperatorORM).where(OperatorORM.operator_id == submitter_id)
            )
            session.commit()
            refreshed_job = session.get(PipelineJobORM, "job-restrict-01")
            refreshed_triage = session.get(
                CandidateTriageStateORM, "job-restrict-01_e0_c0"
            )
            assert refreshed_job is not None
            assert refreshed_job.submitted_by_operator_id is None
            assert refreshed_triage is not None
            assert refreshed_triage.reviewed_by_operator_id is None

        # 2. Deleting telescope_profile referenced by pipeline_jobs is RESTRICTED
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.execute(
                sa.delete(TelescopeProfileORM).where(
                    TelescopeProfileORM.profile_id == "LSST_SIM"
                )
            )
            session.commit()

        # 3. Deleting pipeline_job referenced by triage_state / discovery is RESTRICTED
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.execute(
                sa.delete(PipelineJobORM).where(
                    PipelineJobORM.job_id == "job-restrict-01"
                )
            )
            session.commit()

        # 4. Deleting op_confirmer referenced by curated_discoveries / mpc_submissions is RESTRICTED
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.execute(
                sa.delete(OperatorORM).where(OperatorORM.operator_id == confirmer_id)
            )
            session.commit()

        # 5. Deleting curated_discovery referenced by mpc_submission_items is RESTRICTED
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.execute(
                sa.delete(CuratedDiscoveryORM).where(
                    CuratedDiscoveryORM.discovery_id == disc_id
                )
            )
            session.commit()


# ---------------------------------------------------------------------------
# 3. OLTP Domain CHECK Constraints & Partial Unique Index Enforcement
# ---------------------------------------------------------------------------


class TestOLTPCheckAndUniqueConstraints:
    """Verify domain CHECK constraints and partial unique orbital solution index."""

    def test_operator_and_session_constraints(self, sqlite_engine: sa.Engine) -> None:
        now = datetime.now(UTC)
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            _seed_operator(session, role="superadmin")
            session.commit()

        with Session(sqlite_engine) as session:
            op = _seed_operator(session, username="valid_op", email="valid@lumia.org")
            op_id = op.operator_id
            session.commit()

        # expires_at_utc must be strictly greater than issued_at_utc
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                AuthRefreshSessionORM(
                    operator_id=op_id,
                    token_jti_hash="e" * 64,
                    issued_at_utc=now,
                    expires_at_utc=now,
                )
            )
            session.commit()

    def test_telescope_profile_and_pipeline_job_constraints(
        self,
        sqlite_engine: sa.Engine,
    ) -> None:
        # Invalid profile_id regex
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            _seed_telescope_profile(session, profile_id="BAD PROFILE/ID")
            session.commit()

        # Invalid 2-char MPC code
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            profile = _seed_telescope_profile(
                session, profile_id="BAD_MPC", name="Bad MPC"
            )
            profile.mpc_code = "I4"
            session.commit()

        # Valid meridian wrap-around job (ra_min_deg=358.5, ra_max_deg=1.5) succeeds
        with Session(sqlite_engine) as session:
            _seed_telescope_profile(session, profile_id="WRAP_TEL", name="Wrap Tel")
            wrap_job = PipelineJobORM(
                job_id="job-meridian-wrap",
                profile_id="WRAP_TEL",
                target_field_id="ZTF_WRAP_00",
                observation_night_utc=date(2026, 11, 7),
                ra_min_deg=358.5,
                ra_max_deg=1.5,
                dec_min_deg=-5.0,
                dec_max_deg=5.0,
                max_mag=21.0,
                status="running",
            )
            session.add(wrap_job)
            session.commit()

        # Inverted declination bounds (dec_min_deg > dec_max_deg) fails
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                PipelineJobORM(
                    job_id="job-bad-dec",
                    profile_id="WRAP_TEL",
                    target_field_id="ZTF_BAD_DEC",
                    observation_night_utc=date(2026, 11, 7),
                    ra_min_deg=10.0,
                    ra_max_deg=12.0,
                    dec_min_deg=15.0,
                    dec_max_deg=10.0,
                    max_mag=21.0,
                )
            )
            session.commit()

        # Invalid module_number (11) on pipeline_tasks fails
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                PipelineTaskORM(
                    job_id="job-meridian-wrap",
                    task_name="bad-module",
                    module_number=11,
                )
            )
            session.commit()

    def test_orbital_solutions_partial_unique_current_index_and_discovery_checks(
        self,
        sqlite_engine: sa.Engine,
    ) -> None:
        """At most one is_current=True orbit per discovery while historical False rows succeed."""
        with Session(sqlite_engine) as session:
            op = _seed_operator(session, username="orb_op", email="orb@lumia.org")
            _seed_telescope_profile(session, profile_id="ORB_TEL", name="Orb Tel")
            job = _seed_pipeline_job(
                session,
                job_id="job-orb-01",
                operator_id=op.operator_id,
                profile_id="ORB_TEL",
            )
            discovery = CuratedDiscoveryORM(
                tracklet_id="job-orb-01_trk_1",
                internal_designation="LUMIA-2026-00042",
                mpc_provisional_desig="2026 VQ12",
                discovery_job_id=job.job_id,
                confirmed_by_operator_id=op.operator_id,
                dynamical_class="Trojan",
                first_obs_bjd_tdb=2460755.2,
                last_obs_bjd_tdb=2460755.8,
            )
            session.add(discovery)
            session.flush()
            disc_id = discovery.discovery_id

            # Two historical (is_current=False) + one active (is_current=True) succeed
            for idx, is_curr in enumerate([False, False, True]):
                session.add(
                    OrbitalSolutionORM(
                        discovery_id=disc_id,
                        is_current=is_curr,
                        epoch_tdb_jd=2460755.5 + idx * 0.1,
                        semi_major_axis_au=5.20,
                        eccentricity=0.08,
                        inclination_deg=14.2,
                        lon_asc_node_deg=120.0,
                        arg_perihelion_deg=65.0,
                        mean_anomaly_deg=310.0,
                        rms_residual_arcsec=0.09,
                        n_observations_used=3 + idx,
                        orbit_computer="OpenOrb-5.x",
                    )
                )
            session.commit()

        # Second is_current=True orbit for the same discovery violates partial unique index
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                OrbitalSolutionORM(
                    discovery_id=disc_id,
                    is_current=True,
                    epoch_tdb_jd=2460756.0,
                    semi_major_axis_au=5.21,
                    eccentricity=0.07,
                    inclination_deg=14.1,
                    lon_asc_node_deg=120.1,
                    arg_perihelion_deg=65.1,
                    mean_anomaly_deg=310.2,
                    rms_residual_arcsec=0.07,
                    n_observations_used=8,
                    orbit_computer="OpenOrb-5.x",
                )
            )
            session.commit()

        # Discovery with last_obs_bjd_tdb < first_obs_bjd_tdb fails CHECK
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            op2 = session.scalars(
                sa.select(OperatorORM).where(OperatorORM.username == "orb_op")
            ).one()
            session.add(
                CuratedDiscoveryORM(
                    tracklet_id="job-orb-01_trk_bad",
                    internal_designation="LUMIA-2026-BAD",
                    discovery_job_id="job-orb-01",
                    confirmed_by_operator_id=op2.operator_id,
                    dynamical_class="NEA",
                    first_obs_bjd_tdb=2460756.0,
                    last_obs_bjd_tdb=2460755.0,
                )
            )
            session.commit()

    def test_triage_task_log_and_mpc_check_constraints_and_orm_relationships(
        self,
        sqlite_engine: sa.Engine,
    ) -> None:
        """Verify remaining CHECK constraints, UTCDateTime tzinfo, and ORM relationships."""
        with Session(sqlite_engine) as session:
            op = _seed_operator(session, username="rel_op", email="rel@lumia.org")
            _seed_telescope_profile(session, profile_id="REL_TEL", name="Rel Tel")
            job = _seed_pipeline_job(
                session,
                job_id="job-rel-01",
                operator_id=op.operator_id,
                profile_id="REL_TEL",
            )
            task = PipelineTaskORM(
                job_id=job.job_id,
                task_name="cnn-classify",
                module_number=7,
                status="completed",
            )
            session.add(task)
            session.flush()
            log1 = PipelineTaskLogORM(
                task_run_id=task.task_run_id,
                attempt_number=1,
                severity="INFO",
                message="Batch 1 classified.",
            )
            log2 = PipelineTaskLogORM(
                task_run_id=task.task_run_id,
                attempt_number=1,
                severity="WARNING",
                message="Low SNR stamp encountered.",
            )
            session.add_all([log1, log2])
            session.commit()

            assert log1.log_id < log2.log_id
            assert op.created_at_utc.tzinfo == UTC
            assert len(job.tasks) == 1
            assert len(job.tasks[0].logs) == 2
            task_run_id = task.task_run_id
            op_id = op.operator_id

        # Invalid log severity
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                PipelineTaskLogORM(
                    task_run_id=task_run_id,
                    attempt_number=1,
                    severity="TRACE",
                    message="Invalid severity.",
                )
            )
            session.commit()

        # Invalid triage cnn_score_at_review (> 1.0)
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                CandidateTriageStateORM(
                    candidate_id="job-rel-01_e0_c0",
                    job_id="job-rel-01",
                    field_id="ZTF_0412",
                    disposition="confirmed_real",
                    cnn_score_at_review=1.25,
                )
            )
            session.commit()

        # Invalid MPC submission format
        with pytest.raises(IntegrityError), Session(sqlite_engine) as session:
            session.add(
                MPCSubmissionORM(
                    submitted_by_operator_id=op_id,
                    submission_format="JSON",
                    status="draft",
                    payload_sha256="f" * 64,
                )
            )
            session.commit()


# ---------------------------------------------------------------------------
# 4. OLAP DuckDB Tables, Analytical Views & Hive-Partitioned Parquet Export
# ---------------------------------------------------------------------------


class TestOLAPDuckDBAndParquetSchema:
    """Verify DuckDB OLAP DDL, v_tracklet_triage_summary view, and ZSTD Parquet partitioning."""

    EXPECTED_OLAP_TABLES: frozenset[str] = frozenset(
        {
            "stellar_catalog",
            "candidate_detections",
            "candidate_cutout_stamps",
            "space_velocity_states",
            "tracklets",
            "tracklet_members",
        }
    )

    def test_olap_pydantic_models_enforce_strict_finite_policy(self) -> None:
        for model_cls in (OLAPSchemaConfig, OLAPColumnSpec, OLAPTableDefinition):
            assert issubclass(model_cls, BaseModel)
            cfg = model_cls.model_config
            assert cfg.get("strict") is True
            assert cfg.get("extra") == "forbid"
            assert cfg.get("frozen") is True
            assert cfg.get("allow_inf_nan") is False

        assert frozenset(OLAP_TABLE_DEFINITIONS.keys()) == self.EXPECTED_OLAP_TABLES

    def test_initialize_olap_schema_creates_tables_constraints_and_triage_view(
        self,
    ) -> None:
        conn = duckdb.connect(":memory:")
        try:
            initialize_olap_schema(conn)

            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
                ).fetchall()
            }
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main' AND table_type = 'VIEW'"
                ).fetchall()
            }
            assert tables == self.EXPECTED_OLAP_TABLES
            assert "v_tracklet_triage_summary" in views

            # Insert 3 candidate detections, 1 tracklet, 3 members, and 1 31x31 cutout stamp
            obs_night = date(2026, 11, 7)
            for idx, (mag, snr, score) in enumerate(
                [(20.2, 12.0, 0.96), (20.5, 10.0, 0.90), (20.8, 8.0, 0.84)]
            ):
                conn.execute(
                    """
                    INSERT INTO candidate_detections (
                        candidate_id, job_id, observation_night, field_id,
                        ra_deg, dec_deg, bjd_tdb, x_centroid_px, y_centroid_px,
                        net_flux_adu, flux_err_adu, mag, mag_err, snr,
                        cnn_score, model_version, astrometric_uncertainty_arcsec,
                        astrometric_residual_arcsec, tracklet_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        f"job-01_e{idx}_c0",
                        "job-01",
                        obs_night,
                        "ZTF_0412",
                        180.1 + idx * 0.01,
                        -15.2 + idx * 0.01,
                        2460755.5 + idx * 0.05,
                        512.25,
                        1024.75,
                        1450.0,
                        35.0,
                        mag,
                        0.04,
                        snr,
                        score,
                        "rb-1.0",
                        0.10,
                        0.07,
                        "job-01_trk_0",
                    ],
                )

            conn.execute(
                """
                INSERT INTO tracklets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    "job-01_trk_0",
                    "job-01",
                    obs_night,
                    "ZTF_0412",
                    3,
                    2460755.55,
                    14.2,
                    -6.5,
                    15.6169,
                    1.2,
                    0.07,
                    18.4,
                ],
            )
            for idx in range(3):
                conn.execute(
                    "INSERT INTO tracklet_members VALUES (?, ?, ?, ?)",
                    ["job-01_trk_0", idx, f"job-01_e{idx}_c0", 0.07],
                )

            stamp_grid = [[1.25] * 31 for _ in range(31)]
            conn.execute(
                """
                INSERT INTO candidate_cutout_stamps VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    "job-01_e0_c0",
                    0,
                    obs_night,
                    "ZTF_0412",
                    31,
                    2460755.5,
                    20.2,
                    0.04,
                    15.0,
                    15.0,
                    stamp_grid,
                    stamp_grid,
                    stamp_grid,
                ],
            )

            # Reject invalid cutout_size (e.g., even 32)
            with pytest.raises(duckdb.ConstraintException):
                conn.execute(
                    """
                    INSERT INTO candidate_cutout_stamps VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        "job-01_e0_c0",
                        1,
                        obs_night,
                        "ZTF_0412",
                        32,
                        2460755.5,
                        20.2,
                        0.04,
                        15.0,
                        15.0,
                        [[1.0] * 32 for _ in range(32)],
                        None,
                        None,
                    ],
                )

            # Verify v_tracklet_triage_summary analytical aggregation
            summary_row = conn.execute(
                """
                SELECT
                    tracklet_id, n_epochs, mean_cnn_score,
                    min_cnn_score, brightest_mag, mean_snr
                FROM v_tracklet_triage_summary
                WHERE tracklet_id = ?
                """,
                ["job-01_trk_0"],
            ).fetchone()
            assert summary_row is not None
            assert summary_row[0] == "job-01_trk_0"
            assert int(summary_row[1]) == 3
            assert float(summary_row[2]) == pytest.approx(0.90)
            assert float(summary_row[3]) == pytest.approx(0.84)
            assert float(summary_row[4]) == pytest.approx(20.2)
            assert float(summary_row[5]) == pytest.approx(10.0)
        finally:
            conn.close()

    def test_hive_partitioned_zstd_parquet_export_and_traversal_guard(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify Hive directories, ZSTD Parquet export across job datasets, and pruning."""
        from LUMIA.data_pipeline.database import CatalogDatabase, DuckDBConfig

        olap_cfg = OLAPSchemaConfig(
            parquet_root_dir=tmp_path / "olap",
            compression="ZSTD",
            row_group_size=100_000,
        )
        dirs = initialize_parquet_directory_layout(olap_cfg)
        assert set(dirs.keys()) == self.EXPECTED_OLAP_TABLES
        assert all(p.is_dir() for p in dirs.values())

        # Path traversal in field_id or job_id must raise ValueError
        with pytest.raises(ValueError, match="identifier"):
            job_dataset_partition_path(
                olap_cfg.parquet_root_dir,
                "candidate_detections",
                observation_night=date(2026, 11, 7),
                field_id="../evil",
                job_id="job-001",
            )

        with pytest.raises(ValueError, match="dec_zone"):
            stellar_catalog_partition_path(olap_cfg.parquet_root_dir, dec_zone=-95)

        with CatalogDatabase(DuckDBConfig(database=":memory:")) as cat_db:
            initialize_olap_schema(cat_db)
            conn = cat_db._require_connection()  # noqa: SLF001
            conn.execute(
                """
                INSERT INTO stellar_catalog (
                    dec_zone, healpix_ipix, source_id, ra_deg, dec_deg, mag_g,
                    mag_g_err, mag_bp, mag_rp, pm_ra_mas_yr, pm_dec_mas_yr, ref_epoch_jy
                ) VALUES
                    (-15, 10200, 5002, 180.4, -14.8, 17.5, 0.01, 17.9, 16.9, -3.2, 1.4, 2016.0),
                    (-15, 10199, 5001, 180.1, -14.9, 16.2, 0.01, 16.5, 15.8, 2.1, -0.8, 2016.0)
                """
            )
            cat_parquet = export_stellar_catalog_zone_to_parquet(
                cat_db,
                olap_cfg,
                dec_zone=-15,
                part_index=0,
            )
            assert (
                cat_parquet
                == olap_cfg.parquet_root_dir
                / "stellar_catalog"
                / "dec_zone=-15"
                / "part-0000.parquet"
            )
            assert cat_parquet.is_file()

            conn.execute(
                """
                INSERT INTO candidate_detections (
                    candidate_id, job_id, observation_night, field_id,
                    ra_deg, dec_deg, bjd_tdb, mag, snr, cnn_score, model_version,
                    astrometric_uncertainty_arcsec
                ) VALUES
                    (
                        'c-low', 'job-001', '2026-11-07', 'ZTF_0412',
                        180.2, -14.5, 2460755.5, 21.0, 6.0, 0.55, 'rb-1', 0.12
                    ),
                    (
                        'c-high', 'job-001', '2026-11-07', 'ZTF_0412',
                        180.3, -14.6, 2460755.5, 19.8, 14.0, 0.98, 'rb-1', 0.08
                    )
                """
            )
            grid_31 = [[0.5] * 31 for _ in range(31)]
            conn.execute(
                "INSERT INTO candidate_cutout_stamps "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "c-high",
                    0,
                    date(2026, 11, 7),
                    "ZTF_0412",
                    31,
                    2460755.5,
                    19.8,
                    0.03,
                    15.0,
                    15.0,
                    grid_31,
                    None,
                    None,
                ],
            )
            conn.execute(
                "INSERT INTO tracklets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "job-001_trk_0",
                    "job-001",
                    date(2026, 11, 7),
                    "ZTF_0412",
                    3,
                    2460755.5,
                    10.0,
                    5.0,
                    11.18,
                    0.5,
                    0.06,
                    15.0,
                ],
            )
            conn.execute(
                "INSERT INTO tracklet_members VALUES (?, ?, ?, ?)",
                ["job-001_trk_0", 0, "c-high", 0.06],
            )

            cand_parquet = export_job_table_to_parquet(
                cat_db,
                olap_cfg,
                dataset_name="candidate_detections",
                observation_night=date(2026, 11, 7),
                field_id="ZTF_0412",
                job_id="job-001",
            )
            cutout_parquet = export_job_table_to_parquet(
                cat_db,
                olap_cfg,
                dataset_name="candidate_cutout_stamps",
                observation_night=date(2026, 11, 7),
                field_id="ZTF_0412",
                job_id="job-001",
            )
            members_parquet = export_job_table_to_parquet(
                cat_db,
                olap_cfg,
                dataset_name="tracklet_members",
                observation_night=date(2026, 11, 7),
                field_id="ZTF_0412",
                job_id="job-001",
            )
            assert (
                cand_parquet
                == olap_cfg.parquet_root_dir
                / "candidate_detections"
                / "observation_night=2026-11-07"
                / "field_id=ZTF_0412"
                / "job_id=job-001.parquet"
            )
            assert cand_parquet.is_file()
            assert cutout_parquet.is_file()
            assert members_parquet.is_file()

        # Verify zero-copy Parquet scan from a separate read-only / :memory: DuckDB reader
        reader = duckdb.connect(":memory:")
        try:
            cand_glob = (
                olap_cfg.parquet_root_dir / "candidate_detections" / "**" / "*.parquet"
            ).as_posix()
            rows = reader.execute(
                "SELECT candidate_id, cnn_score FROM read_parquet(?, hive_partitioning=true)",
                [cand_glob],
            ).fetchall()
            # Row-group sorting orders by cnn_score DESC, snr DESC, candidate_id ASC
            assert [r[0] for r in rows] == ["c-high", "c-low"]
        finally:
            reader.close()


# ---------------------------------------------------------------------------
# 5. Strict Polyglot Database Import Separation (ADR-0001 & ADR-0003)
# ---------------------------------------------------------------------------


class TestStrictDatabaseImportIsolation:
    """Ensure SQLAlchemy and DuckDB imports never leak outside src/LUMIA/data_pipeline/."""

    FORBIDDEN_DB_PACKAGES: tuple[str, ...] = ("sqlalchemy", "duckdb")

    def test_no_sqlalchemy_or_duckdb_imports_outside_data_pipeline(self) -> None:
        """Scan all non-data_pipeline modules (including TYPE_CHECKING blocks) for DB imports."""
        lumia_root = _REPO_ROOT / "src" / "LUMIA"
        data_pipeline_dir = (lumia_root / "data_pipeline").resolve()

        checked_files = 0
        violations: list[str] = []

        for py_file in sorted(lumia_root.rglob("*.py")):
            if py_file.resolve().is_relative_to(data_pipeline_dir):
                continue
            checked_files += 1
            rel_path = py_file.relative_to(_REPO_ROOT)
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(
                            alias.name == pkg or alias.name.startswith(f"{pkg}.")
                            for pkg in self.FORBIDDEN_DB_PACKAGES
                        ):
                            violations.append(
                                f"{rel_path}:{node.lineno} -> import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if any(
                        mod == pkg or mod.startswith(f"{pkg}.")
                        for pkg in self.FORBIDDEN_DB_PACKAGES
                    ):
                        violations.append(
                            f"{rel_path}:{node.lineno} -> from {mod} import ..."
                        )
                    if mod in {
                        "LUMIA.data_pipeline.schema_oltp",
                        "LUMIA.data_pipeline.schema_olap",
                    }:
                        violations.append(
                            f"{rel_path}:{node.lineno} -> leaked internal schema {mod}"
                        )

        assert checked_files > 0
        assert violations == [], (
            "Database imports leaked outside src/LUMIA/data_pipeline/:\n"
            + "\n".join(violations)
        )
