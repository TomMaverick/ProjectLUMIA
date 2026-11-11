# ADR-0003: Polyglot Persistence — DuckDB/Parquet for Analytics vs. PostgreSQL for Transactional State

* **Status:** Accepted
* **Date:** 2026-11-07
* **Deciders:** Project LUMIA Architecture Team
* **Related Documents:** `docs/lastenheft.md` (Sections 3, 4.2, 4.10), `docs/modules/module_02_pipeline.md`, `docs/audits/architecture_audit_v1.md` (Findings `P0-1`, `P2-3`, `P2-5`)

---

## Context

Project LUMIA processes wide-field astronomical exposures generating high-volume stellar catalog lookups and transient candidate detections alongside interactive, multi-user pipeline orchestration and human review workflows. These workloads impose conflicting database engineering requirements:

1. **Analytical & Columnar Triage Workload (OLAP):**
   * Spatial sky-box queries (`RA`/`Dec` including `0°/360°` meridian wrap-around) and limiting-magnitude cuts across millions of reference catalog stars.
   * Bulk insertion and multi-criteria ranking (`WHERE cnn_score >= ? ORDER BY cnn_score DESC, snr DESC`) across tens of thousands of single-epoch candidate detections and tracklet metrics.
   * Zero-copy or vectorized extraction directly into typed NumPy arrays for `photutils`, `scipy`, and `torch` without ORM object-materialization overhead.
   * Strict SSD write-endurance protection (`docs/lastenheft.md` Section 3), favoring in-memory query execution and immutable columnar files over write-heavy row-store B-tree churn.

2. **Transactional State, Triage & Long-Term Operational Workload (OLTP):**
   * Concurrent row-level state transitions for Prefect/API pipeline jobs (`queued` -> `running` -> `completed` / `failed`), retry counters, and structured execution audit logs.
   * Multi-client Web-UI operations including JWT user/session management and human Active Learning triage dispositions (`unreviewed`, `confirmed_real`, `follow_up`, `flagged_bogus`) requiring ACID isolation without blocking analytical scans.
   * Long-term curated storage of verified asteroid discoveries, orbital parameters, and Minor Planet Center (MPC) submission histories.

In the initial prototype (`docs/audits/architecture_audit_v1.md`, Finding `P0-1`), job state was held in an ephemeral Python `dict` inside `src/LUMIA/api/routes.py`, and raw DuckDB SQL DDL/DML was embedded directly inside FastAPI route handlers serialized behind a single process-wide `threading.Lock()`. Relying solely on DuckDB for concurrent OLTP mutations causes writer lock contention across processes, whereas relying solely on PostgreSQL for wide-field star-catalog and bulk array analytics sacrifices columnar vectorization speed and increases disk I/O overhead.

---

## Decision

We adopt a **Polyglot Dual-Database Architecture** strictly encapsulated behind repository interfaces inside **Module 2** (`src/LUMIA/data_pipeline/`):

### 1. DuckDB + Apache Parquet (Analytical & Triage OLAP Plane)
* **Scope:** Reference stellar catalogs (`stellar_catalog`), bulk single-epoch `CandidateDetection` tables, velocity-space / `Tracklet` analytical views, field-level summary aggregations, and fast Web-UI candidate filtering/sorting.
* **Storage Strategy:** Persistent reference catalogs and finalized field detections are stored as partitioned, immutable **Apache Parquet** files queried directly or cached via **DuckDB** (`:memory:` instances for transient pipeline runs and `read_only=True` connections for concurrent Web-UI readers).
* **Data Access:** Exposes typed repository methods returning validated Pydantic models or explicitly typed structured NumPy arrays (`float64`/`int64`), utilizing `executemany` / Arrow / Parquet bulk ingestion rather than row-by-row loops.

### 2. PostgreSQL via SQLAlchemy 2.x (Transactional State & Long-Term OLTP Plane)
* **Scope:** Pipeline job lifecycle records, Prefect task/flow audit logs, Web-UI authentication and operator metadata, human candidate triage dispositions (feeding **Module 8** Active Learning HDF5 exports), and curated long-term asteroid tracklet/MPC reporting records.
* **Storage Strategy:** Managed relational schema accessed through SQLAlchemy 2.x connection pooling with ACID transactions (supporting lightweight SQLite/in-memory adapters exclusively within isolated `pytest` unit test fixtures).

### 3. Strict Repository Boundary Enforcement
* All SQL schemas, migrations, connection lifecycles, and parameterized queries must live exclusively inside `src/LUMIA/data_pipeline/`.
* **Modules 3, 4, 5, 7, and 8** (`physics`, `ml`) must never import `duckdb`, `sqlalchemy`, or repository classes.
* **Module 10** (`src/LUMIA/api/`) must interact with persistence exclusively through injected Module 2 repository/service interfaces—never executing raw SQL strings or holding ad-hoc in-memory job dictionaries.
* Cross-database linkage between PostgreSQL state records and DuckDB/Parquet analytical artifacts uses deterministic, immutable string identifiers (`job_id`, `field_id`, `candidate_id`, `tracklet_id`) validated via ADR-0002 Pydantic contracts.

---

## Consequences

### Positive
* Aligns each storage engine with its native strength: sub-millisecond vectorized columnar scans and zero-churn Parquet analytics in DuckDB paired with lock-free concurrent ACID state management and durable triage auditability in PostgreSQL.
* Protects host SSD longevity during heavy transient-hunting runs by combining `:memory:` DuckDB execution, `SpooledTemporaryFile` FITS ingestion, and sequential Parquet writes.
* Decouples FastAPI HTTP/WebSocket handlers (`Module 10`) and Prefect flows (`Module 2`) from SQL implementation details, enabling clean dependency injection and fast offline unit testing.

### Negative / Trade-offs
* Introduces operational overhead of running PostgreSQL alongside DuckDB/Parquet file management in production deployments (mitigated via declarative Docker Compose service definitions).
* Cross-engine relationships (`PostgreSQL` triage dispositions referencing `DuckDB`/Parquet `candidate_id` records) do not have single-engine foreign-key enforcement; application-level repository services in Module 2 must validate referential existence before recording triage dispositions or exporting Active Learning datasets.
