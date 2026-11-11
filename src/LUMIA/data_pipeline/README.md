# LUMIA Data Pipeline

## Overview
LUMIA Data Pipeline implements Module 2 of Project LUMIA: data engineering, polyglot persistence, and workflow orchestration. It enforces the polyglot persistence architecture (ADR-0003), splitting workloads into an OLAP analytical plane (DuckDB and Apache Parquet) for high-throughput spatial queries and an OLTP transactional plane (PostgreSQL via SQLAlchemy 2.x) for ACID candidate triage, operator accounts, and Prefect pipeline task logging.

## Interface
- `CatalogDatabase` / `CandidateDetectionRepository`: Encapsulates the DuckDB analytical engine, supporting in-memory processing, candidate queries, spatial boundary lookups with meridian wrap-around handling, and Parquet persistence.
- `OperatorRepository` / `TriageRepository` / `JobRepository`: Manages relational PostgreSQL operations, including operator authentication records, candidate triage events, and job execution metadata.
- `FITSIngestionGatekeeper`: Validates incoming calibrated FITS files, primary header keys, celestial WCS coordinates, observation times, and telescope metadata.
- `HDF5DatasetManager`: Provides low-level out-of-core reading and chunked writing for HDF5 ground truth dataset archives.
- `GroundTruthDatasetOrchestrator`: Orchestrates the transformation of pipeline detections and cutouts into standardized HDF5 training batches.
- `TriageBatchLoader`: Loads candidates and associated cutout sequences from HDF5 or database tables for review queues.
- `DuckDBConfig` / `OLTPDatabaseConfig`: Validated Pydantic configuration contracts for database connections and connection pools.
- `CatalogQueryParams`: Schema defining spatial coordinates, search radii, and magnitude bounds for reference star catalog queries.

## Invariants
- `DuckDBConfig`: Setting `database=":memory:"` with `read_only=True` is prohibited and raises `ValueError`.
- SQL identifiers are validated against `^[A-Za-z_][A-Za-z0-9_]*$` to prevent SQL injection during dynamic table construction.
- Strict separation of concerns: analytical spatial queries execute in DuckDB/Parquet; transactional state and triage dispositions reside in PostgreSQL.
- Temporary files must use `tempfile.SpooledTemporaryFile` to prevent excessive NVMe/SSD wear.

## Side Effects
- Creates, queries, and closes DuckDB database files and in-memory tables.
- Opens connection pools and executes SQL transactions against PostgreSQL.
- Reads raw FITS files from the telescope archive and writes Apache Parquet files to disk.
- Creates and updates HDF5 dataset files in storage directories.

## Verification
- Run: `uv run pytest tests/test_data_pipeline`
