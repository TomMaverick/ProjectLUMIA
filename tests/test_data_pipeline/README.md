# Test Data Pipeline

## Overview
The `tests/test_data_pipeline/` test package validates Module 2 data persistence, FITS ingestion, out-of-core HDF5 storage, Prefect workflow orchestration, and polyglot repository contracts (DuckDB OLAP and PostgreSQL OLTP).

## Interface
- `test_database_schema.py`: Verifies SQL DDL generation, column types, indexing, and foreign key integrity.
- `test_fits_ingest.py`: Tests FITS header validation, celestial WCS coordinate parsing, and exposure metadata extraction.
- `test_ground_truth_orchestrator.py`: Validates the pipeline transforming candidate detections into labeled ground truth datasets.
- `test_hdf5_storage.py`: Tests low-level HDF5 dataset creation, chunking, and concurrent read/write operations.
- `test_orchestration.py`: Tests Prefect task and flow DAG execution for end-to-end pipeline runs.
- `test_pipeline_persistence_bridge.py`: Tests data conversion and handoff between DuckDB analytical queries and relational tables.
- `test_repositories.py`: Validates spatial queries, magnitude filtering, and candidate persistence in `DuckDBRepository` and `PostgresRepository`.

## Invariants
- Tests run against volatile in-memory DuckDB (`:memory:`) or isolated test databases.
- All database fixtures clean up open connections upon test completion.

## Side Effects
- Spools temporary test database files and Parquet partitions in temporary test folders.
- Writes and deletes temporary FITS and HDF5 test files.

## Verification
- Run: `uv run pytest tests/test_data_pipeline`
