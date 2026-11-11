# Test API

## Overview
The `tests/test_api/` test package provides unit and integration tests verifying Module 10 (FastAPI REST and WebSocket gateway). It validates authentication flows, candidate listing endpoints, candidate triage decision persistence, WebSocket frame delivery, and Pydantic schema validation.

## Interface
- `test_module10_auth_jobs_ws.py`: Tests for JWT token generation, Argon2 authentication, job submission endpoints, and real-time WebSocket events.
- `test_schema_contracts.py`: Validates strict Pydantic contract parsing, rejection of extra fields, and format constraints on API request/response models.
- `test_settings.py`: Tests for runtime resource budget queries and dynamic settings updates.
- `test_triage_hdf5.py`: Integration tests for triage queue streaming from out-of-core HDF5 datasets and database tables.
- `test_web_ui.py`: Verifies endpoints serving frontend application data and CORS configurations.

## Invariants
- Tests execute with isolated in-memory DuckDB and SQLite/PostgreSQL test engines without modifying production databases.
- All test functions assert strict status codes, response envelope schemas, and authenticated session requirements.

## Side Effects
- Spawns in-memory ASGI test clients via `httpx.AsyncClient` or `starlette.testclient.TestClient`.
- Allocates temporary file buffers for test cutout data.

## Verification
- Run: `uv run pytest tests/test_api`
