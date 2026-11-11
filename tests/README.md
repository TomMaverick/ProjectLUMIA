# Tests

## Overview
The `tests/` directory contains the automated test suite for Project LUMIA, structured into subsystem-specific test packages and root architectural boundary tests. It enforces strict type safety, mathematical fidelity, security constraints, and architectural isolation across all pipeline modules.

## Local Interface & Invariants
- `test_security_and_boundaries.py`: Architectural boundary and security test suite enforcing module isolation (e.g. physics package must not import database drivers), pickle deserialization safety (`weights_only=True`), CORS security, SQL injection defenses, and YAML safe loading.

## Subdirectories & Boundaries
- [test_api](./test_api/README.md): REST API endpoints, WebSocket event streaming, authentication, and triage contracts.
- [test_core](./test_core/README.md): Canonical Pydantic schemas, finite floating-point policy, and runtime determinism.
- [test_data_mining](./test_data_mining/README.md): Data mining pipeline, JPL Horizons ephemerides, artifact filters, and dataset loaders.
- [test_data_pipeline](./test_data_pipeline/README.md): DuckDB and PostgreSQL persistence, FITS ingestion, HDF5 storage, and Prefect workflows.
- [test_ml](./test_ml/README.md): PyTorch Real/Bogus CNN inference, preprocessing, and classification batches.
- [test_physics](./test_physics/README.md): WCS plate-solving, aperture photometry, GPU ZOGY subtraction, and HelioLinC linking.
- [test_training](./test_training/README.md): Deep learning training loops, tracklet network backbones, focal loss, and calibration.

## Verification
- Run: `uv run pytest`
