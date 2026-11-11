# LUMIA Python Package

## Overview
`src/LUMIA/` is the primary Python package root for Project LUMIA (Low-SNR Unidentified Moving Interplanetary Analyzer). It organizes the ten canonical pipeline modules into decoupled subsystems covering core schemas, astrophysical processing, deep learning classification, data persistence, and web interfaces.

## Local Interface & Invariants
- `CandidateDetection`: Canonical moving-object detection contract exposed at the package root.
- `__version__`: Package version identifier (`0.1.0`).
- `py.typed`: PEP 561 marker asserting full static type annotations across all submodules.
- Lazy subpackage loading: top-level subpackages (`api`, `core`, `data_pipeline`, `ml`, `physics`) are loaded dynamically on demand via `__getattr__` to prevent expensive initialization side effects during lightweight imports.

## Subdirectories & Boundaries
- [api](./api/README.md): Module 10 – FastAPI REST and WebSocket service managing client communication and triage review.
- [core](./core/README.md): Shared strict Pydantic v2 schemas, finite floating-point policy, determinism, and CPU/GPU resource budgeting.
- [data_mining](./data_mining/README.md): Automated survey crawler, JPL Horizons ephemeris querying, artifact screening, and HDF5 dataset curation.
- [data_pipeline](./data_pipeline/README.md): Module 2 – Prefect orchestration, DuckDB analytical storage, PostgreSQL transactional persistence, and FITS ingestion.
- [ml](./ml/README.md): Module 7 – Pure PyTorch Real/Bogus inference engine, MAD normalization, and confidence scoring.
- [models](./models/README.md): Deep neural network architectures (`ResNetLite`, `BiGRUAggregator`, `TransformerAggregator`) for spatial-temporal sequence classification.
- [physics](./physics/README.md): Modules 1, 3, 4, 5 – WCS astrometry, aperture photometry, GPU ZOGY subtraction, and HelioLinC tracklet linking.
- [training](./training/README.md): Model training framework, focal loss functions, temperature calibration, and atomic checkpointing.

## Verification
- Run: `uv run pytest`
