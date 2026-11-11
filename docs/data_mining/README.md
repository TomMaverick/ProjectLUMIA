# Data Mining Architecture Blueprint

## Overview
The `docs/data_mining/` directory contains the detailed architectural blueprints and algorithmic specifications for Project LUMIA's solar system object data-mining, tracklet harvesting, artifact routing, and HDF5 serialization subsystem.

## Interface
- [crawler_operations_guide.md](./crawler_operations_guide.md): Operations guide, CLI reference, and execution runbook for running `scripts/run_mining.py`.
- [architecture_blueprint.md](./architecture_blueprint.md): Complete technical blueprint detailing the 7-stage data-mining architecture: Horizons topocentric ephemerides, fixed-sky centering, adaptive cadences, DeepStreaks artifact filters, synthetic injections, in-memory D4 augmentation, and variable-length CSR HDF5 storage.

## Invariants
- Specifications define the structural and mathematical contracts implemented by `src/LUMIA/data_mining/`.

## Side Effects
- None. Static documentation.

## Verification
- Run: `uv run pytest tests/test_data_mining`
