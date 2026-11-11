# Training Architecture & Optimization

## Overview
The `docs/training/` directory contains technical specifications, architectural blueprints, and optimization proposals for Project LUMIA's multi-epoch spatial-temporal tracklet classifier training infrastructure.

## Interface
- [architecture_blueprint.md](./architecture_blueprint.md): Approved technical blueprint specifying tracklet neural network topologies (`ResNetLite`, `BiGRUAggregator`, `TransformerAggregator`), 4-class taxonomy, Focal Loss formulations, and out-of-core CSR HDF5 data loading.
- [optimization_proposal.md](./optimization_proposal.md): Detailed proposal covering memory optimization, mixed-precision training, learning rate schedules, and temperature scaling calibration.

## Invariants
- Specifications define the structural and hyperparameter requirements implemented in `src/LUMIA/models/` and `src/LUMIA/training/`.

## Side Effects
- None. Static architectural specifications.

## Verification
- Run: `uv run pytest tests/test_training`
