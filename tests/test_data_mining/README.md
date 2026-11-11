# Test Data Mining

## Overview
The `tests/test_data_mining/` test package verifies the data mining pipeline, NASA JPL Horizons integration, astronomical archive crawlers, artifact screening, synthetic injection algorithms, and out-of-core PyTorch dataset loaders.

## Interface
- `test_cli_dry_run.py`: Verifies CLI options and dry-run pipeline execution paths for `scripts/run_mining.py`.
- `test_data_mining.py`: Tests `DataMiningPipeline`, `DeepStreaksArtifactFilter`, `FaintSyntheticInjector`, and `HDF5TrainingStorageManager`.
- `test_dataset.py`: Validates PyTorch `ZTFSSODataset` item indexing, sequence length handling, and batch collation.

## Invariants
- Tests use mock HTTP fixtures or local synthetic data to prevent external network dependencies during automated runs.
- Synthetic injection tests verify that injected flux profiles preserve background noise variance.

## Side Effects
- Creates temporary mock HDF5 files and directories within pytest-managed temporary directories (`tmp_path`).

## Verification
- Run: `uv run pytest tests/test_data_mining`
