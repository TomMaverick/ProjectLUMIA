# Scripts

## Overview
The `scripts/` package contains executable command-line interfaces, data ingestion utilities, training drivers, and diagnostic inspection tools for Project LUMIA. These entry points provide automated workflows for running pipeline stages, training neural network classifiers, and inspecting tracklet detections.

## Interface
- `evaluate_model.py`: Evaluates saved model checkpoints against test datasets, reporting classification metrics and ROC/PR statistics.
- `import_tomoe_data.py`: Ingests and formats Tomo-e Gozen optical survey image archives into standardized HDF5 datasets.
- `inspect_tracklets.py`: CLI utility for inspecting, plotting, and diagnosing multi-epoch candidate tracklet sequences and cutouts.
- `run_mining.py`: Execution driver for the automated solar system data-mining and synthetic augmentation pipeline (`LUMIA.data_mining`).
- `train_model.py`: CLI entry point for training and evaluating single-epoch Real/Bogus CNN models (`LUMIA.ml`).
- `train_tracklets.py`: CLI entry point for training multi-epoch spatial-temporal tracklet models (`LUMIA.training`).
- `tune_model.py`: Hyperparameter optimization script for tuning model architectures and training hyperparameters.

## Invariants
- Scripts parse arguments strictly using `argparse` or Pydantic configurations.
- Deterministic random seeding is enforced across execution scripts to guarantee reproducibility.
- Scripts execute using project-level dependencies via `uv run python scripts/<script_name>.py`.

## Side Effects
- Reads raw image data, configuration files, and saved model weights from the local filesystem.
- Writes training checkpoints, evaluation summaries, and generated HDF5 datasets to disk.
- Logs execution progress to standard output and diagnostic log files.

## Verification
- Run: `uv run python scripts/evaluate_model.py --help`
- Run: `uv run python scripts/run_mining.py --help`
