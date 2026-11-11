# Test Training

## Overview
The `tests/test_training/` test package verifies the deep learning training pipeline, spatial-temporal tracklet model backbones, loss functions, evaluation metrics, atomic checkpointing, and post-hoc temperature calibration.

## Interface
- `test_calibration.py`: Tests temperature scaling calibration optimization and reliability diagram metrics.
- `test_config_and_cli.py`: Validates Pydantic training configurations and command-line argument parsing.
- `test_loss.py`: Tests cross-entropy, Focal Loss, and Class-Balanced Focal Loss on imbalanced class distributions.
- `test_metrics_and_checkpointing.py`: Verifies mAP, Macro-F1 computation, and atomic checkpoint saving/restoration.
- `test_tracklet_models.py`: Validates forward passes for `TrackletClassifier`, `ResNetLite`, `BiGRUAggregator`, and `TransformerAggregator`.
- `test_trainer.py`: Integration tests for training loops, validation steps, and early stopping triggers.

## Invariants
- Forward and backward pass tests check for non-zero gradients across all trainable parameter tensors.
- Checkpointing tests verify atomic writes preventing corruption during process interruption.

## Side Effects
- Allocates CPU and GPU tensors during training loop execution.
- Reads and writes temporary test checkpoint files in temporary directories.

## Verification
- Run: `uv run pytest tests/test_training`
