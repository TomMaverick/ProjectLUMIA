# LUMIA Training

## Overview
LUMIA Training provides the complete model training, evaluation, validation, post-hoc temperature scaling calibration, and checkpointing framework for Project LUMIA tracklet classifiers. It features class-imbalance robust loss functions (Focal Loss, Class-Balanced Loss), learning rate scheduling, and multi-metric tracking.

## Interface
- `TrackletModelConfig`: Validated schema configuring spatial backbones, temporal aggregators, embedding dimensions, and dropout rates.
- `TrackletTrainingConfig`: Comprehensive Pydantic configuration covering batch sizes, learning rates, weight decay, loss types, and early stopping patience.
- `TrackletTrainer`: Core trainer class handling training and validation loops, backpropagation, and metric evaluation.
- `FocalLoss` / `ClassBalancedFocalLoss`: Loss functions mitigating extreme dataset imbalance between common bogus detections and rare real moving objects.
- `TrackletMetricsTracker`: Metric evaluation utility calculating multi-class Accuracy, Macro-F1, per-class Average Precision, and mAP.
- `TemperatureScaler`: Post-training calibration module optimizing a single temperature parameter using validation set logits.
- `TopKModelCheckpointer`: Checkpoint persistence utility saving top-K model state dicts, optimizer states, and epoch metadata.

## Invariants
- `TrackletTrainingConfig`: Requires `batch_size >= 1`, `learning_rate > 0.0`, and recognized loss function specifiers (`cross_entropy`, `focal`, `class_balanced_focal`).
- Targets passed to loss functions must be integer class indices in the range `[0, num_classes - 1]`.
- Checkpoints are saved atomically using temporary file writes followed by file replacements to guard against corruption during interruptions.

## Side Effects
- Reads and writes PyTorch checkpoint binaries (`.pt`) in the designated checkpoint directory on disk.
- Updates model weights and optimizer internal states during training iterations.
- Logs training and validation metrics to standard output and logger streams.

## Verification
- Run: `uv run pytest tests/test_training`
