# Test ML

## Overview
The `tests/test_ml/` test package verifies Module 7 (Machine Learning CNN Classifier). It tests model architecture initialization, deterministic forward passes, robust MAD normalization, batch inference partitioning, and CUDA out-of-memory handling.

## Interface
- `test_cnn_classifier.py`: Validates `RealBogusClassifier`, `RealBogusNet`, `preprocess_triplets`, decision threshold filtering, and score attachment to `CandidateDetection` instances.

## Invariants
- Tests assert that forward pass outputs are deterministic when initialized with identical RNG seeds.
- Verifies that `CutoutClassificationBatchResult` properly partitions accepted and rejected candidate IDs.

## Side Effects
- Allocates CPU and GPU tensors during forward inference passes.
- Saves and loads temporary model checkpoint weights for testing.

## Verification
- Run: `uv run pytest tests/test_ml`
