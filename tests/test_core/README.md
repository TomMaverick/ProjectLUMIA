# Test Core

## Overview
The `tests/test_core/` test package verifies the fundamental domain contracts, strict Pydantic models, finite floating-point policies, PyTorch runtime determinism, and central concurrency budgeting of Project LUMIA.

## Interface
- `test_concurrency.py`: Verifies thread allocation enforcement across PyTorch, OpenMP, MKL, OpenCV, and GPU memory ceiling settings.
- `test_core_schemas.py`: Tests `CandidateDetection`, `SNRPeakCandidate`, `CutoutBatchResult`, and other canonical models against validation rules and immutability invariants.
- `test_finite_policy.py`: Verifies strict rejection of non-finite floats (`NaN`, `+Inf`, `-Inf`) across all domain models (ADR-0002).
- `test_packaging_and_metadata.py`: Validates package metadata, entry points, and schema exports.
- `test_runtime.py`: Verifies deterministic RNG initialization, cuDNN deterministic configurations, and device resolution (`cpu`, `cuda`, `auto`).

## Invariants
- Tests ensure that any model receiving invalid types or non-finite numbers raises `ValidationError` immediately.
- Global RNG tests restore determinism settings upon test teardown.

## Side Effects
- Mutates and tests thread allocation settings and temporary environment variables.

## Verification
- Run: `uv run pytest tests/test_core`
