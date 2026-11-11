# LUMIA Core

## Overview
LUMIA Core defines the canonical cross-module domain contracts, strict Pydantic schemas, deterministic PyTorch runtime configuration, and global CPU/GPU resource management for Project LUMIA. It serves as the foundational substrate shared across physics, machine learning, data engineering, and API presentation layers without depending on external persistence engines or web frameworks.

## Interface
- `LumiaBaseModel`: Canonical base model enforcing strict typing, immutability (`frozen=True`), forbidden extra fields, and rejection of non-finite floats (`allow_inf_nan=False`).
- `CandidateDetection`: Authoritative moving-object candidate detection schema containing celestial coordinates, light-time corrected BJD_TDB, astrometric error metrics, survey/filter metadata, observer state vectors, and optional CNN classification scores.
- `GroundObservatoryProfile` / `SpacecraftObservatoryProfile`: Discriminated observatory metadata models specifying geocentric coordinates or NAIF SPICE body IDs.
- `SNRPeakCandidate`: Deterministic significance peak candidate extracted from difference imaging significance maps.
- `CutoutBatchResult`: Validated batch container of aligned `(Science, Reference, Difference)` tensor triplets bridging Module 4 image processing to Module 7 machine learning.
- `ClassificationResult`: Validated Real/Bogus prediction container linking candidates to confidence scores and model version provenance.
- `KnownObjectEphemeris`: Ephemeris contract for known solar system bodies from MPC or JPL Horizons.
- `GroundTruthBatch`: Validated out-of-core batch container of 3-channel cutouts, binary labels, and aligned celestial metadata.
- `ResourceBudgetConfig`: Validated resource configuration managing CPU thread budgets, VRAM allocations, and archive paths.
- `configure_deterministic_torch`: Seeds global RNGs, forces cuDNN determinism, and enables deterministic PyTorch execution.
- `resolve_torch_device`: Validates and parses device specifiers (`cpu`, `cuda`, `cuda:N`, `auto`).
- `configure_global_runtime`: Enforces CPU thread limits across PyTorch, OpenMP, MKL, OpenBLAS, OpenCV, and restricts CUDA memory fractions.
- `get_active_resource_budget`: Retrieves the currently active `ResourceBudgetConfig` singleton.

## Invariants
- All domain schemas inherit from `LumiaBaseModel` and enforce zero type coercion, immutability, and rejection of `NaN` or `+/-Inf`.
- `CandidateDetection`: Setting `model_version` requires an accompanying `cnn_score` (`_model_version_requires_score`). Coordinates must satisfy `0.0 <= ra_deg < 360.0` and `-90.0 <= dec_deg <= 90.0`.
- `CutoutBatchResult`: Triplet tensor shape must be `(B, 3, stamp_size, stamp_size)` with odd `stamp_size` (31 or 51). Tensor elements must be strictly finite. Batch size must match candidate count with no duplicate `candidate_id` values.
- `GroundTruthBatch`: Triplet array must have shape `(N, 3, 31, 31)` with `float32` dtype and finite values; labels must be `(N,)` with `int8` dtype; required metadata keys (`bjd_tdb`, `ra_deg`, `dec_deg`, `candidate_id`) must have length `N`.
- `resolve_torch_device`: Rejects invalid strings with `ValueError` and raises `RuntimeError` if CUDA is requested when unavailable.

## Side Effects
- `configure_deterministic_torch`: Seeds `random`, `np.random`, `torch`, and `torch.cuda` RNGs; sets `CUBLAS_WORKSPACE_CONFIG` in `os.environ`; mutates `torch.backends.cudnn.deterministic` and `torch.use_deterministic_algorithms`.
- `configure_global_runtime`: Modifies CPU threading state via `torch.set_num_threads`, `cv2.setNumThreads`, and environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`); adjusts CUDA per-process memory limits; creates filesystem directory at `archive_storage_path`.

## Verification
- Run: `uv run pytest tests/test_core`
