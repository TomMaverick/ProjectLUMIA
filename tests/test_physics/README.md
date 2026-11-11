# Test Physics

## Overview
The `tests/test_physics/` test package verifies the mathematical and physical algorithms across Modules 1, 3, 4, and 5. It validates WCS astrometry, aperture photometry, ZOGY image subtraction, likelihood shift-and-stacking, candidate peak slicing, HelioLinC kinematic tracklet linking, and synthetic asteroid injection.

## Interface
- `test_astrometry_photometry.py`: Tests WCS coordinate transforms, BJD_TDB light-time corrections, and aperture photometry zero-points.
- `test_cutout_extraction.py`: Validates deterministic SNR peak detection and 3-channel cutout extraction (`CutoutBatchResult`).
- `test_gpu_processing.py`: Tests GPU ZOGY subtraction, significance map generation, and $\Psi/\Phi$ shift-and-stacking.
- `test_mpc_client.py`: Validates Minor Planet Center client requests and response parsing using mock network fixtures.
- `test_synthetic_injection.py`: Tests synthetic point source convolving and PSF flux injection.
- `test_telescope_config.py`: Validates optical specifications and telescope site coordinates.
- `test_tracklet_linking.py`: Verifies HelioLinC clustering, space-velocity projections, and kinematic velocity bounds.

## Invariants
- Numerical physics assertions enforce conservation of flux and astrometric closure (pixel to sky and back).
- Tests enforce that the physics package remains completely decoupled from database models and web libraries.

## Side Effects
- Allocates CPU and CUDA memory for numerical grid processing and FFTs.

## Verification
- Run: `uv run pytest tests/test_physics`
