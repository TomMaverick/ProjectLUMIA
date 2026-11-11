# Canonical Module Specifications

## Overview
The `docs/modules/` directory contains the detailed engineering specifications and interface contracts for the ten canonical modules comprising Project LUMIA (ADR-0001).

## Interface
- [module_01_config.md](./module_01_config.md): Module 1 – Configuration, telescope profiles, and central CPU/GPU resource allocation.
- [module_02_pipeline.md](./module_02_pipeline.md): Module 2 – Prefect task orchestration, polyglot persistence (DuckDB + PostgreSQL), and ingestion staging.
- [module_03_astrometry_photometry.md](./module_03_astrometry_photometry.md): Module 3 – WCS plate-solving, BJD_TDB light-time correction, and aperture photometry.
- [module_04_gpu_processing.md](./module_04_gpu_processing.md): Module 4 – GPU-accelerated ZOGY image subtraction, $\Psi/\Phi$ shift-and-stacking, and cutout slicing.
- [module_05_tracklet_linking.md](./module_05_tracklet_linking.md): Module 5 – Two-Branch HelioLinC3D engine (stereoscopic triangulation across Earth-space baselines and universal Keplerian dynamical association) with SNR-weighted astrometric gating.
- [module_07_cnn_classifier.md](./module_07_cnn_classifier.md): Module 7 – Pure PyTorch Real/Bogus CNN inference, Two-Stream adaptation heads, and U-Net artifact masking.
- [module_08_ground_truth_mocking.md](./module_08_ground_truth_mocking.md): Module 8 – Out-of-core HDF5/Zarr ground truth curation, multi-survey SODA harvesting, and worker-partitioned storage.
- [module_10_api.md](./module_10_api.md): Module 10 – FastAPI REST gateway, Argon2/JWT authentication, and real-time WebSocket event streaming.
- [module_10_web_ui.md](./module_10_web_ui.md): Module 10 – React / TypeScript dark-mode web console for candidate triage and Interactive Multi-Survey Blink Viewer with FOV physical scale normalization.

## Invariants
- Module specifications define the non-overlapping boundaries, input/output contracts, and validation rules for each subsystem.

## Side Effects
- None. Static documentation records.

## Verification
- Run: `uv run pytest tests/test_security_and_boundaries.py`
