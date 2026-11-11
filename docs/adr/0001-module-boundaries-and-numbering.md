# ADR-0001: Canonical Module Boundaries and Numbering

* **Status:** Accepted
* **Date:** 2026-11-07
* **Deciders:** Project LUMIA Architecture Team
* **Related Documents:** `docs/lastenheft.md`, `docs/audits/architecture_audit_v1.md` (Finding `P2-1`)

---

## Context

During Phase 1 MVP development, an architectural consistency audit (`docs/audits/architecture_audit_v1.md`, Finding `P2-1`) identified a numbering and scope collision across `docs/lastenheft.md`, the module specifications in `docs/modules/`, and the implemented Python packages:

1. In the initial `docs/lastenheft.md`, **Module 4** combined GPU image subtraction (`ZOGY`), likelihood shift-and-stacking ($\Psi/\Phi$), *and* `HelioLinC` tracklet linking, while **Module 5** was assigned to *Out-of-Core ML Training, HDF5 Cutout Extraction, and Synthetic Mocking*.
2. During implementation, multi-epoch kinematic tracklet linking (`HelioLinC`) was rightly isolated into **Module 5** (`docs/modules/module_05_tracklet_linking.md`, `src/LUMIA/physics/linking.py`), while **Module 7** (`docs/modules/module_07_cnn_classifier.md`, `src/LUMIA/ml/`) was established as the PyTorch CNN Real/Bogus classifier and **Module 10** (`docs/modules/module_10_web_ui.md`, `src/LUMIA/api/`) was established as the FastAPI/React Web-UI.
3. Consequently, `docs/modules/module_04_gpu_processing.md` retained stale `HelioLinC` bullets duplicating Module 5, `docs/lastenheft.md` contradicted `module_05_tracklet_linking.md`, and the *Out-of-Core HDF5 Ground Truth & Synthetic Mocking* pipeline lacked a dedicated module identifier and specification.

Renumbering existing implemented modules (such as shifting Module 7 or Module 10) would introduce unnecessary churn across Python package docstrings, API documentation, and automated test suites.

---

## Decision

We establish a canonical, non-overlapping 10-module architecture across all Project LUMIA specifications and source packages:

| Module | Canonical Name | Specification File | Core Responsibility & Boundary |
| :--- | :--- | :--- | :--- |
| **Module 1** | Config & Profile Management | `docs/modules/module_01_config.md` | Validated telescope hardware profiles, astrometry solver routing, and central CPU/GPU resource allocation (`src/LUMIA/config/`). |
| **Module 2** | Data Engineering & Task Orchestration | `docs/modules/module_02_pipeline.md` | Prefect DAG flows, polyglot persistence repositories (PostgreSQL OLTP + DuckDB/Parquet OLAP), and `SpooledTemporaryFile` in-memory buffering (`src/LUMIA/data_pipeline/`). |
| **Module 3** | Data Ingestion, Astrometry & Photometry | `docs/modules/module_03_astrometry_photometry.md` | Science-ready FITS validation, WCS plate-solving, BJD_TDB light-time correction, and calibrated `photutils` aperture photometry (`src/LUMIA/physics/`). |
| **Module 4** | GPU-Optimized Image Processing & Stacking | `docs/modules/module_04_gpu_processing.md` | Strictly single-/multi-frame GPU tensor processing: optimal `ZOGY` image subtraction, $\Psi/\Phi$ shift-and-stacking, SNR peak extraction, and in-memory 3-channel cutout slicing (`src/LUMIA/physics/`). |
| **Module 5** | Tracklet Linking & HelioLinC | `docs/modules/module_05_tracklet_linking.md` | Strictly multi-epoch kinematic linking: gnomonic space-velocity projection, 4D phase-space clustering, and Solar System acceleration/escape-velocity validation (`src/LUMIA/physics/linking.py`). |
| **Module 6** | Multi-Facility Archival Precovery & Spectral Analysis | *(Phase 4)* | Orbit projection (`OpenOrb`), Virtual Observatory SIA archival queries, and multi-band photometric albedo fusion. |
| **Module 7** | Machine Learning CNN Classifier | `docs/modules/module_07_cnn_classifier.md` | Pure PyTorch neural inference and model definitions: 3-channel (`Science`, `Reference`, `Difference`) VGG/ResNet-Lite Real/Bogus CNNs, U-Net trail masking, and Active Learning fine-tuning (`src/LUMIA/ml/`). |
| **Module 8** | Out-of-Core HDF5 Ground Truth & Synthetic Mocking | `docs/modules/module_08_ground_truth_mocking.md` | Archival MPC cutout extraction into compressed out-of-core HDF5 tensor datasets (with immediate FITS cleanup) and synthetic asteroid injection for completeness characterization. |
| **Module 9** | Known-Object Filtering, Astrophysical Analysis & Archival | *(Phase 3–4)* | Post-linking MPC/SkyBoT known-object rejection, Stellarium live bridge, MOID orbit simulation, light-curve analysis, long-term data retention, and MPC reporting. |
| **Module 10** | Output, Review & Web-UI | `docs/modules/module_10_web_ui.md` | JWT-secured FastAPI REST/WebSocket backend (delegating persistence to Module 2) and dark-mode React candidate blink-loop and Active Learning triage dashboard (`src/LUMIA/api/`, `frontend/`). |

Additionally, strict unidirectional dependency boundaries apply:
* Computational domain modules (**Modules 3, 4, 5, 7, 8**) must remain pure and persistence-agnostic, containing zero SQL, database driver, or HTTP framework logic.
* **Module 2** (`data_pipeline`) owns all database schemas, repository queries, and workflow orchestration wiring.

---

## Consequences

### Positive
* Eliminates all numbering and scope ambiguity between `docs/lastenheft.md` and `docs/modules/`.
* Preserves backward compatibility with all existing codebase identifiers for Modules 1, 2, 3, 4, 5, 7, and 10 while giving **Out-of-Core HDF5 Ground Truth & Synthetic Mocking** a first-class specification as **Module 8**.
* Establishes an unambiguous dataflow pipeline: **Module 3** (calibrated frames/WCS/BJD) -> **Module 4** (ZOGY subtraction & 3-channel stamps) -> **Module 7** (CNN Real/Bogus scores) -> **Module 5** (kinematic `Tracklet` entities) -> **Module 2** (PostgreSQL/DuckDB persistence) -> **Module 10** (human triage) -> **Module 8** (HDF5 Active Learning retraining & completeness mocking).

### Negative / Trade-offs
* Consolidating Known-Object Rejection/Stellarium Integration and Astrophysical Analysis/MPC Archival into **Module 9** broadens Module 9's scope across ephemeris cross-matching and post-discovery reporting; if necessary in Phase 4, Module 9 can be subdivided into internal submodules (`filtering` and `reporting`) under the same top-level module number.
