# Project LUMIA: Design Specifications Index (Sitemap)

## Overview
This document serves as the central index and registry for all Elephant-and-Goldfish Model (EGM) technical design documents, RFCs, and subsystem specifications maintained in the `docs/designs/` directory.

All new design documents produced via `egm-brainstorm`, `egm-draft`, and `egm-review` must be registered in this index with their target subsystem, status, and tracking dates.

---

## Specification Lifecycle

| State | Definition |
| :--- | :--- |
| **DRAFT** | Initial design under active drafting via `egm-draft`. Incomplete sections allowed. |
| **IN_REVIEW** | Draft completed; undergoing multi-lens Goldfish review and human inspection. |
| **APPROVED** | Formally accepted architecture blueprint ready for implementation. |
| **IMPLEMENTED** | Architecture blueprint fully implemented in codebase and verified by automated tests. |
| **SUPERSEDED** | Replaced by a newer specification; preserved for historical audit trails. |

---

## Active & Related Architecture Blueprints

### Subsystem A: Single-Epoch Science Ingestion & Pipeline Gateway
* [Module 01: Config Management](../../docs/modules/module_01_config.md) – `APPROVED` – Optical parameters, telescope routing, and resource budgets.
* [Module 02: Data Engineering & Pipeline](../../docs/modules/module_02_pipeline.md) – `APPROVED` – Prefect orchestration and polyglot persistence.
* [Module 03: Astrometry & Photometry](../../docs/modules/module_03_astrometry_photometry.md) – `APPROVED` – WCS plate-solving, BJD_TDB, and aperture photometry.
* [Module 04: GPU Processing](../../docs/modules/module_04_gpu_processing.md) – `APPROVED` – ZOGY image subtraction and $\Psi/\Phi$ shift-and-stacking.
* [Module 05: Tracklet Linking](../../docs/modules/module_05_tracklet_linking.md) – `APPROVED` – Kinematic HelioLinC clustering and velocity bounds.
* [Module 07: Real/Bogus CNN Classifier](../../docs/modules/module_07_cnn_classifier.md) – `APPROVED` – Single-epoch 3-channel cutout classification.
* [Module 10: Web API Gateway](../../docs/modules/module_10_api.md) – `APPROVED` – FastAPI REST and WebSocket telemetry gateway.
* [Module 10: Frontend UI/UX](../../docs/frontend/ui_design_specification.md) – `APPROVED` – Dark-mode observatory ergonomics and triage dashboard.

### Subsystem B: Multi-Epoch ZTF Tracklet Subsystem
* [Data-Mining & Curation Blueprint](../../docs/data_mining/architecture_blueprint.md) – `APPROVED` – Variable-length tracklet harvesting, JPL Horizons, and CSR HDF5 archives.
* [Multi-Epoch Tracklet Training Blueprint](../../docs/training/architecture_blueprint.md) – `APPROVED` – Spatial-temporal sequence neural architectures and focal loss optimization.
* [Training Optimization Proposal](../../docs/training/optimization_proposal.md) – `APPROVED` – Mixed precision, learning rate schedules, and temperature scaling.
* [`ztf_tracklet_real_data_migration.md`](ztf_tracklet_real_data_migration.md) – `IMPLEMENTED` – IPAC IBE cutout harvesting, CSR HDF5 v2.1.0, 144x144 dynamic jitter, GroupNorm ResNetLite, and 4-class classification.
* [`des_005_spacecraft_ephemerides_and_multi_survey_fusion.md`](des_005_spacecraft_ephemerides_and_multi_survey_fusion.md) – `IMPLEMENTED` – NAIF SPICE kernels, Planetary Retarded-Time Solver, Two-Branch HelioLinC3D (Stereoscopic Triangulation & Keplerian Dynamical Association), Multi-Survey SODA ingestion (Pan-STARRS & Rubin), Worker-Partitioned Zarr v3 storage, Active DIA artifact mining, Two-Stream adaptation head, and React Multi-Survey Blink Viewer with FOV scale normalization (605 tests verified).

---

## Upcoming EGM Design Specifications (`docs/designs/`)

The following designs are scheduled for exploration and drafting:

| Design Document | Subsystem / Domain | Target Scope | Status |
| :--- | :--- | :--- | :--- |
| [`des_001_completeness_mapping.md`](des_001_completeness_mapping.md) | Subsystem B (Physics/ML) | Automated completeness mapping and synthetic injection recovery benchmarking. | `IN_REVIEW` |
| `des_002_prefect_production_dag.md` | Subsystem A (Data Pipeline) | Full production Prefect 3.x deployment topology, worker queues, and artifact retention. | `PLANNED` |
| `des_003_mpc_submission_reporting.md` | Subsystem A (Physics/API) | Automated Minor Planet Center (MPC) 80-column and ADES format export pipeline. | `PLANNED` |
| `des_004_active_learning_loop.md` | Cross-Subsystem (UI/ML) | Human candidate triage feedback loop retraining tracklet classifiers from HDF5. | `PLANNED` |
| [`des_005_spacecraft_ephemerides_and_multi_survey_fusion.md`](des_005_spacecraft_ephemerides_and_multi_survey_fusion.md) | Subsystem A/B (Physics/Astrometry) | NAIF SPICE kernels, Sun-Earth L2 parallax baselines, BCRS orbital linking, multi-survey adapters (Rubin, Roman, JWST), Zarr v3, and sub-5-sigma discovery. | `IMPLEMENTED` |

---

## Architectural Invariants & Governance

1. **Document Language:** All specifications must be written in English.
2. **Emoji Prohibition:** Emojis are strictly forbidden in all design documents and specifications.
3. **Contract Adherence:** All interfaces specified in design docs must use strict Pydantic v2 schemas adhering to `ADR-0002`.
4. **Traceability:** Every new design file in `docs/designs/` must cite its upstream ADRs and register a corresponding entry in this sitemap.
