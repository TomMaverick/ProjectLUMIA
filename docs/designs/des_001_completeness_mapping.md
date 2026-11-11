# RFC des_001: Automated Completeness Mapping & Synthetic Recovery Benchmarking

* **Author:** TomMaverick
* **Status:** IN_REVIEW (Goldfish Swarm Review Synthesis)
* **Date:** 2026-11-10
* **Target Scope:** Subsystem B (Physics / Multi-Epoch ML), Module 4 (Shift-and-Stack Likelihood), Module 5 (HelioLinC3D Linking), Module 7 (Two-Stream Adaptation Head), `src/LUMIA/benchmarking/` (Evaluation Orchestrator)
* **Governance Alignment:** `ADR-0001` (System Boundaries), `ADR-0002` (Strict Pydantic v2 Contracts), `ADR-0003` (Polyglot Persistence)

--------------------------------------------------------------------------------

## 1. Overview

Project LUMIA (Low-SNR Unidentified Moving Interplanetary Analyzer) has achieved multi-survey data ingestion, spacecraft ephemeris handling in the Barycentric Celestial Reference System (BCRS), and kinematic linking down into the sub-5-sigma regime ($1.5\sigma \le SNR < 5.0\sigma$) via `des_005`. Operating below $5\sigma$ introduces severe signal contamination risks from False Discovery Rate (FDR) inflation, optical subtraction artifacts (dipoles, bleeding columns), and false-association combinatorial explosions.

`des_001` establishes the architectural blueprint for an **Automated Completeness Mapping & Synthetic Recovery Benchmarking Harness** (`src/LUMIA/benchmarking/`). This subsystem provides an automated, statistically rigorous Monte Carlo injection-and-recovery test pipeline that:

1. Injects synthetic moving asteroids into real calibrated difference images and multi-epoch cutouts using closed-form analytical error-function (erf) motion trailing convolution, bivariate Moffat seeing profiles, and Poisson detector shot noise.
2. Characterizes the two-dimensional detection efficiency surface $\epsilon(V, \mu)$ across apparent visual magnitudes ($18.0 \le V \le 23.5$) and celestial motion rates ($\mu \in [0.05''/\text{hr}, 10.0^\circ/\text{day}]$), accounting for intra-exposure PSF trailing flux dilution.
3. Decouples failure attribution across four discrete pipeline gates:
   $$\text{Pixels} \xrightarrow{\epsilon_{\text{ext}}} \text{Extractions} \xrightarrow{\epsilon_{\text{stack}}} \text{Candidates} \xrightarrow{\epsilon_{\text{ml}}} \text{Classified Detections} \xrightarrow{\epsilon_{\text{link}}} \text{Linked Orbits}$$
4. Emits a compact, tabular rejected-objects catalog (`rejected_candidates.parquet`) referencing sharded Zarr v3 cutouts via zero-copy slice URIs, completely avoiding intermediate image disk writes.
5. Bridges directly to an interactive manual triage CLI and web viewer, outputting cryptographically verified annotations that feed the upcoming Active Learning retraining loop (`des_004`).
6. Derives empirical 50% ($V_{50}$) and 90% ($V_{90}$) limiting recovery curves with field-level clustered bootstrap confidence intervals, benchmarking the performance delta of the Two-Stream Adaptation Head and SNR-weighted HelioLinC3D gating against historical baselines.

--------------------------------------------------------------------------------

## 2. The Problem Statement

Operating a moving object detection pipeline in the low-SNR regime ($1.5\sigma \le SNR < 5.0\sigma$) exposes the system to five fundamental vulnerabilities:

1. **Uncalibrated Sub-5-Sigma Horizons & Synthetic Detection Bias:**
   Detection horizons cannot be estimated from background noise variance alone. Moreover, injecting synthetic sources with idealized Gaussian PSFs into Difference Image Analysis (DIA) cutouts concentrates excess flux in the central pixel, creating a synthetic detection bias of $0.15\text{--}0.35$ magnitudes. Injections must model realistic Moffat seeing wings ($\beta \approx 3.5$) and convolve with the difference image matched filter kernel ($P_D$).
2. **Phase Space Sampling vs. Physical Orbital Dynamics:**
   A naive Cartesian grid over $(V, \mu)$ samples unphysical empty regions (e.g., $V=18.0$ at $\mu=0.05''/\text{hr}$ implies a body larger than Pluto at $40\text{ AU}$, while $V=18.0$ at $10^\circ/\text{day}$ implies a bolide at lunar distance), while under-sampling the dominant orbital manifolds of Main Belt Asteroids ($25''\text{--}45''/\text{hr}$).
3. **Circular Synthetic Leakage in ML Evaluation:**
   Evaluating the Two-Stream Adaptation Head against the same generative assumptions embedded in `SyntheticInjector` measures neural memorization of the generator's inductive bias rather than astrophysical recovery. The benchmark requires generative disjointness (Moffat profiles, rotational light curves with tumbling periods $P \in [2, 12]\text{ hr}$, and empirical high-SNR anchor sets).
4. **SSD Endurance Destruction & Windows Concurrency Crashes:**
   A brute-force run of 69,000 synthetic asteroids across 5 epochs generates $> 1.45\text{ TB}$ of ephemeral image writes if intermediate frames are saved, destroying SSD write endurance (TBW) in 12-18 months. Furthermore, multi-process workers opening monolithic HDF5 files on Windows NTFS inevitably trigger mandatory file locking violations (`WinError 32`).
5. **Coupled Multi-Stage Attribution Ambiguity:**
   In `TwoStreamAdaptationHead`, spatial features (`feat_a`, 128-d) and temporal features (`feat_b`, 64-d) are concatenated into a joint non-linear MLP. Individual stream probabilities do not exist natively. Attributing ML failure strictly to Stream A or Stream B without auxiliary classification heads or counterfactual swap testing introduces severe attribution bias.

--------------------------------------------------------------------------------

## 3. Goals and Non-Goals

### 3.1 Goals
* **G1 (Vectorized Multi-Mode Benchmark Engine):** Implement `src/LUMIA/benchmarking/` supporting multi-mode execution: fast deterministic smoke tests (< 60s), Active Sigmoidal Uncertainty Search (~3-5m), and offline population-weighted sweeps.
* **G2 (Astrophysical & Optical Fidelity):** Replace discrete quadrature stepping with closed-form analytical error-function trailing convolution; support bivariate Moffat seeing profiles ($\beta = 3.5$) and directional Differential Chromatic Refraction (DCR).
* **G3 (Spatial Multiplexing):** Support injecting $K = 10\text{--}15$ non-interfering synthetic sources per cutout using Poisson disk sampling ($d_{\min} \ge 50\text{ px}$), amortizing image read and stacking overhead by $>10\times$.
* **G4 (Strict Staged Attribution):** Track candidate fates across four gates (`FAILED_EXTRACTION`, `FAILED_STACKING`, `FAILED_ML`, `FAILED_LINKING`, `RECOVERED`).
* **G5 (Two-Stream Disentanglement):** Equip `TwoStreamAdaptationHead` with calibrated auxiliary linear heads or counterfactual swap testing to mathematically isolate Stream A vs Stream B failure modes.
* **G6 (Analytical Rejected Objects Catalog):** Emit `rejected_candidates.parquet` containing tabular ground truth and diagnostics, referencing sharded Zarr v3 cutouts via zero-copy slice URIs. Zero ephemeral images written to disk.
* **G7 (Manual Triage Bridge):** Emit `triage_manifest.json` for React UI / CLI triage, generating cryptographically hashed `TriageAnnotationRecord` entries feeding Active Learning (`des_004`).
* **G8 (Bivariate Completeness Modeling):** Model $\epsilon(V, \mu)$ via Bivariate Generalized Additive Models (GAM) and generalized Richards' curves with asymptotic recovery ceiling ($\epsilon_{\max} < 1.0$) and field-level clustered bootstrap confidence intervals.
* **G9 (Governance & Concurrency):** Enforce strict Pydantic v2 schemas (`ADR-0002`), Zarr v3 sharded storage (`ADR-0003`), and Level 4 package hierarchy placement (`ADR-0001`).

### 3.2 Non-Goals
* **NG1 (Git Binary Bloat):** Committing multi-gigabyte static cutout datasets directly into Git or Git LFS is strictly prohibited.
* **NG2 (Production Database Pollution):** Synthetic benchmark runs will never write records to production PostgreSQL or DuckDB databases.
* **NG3 (Real-Time Ingestion Interception):** Synthetic injection will not run inside production live-stream telescope ingestion pipelines.
* **NG4 (Non-Gravitational Trajectory Dynamics):** Modeling cometary outgassing or Yarkovsky acceleration is out of scope for recovery certification.

--------------------------------------------------------------------------------

## 4. Execution Strategies & Storage Architecture

### 4.1 Swarm-Synthesized Execution Models
Following the Goldfish review, four execution models are formulated for author selection:

* **Model 1: Active Sigmoidal Uncertainty Search (ASUS)**
  * Replaces uniform 23-bin grids with sequential two-phase active sampling.
  * Phase 1 evaluates a coarse exploratory grid ($N=15$, 750 injections). Phase 2 concentrates 80% of remaining trials within $\pm 0.75\text{ mag}$ of the estimated transition threshold $V \in [V_{90}, V_{50}]$, where binomial variance is maximized.
  * *Result:* 80-90% reduction in total trials (~6,900 injections) with equal or tighter confidence intervals. Full run finishes in ~90 seconds on GPU.
* **Model 2: Stratified Multi-Source In-Memory Multiplexing**
  * Injects $K = 15$ synthetic asteroids simultaneously into a single $512 \times 512$ cutout field using Poisson disk spatial sampling ($d_{\min} \ge 50\text{ px}$).
  * *Result:* Amortizes I/O and stacking overhead by $15\times$, eliminating the need to run 69,000 isolated pipeline invocations.
* **Model 3: Two-Tier Factorized Emulation (TTFE)**
  * Decouples the benchmark into a *Pixel Transfer Tier* (local cutouts parameterizing extraction/ML probabilities) and a *Clutter Kinematic Tier* (catalog-level trajectory propagation co-mingled with real DIA false-alarm clutter in DuckDB/Polars).
  * *Result:* Kinematic linking can evaluate $10^6$ trajectories in seconds without image I/O.
* **Model 4: Synthesized Tri-Modal Pipeline (Recommended)**
  * Combines Model 1 and Model 2:
    1. *PR CI (`--smoke-test`):* Pinned deterministic vectors ($N=25$, < 60s). Zero binomial variance; any delta is code regression.
    2. *Nightly CI (`--active-bisection`):* ASUS adaptive bisection with spatial multiplexing (~3-5m), providing full mathematical certification of $V_{50}$ and $V_{90}$.
    3. *Release Milestone (`--population-weighted`):* Full population-weighted sweep drawing $(a, e, i, H)$ from Granvik/Grav orbital distributions.

### 4.2 Three-Tier Storage Topology & Windows Concurrency Defense
To eliminate Windows NTFS `WinError 32` file locking collisions and prevent Git bloat:
1. **Tier 1 (CI Regression Fixtures):** Lightweight deterministic micro-cutouts ($\le 3\text{ MB}$) committed to `tests/fixtures/benchmarks/` in Git.
2. **Tier 2 (Remote Canonical Archives):** Multi-gigabyte canonical datasets (`ztf_curated_benchmarks.zarr`, `rubin_curated_benchmarks.zarr`) stored on an external S3-compatible / Zenodo store, versioned with SHA-256 integrity digests in `curated_manifest.json`.
3. **Tier 3 (Local Cache-Aside):** Downloaded locally via `lumia benchmark fetch-data --telescope <name>` into `$LUMIA_CACHE_DIR/benchmarks/curated/` (strictly `.gitignore`d).
4. **Zarr v3 Format Standard:** Static cutouts use sharded Zarr v3 with Blosc/Zstandard compression, permitting lock-free concurrent memory-mapped chunk reads across multi-process workers on Windows.

### 4.3 Inversion of Control: Telescope Benchmark Adapter
To decouple the benchmark runner from observatory-specific file formats, all cutouts are ingested via the `TelescopeBenchmarkAdapter` protocol:

```python
from typing import Protocol, runtime_checkable
import torch
from LUMIA.core.schemas import LumiaBaseModel

class CalibratedFrameTriplet(LumiaBaseModel):
    """Normalized in-memory representation of multi-epoch science observations."""
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, arbitrary_types_allowed=True)
    
    science: torch.Tensor       # Shape (H, W), dtype float32
    reference: torch.Tensor     # Shape (H, W), dtype float32
    difference: torch.Tensor    # Shape (H, W), dtype float32
    variance: torch.Tensor      # Shape (H, W), dtype float32
    mask: torch.Tensor          # Shape (H, W), dtype int32
    
    zero_point: float
    pixel_scale_arcsec: float
    exposure_time_sec: float
    gain_e_per_adu: float
    saturation_adu: float
    bjd_tdb: float
    wcs_header_json: str

@runtime_checkable
class TelescopeBenchmarkAdapter(Protocol):
    """Abstract contract decoupling the benchmark runner from physical data formats."""
    def get_telescope_name(self) -> str: ...
    def load_triplet(self, sample_id: str) -> CalibratedFrameTriplet: ...
    def list_curated_samples(self) -> tuple[str, ...]: ...
    def sky_to_pixel(self, triplet: CalibratedFrameTriplet, ra_deg: float, dec_deg: float) -> tuple[float, float]: ...
    def pixel_to_sky(self, triplet: CalibratedFrameTriplet, x: float, y: float) -> tuple[float, float]: ...
```

--------------------------------------------------------------------------------

## 5. Attribution, Rejected Objects Catalog & Manual Triage Loop

### 5.1 Strict Pydantic v2 Contracts (ADR-0002)

```python
from enum import StrEnum
from typing import Literal
from pydantic import Field
from LUMIA.core.schemas import LumiaBaseModel

class BenchmarkFailureStage(StrEnum):
    FAILED_EXTRACTION = "FAILED_EXTRACTION"
    FAILED_STACKING = "FAILED_STACKING"
    FAILED_ML = "FAILED_ML"
    FAILED_LINKING = "FAILED_LINKING"
    RECOVERED = "RECOVERED"

class InjectedGroundTruth(LumiaBaseModel):
    synthetic_id: str = Field(..., pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    magnitude_v: float = Field(..., ge=10.0, le=30.0)
    apparent_rate_arcsec_hr: float = Field(..., ge=0.01, le=1000.0)
    position_angle_deg: float = Field(..., ge=0.0, lt=360.0)
    trail_length_px: float = Field(..., ge=0.0)
    injected_snr: float = Field(..., gt=0.0)
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)

class StageDiagnostics(LumiaBaseModel):
    measured_snr: float | None = Field(default=None, gt=0.0)
    stream_a_morphology_score: float | None = Field(default=None, ge=0.0, le=1.0)
    stream_b_temporal_score: float | None = Field(default=None, ge=0.0, le=1.0)
    two_stream_fused_p: float | None = Field(default=None, ge=0.0, le=1.0)
    heliolinc_chi2: float | None = Field(default=None, ge=0.0)
    heliolinc_dof: int | None = Field(default=None, ge=1)
    astrometric_residual_arcsec: float | None = Field(default=None, ge=0.0)

class CutoutReferencePointer(LumiaBaseModel):
    telescope: Literal["ZTF", "PANSTARRS", "RUBIN", "JWST", "ROMAN"]
    dataset_relative_path: str = Field(..., pattern=r"^[A-Za-z0-9_/.-]+$")
    slice_index: int = Field(..., ge=0)

class SyntheticInjectionFailureRecord(LumiaBaseModel):
    run_id: str = Field(..., min_length=1, max_length=128)
    ground_truth: InjectedGroundTruth
    failure_stage: BenchmarkFailureStage
    diagnostics: StageDiagnostics
    cutout_ref: CutoutReferencePointer

class TriageActionType(StrEnum):
    ACCEPT_AS_HARD_POSITIVE = "ACCEPT_AS_HARD_POSITIVE"
    CONFIRM_ARTIFACT = "CONFIRM_ARTIFACT"
    AMBIGUOUS_SKIP = "AMBIGUOUS_SKIP"

class TriageAnnotationRecord(LumiaBaseModel):
    """Immutable human review disposition directly consumed by des_004 Active Learning."""
    annotation_id: str = Field(..., pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    synthetic_id: str = Field(...)
    run_id: str = Field(...)
    operator_username: str = Field(..., min_length=1, max_length=64)
    action: TriageActionType
    target_class_override: int | None = Field(default=None, ge=0, le=3)
    timestamp_utc: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
    sha256_provenance: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
```

### 5.2 Two-Stream Failure Attribution Disentanglement
To resolve the joint feature concatenation in `TwoStreamAdaptationHead`, the benchmark employs two complementary mechanisms:
1. **Auxiliary Readout Heads:** During training, linear projection heads evaluate `feat_a` and `feat_b` independently against auxiliary classification loss:
   $$\mathcal{L} = \mathcal{L}_{\text{fused}} + 0.3 \mathcal{L}_{\text{aux},A} + 0.3 \mathcal{L}_{\text{aux},B}$$
   providing calibrated marginal scores `stream_a_morphology_score` and `stream_b_temporal_score`.
2. **Counterfactual Swap Testing:** For candidates failing at `FAILED_ML`, the harness pairs the real spatial stamp with a noise-free synthetic temporal profile. If the candidate recovers, the failure was purely temporal (Stream B). If it still fails, spatial morphology (Stream A) was the primary cause.

### 5.3 Parquet Storage & Triage Bridge
* `rejected_candidates.parquet` is partitioned by `campaign_id` and `failure_stage`, compressed via `zstd` with dictionary encoding ($< 3.5\text{ MB}$ per full run).
* The harness emits `triage_manifest.json` enabling immediate visual triage:
  ```bash
  python -m LUMIA.benchmarking.triage --run-id <run_id> --stage failed_ml --limit 50
  ```
* Validated annotations commit as `TriageAnnotationRecord` JSON files into `data/triage/annotations/`, directly ingested by `des_004` to retrain the neural network on hard sub-threshold edge cases.

--------------------------------------------------------------------------------

## 6. Astrophysical & Statistical Modeling

### 6.1 Analytical Trailing Convolution
In `src/LUMIA/physics/synthetic_injection.py`, discrete Riemann stepping is replaced with the exact closed-form analytical error-function convolution along the trail vector $(u, v)$:
$$I(u, v) = \frac{F_{\text{ADU}}}{2 L \sqrt{2\pi} \sigma} \exp\left(-\frac{v^2}{2\sigma^2}\right) \left[ \text{erf}\left(\frac{u + L/2}{\sqrt{2}\sigma}\right) - \text{erf}\left(\frac{u - L/2}{\sqrt{2}\sigma}\right) \right]$$
Eliminates discrete stepping ripples, enforces continuous photon conservation, and executes vectorized via `torch.special.erf`.

### 6.2 Bivariate Completeness Modeling
Instead of unconstrained symmetric 1D logistics, the 2D surface $\epsilon(V, \mu)$ is modeled using a generalized Richards' curve incorporating intra-exposure trailing shift and an asymptotic ceiling $\epsilon_{\max} < 1.0$:
$$\epsilon(V, \mu) = \frac{\epsilon_{\max}}{\left[ 1 + \nu \exp\left( k \left( V - V_{50}(\mu) \right) \right) \right]^{1/\nu}}$$
$$V_{50}(\mu) = V_{50}(0) - 1.25 \log_{10}\left(1 + \left(\frac{\mu \cdot t_{\text{exp}}}{2 \cdot \text{FWHM}}\right)^2\right)$$

### 6.3 Field-Level Clustered Bootstrap Confidence Intervals
Because synthetic sources injected into the same image frame share atmospheric seeing, background noise, and subtraction residuals, independent Bernoulli trial assumptions fail. The harness computes confidence bounds using **Field-Level Clustered Bootstrap Resampling** (resampling entire pointings/cutouts with replacement, $B = 1,000$ iterations), providing honest, non-deflated variance estimates for $V_{50}$ and $V_{90}$.

--------------------------------------------------------------------------------

## 7. Architectural Invariants & Governance Alignment

1. **Package Hierarchy Placement (ADR-0001):** The benchmarking harness resides in `src/LUMIA/benchmarking/` (Level 4 Orchestration). It may import from `core`, `physics`, `ml`, `models`, and `data_pipeline`. Computational modules must never import from `benchmarking/`. This invariant is verified by AST firewalls in `tests/test_security_and_boundaries.py`.
2. **Strict Bitwise Determinism:**
   * Global runtime enforces `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `torch.use_deterministic_algorithms(True)`, and disables TF32.
   * Injection generators use isolated `torch.Generator` instances seeded via BLAKE3-64 hash of `master_seed` and UUIDv7.
3. **CI Regression Guardrails:**
   * Bright regime ($V \le 20.5$): Zero tolerance ($\ge 1$ dropped bright object triggers immediate P0 build failure).
   * Faint transition regime ($V > 20.5$): Statistical shift $> 3.0\sigma$ triggers build failure; ambiguous shifts ($1.96\sigma < \Delta \le 3.0\sigma$) trigger an automated Sequential Probability Ratio Test (SPRT) mini-batch retry.
4. **Zero-Image Invariant (ADR-0003):** Under no circumstances are intermediate image arrays written to persistent storage. All pixel operations execute in volatile memory.
