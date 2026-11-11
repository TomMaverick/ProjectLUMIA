# Module 5: Tracklet Linking & Two-Branch HelioLinC3D Specification

## Overview
Module 5 is responsible for connecting individual transient detections across multiple epochs and disparate survey observatories (ground and space) into consistent moving-object tracklets and orbital arcs. It implements Project LUMIA's Two-Branch HelioLinC3D engine (`src/LUMIA/physics/heliolinc3d.py`) and topocentric-to-barycentric kinematic linking (`src/LUMIA/physics/linking.py`).

## Key Components

### 1. Two-Branch HelioLinC3D Engine (`src/LUMIA/physics/heliolinc3d.py`)
To prevent division-by-zero singularities when linking observations across large interplanetary parallax baselines (e.g., Cerro Pachón to Sun-Earth L2, $\Delta r \ge 500,000\text{ km}$):
* **Branch A (Stereoscopic Triangulation):** Activated when observation epochs satisfy $|t_j - t_i| < 1.0\text{ hour}$ and baseline separation $\|\vec{x}_{\text{obs}, j} - \vec{x}_{\text{obs}, i}\| \ge 500,000\text{ km}$. Solves the non-degenerate 3D intersection of skew line-of-sight unit vectors $\hat{u}_i$ and $\hat{u}_j$:

  $$\vec{r} = \vec{x}_{\text{obs}, i} + \rho_i \hat{u}_i = \vec{x}_{\text{obs}, j} + \rho_j \hat{u}_j$$

  Yields instantaneous 3D heliocentric position $\vec{r}$ directly without requiring numerical velocity differencing.
* **Branch B (Dynamical Association):** Activated for observations separated by $|t_j - t_i| \ge 1.0\text{ hour}$ or ground-only baselines. Inverts topocentric angles into heliocentric states across an $(r_H, \dot{r}_H)$ grid and propagates state vectors across epochs using universal Keplerian $f$ and $g$ functions (`propagate_kepler_universal`).

### 2. Planetary Retarded-Time Integration (`src/LUMIA/physics/ephemerides.py`)
Standard stellar plane-wave barycentric formulas ($BJD_{TDB}$) produce $\sim 0.21^{\prime\prime}$ apparent astrometric errors across Earth-L2 baselines for moving targets. Module 5 integrates the `PlanetaryRetardedTimeSolver` solving:

$$\tau = t_{\text{obs}} - \frac{\|\vec{r}(\tau) - \vec{x}_{\text{obs}}(t_{\text{obs}})\|}{c}$$

iteratively in the Barycentric Celestial Reference System (BCRS) using thread-safe NAIF SPICE kernels (`SPICEKernelManager`).

### 3. SNR-Weighted Astrometric Gating
Replaces rigid static residual cutoffs ($RMS \le 0.15^{\prime\prime}$) that would falsely reject sub-$5\sigma$ candidates. Dynamically accommodates photon-noise centroiding scatter ($\sigma_{\text{centroid}} \approx \text{FWHM} / (2 \cdot SNR)$) via normalized chi-squared evaluation:

$$\chi^2 = \sum_{k=1}^N \frac{(\Delta \theta_k)^2}{\sigma_{\text{centroid}, k}^2 + \sigma_{\text{sys}}^2 + \sigma_{\text{DCR}, k}^2}$$

### 4. Space-Velocity Transformation & Clustering
* Groups isolated spatial detections by hypothesized velocity vectors using KD-trees.
* Rejects tracklets exceeding physical acceleration or solar system escape velocity bounds.

## Architecture & Boundary Guidelines
* **Clear Upstream/Downstream Delineation:** Module 5 owns multi-epoch kinematic linking exclusively. Single-epoch GPU image subtraction and stack peak extraction belong to **Module 4** (`docs/modules/module_04_gpu_processing.md`), and Real/Bogus neural stamp classification belongs to **Module 7** (`docs/modules/module_07_cnn_classifier.md`).
* **AST Architectural Firewall:** Module 5 computational routines remain purely algorithmic and must never import database drivers (`duckdb`, `sqlalchemy`) or web frameworks (`fastapi`).
* **Astropy Units Enforcement:** All kinematic calculations utilize Astropy units and constants throughout.
* **Pure Domain Contracts (ADR-0002):** Module 5 operates strictly on immutable Pydantic `CandidateDetection` entities (enriched with multi-survey metadata: `survey_id`, `filter_band`, `observer_pos_au`, `light_travel_time_sec`) and outputs validated `Tracklet` objects.