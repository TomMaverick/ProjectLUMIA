# Module 7: Machine Learning CNN Classifier Specification

## 1. Overview
Module 7 implements a PyTorch-based Convolutional Neural Network (CNN) for automated Real/Bogus (R/B) transient and moving-object classification within Project LUMIA. Operating downstream of optimal image subtraction and candidate extraction (Module 4), this module distinguishes genuine low-Signal-to-Noise Ratio (SNR) interplanetary moving objects from instrumental and algorithmic false positives—including cosmic ray hits, hot or dead pixels, saturated star halos, satellite trail residuals, and Point Spread Function (PSF) dipole artifacts resulting from imperfect image subtraction.

In accordance with Project LUMIA architectural directives, Module 7 enforces strict Separation of Concerns: it encapsulates pure tensor preprocessing, neural network inference, and out-of-core training workflows without containing any database persistence logic (PostgreSQL or DuckDB) or workflow orchestration code.

## 2. Network Architecture
* **Three-Channel Input Triplets:**
  * The classifier accepts aligned, multi-channel 2D image stamps structured as 4D floating-point tensors of shape `(Batch, 3, Height, Width)` centered on each candidate detection's sub-pixel coordinates:
    1. **Channel 0 (Science):** Calibrated science exposure cutout capturing the candidate detection epoch.
    2. **Channel 1 (Reference):** Co-added static reference template cutout representing the unperturbed background sky.
    3. **Channel 2 (Difference / ZOGY):** Statistically optimal subtraction cutout (ZOGY difference image or matched-filter significance map from Module 4) isolating transient flux.
* **Spatial Cutout Dimensions:**
  * Input stamps must use configurable, odd-numbered square dimensions to preserve exact center-pixel symmetry around the target source.
  * Standardized supported input resolutions are `31x31` pixels (default for nominal seeing and compact PSFs) and `51x51` pixels (for broader PSF wings, degraded seeing, or slightly trailed fast-moving near-Earth objects).
* **Selectable Backbone Topologies:**
  * **VGG-Style Architecture:** A compact sequential feature extractor composed of stacked `3x3` convolutional layers, batch normalization, non-linear activations, progressive spatial pooling, global adaptive pooling, and dropout-regularized dense classification layers. Designed for minimal parameter overhead and high-throughput GPU inference on small stamps.
  * **ResNet-Lite Architecture:** A shallow residual network incorporating shortcut skip connections across convolutional blocks, batch normalization, and global average pooling. Preserves fine pixel-level spatial morphology—such as subtle positive-negative ZOGY dipoles versus symmetric stellar PSFs—while maintaining stable gradient propagation during training.
* **Input Normalization & Sanitization:**
  * Prior to feature extraction, input triplets undergo vectorized per-channel robust scaling (e.g., median background subtraction and Median Absolute Deviation or Z-score normalization) alongside deterministic masking/replacement of non-finite pixel values (`NaN` or `Inf`) near detector edges or masked regions.

## 3. Data Pipeline, Configuration & Execution
* **Pydantic Configuration & Interface Schemas:**
  * All runtime, training, and architectural parameters—including backbone type (`vgg` or `resnet_lite`), stamp size (`31` or `51`), inference batch size, decision threshold, model checkpoint path/version, CPU thread limits, and target GPU device identifiers—must be strictly validated via Pydantic configuration models.
  * All data exchanged with upstream and downstream modules (candidate cutout payloads, batch inference requests, and classification outputs) must use immutable, strictly typed Pydantic v2 schemas (`strict=True`, `extra="forbid"`, `frozen=True`, `allow_inf_nan=False`). Invalid tensor shapes, non-finite metadata, mismatched channel counts, or out-of-bounds parameters must fail fast with descriptive validation errors.
* **Deterministic PyTorch Execution:**
  * To uphold FAIR data standards and scientific reproducibility across Windows, Linux, and macOS environments, all training and inference routines must enforce strict PyTorch determinism.
  * Execution requires fixed random seeds across Python, NumPy, and PyTorch, mandatory activation of deterministic PyTorch algorithms, disabling CuDNN benchmark auto-tuning, and enforcing deterministic convolution kernels.
  * Inference passes must explicitly operate in network evaluation mode with gradient tracking disabled to guarantee bit-consistent scoring and minimal memory overhead.
* **GPU Inference Batching & Memory Management:**
  * **In-Memory Pipeline Operation:** During live pipeline execution, candidate cutouts are sliced and assembled directly in RAM/VRAM from calibrated image arrays (via **Module 4**, `docs/modules/module_04_gpu_processing.md`) without writing temporary FITS or image files to disk, protecting SSD write endurance.
  * **VRAM-Bounded Mini-Batching:** Candidate triplets from dense fields are aggregated and processed through the GPU in configurable mini-batch chunks. Batch sizing adapts to user-defined hardware resource limits to maximize tensor-core utilization while preventing GPU Out-Of-Memory (OOM) errors.
  * **Out-of-Core Training & Active Learning:** Model training and periodic Active Learning retraining ingest labeled Real/Bogus triplets streamed from compact HDF5 tensor archives produced by **Module 8** (`docs/modules/module_08_ground_truth_mocking.md`, incorporating MPC archival extractions, synthetic injections, and human-vetted candidates from Module 10), utilizing dihedral rotation/reflection augmentations that preserve astrophysical PSF validity.

## 4. Output Specification
* **Continuous Confidence Score:**
  * For every input triplet, the network produces a single scalar logit transformed via a sigmoid activation function during inference (paired with numerically stable logit-based binary cross-entropy loss during training).
  * The resulting Real/Bogus confidence score is a floating-point value strictly bounded within the closed interval `[0.0, 1.0]`:
    * **`0.0` (Bogus):** Indicates maximum certainty that the candidate is a non-astrophysical artifact, noise spike, or subtraction residual.
    * **`1.0` (Real):** Indicates maximum certainty that the candidate represents a genuine astronomical transient or moving solar system body.
* **Downstream Integration:**
  * Each prediction is emitted within a validated result schema containing the candidate identifier, the continuous Real/Bogus score in `[0.0, 1.0]`, a boolean pass/reject flag evaluated against a configurable classification threshold, and the active model version identifier for provenance tracking in Module 5 (Tracklet Linking) and Module 10 (Candidate Triage Web-UI).
