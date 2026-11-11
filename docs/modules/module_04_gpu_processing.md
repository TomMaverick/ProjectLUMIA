# Module 4: GPU-Optimized Image Processing & Stacking Specification

## Overview
Module 4 handles high-performance, GPU-accelerated image processing, optimal image subtraction, and tensor-based shifting-and-stacking operations. It forms the computational core for detecting faint, low-Signal-to-Noise Ratio (SNR) moving interplanetary objects.

## Key Components
* **Optimal Image Subtraction (ZOGY):** 
  * Implementation of the Zackay-Ofek-Gal-Yam algorithm for statistically optimal image subtraction.
  * Handles point spread function (PSF) variations, differential flux adjustments, and noise matching between reference templates and science images to isolate transient and moving sources.
* **Tensor-Based Stacking ($\Psi / \Phi$):** 
  * Matrix-vectorized shift-and-stack operations executed on the GPU using PyTorch tensors.
  * Stacks sequential frames along hypothetical velocity and direction vectors to enhance the SNR of fast-moving or distant asteroids.
* **Candidate Peak Extraction & Cutout Slicing:**
  * Thresholds ZOGY significance maps and $\Psi / \Phi$ SNR stacks to isolate candidate transient peaks with sub-pixel coordinates.
  * Slices aligned, in-memory 3-channel `(Science, Reference, Difference/ZOGY)` floating-point stamps (`31x31` or `51x51` pixels) for downstream Real/Bogus scoring in **Module 7** (`docs/modules/module_07_cnn_classifier.md`), while delegating all multi-epoch kinematic tracklet linking exclusively to **Module 5** (`docs/modules/module_05_tracklet_linking.md`).

## Performance, Contracts & VRAM Optimization
* **Strict Pydantic Contracts:** All configuration schemas and tensor result containers must use immutable Pydantic v2 models (`strict=True`, `extra="forbid"`, `frozen=True`, `allow_inf_nan=False`) rather than standard-library dataclasses.
* **Out-of-Core VRAM Chunking:** Large image matrices and velocity batches must be chunked into VRAM efficiently to prevent hardware overflow.
* **Determinism:** All PyTorch operations must enforce strict determinism flags (`torch.use_deterministic_algorithms(True)`) after upfront target device validation to ensure reproducible scientific results.