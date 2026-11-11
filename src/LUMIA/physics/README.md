# LUMIA Physics

## Overview
LUMIA Physics encapsulates the scientific and computational core of Project LUMIA, implementing Modules 1, 3, 4, and 5. It provides pure astrophysical algorithms for WCS coordinate transformations, BJD_TDB light-time corrections, calibrated aperture photometry, GPU-accelerated ZOGY image subtraction, $\Psi/\Phi$ shift-and-stacking, candidate peak extraction, multi-epoch HelioLinC tracklet linking, and synthetic asteroid injection.

## Interface
- `extract_celestial_wcs` / `compute_bjd_tdb`: Extracts celestial WCS models from FITS headers and calculates Barycentric Julian Dates in Barycentric Dynamical Time.
- `pixel_to_sky` / `sky_to_pixel`: Bidirectional coordinate transformations between detector pixels and ICRS celestial coordinates.
- `perform_aperture_photometry` / `compute_zero_point`: Executes aperture photometry with local annulus background estimation and photometric zero-point calibration.
- `zogy_subtract`: GPU-accelerated, mathematically optimal image subtraction yielding difference images, significance maps ($S_{corr}$), and subtraction PSFs.
- `shift_and_stack` / `compute_psi_phi_stack`: Multi-exposure likelihood coaddition searching for faint sub-threshold moving targets along velocity vectors.
- `extract_zogy_cutout_batch`: Extracts deterministic SNR peaks and slices standardized 3-channel cutout batches (`CutoutBatchResult`).
- `link_tracklets`: Implements the HelioLinC algorithm, projecting detections into gnomonic space-velocity coordinates and clustering candidates into multi-epoch tracklets.
- `TwoBranchHelioLinC3D` / `HelioLinC3DEngine`: Two-branch orbital linking engine routing wide-baseline ($>500,000\text{ km}$) pairs to stereoscopic sightline triangulation and single-observatory arcs to Keplerian dynamical association.
- `SPICEKernelManager` / `PlanetaryRetardedTimeSolver`: NAIF SPICE kernel manager and iterative BCRS planetary light-travel time solver ($\tau = t_{\text{obs}} - \rho(\tau)/c$).
- `validate_tracklet_kinematics`: Verifies candidate tracklet motions against solar system bound orbital velocity ceilings.
- `inject_synthetic_asteroid` / `inject_faint_moving_source`: Convolves synthetic point-source profiles and faint moving streaks into science frames for completeness testing.
- `MinorPlanetCenterClient`: Client for querying known solar system ephemerides and cross-matching known objects.
- `TelescopeConfig` / `ZOGYConfig` / `GPUProcessingConfig` / `TrackletLinkingConfig`: Validated Pydantic domain configurations.

## Invariants
- Strict separation of concerns: this module remains completely free of database engines, SQL schemas, and web framework dependencies.
- Astrometric coordinates must strictly satisfy `0.0 <= ra_deg < 360.0` and `-90.0 <= dec_deg <= 90.0`.
- All image subtraction inputs (science image, reference image, PSFs, noise variances) must be finite 2D floating-point arrays with matching dimensions.
- Tracklet linking discards candidate clusters exceeding solar system parabolic escape velocities.

## Side Effects
- Allocates CPU and CUDA tensor memory for FFTs, grid clustering, and spatial slicing.
- Queries external astronomical services via `MinorPlanetCenterClient` over HTTP.

## Verification
- Run: `uv run pytest tests/test_physics`
