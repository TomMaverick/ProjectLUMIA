"""Integration and unit tests for HelioLinC3D fusion and astrometric gating."""

from __future__ import annotations

import math
import numpy as np
import pytest

from LUMIA.core.schemas import CandidateDetection
from LUMIA.physics.heliolinc3d import (
    MU_SUN_AU_DAY,
    HelioLinC3DEngine,
    StereoscopicTriangulationBranch,
    compute_snr_weighted_chi2,
    propagate_kepler_universal,
    radec_to_unit_vector,
    unit_vector_to_radec,
)


def test_radec_unit_vector_roundtrip() -> None:
    """Test spherical to Cartesian unit vector roundtrip."""
    for ra, dec in [(0.0, 0.0), (180.25, -45.5), (359.9, 89.0), (45.0, 30.0)]:
        u = radec_to_unit_vector(ra, dec)
        assert np.linalg.norm(u) == pytest.approx(1.0, rel=1e-10)
        ra_rec, dec_rec = unit_vector_to_radec(u)
        assert ra_rec == pytest.approx(ra, abs=1e-7)
        assert dec_rec == pytest.approx(dec, abs=1e-7)


def test_kepler_universal_circular_orbit() -> None:
    """Propagate a 1 AU circular orbit for 1/4 period (approx 91.31 days)."""
    # Circular velocity at 1 AU: v = sqrt(mu / 1.0) AU/day
    v_circ = math.sqrt(MU_SUN_AU_DAY)
    r0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([0.0, v_circ, 0.0], dtype=np.float64)

    period_days = 2.0 * math.pi / v_circ  # ~365.256 days
    quarter_period = period_days / 4.0

    r_next, v_next = propagate_kepler_universal(r0, v0, quarter_period)

    # After 90 degrees in circular orbit: pos should be near [0, 1, 0], vel near [-v_circ, 0, 0]
    assert np.linalg.norm(r_next) == pytest.approx(1.0, rel=1e-5)
    assert r_next[0] == pytest.approx(0.0, abs=1e-4)
    assert r_next[1] == pytest.approx(1.0, abs=1e-4)
    assert v_next[0] == pytest.approx(-v_circ, abs=1e-4)
    assert v_next[1] == pytest.approx(0.0, abs=1e-4)


def test_stereoscopic_triangulation_accuracy() -> None:
    """Validate 3D target recovery across Earth - Sun-Earth L2 baseline."""
    # Target asteroid at (1.4 AU, 0.3 AU, 0.1 AU)
    r_target = np.array([1.4, 0.3, 0.1], dtype=np.float64)

    # Earth at (1.0, 0.0, 0.0) AU
    obs1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # L2 spacecraft at (1.01, 0.0, 0.0) AU (~1.5e6 km baseline)
    obs2 = np.array([1.01, 0.0, 0.0], dtype=np.float64)

    u1 = (r_target - obs1) / np.linalg.norm(r_target - obs1)
    u2 = (r_target - obs2) / np.linalg.norm(r_target - obs2)

    midpoint, separation = StereoscopicTriangulationBranch.triangulate(obs1, u1, obs2, u2)

    assert separation < 1e-10
    np.testing.assert_allclose(midpoint, r_target, rtol=1e-7, atol=1e-7)


def test_stereoscopic_triangulation_degenerate_parallel() -> None:
    """Degenerate parallel sightlines raise ValueError."""
    obs1 = np.array([1.0, 0.0, 0.0])
    obs2 = np.array([1.01, 0.0, 0.0])
    u = np.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="parallel"):
        StereoscopicTriangulationBranch.triangulate(obs1, u, obs2, u)


def test_snr_weighted_chi2_gating_behavior() -> None:
    """Test that low-SNR detections pass with larger centroid scatter while high-SNR fails."""
    # 4 epochs with 0.25 arcsec position residual
    residuals_deg = 0.25 / 3600.0
    obs_radec = np.array([
        [100.0 + residuals_deg, 20.0],
        [100.1 + residuals_deg, 20.0],
        [100.2 + residuals_deg, 20.0],
        [100.3 + residuals_deg, 20.0],
    ])
    pred_radec = np.array([
        [100.0, 20.0],
        [100.1, 20.0],
        [100.2, 20.0],
        [100.3, 20.0],
    ])

    # Case A: Low SNR = 2.0 (centroid uncertainty sigma ~ 1.5 / 4.0 = 0.375 arcsec)
    # 0.25 arcsec residual is well within 1-sigma scatter -> must accept!
    snrs_low = np.array([2.0, 2.0, 2.0, 2.0])
    chi2_low, red_low, accepted_low = compute_snr_weighted_chi2(
        obs_radec, pred_radec, snrs_low, fwhm_arcsec=1.5
    )
    assert accepted_low is True
    assert red_low < 3.0

    # Case B: High SNR = 25.0 (centroid uncertainty sigma ~ 1.5 / 50.0 = 0.03 arcsec)
    # 0.25 arcsec residual is an 8-sigma failure -> must reject!
    snrs_high = np.array([25.0, 25.0, 25.0, 25.0])
    chi2_high, red_high, accepted_high = compute_snr_weighted_chi2(
        obs_radec, pred_radec, snrs_high, fwhm_arcsec=1.5
    )
    assert accepted_high is False
    assert red_high > 10.0


def test_heliolinc3d_engine_routing() -> None:
    """Verify branch routing based on baseline and epoch difference."""
    engine = HelioLinC3DEngine()

    det_ground = CandidateDetection(
        candidate_id="rubin_001",
        field_id="f1",
        ra_deg=120.0,
        dec_deg=10.0,
        bjd_tdb=2460000.5000,
        snr=3.2,
        observer_pos_au=(1.0, 0.0, 0.0),
    )
    # JWST concurrent detection (10 minutes later, baseline ~0.01 AU)
    det_space_concurrent = CandidateDetection(
        candidate_id="jwst_001",
        field_id="f2",
        ra_deg=120.05,
        dec_deg=10.02,
        bjd_tdb=2460000.5070,  # ~10 min later
        snr=4.1,
        observer_pos_au=(1.01, 0.0, 0.0),  # L2 baseline
    )
    # JWST multi-day detection (2 days later)
    det_space_multiday = CandidateDetection(
        candidate_id="jwst_002",
        field_id="f2",
        ra_deg=121.5,
        dec_deg=10.8,
        bjd_tdb=2460002.5000,
        snr=4.5,
        observer_pos_au=(1.01, 0.0, 0.0),
    )

    # Concurrent across space baseline -> stereoscopic
    assert engine.determine_branch(det_ground, det_space_concurrent) == "stereoscopic"
    # Multi-day arc -> dynamical
    assert engine.determine_branch(det_ground, det_space_multiday) == "dynamical"


def test_faint_tracklet_passes_snr_weighted_gate_in_linking() -> None:
    """Verify that faint detections with physical photon-noise scatter pass SNR-weighted linking."""
    from LUMIA.physics.linking import TrackletLinkingConfig, validate_tracklet_kinematics

    # 4 epochs with residual ~1.1 arcsec, but SNR = 2.0 (sigma_centroid ~ 1.5 / 4.0 = 0.375 arcsec)
    detections = [
        CandidateDetection(
            candidate_id=f"faint_{i}",
            field_id="f",
            ra_deg=100.0 + i * 0.01 + (1.1 / 3600.0 if i % 2 else 0.0),
            dec_deg=20.0,
            bjd_tdb=2460000.0 + i * 0.05,
            snr=2.0,
        )
        for i in range(4)
    ]
    # Static gate (max_rms_residual_arcsec = 0.35, use_snr_weighted_gating = False) -> Rejects!
    cfg_static = TrackletLinkingConfig(max_rms_residual_arcsec=0.35, use_snr_weighted_gating=False)
    res_static = validate_tracklet_kinematics(detections, config=cfg_static)
    assert not res_static.is_valid

    # Dynamic SNR-weighted gate (use_snr_weighted_gating = True) -> Accepts!
    cfg_snr = TrackletLinkingConfig(max_rms_residual_arcsec=0.35, use_snr_weighted_gating=True)
    res_snr = validate_tracklet_kinematics(detections, config=cfg_snr)
    assert res_snr.is_valid

