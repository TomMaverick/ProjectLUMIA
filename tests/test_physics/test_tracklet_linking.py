"""Strict TDD tests for Module 5: Tracklet Linking & HelioLinC kinematics."""

from __future__ import annotations

import ast
import math
import time
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from pydantic import ValidationError

from LUMIA.core.schemas import CandidateDetection
from LUMIA.physics.linking import (
    KinematicValidationResult,
    SpaceVelocityBatch,
    SpaceVelocityState,
    Tracklet,
    TrackletLinkingConfig,
    _compute_tangent_basis,
    _gnomonic_project_arcsec,
    _sky_to_unit_vectors,
    cluster_velocity_states,
    compute_max_bound_angular_speed_arcsec_per_hr,
    link_tracklets,
    project_detections_to_velocity_batch,
    project_detections_to_velocity_space,
    validate_tracklet_kinematics,
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _generate_linear_detections(
    *,
    prefix: str,
    ra0_deg: float,
    dec0_deg: float,
    v_ra_cosdec_arcsec_per_hr: float,
    v_dec_arcsec_per_hr: float,
    epochs_bjd: list[float],
    accel_ra_arcsec_per_day2: float = 0.0,
    accel_dec_arcsec_per_day2: float = 0.0,
) -> list[CandidateDetection]:
    """Synthesize multi-epoch sky detections for a moving object on the celestial sphere."""
    t_ref = epochs_bjd[0]
    detections: list[CandidateDetection] = []

    for idx, bjd in enumerate(epochs_bjd):
        dt_days = bjd - t_ref
        dt_hours = dt_days * 24.0

        d_ra_cosdec_arcsec = (
            v_ra_cosdec_arcsec_per_hr * dt_hours
            + 0.5 * accel_ra_arcsec_per_day2 * (dt_days**2)
        )
        d_dec_arcsec = (
            v_dec_arcsec_per_hr * dt_hours
            + 0.5 * accel_dec_arcsec_per_day2 * (dt_days**2)
        )

        dec_deg = dec0_deg + (d_dec_arcsec / 3600.0)
        cos_dec = math.cos(math.radians(dec0_deg))
        ra_deg = (ra0_deg + (d_ra_cosdec_arcsec / (3600.0 * cos_dec))) % 360.0

        detections.append(
            CandidateDetection(
                candidate_id=f"{prefix}_ep{idx}",
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                bjd_tdb=bjd,
                mag=20.5,
                snr=11.0,
            )
        )
    return detections


def _naive_pairwise_project(
    detections: list[CandidateDetection],
    *,
    reference_bjd_tdb: float | None = None,
    config: TrackletLinkingConfig | None = None,
    filter_by_speed_bounds: bool = False,
) -> list[SpaceVelocityState]:
    """Reference O(N^2) naive nested-loop implementation for numerical equivalence validation."""
    cfg = config if config is not None else TrackletLinkingConfig()
    if len(detections) < 2:
        return []

    ordered_indices = sorted(
        range(len(detections)),
        key=lambda idx: (detections[idx].bjd_tdb, detections[idx].candidate_id),
    )
    ordered_dets = [detections[i] for i in ordered_indices]

    ref_bjd = float(
        reference_bjd_tdb if reference_bjd_tdb is not None else ordered_dets[0].bjd_tdb
    )

    bjd_arr = np.array([d.bjd_tdb for d in ordered_dets], dtype=np.float64)
    min_dt = float(np.min(np.abs(bjd_arr - ref_bjd)))
    half_min_baseline_days = float((0.5 * cfg.min_baseline_hours * u.hour).to_value(u.day))
    ref_mask = np.abs(bjd_arr - ref_bjd) <= (min_dt + half_min_baseline_days)
    all_u_vecs = _sky_to_unit_vectors(
        np.array([d.ra_deg for d in ordered_dets], dtype=np.float64),
        np.array([d.dec_deg for d in ordered_dets], dtype=np.float64),
    )
    u0, east_vec, north_vec = _compute_tangent_basis(all_u_vecs[ref_mask])
    xi_arcsec, eta_arcsec = _gnomonic_project_arcsec(ordered_dets, u0, east_vec, north_vec)

    t_rel_hr = ((bjd_arr - ref_bjd) * u.day).to_value(u.hour)
    min_dt_hr = float((cfg.min_baseline_hours * u.hour).to_value(u.hour))
    max_dt_hr = float((cfg.max_baseline_days * u.day).to_value(u.hour))

    max_speed_ceiling = cfg.max_angular_speed_arcsec_per_hr
    if cfg.enforce_escape_velocity_bound:
        max_speed_ceiling = min(
            max_speed_ceiling,
            compute_max_bound_angular_speed_arcsec_per_hr(cfg.assumed_heliocentric_distance_au),
        )

    states: list[SpaceVelocityState] = []
    n = len(ordered_dets)

    for i in range(n):
        t_i = float(t_rel_hr[i])
        xi_i = float(xi_arcsec[i])
        eta_i = float(eta_arcsec[i])
        for j in range(i + 1, n):
            t_j = float(t_rel_hr[j])
            dt = t_j - t_i
            if dt < min_dt_hr or dt > max_dt_hr:
                continue

            v_xi = (float(xi_arcsec[j]) - xi_i) / dt
            v_eta = (float(eta_arcsec[j]) - eta_i) / dt

            if filter_by_speed_bounds:
                speed = math.hypot(v_xi, v_eta)
                if speed < cfg.min_angular_speed_arcsec_per_hr or speed > max_speed_ceiling:
                    continue

            ref_xi = xi_i - v_xi * t_i
            ref_eta = eta_i - v_eta * t_i

            states.append(
                SpaceVelocityState(
                    source_candidate_id=ordered_dets[i].candidate_id,
                    target_candidate_id=ordered_dets[j].candidate_id,
                    source_index=int(ordered_indices[i]),
                    target_index=int(ordered_indices[j]),
                    reference_bjd_tdb=ref_bjd,
                    baseline_hours=dt,
                    ref_xi_arcsec=ref_xi,
                    ref_eta_arcsec=ref_eta,
                    v_ra_cosdec_arcsec_per_hr=v_xi,
                    v_dec_arcsec_per_hr=v_eta,
                )
            )

    return states


class TestSpaceVelocityTransformations:
    """Verify projection of temporal RA/DEC detections into space-velocity coordinates."""

    def test_projects_temporal_ra_dec_into_velocity_space_accurately(self) -> None:
        """Linear sky motion at non-zero declination recovers exact tangent-plane velocity."""
        epochs = [2460755.50, 2460755.5416667, 2460755.5833333, 2460755.625]
        expected_vra = 36.0  # arcsec / hr (typical main-belt asteroid ~ 0.24 deg/day)
        expected_vdec = -18.0  # arcsec / hr

        detections = _generate_linear_detections(
            prefix="mba",
            ra0_deg=120.0,
            dec0_deg=30.0,
            v_ra_cosdec_arcsec_per_hr=expected_vra,
            v_dec_arcsec_per_hr=expected_vdec,
            epochs_bjd=epochs,
        )

        config = TrackletLinkingConfig()
        states: list[SpaceVelocityState] = project_detections_to_velocity_space(
            detections,
            reference_bjd_tdb=epochs[0],
            config=config,
        )

        # 4 epochs -> C(4, 2) = 6 ordered temporal pairs
        assert len(states) == 6

        for state in states:
            assert state.v_ra_cosdec_arcsec_per_hr == pytest.approx(expected_vra, rel=1e-3)
            assert state.v_dec_arcsec_per_hr == pytest.approx(expected_vdec, rel=1e-3)
            assert state.ref_xi_arcsec == pytest.approx(0.0, abs=0.05)
            assert state.ref_eta_arcsec == pytest.approx(0.0, abs=0.05)

    def test_handles_ra_zero_meridian_wraparound(self) -> None:
        """Detections crossing RA 359.99 deg -> 0.02 deg compute continuous Eastward velocity."""
        epochs = [2460755.50, 2460755.55, 2460755.60]
        expected_vra = 45.0  # arcsec / hr Eastward across 0 deg boundary
        expected_vdec = 12.0  # arcsec / hr Northward

        detections = _generate_linear_detections(
            prefix="wrap",
            ra0_deg=359.992,
            dec0_deg=10.0,
            v_ra_cosdec_arcsec_per_hr=expected_vra,
            v_dec_arcsec_per_hr=expected_vdec,
            epochs_bjd=epochs,
        )
        # Sanity check: RA actually wrapped across 360 -> 0 deg
        assert detections[0].ra_deg > 359.9
        assert detections[-1].ra_deg < 0.1

        states = project_detections_to_velocity_space(
            detections,
            reference_bjd_tdb=epochs[0],
            config=TrackletLinkingConfig(),
        )
        assert len(states) == 3
        for state in states:
            assert state.v_ra_cosdec_arcsec_per_hr == pytest.approx(expected_vra, rel=1e-3)
            assert state.v_dec_arcsec_per_hr == pytest.approx(expected_vdec, rel=1e-3)


class TestVelocitySpaceClustering:
    """Verify KD-Tree / DBSCAN clustering groups real moving objects and ignores noise."""

    def test_clusters_multiple_moving_objects_and_ignores_random_noise(self) -> None:
        """Two distinct asteroids across 4 epochs are linked cleanly amidst 35 noise points."""
        epochs = [2460755.50, 2460755.54, 2460755.58, 2460755.62]

        asteroid_a = _generate_linear_detections(
            prefix="astA",
            ra0_deg=180.10,
            dec0_deg=15.20,
            v_ra_cosdec_arcsec_per_hr=-32.0,
            v_dec_arcsec_per_hr=14.0,
            epochs_bjd=epochs,
        )
        asteroid_b = _generate_linear_detections(
            prefix="astB",
            ra0_deg=180.35,
            dec0_deg=15.45,
            v_ra_cosdec_arcsec_per_hr=54.0,
            v_dec_arcsec_per_hr=-22.0,
            epochs_bjd=epochs,
        )

        rng = np.random.default_rng(seed=202605)
        noise_detections: list[CandidateDetection] = []
        for idx in range(35):
            noise_detections.append(
                CandidateDetection(
                    candidate_id=f"noise_{idx:02d}",
                    ra_deg=float(rng.uniform(179.8, 180.8)),
                    dec_deg=float(rng.uniform(14.9, 15.8)),
                    bjd_tdb=float(epochs[idx % len(epochs)]),
                    mag=21.2,
                    snr=5.5,
                )
            )

        mixed_catalog = asteroid_a + asteroid_b + noise_detections
        rng.shuffle(mixed_catalog)  # type: ignore[arg-type]

        config = TrackletLinkingConfig(
            min_epochs=4,
            position_eps_arcsec=1.5,
            velocity_eps_arcsec_per_hr=1.5,
        )

        # Verify low-level clustering finds at least the two dense velocity clusters
        states = project_detections_to_velocity_space(mixed_catalog, config=config)
        clusters = cluster_velocity_states(states, config=config)
        assert len(clusters) >= 2

        # Verify end-to-end linking recovers exactly the two true 4-epoch tracklets
        tracklets: list[Tracklet] = link_tracklets(mixed_catalog, config=config)
        assert len(tracklets) == 2

        recovered_sets = {
            frozenset(d.candidate_id for d in trk.detections) for trk in tracklets
        }
        expected_a = frozenset(d.candidate_id for d in asteroid_a)
        expected_b = frozenset(d.candidate_id for d in asteroid_b)
        assert recovered_sets == {expected_a, expected_b}

        for trk in tracklets:
            assert trk.rms_residual_arcsec < 0.25


class TestKinematicValidation:
    """Verify astrophysical velocity and acceleration bounds using Astropy constants/units."""

    def test_accepts_physically_valid_main_belt_tracklet(self) -> None:
        """Bound main-belt asteroid trajectory passes velocity and acceleration checks."""
        epochs = [2460755.50, 2460755.55, 2460755.60, 2460755.65]
        detections = _generate_linear_detections(
            prefix="valid_mba",
            ra0_deg=210.0,
            dec0_deg=-5.0,
            v_ra_cosdec_arcsec_per_hr=-35.0,
            v_dec_arcsec_per_hr=10.0,
            epochs_bjd=epochs,
            accel_ra_arcsec_per_day2=0.4,
            accel_dec_arcsec_per_day2=-0.2,
        )

        result: KinematicValidationResult = validate_tracklet_kinematics(
            detections,
            config=TrackletLinkingConfig(),
        )
        assert result.is_valid is True
        assert result.rejection_reason is None
        assert result.angular_speed_arcsec_per_hr == pytest.approx(
            math.hypot(-35.0, 10.0),
            rel=1e-2,
        )

        # Verify implied transverse heliocentric velocity uses Astropy AU/GM_sun and is < v_esc
        v_esc_kms = float(
            np.sqrt(2.0 * const.GM_sun / (2.5 * const.au)).to_value(u.km / u.s)
        )
        assert result.implied_transverse_speed_km_s < v_esc_kms

    def test_rejects_tracklet_with_impossible_acceleration(self) -> None:
        """Tracklet exhibiting extreme non-gravitational sky acceleration is rejected."""
        epochs = [2460755.50, 2460755.55, 2460755.60, 2460755.65]
        accelerating_detections = _generate_linear_detections(
            prefix="bad_accel",
            ra0_deg=210.0,
            dec0_deg=-5.0,
            v_ra_cosdec_arcsec_per_hr=-30.0,
            v_dec_arcsec_per_hr=12.0,
            epochs_bjd=epochs,
            accel_ra_arcsec_per_day2=2500.0,
            accel_dec_arcsec_per_day2=-1800.0,
        )

        result = validate_tracklet_kinematics(
            accelerating_detections,
            config=TrackletLinkingConfig(max_angular_accel_arcsec_per_day2=50.0),
        )
        assert result.is_valid is False
        assert result.rejection_reason is not None
        assert "acceleration" in result.rejection_reason.lower()

    def test_rejects_unbound_hypervelocity_and_stationary_false_positives(self) -> None:
        """Objects exceeding Solar escape velocity or stationary stars are rejected."""
        epochs = [2460755.50, 2460755.52, 2460755.54, 2460755.56]

        # 1. Hypervelocity object (e.g. > 20,000 arcsec/hr -> unbound at asteroid distances)
        fast_detections = _generate_linear_detections(
            prefix="unbound_fast",
            ra0_deg=45.0,
            dec0_deg=20.0,
            v_ra_cosdec_arcsec_per_hr=25000.0,
            v_dec_arcsec_per_hr=5000.0,
            epochs_bjd=epochs,
        )
        fast_res = validate_tracklet_kinematics(
            fast_detections,
            config=TrackletLinkingConfig(assumed_heliocentric_distance_au=2.0),
        )
        assert fast_res.is_valid is False
        assert fast_res.rejection_reason is not None
        assert (
            "velocity" in fast_res.rejection_reason.lower()
            or "escape" in fast_res.rejection_reason.lower()
        )

        # 2. Stationary star / subtraction residual (~0.01 arcsec/hr)
        stationary_detections = _generate_linear_detections(
            prefix="stationary_star",
            ra0_deg=45.0,
            dec0_deg=20.0,
            v_ra_cosdec_arcsec_per_hr=0.01,
            v_dec_arcsec_per_hr=-0.01,
            epochs_bjd=epochs,
        )
        slow_res = validate_tracklet_kinematics(
            stationary_detections,
            config=TrackletLinkingConfig(min_angular_speed_arcsec_per_hr=1.0),
        )
        assert slow_res.is_valid is False
        assert slow_res.rejection_reason is not None
        assert "velocity" in slow_res.rejection_reason.lower()

    def test_pydantic_validation_enforces_strict_bounds(self) -> None:
        """Invalid sky coordinates or negative thresholds raise Pydantic ValidationError."""
        with pytest.raises(ValidationError):
            CandidateDetection(
                candidate_id="bad_dec",
                ra_deg=120.0,
                dec_deg=95.0,
                bjd_tdb=2460755.5,
            )

        with pytest.raises(ValidationError):
            TrackletLinkingConfig(min_epochs=2)


class TestLinkingModelStrictness:
    """NaN/Inf and extra-field rejection on the remaining linking models (P0-2)."""

    def test_config_rejects_non_finite_thresholds(self) -> None:
        for field in (
            "min_baseline_hours",
            "max_baseline_days",
            "max_angular_speed_arcsec_per_hr",
            "max_angular_accel_arcsec_per_day2",
            "max_rms_residual_arcsec",
            "assumed_heliocentric_distance_au",
            "position_eps_arcsec",
            "velocity_eps_arcsec_per_hr",
        ):
            for bad in (float("nan"), float("inf")):
                with pytest.raises(ValidationError):
                    TrackletLinkingConfig(**{field: bad})  # type: ignore[arg-type]

    def test_config_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TrackletLinkingConfig(unknown=1)  # type: ignore[call-arg]

    def test_space_velocity_state_rejects_non_finite_values(self) -> None:
        good = {
            "source_candidate_id": "a",
            "target_candidate_id": "b",
            "source_index": 0,
            "target_index": 1,
            "reference_bjd_tdb": 2460755.5,
            "baseline_hours": 1.0,
            "ref_xi_arcsec": 0.0,
            "ref_eta_arcsec": 0.0,
            "v_ra_cosdec_arcsec_per_hr": 10.0,
            "v_dec_arcsec_per_hr": -5.0,
        }
        SpaceVelocityState(**good)  # type: ignore[arg-type]
        for field in (
            "reference_bjd_tdb",
            "baseline_hours",
            "ref_xi_arcsec",
            "ref_eta_arcsec",
            "v_ra_cosdec_arcsec_per_hr",
            "v_dec_arcsec_per_hr",
        ):
            for bad in (float("nan"), float("inf")):
                with pytest.raises(ValidationError):
                    SpaceVelocityState(**{**good, field: bad})  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            SpaceVelocityState(**good, extra=1)  # type: ignore[arg-type]

    def test_kinematic_result_and_tracklet_reject_non_finite_values(self) -> None:
        with pytest.raises(ValidationError):
            KinematicValidationResult(
                is_valid=True,
                v_ra_cosdec_arcsec_per_hr=float("nan"),
                v_dec_arcsec_per_hr=0.0,
                angular_speed_arcsec_per_hr=1.0,
                angular_accel_arcsec_per_day2=0.0,
                rms_residual_arcsec=0.0,
                implied_transverse_speed_km_s=1.0,
                max_bound_transverse_speed_km_s=10.0,
            )
        det = CandidateDetection(
            candidate_id="d0",
            ra_deg=10.0,
            dec_deg=5.0,
            bjd_tdb=2460755.5,
        )
        with pytest.raises(ValidationError):
            Tracklet(
                tracklet_id="trk_0001",
                detections=(det,),
                reference_bjd_tdb=2460755.5,
                v_ra_cosdec_arcsec_per_hr=1.0,
                v_dec_arcsec_per_hr=1.0,
                angular_speed_arcsec_per_hr=float("inf"),
                angular_accel_arcsec_per_day2=0.0,
                rms_residual_arcsec=0.0,
                implied_transverse_speed_km_s=1.0,
            )

    def test_detection_rejects_non_finite_optional_photometry(self) -> None:
        """Optional mag/snr/cnn_score were previously excluded from the finite check."""
        for field in ("mag", "snr", "cnn_score"):
            with pytest.raises(ValidationError):
                CandidateDetection(
                    candidate_id="d0",
                    ra_deg=10.0,
                    dec_deg=5.0,
                    bjd_tdb=2460755.5,
                    **{field: float("nan")},  # type: ignore[arg-type]
                )


class TestVectorizedBroadcastingAndAstropyPurity:
    """Verify vectorized NumPy/PyTorch broadcasting and pure Astropy unit math."""

    def test_numpy_and_torch_backends_match_and_scale_without_nested_loops(self) -> None:
        """Verify vectorized NumPy and PyTorch broadcasting agree and run rapidly on N=260."""
        rng = np.random.default_rng(202611)
        epochs = [2460755.50, 2460755.55, 2460755.60, 2460755.65]
        dets = _generate_linear_detections(
            prefix="vec_ast",
            ra0_deg=205.2,
            dec0_deg=-12.4,
            v_ra_cosdec_arcsec_per_hr=42.5,
            v_dec_arcsec_per_hr=-19.0,
            epochs_bjd=epochs,
        )
        for idx in range(256):
            dets.append(
                CandidateDetection(
                    candidate_id=f"bg_{idx:03d}",
                    ra_deg=float(rng.uniform(204.7, 205.8)),
                    dec_deg=float(rng.uniform(-12.9, -11.9)),
                    bjd_tdb=float(epochs[idx % len(epochs)]),
                    mag=21.4,
                    snr=6.2,
                )
            )

        cfg = TrackletLinkingConfig(
            min_baseline_hours=0.5,
            max_baseline_days=2.0,
            min_angular_speed_arcsec_per_hr=5.0,
            max_angular_speed_arcsec_per_hr=600.0,
        )

        t0 = time.perf_counter()
        batch_np: SpaceVelocityBatch = project_detections_to_velocity_batch(
            dets,
            config=cfg,
            filter_by_speed_bounds=True,
            backend="numpy",
        )
        elapsed_s = time.perf_counter() - t0
        assert elapsed_s < 1.0, f"Vectorized projection took too long: {elapsed_s:.3f}s"

        batch_torch: SpaceVelocityBatch = project_detections_to_velocity_batch(
            dets,
            config=cfg,
            filter_by_speed_bounds=True,
            backend="torch",
        )

        assert batch_np.pair_count > 0
        assert batch_np.pair_count == batch_torch.pair_count
        np.testing.assert_array_equal(batch_np.source_indices, batch_torch.source_indices)
        np.testing.assert_array_equal(batch_np.target_indices, batch_torch.target_indices)
        np.testing.assert_allclose(batch_np.baseline_hours, batch_torch.baseline_hours, rtol=1e-11)
        np.testing.assert_allclose(batch_np.ref_xi_arcsec, batch_torch.ref_xi_arcsec, atol=1e-9)
        np.testing.assert_allclose(batch_np.ref_eta_arcsec, batch_torch.ref_eta_arcsec, atol=1e-9)
        np.testing.assert_allclose(
            batch_np.v_ra_cosdec_arcsec_per_hr,
            batch_torch.v_ra_cosdec_arcsec_per_hr,
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            batch_np.v_dec_arcsec_per_hr,
            batch_torch.v_dec_arcsec_per_hr,
            rtol=1e-11,
        )

        states_np = project_detections_to_velocity_space(
            dets,
            config=cfg,
            filter_by_speed_bounds=True,
            backend="numpy",
        )
        states_torch = project_detections_to_velocity_space(
            dets,
            config=cfg,
            filter_by_speed_bounds=True,
            backend="torch",
        )
        assert len(states_np) == batch_np.pair_count
        assert len(states_torch) == batch_torch.pair_count
        assert [s.source_candidate_id for s in states_np] == [
            s.source_candidate_id for s in states_torch
        ]
        assert [s.target_candidate_id for s in states_np] == [
            s.target_candidate_id for s in states_torch
        ]
        assert batch_np.to_states() == states_np

    def test_high_declination_polar_gnomonic_projection_and_hemisphere_guard(self) -> None:
        """Verify spherical tangent-plane projection near +88.5 deg Dec and >90 deg rejection."""
        epochs = [2460755.50, 2460755.55, 2460755.60]
        polar_dets = _generate_linear_detections(
            prefix="polar",
            ra0_deg=45.0,
            dec0_deg=88.5,
            v_ra_cosdec_arcsec_per_hr=24.0,
            v_dec_arcsec_per_hr=-15.0,
            epochs_bjd=epochs,
        )
        states = project_detections_to_velocity_space(
            polar_dets,
            reference_bjd_tdb=epochs[0],
        )
        assert len(states) == 3
        for st in states:
            speed = math.hypot(st.v_ra_cosdec_arcsec_per_hr, st.v_dec_arcsec_per_hr)
            assert speed == pytest.approx(math.hypot(24.0, -15.0), rel=5e-3)

        # Antipodal detections (> 90 deg apart) cannot share a single gnomonic tangent plane
        antipodal = [
            CandidateDetection(
                candidate_id="north_hem",
                ra_deg=10.0,
                dec_deg=45.0,
                bjd_tdb=2460755.50,
            ),
            CandidateDetection(
                candidate_id="south_anti",
                ra_deg=190.0,
                dec_deg=-45.0,
                bjd_tdb=2460755.60,
            ),
        ]
        with pytest.raises(ValueError, match="hemisphere|tangent"):
            project_detections_to_velocity_space(antipodal)

    def test_ast_enforces_zero_nested_for_loops_and_zero_magic_unit_scalars(self) -> None:
        """Verify linking.py has no nested O(N^2) loops and no 24.0/48.0/3600.0 magic scalars."""
        linking_path = _REPO_ROOT / "src" / "LUMIA" / "physics" / "linking.py"
        tree = ast.parse(linking_path.read_text(encoding="utf-8"), filename=str(linking_path))

        # 1. Zero nested ast.For loops in velocity projection functions
        projection_funcs = {
            "project_detections_to_velocity_space",
            "project_detections_to_velocity_batch",
        }
        nested_for_violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in projection_funcs:
                for child in ast.walk(node):
                    if isinstance(child, ast.For):
                        inner_fors = [
                            sub
                            for sub in ast.walk(child)
                            if isinstance(sub, ast.For) and sub is not child
                        ]
                        if inner_fors:
                            nested_for_violations.append(
                                f"{node.name}:{child.lineno} contains nested for-loop"
                            )
        assert nested_for_violations == []

        # 2. Zero hardcoded 24.0 / 48.0 time-conversion literals outside Field defaults
        forbidden_scalars = {24.0, 48.0}
        scalar_violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Constant)
                        and isinstance(sub.value, (int, float))
                        and not isinstance(sub.value, bool)
                        and float(sub.value) in forbidden_scalars
                    ):
                        scalar_violations.append(
                            f"{node.name}:{sub.lineno} uses magic scalar {sub.value!r}"
                        )
        assert scalar_violations == []

    def test_equivalence_vectorized_and_naive_projection(self) -> None:
        """Vectorized NumPy and PyTorch implementations match naive O(N^2) reference loop."""
        epochs = [2460755.50, 2460755.54, 2460755.58, 2460755.62]
        rng = np.random.default_rng(42)

        # 2 true moving objects + 30 background detections
        dets = _generate_linear_detections(
            prefix="obj1",
            ra0_deg=100.0,
            dec0_deg=20.0,
            v_ra_cosdec_arcsec_per_hr=25.0,
            v_dec_arcsec_per_hr=-10.0,
            epochs_bjd=epochs,
        ) + _generate_linear_detections(
            prefix="obj2",
            ra0_deg=100.2,
            dec0_deg=20.1,
            v_ra_cosdec_arcsec_per_hr=-30.0,
            v_dec_arcsec_per_hr=15.0,
            epochs_bjd=epochs,
        )
        for idx in range(30):
            dets.append(
                CandidateDetection(
                    candidate_id=f"rand_{idx:02d}",
                    ra_deg=float(rng.uniform(99.8, 100.5)),
                    dec_deg=float(rng.uniform(19.8, 20.3)),
                    bjd_tdb=float(epochs[idx % len(epochs)]),
                    mag=21.0,
                    snr=8.0,
                )
            )

        cfg = TrackletLinkingConfig(
            min_baseline_hours=0.5,
            max_baseline_days=2.0,
            min_angular_speed_arcsec_per_hr=2.0,
            max_angular_speed_arcsec_per_hr=500.0,
        )

        for filter_speeds in (False, True):
            naive_states = _naive_pairwise_project(
                dets,
                config=cfg,
                filter_by_speed_bounds=filter_speeds,
            )
            np_states = project_detections_to_velocity_space(
                dets,
                config=cfg,
                filter_by_speed_bounds=filter_speeds,
                backend="numpy",
            )
            torch_states = project_detections_to_velocity_space(
                dets,
                config=cfg,
                filter_by_speed_bounds=filter_speeds,
                backend="torch",
            )

            assert len(naive_states) > 0
            assert len(naive_states) == len(np_states) == len(torch_states)

            for st_naive, st_np, st_torch in zip(
                naive_states, np_states, torch_states, strict=True
            ):
                assert st_naive.source_candidate_id == st_np.source_candidate_id == st_torch.source_candidate_id
                assert st_naive.target_candidate_id == st_np.target_candidate_id == st_torch.target_candidate_id
                assert st_naive.source_index == st_np.source_index == st_torch.source_index
                assert st_naive.target_index == st_np.target_index == st_torch.target_index
                assert st_np.baseline_hours == pytest.approx(st_naive.baseline_hours, rel=1e-10)
                assert st_torch.baseline_hours == pytest.approx(st_naive.baseline_hours, rel=1e-10)
                assert st_np.v_ra_cosdec_arcsec_per_hr == pytest.approx(
                    st_naive.v_ra_cosdec_arcsec_per_hr, rel=1e-10
                )
                assert st_torch.v_ra_cosdec_arcsec_per_hr == pytest.approx(
                    st_naive.v_ra_cosdec_arcsec_per_hr, rel=1e-10
                )
                assert st_np.v_dec_arcsec_per_hr == pytest.approx(
                    st_naive.v_dec_arcsec_per_hr, rel=1e-10
                )
                assert st_torch.v_dec_arcsec_per_hr == pytest.approx(
                    st_naive.v_dec_arcsec_per_hr, rel=1e-10
                )
                assert st_np.ref_xi_arcsec == pytest.approx(
                    st_naive.ref_xi_arcsec, rel=1e-9, abs=1e-9
                )
                assert st_torch.ref_xi_arcsec == pytest.approx(
                    st_naive.ref_xi_arcsec, rel=1e-9, abs=1e-9
                )
                assert st_np.ref_eta_arcsec == pytest.approx(
                    st_naive.ref_eta_arcsec, rel=1e-9, abs=1e-9
                )
                assert st_torch.ref_eta_arcsec == pytest.approx(
                    st_naive.ref_eta_arcsec, rel=1e-9, abs=1e-9
                )

    def test_vectorized_kinematic_escape_velocity_masking(self) -> None:
        """Tensor-level speed masks discard stationary, unbound escape-speed, and extreme pairs."""
        t0 = 2460755.50
        t1 = 2460755.55  # dt = 0.05 days = 1.2 hours
        dt_hr = 1.2

        cfg = TrackletLinkingConfig(
            min_baseline_hours=0.5,
            max_baseline_days=1.0,
            min_angular_speed_arcsec_per_hr=1.0,
            max_angular_speed_arcsec_per_hr=3600.0,
            assumed_heliocentric_distance_au=2.5,
            enforce_escape_velocity_bound=True,
        )
        esc_ceiling = compute_max_bound_angular_speed_arcsec_per_hr(2.5)
        assert 150.0 < esc_ceiling < 220.0

        # Construct pairs:
        # 1. Stationary pair (speed = 0.05" / 1.2hr ~ 0.04 arcsec/hr < min 1.0)
        # 2. Bound asteroid pair (speed = 42" / 1.2hr = 35.0 arcsec/hr, bound)
        # 3. Unbound pair (speed = 300" / 1.2hr = 250.0 arcsec/hr > esc_ceiling)
        # 4. Extreme pair (speed = 6000" / 1.2hr = 5000.0 arcsec/hr > max 3600.0)
        base_ra = 150.0
        base_dec = 10.0

        dets = [
            # Stationary (src, tgt)
            CandidateDetection(candidate_id="stat_0", ra_deg=base_ra, dec_deg=base_dec, bjd_tdb=t0),
            CandidateDetection(
                candidate_id="stat_1",
                ra_deg=base_ra + (0.05 / 3600.0),
                dec_deg=base_dec,
                bjd_tdb=t1,
            ),
            # Bound asteroid (src, tgt)
            CandidateDetection(candidate_id="bound_0", ra_deg=base_ra + 0.1, dec_deg=base_dec, bjd_tdb=t0),
            CandidateDetection(
                candidate_id="bound_1",
                ra_deg=base_ra + 0.1 + (42.0 / 3600.0),
                dec_deg=base_dec,
                bjd_tdb=t1,
            ),
            # Unbound (src, tgt)
            CandidateDetection(candidate_id="unbound_0", ra_deg=base_ra + 0.2, dec_deg=base_dec, bjd_tdb=t0),
            CandidateDetection(
                candidate_id="unbound_1",
                ra_deg=base_ra + 0.2 + (300.0 / 3600.0),
                dec_deg=base_dec,
                bjd_tdb=t1,
            ),
            # Extreme (src, tgt)
            CandidateDetection(candidate_id="extreme_0", ra_deg=base_ra + 0.3, dec_deg=base_dec, bjd_tdb=t0),
            CandidateDetection(
                candidate_id="extreme_1",
                ra_deg=base_ra + 0.3 + (6000.0 / 3600.0),
                dec_deg=base_dec,
                bjd_tdb=t1,
            ),
        ]

        for backend in ("numpy", "torch"):
            # With escape bound enforced: unbound and extreme are pruned
            batch_esc = project_detections_to_velocity_batch(
                dets,
                config=cfg,
                filter_by_speed_bounds=True,
                backend=backend,  # type: ignore[arg-type]
            )
            esc_pairs = list(zip(batch_esc.source_candidate_ids, batch_esc.target_candidate_ids, strict=True))
            assert ("bound_0", "bound_1") in esc_pairs
            assert ("stat_0", "stat_1") not in esc_pairs
            assert ("unbound_0", "unbound_1") not in esc_pairs
            assert ("extreme_0", "extreme_1") not in esc_pairs

            # Without escape bound enforced: unbound survives, extreme is pruned
            cfg_no_esc = cfg.model_copy(update={"enforce_escape_velocity_bound": False})
            batch_no_esc = project_detections_to_velocity_batch(
                dets,
                config=cfg_no_esc,
                filter_by_speed_bounds=True,
                backend=backend,  # type: ignore[arg-type]
            )
            no_esc_pairs = list(
                zip(batch_no_esc.source_candidate_ids, batch_no_esc.target_candidate_ids, strict=True)
            )
            assert ("bound_0", "bound_1") in no_esc_pairs
            assert ("unbound_0", "unbound_1") in no_esc_pairs
            assert ("stat_0", "stat_1") not in no_esc_pairs
            assert ("extreme_0", "extreme_1") not in no_esc_pairs

    def test_stress_benchmark_n1000_detections(self) -> None:
        """N=1000 multi-epoch detections scale sub-second with bounded memory."""
        rng = np.random.default_rng(202611)
        epochs = [2460755.50, 2460755.55, 2460755.60, 2460755.65]

        # 5 real moving objects (20 detections)
        all_dets: list[CandidateDetection] = []
        expected_pairs: set[tuple[str, str]] = set()

        for obj_idx in range(5):
            obj_dets = _generate_linear_detections(
                prefix=f"stress_obj_{obj_idx}",
                ra0_deg=180.0 + (obj_idx * 0.15),
                dec0_deg=10.0 + (obj_idx * 0.15),
                v_ra_cosdec_arcsec_per_hr=20.0 + (obj_idx * 8.0),
                v_dec_arcsec_per_hr=-15.0 + (obj_idx * 5.0),
                epochs_bjd=epochs,
            )
            all_dets.extend(obj_dets)
            # Add all pairwise combinations for this object
            for i in range(len(obj_dets)):
                for j in range(i + 1, len(obj_dets)):
                    expected_pairs.add((obj_dets[i].candidate_id, obj_dets[j].candidate_id))

        # 980 background noise detections spread across 4 epochs (total = 1000 detections)
        for idx in range(980):
            all_dets.append(
                CandidateDetection(
                    candidate_id=f"bg_stress_{idx:04d}",
                    ra_deg=float(rng.uniform(179.5, 181.5)),
                    dec_deg=float(rng.uniform(9.5, 11.5)),
                    bjd_tdb=float(epochs[idx % len(epochs)]),
                    mag=21.5,
                    snr=6.0,
                )
            )

        assert len(all_dets) == 1000

        cfg = TrackletLinkingConfig(
            min_baseline_hours=0.5,
            max_baseline_days=1.0,
            min_angular_speed_arcsec_per_hr=5.0,
            max_angular_speed_arcsec_per_hr=500.0,
        )

        # Benchmark NumPy backend
        t0 = time.perf_counter()
        batch_np = project_detections_to_velocity_batch(
            all_dets,
            config=cfg,
            filter_by_speed_bounds=True,
            backend="numpy",
        )
        elapsed_np = time.perf_counter() - t0
        assert elapsed_np < 1.0, f"NumPy backend on N=1000 took {elapsed_np:.3f}s (must be < 1.0s)"

        # Benchmark PyTorch backend
        t0 = time.perf_counter()
        batch_torch = project_detections_to_velocity_batch(
            all_dets,
            config=cfg,
            filter_by_speed_bounds=True,
            backend="torch",
        )
        elapsed_torch = time.perf_counter() - t0
        assert elapsed_torch < 1.0, f"Torch backend on N=1000 took {elapsed_torch:.3f}s (must be < 1.0s)"

        # Structural & numerical integrity
        assert batch_np.pair_count == batch_torch.pair_count
        assert batch_np.pair_count > 0
        assert np.all(np.isfinite(batch_np.v_ra_cosdec_arcsec_per_hr))
        assert np.all(np.isfinite(batch_np.v_dec_arcsec_per_hr))
        assert np.all(np.isfinite(batch_np.ref_xi_arcsec))
        assert np.all(np.isfinite(batch_np.ref_eta_arcsec))

        # Check all real object pairs were preserved
        recovered_pairs = set(
            zip(batch_np.source_candidate_ids, batch_np.target_candidate_ids, strict=True)
        )
        assert expected_pairs.issubset(recovered_pairs)


class TestEndToEndLinkingProvenanceAndIsolation:
    """Verify per-epoch residual population, CNN score gating, and zero DB coupling."""

    def test_link_tracklets_populates_residuals_immutably_and_honors_min_cnn_score(
        self,
    ) -> None:
        """Linked Tracklet detections carry astrometric_residual_arcsec without mutating inputs."""
        epochs = [2460755.50, 2460755.54, 2460755.58, 2460755.62]
        real_ast = [
            d.model_copy(update={"cnn_score": 0.93, "model_version": "rb-1.0"})
            for d in _generate_linear_detections(
                prefix="real_neo",
                ra0_deg=88.0,
                dec0_deg=22.0,
                v_ra_cosdec_arcsec_per_hr=68.0,
                v_dec_arcsec_per_hr=-25.0,
                epochs_bjd=epochs,
            )
        ]
        bogus_artifact_chain = [
            d.model_copy(update={"cnn_score": 0.12, "model_version": "rb-1.0"})
            for d in _generate_linear_detections(
                prefix="bogus_spike",
                ra0_deg=88.25,
                dec0_deg=22.15,
                v_ra_cosdec_arcsec_per_hr=-40.0,
                v_dec_arcsec_per_hr=18.0,
                epochs_bjd=epochs,
            )
        ]

        # Also add an inferior off-track candidate at epoch 1 near real_neo to test epoch dedup
        ambiguous_ep1 = real_ast[1].model_copy(
            update={
                "candidate_id": "real_neo_ep1_off",
                "dec_deg": real_ast[1].dec_deg + (0.8 / 3600.0),
            }
        )

        combined = [*real_ast, *bogus_artifact_chain, ambiguous_ep1]
        cfg = TrackletLinkingConfig(
            min_epochs=4,
            min_cnn_score=0.75,
            position_eps_arcsec=2.0,
            velocity_eps_arcsec_per_hr=2.0,
        )
        linked = link_tracklets(combined, config=cfg)

        assert len(linked) == 1
        trk = linked[0]
        assert trk.tracklet_id == "trk_0001"
        assert [d.candidate_id for d in trk.detections] == [d.candidate_id for d in real_ast]

        # Caller's input objects retain astrometric_residual_arcsec=None
        assert all(d.astrometric_residual_arcsec is None for d in real_ast)
        # Linked Tracklet detections have non-negative astrometric_residual_arcsec populated
        assert all(
            d.astrometric_residual_arcsec is not None and d.astrometric_residual_arcsec >= 0.0
            for d in trk.detections
        )

    def test_kinematic_validation_rejects_jitter_duplicate_epochs_and_short_arcs(
        self,
    ) -> None:
        """Verify RMS residual ceiling, duplicate-epoch guard, and min_epochs enforcement."""
        epochs = [2460755.50, 2460755.55, 2460755.60, 2460755.65]
        base = _generate_linear_detections(
            prefix="jitter",
            ra0_deg=150.0,
            dec0_deg=5.0,
            v_ra_cosdec_arcsec_per_hr=30.0,
            v_dec_arcsec_per_hr=10.0,
            epochs_bjd=epochs,
        )
        # Displace epoch 1 perpendicular to motion by 4 arcsec without large net curvature
        # by offsetting ep1 (+4") and ep2 (-4") alternately -> huge RMS residual
        jittered = [
            base[0],
            base[1].model_copy(update={"dec_deg": base[1].dec_deg + 4.0 / 3600.0}),
            base[2].model_copy(update={"dec_deg": base[2].dec_deg - 4.0 / 3600.0}),
            base[3],
        ]
        res_jitter = validate_tracklet_kinematics(
            jittered,
            config=TrackletLinkingConfig(
                max_angular_accel_arcsec_per_day2=5000.0,
                max_rms_residual_arcsec=0.5,
            ),
        )
        assert res_jitter.is_valid is False
        assert res_jitter.rejection_reason is not None
        assert "residual" in res_jitter.rejection_reason.lower()
        assert len(res_jitter.per_epoch_residuals_arcsec) == 4

        # Duplicate epoch rejection
        dup_epoch = [base[0], base[0].model_copy(update={"candidate_id": "dup_t0"}), base[2]]
        res_dup = validate_tracklet_kinematics(dup_epoch)
        assert res_dup.is_valid is False
        assert res_dup.rejection_reason is not None
        assert "simultaneous" in res_dup.rejection_reason.lower() or (
            "duplicate" in res_dup.rejection_reason.lower()
        )

        # Short arc (< min_epochs) rejection
        res_short = validate_tracklet_kinematics(base[:2])
        assert res_short.is_valid is False
        assert res_short.rejection_reason is not None
        assert "insufficient" in res_short.rejection_reason.lower()

    def test_module_5_contains_zero_database_or_orm_imports(self) -> None:
        """AST guard: linking.py must contain zero persistence, ORM, or web imports."""
        linking_path = _REPO_ROOT / "src" / "LUMIA" / "physics" / "linking.py"
        forbidden = (
            "duckdb",
            "sqlalchemy",
            "psycopg",
            "sqlite3",
            "prefect",
            "fastapi",
            "LUMIA.data_pipeline",
            "LUMIA.ml",
            "LUMIA.api",
        )
        tree = ast.parse(linking_path.read_text(encoding="utf-8"), filename=str(linking_path))
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == f or alias.name.startswith(f"{f}.")
                        for f in forbidden
                    ):
                        violations.append(f"line {node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if any(mod == f or mod.startswith(f"{f}.") for f in forbidden):
                    violations.append(f"line {node.lineno}: from {mod}")
        assert violations == []

