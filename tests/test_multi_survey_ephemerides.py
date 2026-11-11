"""Unit and integration tests for SPICE ephemerides and planetary retarded-time solver."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from LUMIA.core.schemas import GroundObservatoryProfile, SpacecraftObservatoryProfile
from LUMIA.physics.ephemerides import (
    C_AU_PER_DAY,
    PlanetaryRetardedTimeSolver,
    SPICEKernelManager,
)


def test_planetary_retarded_time_solver_fixed_point_convergence() -> None:
    """Test convergence of retarded-time solver for a moving target."""
    # Target is an asteroid at ~1 AU moving along Y at 30 km/s (~0.01735 AU/day)
    v_target_y = 30.0 * 86400.0 / 149597870.7  # AU/day
    r0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    t_obs = 60000.0  # MJD

    def target_pos_func(tau: float) -> np.ndarray:
        dt = tau - t_obs
        return r0 + np.array([0.0, v_target_y * dt, 0.0], dtype=np.float64)

    # Observer at Earth (0, 0, 0)
    obs_pos_earth = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    tau_earth, lt_earth_sec = PlanetaryRetardedTimeSolver.solve(
        t_obs, obs_pos_earth, target_pos_func, tol_sec=1e-7
    )

    # Light travel time for 1 AU is ~499.00478 seconds
    expected_lt = 1.0 / C_AU_PER_DAY * 86400.0
    assert lt_earth_sec == pytest.approx(expected_lt, rel=1e-3)
    assert tau_earth < t_obs
    assert (t_obs - tau_earth) * 86400.0 == pytest.approx(lt_earth_sec, abs=1e-5)

    # Observer at L2 (~1.5e6 km further along X: ~0.0100268 AU)
    l2_baseline_au = 1500000.0 / 149597870.7  # ~0.0100268 AU
    obs_pos_l2 = np.array([-l2_baseline_au, 0.0, 0.0], dtype=np.float64)
    tau_l2, lt_l2_sec = PlanetaryRetardedTimeSolver.solve(
        t_obs, obs_pos_l2, target_pos_func, tol_sec=1e-7
    )

    # Differential light time between Earth and L2 baseline (~5.003 seconds)
    delta_lt_sec = lt_l2_sec - lt_earth_sec
    assert delta_lt_sec == pytest.approx(5.003, abs=0.05)


def test_planetary_retarded_time_solver_max_iter() -> None:
    """Ensure solver terminates safely even if max_iter is 1."""
    obs_pos = np.array([0.0, 0.0, 0.0])
    target_func = lambda tau: np.array([1.0, 0.0, 0.0])
    tau, lt_sec = PlanetaryRetardedTimeSolver.solve(60000.0, obs_pos, target_func, max_iter=1)
    assert tau < 60000.0
    assert lt_sec > 400.0


def test_spice_kernel_manager_thread_safety() -> None:
    """Verify SPICEKernelManager lock behavior."""
    mgr = SPICEKernelManager()
    assert mgr._lock is not None

    with patch("spiceypy.spkezr") as mock_spkezr:
        # Mock SPICE returning state vector in km and km/s
        mock_spkezr.return_value = (
            [149597870.7, 0.0, 0.0, 0.0, 29.78, 0.0],
            0.0,
        )
        pos, vel = mgr.get_observer_bcrs_state(-170, 0.0)
        assert pos[0] == pytest.approx(1.0, rel=1e-6)
        assert vel[1] == pytest.approx(29.78 * 86400.0 / 149597870.7, rel=1e-6)
        mock_spkezr.assert_called_once()


def test_spice_kernel_manager_ground_observer_state() -> None:
    """Verify ground observer state incorporates Earth barycenter and topocentric rotation."""
    mgr = SPICEKernelManager()
    profile = GroundObservatoryProfile(
        mpc_code="W84",
        name="Cerro Pachon / Rubin",
        latitude_deg=-30.2446,
        longitude_deg=-70.7494,
        elevation_meters=2663.0,
        pixel_scale_arcsec_per_px=0.2,
        default_filter="r",
    )

    with patch.object(mgr, "get_observer_bcrs_state") as mock_earth_state:
        # Earth center at 1 AU
        mock_earth_state.return_value = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.017, 0.0]),
        )
        pos, vel = mgr.get_ground_observer_bcrs_state(profile, epoch_et=0.0)
        # Position should be ~1 AU with a tiny topocentric shift (~6370 km / 1.5e8 km ~ 4e-5 AU)
        assert pos[0] == pytest.approx(1.0, abs=1e-3)
        assert np.linalg.norm(pos - np.array([1.0, 0.0, 0.0])) < 1e-3


def test_spice_kernel_manager_missing_kernel_raises() -> None:
    """Loading non-existent kernel raises FileNotFoundError."""
    mgr = SPICEKernelManager(kernel_dir="non_existent_spice_dir")
    with pytest.raises(FileNotFoundError, match="SPICE kernel not found"):
        mgr.load_kernel_chain(["de440.bsp"])
