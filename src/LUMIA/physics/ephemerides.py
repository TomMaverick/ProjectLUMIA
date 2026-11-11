"""Thread-safe NAIF SPICE ephemerides manager and planetary retarded-time solver."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import spiceypy as spice

if TYPE_CHECKING:
    from LUMIA.core.schemas import GroundObservatoryProfile

__all__ = [
    "C_AU_PER_DAY",
    "C_KM_PER_S",
    "KM_PER_AU",
    "PlanetaryRetardedTimeSolver",
    "SPICEKernelManager",
]

KM_PER_AU: float = 149597870.7
C_KM_PER_S: float = 299792.458
C_AU_PER_DAY: float = (C_KM_PER_S * 86400.0) / KM_PER_AU  # ~173.144632674240 AU/day
WGS84_A_KM: float = 6378.137
WGS84_F: float = 1.0 / 298.257223563


class SPICEKernelManager:
    """Thread-safe and process-guarded manager for NAIF SPICE kernels."""

    _lock = threading.Lock()

    def __init__(self, kernel_dir: Path | str | None = None) -> None:
        self.kernel_dir = Path(kernel_dir) if kernel_dir is not None else Path("data/spice")
        self.loaded_kernels: set[str] = set()

    def load_kernel_chain(self, kernel_names: list[str]) -> None:
        """Loads an ordered list of SPICE kernels under the process-wide lock."""
        with self._lock:
            for name in kernel_names:
                kernel_path = self.kernel_dir / name
                if not kernel_path.exists():
                    msg = f"SPICE kernel not found: {kernel_path.resolve()}"
                    raise FileNotFoundError(msg)
                resolved = str(kernel_path.resolve())
                if resolved not in self.loaded_kernels:
                    spice.furnsh(resolved)
                    self.loaded_kernels.add(resolved)

    def load_kernel_file(self, kernel_path: Path | str) -> None:
        """Loads a single SPICE kernel file by direct path."""
        p = Path(kernel_path).resolve()
        if not p.exists():
            msg = f"SPICE kernel not found: {p}"
            raise FileNotFoundError(msg)
        with self._lock:
            resolved = str(p)
            if resolved not in self.loaded_kernels:
                spice.furnsh(resolved)
                self.loaded_kernels.add(resolved)

    def unload_all(self) -> None:
        """Unloads all kernels loaded in CSPICE and resets tracker."""
        with self._lock:
            spice.kclear()
            self.loaded_kernels.clear()

    def get_observer_bcrs_state(
        self,
        naif_id: int | str,
        epoch_et: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns BCRS observer state (position [AU], velocity [AU/day]) relative to SSB."""
        with self._lock:
            state, _ = spice.spkezr(str(naif_id), epoch_et, "J2000", "NONE", "0")
        pos_au = np.array(state[:3], dtype=np.float64) / KM_PER_AU
        vel_au_day = (np.array(state[3:], dtype=np.float64) * 86400.0) / KM_PER_AU
        return pos_au, vel_au_day

    def get_ground_observer_bcrs_state(
        self,
        profile: GroundObservatoryProfile,
        epoch_et: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates BCRS observer state for a ground-based station.

        Combines Earth's barycentric state (NAIF 399) with geodetic topocentric offset.
        """
        earth_pos_au, earth_vel_au_day = self.get_observer_bcrs_state(399, epoch_et)

        # Geodetic to Earth-Centered coordinates (WGS-84)
        lat_rad = math.radians(profile.latitude_deg)
        lon_rad = math.radians(profile.longitude_deg)
        h_km = profile.elevation_meters / 1000.0

        e2 = 2.0 * WGS84_F - WGS84_F**2
        n = WGS84_A_KM / math.sqrt(1.0 - e2 * math.sin(lat_rad) ** 2)
        x_ecef = (n + h_km) * math.cos(lat_rad) * math.cos(lon_rad)
        y_ecef = (n + h_km) * math.cos(lat_rad) * math.sin(lon_rad)
        z_ecef = (n * (1.0 - e2) + h_km) * math.sin(lat_rad)

        # Greenwich Mean Sidereal Time (GMST) rotation
        # J2000 epoch is ET = 0.0, JD 2451545.0
        d_days = epoch_et / 86400.0
        theta_gmst = math.radians((280.46061837 + 360.98564736629 * d_days) % 360.0)

        # Earth rotation rate in radians per day
        omega_earth = 2.0 * math.pi * 1.00273790935  # rad/day

        # Rotate from Earth-Fixed to J2000 equatorial plane
        # Note: True obliquity of ecliptic eps ~ 23.439291 deg
        cos_t = math.cos(theta_gmst)
        sin_t = math.sin(theta_gmst)

        rx_km = cos_t * x_ecef - sin_t * y_ecef
        ry_km = sin_t * x_ecef + cos_t * y_ecef
        rz_km = z_ecef

        topo_pos_au = np.array([rx_km, ry_km, rz_km], dtype=np.float64) / KM_PER_AU

        # Topocentric rotational velocity (d/dt of position in inertial frame)
        vx_km_day = (-sin_t * x_ecef - cos_t * y_ecef) * omega_earth
        vy_km_day = (cos_t * x_ecef - sin_t * y_ecef) * omega_earth
        vz_km_day = 0.0

        topo_vel_au_day = np.array([vx_km_day, vy_km_day, vz_km_day], dtype=np.float64) / KM_PER_AU

        return earth_pos_au + topo_pos_au, earth_vel_au_day + topo_vel_au_day

    @staticmethod
    def jd_to_et(jd: float) -> float:
        """Converts Julian Date to Ephemeris Time (seconds past J2000)."""
        return (jd - 2451545.0) * 86400.0

    @staticmethod
    def et_to_jd(et: float) -> float:
        """Converts Ephemeris Time (seconds past J2000) to Julian Date."""
        return 2451545.0 + et / 86400.0

    def utc_to_et(self, utc_str: str) -> float:
        """Converts UTC string to Ephemeris Time via SPICE or astropy fallback."""
        with self._lock:
            try:
                return float(spice.str2et(utc_str))
            except Exception:
                pass
        from astropy.time import Time

        t = Time(utc_str, scale="utc")
        return self.jd_to_et(t.jd)


class PlanetaryRetardedTimeSolver:
    """Fixed-point solver for finite-range planetary retarded-time in BCRS.

    Solves the implicit light-travel time equation:
    tau = t_obs - ||r_obj(tau) - x_obs(t_obs)|| / c
    """

    C_AU_PER_DAY = C_AU_PER_DAY

    @classmethod
    def solve(
        cls,
        t_obs_mjd: float,
        obs_pos_au: np.ndarray,
        target_pos_func: Callable[[float], np.ndarray],
        max_iter: int = 10,
        tol_sec: float = 1e-6,
    ) -> tuple[float, float]:
        """Solves tau = t_obs - ||r(tau) - x_obs(t_obs)|| / c.

        Parameters
        ----------
        t_obs_mjd : float
            Observation epoch in MJD or Julian Days.
        obs_pos_au : np.ndarray
            3D barycentric observer position at observation epoch [AU].
        target_pos_func : Callable[[float], np.ndarray]
            Function taking epoch tau (same time scale as t_obs_mjd) and returning
            3D barycentric target position [AU].
        max_iter : int
            Maximum number of fixed-point iterations.
        tol_sec : float
            Convergence tolerance in seconds.

        Returns
        -------
        tau : float
            Retarded emission epoch in the same time scale as t_obs_mjd.
        light_travel_time_sec : float
            One-way light-travel time differential in seconds.
        """
        tol_days = tol_sec / 86400.0
        tau = float(t_obs_mjd)
        obs_pos = np.asarray(obs_pos_au, dtype=np.float64)

        for _ in range(max_iter):
            r_target = np.asarray(target_pos_func(tau), dtype=np.float64)
            range_au = float(np.linalg.norm(r_target - obs_pos))
            dt_days = range_au / cls.C_AU_PER_DAY
            tau_next = float(t_obs_mjd - dt_days)
            if abs(tau_next - tau) < tol_days:
                return tau_next, dt_days * 86400.0
            tau = tau_next

        r_target = np.asarray(target_pos_func(tau), dtype=np.float64)
        range_au = float(np.linalg.norm(r_target - obs_pos))
        dt_days = range_au / cls.C_AU_PER_DAY
        return tau, dt_days * 86400.0
