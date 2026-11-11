"""Two-Branch HelioLinC3D linking engine with SNR-weighted chi-squared astrometric gating."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy import stats

from LUMIA.core.schemas import CandidateDetection

__all__ = [
    "DynamicalAssociationBranch",
    "HelioLinC3DEngine",
    "StereoscopicTriangulationBranch",
    "compute_snr_weighted_chi2",
    "propagate_kepler_universal",
    "radec_to_unit_vector",
    "stumpff_c2",
    "stumpff_c3",
    "unit_vector_to_radec",
]

# Gravitational parameter of the Sun: mu = k^2 in AU^3 / day^2
MU_SUN_AU_DAY: float = 0.01720209895**2  # ~0.0002959122082855911
EARTH_L2_MIN_BASELINE_AU: float = 500000.0 / 149597870.7  # ~0.003342 AU (500,000 km)
ONE_HOUR_DAYS: float = 1.0 / 24.0


def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Converts ICRS Right Ascension and Declination (in degrees) to a 3D unit vector."""
    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)
    cos_dec = math.cos(dec_rad)
    return np.array(
        [
            cos_dec * math.cos(ra_rad),
            cos_dec * math.sin(ra_rad),
            math.sin(dec_rad),
        ],
        dtype=np.float64,
    )


def unit_vector_to_radec(u_unit: np.ndarray) -> tuple[float, float]:
    """Converts a 3D unit vector to ICRS Right Ascension and Declination (in degrees)."""
    u = np.asarray(u_unit, dtype=np.float64)
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        msg = "Cannot convert zero-length vector to celestial coordinates."
        raise ValueError(msg)
    u_norm = u / norm
    ra_deg = math.degrees(math.atan2(u_norm[1], u_norm[0])) % 360.0
    dec_deg = math.degrees(math.asin(np.clip(u_norm[2], -1.0, 1.0)))
    return ra_deg, dec_deg


def stumpff_c2(z: float) -> float:
    """Computes Stumpff function c2(z) = (1 - cos(sqrt(z))) / z."""
    if z > 1e-6:
        sqrt_z = math.sqrt(z)
        return (1.0 - math.cos(sqrt_z)) / z
    if z < -1e-6:
        sqrt_neg_z = math.sqrt(-z)
        return (math.cosh(sqrt_neg_z) - 1.0) / (-z)
    # Series expansion around z=0
    return 0.5 - z / 24.0 + (z**2) / 720.0 - (z**3) / 40320.0


def stumpff_c3(z: float) -> float:
    """Computes Stumpff function c3(z) = (sqrt(z) - sin(sqrt(z))) / (z^(3/2))."""
    if z > 1e-6:
        sqrt_z = math.sqrt(z)
        return (sqrt_z - math.sin(sqrt_z)) / (z * sqrt_z)
    if z < -1e-6:
        sqrt_neg_z = math.sqrt(-z)
        return (math.sinh(sqrt_neg_z) - sqrt_neg_z) / ((-z) * sqrt_neg_z)
    # Series expansion around z=0
    return 1.0 / 6.0 - z / 120.0 + (z**2) / 5040.0 - (z**3) / 362880.0


def propagate_kepler_universal(
    r0_au: np.ndarray,
    v0_au_day: np.ndarray,
    dt_days: float,
    mu: float = MU_SUN_AU_DAY,
    max_iter: int = 50,
    tol: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagates a 2-body Keplerian state vector using universal variable formulation.

    Parameters
    ----------
    r0_au : np.ndarray
        Initial heliocentric position vector [AU].
    v0_au_day : np.ndarray
        Initial heliocentric velocity vector [AU/day].
    dt_days : float
        Propagation time delta [days].
    mu : float
        Gravitational parameter [AU^3 / day^2].
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance on universal anomaly chi.

    Returns
    -------
    r_au : np.ndarray
        Propagated position [AU].
    v_au_day : np.ndarray
        Propagated velocity [AU/day].
    """
    if abs(dt_days) < 1e-12:
        return np.copy(r0_au), np.copy(v0_au_day)

    r0 = float(np.linalg.norm(r0_au))
    v0 = float(np.linalg.norm(v0_au_day))
    if r0 < 1e-8:
        msg = "Initial radius too close to barycenter origin."
        raise ValueError(msg)

    r0_dot_v0 = float(np.dot(r0_au, v0_au_day))
    # alpha = 1 / a (reciprocal semi-major axis)
    alpha = (2.0 / r0) - (v0**2 / mu)
    sqrt_mu = math.sqrt(mu)

    # Initial guess for universal anomaly chi
    if alpha > 1e-6:
        # Elliptic orbit
        chi = sqrt_mu * dt_days * alpha
    elif alpha < -1e-6:
        # Hyperbolic orbit
        a = 1.0 / alpha
        term = -2.0 * mu * dt_days / (a * (r0_dot_v0 + math.copysign(math.sqrt(-mu * a), dt_days)))
        chi = math.copysign(math.sqrt(-a), dt_days) * math.log(max(1e-12, abs(term)))
    else:
        # Parabolic orbit
        h_vec = np.cross(r0_au, v0_au_day)
        p = float(np.dot(h_vec, h_vec)) / mu
        s = 0.5 * (math.pi / 2.0 - math.atan(3.0 * math.sqrt(mu / (p**3)) * dt_days))
        w = math.atan((math.tan(s)) ** (1.0 / 3.0))
        chi = math.sqrt(p) * (2.0 / math.tan(2.0 * w))

    # Newton-Raphson iteration
    for _ in range(max_iter):
        z = alpha * (chi**2)
        c2 = stumpff_c2(z)
        c3 = stumpff_c3(z)

        # Kepler's universal equation F(chi)
        f_val = (r0_dot_v0 / sqrt_mu) * (chi**2) * c2 + (1.0 - alpha * r0) * (chi**3) * c3 + r0 * chi - sqrt_mu * dt_days
        # Derivative dF/dchi
        df_val = (r0_dot_v0 / sqrt_mu) * chi * (1.0 - z * c3) + (1.0 - alpha * r0) * (chi**2) * c2 + r0

        delta_chi = f_val / df_val
        chi -= delta_chi
        if abs(delta_chi) < tol:
            break

    z = alpha * (chi**2)
    c2 = stumpff_c2(z)
    c3 = stumpff_c3(z)

    # Lagrange f and g functions
    f = 1.0 - (chi**2 / r0) * c2
    g = dt_days - (chi**3 / sqrt_mu) * c3

    r_next = f * r0_au + g * v0_au_day
    r = float(np.linalg.norm(r_next))

    # Time derivatives of f and g
    f_dot = (sqrt_mu / (r * r0)) * (alpha * (chi**3) * c3 - chi)
    g_dot = 1.0 - (chi**2 / r) * c2

    v_next = f_dot * r0_au + g_dot * v0_au_day
    return r_next, v_next


class StereoscopicTriangulationBranch:
    """Direct 3D position determination across Earth-space interplanetary baselines."""

    @staticmethod
    def triangulate(
        obs1_pos_au: np.ndarray,
        u1_unit: np.ndarray,
        obs2_pos_au: np.ndarray,
        u2_unit: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Calculates closest point of approach between two skew sightlines.

        Parameters
        ----------
        obs1_pos_au : np.ndarray
            3D barycentric position of observer 1 [AU].
        u1_unit : np.ndarray
            Line-of-sight unit vector from observer 1.
        obs2_pos_au : np.ndarray
            3D barycentric position of observer 2 [AU].
        u2_unit : np.ndarray
            Line-of-sight unit vector from observer 2.

        Returns
        -------
        midpoint_au : np.ndarray
            Heliocentric 3D midpoint of closest approach [AU].
        separation_au : float
            Euclidean miss distance between the two skew sightlines [AU].
        """
        u1 = np.asarray(u1_unit, dtype=np.float64) / np.linalg.norm(u1_unit)
        u2 = np.asarray(u2_unit, dtype=np.float64) / np.linalg.norm(u2_unit)
        w0 = np.asarray(obs1_pos_au - obs2_pos_au, dtype=np.float64)

        a = float(np.dot(u1, u1))
        b = float(np.dot(u1, u2))
        c = float(np.dot(u2, u2))
        d = float(np.dot(u1, w0))
        e = float(np.dot(u2, w0))

        denom = a * c - b * b
        if abs(denom) < 1e-9:
            msg = "Sightlines are parallel; triangulation degenerate."
            raise ValueError(msg)

        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom

        p1 = obs1_pos_au + sc * u1
        p2 = obs2_pos_au + tc * u2

        midpoint = 0.5 * (p1 + p2)
        separation_au = float(np.linalg.norm(p1 - p2))
        return midpoint, separation_au


class DynamicalAssociationBranch:
    """Multi-day arc association branch using universal Keplerian f and g propagation."""

    @staticmethod
    def propagate_and_predict_radec(
        r0_au: np.ndarray,
        v0_au_day: np.ndarray,
        t0_mjd: float,
        target_mjd: float,
        target_obs_pos_au: np.ndarray,
    ) -> tuple[float, float, float]:
        """Propagates state vector to target epoch and computes predicted (RA, Dec) and range.

        Returns
        -------
        ra_pred_deg : float
        dec_pred_deg : float
        range_pred_au : float
        """
        dt_days = target_mjd - t0_mjd
        r_helio, _ = propagate_kepler_universal(r0_au, v0_au_day, dt_days)
        rho_vec = r_helio - np.asarray(target_obs_pos_au, dtype=np.float64)
        range_au = float(np.linalg.norm(rho_vec))
        if range_au < 1e-8:
            msg = "Target predicted at observer position."
            raise ValueError(msg)
        u_unit = rho_vec / range_au
        ra_deg, dec_deg = unit_vector_to_radec(u_unit)
        return ra_deg, dec_deg, range_au


def compute_snr_weighted_chi2(
    observed_radec: np.ndarray,
    predicted_radec: np.ndarray,
    snrs: np.ndarray,
    fwhm_arcsec: float | np.ndarray = 1.5,
    sigma_sys_arcsec: float = 0.05,
    sigma_dcr_arcsec: float = 0.02,
    num_fit_params: int = 4,
    alpha: float = 0.01,
) -> tuple[float, float, bool]:
    """Evaluates an SNR-weighted chi-squared test for candidate tracklet astrometry.

    Accounts for photon-noise centroid limits (sigma_centroid ~ FWHM / (2 * SNR)),
    systematic astrometric floor, and atmospheric DCR:
    sigma_total^2 = (FWHM / (2 * SNR))^2 + sigma_sys^2 + sigma_dcr^2

    Parameters
    ----------
    observed_radec : np.ndarray
        Array of shape (N, 2) of observed [RA, Dec] in degrees.
    predicted_radec : np.ndarray
        Array of shape (N, 2) of predicted [RA, Dec] in degrees.
    snrs : np.ndarray
        Array of shape (N,) of detection SNRs.
    fwhm_arcsec : float | np.ndarray
        Observing seeing FWHM in arcseconds.
    sigma_sys_arcsec : float
        Systematic WCS registration error [arcsec].
    sigma_dcr_arcsec : float
        Atmospheric differential chromatic refraction uncertainty [arcsec].
    num_fit_params : int
        Number of motion model parameters (typically 4 for linear tracklet, 6 for orbit).
    alpha : float
        Significance level for chi-squared rejection.

    Returns
    -------
    chi2_val : float
        Total chi-squared statistic.
    reduced_chi2 : float
        Reduced chi-squared (chi2 / dof).
    is_accepted : bool
        True if the candidate association passes the chi-squared gate.
    """
    obs = np.asarray(observed_radec, dtype=np.float64)
    pred = np.asarray(predicted_radec, dtype=np.float64)
    snr_arr = np.maximum(np.asarray(snrs, dtype=np.float64), 0.1)

    n_points = obs.shape[0]
    if n_points < 2:
        return 0.0, 0.0, True

    fwhm_arr = np.broadcast_to(np.asarray(fwhm_arcsec, dtype=np.float64), (n_points,))

    # Positional centroiding error from photon noise (Cramer-Rao lower bound)
    sigma_centroid = fwhm_arr / (2.0 * snr_arr)
    # Total positional variance per axis in arcseconds
    sigma_total_sq = sigma_centroid**2 + sigma_sys_arcsec**2 + sigma_dcr_arcsec**2

    # Residuals in arcseconds: RA residual scaled by cos(Dec)
    dec_rad = np.radians(obs[:, 1])
    delta_ra_arcsec = (obs[:, 0] - pred[:, 0]) * np.cos(dec_rad) * 3600.0
    delta_dec_arcsec = (obs[:, 1] - pred[:, 1]) * 3600.0

    delta_theta_sq = delta_ra_arcsec**2 + delta_dec_arcsec**2

    # 2 coordinates (RA, Dec) per observation epoch
    chi2_val = float(np.sum(delta_theta_sq / sigma_total_sq))
    dof = max(1, 2 * n_points - num_fit_params)
    reduced_chi2 = chi2_val / dof

    crit_chi2 = float(stats.chi2.ppf(1.0 - alpha, dof))
    is_accepted = bool(chi2_val <= crit_chi2 or reduced_chi2 <= 3.0)

    return chi2_val, reduced_chi2, is_accepted


class HelioLinC3DEngine:
    """Unified HelioLinC3D linking engine bifurcating stereoscopic and dynamical branches."""

    def __init__(
        self,
        min_space_baseline_au: float = EARTH_L2_MIN_BASELINE_AU,
        max_stereo_dt_days: float = ONE_HOUR_DAYS,
        fwhm_arcsec: float = 1.5,
    ) -> None:
        self.min_space_baseline_au = min_space_baseline_au
        self.max_stereo_dt_days = max_stereo_dt_days
        self.fwhm_arcsec = fwhm_arcsec

    def determine_branch(
        self,
        det1: CandidateDetection,
        det2: CandidateDetection,
    ) -> str:
        """Determines whether a detection pair routes to stereoscopic or dynamical branch."""
        dt_days = abs(det2.bjd_tdb - det1.bjd_tdb)
        pos1 = det1.observer_pos_au
        pos2 = det2.observer_pos_au

        if pos1 is not None and pos2 is not None:
            baseline_au = float(
                np.linalg.norm(np.array(pos2, dtype=np.float64) - np.array(pos1, dtype=np.float64))
            )
            if dt_days <= self.max_stereo_dt_days and baseline_au >= self.min_space_baseline_au:
                return "stereoscopic"

        return "dynamical"

    def triangulate_pair(
        self,
        det1: CandidateDetection,
        det2: CandidateDetection,
    ) -> tuple[np.ndarray, float]:
        """Triangulates simultaneous detections across ground and space observatories."""
        if det1.observer_pos_au is None or det2.observer_pos_au is None:
            msg = "Observer state vectors required for stereoscopic triangulation."
            raise ValueError(msg)

        u1 = radec_to_unit_vector(det1.ra_deg, det1.dec_deg)
        u2 = radec_to_unit_vector(det2.ra_deg, det2.dec_deg)
        obs1 = np.array(det1.observer_pos_au, dtype=np.float64)
        obs2 = np.array(det2.observer_pos_au, dtype=np.float64)

        return StereoscopicTriangulationBranch.triangulate(obs1, u1, obs2, u2)

    def validate_tracklet(
        self,
        detections: Sequence[CandidateDetection],
        predicted_radec: np.ndarray,
    ) -> tuple[float, float, bool]:
        """Validates tracklet astrometry using SNR-weighted chi-squared gate."""
        obs_radec = np.array([[d.ra_deg, d.dec_deg] for d in detections], dtype=np.float64)
        snrs = np.array([d.snr if d.snr is not None else 5.0 for d in detections], dtype=np.float64)
        return compute_snr_weighted_chi2(
            obs_radec,
            predicted_radec,
            snrs,
            fwhm_arcsec=self.fwhm_arcsec,
        )
