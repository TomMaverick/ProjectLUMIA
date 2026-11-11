"""Module 5: HelioLinC-inspired space-velocity tracklet linking and kinematic validation."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Literal

import astropy.constants as const
import astropy.units as u
import numpy as np
import torch
from pydantic import ConfigDict, Field, model_validator
from scipy.spatial import cKDTree

from LUMIA.core.schemas import (
    CandidateDetection,
    LumiaBaseModel,
)
from LUMIA.physics.heliolinc3d import HelioLinC3DEngine

if TYPE_CHECKING:
    from collections.abc import Sequence

ProjectionBackend = Literal["numpy", "torch"]


class TrackletLinkingConfig(LumiaBaseModel):
    """Configuration parameters governing space-velocity clustering and kinematic cuts."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    min_epochs: int = Field(
        default=3,
        ge=3,
        description="Minimum distinct observation epochs required to form a valid tracklet.",
    )
    min_baseline_hours: float = Field(
        default=0.05,
        gt=0.0,
        description="Minimum time separation between paired detections [hours].",
    )
    max_baseline_days: float = Field(
        default=15.0,
        gt=0.0,
        description="Maximum time span between paired detections [days].",
    )
    min_angular_speed_arcsec_per_hr: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum sky-plane motion rate to reject stationary stars [arcsec/hr].",
    )
    max_angular_speed_arcsec_per_hr: float = Field(
        default=3600.0,
        gt=0.0,
        description="Maximum sky-plane angular speed ceiling [arcsec/hr].",
    )
    max_angular_accel_arcsec_per_day2: float = Field(
        default=120.0,
        gt=0.0,
        description="Maximum allowed sky-plane angular acceleration [arcsec/day^2].",
    )
    max_rms_residual_arcsec: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum linear-fit astrometric RMS residual [arcsec].",
    )
    assumed_heliocentric_distance_au: float = Field(
        default=2.5,
        gt=0.05,
        description="Hypothesized heliocentric distance for HelioLinC escape-speed check [AU].",
    )
    enforce_escape_velocity_bound: bool = Field(
        default=True,
        description="Reject candidates whose implied speed exceeds Solar escape limits.",
    )
    position_eps_arcsec: float = Field(
        default=2.0,
        gt=0.0,
        description="KD-Tree / DBSCAN clustering radius in reference sky position [arcsec].",
    )
    velocity_eps_arcsec_per_hr: float = Field(
        default=2.0,
        gt=0.0,
        description="KD-Tree / DBSCAN clustering radius in angular velocity [arcsec/hr].",
    )
    dbscan_min_samples: int | None = Field(
        default=None,
        ge=1,
        description="Optional override for minimum neighbors in 4D phase-space DBSCAN.",
    )
    min_cnn_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional Real/Bogus CNN score gate applied prior to linking.",
    )
    use_snr_weighted_gating: bool = Field(
        default=True,
        description="Enable dynamic SNR-weighted chi-squared gating for faint candidate triage.",
    )
    seeing_fwhm_arcsec: float = Field(
        default=1.5,
        gt=0.1,
        description="Observing seeing FWHM in arcseconds for centroid noise scaling.",
    )

    @model_validator(mode="after")
    def _validate_velocity_window(self) -> TrackletLinkingConfig:
        if self.max_angular_speed_arcsec_per_hr <= self.min_angular_speed_arcsec_per_hr:
            msg = (
                "max_angular_speed_arcsec_per_hr must be strictly greater than "
                "min_angular_speed_arcsec_per_hr."
            )
            raise ValueError(msg)
        max_baseline_hr = float((self.max_baseline_days * u.day).to_value(u.hour))
        if max_baseline_hr <= self.min_baseline_hours:
            msg = "max_baseline_days must exceed min_baseline_hours."
            raise ValueError(msg)
        return self


class SpaceVelocityState(LumiaBaseModel):
    """Projected 4D tangent-plane position and velocity vector at a reference epoch."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    source_candidate_id: str = Field(..., min_length=1)
    target_candidate_id: str = Field(..., min_length=1)
    source_index: int = Field(..., ge=0)
    target_index: int = Field(..., ge=0)
    reference_bjd_tdb: float
    baseline_hours: float = Field(..., gt=0.0)
    ref_xi_arcsec: float = Field(
        ...,
        description="Back-projected Eastward tangent-plane offset at reference_bjd_tdb [arcsec].",
    )
    ref_eta_arcsec: float = Field(
        ...,
        description="Back-projected Northward tangent-plane offset at reference_bjd_tdb [arcsec].",
    )
    v_ra_cosdec_arcsec_per_hr: float = Field(
        ...,
        description="Sky-plane angular velocity along RA*cos(Dec) [arcsec/hr].",
    )
    v_dec_arcsec_per_hr: float = Field(
        ...,
        description="Sky-plane angular velocity along Dec [arcsec/hr].",
    )


class SpaceVelocityBatch(LumiaBaseModel):
    """Vectorized batch of projected 4D tangent-plane space-velocity pairs."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    source_candidate_ids: tuple[str, ...]
    target_candidate_ids: tuple[str, ...]
    source_indices: np.ndarray
    target_indices: np.ndarray
    reference_bjd_tdb: float
    baseline_hours: np.ndarray
    ref_xi_arcsec: np.ndarray
    ref_eta_arcsec: np.ndarray
    v_ra_cosdec_arcsec_per_hr: np.ndarray
    v_dec_arcsec_per_hr: np.ndarray

    @model_validator(mode="after")
    def _validate_batch_arrays(self) -> SpaceVelocityBatch:
        n_pairs = len(self.source_candidate_ids)
        if len(self.target_candidate_ids) != n_pairs:
            msg = "source_candidate_ids and target_candidate_ids must have identical length."
            raise ValueError(msg)
        for name, arr in (
            ("source_indices", self.source_indices),
            ("target_indices", self.target_indices),
            ("baseline_hours", self.baseline_hours),
            ("ref_xi_arcsec", self.ref_xi_arcsec),
            ("ref_eta_arcsec", self.ref_eta_arcsec),
            ("v_ra_cosdec_arcsec_per_hr", self.v_ra_cosdec_arcsec_per_hr),
            ("v_dec_arcsec_per_hr", self.v_dec_arcsec_per_hr),
        ):
            if arr.ndim != 1 or arr.shape[0] != n_pairs:
                msg = f"{name} must be 1D with length {n_pairs}, got shape {arr.shape}."
                raise ValueError(msg)
            if not np.all(np.isfinite(arr)):
                msg = f"{name} contains non-finite values."
                raise ValueError(msg)
        return self

    @property
    def pair_count(self) -> int:
        """Return total number of projected detection pairs in the batch."""
        return int(self.source_indices.shape[0])

    def to_states(self) -> list[SpaceVelocityState]:
        """Materialize immutable SpaceVelocityState records from vectorized arrays."""
        return [
            SpaceVelocityState(
                source_candidate_id=self.source_candidate_ids[idx],
                target_candidate_id=self.target_candidate_ids[idx],
                source_index=int(self.source_indices[idx]),
                target_index=int(self.target_indices[idx]),
                reference_bjd_tdb=self.reference_bjd_tdb,
                baseline_hours=float(self.baseline_hours[idx]),
                ref_xi_arcsec=float(self.ref_xi_arcsec[idx]),
                ref_eta_arcsec=float(self.ref_eta_arcsec[idx]),
                v_ra_cosdec_arcsec_per_hr=float(self.v_ra_cosdec_arcsec_per_hr[idx]),
                v_dec_arcsec_per_hr=float(self.v_dec_arcsec_per_hr[idx]),
            )
            for idx in range(self.pair_count)
        ]


class KinematicValidationResult(LumiaBaseModel):
    """Astrophysical kinematic fit metrics and pass/reject verdict for a candidate tracklet."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    is_valid: bool
    rejection_reason: str | None = Field(default=None)
    v_ra_cosdec_arcsec_per_hr: float
    v_dec_arcsec_per_hr: float
    angular_speed_arcsec_per_hr: float = Field(..., ge=0.0)
    angular_accel_arcsec_per_day2: float = Field(..., ge=0.0)
    rms_residual_arcsec: float = Field(..., ge=0.0)
    implied_transverse_speed_km_s: float = Field(..., ge=0.0)
    max_bound_transverse_speed_km_s: float = Field(..., ge=0.0)
    per_epoch_residuals_arcsec: tuple[float, ...] = Field(default=())


class Tracklet(LumiaBaseModel):
    """Validated multi-epoch moving-object tracklet linked in space-velocity coordinates."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    tracklet_id: str = Field(..., min_length=1)
    detections: tuple[CandidateDetection, ...]
    reference_bjd_tdb: float
    v_ra_cosdec_arcsec_per_hr: float
    v_dec_arcsec_per_hr: float
    angular_speed_arcsec_per_hr: float = Field(..., ge=0.0)
    angular_accel_arcsec_per_day2: float = Field(..., ge=0.0)
    rms_residual_arcsec: float = Field(..., ge=0.0)
    implied_transverse_speed_km_s: float = Field(..., ge=0.0)


def _sky_to_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert ICRS (ra_deg, dec_deg) arrays to 3D Cartesian unit vectors via Astropy units."""
    ra_rad: np.ndarray = (ra_deg * u.deg).to_value(u.rad)
    dec_rad: np.ndarray = (dec_deg * u.deg).to_value(u.rad)
    cos_dec = np.cos(dec_rad)
    return np.column_stack(
        (
            cos_dec * np.cos(ra_rad),
            cos_dec * np.sin(ra_rad),
            np.sin(dec_rad),
        )
    )


def _compute_tangent_basis(
    unit_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct local orthonormal (u0, east, north) tangent basis at the mean sky vector."""
    mean_vec = np.mean(unit_vectors, axis=0)
    norm = float(np.linalg.norm(mean_vec))
    u0 = unit_vectors[0].copy() if norm < 1e-12 else mean_vec / norm

    xy_norm = math.hypot(float(u0[0]), float(u0[1]))
    if xy_norm < 1e-12:
        # Celestial pole fallback
        east = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        east = np.array([-u0[1] / xy_norm, u0[0] / xy_norm, 0.0], dtype=np.float64)

    north = np.cross(u0, east)
    north /= float(np.linalg.norm(north))
    return u0, east, north


def _gnomonic_project_arcsec(
    detections: Sequence[CandidateDetection],
    tangent_u0: np.ndarray,
    east_vec: np.ndarray,
    north_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project sky detections onto a gnomonic tangent plane in arcseconds."""
    ra_arr = np.array([d.ra_deg for d in detections], dtype=np.float64)
    dec_arr = np.array([d.dec_deg for d in detections], dtype=np.float64)
    u_vecs = _sky_to_unit_vectors(ra_arr, dec_arr)

    denom = u_vecs @ tangent_u0
    if np.any(denom <= 0.0):
        msg = "Detections span more than a hemisphere; cannot share a single tangent plane."
        raise ValueError(msg)

    xi_rad = (u_vecs @ east_vec) / denom
    eta_rad = (u_vecs @ north_vec) / denom

    xi_arcsec: np.ndarray = (xi_rad * u.rad).to_value(u.arcsec)
    eta_arcsec: np.ndarray = (eta_rad * u.rad).to_value(u.arcsec)
    return xi_arcsec, eta_arcsec


def compute_max_bound_angular_speed_arcsec_per_hr(
    assumed_heliocentric_distance_au: float,
) -> float:
    """Compute topocentric angular speed ceiling [arcsec/hr] for a bound Solar System body at opposition."""
    r_helio = assumed_heliocentric_distance_au * const.au
    v_esc_helio = np.sqrt(2.0 * const.GM_sun / r_helio).to(u.km / u.s)
    v_earth_orb = np.sqrt(const.GM_sun / (1.0 * const.au)).to(u.km / u.s)
    max_bound_kms = float((v_esc_helio + v_earth_orb).to_value(u.km / u.s))

    delta_topocentric_au = max(0.1, abs(assumed_heliocentric_distance_au - 1.0))
    delta_dist = delta_topocentric_au * const.au
    ang_speed_ceiling = (max_bound_kms * (u.km / u.s)) / delta_dist
    return float(
        ang_speed_ceiling.to_value(
            u.arcsec / u.hour,
            equivalencies=u.dimensionless_angles(),
        )
    )


def _empty_velocity_batch(reference_bjd_tdb: float) -> SpaceVelocityBatch:
    """Return an empty validated SpaceVelocityBatch with 0 pairs."""
    return SpaceVelocityBatch(
        source_candidate_ids=(),
        target_candidate_ids=(),
        source_indices=np.empty(0, dtype=np.int64),
        target_indices=np.empty(0, dtype=np.int64),
        reference_bjd_tdb=reference_bjd_tdb,
        baseline_hours=np.empty(0, dtype=np.float64),
        ref_xi_arcsec=np.empty(0, dtype=np.float64),
        ref_eta_arcsec=np.empty(0, dtype=np.float64),
        v_ra_cosdec_arcsec_per_hr=np.empty(0, dtype=np.float64),
        v_dec_arcsec_per_hr=np.empty(0, dtype=np.float64),
    )


def project_detections_to_velocity_batch(
    detections: Sequence[CandidateDetection],
    *,
    reference_bjd_tdb: float | None = None,
    config: TrackletLinkingConfig | None = None,
    filter_by_speed_bounds: bool = False,
    backend: ProjectionBackend = "numpy",
) -> SpaceVelocityBatch:
    """Project multi-epoch detections into a vectorized 4D space-velocity batch."""
    if backend not in ("numpy", "torch"):
        msg = f"Unsupported projection backend: {backend!r}. Expected 'numpy' or 'torch'."
        raise ValueError(msg)

    cfg = config if config is not None else TrackletLinkingConfig()
    if len(detections) < 2:
        fallback_ref = (
            float(reference_bjd_tdb)
            if reference_bjd_tdb is not None
            else (detections[0].bjd_tdb if detections else 0.0)
        )
        return _empty_velocity_batch(fallback_ref)

    ordered_indices = np.asarray(
        sorted(
            range(len(detections)),
            key=lambda idx: (detections[idx].bjd_tdb, detections[idx].candidate_id),
        ),
        dtype=np.int64,
    )
    ordered_dets = [detections[int(i)] for i in ordered_indices]

    ref_bjd: float = (
        float(reference_bjd_tdb)
        if reference_bjd_tdb is not None
        else ordered_dets[0].bjd_tdb
    )

    # Anchor the tangent plane at detections nearest the reference epoch so that
    # single-object trajectories back-project to the origin while full fields center cleanly.
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

    t_rel_hr: np.ndarray = ((bjd_arr - ref_bjd) * u.day).to_value(u.hour)
    min_dt_hr = float((cfg.min_baseline_hours * u.hour).to_value(u.hour))
    max_dt_hr = float((cfg.max_baseline_days * u.day).to_value(u.hour))
    rate_scale = float((1.0 * u.arcsec / (1.0 * u.hour)).to_value(u.arcsec / u.hour))
    offset_scale = float(((1.0 * (u.arcsec / u.hour)) * (1.0 * u.hour)).to_value(u.arcsec))

    max_speed_ceiling = cfg.max_angular_speed_arcsec_per_hr
    if cfg.enforce_escape_velocity_bound:
        escape_bound = compute_max_bound_angular_speed_arcsec_per_hr(
            cfg.assumed_heliocentric_distance_au
        )
        max_speed_ceiling = min(max_speed_ceiling, escape_bound)

    n_dets = len(ordered_dets)

    if backend == "numpy":
        # Vectorized 2D (1, N) - (N, 1) broadcasting over upper-triangular pairs
        dt_mat = t_rel_hr[None, :] - t_rel_hr[:, None]
        upper_mask = np.triu(np.ones((n_dets, n_dets), dtype=bool), k=1)
        valid_mask = upper_mask & (dt_mat >= min_dt_hr) & (dt_mat <= max_dt_hr)
        src_pos, tgt_pos = np.nonzero(valid_mask)
        if src_pos.size == 0:
            return _empty_velocity_batch(ref_bjd)

        dt_hr = dt_mat[src_pos, tgt_pos]
        v_xi = ((xi_arcsec[tgt_pos] - xi_arcsec[src_pos]) / dt_hr) * rate_scale
        v_eta = ((eta_arcsec[tgt_pos] - eta_arcsec[src_pos]) / dt_hr) * rate_scale

        if filter_by_speed_bounds:
            speed = np.hypot(v_xi, v_eta)
            keep = (speed >= cfg.min_angular_speed_arcsec_per_hr) & (
                speed <= max_speed_ceiling
            )
            src_pos = src_pos[keep]
            tgt_pos = tgt_pos[keep]
            dt_hr = dt_hr[keep]
            v_xi = v_xi[keep]
            v_eta = v_eta[keep]
            if src_pos.size == 0:
                return _empty_velocity_batch(ref_bjd)

        ref_xi = xi_arcsec[src_pos] - (v_xi * t_rel_hr[src_pos]) * offset_scale
        ref_eta = eta_arcsec[src_pos] - (v_eta * t_rel_hr[src_pos]) * offset_scale
    else:
        with torch.no_grad():
            t_t = torch.as_tensor(t_rel_hr, dtype=torch.float64)
            xi_t = torch.as_tensor(xi_arcsec, dtype=torch.float64)
            eta_t = torch.as_tensor(eta_arcsec, dtype=torch.float64)

            dt_mat_t = t_t.unsqueeze(0) - t_t.unsqueeze(1)
            upper_t = torch.triu(
                torch.ones((n_dets, n_dets), dtype=torch.bool),
                diagonal=1,
            )
            valid_t = upper_t & (dt_mat_t >= min_dt_hr) & (dt_mat_t <= max_dt_hr)
            src_t, tgt_t = torch.nonzero(valid_t, as_tuple=True)
            if src_t.numel() == 0:
                return _empty_velocity_batch(ref_bjd)

            dt_t = dt_mat_t[src_t, tgt_t]
            v_xi_t = ((xi_t[tgt_t] - xi_t[src_t]) / dt_t) * rate_scale
            v_eta_t = ((eta_t[tgt_t] - eta_t[src_t]) / dt_t) * rate_scale

            if filter_by_speed_bounds:
                speed_t = torch.hypot(v_xi_t, v_eta_t)
                keep_t = (speed_t >= cfg.min_angular_speed_arcsec_per_hr) & (
                    speed_t <= max_speed_ceiling
                )
                src_t = src_t[keep_t]
                tgt_t = tgt_t[keep_t]
                dt_t = dt_t[keep_t]
                v_xi_t = v_xi_t[keep_t]
                v_eta_t = v_eta_t[keep_t]
                if src_t.numel() == 0:
                    return _empty_velocity_batch(ref_bjd)

            ref_xi_t = xi_t[src_t] - (v_xi_t * t_t[src_t]) * offset_scale
            ref_eta_t = eta_t[src_t] - (v_eta_t * t_t[src_t]) * offset_scale

            src_pos = src_t.cpu().numpy()
            tgt_pos = tgt_t.cpu().numpy()
            dt_hr = dt_t.cpu().numpy()
            v_xi = v_xi_t.cpu().numpy()
            v_eta = v_eta_t.cpu().numpy()
            ref_xi = ref_xi_t.cpu().numpy()
            ref_eta = ref_eta_t.cpu().numpy()

    ordered_ids = [d.candidate_id for d in ordered_dets]
    src_ids = tuple(ordered_ids[int(p)] for p in src_pos)
    tgt_ids = tuple(ordered_ids[int(p)] for p in tgt_pos)

    return SpaceVelocityBatch(
        source_candidate_ids=src_ids,
        target_candidate_ids=tgt_ids,
        source_indices=np.asarray(ordered_indices[src_pos], dtype=np.int64),
        target_indices=np.asarray(ordered_indices[tgt_pos], dtype=np.int64),
        reference_bjd_tdb=ref_bjd,
        baseline_hours=np.asarray(dt_hr, dtype=np.float64),
        ref_xi_arcsec=np.asarray(ref_xi, dtype=np.float64),
        ref_eta_arcsec=np.asarray(ref_eta, dtype=np.float64),
        v_ra_cosdec_arcsec_per_hr=np.asarray(v_xi, dtype=np.float64),
        v_dec_arcsec_per_hr=np.asarray(v_eta, dtype=np.float64),
    )


def project_detections_to_velocity_space(
    detections: Sequence[CandidateDetection],
    *,
    reference_bjd_tdb: float | None = None,
    config: TrackletLinkingConfig | None = None,
    filter_by_speed_bounds: bool = False,
    backend: ProjectionBackend = "numpy",
) -> list[SpaceVelocityState]:
    """Project multi-epoch RA/DEC detections into 4D reference-epoch space-velocity vectors."""
    batch = project_detections_to_velocity_batch(
        detections,
        reference_bjd_tdb=reference_bjd_tdb,
        config=config,
        filter_by_speed_bounds=filter_by_speed_bounds,
        backend=backend,
    )
    return batch.to_states()


def cluster_velocity_states(
    states: Sequence[SpaceVelocityState],
    *,
    config: TrackletLinkingConfig | None = None,
) -> list[list[SpaceVelocityState]]:
    """Group space-velocity vectors via KD-Tree backed 4D DBSCAN density clustering."""
    cfg = config if config is not None else TrackletLinkingConfig()
    n_states = len(states)
    if n_states == 0:
        return []

    min_samples: int = (
        cfg.dbscan_min_samples
        if cfg.dbscan_min_samples is not None
        else max(2, cfg.min_epochs - 1)
    )
    if n_states < min_samples:
        return []

    scaled_coords = np.array(
        [
            [
                s.ref_xi_arcsec / cfg.position_eps_arcsec,
                s.ref_eta_arcsec / cfg.position_eps_arcsec,
                s.v_ra_cosdec_arcsec_per_hr / cfg.velocity_eps_arcsec_per_hr,
                s.v_dec_arcsec_per_hr / cfg.velocity_eps_arcsec_per_hr,
            ]
            for s in states
        ],
        dtype=np.float64,
    )

    tree = cKDTree(scaled_coords)
    neighborhoods: list[list[int]] = [
        list(nbrs) for nbrs in tree.query_ball_point(scaled_coords, r=1.0)
    ]
    is_core = np.array(
        [len(nbrs) >= min_samples for nbrs in neighborhoods],
        dtype=bool,
    )

    labels = np.full(n_states, -1, dtype=np.int64)
    cluster_id = 0

    for idx in range(n_states):
        if not is_core[idx] or labels[idx] != -1:
            continue

        labels[idx] = cluster_id
        queue: deque[int] = deque([idx])

        while queue:
            current = queue.popleft()
            if not is_core[current]:
                continue
            for neighbor in neighborhoods[current]:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    queue.append(neighbor)

        cluster_id += 1

    clusters: list[list[SpaceVelocityState]] = []
    for cid in range(cluster_id):
        member_indices = np.flatnonzero(labels == cid)
        clusters.append([states[int(i)] for i in member_indices])

    return clusters


def validate_tracklet_kinematics(
    detections: Sequence[CandidateDetection],
    *,
    config: TrackletLinkingConfig | None = None,
) -> KinematicValidationResult:
    """Fit linear and quadratic sky motion and validate against Solar System physics bounds."""
    cfg = config if config is not None else TrackletLinkingConfig()

    # Compute Solar escape speed and topocentric bound ceiling using Astropy constants
    r_helio = cfg.assumed_heliocentric_distance_au * const.au
    v_esc_helio = np.sqrt(2.0 * const.GM_sun / r_helio).to(u.km / u.s)
    v_earth_orb = np.sqrt(const.GM_sun / (1.0 * const.au)).to(u.km / u.s)
    max_bound_kms = float((v_esc_helio + v_earth_orb).to_value(u.km / u.s))

    if len(detections) < cfg.min_epochs:
        return KinematicValidationResult(
            is_valid=False,
            rejection_reason=(
                f"Insufficient detections ({len(detections)} < min_epochs={cfg.min_epochs})."
            ),
            v_ra_cosdec_arcsec_per_hr=0.0,
            v_dec_arcsec_per_hr=0.0,
            angular_speed_arcsec_per_hr=0.0,
            angular_accel_arcsec_per_day2=0.0,
            rms_residual_arcsec=0.0,
            implied_transverse_speed_km_s=0.0,
            max_bound_transverse_speed_km_s=max_bound_kms,
        )

    ordered = sorted(detections, key=lambda d: d.bjd_tdb)
    bjd_arr = np.array([d.bjd_tdb for d in ordered], dtype=np.float64)
    min_sep_days = float((0.5 * cfg.min_baseline_hours * u.hour).to_value(u.day))

    if np.any(np.diff(bjd_arr) < min_sep_days):
        return KinematicValidationResult(
            is_valid=False,
            rejection_reason="Duplicate or near-simultaneous epochs in candidate tracklet.",
            v_ra_cosdec_arcsec_per_hr=0.0,
            v_dec_arcsec_per_hr=0.0,
            angular_speed_arcsec_per_hr=0.0,
            angular_accel_arcsec_per_day2=0.0,
            rms_residual_arcsec=0.0,
            implied_transverse_speed_km_s=0.0,
            max_bound_transverse_speed_km_s=max_bound_kms,
        )

    all_u = _sky_to_unit_vectors(
        np.array([d.ra_deg for d in ordered], dtype=np.float64),
        np.array([d.dec_deg for d in ordered], dtype=np.float64),
    )
    u0, east_vec, north_vec = _compute_tangent_basis(all_u)
    xi_arcsec, eta_arcsec = _gnomonic_project_arcsec(ordered, u0, east_vec, north_vec)

    t_mid = float(np.mean(bjd_arr))
    dt_days = (bjd_arr - t_mid) * u.day
    dt_hours_val: np.ndarray = dt_days.to_value(u.hour)
    dt_days_val: np.ndarray = dt_days.to_value(u.day)

    # 1. Linear constant-velocity fit: x(t) = x0 + v * dt_hours
    design_lin = np.column_stack((np.ones_like(dt_hours_val), dt_hours_val))
    coef_xi_lin, _, _, _ = np.linalg.lstsq(design_lin, xi_arcsec, rcond=None)
    coef_eta_lin, _, _, _ = np.linalg.lstsq(design_lin, eta_arcsec, rcond=None)

    v_xi = float(coef_xi_lin[1])
    v_eta = float(coef_eta_lin[1])
    ang_speed_quant = math.hypot(v_xi, v_eta) * (u.arcsec / u.hour)
    ang_speed = float(ang_speed_quant.to_value(u.arcsec / u.hour))

    res_xi = xi_arcsec - (design_lin @ coef_xi_lin)
    res_eta = eta_arcsec - (design_lin @ coef_eta_lin)
    per_epoch_res = np.hypot(res_xi, res_eta)
    rms_residual = float(np.sqrt(np.mean(per_epoch_res**2)))

    # 2. Quadratic kinematic fit: x(t) = x0 + v_d * dt_days + 0.5 * a * dt_days^2
    design_quad = np.column_stack(
        (np.ones_like(dt_days_val), dt_days_val, 0.5 * (dt_days_val**2))
    )
    coef_xi_quad, _, _, _ = np.linalg.lstsq(design_quad, xi_arcsec, rcond=None)
    coef_eta_quad, _, _, _ = np.linalg.lstsq(design_quad, eta_arcsec, rcond=None)

    accel_mag = math.hypot(float(coef_xi_quad[2]), float(coef_eta_quad[2]))

    # 3. Implied physical transverse speed at opposition topocentric distance
    delta_topocentric_au = max(0.1, abs(cfg.assumed_heliocentric_distance_au - 1.0))
    delta_dist = delta_topocentric_au * const.au
    implied_v_kms = float(
        (ang_speed_quant * delta_dist).to_value(
            u.km / u.s,
            equivalencies=u.dimensionless_angles(),
        )
    )

    # Solar gravitational acceleration ceiling + Earth reflex bound check
    if ang_speed < cfg.min_angular_speed_arcsec_per_hr:
        reason: str | None = (
            f"Angular velocity {ang_speed:.3f} arcsec/hr is below minimum moving-object "
            f"threshold ({cfg.min_angular_speed_arcsec_per_hr:.3f} arcsec/hr)."
        )
    elif cfg.enforce_escape_velocity_bound and implied_v_kms > max_bound_kms:
        reason = (
            f"Implied transverse velocity {implied_v_kms:.2f} km/s exceeds bound Solar "
            f"escape velocity limit ({max_bound_kms:.2f} km/s)."
        )
    elif ang_speed > cfg.max_angular_speed_arcsec_per_hr:
        reason = (
            f"Angular velocity {ang_speed:.2f} arcsec/hr exceeds maximum configured "
            f"limit ({cfg.max_angular_speed_arcsec_per_hr:.2f} arcsec/hr)."
        )
    elif accel_mag > cfg.max_angular_accel_arcsec_per_day2:
        reason = (
            f"Angular acceleration {accel_mag:.2f} arcsec/day^2 exceeds physical "
            f"threshold ({cfg.max_angular_accel_arcsec_per_day2:.2f} arcsec/day^2)."
        )
    elif rms_residual > cfg.max_rms_residual_arcsec:
        if cfg.use_snr_weighted_gating:
            snrs = np.array(
                [d.snr if d.snr is not None else 5.0 for d in ordered], dtype=np.float64
            )
            sigma_centroid = cfg.seeing_fwhm_arcsec / (2.0 * np.maximum(snrs, 0.1))
            sigma_tot_sq = sigma_centroid**2 + 0.05**2 + 0.02**2
            chi2_val = float(np.sum((per_epoch_res**2) / sigma_tot_sq))
            dof = max(1, 2 * len(ordered) - 4)
            reduced_chi2 = chi2_val / dof
            if reduced_chi2 > 3.0:
                reason = (
                    f"Astrometric RMS residual {rms_residual:.3f} arcsec (reduced chi2 {reduced_chi2:.2f}) "
                    f"exceeds tolerance for detection SNRs."
                )
            else:
                reason = None
        else:
            reason = (
                f"Astrometric RMS residual {rms_residual:.3f} arcsec exceeds threshold "
                f"({cfg.max_rms_residual_arcsec:.3f} arcsec)."
            )
    else:
        reason = None

    return KinematicValidationResult(
        is_valid=reason is None,
        rejection_reason=reason,
        v_ra_cosdec_arcsec_per_hr=v_xi,
        v_dec_arcsec_per_hr=v_eta,
        angular_speed_arcsec_per_hr=ang_speed,
        angular_accel_arcsec_per_day2=accel_mag,
        rms_residual_arcsec=rms_residual,
        implied_transverse_speed_km_s=implied_v_kms,
        max_bound_transverse_speed_km_s=max_bound_kms,
        per_epoch_residuals_arcsec=tuple(float(v) for v in per_epoch_res),
    )


def _select_unique_epoch_detections(
    cluster_states: Sequence[SpaceVelocityState],
    detections: Sequence[CandidateDetection],
    config: TrackletLinkingConfig,
) -> list[CandidateDetection]:
    """Extract at most one trajectory-consistent detection per epoch from a pair cluster."""
    candidate_indices: set[int] = set()
    for state in cluster_states:
        candidate_indices.add(state.source_index)
        candidate_indices.add(state.target_index)

    member_dets = sorted(
        (detections[idx] for idx in candidate_indices),
        key=lambda d: (d.bjd_tdb, d.candidate_id),
    )
    if len(member_dets) < config.min_epochs:
        return []

    ref_bjd = cluster_states[0].reference_bjd_tdb
    med_xi0 = float(np.median([s.ref_xi_arcsec for s in cluster_states]))
    med_eta0 = float(np.median([s.ref_eta_arcsec for s in cluster_states]))
    med_vxi = float(np.median([s.v_ra_cosdec_arcsec_per_hr for s in cluster_states]))
    med_veta = float(np.median([s.v_dec_arcsec_per_hr for s in cluster_states]))

    # Project using the same reference-epoch basis used during velocity projection
    all_bjd = np.array([d.bjd_tdb for d in detections], dtype=np.float64)
    min_dt = float(np.min(np.abs(all_bjd - ref_bjd)))
    half_min_baseline_days = float((0.5 * config.min_baseline_hours * u.hour).to_value(u.day))
    ref_mask = np.abs(all_bjd - ref_bjd) <= (min_dt + half_min_baseline_days)
    all_u = _sky_to_unit_vectors(
        np.array([d.ra_deg for d in detections], dtype=np.float64),
        np.array([d.dec_deg for d in detections], dtype=np.float64),
    )
    u0, east_vec, north_vec = _compute_tangent_basis(all_u[ref_mask])
    xi_mem, eta_mem = _gnomonic_project_arcsec(member_dets, u0, east_vec, north_vec)

    # Group detections into distinct observation epochs
    epoch_groups: list[list[int]] = []
    for idx, det in enumerate(member_dets):
        if (
            not epoch_groups
            or abs(det.bjd_tdb - member_dets[epoch_groups[-1][0]].bjd_tdb)
            > half_min_baseline_days
        ):
            epoch_groups.append([idx])
        else:
            epoch_groups[-1].append(idx)

    member_bjd = np.array([d.bjd_tdb for d in member_dets], dtype=np.float64)
    member_dt_hr: np.ndarray = ((member_bjd - ref_bjd) * u.day).to_value(u.hour)
    pred_xi_arr = med_xi0 + med_vxi * member_dt_hr
    pred_eta_arr = med_eta0 + med_veta * member_dt_hr
    err_arr = np.hypot(xi_mem - pred_xi_arr, eta_mem - pred_eta_arr)

    selected: list[CandidateDetection] = []
    max_gate_arcsec = 3.0 * config.position_eps_arcsec

    for group in epoch_groups:
        group_errs = err_arr[group]
        best_local = int(np.argmin(group_errs))
        best_idx = group[best_local]
        best_err = float(err_arr[best_idx])
        if best_err <= max_gate_arcsec:
            selected.append(member_dets[best_idx])

    return selected


def link_tracklets(
    detections: Sequence[CandidateDetection],
    *,
    config: TrackletLinkingConfig | None = None,
    backend: ProjectionBackend = "numpy",
) -> list[Tracklet]:
    """Link isolated multi-epoch transient detections into validated moving-object tracklets."""
    cfg = config if config is not None else TrackletLinkingConfig()
    working_dets: Sequence[CandidateDetection] = (
        [
            det
            for det in detections
            if det.cnn_score is not None and det.cnn_score >= cfg.min_cnn_score
        ]
        if cfg.min_cnn_score is not None
        else detections
    )
    if len(working_dets) < cfg.min_epochs:
        return []

    states = project_detections_to_velocity_space(
        working_dets,
        config=cfg,
        filter_by_speed_bounds=True,
        backend=backend,
    )
    if not states:
        return []

    clusters = cluster_velocity_states(states, config=cfg)

    seen_detection_sets: set[frozenset[str]] = set()
    valid_tracklets: list[Tracklet] = []

    for cluster in clusters:
        selected_dets = _select_unique_epoch_detections(cluster, working_dets, cfg)
        if len(selected_dets) < cfg.min_epochs:
            continue

        det_key = frozenset(d.candidate_id for d in selected_dets)
        if det_key in seen_detection_sets:
            continue

        kinematics = validate_tracklet_kinematics(selected_dets, config=cfg)
        if not kinematics.is_valid:
            continue

        seen_detection_sets.add(det_key)
        ordered_sorted = sorted(selected_dets, key=lambda d: d.bjd_tdb)
        enriched_tuple = tuple(
            det.model_copy(update={"astrometric_residual_arcsec": res_val})
            for det, res_val in zip(
                ordered_sorted,
                kinematics.per_epoch_residuals_arcsec,
                strict=True,
            )
        )
        valid_tracklets.append(
            Tracklet(
                tracklet_id=f"trk_{len(valid_tracklets) + 1:04d}",
                detections=enriched_tuple,
                reference_bjd_tdb=enriched_tuple[0].bjd_tdb,
                v_ra_cosdec_arcsec_per_hr=kinematics.v_ra_cosdec_arcsec_per_hr,
                v_dec_arcsec_per_hr=kinematics.v_dec_arcsec_per_hr,
                angular_speed_arcsec_per_hr=kinematics.angular_speed_arcsec_per_hr,
                angular_accel_arcsec_per_day2=kinematics.angular_accel_arcsec_per_day2,
                rms_residual_arcsec=kinematics.rms_residual_arcsec,
                implied_transverse_speed_km_s=kinematics.implied_transverse_speed_km_s,
            )
        )

    valid_tracklets.sort(
        key=lambda t: (t.reference_bjd_tdb, t.detections[0].candidate_id),
    )
    return [
        t.model_copy(update={"tracklet_id": f"trk_{idx + 1:04d}"})
        for idx, t in enumerate(valid_tracklets)
    ]

