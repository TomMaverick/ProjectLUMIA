"""Spatial D4 dihedral group transformations, crop jittering, and faint synthetic injection."""

from __future__ import annotations

import math
from collections.abc import Sequence  # noqa: TC003
from typing import Literal, overload

import numpy as np
from pydantic import Field

from LUMIA.core.schemas import LumiaBaseModel
from LUMIA.data_mining.archive_crawler import CutoutTriplet, WCSMetadata

DihedralOp = Literal[0, 1, 2, 3, 4, 5, 6, 7]


class FaintSyntheticConfig(LumiaBaseModel):
    """Configuration for physical synthetic moving-object injection."""

    mag_min: float = Field(
        default=19.0, ge=12.0, le=25.0, description="Brightest injection magnitude."
    )
    mag_max: float = Field(
        default=22.0, ge=12.0, le=25.0, description="Faintest injection magnitude."
    )
    zero_point: float = Field(
        default=26.3, gt=0.0, description="Photometric zero-point (ZTF r ~ 26.3)."
    )
    exposure_time_s: float = Field(
        default=30.0, gt=0.0, description="Exposure duration in seconds."
    )
    gain_e_per_adu: float = Field(default=1.0, gt=0.0, description="CCD conversion gain.")
    psf_fwhm_px: float = Field(
        default=2.0, gt=0.0, description="Atmospheric seeing FWHM in pixels."
    )
    streak_fraction: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Fraction of injected sources that are trailed streaks.",
    )
    max_streak_length_px: float = Field(
        default=15.0, ge=1.0, le=40.0, description="Max intra-exposure trailing length."
    )
    seed: int = Field(default=42, ge=0)


class InjectedSampleMetadata(LumiaBaseModel):
    """Ground-truth metadata describing an injected synthetic moving object."""

    target_id: str = Field(...)
    apparent_mag: float = Field(...)
    snr: float = Field(..., ge=0.0)
    is_streak: bool = Field(...)
    streak_length_px: float = Field(..., ge=0.0)
    position_angle_deg: float = Field(..., ge=0.0, lt=360.0)
    centroid_x_px: float = Field(...)
    centroid_y_px: float = Field(...)
    injected_flux_adu: float = Field(..., ge=0.0)


def transform_coordinates_d4(
    target_x: float,
    target_y: float,
    op: DihedralOp = 0,
    stamp_size: int = 128,
) -> tuple[float, float]:
    """Transform sub-pixel target coordinates (x, y) under D4 symmetry operations."""
    s = float(stamp_size - 1)
    if op == 0:
        return target_x, target_y
    if op == 1:  # rot90 CCW: y' = s - x, x' = y
        return target_y, s - target_x
    if op == 2:  # rot180: x' = s - x, y' = s - y
        return s - target_x, s - target_y
    if op == 3:  # rot270 CCW: x' = s - y, y' = x
        return s - target_y, target_x
    if op == 4:  # horizontal flip: x' = s - x, y' = y
        return s - target_x, target_y
    if op == 5:  # vertical flip: x' = x, y' = s - y
        return target_x, s - target_y
    if op == 6:  # transpose: x' = y, y' = x
        return target_y, target_x
    if op == 7:  # anti-transpose: x' = s - y, y' = s - x
        return s - target_y, s - target_x
    return target_x, target_y


def apply_d4_transform_with_coords(
    tensor_3ch: np.ndarray,
    target_x: float,
    target_y: float,
    op: DihedralOp = 0,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Apply D4 transformation to a 3-channel tensor and transform target coordinates."""
    res = apply_d4_transform(tensor_3ch, op, target_x=target_x, target_y=target_y)
    return res[0], (res[1], res[2])


@overload
def apply_d4_transform(
    tensor_3ch: np.ndarray,
    op: DihedralOp,
) -> np.ndarray: ...


@overload
def apply_d4_transform(
    tensor_3ch: np.ndarray,
    op: DihedralOp,
    target_x: float,
    target_y: float,
) -> tuple[np.ndarray, float, float]: ...


def apply_d4_transform(
    tensor_3ch: np.ndarray,
    op: DihedralOp,
    target_x: float | None = None,
    target_y: float | None = None,
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """Apply an exact spatial D4 dihedral group transformation to a 3-channel tensor.

    Kept strictly in-memory/on-the-fly to prevent 8x disk bloat and WCS misalignment.
    """
    if op == 0:
        transformed = tensor_3ch.copy()
    elif op == 1:
        transformed = np.rot90(tensor_3ch, k=1, axes=(-2, -1)).copy()
    elif op == 2:
        transformed = np.rot90(tensor_3ch, k=2, axes=(-2, -1)).copy()
    elif op == 3:
        transformed = np.rot90(tensor_3ch, k=3, axes=(-2, -1)).copy()
    elif op == 4:
        transformed = np.flip(tensor_3ch, axis=-1).copy()
    elif op == 5:
        transformed = np.flip(tensor_3ch, axis=-2).copy()
    elif op == 6:
        transformed = np.swapaxes(tensor_3ch, -2, -1).copy()
    elif op == 7:
        transformed = np.rot90(np.swapaxes(tensor_3ch, -2, -1), k=2, axes=(-2, -1)).copy()
    else:
        transformed = tensor_3ch.copy()

    if target_x is not None and target_y is not None:
        stamp_size = tensor_3ch.shape[-1]
        tx, ty = transform_coordinates_d4(target_x, target_y, op, stamp_size)
        return transformed, tx, ty

    return transformed


def _apply_array_crop_jitter(
    cutout_3ch: np.ndarray,
    target_x: float,
    target_y: float,
    crpix: tuple[float, float],
    shift_x: int,
    shift_y: int,
) -> tuple[np.ndarray, float, float, tuple[float, float]]:
    """Shift crop window by (shift_x, shift_y) while updating target coordinates and WCS crpix."""
    c, h, w = cutout_3ch.shape
    jittered = np.zeros_like(cutout_3ch)

    src_x0 = max(0, shift_x)
    src_x1 = min(w, w + shift_x)
    src_y0 = max(0, shift_y)
    src_y1 = min(h, h + shift_y)

    dst_x0 = max(0, -shift_x)
    dst_x1 = min(w, w - shift_x)
    dst_y0 = max(0, -shift_y)
    dst_y1 = min(h, h - shift_y)

    if (src_x1 > src_x0) and (src_y1 > src_y0) and (dst_x1 > dst_x0) and (dst_y1 > dst_y0):
        jittered[:, dst_y0:dst_y1, dst_x0:dst_x1] = cutout_3ch[:, src_y0:src_y1, src_x0:src_x1]

    new_tx = float(target_x - shift_x)
    new_ty = float(target_y - shift_y)
    new_crpix = (float(crpix[0] - shift_x), float(crpix[1] - shift_y))

    return jittered, new_tx, new_ty, new_crpix


def apply_spatial_crop_jitter_frames(
    frames: Sequence[CutoutTriplet],
    max_offset_px: int = 4,
    shift_x: int | None = None,
    shift_y: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[CutoutTriplet]:
    """Apply coherent spatial crop jitter across all frames of a multi-epoch tracklet."""
    if not frames:
        return []

    if shift_x is None or shift_y is None:
        local_rng = rng if rng is not None else np.random.default_rng()
        candidates = [v for v in range(-max_offset_px, max_offset_px + 1) if v != 0]
        sx = int(local_rng.choice(candidates)) if candidates else 0
        sy = int(local_rng.choice(candidates)) if candidates else 0
    else:
        sx, sy = shift_x, shift_y

    jittered_frames: list[CutoutTriplet] = []
    for fr in frames:
        new_data, new_tx, new_ty, new_crpix = _apply_array_crop_jitter(
            fr.data,
            fr.target_x,
            fr.target_y,
            fr.wcs_params.crpix,
            sx,
            sy,
        )
        new_wcs = WCSMetadata(
            crpix=new_crpix,
            crval=fr.wcs_params.crval,
            cd=fr.wcs_params.cd,
        )
        jittered_frames.append(
            CutoutTriplet(
                target_id=fr.target_id,
                tracklet_id=f"{fr.tracklet_id}_jit",
                epoch_index=fr.epoch_index,
                timestamp_utc=fr.timestamp_utc,
                bjd_tdb=fr.bjd_tdb,
                ra_deg=fr.ra_deg,
                dec_deg=fr.dec_deg,
                target_x=new_tx,
                target_y=new_ty,
                wcs_params=new_wcs,
                stamp_size=fr.stamp_size,
                filter_band=fr.filter_band,
                survey=fr.survey,
                snr=fr.snr,
                mag_apparent=fr.mag_apparent,
                is_synthetic=fr.is_synthetic,
                is_artifact=fr.is_artifact,
                artifact_type=fr.artifact_type,
                data=new_data,
            )
        )
    return jittered_frames


@overload
def apply_spatial_crop_jitter(
    cutout_3ch: np.ndarray,
    target_x: float,
    target_y: float,
    crpix: tuple[float, float],
    shift_x: int,
    shift_y: int,
) -> tuple[np.ndarray, float, float, tuple[float, float]]: ...


@overload
def apply_spatial_crop_jitter(
    cutout_3ch: Sequence[CutoutTriplet],
    *,
    max_offset_px: int = 4,
    shift_x: int | None = None,
    shift_y: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[CutoutTriplet]: ...


def apply_spatial_crop_jitter(
    cutout_3ch: np.ndarray | Sequence[CutoutTriplet],
    target_x: float | None = None,
    target_y: float | None = None,
    crpix: tuple[float, float] | None = None,
    shift_x: int | None = None,
    shift_y: int | None = None,
    *,
    max_offset_px: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, float, tuple[float, float]] | list[CutoutTriplet]:
    """Shift crop window for a raw array or coherently across tracklet frames."""
    if isinstance(cutout_3ch, (list, tuple)):
        return apply_spatial_crop_jitter_frames(
            frames=cutout_3ch,
            max_offset_px=max_offset_px,
            shift_x=shift_x,
            shift_y=shift_y,
            rng=rng,
        )

    if target_x is None or target_y is None or crpix is None or shift_x is None or shift_y is None:
        raise ValueError("Raw array jittering requires target_x, target_y, crpix, shift_x, shift_y")

    return _apply_array_crop_jitter(cutout_3ch, target_x, target_y, crpix, shift_x, shift_y)


class FaintSyntheticInjector:
    """Engine for injecting physically modeled faint moving sources into real sky noise."""

    def __init__(self, config: FaintSyntheticConfig | None = None) -> None:
        self.config = config if config is not None else FaintSyntheticConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def inject_synthetic_source(
        self,
        clean_background_triplet: np.ndarray,
        target_id: str | None = None,
        force_mag: float | None = None,
        force_streak: bool | None = None,
        center_xy: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, InjectedSampleMetadata]:
        """Inject a synthetic faint point source or trailed streak into a background triplet."""
        triplet = clean_background_triplet.copy().astype(np.float32)
        _, h, w = triplet.shape

        mag = (
            float(force_mag)
            if force_mag is not None
            else float(self._rng.uniform(self.config.mag_min, self.config.mag_max))
        )

        is_streak = (
            force_streak
            if force_streak is not None
            else bool(self._rng.random() < self.config.streak_fraction)
        )

        if center_xy is not None:
            cx, cy = center_xy
        else:
            cx = (w / 2.0) + float(self._rng.uniform(-4.0, 4.0))
            cy = (h / 2.0) + float(self._rng.uniform(-4.0, 4.0))

        if is_streak:
            length_px = float(self._rng.uniform(2.5, self.config.max_streak_length_px))
            pa_deg = float(self._rng.uniform(0.0, 360.0))
            rad = math.radians(pa_deg)
            dx = length_px * math.cos(rad)
            dy = length_px * math.sin(rad)
        else:
            length_px = float(self._rng.uniform(0.0, 0.4))
            pa_deg = float(self._rng.uniform(0.0, 360.0))
            dx = 0.0
            dy = 0.0

        flux_electrons = (
            math.pow(10.0, -0.4 * (mag - self.config.zero_point)) * self.config.exposure_time_s
        )
        total_flux_adu = flux_electrons / self.config.gain_e_per_adu

        source_profile = self._render_trailed_source(
            h=h,
            w=w,
            cx=cx,
            cy=cy,
            dx=dx,
            dy=dy,
            total_flux=total_flux_adu,
            fwhm=self.config.psf_fwhm_px,
        )

        source_electrons = np.clip(source_profile * self.config.gain_e_per_adu, 0.0, None)
        noisy_electrons = self._rng.poisson(source_electrons).astype(np.float32)
        noisy_source_adu = noisy_electrons / self.config.gain_e_per_adu

        # Add to Science (Channel 0) and Difference (Channel 2); Reference (Channel 1) stays static
        triplet[0] += noisy_source_adu
        triplet[2] += noisy_source_adu

        diff_noise = (
            float(
                np.median(
                    np.abs(clean_background_triplet[2] - np.median(clean_background_triplet[2]))
                )
            )
            * 1.4826
        )
        snr_est = float(np.max(source_profile) / max(diff_noise, 1e-3))

        synth_id = (
            target_id if target_id is not None else f"SYNTH_{self._rng.integers(100000, 999999)}"
        )

        metadata = InjectedSampleMetadata(
            target_id=synth_id,
            apparent_mag=mag,
            snr=snr_est,
            is_streak=is_streak,
            streak_length_px=length_px,
            position_angle_deg=pa_deg,
            centroid_x_px=cx,
            centroid_y_px=cy,
            injected_flux_adu=total_flux_adu,
        )

        return triplet, metadata

    def inject_synthetic_tracklet(
        self,
        clean_background_triplets: Sequence[np.ndarray],
        target_id: str | None = None,
        speed_px_per_frame: float = 3.5,
        position_angle_deg: float = 45.0,
        mag: float = 20.5,
    ) -> tuple[list[np.ndarray], list[InjectedSampleMetadata]]:
        """Inject a coherent moving object across an N-frame Fixed-Sky sequence."""
        n_frames = len(clean_background_triplets)
        synth_id = (
            target_id
            if target_id is not None
            else f"SYNTH_TRK_{self._rng.integers(100000, 999999)}"
        )

        rad = math.radians(position_angle_deg)
        step_x = speed_px_per_frame * math.cos(rad)
        step_y = speed_px_per_frame * math.sin(rad)

        h, w = clean_background_triplets[0].shape[1], clean_background_triplets[0].shape[2]
        # Start offset so tracklet passes through the interior
        half_span = (n_frames - 1) * 0.5
        x0 = (w / 2.0) - (half_span * step_x)
        y0 = (h / 2.0) - (half_span * step_y)

        out_triplets: list[np.ndarray] = []
        out_metadata: list[InjectedSampleMetadata] = []

        is_streak = speed_px_per_frame > 4.0

        for idx, bg in enumerate(clean_background_triplets):
            cur_x = x0 + (idx * step_x)
            cur_y = y0 + (idx * step_y)
            triplet, meta = self.inject_synthetic_source(
                clean_background_triplet=bg,
                target_id=synth_id,
                force_mag=mag,
                force_streak=is_streak,
                center_xy=(cur_x, cur_y),
            )
            out_triplets.append(triplet)
            out_metadata.append(meta)

        return out_triplets, out_metadata

    def _render_trailed_source(
        self,
        h: int,
        w: int,
        cx: float,
        cy: float,
        dx: float,
        dy: float,
        total_flux: float,
        fwhm: float,
    ) -> np.ndarray:
        """Render a trailed moving point source convolved with a 2D Gaussian PSF."""
        sigma = fwhm / 2.3548200450309493
        two_sigma_sq = 2.0 * sigma * sigma

        y_indices, x_indices = np.indices((h, w), dtype=np.float32)

        n_substeps = 20 if (abs(dx) > 0.5 or abs(dy) > 0.5) else 1
        sub_steps = np.linspace(-0.5, 0.5, n_substeps)

        profile = np.zeros((h, w), dtype=np.float32)
        norm_factor = 1.0 / (2.0 * math.pi * sigma * sigma)

        for s in sub_steps:
            sub_x = cx + (s * dx)
            sub_y = cy + (s * dy)
            dist_sq = (x_indices - sub_x) ** 2 + (y_indices - sub_y) ** 2
            profile += norm_factor * np.exp(-dist_sq / two_sigma_sq)

        profile /= n_substeps
        profile *= total_flux
        return profile
