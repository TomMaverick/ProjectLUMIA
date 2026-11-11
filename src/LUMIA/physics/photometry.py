"""Aperture photometry and photometric zero-point calibration using Photutils."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Self

import numpy as np
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from pydantic import Field, model_validator

from LUMIA.core.schemas import LumiaBaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from LUMIA.physics.config import TelescopeConfig

_MAG_PREFACTOR: float = 2.5 / math.log(10.0)


class PhotometryConfig(LumiaBaseModel):
    """Validated aperture, sky-annulus, and detector noise parameters for photometry."""

    aperture_radius_px: float = Field(
        ...,
        gt=0.0,
        description="Circular aperture radius in pixels.",
    )
    annulus_r_in_px: float = Field(
        ...,
        gt=0.0,
        description="Inner sky-background annulus radius in pixels.",
    )
    annulus_r_out_px: float = Field(
        ...,
        gt=0.0,
        description="Outer sky-background annulus radius in pixels.",
    )
    gain: float = Field(
        ...,
        gt=0.0,
        description="Detector conversion gain in electrons per ADU (e-/ADU).",
    )
    read_noise: float = Field(
        ...,
        ge=0.0,
        description="Detector readout noise RMS in electrons (e-).",
    )
    dark_current: float = Field(
        default=0.0,
        ge=0.0,
        description="Residual dark current in electrons per pixel per second (e-/px/s).",
    )
    exposure_time_s: float = Field(
        default=1.0,
        gt=0.0,
        description="Science frame integration time in seconds.",
    )

    @model_validator(mode="after")
    def _validate_annulus_radii(self) -> Self:
        if self.annulus_r_in_px < self.aperture_radius_px:
            msg = "annulus_r_in_px must be greater than or equal to aperture_radius_px."
            raise ValueError(msg)
        if self.annulus_r_out_px <= self.annulus_r_in_px:
            msg = "annulus_r_out_px must be strictly greater than annulus_r_in_px."
            raise ValueError(msg)
        return self

    @classmethod
    def from_telescope_config(
        cls,
        telescope: TelescopeConfig,
        *,
        aperture_radius_px: float,
        annulus_r_in_px: float,
        annulus_r_out_px: float,
        exposure_time_s: float = 1.0,
    ) -> Self:
        """Build a PhotometryConfig inheriting gain, read_noise, and dark_current from TelescopeConfig."""
        return cls(
            aperture_radius_px=aperture_radius_px,
            annulus_r_in_px=annulus_r_in_px,
            annulus_r_out_px=annulus_r_out_px,
            gain=telescope.gain,
            read_noise=telescope.read_noise,
            dark_current=telescope.dark_current,
            exposure_time_s=exposure_time_s,
        )


class ZeroPointResult(LumiaBaseModel):
    """Validated photometric zero-point and propagated uncertainty in magnitudes."""

    zero_point_mag: float = Field(..., description="Calibrated zero-point magnitude.")
    zero_point_err_mag: float = Field(
        ...,
        ge=0.0,
        description="1-sigma uncertainty on the photometric zero-point [mag].",
    )
    n_stars_used: int = Field(..., ge=1, description="Number of valid reference stars.")
    rms_residual_mag: float = Field(
        ...,
        ge=0.0,
        description="RMS scatter of per-star zero-point residuals [mag].",
    )


class AperturePhotometryMeasurement(LumiaBaseModel):
    """Validated flux, background, and magnitude measurement for a single source."""

    x_px: float
    y_px: float
    aperture_sum_adu: float
    sky_background_per_px_adu: float
    net_flux_adu: float
    flux_err_adu: float = Field(..., ge=0.0)
    instrumental_mag: float | None = None
    instrumental_mag_err: float | None = Field(default=None, ge=0.0)
    calibrated_mag: float | None = None
    calibrated_mag_err: float | None = Field(default=None, ge=0.0)


def compute_effective_noise_std(
    sky_background_adu: float,
    gain: float,
    read_noise: float,
    *,
    dark_current: float = 0.0,
    exposure_time_s: float = 1.0,
) -> float:
    """Compute effective pixel-level 1-sigma noise standard deviation in ADU for ZOGY/photometry."""
    safe_bg = max(0.0, float(sky_background_adu))
    g = float(gain)
    if g <= 0.0:
        msg = f"gain must be strictly positive, got {gain}."
        raise ValueError(msg)
    rn_var_adu = (float(read_noise) / g) ** 2
    dark_var_adu = (max(0.0, float(dark_current)) * max(0.0, float(exposure_time_s))) / (g**2)
    return math.sqrt(safe_bg / g + rn_var_adu + dark_var_adu)


def compute_zero_point(
    flux_adu: Sequence[float] | np.ndarray,
    flux_err_adu: Sequence[float] | np.ndarray,
    catalog_mag: Sequence[float] | np.ndarray,
    *,
    exposure_time_s: float = 1.0,
    catalog_mag_err: Sequence[float] | np.ndarray | None = None,
) -> ZeroPointResult:
    """Compute an inverse-variance weighted photometric zero-point from reference stars."""
    if not math.isfinite(exposure_time_s) or exposure_time_s <= 0.0:
        msg = "exposure_time_s must be finite and strictly positive."
        raise ValueError(msg)

    f_arr: np.ndarray = np.asarray(flux_adu, dtype=np.float64)
    fe_arr: np.ndarray = np.asarray(flux_err_adu, dtype=np.float64)
    m_cat: np.ndarray = np.asarray(catalog_mag, dtype=np.float64)

    if f_arr.ndim != 1 or f_arr.shape != fe_arr.shape or f_arr.shape != m_cat.shape:
        msg = "flux_adu, flux_err_adu, and catalog_mag must be 1D arrays of equal length."
        raise ValueError(msg)
    if np.any(fe_arr < 0.0):
        msg = "flux_err_adu values must be non-negative."
        raise ValueError(msg)

    m_cat_err: np.ndarray = (
        np.zeros_like(m_cat)
        if catalog_mag_err is None
        else np.asarray(catalog_mag_err, dtype=np.float64)
    )
    if m_cat_err.shape != m_cat.shape or np.any(m_cat_err < 0.0):
        msg = "catalog_mag_err must match catalog_mag shape and be non-negative."
        raise ValueError(msg)

    valid_mask: np.ndarray = (
        np.isfinite(f_arr)
        & np.isfinite(fe_arr)
        & np.isfinite(m_cat)
        & np.isfinite(m_cat_err)
        & (f_arr > 0.0)
    )
    n_valid: int = int(np.count_nonzero(valid_mask))
    if n_valid == 0:
        msg = "At least one reference star with finite positive flux is required."
        raise ValueError(msg)

    f_v: np.ndarray = f_arr[valid_mask]
    fe_v: np.ndarray = fe_arr[valid_mask]
    m_v: np.ndarray = m_cat[valid_mask]
    me_v: np.ndarray = m_cat_err[valid_mask]

    inst_mag: np.ndarray = -2.5 * np.log10(f_v / exposure_time_s)
    zp_stars: np.ndarray = m_v - inst_mag
    zp_var: np.ndarray = (_MAG_PREFACTOR * (fe_v / f_v)) ** 2 + me_v**2

    if np.all(zp_var > 0.0):
        weights: np.ndarray = 1.0 / zp_var
        w_sum: float = float(np.sum(weights))
        zp_mean: float = float(np.sum(weights * zp_stars) / w_sum)
        zp_err: float = math.sqrt(1.0 / w_sum)
    else:
        zp_mean = float(np.mean(zp_stars))
        zp_err = (
            float(np.std(zp_stars, ddof=1) / math.sqrt(n_valid))
            if n_valid > 1
            else 0.0
        )

    rms_resid: float = float(np.sqrt(np.mean((zp_stars - zp_mean) ** 2)))
    return ZeroPointResult(
        zero_point_mag=zp_mean,
        zero_point_err_mag=zp_err,
        n_stars_used=n_valid,
        rms_residual_mag=rms_resid,
    )


def perform_aperture_photometry(
    image: np.ndarray,
    positions: Sequence[tuple[float, float]] | np.ndarray,
    config: PhotometryConfig,
    *,
    zero_point: ZeroPointResult | None = None,
    mask: np.ndarray | None = None,
) -> list[AperturePhotometryMeasurement]:
    """Perform local-annulus background-subtracted circular aperture photometry."""
    img: np.ndarray = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        msg = f"Expected non-empty 2D image array, got shape {img.shape}."
        raise ValueError(msg)

    pos_arr: np.ndarray = np.asarray(positions, dtype=np.float64)
    if pos_arr.ndim != 2 or pos_arr.shape[1] != 2 or pos_arr.shape[0] == 0:
        msg = "positions must be a non-empty (N, 2) sequence of (x_px, y_px) pairs."
        raise ValueError(msg)
    if not np.all(np.isfinite(pos_arr)):
        msg = "Aperture center coordinates must be finite."
        raise ValueError(msg)

    non_finite_mask: np.ndarray = ~np.isfinite(img)
    effective_mask: np.ndarray | None
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != img.shape:
            msg = f"mask shape {mask_arr.shape} must match image shape {img.shape}."
            raise ValueError(msg)
        effective_mask = mask_arr | non_finite_mask
    elif np.any(non_finite_mask):
        effective_mask = non_finite_mask
    else:
        effective_mask = None

    safe_img: np.ndarray = np.where(np.isfinite(img), img, 0.0)
    # Pixel-level 1-sigma CCD noise map in ADU:
    # sigma_adu = sqrt(max(I, 0)/gain + (read_noise/gain)^2 + (dark_current * t_exp)/gain^2)
    rn_adu_sq: float = (config.read_noise / config.gain) ** 2
    dark_adu_sq: float = (config.dark_current * config.exposure_time_s) / (config.gain**2)
    pixel_err_adu: np.ndarray = np.sqrt(
        np.maximum(safe_img, 0.0) / config.gain + rn_adu_sq + dark_adu_sq
    )

    aperture = CircularAperture(pos_arr, r=config.aperture_radius_px)
    annulus = CircularAnnulus(
        pos_arr,
        r_in=config.annulus_r_in_px,
        r_out=config.annulus_r_out_px,
    )

    phot_table = aperture_photometry(
        safe_img,
        aperture,
        error=pixel_err_adu,
        mask=effective_mask,
    )
    bkg_stats = ApertureStats(safe_img, annulus, mask=effective_mask)

    ap_areas: np.ndarray = np.asarray(
        aperture.area_overlap(safe_img, mask=effective_mask),
        dtype=np.float64,
    )
    bkg_medians: np.ndarray = np.asarray(bkg_stats.median, dtype=np.float64)
    bkg_stds: np.ndarray = np.asarray(bkg_stats.std, dtype=np.float64)
    bkg_areas: np.ndarray = np.asarray(bkg_stats.sum_aper_area.value, dtype=np.float64)

    measurements: list[AperturePhotometryMeasurement] = []
    for idx in range(pos_arr.shape[0]):
        raw_sum: float = float(phot_table["aperture_sum"][idx])
        raw_sum_err: float = float(phot_table["aperture_sum_err"][idx])
        n_pix: float = float(ap_areas[idx])
        sky_med: float = (
            float(bkg_medians[idx]) if math.isfinite(float(bkg_medians[idx])) else 0.0
        )
        sky_std: float = (
            float(bkg_stds[idx]) if math.isfinite(float(bkg_stds[idx])) else 0.0
        )
        n_bkg: float = float(bkg_areas[idx])

        net_flux: float = raw_sum - n_pix * sky_med
        bkg_sub_var: float = (
            (0.5 * math.pi) * (n_pix**2 / n_bkg) * (sky_std**2)
            if n_bkg > 0.0
            else 0.0
        )
        total_flux_err: float = math.sqrt(raw_sum_err**2 + bkg_sub_var)

        inst_mag: float | None = None
        inst_mag_err: float | None = None
        cal_mag: float | None = None
        cal_mag_err: float | None = None

        if net_flux > 0.0:
            inst_mag = -2.5 * math.log10(net_flux / config.exposure_time_s)
            inst_mag_err = _MAG_PREFACTOR * (total_flux_err / net_flux)
            if zero_point is not None:
                cal_mag = inst_mag + zero_point.zero_point_mag
                cal_mag_err = math.hypot(inst_mag_err, zero_point.zero_point_err_mag)

        measurements.append(
            AperturePhotometryMeasurement(
                x_px=float(pos_arr[idx, 0]),
                y_px=float(pos_arr[idx, 1]),
                aperture_sum_adu=raw_sum,
                sky_background_per_px_adu=sky_med,
                net_flux_adu=net_flux,
                flux_err_adu=total_flux_err,
                instrumental_mag=inst_mag,
                instrumental_mag_err=inst_mag_err,
                calibrated_mag=cal_mag,
                calibrated_mag_err=cal_mag_err,
            )
        )

    return measurements
