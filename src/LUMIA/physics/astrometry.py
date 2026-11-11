"""WCS coordinate transformations, astrometric uncertainty propagation, and BJD timing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers
from astropy.wcs import WCS
from pydantic import BaseModel, ConfigDict, Field

from LUMIA.physics.config import ObservatoryLocation  # noqa: TC001

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import ArrayLike


_ASTROMETRY_MODEL_CONFIG: ConfigDict = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)


class AstrometricCalibrationConfig(BaseModel):
    """Systematic WCS plate-solution uncertainty floor in tangent-plane arcseconds."""

    model_config = _ASTROMETRY_MODEL_CONFIG

    wcs_rms_ra_arcsec: float = Field(
        default=0.0,
        ge=0.0,
        description="Systematic WCS plate-solve RMS residual in RA*cos(Dec) [arcsec].",
    )
    wcs_rms_dec_arcsec: float = Field(
        default=0.0,
        ge=0.0,
        description="Systematic WCS plate-solve RMS residual in Dec [arcsec].",
    )


class AstrometricMeasurement(BaseModel):
    """Validated ICRS sky coordinates and propagated 1-sigma astrometric uncertainties."""

    model_config = _ASTROMETRY_MODEL_CONFIG

    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    sigma_ra_cosdec_arcsec: float = Field(..., ge=0.0)
    sigma_dec_arcsec: float = Field(..., ge=0.0)


class BarycentricTimeResult(BaseModel):
    """Validated Barycentric Julian Date (TDB) and light-travel time correction."""

    model_config = _ASTROMETRY_MODEL_CONFIG

    jd_utc: float = Field(..., description="Mid-exposure Julian Date in UTC scale.")
    bjd_tdb: float = Field(
        ...,
        description="Light-time corrected Barycentric Julian Date in TDB scale.",
    )
    light_travel_time_s: float = Field(
        ...,
        description="Barycentric Romer light-travel time offset in seconds.",
    )


def extract_celestial_wcs(wcs_or_header: WCS | fits.Header) -> WCS:
    """Extract and validate a non-singular 2D celestial WCS from a FITS Header or WCS."""
    raw_wcs: WCS = (
        WCS(wcs_or_header, relax=True, fix=False)
        if isinstance(wcs_or_header, fits.Header)
        else wcs_or_header
    )
    if not raw_wcs.has_celestial:
        msg = "Input FITS header/WCS does not contain valid celestial WCS axes."
        raise ValueError(msg)

    celestial_wcs: WCS = raw_wcs.celestial
    if celestial_wcs.pixel_n_dim != 2:
        msg = f"Expected 2 celestial WCS axes, got {celestial_wcs.pixel_n_dim}."
        raise ValueError(msg)

    scale_matrix: np.ndarray = np.asarray(
        celestial_wcs.pixel_scale_matrix,
        dtype=np.float64,
    )
    det: float = float(np.linalg.det(scale_matrix))
    if not np.isfinite(det) or abs(det) < 1e-30:
        msg = "Celestial WCS pixel_scale_matrix is singular or non-invertible."
        raise ValueError(msg)

    return celestial_wcs


def pixel_to_sky(
    wcs_or_header: WCS | fits.Header,
    x_px: ArrayLike,
    y_px: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 0-based pixel coordinates (x_px, y_px) to ICRS (ra_deg, dec_deg)."""
    cel_wcs: WCS = extract_celestial_wcs(wcs_or_header)
    x_arr: np.ndarray = np.asarray(x_px, dtype=np.float64)
    y_arr: np.ndarray = np.asarray(y_px, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        msg = f"x_px shape {x_arr.shape} does not match y_px shape {y_arr.shape}."
        raise ValueError(msg)
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        msg = "Pixel coordinates (x_px, y_px) must be finite."
        raise ValueError(msg)

    sky_coord: SkyCoord = cel_wcs.pixel_to_world(x_arr, y_arr).icrs
    ra_deg: np.ndarray = np.mod(np.asarray(sky_coord.ra.deg, dtype=np.float64), 360.0)
    dec_deg: np.ndarray = np.asarray(sky_coord.dec.deg, dtype=np.float64)
    if not (np.all(np.isfinite(ra_deg)) and np.all(np.isfinite(dec_deg))):
        msg = "WCS transformation produced non-finite sky coordinates."
        raise ValueError(msg)
    return ra_deg, dec_deg


def sky_to_pixel(
    wcs_or_header: WCS | fits.Header,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ICRS sky coordinates (ra_deg, dec_deg) to 0-based pixel coordinates."""
    cel_wcs: WCS = extract_celestial_wcs(wcs_or_header)
    ra_arr: np.ndarray = np.asarray(ra_deg, dtype=np.float64)
    dec_arr: np.ndarray = np.asarray(dec_deg, dtype=np.float64)
    if ra_arr.shape != dec_arr.shape:
        msg = f"ra_deg shape {ra_arr.shape} does not match dec_deg shape {dec_arr.shape}."
        raise ValueError(msg)
    if not (np.all(np.isfinite(ra_arr)) and np.all(np.isfinite(dec_arr))):
        msg = "Sky coordinates (ra_deg, dec_deg) must be finite."
        raise ValueError(msg)
    if np.any((dec_arr < -90.0) | (dec_arr > 90.0)):
        msg = "dec_deg values must lie within [-90.0, 90.0] degrees."
        raise ValueError(msg)

    coord = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame="icrs")
    x_raw, y_raw = cel_wcs.world_to_pixel(coord)
    x_px: np.ndarray = np.asarray(x_raw, dtype=np.float64)
    y_px: np.ndarray = np.asarray(y_raw, dtype=np.float64)
    if not (np.all(np.isfinite(x_px)) and np.all(np.isfinite(y_px))):
        msg = "WCS inverse transformation produced non-finite pixel coordinates."
        raise ValueError(msg)
    return x_px, y_px


def propagate_astrometric_uncertainty(
    wcs_or_header: WCS | fits.Header,
    x_px: float,
    y_px: float,
    sigma_x_px: float,
    sigma_y_px: float,
    calibration: AstrometricCalibrationConfig | None = None,
) -> AstrometricMeasurement:
    """Propagate pixel centroid uncertainties and WCS RMS into sky tangent-plane errors."""
    if sigma_x_px < 0.0 or sigma_y_px < 0.0:
        msg = "Pixel centroid uncertainties (sigma_x_px, sigma_y_px) must be non-negative."
        raise ValueError(msg)
    if not (math.isfinite(sigma_x_px) and math.isfinite(sigma_y_px)):
        msg = "Pixel centroid uncertainties must be finite."
        raise ValueError(msg)

    cal: AstrometricCalibrationConfig = (
        calibration if calibration is not None else AstrometricCalibrationConfig()
    )
    cel_wcs: WCS = extract_celestial_wcs(wcs_or_header)
    ra_arr, dec_arr = pixel_to_sky(cel_wcs, x_px, y_px)

    # Local Jacobian mapping (dx, dy) [px] -> world tangent-plane offsets [arcsec]
    jac_arcsec_per_px: np.ndarray = (
        np.asarray(cel_wcs.pixel_scale_matrix, dtype=np.float64) * 3600.0
    )
    cov_px: np.ndarray = np.array(
        [[sigma_x_px**2, 0.0], [0.0, sigma_y_px**2]],
        dtype=np.float64,
    )
    cov_world: np.ndarray = jac_arcsec_per_px @ cov_px @ jac_arcsec_per_px.T

    lng_idx: int = int(cel_wcs.wcs.lng)
    lat_idx: int = int(cel_wcs.wcs.lat)

    sigma_ra_cosdec: float = math.sqrt(
        float(cov_world[lng_idx, lng_idx]) + cal.wcs_rms_ra_arcsec**2
    )
    sigma_dec: float = math.sqrt(
        float(cov_world[lat_idx, lat_idx]) + cal.wcs_rms_dec_arcsec**2
    )

    return AstrometricMeasurement(
        ra_deg=float(ra_arr),
        dec_deg=float(dec_arr),
        sigma_ra_cosdec_arcsec=sigma_ra_cosdec,
        sigma_dec_arcsec=sigma_dec,
    )


def compute_bjd_tdb(
    obs_time: str | datetime | Time,
    ra_deg: float,
    dec_deg: float,
    observatory: ObservatoryLocation,
    *,
    exposure_time_s: float = 0.0,
    timestamp_is_start: bool = False,
) -> BarycentricTimeResult:
    """Convert an observation UTC timestamp to Barycentric Julian Date (BJD_TDB)."""
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
        msg = "Target sky coordinates (ra_deg, dec_deg) must be finite."
        raise ValueError(msg)
    if dec_deg < -90.0 or dec_deg > 90.0:
        msg = "dec_deg must lie within [-90.0, 90.0] degrees."
        raise ValueError(msg)
    if not math.isfinite(exposure_time_s) or exposure_time_s < 0.0:
        msg = "exposure_time_s must be finite and non-negative."
        raise ValueError(msg)

    earth_loc = EarthLocation.from_geodetic(
        lon=observatory.longitude_deg * u.deg,
        lat=observatory.latitude_deg * u.deg,
        height=observatory.elevation_m * u.m,
    )

    with (
        iers.conf.set_temp("auto_download", False),
        iers.conf.set_temp("auto_max_age", None),
    ):
        base_time: Time = (
            Time(obs_time.isot, format="isot", scale="utc", location=earth_loc)
            if isinstance(obs_time, Time)
            else Time(obs_time, scale="utc", location=earth_loc)
        )
        mid_time: Time = (
            base_time + 0.5 * exposure_time_s * u.s
            if timestamp_is_start and exposure_time_s > 0.0
            else base_time
        )
        target_coord = SkyCoord(
            ra=(ra_deg % 360.0) * u.deg,
            dec=dec_deg * u.deg,
            frame="icrs",
        )
        ltt_bary = mid_time.light_travel_time(
            target_coord,
            kind="barycentric",
            ephemeris="builtin",
        )
        bjd_tdb_val: float = float((mid_time.tdb + ltt_bary).jd)
        jd_utc_val: float = float(mid_time.utc.jd)
        ltt_sec: float = float(ltt_bary.to_value(u.s))

    return BarycentricTimeResult(
        jd_utc=jd_utc_val,
        bjd_tdb=bjd_tdb_val,
        light_travel_time_s=ltt_sec,
    )
