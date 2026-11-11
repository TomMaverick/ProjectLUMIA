"""Unit tests for Module 3 WCS astrometry, BJD time-referencing, and Photutils photometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from pydantic import BaseModel, ValidationError

from LUMIA.physics.astrometry import (
    AstrometricCalibrationConfig,
    AstrometricMeasurement,
    BarycentricTimeResult,
    compute_bjd_tdb,
    extract_celestial_wcs,
    pixel_to_sky,
    propagate_astrometric_uncertainty,
    sky_to_pixel,
)
from LUMIA.physics.config import ObservatoryLocation
from LUMIA.physics.photometry import (
    AperturePhotometryMeasurement,
    PhotometryConfig,
    ZeroPointResult,
    compute_zero_point,
    perform_aperture_photometry,
)


def _make_tan_fits_header(
    *,
    ra0_deg: float = 180.0,
    dec0_deg: float = 25.0,
    scale_arcsec_px: float = 1.0,
    rotation_deg: float = 0.0,
    swap_axes: bool = False,
) -> fits.Header:
    """Build a synthetic TAN-projected celestial FITS header."""
    scale_deg: float = scale_arcsec_px / 3600.0
    theta: float = math.radians(rotation_deg)
    cos_t: float = math.cos(theta)
    sin_t: float = math.sin(theta)

    # Standard East-left (-scale on RA), North-up (+scale on Dec) before rotation
    cd_ra_x: float = -scale_deg * cos_t
    cd_ra_y: float = scale_deg * sin_t
    cd_dec_x: float = scale_deg * sin_t
    cd_dec_y: float = scale_deg * cos_t

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0

    if not swap_axes:
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = ra0_deg
        header["CRVAL2"] = dec0_deg
        header["CD1_1"] = cd_ra_x
        header["CD1_2"] = cd_ra_y
        header["CD2_1"] = cd_dec_x
        header["CD2_2"] = cd_dec_y
    else:
        header["CTYPE1"] = "DEC--TAN"
        header["CTYPE2"] = "RA---TAN"
        header["CRVAL1"] = dec0_deg
        header["CRVAL2"] = ra0_deg
        header["CD1_1"] = cd_dec_x
        header["CD1_2"] = cd_dec_y
        header["CD2_1"] = cd_ra_x
        header["CD2_2"] = cd_ra_y

    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["RADESYS"] = "ICRS"
    return header


def _inject_gaussian_star(
    image: np.ndarray,
    x0: float,
    y0: float,
    total_flux_adu: float,
    sigma_px: float = 1.5,
) -> None:
    """Add an analytic 2D circular Gaussian star evaluated at pixel centers."""
    ny, nx = image.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    norm: float = total_flux_adu / (2.0 * math.pi * sigma_px**2)
    r2: np.ndarray = (x_grid - x0) ** 2 + (y_grid - y0) ** 2
    image += norm * np.exp(-0.5 * r2 / (sigma_px**2))


def test_wcs_pixel_sky_roundtrip_and_3d_cube_extraction() -> None:
    """Verify WCS pixel-to-sky and sky-to-pixel transformations and 3D cube slicing."""
    header: fits.Header = _make_tan_fits_header(ra0_deg=180.0, dec0_deg=30.0, scale_arcsec_px=1.5)

    # CRPIX in FITS is 1-based (512.0, 512.0) -> 0-based (511.0, 511.0)
    ra_center, dec_center = pixel_to_sky(header, 511.0, 511.0)
    assert float(ra_center) == pytest.approx(180.0, abs=1e-9)
    assert float(dec_center) == pytest.approx(30.0, abs=1e-9)

    x_in = np.array([100.0, 511.0, 820.5])
    y_in = np.array([200.0, 511.0, 740.25])
    ra_deg, dec_deg = pixel_to_sky(header, x_in, y_in)
    x_out, y_out = sky_to_pixel(header, ra_deg, dec_deg)

    np.testing.assert_allclose(x_out, x_in, atol=1e-7)
    np.testing.assert_allclose(y_out, y_in, atol=1e-7)

    # 3D FITS cube edge case (e.g. Stokes or wavelength axis 3)
    cube_header = header.copy()
    cube_header["NAXIS"] = 3
    cube_header["NAXIS3"] = 4
    cube_header["CTYPE3"] = "FREQ"
    cube_header["CRPIX3"] = 1.0
    cube_header["CRVAL3"] = 1.4e9
    cube_header["CD3_3"] = 1.0e6

    celestial_wcs: WCS = extract_celestial_wcs(cube_header)
    assert celestial_wcs.pixel_n_dim == 2
    ra_cube, dec_cube = pixel_to_sky(celestial_wcs, 511.0, 511.0)
    assert float(ra_cube) == pytest.approx(180.0, abs=1e-9)
    assert float(dec_cube) == pytest.approx(30.0, abs=1e-9)


def test_astrometric_uncertainty_propagation_and_swapped_axes() -> None:
    """Verify Jacobian centroid error propagation and WCS RMS floor across axis orders."""
    cal_config = AstrometricCalibrationConfig(
        wcs_rms_ra_arcsec=0.10,
        wcs_rms_dec_arcsec=0.15,
    )

    # Unrotated 2.0 arcsec/px: sigma_x=0.2 px -> 0.4 arcsec RA, sigma_y=0.3 px -> 0.6 arcsec Dec
    header_std: fits.Header = _make_tan_fits_header(scale_arcsec_px=2.0, swap_axes=False)
    meas_std: AstrometricMeasurement = propagate_astrometric_uncertainty(
        header_std,
        x_px=511.0,
        y_px=511.0,
        sigma_x_px=0.2,
        sigma_y_px=0.3,
        calibration=cal_config,
    )
    assert isinstance(meas_std, BaseModel)
    assert meas_std.sigma_ra_cosdec_arcsec == pytest.approx(math.hypot(0.4, 0.10), rel=1e-6)
    assert meas_std.sigma_dec_arcsec == pytest.approx(math.hypot(0.6, 0.15), rel=1e-6)

    # Swapped FITS axes (CTYPE1=DEC--TAN, CTYPE2=RA---TAN) with same physical CD rows
    header_swapped: fits.Header = _make_tan_fits_header(scale_arcsec_px=2.0, swap_axes=True)
    meas_swapped: AstrometricMeasurement = propagate_astrometric_uncertainty(
        header_swapped,
        x_px=511.0,
        y_px=511.0,
        sigma_x_px=0.2,
        sigma_y_px=0.3,
        calibration=cal_config,
    )
    assert meas_swapped.ra_deg == pytest.approx(meas_std.ra_deg, abs=1e-9)
    assert meas_swapped.dec_deg == pytest.approx(meas_std.dec_deg, abs=1e-9)
    assert meas_swapped.sigma_ra_cosdec_arcsec == pytest.approx(
        meas_std.sigma_ra_cosdec_arcsec,
        rel=1e-6,
    )
    assert meas_swapped.sigma_dec_arcsec == pytest.approx(
        meas_std.sigma_dec_arcsec,
        rel=1e-6,
    )


def test_wcs_edge_cases_raise_errors() -> None:
    """Verify missing celestial WCS, singular CD matrices, and invalid coords fail cleanly."""
    uncalibrated_header = fits.Header()
    uncalibrated_header["NAXIS"] = 2
    uncalibrated_header["NAXIS1"] = 100
    uncalibrated_header["NAXIS2"] = 100
    with pytest.raises(ValueError, match="celestial"):
        extract_celestial_wcs(uncalibrated_header)

    singular_header: fits.Header = _make_tan_fits_header()
    singular_header["CD1_1"] = 0.0
    singular_header["CD1_2"] = 0.0
    with pytest.raises(ValueError, match="singular"):
        extract_celestial_wcs(singular_header)

    valid_header: fits.Header = _make_tan_fits_header()
    with pytest.raises(ValueError, match="finite"):
        pixel_to_sky(valid_header, float("nan"), 100.0)

    with pytest.raises(ValueError, match="dec_deg"):
        sky_to_pixel(valid_header, 180.0, 95.0)


def test_bjd_tdb_light_travel_time_correction() -> None:
    """Verify UTC-to-BJD_TDB conversion at opposition, conjunction, and exposure midpoint."""
    palomar = ObservatoryLocation(
        latitude_deg=33.3563,
        longitude_deg=-116.8648,
        elevation_m=1712.0,
        mpc_code="I41",
    )

    # Near vernal equinox (Sun near RA=0 deg, Dec=0 deg):
    # Target at RA=180 deg, Dec=0 deg is near opposition -> positive light-travel time (~+499 s)
    opp_res: BarycentricTimeResult = compute_bjd_tdb(
        obs_time="2025-03-20T12:00:00",
        ra_deg=180.0,
        dec_deg=0.0,
        observatory=palomar,
    )
    assert isinstance(opp_res, BaseModel)
    assert opp_res.light_travel_time_s > 490.0
    assert opp_res.light_travel_time_s < 510.0
    assert opp_res.bjd_tdb > opp_res.jd_utc

    # Target at RA=0 deg, Dec=0 deg is near solar conjunction -> negative light-travel time (~-499 s)
    conj_res: BarycentricTimeResult = compute_bjd_tdb(
        obs_time="2025-03-20T12:00:00",
        ra_deg=0.0,
        dec_deg=0.0,
        observatory=palomar,
    )
    assert conj_res.light_travel_time_s < -490.0
    assert conj_res.light_travel_time_s > -510.0

    # Exposure start timestamp shifted by half of 60 s exposure (+30 s)
    midpoint_res: BarycentricTimeResult = compute_bjd_tdb(
        obs_time="2025-03-20T12:00:00",
        ra_deg=180.0,
        dec_deg=0.0,
        observatory=palomar,
        exposure_time_s=60.0,
        timestamp_is_start=True,
    )
    delta_days: float = midpoint_res.bjd_tdb - opp_res.bjd_tdb
    assert delta_days * 86400.0 == pytest.approx(30.0, abs=0.05)


def test_aperture_photometry_and_zero_point_determination() -> None:
    """Verify Photutils aperture photometry, uncertainties, and zero-point calibration."""
    sky_bg_adu: float = 100.0
    image: np.ndarray = np.full((128, 128), sky_bg_adu, dtype=np.float64)

    gain: float = 2.0
    read_noise: float = 5.0
    exptime_s: float = 30.0
    true_zp: float = 25.0

    star_positions: list[tuple[float, float]] = [
        (32.0, 32.0),
        (96.0, 32.0),
        (64.0, 96.0),
    ]
    injected_fluxes_adu: list[float] = [60000.0, 24000.0, 6000.0]
    true_mags: list[float] = [
        true_zp - 2.5 * math.log10(f / exptime_s) for f in injected_fluxes_adu
    ]

    for (x0, y0), flux in zip(star_positions, injected_fluxes_adu, strict=True):
        _inject_gaussian_star(image, x0=x0, y0=y0, total_flux_adu=flux, sigma_px=1.5)

    phot_config = PhotometryConfig(
        aperture_radius_px=6.0,
        annulus_r_in_px=10.0,
        annulus_r_out_px=15.0,
        gain=gain,
        read_noise=read_noise,
        exposure_time_s=exptime_s,
    )

    uncalibrated: list[AperturePhotometryMeasurement] = perform_aperture_photometry(
        image=image,
        positions=star_positions,
        config=phot_config,
    )
    assert len(uncalibrated) == 3

    # Verify net fluxes and exact subpixel-weighted CCD variance sum(w_ij^2 * var_ij)
    from photutils.aperture import CircularAperture

    pixel_var_adu: np.ndarray = image / gain + (read_noise / gain) ** 2
    ap_masks = CircularAperture(
        star_positions,
        r=phot_config.aperture_radius_px,
    ).to_mask(method="exact")

    for meas, expected_flux, mask_obj in zip(
        uncalibrated,
        injected_fluxes_adu,
        ap_masks,
        strict=True,
    ):
        assert isinstance(meas, BaseModel)
        assert meas.net_flux_adu == pytest.approx(expected_flux, rel=2e-3)
        weights_2d: np.ndarray = mask_obj.to_image(image.shape)
        expected_var_adu: float = float(np.sum((weights_2d**2) * pixel_var_adu))
        assert meas.flux_err_adu == pytest.approx(math.sqrt(expected_var_adu), rel=1e-6)
        assert meas.calibrated_mag is None

    # Determine zero-point using first two stars as reference catalog stars
    zp_result: ZeroPointResult = compute_zero_point(
        flux_adu=[uncalibrated[0].net_flux_adu, uncalibrated[1].net_flux_adu],
        flux_err_adu=[uncalibrated[0].flux_err_adu, uncalibrated[1].flux_err_adu],
        catalog_mag=[true_mags[0], true_mags[1]],
        exposure_time_s=exptime_s,
    )
    assert isinstance(zp_result, BaseModel)
    assert zp_result.n_stars_used == 2
    assert zp_result.zero_point_mag == pytest.approx(true_zp, abs=0.01)
    assert zp_result.zero_point_err_mag > 0.0

    # Perform calibrated photometry and verify target star 3 magnitude and error
    calibrated: list[AperturePhotometryMeasurement] = perform_aperture_photometry(
        image=image,
        positions=[star_positions[2]],
        config=phot_config,
        zero_point=zp_result,
    )
    target_meas: AperturePhotometryMeasurement = calibrated[0]
    assert target_meas.calibrated_mag is not None
    assert target_meas.calibrated_mag_err is not None
    assert target_meas.calibrated_mag == pytest.approx(true_mags[2], abs=0.01)
    expected_inst_err: float = (2.5 / math.log(10.0)) * (
        target_meas.flux_err_adu / target_meas.net_flux_adu
    )
    expected_cal_err: float = math.hypot(expected_inst_err, zp_result.zero_point_err_mag)
    assert target_meas.calibrated_mag_err == pytest.approx(expected_cal_err, rel=1e-5)


def test_photometry_validation_and_non_detection_edge_cases() -> None:
    """Verify annulus geometry validation and graceful handling of zero/negative net flux."""
    with pytest.raises(ValidationError):
        PhotometryConfig(
            aperture_radius_px=8.0,
            annulus_r_in_px=6.0,
            annulus_r_out_px=12.0,
            gain=2.0,
            read_noise=5.0,
        )

    with pytest.raises(ValidationError):
        PhotometryConfig(
            aperture_radius_px=5.0,
            annulus_r_in_px=10.0,
            annulus_r_out_px=10.0,
            gain=2.0,
            read_noise=5.0,
        )

    # Pure flat background -> net flux == 0 -> instrumental and calibrated mag are None
    blank_image: np.ndarray = np.full((64, 64), 50.0, dtype=np.float64)
    config = PhotometryConfig(
        aperture_radius_px=4.0,
        annulus_r_in_px=8.0,
        annulus_r_out_px=12.0,
        gain=1.5,
        read_noise=4.0,
    )
    results = perform_aperture_photometry(blank_image, [(32.0, 32.0)], config)
    assert results[0].net_flux_adu == pytest.approx(0.0, abs=1e-9)
    assert results[0].instrumental_mag is None
    assert results[0].calibrated_mag is None
