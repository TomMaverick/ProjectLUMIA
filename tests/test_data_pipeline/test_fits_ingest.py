"""Hermetic unit tests for Module 3: Science-Ready FITS Ingestion & Astrometry Gatekeeper."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from LUMIA.data_pipeline.fits_ingest import (
    AstrometrySolverClient,
    FITSHeaderValidationError,
    FITSIngestionGatekeeper,
    ScienceReadyFITS,
    calculate_fits_noise_std,
)
from LUMIA.physics.config import (
    AstrometryRoutingConfig,
    ObservatoryLocation,
    TelescopeConfig,
)


# ---------------------------------------------------------------------------
# Helpers and Fixtures
# ---------------------------------------------------------------------------


def make_valid_fits_header(
    *,
    with_wcs: bool = True,
    exptime: float = 60.0,
    gain: float = 1.5,
    rdnoise: float = 5.0,
    bjd_obs: float | None = 2460000.5,
    mjd_obs: float | None = None,
    date_obs: str | None = None,
    naxis1: int = 100,
    naxis2: int = 100,
) -> fits.Header:
    """Create a synthetic FITS header populated with mandatory and optional keywords."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    header["EXPTIME"] = exptime
    header["GAIN"] = gain
    header["RDNOISE"] = rdnoise

    if bjd_obs is not None:
        header["BJD-OBS"] = bjd_obs
    if mjd_obs is not None:
        header["MJD-OBS"] = mjd_obs
    if date_obs is not None:
        header["DATE-OBS"] = date_obs

    if with_wcs:
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRPIX1"] = float(naxis1 // 2)
        header["CRPIX2"] = float(naxis2 // 2)
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 10.0
        header["CDELT1"] = -0.0005
        header["CDELT2"] = 0.0005
        header["CUNIT1"] = "deg"
        header["CUNIT2"] = "deg"

    return header


def make_fits_bytes(header: fits.Header, data: np.ndarray) -> bytes:
    """Serialize a FITS header and image array into in-memory bytes."""
    bio = io.BytesIO()
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(bio)
    return bio.getvalue()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a 100x100 synthetic science test frame."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=500.0, scale=20.0, size=(100, 100)).astype(np.float32)


@pytest.fixture
def valid_header() -> fits.Header:
    """Return a baseline valid header with TAN WCS."""
    return make_valid_fits_header()


@pytest.fixture
def default_observatory() -> ObservatoryLocation:
    """Return standard test observatory coordinates."""
    return ObservatoryLocation(
        latitude_deg=28.7566,
        longitude_deg=-17.8920,
        elevation_m=2326.0,
        mpc_code="J43",
    )


@pytest.fixture
def telescope_profile(default_observatory: ObservatoryLocation) -> TelescopeConfig:
    """Return a baseline telescope configuration."""
    return TelescopeConfig(
        name="TEST-TEL-01",
        pixel_scale=1.2,
        gain=1.8,
        read_noise=4.5,
        dark_current=0.02,
        saturation_limit=60000.0,
        observatory=default_observatory,
    )


# ---------------------------------------------------------------------------
# 1. Gatekeeper Ingestion & Source Type Tests
# ---------------------------------------------------------------------------


class TestFITSIngestionSourceTypes:
    """Verify ingestion from file paths, in-memory bytes, and HDULists."""

    def test_ingest_from_hdulist(
        self, sample_image: np.ndarray, valid_header: fits.Header
    ) -> None:
        """Gatekeeper ingests directly from an active fits.HDUList."""
        hdu = fits.PrimaryHDU(data=sample_image, header=valid_header)
        hdul = fits.HDUList([hdu])

        gatekeeper = FITSIngestionGatekeeper()
        result = gatekeeper.validate_and_ingest(hdul, field_id="FIELD_ALPHA")

        assert isinstance(result, ScienceReadyFITS)
        assert result.metadata.field_id == "FIELD_ALPHA"
        assert result.metadata.naxis1 == 100
        assert result.metadata.naxis2 == 100
        assert result.data.shape == (100, 100)
        assert isinstance(result.wcs, WCS)
        assert pytest.approx(result.metadata.gain, rel=1e-5) == 1.5
        assert pytest.approx(result.metadata.read_noise, rel=1e-5) == 5.0
        assert pytest.approx(result.metadata.exposure_time_s, rel=1e-5) == 60.0

    def test_ingest_from_bytes(
        self, sample_image: np.ndarray, valid_header: fits.Header
    ) -> None:
        """Gatekeeper ingests from raw in-memory FITS byte streams."""
        fits_bytes = make_fits_bytes(valid_header, sample_image)

        gatekeeper = FITSIngestionGatekeeper()
        result = gatekeeper.validate_and_ingest(fits_bytes)

        assert isinstance(result, ScienceReadyFITS)
        assert result.data.shape == (100, 100)
        assert pytest.approx(result.metadata.bjd_tdb, rel=1e-5) == 2460000.5

    def test_ingest_from_file_path(
        self, tmp_path: Path, sample_image: np.ndarray, valid_header: fits.Header
    ) -> None:
        """Gatekeeper ingests from a local filesystem Path."""
        fits_file = tmp_path / "science_frame.fits"
        hdu = fits.PrimaryHDU(data=sample_image, header=valid_header)
        hdu.writeto(fits_file)

        gatekeeper = FITSIngestionGatekeeper()
        result = gatekeeper.validate_and_ingest(fits_file)

        assert isinstance(result, ScienceReadyFITS)
        assert result.metadata.naxis1 == 100
        assert result.metadata.ra_center_deg > 0.0

    def test_unsupported_source_type_raises_type_error(self) -> None:
        """Passing an invalid object type raises TypeError."""
        gatekeeper = FITSIngestionGatekeeper()
        with pytest.raises(TypeError, match="Unsupported fits_source type"):
            gatekeeper.validate_and_ingest(12345)  # type: ignore[arg-type]

    def test_nan_and_inf_pixel_sanitization(
        self, valid_header: fits.Header
    ) -> None:
        """Corrupted pixels containing NaN or Inf are sanitized to 0.0."""
        dirty_data = np.full((50, 50), 100.0, dtype=np.float32)
        dirty_data[10, 10] = np.nan
        dirty_data[20, 20] = np.inf
        dirty_data[30, 30] = -np.inf

        fits_bytes = make_fits_bytes(valid_header, dirty_data)
        gatekeeper = FITSIngestionGatekeeper()
        result = gatekeeper.validate_and_ingest(fits_bytes)

        assert np.all(np.isfinite(result.data))
        assert result.data[10, 10] == 0.0
        assert result.data[20, 20] == 0.0
        assert result.data[30, 30] == 0.0


# ---------------------------------------------------------------------------
# 2. Mandatory Header Keyword Enforcement Tests
# ---------------------------------------------------------------------------


class TestMandatoryHeaderValidation:
    """Enforce strict validation of mandatory physical and temporal keywords."""

    def test_missing_or_invalid_exposure_time_raises(
        self, sample_image: np.ndarray
    ) -> None:
        """Header missing EXPTIME or non-positive value raises FITSHeaderValidationError."""
        gatekeeper = FITSIngestionGatekeeper()

        # Missing EXPTIME
        h1 = make_valid_fits_header()
        del h1["EXPTIME"]
        with pytest.raises(FITSHeaderValidationError, match="mandatory exposure time"):
            gatekeeper.validate_and_ingest(make_fits_bytes(h1, sample_image))

        # Non-positive EXPTIME
        h2 = make_valid_fits_header(exptime=0.0)
        with pytest.raises(FITSHeaderValidationError, match="mandatory exposure time"):
            gatekeeper.validate_and_ingest(make_fits_bytes(h2, sample_image))

    def test_missing_gain_raises_without_telescope_profile(
        self, sample_image: np.ndarray
    ) -> None:
        """Missing GAIN keyword without telescope configuration profile raises error."""
        h = make_valid_fits_header()
        del h["GAIN"]

        gatekeeper = FITSIngestionGatekeeper(telescope_profile=None)
        with pytest.raises(FITSHeaderValidationError, match="Missing mandatory detector gain"):
            gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))

    def test_missing_gain_falls_back_to_telescope_profile(
        self, sample_image: np.ndarray, telescope_profile: TelescopeConfig
    ) -> None:
        """Missing GAIN keyword is successfully retrieved from telescope profile."""
        h = make_valid_fits_header()
        del h["GAIN"]

        gatekeeper = FITSIngestionGatekeeper(telescope_profile=telescope_profile)
        result = gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))
        assert pytest.approx(result.metadata.gain, rel=1e-5) == telescope_profile.gain

    def test_invalid_negative_gain_raises(self, sample_image: np.ndarray) -> None:
        """Non-positive gain values are rejected."""
        h = make_valid_fits_header(gain=-1.0)
        gatekeeper = FITSIngestionGatekeeper()
        with pytest.raises(FITSHeaderValidationError, match="Detector conversion gain must be strictly positive"):
            gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))

    def test_missing_read_noise_raises_without_profile(
        self, sample_image: np.ndarray
    ) -> None:
        """Missing RDNOISE keyword without telescope configuration profile raises error."""
        h = make_valid_fits_header()
        del h["RDNOISE"]

        gatekeeper = FITSIngestionGatekeeper(telescope_profile=None)
        with pytest.raises(FITSHeaderValidationError, match="Missing mandatory readout noise"):
            gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))

    def test_missing_read_noise_falls_back_to_telescope_profile(
        self, sample_image: np.ndarray, telescope_profile: TelescopeConfig
    ) -> None:
        """Missing RDNOISE keyword is successfully retrieved from telescope profile."""
        h = make_valid_fits_header()
        del h["RDNOISE"]

        gatekeeper = FITSIngestionGatekeeper(telescope_profile=telescope_profile)
        result = gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))
        assert pytest.approx(result.metadata.read_noise, rel=1e-5) == telescope_profile.read_noise

    def test_missing_time_keywords_raises(self, sample_image: np.ndarray) -> None:
        """Missing observation time keywords (BJD, MJD, and DATE-OBS) raises error."""
        h = make_valid_fits_header(bjd_obs=None, mjd_obs=None, date_obs=None)
        gatekeeper = FITSIngestionGatekeeper()
        with pytest.raises(FITSHeaderValidationError, match="Missing mandatory observation time keyword"):
            gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))

    def test_invalid_date_obs_string_raises(self, sample_image: np.ndarray) -> None:
        """Corrupted DATE-OBS string raises FITSHeaderValidationError."""
        h = make_valid_fits_header(bjd_obs=None, mjd_obs=None, date_obs="not-a-valid-iso-date")
        gatekeeper = FITSIngestionGatekeeper()
        with pytest.raises(FITSHeaderValidationError, match="Invalid DATE-OBS string"):
            gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))

    def test_empty_hdu_list_raises(self) -> None:
        """FITS HDUList lacking any 2D image data raises FITSHeaderValidationError."""
        table_hdu = fits.BinTableHDU()
        hdul = fits.HDUList([table_hdu])

        gatekeeper = FITSIngestionGatekeeper()
        with pytest.raises(FITSHeaderValidationError, match="contains no 2D image data HDU"):
            gatekeeper.validate_and_ingest(hdul)


# ---------------------------------------------------------------------------
# 3. Temporal Resolution & BJD Conversion Tests
# ---------------------------------------------------------------------------


class TestTemporalBJDResolution:
    """Verify BJD(TDB) computation across direct and indirect time representations."""

    def test_direct_bjd_obs_keyword(self, sample_image: np.ndarray) -> None:
        """When BJD-OBS is present, it is directly adopted."""
        expected_bjd = 2460123.4567
        h = make_valid_fits_header(bjd_obs=expected_bjd)
        gatekeeper = FITSIngestionGatekeeper()
        result = gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))
        assert pytest.approx(result.metadata.bjd_tdb, rel=1e-6) == expected_bjd

    def test_mjd_obs_to_bjd_tdb_conversion(
        self, sample_image: np.ndarray, default_observatory: ObservatoryLocation
    ) -> None:
        """When only MJD-OBS is provided, gatekeeper computes mid-exposure BJD(TDB)."""
        # MJD 60000.0 corresponds to UTC 2023-02-25 00:00:00
        # JD = 2460000.5
        h = make_valid_fits_header(bjd_obs=None, mjd_obs=60000.0, exptime=120.0)
        gatekeeper = FITSIngestionGatekeeper(observatory_location=default_observatory)
        result = gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))

        # Mid-exposure offset is +60s = 60 / 86400 days ~ 0.0006944 days
        # Light travel time correction adds a small offset (typically ~ few minutes max)
        assert 2460000.0 < result.metadata.bjd_tdb < 2460002.0

    def test_date_obs_iso_to_bjd_tdb_conversion(
        self, sample_image: np.ndarray, default_observatory: ObservatoryLocation
    ) -> None:
        """When only DATE-OBS ISO string is provided, gatekeeper computes mid-exposure BJD(TDB)."""
        h = make_valid_fits_header(bjd_obs=None, mjd_obs=None, date_obs="2023-05-15T22:30:00.000")
        gatekeeper = FITSIngestionGatekeeper(observatory_location=default_observatory)
        result = gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))

        assert result.metadata.bjd_tdb > 2460000.0
        assert result.metadata.date_obs == "2023-05-15T22:30:00.000"


# ---------------------------------------------------------------------------
# 4. Astrometry Solver Client & Fallback Routing Tests
# ---------------------------------------------------------------------------


class TestAstrometrySolverRouting:
    """Verify WCS verification, local solve-field fallback, and remote API fallback."""

    def test_missing_wcs_with_solver_disabled_raises(
        self, sample_image: np.ndarray
    ) -> None:
        """Missing WCS raises error when solver routing is disabled."""
        h = make_valid_fits_header(with_wcs=False)
        astrom_config = AstrometryRoutingConfig(solver_mode="disabled")
        gatekeeper = FITSIngestionGatekeeper(astrometry_config=astrom_config)

        with pytest.raises(FITSHeaderValidationError, match="lacks valid celestial WCS and astrometry solving is disabled"):
            gatekeeper.validate_and_ingest(make_fits_bytes(h, sample_image))

    def test_local_solve_field_success(
        self, sample_image: np.ndarray, valid_header: fits.Header
    ) -> None:
        """Local solve-field subprocess successfully returns solved WCS."""
        h_no_wcs = make_valid_fits_header(with_wcs=False)

        # Mock subprocess.run to create a mock .wcs output file in the tempdir
        def mock_subprocess_run(cmd, cwd=None, **kwargs):  # type: ignore[no-untyped-def]
            out_wcs_file = Path(cwd) / "solve_input.wcs"
            # Write a valid WCS header
            valid_wcs_hdr = fits.Header()
            valid_wcs_hdr["CTYPE1"] = "RA---TAN"
            valid_wcs_hdr["CTYPE2"] = "DEC--TAN"
            valid_wcs_hdr["CRPIX1"] = 50.0
            valid_wcs_hdr["CRPIX2"] = 50.0
            valid_wcs_hdr["CRVAL1"] = 120.0
            valid_wcs_hdr["CRVAL2"] = -15.0
            valid_wcs_hdr["CDELT1"] = -0.0004
            valid_wcs_hdr["CDELT2"] = 0.0004
            valid_wcs_hdr["CUNIT1"] = "deg"
            valid_wcs_hdr["CUNIT2"] = "deg"
            fits.PrimaryHDU(header=valid_wcs_hdr).writeto(out_wcs_file)

            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "Solved!"
            mock_res.stderr = ""
            return mock_res

        astrom_config = AstrometryRoutingConfig(
            solver_mode="local",
            local_solver_binary="solve-field",
            scale_lower_arcsec_per_px=1.0,
            scale_upper_arcsec_per_px=2.0,
            ra_hint_deg=120.0,
            dec_hint_deg=-15.0,
        )
        gatekeeper = FITSIngestionGatekeeper(astrometry_config=astrom_config)

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = gatekeeper.validate_and_ingest(make_fits_bytes(h_no_wcs, sample_image))

        assert isinstance(result.wcs, WCS)
        assert pytest.approx(result.metadata.ra_center_deg, rel=1e-4) == 120.0
        assert pytest.approx(result.metadata.dec_center_deg, rel=1e-4) == -15.0

    def test_local_solve_field_failure_raises(
        self, sample_image: np.ndarray
    ) -> None:
        """Non-zero exit code from local solve-field raises FITSHeaderValidationError."""
        h_no_wcs = make_valid_fits_header(with_wcs=False)

        def mock_subprocess_failed(cmd, **kwargs):  # type: ignore[no-untyped-def]
            mock_res = MagicMock()
            mock_res.returncode = 1
            mock_res.stdout = ""
            mock_res.stderr = "No match found in indices."
            return mock_res

        astrom_config = AstrometryRoutingConfig(solver_mode="local")
        gatekeeper = FITSIngestionGatekeeper(astrometry_config=astrom_config)

        with patch("subprocess.run", side_effect=mock_subprocess_failed):
            with pytest.raises(FITSHeaderValidationError, match="Local Astrometry.net solving failed"):
                gatekeeper.validate_and_ingest(make_fits_bytes(h_no_wcs, sample_image))

    def test_remote_api_solving_success(
        self, sample_image: np.ndarray
    ) -> None:
        """Remote Astrometry.net REST API mock successfully uploads and retrieves solved WCS."""
        h_no_wcs = make_valid_fits_header(with_wcs=False)

        # Mock WCS header text returned by Nova ASTROMETRY.NET API
        solved_wcs_header_str = (
            "SIMPLE  =                    T / file does conform to FITS standard             \n"
            "BITPIX  =                  -32 / number of bits per data pixel                  \n"
            "NAXIS   =                    2 / number of data axes                            \n"
            "CTYPE1  = 'RA---TAN'           / celestial coordinate                              \n"
            "CTYPE2  = 'DEC--TAN'           / celestial coordinate                              \n"
            "CRPIX1  =                 50.0 / Reference pixel in X                           \n"
            "CRPIX2  =                 50.0 / Reference pixel in Y                           \n"
            "CRVAL1  =                210.0 / Reference celestial RA                         \n"
            "CRVAL2  =                 25.0 / Reference celestial Dec                        \n"
            "CDELT1  =              -0.0003 / Degree per pixel                               \n"
            "CDELT2  =               0.0003 / Degree per pixel                               \n"
            "END                                                                             \n"
        )

        def make_http_response(body_bytes: bytes, status: int = 200) -> MagicMock:
            resp = MagicMock()
            resp.status = status
            resp.read.return_value = body_bytes
            resp.__enter__.return_value = resp
            return resp

        def mock_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "/login" in url:
                return make_http_response(b'{"status": "success", "session": "test-session-123"}')
            if "/upload" in url:
                return make_http_response(b'{"status": "success", "subid": 9999}')
            if "/submissions/" in url:
                return make_http_response(b'{"jobs": [12345]}')
            if "/status" in url:
                return make_http_response(b'{"status": "success"}')
            if "/wcs_file" in url:
                return make_http_response(solved_wcs_header_str.encode("utf-8"))
            raise ValueError(f"Unexpected urlopen URL: {url}")

        astrom_config = AstrometryRoutingConfig(
            solver_mode="remote_api",
            remote_api_key="mock-api-key",
            remote_api_url="https://nova.astrometry.net/api",
        )
        gatekeeper = FITSIngestionGatekeeper(astrometry_config=astrom_config)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = gatekeeper.validate_and_ingest(make_fits_bytes(h_no_wcs, sample_image))

        assert isinstance(result.wcs, WCS)
        assert pytest.approx(result.metadata.ra_center_deg, rel=1e-4) == 210.0
        assert pytest.approx(result.metadata.dec_center_deg, rel=1e-4) == 25.0

    def test_remote_api_job_failure_raises(
        self, sample_image: np.ndarray
    ) -> None:
        """Remote plate solving returning status='failure' raises FITSHeaderValidationError."""
        h_no_wcs = make_valid_fits_header(with_wcs=False)

        def make_http_response(body_bytes: bytes, status: int = 200) -> MagicMock:
            resp = MagicMock()
            resp.status = status
            resp.read.return_value = body_bytes
            resp.__enter__.return_value = resp
            return resp

        def mock_urlopen_failed(req, timeout=None):  # type: ignore[no-untyped-def]
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "/upload" in url:
                return make_http_response(b'{"status": "success", "subid": 8888}')
            if "/submissions/" in url:
                return make_http_response(b'{"jobs": [54321]}')
            if "/status" in url:
                return make_http_response(b'{"status": "failure"}')
            raise ValueError(f"Unexpected urlopen URL: {url}")

        astrom_config = AstrometryRoutingConfig(
            solver_mode="remote_api",
            remote_api_key=None,
        )
        gatekeeper = FITSIngestionGatekeeper(astrometry_config=astrom_config)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen_failed):
            with pytest.raises(FITSHeaderValidationError, match="remote plate solving failed for job"):
                gatekeeper.validate_and_ingest(make_fits_bytes(h_no_wcs, sample_image))


# ---------------------------------------------------------------------------
# 5. Photometric Noise Modeling Tests
# ---------------------------------------------------------------------------


class TestFITSNoiseModeling:
    """Verify integration of dark current and readout noise into background variance."""

    def test_noise_calculation_with_explicit_background(
        self, sample_image: np.ndarray, valid_header: fits.Header
    ) -> None:
        """calculate_fits_noise_std accurately computes total noise RMS from inputs."""
        valid_header["DARKCUR"] = 0.05
        fits_bytes = make_fits_bytes(valid_header, sample_image)
        gatekeeper = FITSIngestionGatekeeper()
        fits_item = gatekeeper.validate_and_ingest(fits_bytes)

        # Sky BG = 400.0 ADU
        # Gain = 1.5 e-/ADU
        # Read noise = 5.0 e-
        # Dark current = 0.05 e-/px/s, exptime = 60s -> 3.0 e-
        # Total variance in electrons:
        # Var_e = (400 * 1.5) + (5.0^2) + 3.0 = 600 + 25 + 3 = 628.0 e-^2
        # std_e = sqrt(628.0) ~ 25.0599
        # std_adu = std_e / 1.5 ~ 16.7066
        noise_std = calculate_fits_noise_std(fits_item, sky_background_adu=400.0)

        expected_var_e = (400.0 * 1.5) + (5.0**2) + (0.05 * 60.0)
        expected_noise_std = (expected_var_e**0.5) / 1.5
        assert pytest.approx(noise_std, rel=1e-5) == expected_noise_std

    def test_noise_calculation_from_median_image_background(
        self, sample_image: np.ndarray, valid_header: fits.Header
    ) -> None:
        """calculate_fits_noise_std infers sky background from median image counts."""
        fits_bytes = make_fits_bytes(valid_header, sample_image)
        gatekeeper = FITSIngestionGatekeeper()
        fits_item = gatekeeper.validate_and_ingest(fits_bytes)

        median_bg = float(np.median(sample_image))
        noise_std = calculate_fits_noise_std(fits_item)

        expected_var_e = (median_bg * fits_item.metadata.gain) + (fits_item.metadata.read_noise**2)
        expected_noise_std = (expected_var_e**0.5) / fits_item.metadata.gain
        assert pytest.approx(noise_std, rel=1e-5) == expected_noise_std
