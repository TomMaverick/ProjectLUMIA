"""Unit and integration tests for multi-survey SODA cutout ingestion and streaming."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import responses
from astropy.io import fits
from scripts.run_mining import run_cli

from LUMIA.data_mining.archive_crawler import (
    ArchiveCrawler,
    SURVEY_PIXEL_SCALES,
    build_panstarrs_cutout_url,
    build_soda_cutout_url,
    build_ztf_ibe_cutout_url,
)
from LUMIA.data_mining.taxonomy import EphemerisPoint


def _create_synthetic_fits_bytes(stamp_size: int = 144) -> bytes:
    """Generate in-memory valid FITS byte stream with WCS header."""
    data = np.full((stamp_size, stamp_size), 50.0, dtype=np.float32)
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = stamp_size
    header["NAXIS2"] = stamp_size
    header["CRPIX1"] = stamp_size / 2.0
    header["CRPIX2"] = stamp_size / 2.0
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 15.0
    header["CDELT1"] = -0.000281
    header["CDELT2"] = 0.000281
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"

    hdu = fits.PrimaryHDU(data=data, header=header)
    buf = io.BytesIO()
    hdu.writeto(buf)
    return buf.getvalue()


def test_panstarrs_cutout_url_formatting() -> None:
    """Validate STScI Pan-STARRS fitscut URL construction."""
    url = build_panstarrs_cutout_url(ra_ref=120.5, dec_ref=25.2, size_px=144, filter_band="g")
    assert "ra=120.500000" in url
    assert "dec=25.200000" in url
    assert "size=144" in url
    assert "format=fits" in url
    assert "filter=g" in url
    assert "type=stack" in url


def test_soda_cutout_url_formatting() -> None:
    """Validate IVOA standard SODA cutout URL parameterization."""
    base_soda = "https://data.lsst.cloud/api/soda"
    url = build_soda_cutout_url(base_soda, ra_ref=150.0, dec_ref=-10.0, radius_arcsec=36.0, filter_band="r")
    assert base_soda in url
    assert "POS=CIRCLE" in url
    assert "150.000000" in url
    assert "-10.000000" in url
    # 36 arcsec = 0.01 degrees
    assert "0.010000" in url
    assert "BAND=r" in url


def test_query_overlapping_exposures_multi_survey() -> None:
    """Test query_overlapping_exposures across ZTF, Pan-STARRS, and Rubin."""
    crawler = ArchiveCrawler()
    point = EphemerisPoint(
        target_id="1",
        timestamp_utc=datetime(2026, 3, 15, 4, 30, tzinfo=UTC),
        bjd_tdb=2460750.5,
        ra_deg=180.5,
        dec_deg=12.2,
        ra_rate_arcsec_min=0.15,
        dec_rate_arcsec_min=-0.08,
        total_speed_arcsec_min=0.17,
        apparent_mag=18.5,
        delta_au=1.8,
        r_au=2.5,
        phase_angle_deg=15.0,
    )

    # 1. ZTF
    ztf_exposures = crawler.query_overlapping_exposures([point], survey="ZTF")
    assert len(ztf_exposures) == 1
    assert ztf_exposures[0].pixel_scale_arcsec == pytest.approx(1.012)
    assert "ipac.caltech.edu" in ztf_exposures[0].science_url

    # 2. Pan-STARRS
    ps_exposures = crawler.query_overlapping_exposures([point], survey="panstarrs", filter_band="g")
    assert len(ps_exposures) == 1
    assert ps_exposures[0].pixel_scale_arcsec == pytest.approx(0.25)
    assert "ps1images.stsci.edu" in ps_exposures[0].science_url
    assert "filter=g" in ps_exposures[0].science_url

    # 3. Rubin / LSST
    rubin_exposures = crawler.query_overlapping_exposures([point], survey="rubin", filter_band="i")
    assert len(rubin_exposures) == 1
    assert rubin_exposures[0].pixel_scale_arcsec == pytest.approx(0.20)
    assert "soda" in rubin_exposures[0].science_url
    assert "BAND=i" in rubin_exposures[0].science_url


@responses.activate
def test_fetch_fits_cutout_streaming_spooled_file() -> None:
    """Verify fetch_fits_cutout buffers into SpooledTemporaryFile without disk thrashing."""
    crawler = ArchiveCrawler(spooled_ram_mb=8)
    fits_bytes = _create_synthetic_fits_bytes(stamp_size=64)
    target_url = "https://data.lsst.cloud/api/soda?POS=CIRCLE%20180%2015%200.01&BAND=r"

    responses.add(
        responses.GET,
        target_url,
        body=fits_bytes,
        status=200,
        content_type="application/fits",
        stream=True,
    )

    downloaded = crawler.fetch_fits_cutout(target_url)
    assert downloaded.startswith(b"SIMPLE  =")
    assert len(downloaded) == len(fits_bytes)


def test_cli_dry_run_multi_survey_and_faint_injection(tmp_path: Path) -> None:
    """Validate CLI run in dry-run mode with Pan-STARRS and faint injection enabled."""
    qc_dir = tmp_path / "qc_output"
    out_h5 = tmp_path / "test_panstarrs.h5"

    exit_code = run_cli([
        "--dry-run",
        "--max-cutouts", "2",
        "--survey", "panstarrs",
        "--faint-injection",
        "--objects", "1",
        "--qc-dir", str(qc_dir),
        "--output-hdf5", str(out_h5),
    ])
    assert exit_code == 0
