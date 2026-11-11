"""Unit and integration tests for SQLite crawler ledger, resilient IPAC transport, and rate limiting."""

from __future__ import annotations

import io
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import responses
from astropy.io import fits
from astropy.wcs import WCS

from LUMIA.data_mining.archive_crawler import (
    ArchiveCrawler,
    TokenBucketRateLimiter,
    compute_geometric_midpoint,
    partition_tracklet_sliding_window,
)
from LUMIA.data_mining.crawler_ledger import CrawlerLedgerEntry, SQLiteCrawlerLedger
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


# -----------------------------------------------------------------------------
# 1. Crawler Ledger Tests
# -----------------------------------------------------------------------------


def test_sqlite_crawler_ledger_lifecycle(tmp_path: Path) -> None:
    """Verify SQLiteCrawlerLedger creates WAL-mode table and tracks completed tracklets idempotently."""
    db_file = tmp_path / "test_ledger.db"
    ledger = SQLiteCrawlerLedger(db_path=db_file)

    # Verify SQLite WAL mode is configured
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        row = cursor.fetchone()
        assert row is not None
        assert row[0].upper() == "WAL"

    # Initially empty
    assert not ledger.is_tracklet_completed("99942", "trk_001")
    assert len(ledger.get_completed_ids()) == 0

    entry1 = CrawlerLedgerEntry(
        object_id="99942",
        tracklet_id="trk_001",
        n_frames=4,
        split="train",
        status="COMPLETED",
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    ledger.commit_tracklet(entry1)

    assert ledger.is_tracklet_completed("99942", "trk_001")
    assert not ledger.is_tracklet_completed("99942", "trk_002")
    assert ("99942", "trk_001") in ledger.get_completed_ids()

    # Idempotent re-commit
    ledger.commit_tracklet(entry1)
    assert len(ledger.get_completed_ids()) == 1


# -----------------------------------------------------------------------------
# 2. Token Bucket Rate Limiter Tests
# -----------------------------------------------------------------------------


def test_token_bucket_rate_limiter_pacing() -> None:
    """Verify token bucket limiter restricts acquisition frequency within 2-4 req/s."""
    target_rate = 4.0  # 4 tokens per second -> ~0.25s per token
    limiter = TokenBucketRateLimiter(rate=target_rate, capacity=1.0)

    # Exhaust initial token
    limiter.acquire(1.0)

    start_time = time.monotonic()
    for _ in range(3):
        limiter.acquire(1.0)
    elapsed = time.monotonic() - start_time

    # 3 acquisitions at 4 req/s should take at least ~0.70 seconds (allowing timing variance)
    assert elapsed >= 0.65, f"Limiter did not pace requests: elapsed {elapsed:.3f}s"


# -----------------------------------------------------------------------------
# 3. Transport Resiliency & Retries with Responses
# -----------------------------------------------------------------------------


@responses.activate
def test_archive_crawler_retry_on_429_and_503() -> None:
    """Verify ArchiveCrawler retries on HTTP 429 and 503 before succeeding."""
    crawler = ArchiveCrawler(rate_limit_rps=100.0)  # High rate for fast test execution
    fits_bytes = _create_synthetic_fits_bytes(144)
    test_url = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/test_sci.fits"

    # Configure responses: 1x 429, 1x 503, then 200 OK
    responses.add(responses.GET, test_url, status=429)
    responses.add(responses.GET, test_url, status=503)
    responses.add(responses.GET, test_url, body=fits_bytes, status=200, content_type="image/fits")

    result = crawler.fetch_fits_cutout(test_url)
    assert result == fits_bytes
    assert len(responses.calls) == 3


@responses.activate
def test_archive_crawler_reference_coadd_caching() -> None:
    """Verify in-memory reference coadd caching avoids duplicate network requests."""
    crawler = ArchiveCrawler(rate_limit_rps=100.0)
    fits_bytes = _create_synthetic_fits_bytes(144)
    ref_url = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/test_ref.fits"

    responses.add(responses.GET, ref_url, body=fits_bytes, status=200, content_type="image/fits")

    cache_key = (650, "r", 4, 2, round(180.1234, 4), round(15.5678, 4))

    # First fetch: triggers network request
    data1 = crawler.fetch_reference_cutout(ref_url, cache_key=cache_key)
    assert data1 == fits_bytes
    assert len(responses.calls) == 1

    # Second fetch with identical key: served from in-memory cache with zero new calls
    data2 = crawler.fetch_reference_cutout(ref_url, cache_key=cache_key)
    assert data2 == fits_bytes
    assert len(responses.calls) == 1


# -----------------------------------------------------------------------------
# 4. FITS Validation & Truncation Rejection
# -----------------------------------------------------------------------------


def test_fits_magic_header_validation() -> None:
    """Verify extract_triplet_from_fits_buffers raises ValueError on non-FITS HTML buffers."""
    crawler = ArchiveCrawler()
    point = EphemerisPoint(
        target_id="99942",
        timestamp_utc=datetime.now(UTC),
        bjd_tdb=2459000.5,
        ra_deg=180.0,
        dec_deg=15.0,
        ra_rate_arcsec_min=1.0,
        dec_rate_arcsec_min=0.0,
        total_speed_arcsec_min=1.0,
        apparent_mag=20.0,
        delta_au=1.0,
        r_au=1.2,
        phase_angle_deg=15.0,
    )

    valid_fits = _create_synthetic_fits_bytes(144)
    html_error = b"<!DOCTYPE html><html><body>500 Internal Server Error</body></html>"

    with pytest.raises(ValueError, match="SIMPLE"):
        crawler.extract_triplet_from_fits_buffers(
            sci_bytes=html_error,
            ref_bytes=valid_fits,
            diff_bytes=valid_fits,
            point=point,
            ra_ref=180.0,
            dec_ref=15.0,
            epoch_index=0,
            tracklet_id="trk_test",
            stamp_size=144,
        )


def test_rejection_of_truncated_cutouts() -> None:
    """Verify cutouts smaller than physical stamp_size (e.g. truncated at detector edge) are rejected."""
    crawler = ArchiveCrawler()
    point = EphemerisPoint(
        target_id="99942",
        timestamp_utc=datetime.now(UTC),
        bjd_tdb=2459000.5,
        ra_deg=180.0,
        dec_deg=15.0,
        ra_rate_arcsec_min=1.0,
        dec_rate_arcsec_min=0.0,
        total_speed_arcsec_min=1.0,
        apparent_mag=20.0,
        delta_au=1.0,
        r_au=1.2,
        phase_angle_deg=15.0,
    )

    # 100x100 is smaller than requested 144x144 stamp
    truncated_fits = _create_synthetic_fits_bytes(100)

    result = crawler.extract_triplet_from_fits_buffers(
        sci_bytes=truncated_fits,
        ref_bytes=truncated_fits,
        diff_bytes=truncated_fits,
        point=point,
        ra_ref=180.0,
        dec_ref=15.0,
        epoch_index=0,
        tracklet_id="trk_test",
        stamp_size=144,
    )
    assert result is None, "Cutout truncated at detector boundaries (< 144px) must be rejected"


# -----------------------------------------------------------------------------
# 5. Kinematic Sliding Window & Containment
# -----------------------------------------------------------------------------


def test_sliding_window_containment_bound_144() -> None:
    """Verify sliding-window partitioning enforces Delta theta <= 52.6 arcsec for stamp_size=144."""
    now = datetime.now(UTC)
    scale_deg_per_px = 1.012 / 3600.0

    # Create 10 points spanning 60 arcsec (exceeds Delta theta_max = 52.624 arcsec)
    points = [
        EphemerisPoint(
            target_id="FAST_NEO",
            timestamp_utc=now,
            bjd_tdb=2459000.5 + (i * 0.005),
            ra_deg=180.0 + (i * 7.0 * scale_deg_per_px),
            dec_deg=15.0,
            apparent_mag=19.0,
            ra_rate_arcsec_min=5.0,
            dec_rate_arcsec_min=0.0,
            total_speed_arcsec_min=5.0,
            delta_au=1.0,
            r_au=1.2,
            phase_angle_deg=15.0,
        )
        for i in range(10)
    ]

    sub_tracklets = partition_tracklet_sliding_window(
        points, stamp_size=144, margin_px=12, min_length=2
    )

    assert len(sub_tracklets) >= 2
    for st in sub_tracklets:
        mid_ra, mid_dec = compute_geometric_midpoint(st)
        for pt in st:
            dra_arcsec = abs(pt.ra_deg - mid_ra) * 3600.0 * np.cos(np.radians(mid_dec))
            ddec_arcsec = abs(pt.dec_deg - mid_dec) * 3600.0
            dist_arcsec = np.hypot(dra_arcsec, ddec_arcsec)
            assert (
                dist_arcsec <= 52.63
            ), f"Point exceeded 52.6 arcsec kinematic bound: {dist_arcsec:.2f}"
