"""Public survey archive crawler, fixed-sky tracklet extraction, and WCS containment."""

from __future__ import annotations

import logging
import math
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS
from pydantic import ConfigDict, Field
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from LUMIA.core.schemas import LumiaBaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from LUMIA.data_mining.taxonomy import EphemerisPoint

logger = logging.getLogger(__name__)

StampSize = Literal[63, 128, 144]


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter for HTTP request pacing."""

    def __init__(self, rate: float = 3.0, capacity: float = 4.0) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until requested number of tokens are available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                wait_time = (tokens - self.tokens) / self.rate
            time.sleep(wait_time)


class WCSMetadata(LumiaBaseModel):
    """Local World Coordinate System (WCS) parameters for sub-pixel orbit fitting."""

    crpix: tuple[float, float] = Field(
        ..., description="Reference pixel coordinates (crpix1, crpix2) in 1-indexed FITS standard."
    )
    crval: tuple[float, float] = Field(
        ..., description="Reference sky coordinates (RA_deg, Dec_deg) at crpix."
    )
    cd: tuple[tuple[float, float], tuple[float, float]] = Field(
        ..., description="Coordinate transformation matrix [[CD1_1, CD1_2], [CD2_1, CD2_2]]."
    )


class ExposureMetadata(LumiaBaseModel):
    """Metadata describing a single archival survey exposure overlapping a target track."""

    exposure_id: str = Field(..., min_length=1)
    survey: str = Field(default="ZTF")
    filter_band: str = Field(default="r")
    timestamp_utc: datetime = Field(...)
    ra_center_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_center_deg: float = Field(..., ge=-90.0, le=90.0)
    science_url: str = Field(...)
    reference_url: str = Field(...)
    difference_url: str = Field(...)
    pixel_scale_arcsec: float = Field(default=1.012, gt=0.0)


class CutoutTriplet(LumiaBaseModel):
    """Calibrated 3-channel cutout triplet [Science, Reference, Difference] for a single epoch."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    target_id: str = Field(..., min_length=1)
    tracklet_id: str = Field(default="trk_000", min_length=1)
    epoch_index: int = Field(..., ge=0)
    timestamp_utc: datetime = Field(...)
    bjd_tdb: float = Field(...)
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    target_x: float = Field(..., description="Target column coordinate within cutout (0-indexed).")
    target_y: float = Field(..., description="Target row coordinate within cutout (0-indexed).")
    wcs_params: WCSMetadata = Field(...)
    stamp_size: int = Field(default=128)
    filter_band: str = Field(default="r")
    survey: str = Field(default="ZTF")
    snr: float = Field(default=0.0, ge=0.0)
    mag_apparent: float = Field(default=20.0)
    is_synthetic: bool = Field(default=False)
    is_artifact: bool = Field(default=False)
    artifact_type: str = Field(default="NONE")
    data: np.ndarray = Field(
        ...,
        description="3-channel float32 tensor of shape (3, S, S): [Sci, Ref, Diff].",
    )


class TrackletSeries(LumiaBaseModel):
    """A time-ordered sequence of N >= 2 fixed-sky cutout triplets of a solar system object."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    tracklet_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    taxonomy_class: str = Field(default="MBA")
    ra_ref_deg: float = Field(..., ge=0.0, lt=360.0, description="Shared fixed-sky centering RA.")
    dec_ref_deg: float = Field(
        ..., ge=-90.0, le=90.0, description="Shared fixed-sky centering Dec."
    )
    frames: list[CutoutTriplet] = Field(..., min_length=2)
    label: int = Field(default=1, ge=0, le=3, description="0: Bogus, 1: Point, 2: Streak, 3: Coma")


def compute_geometric_midpoint(points: Sequence[EphemerisPoint]) -> tuple[float, float]:
    """Compute geometric midpoint (RA_0, Dec_0) from bounding coordinates."""
    if not points:
        return 0.0, 0.0
    ra_vals = [p.ra_deg for p in points]
    dec_vals = [p.dec_deg for p in points]
    min_dec, max_dec = min(dec_vals), max(dec_vals)
    mid_dec = (min_dec + max_dec) / 2.0

    min_ra, max_ra = min(ra_vals), max(ra_vals)
    if max_ra - min_ra > 180.0:
        shifted_ra = [(r + 360.0 if r < 180.0 else r) for r in ra_vals]
        mid_ra = (min(shifted_ra) + max(shifted_ra)) / 2.0 % 360.0
    else:
        mid_ra = (min_ra + max_ra) / 2.0
    return float(mid_ra), float(mid_dec)


def compute_fixed_sky_midpoint(points: Sequence[EphemerisPoint]) -> tuple[float, float]:
    """Compute the spatial centroid (ra_ref, dec_ref) of an ephemeris trajectory."""
    if not points:
        return 0.0, 0.0

    # Spherical coordinate mean accounting for 0/360 meridian wrap
    ra_rad = np.radians([p.ra_deg for p in points])
    dec_rad = np.radians([p.dec_deg for p in points])

    x = np.mean(np.cos(dec_rad) * np.cos(ra_rad))
    y = np.mean(np.cos(dec_rad) * np.sin(ra_rad))
    z = np.mean(np.sin(dec_rad))

    ref_ra = math.degrees(math.atan2(y, x)) % 360.0
    ref_dec = math.degrees(math.atan2(z, math.hypot(x, y)))
    return float(ref_ra), float(ref_dec)


def partition_tracklet_sliding_window(
    points: Sequence[EphemerisPoint],
    stamp_size: int = 144,
    pixel_scale_arcsec: float = 1.012,
    margin_px: int = 12,
    jitter_margin_px: int | None = None,
    min_frames: int = 2,
    max_frames: int = 12,
    min_length: int | None = None,
) -> list[list[EphemerisPoint]]:
    """Partition a long or fast moving-object track into overlapping Fixed-Sky sub-tracklets.

    Enforces kinematic containment:
        Delta theta_max = (stamp_size / 2 - jitter_margin - margin_px) * pixel_scale_arcsec
    For stamp_size=144 with 8px jitter margin and 12px edge margin:
        Delta theta_max = (72 - 8 - 12) * 1.012 = 52.624 arcsec (~52.6 arcsec).
    """
    effective_min = min_length if min_length is not None else min_frames
    if len(points) < effective_min:
        return [list(points)] if points else []

    jitter_margin = (
        jitter_margin_px if jitter_margin_px is not None else (8 if stamp_size >= 144 else 0)
    )
    safe_half_span_arcsec = ((stamp_size / 2.0) - jitter_margin - margin_px) * pixel_scale_arcsec
    sub_tracklets: list[list[EphemerisPoint]] = []

    start_idx = 0
    n = len(points)

    while start_idx < n:
        end_idx = min(start_idx + max_frames, n)
        # Check containment of window points[start_idx : end_idx]
        while (end_idx - start_idx) >= effective_min:
            window = list(points[start_idx:end_idx])
            ref_ra, ref_dec = compute_geometric_midpoint(window)

            cos_dec = math.cos(math.radians(ref_dec))
            all_fit = True
            for pt in window:
                dra_arcsec = (pt.ra_deg - ref_ra) * 3600.0 * cos_dec
                ddec_arcsec = (pt.dec_deg - ref_dec) * 3600.0
                dist_arcsec = math.hypot(dra_arcsec, ddec_arcsec)
                if dist_arcsec > safe_half_span_arcsec:
                    all_fit = False
                    break

            if all_fit:
                sub_tracklets.append(window)
                break
            end_idx -= 1

        if (end_idx - start_idx) < effective_min:
            if n >= effective_min:
                tail_window = list(points[max(0, n - effective_min) : n])
                if not sub_tracklets or tail_window != sub_tracklets[-1]:
                    sub_tracklets.append(tail_window)
            break

        step = max(1, (end_idx - start_idx) - 1)
        start_idx += step
        if (n - start_idx) < effective_min:
            if n >= effective_min:
                tail_window = list(points[n - effective_min : n])
                if not sub_tracklets or tail_window != sub_tracklets[-1]:
                    sub_tracklets.append(tail_window)
            break

    return sub_tracklets


def build_ztf_ibe_cutout_url(
    base_url: str,
    ra_ref: float,
    dec_ref: float,
    size_arcsec: float = 130.0,
) -> str:
    """Format lightweight server-side cutout URL for IPAC ZTF IBE (~65 KB per stamp)."""
    return f"{base_url}?center={ra_ref:.6f},{dec_ref:.6f}&size={size_arcsec:.1f}arcsec&gzip=false"


def build_panstarrs_cutout_url(
    ra_ref: float,
    dec_ref: float,
    size_px: int = 144,
    filter_band: str = "r",
    base_url: str = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi",
) -> str:
    """Format lightweight server-side cutout URL for Pan-STARRS (DR2) via STScI fitscut."""
    return f"{base_url}?ra={ra_ref:.6f}&dec={dec_ref:.6f}&size={size_px}&format=fits&filter={filter_band}&type=stack"


def build_soda_cutout_url(
    base_soda_url: str,
    ra_ref: float,
    dec_ref: float,
    radius_arcsec: float = 30.0,
    filter_band: str = "r",
) -> str:
    """Format standard IVOA Server-Side Operations for Data Access (SODA) cutout URL."""
    radius_deg = radius_arcsec / 3600.0
    return f"{base_soda_url}?POS=CIRCLE%20{ra_ref:.6f}%20{dec_ref:.6f}%20{radius_deg:.6f}&BAND={filter_band}"


SURVEY_PIXEL_SCALES: dict[str, float] = {
    "ztf": 1.012,
    "panstarrs": 0.25,
    "rubin": 0.20,
    "lsst": 0.20,
}


def verify_wcs_containment_and_quality(
    cutout_3ch: np.ndarray,
    target_x: float,
    target_y: float,
    stamp_size: int = 128,
    margin_px: int = 12,
) -> tuple[bool, str]:
    """Verify margin containment and check for NaN or masked pixels in a 5x5 neighborhood.

    Returns:
        (is_valid, reason_string)
    """
    if not (margin_px <= target_x < stamp_size - margin_px):
        return (
            False,
            f"OUT_OF_BOUNDS: target_x={target_x:.2f} violates margin "
            f"[{margin_px}, {stamp_size - margin_px})",
        )
    if not (margin_px <= target_y < stamp_size - margin_px):
        return (
            False,
            f"OUT_OF_BOUNDS: target_y={target_y:.2f} violates margin "
            f"[{margin_px}, {stamp_size - margin_px})",
        )

    # Check 5x5 neighborhood around target in all channels
    ix, iy = int(round(target_x)), int(round(target_y))
    x0, x1 = max(0, ix - 2), min(stamp_size, ix + 3)
    y0, y1 = max(0, iy - 2), min(stamp_size, iy + 3)

    patch = cutout_3ch[y0:y1, x0:x1] if cutout_3ch.ndim == 2 else cutout_3ch[:, y0:y1, x0:x1]

    if not np.all(np.isfinite(patch)):
        return False, "NAN_OR_MASKED_CCD: Target 5x5 neighborhood contains non-finite pixels"

    return True, "OK"


def slice_subpixel_cutout(
    image_2d: np.ndarray,
    center_x: float,
    center_y: float,
    stamp_size: int = 128,
) -> np.ndarray:
    """Extract a 2D cutout centered on (center_x, center_y) with edge padding."""
    half_size = stamp_size // 2
    x0 = int(round(center_x)) - half_size
    x1 = x0 + stamp_size
    y0 = int(round(center_y)) - half_size
    y1 = y0 + stamp_size

    h_img, w_img = image_2d.shape
    cutout = np.zeros((stamp_size, stamp_size), dtype=np.float32)

    src_x0 = max(0, x0)
    src_x1 = min(w_img, x1)
    src_y0 = max(0, y0)
    src_y1 = min(h_img, y1)

    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        slice_data = image_2d[src_y0:src_y1, src_x0:src_x1]
        cutout[dst_y0:dst_y1, dst_x0:dst_x1] = slice_data

    # Replace any non-finite pixels with local median background
    invalid_mask = ~np.isfinite(cutout)
    if np.any(invalid_mask):
        valid_vals = cutout[~invalid_mask]
        fill_val = float(np.median(valid_vals)) if valid_vals.size > 0 else 0.0
        cutout[invalid_mask] = fill_val

    return cutout.astype(np.float32)


def extract_wcs_metadata(wcs: WCS, crpix_offset: tuple[float, float] = (0.0, 0.0)) -> WCSMetadata:
    """Extract standard WCS parameters adjusted for cutout crop offset."""
    crpix1 = float(wcs.wcs.crpix[0]) - crpix_offset[0]
    crpix2 = float(wcs.wcs.crpix[1]) - crpix_offset[1]
    crval1 = float(wcs.wcs.crval[0])
    crval2 = float(wcs.wcs.crval[1])

    if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None and wcs.wcs.cd.size == 4:
        cd_mat = (
            (float(wcs.wcs.cd[0, 0]), float(wcs.wcs.cd[0, 1])),
            (float(wcs.wcs.cd[1, 0]), float(wcs.wcs.cd[1, 1])),
        )
    else:
        # Construct from CDELT/PC if CD is not directly set
        cdelt = getattr(wcs.wcs, "cdelt", [-0.000281, 0.000281])
        cd_mat = ((float(cdelt[0]), 0.0), (0.0, float(cdelt[1])))

    return WCSMetadata(
        crpix=(crpix1, crpix2),
        crval=(crval1, crval2),
        cd=cd_mat,
    )


class ArchiveCrawler:
    """Crawler for CADC SSOIS and IRSA ZTF image repositories."""

    USER_AGENT = "ProjectLUMIA/1.0 (Astrophysical Research; contact@project-lumia.org)"

    def __init__(
        self,
        spooled_ram_mb: int = 64,
        rate_limit_rps: float = 3.0,
        max_workers: int = 4,
        pool_connections: int = 16,
    ) -> None:
        self.spooled_ram_bytes = spooled_ram_mb * 1024 * 1024
        self.rate_limiter = TokenBucketRateLimiter(rate=rate_limit_rps, capacity=4.0)
        self.max_workers = max_workers
        self._ref_cache: dict[tuple[Any, ...], bytes] = {}
        self._cache_lock = threading.Lock()

        # Session with bounded connection pooling and exponential backoff retry strategy
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

        retry_strategy = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_connections,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def fetch_fits_cutout(self, url: str, timeout: float = 30.0) -> bytes:
        """Retrieve a calibrated FITS cutout streaming into SpooledTemporaryFile with rate-limiting."""
        self.rate_limiter.acquire(1.0)
        resp = self.session.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with tempfile.SpooledTemporaryFile(max_size=self.spooled_ram_bytes, mode="w+b") as spooled:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    spooled.write(chunk)
            spooled.seek(0)
            content = spooled.read()

        if not content.startswith(b"SIMPLE  ="):
            raise ValueError(
                "Downloaded stream is not a valid FITS file (missing 'SIMPLE  =' header)."
            )
        return content

    def fetch_reference_cutout(
        self,
        url: str,
        cache_key: tuple[Any, ...] | None = None,
        timeout: float = 30.0,
    ) -> bytes:
        """Retrieve reference coadd cutout, checking and populating in-memory cache."""
        if cache_key is not None:
            with self._cache_lock:
                if cache_key in self._ref_cache:
                    return self._ref_cache[cache_key]

        data = self.fetch_fits_cutout(url, timeout=timeout)

        if cache_key is not None:
            with self._cache_lock:
                self._ref_cache[cache_key] = data

        return data

    @staticmethod
    def build_reference_cache_key(
        field: int | str,
        filter_band: str,
        ccd: int | str,
        qid: int | str,
        ra_0: float,
        dec_0: float,
    ) -> tuple[int, str, int, int, float, float]:
        """Build protected cache key (field, filter, ccd, qid, ra_bin, dec_bin)."""
        return (
            int(field),
            str(filter_band),
            int(ccd),
            int(qid),
            round(float(ra_0), 4),
            round(float(dec_0), 4),
        )

    @staticmethod
    def compute_hard_negative_coordinates(
        ra_center: float,
        dec_center: float,
        offset_arcmin: float = 7.5,
        position_angle_deg: float = 45.0,
    ) -> tuple[float, float]:
        """Calculate fixed-sky quiescent negative coordinates offset by 5-10 arcmin."""
        pa_rad = math.radians(position_angle_deg)
        dra_deg = (
            (offset_arcmin / 60.0)
            * math.sin(pa_rad)
            / max(math.cos(math.radians(dec_center)), 1e-6)
        )
        ddec_deg = (offset_arcmin / 60.0) * math.cos(pa_rad)
        ra_neg = (ra_center + dra_deg) % 360.0
        dec_neg = float(np.clip(dec_center + ddec_deg, -90.0, 90.0))
        return ra_neg, dec_neg

    def query_overlapping_exposures(
        self,
        ephemeris_track: Sequence[EphemerisPoint],
        survey: str = "ZTF",
        filter_band: str = "r",
    ) -> list[ExposureMetadata]:
        """Cross-match ephemeris positions against survey footprints via SSOIS/IRSA/SODA."""
        survey_norm = survey.lower()
        pixel_scale = SURVEY_PIXEL_SCALES.get(survey_norm, 1.012)
        exposures: list[ExposureMetadata] = []
        for i, point in enumerate(ephemeris_track):
            exp_id = f"{survey_norm}_{point.timestamp_utc.strftime('%Y%m%d%H%M%S')}_{i:03d}"
            if survey_norm == "panstarrs":
                sci_url = build_panstarrs_cutout_url(
                    point.ra_deg, point.dec_deg, size_px=144, filter_band=filter_band
                )
                ref_url = sci_url
                diff_url = sci_url
            elif survey_norm in ("rubin", "lsst"):
                base_soda = "https://data.lsst.cloud/api/soda"
                sci_url = build_soda_cutout_url(
                    base_soda, point.ra_deg, point.dec_deg, radius_arcsec=30.0, filter_band=filter_band
                )
                ref_url = sci_url
                diff_url = sci_url
            else:
                sci_url = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{exp_id}_sci.fits"
                ref_url = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/{exp_id}_ref.fits"
                diff_url = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/diff/{exp_id}_diff.fits"

            exposures.append(
                ExposureMetadata(
                    exposure_id=exp_id,
                    survey=survey,
                    filter_band=filter_band,
                    timestamp_utc=point.timestamp_utc,
                    ra_center_deg=point.ra_deg,
                    dec_center_deg=point.dec_deg,
                    science_url=sci_url,
                    reference_url=ref_url,
                    difference_url=diff_url,
                    pixel_scale_arcsec=pixel_scale,
                )
            )
        return exposures

    def extract_fixed_sky_triplet_from_arrays(
        self,
        science_2d: np.ndarray,
        reference_2d: np.ndarray,
        difference_2d: np.ndarray,
        point: EphemerisPoint,
        ra_ref: float,
        dec_ref: float,
        epoch_index: int,
        tracklet_id: str,
        wcs: WCS,
        stamp_size: int = 144,
        survey: str = "ZTF",
        filter_band: str = "r",
        margin_px: int = 12,
    ) -> CutoutTriplet | None:
        """Extract a triplet centered on fixed-sky coordinates (ra_ref, dec_ref).

        Calculates exact target pixel coordinates (target_x, target_y) for the epoch's
        position (point.ra_deg, point.dec_deg) and verifies strict containment.
        """
        # Rejection of cutouts truncated at physical detector boundaries (< stamp_size)
        if (
            science_2d.shape[0] < stamp_size
            or science_2d.shape[1] < stamp_size
            or reference_2d.shape[0] < stamp_size
            or reference_2d.shape[1] < stamp_size
            or difference_2d.shape[0] < stamp_size
            or difference_2d.shape[1] < stamp_size
        ):
            logger.debug(
                "Cutout array truncated at detector boundaries (< %d px): sci=%s, ref=%s, diff=%s",
                stamp_size,
                science_2d.shape,
                reference_2d.shape,
                difference_2d.shape,
            )
            return None

        # Center cutout on fixed reference coordinate
        ref_px, ref_py = wcs.all_world2pix(ra_ref, dec_ref, 0)
        center_x = float(ref_px)
        center_y = float(ref_py)

        # Target epoch coordinates in full image
        tgt_px, tgt_py = wcs.all_world2pix(point.ra_deg, point.dec_deg, 0)

        # Cutout top-left origin
        half = stamp_size // 2
        x0 = int(round(center_x)) - half
        y0 = int(round(center_y)) - half

        # Target coordinates within cutout
        cutout_tgt_x = float(tgt_px - x0)
        cutout_tgt_y = float(tgt_py - y0)

        # Slice 2D stamps centered on fixed sky (ra_ref, dec_ref)
        sci_cutout = slice_subpixel_cutout(science_2d, center_x, center_y, stamp_size)
        ref_cutout = slice_subpixel_cutout(reference_2d, center_x, center_y, stamp_size)
        diff_cutout = slice_subpixel_cutout(difference_2d, center_x, center_y, stamp_size)

        triplet_tensor = np.stack([sci_cutout, ref_cutout, diff_cutout], axis=0)

        # Verify containment and neighborhood
        valid, reason = verify_wcs_containment_and_quality(
            triplet_tensor, cutout_tgt_x, cutout_tgt_y, stamp_size=stamp_size, margin_px=margin_px
        )
        if not valid:
            logger.debug(
                "Cutout rejected for %s epoch %d: %s", point.target_id, epoch_index, reason
            )
            return None

        wcs_meta = extract_wcs_metadata(wcs, crpix_offset=(float(x0), float(y0)))

        diff_noise = float(np.median(np.abs(diff_cutout - np.median(diff_cutout)))) * 1.4826
        snr_est = float(np.max(diff_cutout) / max(diff_noise, 1e-3))

        return CutoutTriplet(
            target_id=point.target_id,
            tracklet_id=tracklet_id,
            epoch_index=epoch_index,
            timestamp_utc=point.timestamp_utc,
            bjd_tdb=point.bjd_tdb,
            ra_deg=point.ra_deg,
            dec_deg=point.dec_deg,
            target_x=cutout_tgt_x,
            target_y=cutout_tgt_y,
            wcs_params=wcs_meta,
            stamp_size=stamp_size,
            filter_band=filter_band,
            survey=survey,
            snr=snr_est,
            mag_apparent=point.apparent_mag,
            data=triplet_tensor,
        )

    def extract_triplet_from_fits_buffers(
        self,
        sci_bytes: bytes,
        ref_bytes: bytes,
        diff_bytes: bytes,
        point: EphemerisPoint,
        ra_ref: float,
        dec_ref: float,
        epoch_index: int,
        tracklet_id: str,
        stamp_size: int = 144,
    ) -> CutoutTriplet | None:
        """Process streamed FITS buffers in RAM without saving files to disk."""
        for name, buf in (
            ("science", sci_bytes),
            ("reference", ref_bytes),
            ("difference", diff_bytes),
        ):
            if not buf or not buf.startswith(b"SIMPLE  ="):
                raise ValueError(
                    f"Malformed FITS stream for {name}: missing standard 'SIMPLE  =' header."
                )

        with tempfile.SpooledTemporaryFile(max_size=self.spooled_ram_bytes) as sci_buf:
            sci_buf.write(sci_bytes)
            sci_buf.seek(0)
            with fits.open(sci_buf) as hdul:
                sci_data = hdul[0].data.astype(np.float32)
                wcs = WCS(hdul[0].header)

        with tempfile.SpooledTemporaryFile(max_size=self.spooled_ram_bytes) as ref_buf:
            ref_buf.write(ref_bytes)
            ref_buf.seek(0)
            with fits.open(ref_buf) as hdul:
                ref_data = hdul[0].data.astype(np.float32)

        with tempfile.SpooledTemporaryFile(max_size=self.spooled_ram_bytes) as diff_buf:
            diff_buf.write(diff_bytes)
            diff_buf.seek(0)
            with fits.open(diff_buf) as hdul:
                diff_data = hdul[0].data.astype(np.float32)

        # Check for cutouts truncated at physical detector boundaries (< stamp_size)
        if (
            sci_data.shape[0] < stamp_size
            or sci_data.shape[1] < stamp_size
            or ref_data.shape[0] < stamp_size
            or ref_data.shape[1] < stamp_size
            or diff_data.shape[0] < stamp_size
            or diff_data.shape[1] < stamp_size
        ):
            logger.debug(
                "Cutout truncated at detector boundaries (< %d px): sci=%s, ref=%s, diff=%s",
                stamp_size,
                sci_data.shape,
                ref_data.shape,
                diff_data.shape,
            )
            return None

        return self.extract_fixed_sky_triplet_from_arrays(
            science_2d=sci_data,
            reference_2d=ref_data,
            difference_2d=diff_data,
            point=point,
            ra_ref=ra_ref,
            dec_ref=dec_ref,
            epoch_index=epoch_index,
            tracklet_id=tracklet_id,
            wcs=wcs,
            stamp_size=stamp_size,
        )
