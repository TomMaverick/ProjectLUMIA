"""Module 3: Science-Ready FITS Ingestion, Astrometry Gatekeeper & Noise Modeling.

Validates incoming calibrated FITS files, enforces physical detector and temporal
header keywords, executes local/remote Astrometry.net plate-solving fallback routing,
and propagates dark-current/readout noise models for ZOGY (Module 4).
"""

from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from pydantic import ConfigDict, Field

from LUMIA.core.schemas import LumiaBaseModel
from LUMIA.physics.astrometry import (
    compute_bjd_tdb,
    extract_celestial_wcs,
    pixel_to_sky,
)
from LUMIA.physics.config import (
    AstrometryRoutingConfig,
    ObservatoryLocation,
    TelescopeConfig,
)
from LUMIA.physics.photometry import compute_effective_noise_std

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "AstrometrySolverClient",
    "FITSHeaderValidationError",
    "FITSIngestionGatekeeper",
    "ScienceReadyFITS",
    "ScienceReadyHeader",
    "calculate_fits_noise_std",
]

_DEFAULT_OBSERVATORY = ObservatoryLocation(
    latitude_deg=28.7566,
    longitude_deg=-17.8920,
    elevation_m=2326.0,
    mpc_code="J43",
)


class FITSHeaderValidationError(ValueError):
    """Raised when essential physical or temporal FITS header keywords are missing or invalid."""


class ScienceReadyHeader(LumiaBaseModel):
    """Validated standardized physical, temporal, and astrometric metadata from FITS headers."""

    bjd_tdb: float = Field(..., description="Barycentric Julian Date in TDB scale.")
    exposure_time_s: float = Field(..., gt=0.0, description="Integration time in seconds.")
    gain: float = Field(..., gt=0.0, description="Detector conversion gain in e-/ADU.")
    read_noise: float = Field(..., ge=0.0, description="Detector readout noise RMS in e-.")
    dark_current: float = Field(default=0.0, ge=0.0, description="Dark current in e-/px/s.")
    saturation_limit: float | None = Field(default=None, gt=0.0, description="Saturation limit in ADU.")
    pixel_scale_arcsec: float | None = Field(default=None, gt=0.0, description="Pixel scale in arcsec/px.")
    filter_name: str | None = Field(default=None, description="Optical filter or passband.")
    field_id: str = Field(default="default", min_length=1, description="Observation field identifier.")
    ra_center_deg: float = Field(..., ge=0.0, lt=360.0, description="Boresight Right Ascension in degrees.")
    dec_center_deg: float = Field(..., ge=-90.0, le=90.0, description="Boresight Declination in degrees.")
    wcs_rms_arcsec: float = Field(default=0.15, ge=0.0, description="Astrometric solution 1-sigma uncertainty.")
    naxis1: int = Field(..., gt=0, description="Image width (columns) in pixels.")
    naxis2: int = Field(..., gt=0, description="Image height (rows) in pixels.")
    date_obs: str | None = Field(default=None, description="Observation start timestamp string.")


class ScienceReadyFITS(LumiaBaseModel):
    """Calibrated in-memory FITS image, parsed header metadata, and verified celestial WCS."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    data: np.ndarray
    metadata: ScienceReadyHeader
    wcs: WCS
    raw_header: dict[str, Any]


class AstrometrySolverClient:
    """Client for solving plate astrometry via local solve-field or remote Astrometry.net API."""

    def __init__(self, config: AstrometryRoutingConfig | None = None) -> None:
        self.config: AstrometryRoutingConfig = config if config is not None else AstrometryRoutingConfig()

    def solve_wcs(
        self,
        header: fits.Header,
        image_data: np.ndarray | None = None,
    ) -> WCS:
        """Verify existing WCS or route to local/remote plate-solving fallback."""
        try:
            return extract_celestial_wcs(header)
        except (ValueError, fits.VerifyError):
            pass

        mode = self.config.solver_mode
        if mode == "disabled":
            msg = "FITS header lacks valid celestial WCS and astrometry solving is disabled."
            raise FITSHeaderValidationError(msg)

        if mode == "local":
            return self._solve_local(header, image_data)
        if mode == "remote_api":
            return self._solve_remote_api(header, image_data)
        if mode == "auto":
            # Auto: try local binary first if available, otherwise fallback to remote API
            if shutil.which(self.config.local_solver_binary) is not None:
                try:
                    return self._solve_local(header, image_data)
                except Exception:
                    return self._solve_remote_api(header, image_data)
            return self._solve_remote_api(header, image_data)

        msg = f"Unknown astrometry solver mode: {mode!r}."
        raise ValueError(msg)

    def _solve_local(
        self,
        header: fits.Header,
        image_data: np.ndarray | None,
    ) -> WCS:
        """Solve plate astrometry using a local solve-field subprocess."""
        if image_data is None:
            msg = "Image pixel data is required for local Astrometry.net plate solving."
            raise FITSHeaderValidationError(msg)

        binary = self.config.local_solver_binary
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fits_in = tmp_path / "solve_input.fits"
            fits.writeto(fits_in, image_data.astype(np.float32), header, overwrite=True)

            cmd: list[str] = [
                binary,
                str(fits_in),
                "--no-plots",
                "--overwrite",
                "--downsample",
                str(self.config.downsample),
            ]
            if self.config.scale_lower_arcsec_per_px is not None:
                cmd.extend(["--scale-low", str(self.config.scale_lower_arcsec_per_px)])
            if self.config.scale_upper_arcsec_per_px is not None:
                cmd.extend(["--scale-high", str(self.config.scale_upper_arcsec_per_px)])
            if self.config.scale_lower_arcsec_per_px or self.config.scale_upper_arcsec_per_px:
                cmd.extend(["--scale-units", "arcsecperpix"])

            if self.config.ra_hint_deg is not None and self.config.dec_hint_deg is not None:
                cmd.extend(["--ra", str(self.config.ra_hint_deg), "--dec", str(self.config.dec_hint_deg)])
                if self.config.radius_hint_deg is not None:
                    cmd.extend(["--radius", str(self.config.radius_hint_deg)])

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                msg = f"Local solve-field timed out after {self.config.timeout_s}s."
                raise FITSHeaderValidationError(msg) from exc
            except OSError as exc:
                msg = f"Failed to execute local solve-field binary '{binary}': {exc}"
                raise FITSHeaderValidationError(msg) from exc

            wcs_file = tmp_path / "solve_input.wcs"
            new_file = tmp_path / "solve_input.new"
            target_out = wcs_file if wcs_file.is_file() else (new_file if new_file.is_file() else None)

            if target_out is None or proc.returncode != 0:
                err_detail = proc.stderr.strip() or proc.stdout.strip()
                msg = f"Local Astrometry.net solving failed (exit {proc.returncode}): {err_detail}"
                raise FITSHeaderValidationError(msg)

            solved_header = fits.getheader(target_out)
            return extract_celestial_wcs(solved_header)

    def _solve_remote_api(
        self,
        header: fits.Header,
        image_data: np.ndarray | None,
    ) -> WCS:
        """Solve plate astrometry via remote Astrometry.net (Nova) REST API."""
        if image_data is None:
            msg = "Image pixel data is required for remote Astrometry.net API solving."
            raise FITSHeaderValidationError(msg)

        api_url = self.config.remote_api_url.rstrip("/")
        timeout = self.config.timeout_s

        session_key = "anonymous"
        if self.config.remote_api_key:
            login_data = urllib.parse.urlencode(
                {"request-json": json.dumps({"apikey": self.config.remote_api_key})}
            ).encode("utf-8")
            login_req = urllib.request.Request(f"{api_url}/login", data=login_data, method="POST")
            try:
                with urllib.request.urlopen(login_req, timeout=timeout) as resp:
                    if resp.status != 200:
                        msg = f"Astrometry.net API login failed with status {resp.status}."
                        raise FITSHeaderValidationError(msg)
                    login_json = json.loads(resp.read().decode("utf-8"))
                    session_key = login_json.get("session", "anonymous")
            except FITSHeaderValidationError:
                raise
            except Exception as exc:
                msg = f"Astrometry.net API login failed: {exc}"
                raise FITSHeaderValidationError(msg) from exc

        bio = io.BytesIO()
        fits.writeto(bio, image_data.astype(np.float32), header)
        fits_bytes = bio.getvalue()

        upload_payload: dict[str, Any] = {
            "session": session_key,
            "allow_commercial_use": "d",
            "allow_modifications": "d",
            "publicly_visible": "n",
        }
        if self.config.scale_lower_arcsec_per_px is not None:
            upload_payload["scale_lower"] = self.config.scale_lower_arcsec_per_px
        if self.config.scale_upper_arcsec_per_px is not None:
            upload_payload["scale_upper"] = self.config.scale_upper_arcsec_per_px
        if self.config.ra_hint_deg is not None:
            upload_payload["center_ra"] = self.config.ra_hint_deg
        if self.config.dec_hint_deg is not None:
            upload_payload["center_dec"] = self.config.dec_hint_deg

        boundary = f"----LumiaBoundary{uuid.uuid4().hex}"
        body = bytearray()
        body.extend(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"request-json\"\r\n\r\n{json.dumps(upload_payload)}\r\n".encode("utf-8")
        )
        body.extend(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"upload.fits\"\r\nContent-Type: application/fits\r\n\r\n".encode("utf-8")
        )
        body.extend(fits_bytes)
        body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

        upload_req = urllib.request.Request(
            f"{api_url}/upload",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(upload_req, timeout=timeout) as resp:
                if resp.status != 200:
                    msg = f"Astrometry.net upload failed with status {resp.status}."
                    raise FITSHeaderValidationError(msg)
                upload_res = json.loads(resp.read().decode("utf-8"))
        except FITSHeaderValidationError:
            raise
        except Exception as exc:
            msg = f"Astrometry.net upload failed: {exc}"
            raise FITSHeaderValidationError(msg) from exc

        sub_id = upload_res.get("subid")
        if not sub_id:
            msg = f"Astrometry.net API did not return a submission ID: {upload_res}"
            raise FITSHeaderValidationError(msg)

        # Poll submission status for jobs
        t_start = time.perf_counter()
        job_id: int | None = None
        while time.perf_counter() - t_start < timeout:
            try:
                sub_req = urllib.request.Request(f"{api_url}/submissions/{sub_id}")
                with urllib.request.urlopen(sub_req, timeout=timeout) as resp:
                    stat_json = json.loads(resp.read().decode("utf-8"))
                    jobs = stat_json.get("jobs", [])
                    if jobs and jobs[0] is not None:
                        job_id = int(jobs[0])
                        break
            except Exception:
                pass
            time.sleep(0.5)

        if job_id is None:
            msg = f"Astrometry.net API solving timed out waiting for job ID (subid {sub_id})."
            raise FITSHeaderValidationError(msg)

        # Poll job status
        while time.perf_counter() - t_start < timeout:
            try:
                job_req = urllib.request.Request(f"{api_url}/jobs/{job_id}/status")
                with urllib.request.urlopen(job_req, timeout=timeout) as resp:
                    status = json.loads(resp.read().decode("utf-8")).get("status")
                    if status == "success":
                        break
                    if status == "failure":
                        msg = f"Astrometry.net remote plate solving failed for job {job_id}."
                        raise FITSHeaderValidationError(msg)
            except FITSHeaderValidationError:
                raise
            except Exception:
                pass
            time.sleep(0.5)

        # Fetch solved WCS header
        try:
            wcs_req = urllib.request.Request(f"{api_url}/jobs/{job_id}/wcs_file")
            with urllib.request.urlopen(wcs_req, timeout=timeout) as resp:
                wcs_text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            msg = f"Failed to download solved WCS header for job {job_id}: {exc}"
            raise FITSHeaderValidationError(msg) from exc

        wcs_header = fits.Header.fromstring(wcs_text, sep="\n")
        return extract_celestial_wcs(wcs_header)


class FITSIngestionGatekeeper:
    """Science-Ready FITS validation gatekeeper enforcing physical keywords and WCS integrity."""

    def __init__(
        self,
        *,
        astrometry_config: AstrometryRoutingConfig | None = None,
        observatory_location: ObservatoryLocation | None = None,
        telescope_profile: TelescopeConfig | None = None,
        solver_client: AstrometrySolverClient | None = None,
    ) -> None:
        self.astrometry_config: AstrometryRoutingConfig = (
            astrometry_config if astrometry_config is not None else AstrometryRoutingConfig()
        )
        self.observatory_location: ObservatoryLocation = (
            observatory_location if observatory_location is not None else _DEFAULT_OBSERVATORY
        )
        self.telescope_profile: TelescopeConfig | None = telescope_profile
        self.solver_client: AstrometrySolverClient = (
            solver_client
            if solver_client is not None
            else AstrometrySolverClient(self.astrometry_config)
        )

    def validate_and_ingest(
        self,
        fits_source: str | Path | bytes | fits.HDUList,
        *,
        field_id: str | None = None,
    ) -> ScienceReadyFITS:
        """Validate FITS data and header keywords into an immutable ScienceReadyFITS container."""
        hdul: fits.HDUList
        should_close: bool = False
        if isinstance(fits_source, (str, Path)):
            hdul = fits.open(str(fits_source))
            should_close = True
        elif isinstance(fits_source, (bytes, bytearray)):
            hdul = fits.open(io.BytesIO(fits_source))
            should_close = True
        elif isinstance(fits_source, fits.HDUList):
            hdul = fits_source
        else:
            msg = f"Unsupported fits_source type: {type(fits_source).__name__}."
            raise TypeError(msg)

        try:
            # Find image HDU (Primary or first ImageExtension)
            image_hdu: fits.ImageHDU | fits.PrimaryHDU | None = None
            for hdu in hdul:
                if hdu.data is not None and isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
                    image_hdu = hdu
                    break

            if image_hdu is None or image_hdu.data is None:
                msg = "FITS source contains no 2D image data HDU."
                raise FITSHeaderValidationError(msg)

            raw_header: fits.Header = image_hdu.header
            data_arr = np.asarray(image_hdu.data, dtype=np.float32)

            if data_arr.shape[0] == 0 or data_arr.shape[1] == 0:
                msg = "FITS image array dimensions must be non-zero."
                raise FITSHeaderValidationError(msg)

            if not np.all(np.isfinite(data_arr)):
                # Clean up non-finite pixels with 0.0 to prevent crashing downstream pipelines
                data_arr = np.nan_to_num(data_arr, nan=0.0, posinf=0.0, neginf=0.0)

            naxis1 = int(raw_header.get("NAXIS1", data_arr.shape[1]))
            naxis2 = int(raw_header.get("NAXIS2", data_arr.shape[0]))

            # 1. Mandatory EXPTIME
            exptime_val: float | None = self._extract_float(
                raw_header,
                ("EXPTIME", "EXPOSURE", "ELAPSED"),
            )
            if exptime_val is None or exptime_val <= 0.0 or not math.isfinite(exptime_val):
                msg = "Missing or invalid mandatory exposure time keyword 'EXPTIME'."
                raise FITSHeaderValidationError(msg)

            # 2. Mandatory GAIN
            gain_val: float | None = self._extract_float(
                raw_header,
                ("GAIN", "E_PER_ADU", "EGAIN"),
            )
            if gain_val is None:
                if self.telescope_profile is not None:
                    gain_val = float(self.telescope_profile.gain)
                else:
                    msg = "Missing mandatory detector gain keyword 'GAIN'."
                    raise FITSHeaderValidationError(msg)
            if gain_val <= 0.0 or not math.isfinite(gain_val):
                msg = f"Detector conversion gain must be strictly positive, got {gain_val}."
                raise FITSHeaderValidationError(msg)

            # 3. Mandatory RDNOISE
            rdnoise_val: float | None = self._extract_float(
                raw_header,
                ("RDNOISE", "READNOIS", "RON", "ENOISE"),
            )
            if rdnoise_val is None:
                if self.telescope_profile is not None:
                    rdnoise_val = float(self.telescope_profile.read_noise)
                else:
                    msg = "Missing mandatory readout noise keyword 'RDNOISE'."
                    raise FITSHeaderValidationError(msg)
            if rdnoise_val < 0.0 or not math.isfinite(rdnoise_val):
                msg = f"Readout noise must be non-negative, got {rdnoise_val}."
                raise FITSHeaderValidationError(msg)

            # 4. Mandatory Observation Time -> BJD(TDB)
            bjd_tdb = self._resolve_bjd_tdb(raw_header, exptime_val)

            # Optional metadata
            dark_val: float = self._extract_float(
                raw_header,
                ("DARKCUR", "DARK_CUR", "DARKCURR"),
            ) or (self.telescope_profile.dark_current if self.telescope_profile else 0.0)

            sat_val: float | None = self._extract_float(
                raw_header,
                ("SATURATE", "SATURAT", "DATAMAX"),
            ) or (self.telescope_profile.saturation_limit if self.telescope_profile else None)

            pix_scale: float | None = self._extract_float(
                raw_header,
                ("PIXSCALE", "SECPIX", "SCALE"),
            ) or (self.telescope_profile.pixel_scale if self.telescope_profile else None)

            filter_name: str | None = self._extract_str(
                raw_header,
                ("FILTER", "FILTER1", "BAND", "FILT"),
            )
            field_name: str = (
                field_id
                or self._extract_str(raw_header, ("FIELD", "OBJECT", "FIELDID", "TARGNAME"))
                or "default"
            )
            date_obs_str: str | None = self._extract_str(raw_header, ("DATE-OBS", "DATE_OBS"))

            # 5. Astrometric Calibration & WCS Validation
            wcs = self.solver_client.solve_wcs(raw_header, data_arr)
            center_x = (naxis1 - 1) / 2.0
            center_y = (naxis2 - 1) / 2.0
            ra_deg, dec_deg = pixel_to_sky(wcs, center_x, center_y)

            # WCS astrometric uncertainty RMS
            wcs_rms: float = float(raw_header.get("WCS_RMS", 0.15))

            header_meta = ScienceReadyHeader(
                bjd_tdb=bjd_tdb,
                exposure_time_s=exptime_val,
                gain=gain_val,
                read_noise=rdnoise_val,
                dark_current=max(0.0, float(dark_val)),
                saturation_limit=sat_val,
                pixel_scale_arcsec=pix_scale,
                filter_name=filter_name,
                field_id=field_name,
                ra_center_deg=ra_deg,
                dec_center_deg=dec_deg,
                wcs_rms_arcsec=max(0.0, wcs_rms),
                naxis1=naxis1,
                naxis2=naxis2,
                date_obs=date_obs_str,
            )

            # Safe raw header dictionary representation
            raw_dict: dict[str, Any] = {}
            for k in raw_header:
                if k:
                    try:
                        raw_dict[str(k)] = raw_header[k]
                    except Exception:
                        pass

            return ScienceReadyFITS(
                data=data_arr,
                metadata=header_meta,
                wcs=wcs,
                raw_header=raw_dict,
            )
        finally:
            if should_close:
                hdul.close()

    def _resolve_bjd_tdb(self, header: fits.Header, exptime_s: float) -> float:
        """Resolve light-travel time corrected BJD in TDB scale from available time keywords."""
        # Case A: Explicit BJD-OBS or BJD_TDB
        bjd_raw = self._extract_float(header, ("BJD-OBS", "BJD_OBS", "BJD_TDB", "BJD"))
        if bjd_raw is not None and bjd_raw > 2400000.0 and math.isfinite(bjd_raw):
            return bjd_raw

        # Case B: MJD-OBS or DATE-OBS
        mjd_raw = self._extract_float(header, ("MJD-OBS", "MJD_OBS", "MJD"))
        date_obs = self._extract_str(header, ("DATE-OBS", "DATE_OBS"))

        utc_time: Time | None = None
        if mjd_raw is not None and math.isfinite(mjd_raw) and mjd_raw > 0.0:
            utc_time = Time(mjd_raw, format="mjd", scale="utc")
        elif date_obs:
            try:
                utc_time = Time(date_obs, scale="utc")
            except Exception as exc:
                msg = f"Invalid DATE-OBS string '{date_obs}': {exc}"
                raise FITSHeaderValidationError(msg) from exc

        if utc_time is None:
            msg = "Missing mandatory observation time keyword ('BJD-OBS', 'MJD-OBS', or 'DATE-OBS')."
            raise FITSHeaderValidationError(msg)

        # Approximate mid-exposure UTC timestamp
        t_mid = utc_time + ((0.5 * exptime_s) * u.s)

        # Rough boresight coordinates for Barycentric Romer delay calculation
        ra_approx = self._extract_float(header, ("CRVAL1", "RA", "RA_DEG")) or 0.0
        dec_approx = self._extract_float(header, ("CRVAL2", "DEC", "DEC_DEG")) or 0.0

        bary_res = compute_bjd_tdb(
            t_mid.datetime,
            ra_deg=ra_approx % 360.0,
            dec_deg=max(-90.0, min(90.0, dec_approx)),
            observatory=self.observatory_location,
        )
        return bary_res.bjd_tdb

    @staticmethod
    def _extract_float(header: fits.Header, keys: tuple[str, ...]) -> float | None:
        """Extract the first matching float value from a tuple of header key aliases."""
        for k in keys:
            if k in header:
                try:
                    val = float(header[k])
                    if math.isfinite(val):
                        return val
                except (ValueError, TypeError):
                    continue
        return None

    @staticmethod
    def _extract_str(header: fits.Header, keys: tuple[str, ...]) -> str | None:
        """Extract the first matching stripped string from a tuple of header key aliases."""
        for k in keys:
            if k in header:
                val = str(header[k]).strip()
                if val:
                    return val
        return None


def calculate_fits_noise_std(
    fits_item: ScienceReadyFITS,
    sky_background_adu: float | None = None,
) -> float:
    """Calculate effective background RMS noise in ADU including dark current and read noise."""
    bg_adu: float
    if sky_background_adu is not None:
        bg_adu = max(0.0, float(sky_background_adu))
    else:
        # Estimate background median from image data
        bg_adu = max(0.0, float(np.median(fits_item.data)))

    meta = fits_item.metadata
    return compute_effective_noise_std(
        sky_background_adu=bg_adu,
        gain=meta.gain,
        read_noise=meta.read_noise,
        dark_current=meta.dark_current,
        exposure_time_s=meta.exposure_time_s,
    )
