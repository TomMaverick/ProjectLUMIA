"""Minor Planet Center (MPC) client for querying known solar system ephemerides."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import TYPE_CHECKING, Any

import requests
from astropy.time import Time

from LUMIA.core.schemas import KnownObjectEphemeris

try:
    import astroquery.mpc as mpc_api
    from astroquery.exceptions import (
        EmptyResponseError,
        InvalidQueryError,
        RemoteServiceError,
        TableParseError,
        TimeoutError as AstroTimeoutError,
    )
except ImportError:  # pragma: no cover
    mpc_api = None  # type: ignore[assignment]
    EmptyResponseError = Exception  # type: ignore[misc, assignment]
    InvalidQueryError = Exception  # type: ignore[misc, assignment]
    RemoteServiceError = Exception  # type: ignore[misc, assignment]
    TableParseError = Exception  # type: ignore[misc, assignment]
    AstroTimeoutError = Exception  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "MinorPlanetCenterClient",
]


class MinorPlanetCenterClient:
    """Client for querying Minor Planet Center (MPC) ephemerides and known solar system bodies."""

    def __init__(
        self,
        default_observatory_code: str = "500",
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize MPC client with observatory site code and request timeout.

        Parameters
        ----------
        default_observatory_code : str, optional
            IAU MPC observatory code (e.g. '500' for geocentric, default '500').
        timeout_seconds : float, optional
            HTTP request timeout ceiling in seconds (default 30.0).
        """
        self._default_observatory_code: str = str(default_observatory_code)
        self._timeout_seconds: float = float(timeout_seconds)

    @property
    def default_observatory_code(self) -> str:
        """Return the default IAU observatory site code."""
        return self._default_observatory_code

    @property
    def timeout_seconds(self) -> float:
        """Return the timeout duration in seconds."""
        return self._timeout_seconds

    def query_field_ephemerides(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float,
        bjd_tdb: float,
        observatory_code: str | None = None,
    ) -> list[KnownObjectEphemeris]:
        """Query known objects intersecting sky cone / box at given observation epoch.

        Parameters
        ----------
        ra_center : float
            Pointing Right Ascension in degrees [0.0, 360.0).
        dec_center : float
            Pointing Declination in degrees [-90.0, 90.0].
        radius_deg : float
            Field-of-view search radius in degrees (> 0.0).
        bjd_tdb : float
            Observation epoch in Barycentric Julian Date (TDB).
        observatory_code : str | None, optional
            IAU observatory site code override. If None, uses default_observatory_code.

        Returns
        -------
        list[KnownObjectEphemeris]
            List of validated known object ephemerides, or empty list on timeout/error.
        """
        if not (0.0 <= ra_center < 360.0):
            msg = f"ra_center must be in [0.0, 360.0), got {ra_center}."
            raise ValueError(msg)
        if not (-90.0 <= dec_center <= 90.0):
            msg = f"dec_center must be in [-90.0, 90.0], got {dec_center}."
            raise ValueError(msg)
        if radius_deg <= 0.0:
            msg = f"radius_deg must be positive, got {radius_deg}."
            raise ValueError(msg)
        if bjd_tdb <= 0.0:
            msg = f"bjd_tdb must be a positive Julian Date, got {bjd_tdb}."
            raise ValueError(msg)

        obs_code = observatory_code or self._default_observatory_code

        # Convert BJD (TDB) to UTC Time representation for MPC
        obs_time = Time(bjd_tdb, format="jd", scale="tdb")
        utc_time = obs_time.utc
        epoch_jd = float(utc_time.jd)

        if mpc_api is None:  # pragma: no cover
            logger.warning("astroquery.mpc is not available.")
            return []

        try:
            # Query MPC web service via astroquery.mpc
            raw_response = mpc_api.MPC.query_objects(
                "box",
                ra=ra_center,
                dec=dec_center,
                radius=radius_deg,
                epoch_jd=epoch_jd,
                obscode=obs_code,
                cache=False,
            )
            return self.parse_response(raw_response)
        except (
            requests.RequestException,
            RemoteServiceError,
            AstroTimeoutError,
            EmptyResponseError,
            InvalidQueryError,
            TableParseError,
            RuntimeError,
            ValueError,
            KeyError,
        ) as exc:
            logger.warning("MPC ephemeris query failed or endpoint unavailable: %s", exc)
            return []
        except Exception as exc:
            logger.warning("Unexpected error during MPC ephemeris query: %s", exc)
            return []

    def parse_response(self, raw_data: Any) -> list[KnownObjectEphemeris]:
        """Parse raw JSON, CSV text, or tabular response into KnownObjectEphemeris objects.

        Parameters
        ----------
        raw_data : Any
            Raw response payload from MPC query (JSON string, dict, list, Table, or CSV).

        Returns
        -------
        list[KnownObjectEphemeris]
            List of strictly validated KnownObjectEphemeris instances.
        """
        if raw_data is None:
            return []

        records: list[dict[str, Any]] = []

        if isinstance(raw_data, str):
            text = raw_data.strip()
            if not text:
                return []
            # Attempt JSON parsing first
            try:
                parsed_json = json.loads(text)
                return self.parse_response(parsed_json)
            except (json.JSONDecodeError, ValueError):
                # Fall back to CSV DictReader parsing
                try:
                    reader = csv.DictReader(io.StringIO(text))
                    records = list(reader)
                except Exception as exc:
                    logger.warning("Failed to parse raw text response as CSV: %s", exc)
                    return []
        elif isinstance(raw_data, dict):
            # Unwrap dictionary responses containing object arrays
            for key in ("objects", "data", "results", "items"):
                if key in raw_data and isinstance(raw_data[key], (list, tuple)):
                    records = list(raw_data[key])
                    break
            else:
                records = [raw_data]
        elif isinstance(raw_data, (list, tuple)):
            records = [item for item in raw_data if isinstance(item, dict)]
        elif hasattr(raw_data, "colnames"):  # Astropy Table or QTable
            for row in raw_data:
                records.append({col: row[col] for col in raw_data.colnames})

        results: list[KnownObjectEphemeris] = []
        for rec in records:
            parsed_item = self._parse_single_record(rec)
            if parsed_item is not None:
                results.append(parsed_item)

        return results

    def _parse_single_record(self, record: dict[str, Any]) -> KnownObjectEphemeris | None:
        """Extract and validate single KnownObjectEphemeris from generic dictionary."""
        # 1. Target Designation / Name
        raw_desig = (
            record.get("designation")
            or record.get("name")
            or record.get("number")
            or record.get("target")
            or record.get("desig")
            or record.get("prov_desig")
        )
        if raw_desig is None:
            return None
        designation = str(raw_desig).strip()
        if not designation:
            return None

        # 2. Astrometric Coordinates
        raw_ra = (
            record.get("ra_deg")
            or record.get("ra")
            or record.get("RA")
            or record.get("ra_j2000")
        )
        raw_dec = (
            record.get("dec_deg")
            or record.get("dec")
            or record.get("DEC")
            or record.get("dec_j2000")
        )
        if raw_ra is None or raw_dec is None:
            return None

        try:
            ra_deg = float(raw_ra) % 360.0
            dec_deg = float(raw_dec)
            if not (-90.0 <= dec_deg <= 90.0):
                return None
        except (ValueError, TypeError):
            return None

        # 3. Visual Magnitude
        raw_vmag = (
            record.get("vmag")
            or record.get("v")
            or record.get("V")
            or record.get("mag")
            or record.get("magnitude")
        )
        vmag: float | None = None
        if raw_vmag is not None and str(raw_vmag).strip() != "":
            try:
                vmag = float(raw_vmag)
            except (ValueError, TypeError):
                vmag = None

        # 4. Sky-Plane Rates (arcsec / hr)
        raw_ra_rate = (
            record.get("ra_rate_arcsec_hr")
            or record.get("ra_rate")
            or record.get("dRA")
            or record.get("dRA cos(Dec)")
            or record.get("proper_motion_ra")
            or record.get("pm_ra")
            or 0.0
        )
        raw_dec_rate = (
            record.get("dec_rate_arcsec_hr")
            or record.get("dec_rate")
            or record.get("dDec")
            or record.get("proper_motion_dec")
            or record.get("pm_dec")
            or 0.0
        )
        try:
            ra_rate = float(raw_ra_rate)
            dec_rate = float(raw_dec_rate)
        except (ValueError, TypeError):
            ra_rate = 0.0
            dec_rate = 0.0

        try:
            return KnownObjectEphemeris(
                designation=designation,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                vmag=vmag,
                ra_rate_arcsec_hr=ra_rate,
                dec_rate_arcsec_hr=dec_rate,
            )
        except Exception as exc:
            logger.debug("Failed to instantiate KnownObjectEphemeris: %s", exc)
            return None
