"""Hermetic tests for Minor Planet Center (MPC) ephemeris client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests
import responses
from astropy.table import Table
from pydantic import ValidationError

from LUMIA.core.schemas import KnownObjectEphemeris
from LUMIA.physics.mpc_client import MinorPlanetCenterClient

if TYPE_CHECKING:
    pass

MPC_WEB_SERVICE_URL = "https://minorplanetcenter.net/web_service"


# ---------------------------------------------------------------------------
# 1. KnownObjectEphemeris Pydantic Contract Tests
# ---------------------------------------------------------------------------


class TestKnownObjectEphemerisContract:
    """Verify strict validation and immutability of KnownObjectEphemeris."""

    def test_valid_ephemeris_creation(self) -> None:
        """Verify successful instantiation with valid parameters."""
        eph = KnownObjectEphemeris(
            designation="1999 RQ36",
            ra_deg=180.25,
            dec_deg=12.5,
            vmag=19.4,
            ra_rate_arcsec_hr=1.2,
            dec_rate_arcsec_hr=-0.8,
        )
        assert eph.designation == "1999 RQ36"
        assert eph.ra_deg == 180.25
        assert eph.dec_deg == 12.5
        assert eph.vmag == pytest.approx(19.4)
        assert eph.ra_rate_arcsec_hr == pytest.approx(1.2)
        assert eph.dec_rate_arcsec_hr == pytest.approx(-0.8)

    def test_optional_vmag_defaults_to_none(self) -> None:
        """Verify vmag is optional and defaults to None."""
        eph = KnownObjectEphemeris(
            designation="2020 CD3",
            ra_deg=45.0,
            dec_deg=-15.0,
        )
        assert eph.vmag is None
        assert eph.ra_rate_arcsec_hr == 0.0
        assert eph.dec_rate_arcsec_hr == 0.0

    @pytest.mark.parametrize(
        ("bad_kwargs", "match_str"),
        [
            ({"ra_deg": 360.0}, "less than 360"),
            ({"ra_deg": -0.1}, "greater than or equal to 0"),
            ({"dec_deg": 90.5}, "less than or equal to 90"),
            ({"dec_deg": -90.5}, "greater than or equal to -90"),
            ({"designation": ""}, "at least 1 character"),
            ({"unexpected_field": 123}, "extra_forbidden"),
        ],
    )
    def test_out_of_range_or_invalid_values_rejected(
        self, bad_kwargs: dict[str, object], match_str: str
    ) -> None:
        """Verify boundary violations fail fast with ValidationError."""
        base = {
            "designation": "Ceres",
            "ra_deg": 100.0,
            "dec_deg": 20.0,
        }
        base.update(bad_kwargs)
        with pytest.raises(ValidationError, match=match_str):
            KnownObjectEphemeris(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Hermetic Network Tests (responses)
# ---------------------------------------------------------------------------


class TestMPCClientHermeticQueries:
    """Verify hermetic network queries via responses mocking all external requests."""

    @responses.activate
    def test_query_field_ephemerides_parses_json_response(self) -> None:
        """Verify successful HTTP 200 JSON query produces validated ephemeris objects."""
        mock_payload = [
            {
                "designation": "1999 RQ36",
                "name": "Bennu",
                "ra": 182.45,
                "dec": 14.20,
                "v": 19.3,
                "ra_rate": 2.15,
                "dec_rate": -1.45,
            },
            {
                "designation": "433",
                "name": "Eros",
                "ra": 183.10,
                "dec": 14.85,
                "v": 14.1,
                "ra_rate": 0.85,
                "dec_rate": 0.35,
            },
        ]

        responses.add(
            responses.GET,
            MPC_WEB_SERVICE_URL,
            json=mock_payload,
            status=200,
        )

        client = MinorPlanetCenterClient(default_observatory_code="500")
        results = client.query_field_ephemerides(
            ra_center=182.5,
            dec_center=14.5,
            radius_deg=1.0,
            bjd_tdb=2460755.5,
        )

        assert len(results) == 2
        assert results[0].designation == "1999 RQ36"
        assert results[0].ra_deg == pytest.approx(182.45)
        assert results[0].dec_deg == pytest.approx(14.20)
        assert results[0].vmag == pytest.approx(19.3)
        assert results[0].ra_rate_arcsec_hr == pytest.approx(2.15)
        assert results[0].dec_rate_arcsec_hr == pytest.approx(-1.45)

        assert results[1].designation == "433"
        assert results[1].ra_deg == pytest.approx(183.10)
        assert results[1].vmag == pytest.approx(14.1)

    @responses.activate
    def test_empty_query_returns_empty_list_without_crash(self) -> None:
        """Verify empty JSON list from MPC returns empty list without exceptions."""
        responses.add(
            responses.GET,
            MPC_WEB_SERVICE_URL,
            json=[],
            status=200,
        )

        client = MinorPlanetCenterClient()
        results = client.query_field_ephemerides(
            ra_center=100.0,
            dec_center=20.0,
            radius_deg=0.5,
            bjd_tdb=2460755.5,
        )

        assert results == []

    @responses.activate
    def test_http_500_server_error_handled_gracefully(self) -> None:
        """Verify HTTP 500 error from MPC server degrades gracefully to empty list."""
        responses.add(
            responses.GET,
            MPC_WEB_SERVICE_URL,
            status=500,
        )

        client = MinorPlanetCenterClient()
        results = client.query_field_ephemerides(
            ra_center=180.0,
            dec_center=10.0,
            radius_deg=1.0,
            bjd_tdb=2460755.5,
        )

        assert results == []

    @responses.activate
    def test_connection_timeout_handled_gracefully(self) -> None:
        """Verify connection timeout degrades gracefully to empty list."""
        responses.add(
            responses.GET,
            MPC_WEB_SERVICE_URL,
            body=requests.exceptions.ConnectTimeout("Connection timed out"),
        )

        client = MinorPlanetCenterClient(timeout_seconds=2.0)
        results = client.query_field_ephemerides(
            ra_center=180.0,
            dec_center=10.0,
            radius_deg=1.0,
            bjd_tdb=2460755.5,
        )

        assert results == []

    @responses.activate
    def test_connection_failure_handled_gracefully(self) -> None:
        """Verify network connection failure degrades gracefully to empty list."""
        responses.add(
            responses.GET,
            MPC_WEB_SERVICE_URL,
            body=requests.exceptions.ConnectionError("Failed to resolve host"),
        )

        client = MinorPlanetCenterClient()
        results = client.query_field_ephemerides(
            ra_center=180.0,
            dec_center=10.0,
            radius_deg=1.0,
            bjd_tdb=2460755.5,
        )

        assert results == []


# ---------------------------------------------------------------------------
# 3. Format Parsing Unit Tests (CSV, JSON, Astropy Table)
# ---------------------------------------------------------------------------


class TestMPCResponseParsing:
    """Verify parse_response extracts KnownObjectEphemeris from multi-format payloads."""

    def test_parse_csv_text_response(self) -> None:
        """Verify CSV text format is parsed correctly."""
        csv_text = """designation,ra,dec,vmag,ra_rate,dec_rate
1999 RQ36,182.45,14.20,19.3,2.15,-1.45
(433) Eros,183.10,14.85,14.1,0.85,0.35"""

        client = MinorPlanetCenterClient()
        results = client.parse_response(csv_text)

        assert len(results) == 2
        assert results[0].designation == "1999 RQ36"
        assert results[0].ra_deg == pytest.approx(182.45)
        assert results[0].dec_deg == pytest.approx(14.20)
        assert results[0].vmag == pytest.approx(19.3)
        assert results[0].ra_rate_arcsec_hr == pytest.approx(2.15)
        assert results[0].dec_rate_arcsec_hr == pytest.approx(-1.45)

        assert results[1].designation == "(433) Eros"
        assert results[1].vmag == pytest.approx(14.1)

    def test_parse_astropy_table_response(self) -> None:
        """Verify Astropy Table / QTable objects are converted seamlessly."""
        tab = Table(
            {
                "designation": ["2014 AA", "2015 BZ509"],
                "ra_deg": [120.5, 121.2],
                "dec_deg": [-5.4, -4.9],
                "vmag": [21.5, 22.1],
                "ra_rate_arcsec_hr": [4.2, -1.8],
                "dec_rate_arcsec_hr": [0.6, 1.2],
            }
        )

        client = MinorPlanetCenterClient()
        results = client.parse_response(tab)

        assert len(results) == 2
        assert results[0].designation == "2014 AA"
        assert results[0].ra_deg == pytest.approx(120.5)
        assert results[1].designation == "2015 BZ509"
        assert results[1].dec_rate_arcsec_hr == pytest.approx(1.2)

    def test_parse_none_or_empty_inputs(self) -> None:
        """Verify None, empty string, or empty dict returns empty list."""
        client = MinorPlanetCenterClient()
        assert client.parse_response(None) == []
        assert client.parse_response("") == []
        assert client.parse_response("   ") == []
        assert client.parse_response([]) == []
        assert client.parse_response({}) == []

    def test_parse_skips_invalid_records(self) -> None:
        """Verify corrupt records are filtered out while preserving valid rows."""
        corrupt_payload = [
            {"designation": "Valid Asteroid", "ra": 100.0, "dec": 20.0, "v": 18.0},
            {"designation": "", "ra": 100.0, "dec": 20.0},  # Empty designation
            {"designation": "Bad Coordinates", "ra": "not_a_number", "dec": 20.0},
            {"designation": "Out of Range Dec", "ra": 100.0, "dec": 120.0},
        ]

        client = MinorPlanetCenterClient()
        results = client.parse_response(corrupt_payload)

        assert len(results) == 1
        assert results[0].designation == "Valid Asteroid"
        assert results[0].ra_deg == 100.0


# ---------------------------------------------------------------------------
# 4. Input Parameter Validation Tests
# ---------------------------------------------------------------------------


class TestMPCClientInputValidation:
    """Verify input validation for query_field_ephemerides."""

    def test_invalid_parameters_raise_value_errors(self) -> None:
        """Verify boundary parameter checking for RA, Dec, radius, and BJD."""
        client = MinorPlanetCenterClient()

        with pytest.raises(ValueError, match="ra_center must be in \\[0.0, 360.0\\)"):
            client.query_field_ephemerides(ra_center=360.0, dec_center=0.0, radius_deg=1.0, bjd_tdb=2460755.0)

        with pytest.raises(ValueError, match="dec_center must be in \\[-90.0, 90.0\\]"):
            client.query_field_ephemerides(ra_center=180.0, dec_center=95.0, radius_deg=1.0, bjd_tdb=2460755.0)

        with pytest.raises(ValueError, match="radius_deg must be positive"):
            client.query_field_ephemerides(ra_center=180.0, dec_center=0.0, radius_deg=-1.0, bjd_tdb=2460755.0)

        with pytest.raises(ValueError, match="bjd_tdb must be a positive Julian Date"):
            client.query_field_ephemerides(ra_center=180.0, dec_center=0.0, radius_deg=1.0, bjd_tdb=-10.0)
