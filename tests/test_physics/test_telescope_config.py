"""Unit tests for telescope hardware and observatory configuration loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from LUMIA.physics.config import ObservatoryLocation, TelescopeConfig, load_telescope_config

if TYPE_CHECKING:
    from pathlib import Path


def _valid_config_payload() -> dict[str, Any]:
    """Return a complete, physically valid telescope configuration dictionary."""
    return {
        "name": "ZTF Samuel Oschin 48-inch",
        "pixel_scale": 1.012,
        "gain": 6.2,
        "read_noise": 10.3,
        "saturation_limit": 65535.0,
        "dark_current": 0.02,
        "aperture_diameter_m": 1.22,
        "focal_length_m": 3.07,
        "observatory": {
            "latitude_deg": 33.3563,
            "longitude_deg": -116.8648,
            "elevation_m": 1712.0,
            "mpc_code": "I41",
        },
    }


def test_load_valid_telescope_config_from_yaml(tmp_path: Path) -> None:
    """Verify a valid YAML telescope profile loads into a validated Pydantic model."""
    config_file: Path = tmp_path / "telescope_profile.yaml"
    payload: dict[str, Any] = _valid_config_payload()
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config: TelescopeConfig = load_telescope_config(config_file)

    assert isinstance(config, BaseModel)
    assert isinstance(config, TelescopeConfig)
    assert isinstance(config.observatory, ObservatoryLocation)
    assert config.name == "ZTF Samuel Oschin 48-inch"
    assert config.pixel_scale == pytest.approx(1.012)
    assert config.gain == pytest.approx(6.2)
    assert config.read_noise == pytest.approx(10.3)
    assert config.saturation_limit == pytest.approx(65535.0)
    assert config.dark_current == pytest.approx(0.02)
    assert config.aperture_diameter_m == pytest.approx(1.22)
    assert config.focal_length_m == pytest.approx(3.07)
    assert config.observatory.latitude_deg == pytest.approx(33.3563)
    assert config.observatory.longitude_deg == pytest.approx(-116.8648)
    assert config.observatory.elevation_m == pytest.approx(1712.0)
    assert config.observatory.mpc_code == "I41"


@pytest.mark.parametrize(
    ("mutator_kind", "target_key", "bad_value", "expected_loc"),
    [
        ("delete", "pixel_scale", None, ("pixel_scale",)),
        ("delete", "gain", None, ("gain",)),
        ("delete", "read_noise", None, ("read_noise",)),
        ("delete", "saturation_limit", None, ("saturation_limit",)),
        ("delete", "observatory", None, ("observatory",)),
        ("set", "pixel_scale", "not_a_float", ("pixel_scale",)),
        ("set", "gain", True, ("gain",)),
        ("set", "read_noise", [10.3], ("read_noise",)),
        ("set", "pixel_scale", -0.5, ("pixel_scale",)),
        ("set_obs", "latitude_deg", 95.0, ("observatory", "latitude_deg")),
        ("set_obs", "longitude_deg", "west", ("observatory", "longitude_deg")),
        ("set_obs", "mpc_code", "TOOLONG", ("observatory", "mpc_code")),
    ],
)
def test_invalid_telescope_config_raises_validation_error(
    tmp_path: Path,
    mutator_kind: str,
    target_key: str,
    bad_value: Any,
    expected_loc: tuple[str, ...],
) -> None:
    """Verify missing mandatory keys, wrong data types, or invalid bounds raise ValidationError."""
    payload: dict[str, Any] = _valid_config_payload()
    if mutator_kind == "delete":
        del payload[target_key]
    elif mutator_kind == "set":
        payload[target_key] = bad_value
    elif mutator_kind == "set_obs":
        payload["observatory"][target_key] = bad_value

    config_file: Path = tmp_path / "invalid_telescope_profile.yaml"
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        load_telescope_config(config_file)

    error_locations: list[tuple[int | str, ...]] = [
        err["loc"] for err in exc_info.value.errors()
    ]
    assert expected_loc in error_locations
