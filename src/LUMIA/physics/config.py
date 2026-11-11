"""Pydantic schemas and YAML loader for telescope and observatory configurations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field

from LUMIA.core.schemas import LumiaBaseModel


class ObservatoryLocation(LumiaBaseModel):
    """Topocentric coordinates and optional MPC identifier for an observatory site."""

    latitude_deg: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Geodetic latitude in decimal degrees [-90, 90].",
    )
    longitude_deg: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Geodetic east-positive longitude in decimal degrees [-180, 180].",
    )
    elevation_m: float = Field(
        ...,
        ge=-500.0,
        le=10000.0,
        description="Site elevation above sea level in meters.",
    )
    mpc_code: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9]{3}$",
        description="Optional 3-character IAU Minor Planet Center observatory code.",
    )


class AstrometryRoutingConfig(LumiaBaseModel):
    """Configuration governing WCS verification and plate-solving fallback routing."""

    solver_mode: Literal["auto", "local", "remote_api", "disabled"] = Field(
        default="auto",
        description="Astrometric solving strategy: 'local' (solve-field), 'remote_api' (Nova), 'auto', or 'disabled'.",
    )
    local_solver_binary: str = Field(
        default="solve-field",
        description="Name or path of local Astrometry.net binary.",
    )
    remote_api_url: str = Field(
        default="https://nova.astrometry.net/api",
        description="Base URL for remote Astrometry.net REST API.",
    )
    remote_api_key: str | None = Field(
        default=None,
        description="Optional API key for remote Astrometry.net submission.",
    )
    timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout in seconds for astrometric solving execution.",
    )
    downsample: int = Field(
        default=2,
        ge=1,
        description="Downsampling factor for faster plate solving.",
    )
    scale_lower_arcsec_per_px: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional lower bound on pixel scale in arcsec/px.",
    )
    scale_upper_arcsec_per_px: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional upper bound on pixel scale in arcsec/px.",
    )
    ra_hint_deg: float | None = Field(
        default=None,
        ge=0.0,
        lt=360.0,
        description="Optional approximate RA center coordinate hint in degrees.",
    )
    dec_hint_deg: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Optional approximate Dec center coordinate hint in degrees.",
    )
    radius_hint_deg: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional search radius around coordinate hints in degrees.",
    )


class TelescopeConfig(LumiaBaseModel):
    """Validated hardware, optical, and site parameters for a telescope profile."""

    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable identifier of the telescope and camera profile.",
    )
    pixel_scale: float = Field(
        ...,
        gt=0.0,
        description="Plate scale in arcseconds per pixel (arcsec/px).",
    )
    gain: float = Field(
        ...,
        gt=0.0,
        description="Detector conversion gain in electrons per ADU (e-/ADU).",
    )
    read_noise: float = Field(
        ...,
        ge=0.0,
        description="Detector readout noise RMS in electrons (e-).",
    )
    saturation_limit: float = Field(
        ...,
        gt=0.0,
        description="Detector linearity/saturation limit in ADU.",
    )
    observatory: ObservatoryLocation = Field(
        ...,
        description="Topocentric observatory coordinates for BJD and parallax calculations.",
    )
    dark_current: float = Field(
        default=0.0,
        ge=0.0,
        description="Residual dark current in electrons per pixel per second (e-/px/s).",
    )
    aperture_diameter_m: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional primary aperture diameter in meters.",
    )
    focal_length_m: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional effective focal length in meters.",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a telescope configuration profile from a YAML file."""
        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw_data: Any = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            msg = (
                f"Invalid or unsafe YAML syntax in telescope configuration "
                f"'{config_path}': {exc}"
            )
            raise ValueError(msg) from exc

        if not isinstance(raw_data, Mapping):
            msg = (
                "Telescope YAML configuration root must be a mapping, "
                f"got {type(raw_data).__name__}."
            )
            raise ValueError(msg)

        return cls.model_validate(dict(raw_data))


def load_telescope_config(path: str | Path) -> TelescopeConfig:
    """Load and validate a TelescopeConfig instance from a YAML file path."""
    return TelescopeConfig.from_yaml(path)
