"""P1-2 finite-float policy: NaN/Inf must be rejected on the remaining Pydantic models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from LUMIA.data_pipeline.orchestration import (
    CatalogQuerySummary,
    PipelineJobResult,
)
from LUMIA.ml.config import CNNClassifierConfig
from LUMIA.physics.astrometry import (
    AstrometricCalibrationConfig,
    AstrometricMeasurement,
    BarycentricTimeResult,
)
from LUMIA.physics.photometry import (
    AperturePhotometryMeasurement,
    PhotometryConfig,
    ZeroPointResult,
)

_NAN: float = float("nan")
_INF: float = float("inf")
_NEG_INF: float = float("-inf")

_PHOTOMETRY_CONFIG_KWARGS: dict[str, Any] = {
    "aperture_radius_px": 4.0,
    "annulus_r_in_px": 6.0,
    "annulus_r_out_px": 9.0,
    "gain": 1.5,
    "read_noise": 5.0,
    "exposure_time_s": 30.0,
}

_PHOTOMETRY_MEASUREMENT_KWARGS: dict[str, Any] = {
    "x_px": 10.0,
    "y_px": 12.0,
    "aperture_sum_adu": 1500.0,
    "sky_background_per_px_adu": 20.0,
    "net_flux_adu": 1200.0,
    "flux_err_adu": 15.0,
    "instrumental_mag": -7.5,
    "instrumental_mag_err": 0.02,
    "calibrated_mag": 19.5,
    "calibrated_mag_err": 0.03,
}

# (model class, valid kwargs, float fields that must reject NaN/Inf)
_MODEL_CASES: list[tuple[type[BaseModel], dict[str, Any], tuple[str, ...]]] = [
    (
        AstrometricCalibrationConfig,
        {"wcs_rms_ra_arcsec": 0.1, "wcs_rms_dec_arcsec": 0.1},
        ("wcs_rms_ra_arcsec", "wcs_rms_dec_arcsec"),
    ),
    (
        AstrometricMeasurement,
        {
            "ra_deg": 180.0,
            "dec_deg": 10.0,
            "sigma_ra_cosdec_arcsec": 0.1,
            "sigma_dec_arcsec": 0.1,
        },
        ("ra_deg", "dec_deg", "sigma_ra_cosdec_arcsec", "sigma_dec_arcsec"),
    ),
    (
        BarycentricTimeResult,
        {"jd_utc": 2460000.5, "bjd_tdb": 2460000.5001, "light_travel_time_s": 12.5},
        ("jd_utc", "bjd_tdb", "light_travel_time_s"),
    ),
    (
        PhotometryConfig,
        _PHOTOMETRY_CONFIG_KWARGS,
        (
            "aperture_radius_px",
            "annulus_r_in_px",
            "annulus_r_out_px",
            "gain",
            "read_noise",
            "exposure_time_s",
        ),
    ),
    (
        ZeroPointResult,
        {
            "zero_point_mag": 25.1,
            "zero_point_err_mag": 0.02,
            "n_stars_used": 12,
            "rms_residual_mag": 0.04,
        },
        ("zero_point_mag", "zero_point_err_mag", "rms_residual_mag"),
    ),
    (
        AperturePhotometryMeasurement,
        _PHOTOMETRY_MEASUREMENT_KWARGS,
        tuple(
            key
            for key, value in _PHOTOMETRY_MEASUREMENT_KWARGS.items()
            if isinstance(value, float)
        ),
    ),
    (
        CatalogQuerySummary,
        {"matched_source_count": 3, "brightest_mag_g": 14.2},
        ("brightest_mag_g",),
    ),
    (
        PipelineJobResult,
        {
            "job_id": "job-1",
            "target_field": "ZTF_0412",
            "matched_source_count": 3,
            "brightest_mag_g": 14.2,
        },
        ("brightest_mag_g",),
    ),
    (
        CNNClassifierConfig,
        {"dropout_rate": 0.25, "decision_threshold": 0.5},
        ("dropout_rate", "decision_threshold"),
    ),
]

# PipelineJobRequest holds no float fields directly (they live in nested models that already
# enforce the policy) but must still declare the policy so future float fields inherit it.
from LUMIA.data_pipeline.orchestration import PipelineJobRequest  # noqa: E402

_ALL_REMAINING_MODELS: list[type[BaseModel]] = [
    *(case[0] for case in _MODEL_CASES),
    PipelineJobRequest,
]


def test_ten_remaining_models_are_covered() -> None:
    """Guard: the 10 models flagged by the P1 audit are all exercised here."""
    assert len(set(_ALL_REMAINING_MODELS)) == 10


@pytest.mark.parametrize("model_cls", _ALL_REMAINING_MODELS, ids=lambda c: c.__name__)
def test_model_config_declares_strict_forbid_frozen_and_finite(
    model_cls: type[BaseModel],
) -> None:
    """Each model must carry the full ConfigDict(strict, extra=forbid, frozen, allow_inf_nan=False)."""
    cfg = model_cls.model_config
    assert cfg.get("strict") is True
    assert cfg.get("extra") == "forbid"
    assert cfg.get("frozen") is True
    assert cfg.get("allow_inf_nan") is False


@pytest.mark.parametrize("bad_value", [_NAN, _INF, _NEG_INF], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize(
    ("model_cls", "valid_kwargs", "float_fields"),
    _MODEL_CASES,
    ids=[case[0].__name__ for case in _MODEL_CASES],
)
def test_models_reject_non_finite_floats_with_validation_error(
    model_cls: type[BaseModel],
    valid_kwargs: dict[str, Any],
    float_fields: tuple[str, ...],
    bad_value: float,
) -> None:
    """Valid payloads construct cleanly; any NaN/Inf float field raises ValidationError."""
    model_cls(**valid_kwargs)

    for field_name in float_fields:
        payload = {**valid_kwargs, field_name: bad_value}
        with pytest.raises(ValidationError):
            model_cls(**payload)
