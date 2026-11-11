"""Tests for the canonical shared schemas and the project-wide Pydantic model policy (P0-2, P0-3)."""

from __future__ import annotations

import dataclasses
import inspect
from types import ModuleType

import pytest
from pydantic import BaseModel, ValidationError

import LUMIA.api.schemas as api_schemas
import LUMIA.physics.gpu_processing as gpu_processing
import LUMIA.physics.linking as linking
from LUMIA.core.schemas import CandidateDetection


def _valid_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "candidate_id": "cand-001",
        "ra_deg": 180.125,
        "dec_deg": 15.34,
        "bjd_tdb": 2460755.75658,
    }
    base.update(overrides)
    return base


class TestCanonicalCandidateDetection:
    """A single authoritative CandidateDetection shared by physics, ML and API layers."""

    def test_single_definition_is_shared_across_modules(self) -> None:
        assert linking.CandidateDetection is CandidateDetection
        assert api_schemas.CandidateDetection is CandidateDetection
        assert CandidateDetection.__module__ == "LUMIA.core.schemas"

    def test_minimal_detection_uses_documented_defaults(self) -> None:
        det = CandidateDetection(**_valid_kwargs())  # type: ignore[arg-type]
        assert det.field_id == "default"
        assert det.mag is None
        assert det.snr is None
        assert det.cnn_score is None
        assert det.model_version is None
        assert det.astrometric_uncertainty_arcsec == pytest.approx(0.15)
        assert det.astrometric_residual_arcsec is None

    def test_full_detection_with_classification_provenance(self) -> None:
        det = CandidateDetection(
            **_valid_kwargs(
                field_id="ZTF_0412",
                mag=20.4,
                snr=12.5,
                cnn_score=0.96,
                model_version="rb-0.1",
                astrometric_uncertainty_arcsec=0.08,
                astrometric_residual_arcsec=0.11,
            ),  # type: ignore[arg-type]
        )
        assert det.cnn_score == pytest.approx(0.96)
        assert det.model_version == "rb-0.1"

    def test_measurement_uncertainty_and_fit_residual_are_distinct_fields(self) -> None:
        det = CandidateDetection(
            **_valid_kwargs(
                astrometric_uncertainty_arcsec=0.05,
                astrometric_residual_arcsec=0.30,
            ),  # type: ignore[arg-type]
        )
        assert det.astrometric_uncertainty_arcsec == pytest.approx(0.05)
        assert det.astrometric_residual_arcsec == pytest.approx(0.30)

    @pytest.mark.parametrize(
        "field",
        [
            "ra_deg",
            "dec_deg",
            "bjd_tdb",
            "mag",
            "snr",
            "cnn_score",
            "astrometric_uncertainty_arcsec",
            "astrometric_residual_arcsec",
        ],
    )
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_values_in_every_float_field(
        self,
        field: str,
        bad: float,
    ) -> None:
        with pytest.raises(ValidationError):
            CandidateDetection(**_valid_kwargs(**{field: bad}))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"ra_deg": 360.0},
            {"ra_deg": -0.1},
            {"dec_deg": 90.5},
            {"snr": -1.0},
            {"cnn_score": 1.01},
            {"cnn_score": -0.01},
            {"astrometric_uncertainty_arcsec": -0.1},
            {"astrometric_residual_arcsec": -0.1},
            {"candidate_id": ""},
            {"field_id": ""},
            {"model_version": ""},
            {"ra_deg": "180.0"},  # strict mode forbids str -> float coercion
        ],
    )
    def test_rejects_out_of_range_or_wrongly_typed_values(
        self,
        bad_kwargs: dict[str, object],
    ) -> None:
        with pytest.raises(ValidationError):
            CandidateDetection(**_valid_kwargs(**bad_kwargs))  # type: ignore[arg-type]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            CandidateDetection(**_valid_kwargs(unexpected=1))  # type: ignore[arg-type]

    def test_model_version_requires_a_classification_score(self) -> None:
        with pytest.raises(ValidationError, match="cnn_score"):
            CandidateDetection(**_valid_kwargs(model_version="rb-0.1"))  # type: ignore[arg-type]

    def test_is_immutable(self) -> None:
        det = CandidateDetection(**_valid_kwargs())  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            det.ra_deg = 10.0  # type: ignore[misc]


def _models_defined_in(module: ModuleType) -> list[type[BaseModel]]:
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, BaseModel)
        and cls is not BaseModel
        and cls.__module__ == module.__name__
    ]


_POLICY_MODULES: list[ModuleType] = [
    linking,
    api_schemas,
    gpu_processing,
    __import__("LUMIA.core.schemas", fromlist=["CandidateDetection"]),
]


class TestPydanticModelPolicy:
    """Every model in the refactored files enforces strict / forbid / no-NaN-Inf."""

    @pytest.mark.parametrize("module", _POLICY_MODULES, ids=lambda m: m.__name__)
    def test_all_models_enforce_strict_forbid_and_finite(
        self,
        module: ModuleType,
    ) -> None:
        models = _models_defined_in(module)
        assert models, f"No Pydantic models discovered in {module.__name__}."
        for model in models:
            cfg = model.model_config
            assert cfg.get("strict") is True, f"{model.__name__}: strict"
            assert cfg.get("extra") == "forbid", f"{model.__name__}: extra"
            assert cfg.get("allow_inf_nan") is False, f"{model.__name__}: allow_inf_nan"
            assert cfg.get("frozen") is True, f"{model.__name__}: frozen"

    def test_gpu_results_are_pydantic_models_not_dataclasses(self) -> None:
        for name in ("ShiftAndStackResult", "PsiPhiStackResult", "ZOGYResult"):
            cls = getattr(gpu_processing, name)
            assert issubclass(cls, BaseModel), name
            assert not dataclasses.is_dataclass(cls), name
