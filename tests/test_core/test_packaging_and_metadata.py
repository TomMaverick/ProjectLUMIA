"""Tests for explicit package initializers, PEP 561 py.typed marker, and pyproject metadata."""

from __future__ import annotations

import ast
import importlib
import re
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_SRC_ROOT: Path = _REPO_ROOT / "src"
_LUMIA_ROOT: Path = _SRC_ROOT / "LUMIA"

_EXPECTED_PACKAGES: tuple[tuple[str, Path], ...] = (
    ("LUMIA", _LUMIA_ROOT / "__init__.py"),
    ("LUMIA.core", _LUMIA_ROOT / "core" / "__init__.py"),
    ("LUMIA.physics", _LUMIA_ROOT / "physics" / "__init__.py"),
    ("LUMIA.ml", _LUMIA_ROOT / "ml" / "__init__.py"),
    ("LUMIA.data_pipeline", _LUMIA_ROOT / "data_pipeline" / "__init__.py"),
    ("LUMIA.api", _LUMIA_ROOT / "api" / "__init__.py"),
)

_EXPECTED_EXPORTS: dict[str, set[str]] = {
    "LUMIA": {
        "CandidateDetection",
        "__version__",
        "api",
        "core",
        "data_pipeline",
        "ml",
        "physics",
    },
    "LUMIA.core": {
        "CandidateDetection",
        "DEVICE_SPEC_PATTERN",
        "configure_deterministic_torch",
        "resolve_torch_device",
    },
    "LUMIA.physics": {
        "AperturePhotometryMeasurement",
        "AstrometricCalibrationConfig",
        "AstrometricMeasurement",
        "BarycentricTimeResult",
        "BoundaryMode",
        "CutoutBatchResult",
        "CutoutConfig",
        "CutoutStampSize",
        "DifferenceChannelSource",
        "GPUProcessingConfig",
        "KinematicValidationResult",
        "ObservatoryLocation",
        "PeakExtractionConfig",
        "PhotometryConfig",
        "PsiPhiStackResult",
        "SNRPeakCandidate",
        "ShiftAndStackResult",
        "SpaceVelocityState",
        "TelescopeConfig",
        "Tracklet",
        "TrackletLinkingConfig",
        "VRAMLimitExceededError",
        "ZOGYConfig",
        "ZOGYResult",
        "ZeroPointResult",
        "cluster_velocity_states",
        "compute_bjd_tdb",
        "compute_psi_phi_stack",
        "compute_zero_point",
        "extract_celestial_wcs",
        "extract_snr_peaks",
        "extract_zogy_cutout_batch",
        "link_tracklets",
        "load_telescope_config",
        "perform_aperture_photometry",
        "pixel_to_sky",
        "project_detections_to_velocity_space",
        "propagate_astrometric_uncertainty",
        "shift_and_stack",
        "sky_to_pixel",
        "slice_cutout_triplets",
        "validate_tracklet_kinematics",
        "zogy_subtract",
    },
    "LUMIA.ml": {
        "BackboneType",
        "CNNClassifierConfig",
        "ClassificationResult",
        "INPUT_CHANNELS",
        "InferenceMemoryError",
        "RealBogusClassifier",
        "RealBogusNet",
        "ResNetLiteClassifier",
        "StampSize",
        "VGGClassifier",
        "build_model",
        "configure_deterministic_torch",
        "preprocess_triplets",
        "resolve_torch_device",
    },
    "LUMIA.data_pipeline": {
        "CandidateCutoutRepository",
        "CandidateCutoutStampRecord",
        "CandidateDetectionRepository",
        "CandidateDetectionSummary",
        "CandidateListQuery",
        "CatalogDatabase",
        "CatalogQueryParams",
        "CatalogQuerySummary",
        "CutoutStampSize",
        "DuckDBConfig",
        "PipelineJobRequest",
        "PipelineJobResult",
        "StellarCatalogRecord",
        "log_task_failure",
        "query_catalog_sources_task",
        "run_catalog_pipeline_flow",
        "validate_job_request_task",
    },
    "LUMIA.api": {
        "APIState",
        "CandidateCutoutEpoch",
        "CandidateCutoutSequenceResponse",
        "CandidateListResponse",
        "CatalogQueryResponse",
        "CatalogStarItem",
        "CutoutStampSize",
        "DetectionSummaryResponse",
        "JobStatusResponse",
        "JobSubmissionRequest",
        "PipelineStatusResponse",
        "SafeIdentifierPath",
        "StateDep",
        "create_app",
        "get_api_state",
        "router",
    },
}


def _extract_pkg_name(requirement: str) -> str:
    """Normalize a PEP 508 dependency specifier down to its lowercase package name."""
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        return requirement.strip().lower()
    return match.group(1).lower().replace("_", "-")


class TestPackageInitializersAndPyTyped:
    """Verify explicit __init__.py initializers, curated __all__, and PEP 561 py.typed."""

    @pytest.mark.parametrize(
        ("pkg_name", "init_path"),
        _EXPECTED_PACKAGES,
        ids=[item[0] for item in _EXPECTED_PACKAGES],
    )
    def test_explicit_init_file_exists_and_declares_future_annotations(
        self,
        pkg_name: str,
        init_path: Path,
    ) -> None:
        """Every LUMIA package must have an explicit __init__.py with deferred annotations."""
        assert init_path.is_file(), f"Missing explicit initializer for {pkg_name}: {init_path}"
        source = init_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(init_path))
        has_future_annotations = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        assert has_future_annotations, (
            f"{init_path} must declare 'from __future__ import annotations'."
        )

    @pytest.mark.parametrize(
        ("pkg_name", "init_path"),
        _EXPECTED_PACKAGES,
        ids=[item[0] for item in _EXPECTED_PACKAGES],
    )
    def test_package_exposes_curated_dunder_all_and_resolves_all_exports(
        self,
        pkg_name: str,
        init_path: Path,
    ) -> None:
        """Each package must be non-namespace, define curated __all__, and resolve every export."""
        assert init_path.is_file(), f"Missing {init_path}"
        module = importlib.import_module(pkg_name)
        assert module.__file__ is not None
        assert Path(module.__file__).resolve() == init_path.resolve()

        assert hasattr(module, "__all__"), f"{pkg_name} must define __all__."
        exported = module.__all__
        assert isinstance(exported, (list, tuple))
        assert len(exported) > 0
        assert len(exported) == len(set(exported)), f"Duplicate entries in {pkg_name}.__all__"

        for symbol in exported:
            assert isinstance(symbol, str)
            if symbol != "__version__":
                assert not symbol.startswith("_"), (
                    f"Private symbol {symbol!r} must not appear in {pkg_name}.__all__."
                )
            assert hasattr(module, symbol), (
                f"{pkg_name}.{symbol} listed in __all__ does not resolve"
            )

        expected_subset = _EXPECTED_EXPORTS[pkg_name]
        missing = expected_subset - set(exported)
        assert not missing, f"{pkg_name}.__all__ is missing expected symbols: {sorted(missing)}"

    def test_data_pipeline_does_not_reexport_internal_schema_modules(self) -> None:
        """Internal schema_oltp and schema_olap modules must not be publicly re-exported."""
        dp_pkg = importlib.import_module("LUMIA.data_pipeline")
        assert "schema_oltp" not in dp_pkg.__all__
        assert "schema_olap" not in dp_pkg.__all__

    def test_pep561_py_typed_marker_exists(self) -> None:
        """Verify PEP 561 py.typed marker file exists at src/LUMIA/py.typed."""
        marker = _LUMIA_ROOT / "py.typed"
        assert marker.is_file(), f"Expected PEP 561 marker at {marker}."


class TestPyprojectToolingMetadata:
    """Verify [build-system], [dependency-groups], [tool.ruff], and [tool.mypy] settings."""

    def _load_pyproject(self) -> dict[str, object]:
        pyproject_path = _REPO_ROOT / "pyproject.toml"
        return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    def test_build_system_configured_with_hatchling_or_flit_core(self) -> None:
        """Verify [build-system] specifies hatchling or flit_core."""
        data = self._load_pyproject()
        assert "build-system" in data, "pyproject.toml must define [build-system]."
        build_sys = data["build-system"]
        assert isinstance(build_sys, dict)

        backend = build_sys.get("build-backend")
        requires = build_sys.get("requires")
        assert backend in {"hatchling.build", "flit_core.buildapi"}
        assert isinstance(requires, list)
        req_names = {_extract_pkg_name(str(item)) for item in requires}
        if backend == "hatchling.build":
            assert "hatchling" in req_names
        else:
            assert "flit-core" in req_names

    def test_dev_dependencies_moved_to_dependency_groups(self) -> None:
        """Verify pytest and httpx reside in [dependency-groups].dev and not runtime deps."""
        data = self._load_pyproject()
        project = data["project"]
        assert isinstance(project, dict)
        runtime_deps = project.get("dependencies", [])
        assert isinstance(runtime_deps, list)
        runtime_names = {_extract_pkg_name(str(dep)) for dep in runtime_deps}

        assert "pytest" not in runtime_names, "pytest must not remain in [project].dependencies."
        assert "httpx" not in runtime_names, "httpx must not remain in [project].dependencies."

        dep_groups = data.get("dependency-groups")
        assert isinstance(dep_groups, dict), "pyproject.toml must define [dependency-groups]."
        dev_group = dep_groups.get("dev")
        assert isinstance(dev_group, list), "[dependency-groups] must define a 'dev' list."
        dev_names = {_extract_pkg_name(str(dep)) for dep in dev_group}
        assert {"pytest", "httpx"}.issubset(dev_names)

    def test_ruff_strict_rules_configured(self) -> None:
        """Verify [tool.ruff] and [tool.ruff.lint] select E, F, W, B, SIM, UP, I, TC."""
        data = self._load_pyproject()
        tool = data.get("tool")
        assert isinstance(tool, dict)

        ruff_cfg = tool.get("ruff")
        assert isinstance(ruff_cfg, dict), "pyproject.toml must define [tool.ruff]."
        lint_cfg = ruff_cfg.get("lint")
        assert isinstance(lint_cfg, dict), "pyproject.toml must define [tool.ruff.lint]."

        selected = set(lint_cfg.get("select", [])) | set(lint_cfg.get("extend-select", []))
        required_rules = {"E", "F", "W", "B", "SIM", "UP", "I", "TC"}
        assert required_rules.issubset(selected), (
            f"Missing required Ruff lint rules: {sorted(required_rules - selected)}"
        )

    def test_mypy_strict_and_pydantic_plugin_configured(self) -> None:
        """Verify [tool.mypy] enables strict = true and plugins = ['pydantic.mypy']."""
        data = self._load_pyproject()
        tool = data.get("tool")
        assert isinstance(tool, dict)

        mypy_cfg = tool.get("mypy")
        assert isinstance(mypy_cfg, dict), "pyproject.toml must define [tool.mypy]."
        assert mypy_cfg.get("strict") is True
        plugins = mypy_cfg.get("plugins")
        assert isinstance(plugins, list)
        assert "pydantic.mypy" in plugins
