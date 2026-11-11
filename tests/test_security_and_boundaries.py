"""Security, fuzzing, and architectural boundary test suite for Project LUMIA."""

from __future__ import annotations

import ast
import os
import pickle
from typing import TYPE_CHECKING
from urllib.parse import quote

import pytest
import torch
import yaml
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import ValidationError

from LUMIA.api.main import create_app
from LUMIA.core.schemas import CandidateDetection
from LUMIA.data_pipeline.database import CatalogDatabase, DuckDBConfig
from LUMIA.data_pipeline.repositories import (
    CandidateDetectionRepository,
    CandidateListQuery,
)
from LUMIA.ml.classifier import RealBogusClassifier
from LUMIA.ml.config import CNNClassifierConfig
from LUMIA.physics.config import TelescopeConfig

if TYPE_CHECKING:
    from pathlib import Path


_REPO_ROOT: Path = (
    # tests/test_security_and_boundaries.py -> repository root
    __import__("pathlib").Path(__file__).resolve().parents[1]
)


class _MaliciousOsSystemPayload:
    """Pickle gadget attempting arbitrary command execution via os.system."""

    def __init__(self, marker_path: str) -> None:
        self.marker_path: str = marker_path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, (f'echo pwned > "{self.marker_path}"',))


class _ArbitraryCustomClassPayload:
    """Custom non-tensor class instance that must be rejected by weights_only=True."""

    def __init__(self) -> None:
        self.secret: str = "unauthorized_global"


def _make_sample_candidate(
    candidate_id: str = "cand-sec-001",
    field_id: str = "ZTF_SEC_01",
) -> CandidateDetection:
    """Construct a valid canonical CandidateDetection record for security tests."""
    return CandidateDetection(
        candidate_id=candidate_id,
        field_id=field_id,
        ra_deg=180.25,
        dec_deg=-12.5,
        bjd_tdb=2460500.5,
        mag=21.1,
        snr=8.4,
        cnn_score=0.92,
        model_version="lumia-rb-0.1.0",
        astrometric_uncertainty_arcsec=0.12,
        astrometric_residual_arcsec=0.09,
    )


def _api_headers(client: TestClient) -> dict[str, str]:
    auth = client.app.state.lumia_state.auth
    auth.register_operator(
        username="security-reviewer",
        email="security-reviewer@observatory.test",
        password="correct-horse-battery",
        role="reviewer",
    )
    response = client.post(
        "/api/v1/auth/login",
        json={
            "username": "security-reviewer",
            "password": "correct-horse-battery",
        },
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


# ---------------------------------------------------------------------------
# 1. PyTorch Pickle RCE Prevention
# ---------------------------------------------------------------------------


class TestPyTorchPickleRCEPrevention:
    """Verify RealBogusClassifier blocks pickle RCE gadgets and unsafe checkpoints."""

    def test_classifier_enforces_weights_only_in_torch_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify RealBogusClassifier passes weights_only=True to torch.load."""
        base_cfg = CNNClassifierConfig(backbone="vgg", stamp_size=51, device="cpu")
        base_clf = RealBogusClassifier(base_cfg)
        ckpt_path: Path = tmp_path / "valid_weights.pt"
        torch.save(base_clf.model.state_dict(), ckpt_path)

        recorded_kwargs: dict[str, object] = {}
        real_torch_load = torch.load

        def _spy_torch_load(*args: object, **kwargs: object) -> object:
            recorded_kwargs.update(kwargs)
            return real_torch_load(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(torch, "load", _spy_torch_load)

        load_cfg = CNNClassifierConfig(
            backbone="vgg",
            stamp_size=51,
            device="cpu",
            checkpoint_path=ckpt_path,
        )
        RealBogusClassifier(load_cfg)

        assert recorded_kwargs.get("weights_only") is True

    def test_classifier_rejects_os_system_pickle_rce_payload(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify a checkpoint containing an os.system __reduce__ gadget never executes."""
        marker_file: Path = tmp_path / "rce_executed.txt"
        malicious_ckpt: Path = tmp_path / "malicious_os_system.pt"
        with malicious_ckpt.open("wb") as handle:
            pickle.dump(
                {"state": _MaliciousOsSystemPayload(str(marker_file))},
                handle,
                protocol=2,
            )

        cfg = CNNClassifierConfig(
            backbone="vgg",
            stamp_size=51,
            device="cpu",
            checkpoint_path=malicious_ckpt,
        )
        with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError)):
            RealBogusClassifier(cfg)

        assert not marker_file.exists(), "Pickle RCE gadget executed os.system!"

    def test_classifier_rejects_custom_class_and_non_mapping_checkpoints(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify custom classes and non-mapping primitive checkpoints are rejected cleanly."""
        custom_ckpt: Path = tmp_path / "custom_class.pt"
        torch.save(_ArbitraryCustomClassPayload(), custom_ckpt)

        cfg_custom = CNNClassifierConfig(
            backbone="resnet_lite",
            stamp_size=51,
            device="cpu",
            checkpoint_path=custom_ckpt,
        )
        with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError)):
            RealBogusClassifier(cfg_custom)

        non_mapping_ckpt: Path = tmp_path / "non_mapping_list.pt"
        torch.save([torch.zeros((3, 3), dtype=torch.float32)], non_mapping_ckpt)

        cfg_list = CNNClassifierConfig(
            backbone="resnet_lite",
            stamp_size=51,
            device="cpu",
            checkpoint_path=non_mapping_ckpt,
        )
        with pytest.raises(ValueError, match="mapping|state_dict"):
            RealBogusClassifier(cfg_list)


# ---------------------------------------------------------------------------
# 2. SQL Injection & Path Parameter Validation
# ---------------------------------------------------------------------------


class TestSQLInjectionAndPathValidation:
    """Verify repository SQL parameterization and FastAPI strict Path validation."""

    @pytest.mark.parametrize(
        "sqli_payload",
        [
            "' OR '1'='1'; DROP TABLE candidate_detections; --",
            "cand-sec-001' UNION SELECT * FROM candidate_detections --",
            "'; DELETE FROM candidate_detections WHERE '1'='1",
            "ZTF_SEC_01' OR 1=1 --",
        ],
    )
    def test_repository_treats_sql_injection_strings_as_inert_literals(
        self,
        sqli_payload: str,
    ) -> None:
        """Verify get_by_id, list_candidates, and insert_many neutralize SQL injection."""
        with CatalogDatabase(DuckDBConfig(database=":memory:")) as db:
            repo = CandidateDetectionRepository(db)
            repo.ensure_table()
            baseline = _make_sample_candidate()
            repo.insert_many([baseline])

            # 1. Lookup with SQLi payload must return None and never drop or leak rows
            assert repo.get_by_id(sqli_payload) is None

            # 2. Field filter with SQLi payload must return zero matches, not all rows
            filtered = repo.list_candidates(
                CandidateListQuery(min_cnn_score=0.0, field_id=sqli_payload)
            )
            assert filtered == []

            # 3. Inserting a record whose string fields contain SQLi text stores it verbatim
            literal_record = _make_sample_candidate(
                candidate_id=sqli_payload,
                field_id=sqli_payload,
            )
            repo.insert_many([literal_record])

            fetched = repo.get_by_id(sqli_payload)
            assert fetched is not None
            assert fetched.candidate_id == sqli_payload
            assert fetched.field_id == sqli_payload

            # 4. Table remains intact with both records preserved
            summary = repo.summarize(min_cnn_score=0.8)
            assert summary.total_candidates == 2

    def test_repository_rejects_invalid_table_identifier_injection(self) -> None:
        """Verify table_name DDL identifier validation blocks SQL injection."""
        with (
            CatalogDatabase(DuckDBConfig(database=":memory:")) as db,
            pytest.raises(ValueError, match="Invalid SQL identifier"),
        ):
            CandidateDetectionRepository(
                db,
                table_name="candidate_detections; DROP TABLE stellar_catalog; --",
            )

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "' OR '1'='1'; DROP TABLE candidate_detections; --",
            "..\\..\\etc\\passwd",
            "..\\..\\windows\\win.ini",
            "cand;DROP_TABLE",
            "cand$001",
            "cand with spaces",
            "<img src=x onerror=alert(1)>",
            "a" * 129,
        ],
    )
    def test_fastapi_path_parameters_reject_injection_and_traversal_with_422(
        self,
        malicious_id: str,
    ) -> None:
        """Verify /candidates/{candidate_id}, /cutouts, and /jobs/{job_id} enforce Path regex."""
        app = create_app(DuckDBConfig(database=":memory:"))
        encoded_id: str = quote(malicious_id, safe="")

        with TestClient(app) as client:
            headers = _api_headers(client)
            resp_cand = client.get(f"/api/v1/candidates/{encoded_id}", headers=headers)
            assert resp_cand.status_code == 422, (
                f"Expected 422 for /candidates/{{id}} with {malicious_id!r}, "
                f"got {resp_cand.status_code}"
            )

            resp_cutouts = client.get(
                f"/api/v1/candidates/{encoded_id}/cutouts", headers=headers
            )
            assert resp_cutouts.status_code == 422, (
                f"Expected 422 for /candidates/{{id}}/cutouts with {malicious_id!r}, "
                f"got {resp_cutouts.status_code}"
            )

            resp_job = client.get(f"/api/v1/jobs/{encoded_id}", headers=headers)
            assert resp_job.status_code == 422, (
                f"Expected 422 for /jobs/{{id}} with {malicious_id!r}, "
                f"got {resp_job.status_code}"
            )

    @pytest.mark.parametrize(
        "slash_payload",
        [
            "../../etc/passwd",
            "../secret/config.yaml",
            "<script>alert(1)</script>",
        ],
    )
    def test_fastapi_rejects_encoded_slash_traversal_payloads(
        self,
        slash_payload: str,
    ) -> None:
        """Verify URL-encoded forward-slash traversal payloads never reach data handlers."""
        app = create_app(DuckDBConfig(database=":memory:"))
        encoded_id: str = quote(slash_payload, safe="")

        with TestClient(app) as client:
            headers = _api_headers(client)
            assert (
                client.get(f"/api/v1/candidates/{encoded_id}", headers=headers).status_code
                in {404, 422}
            )
            assert client.get(
                f"/api/v1/jobs/{encoded_id}", headers=headers
            ).status_code in {404, 422}

    def test_fastapi_candidate_detail_and_openapi_path_schema_constraints(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify valid IDs and exact OpenAPI path constraints."""
        db_file: Path = tmp_path / "sec_api.duckdb"
        cfg = DuckDBConfig(database=db_file)
        with CatalogDatabase(cfg) as db:
            repo = CandidateDetectionRepository(db)
            repo.ensure_table()
            repo.insert_many([_make_sample_candidate("cand_valid.01-A")])

        app = create_app(cfg)
        with TestClient(app) as client:
            headers = _api_headers(client)
            ok_resp = client.get("/api/v1/candidates/cand_valid.01-A", headers=headers)
            assert ok_resp.status_code == 200
            assert ok_resp.json()["candidate_id"] == "cand_valid.01-A"

            missing_resp = client.get(
                "/api/v1/candidates/cand_missing.02-B", headers=headers
            )
            assert missing_resp.status_code == 404

            schema = app.openapi()
            for route_path, param_name in [
                ("/api/v1/candidates/{candidate_id}", "candidate_id"),
                ("/api/v1/candidates/{candidate_id}/cutouts", "candidate_id"),
                ("/api/v1/jobs/{job_id}", "job_id"),
            ]:
                params = schema["paths"][route_path]["get"]["parameters"]
                path_param = next(p for p in params if p["name"] == param_name)
                param_schema = path_param["schema"]
                assert param_schema["minLength"] == 1
                assert param_schema["maxLength"] == 128
                assert param_schema["pattern"] == r"^[A-Za-z0-9_.-]{1,128}$"


# ---------------------------------------------------------------------------
# 3. YAML Deserialization Safety & Malformed Input Handling
# ---------------------------------------------------------------------------


class TestYAMLDeserializationSafety:
    """Verify TelescopeConfig.from_yaml blocks RCE tags and normalizes YAML errors."""

    def test_from_yaml_rejects_python_object_apply_rce_tag_with_domain_value_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify !!python/object/apply is never executed and raises ValueError (not YAMLError)."""
        marker_file: Path = tmp_path / "yaml_rce_pwned.txt"
        yaml_file: Path = tmp_path / "rce_telescope.yaml"
        yaml_file.write_text(
            f'!!python/object/apply:os.system ["echo rce > {marker_file.as_posix()}"]\n',
            encoding="utf-8",
        )

        with pytest.raises(ValueError) as exc_info:
            TelescopeConfig.from_yaml(yaml_file)

        assert not isinstance(exc_info.value, yaml.YAMLError)
        assert not marker_file.exists(), "YAML RCE tag executed os.system!"

    @pytest.mark.parametrize(
        "malformed_yaml",
        [
            "name: [unclosed_bracket\npixel_scale: 1.0\n",
            "name: ZTF\n  bad_indent: : :\n\tinvalid_tab: [}\n",
            ": ::: invalid_flow: }\n",
        ],
    )
    def test_from_yaml_raises_domain_value_error_on_malformed_syntax(
        self,
        tmp_path: Path,
        malformed_yaml: str,
    ) -> None:
        """Verify broken YAML syntax raises a domain ValueError rather than raw yaml.YAMLError."""
        yaml_file: Path = tmp_path / "malformed.yaml"
        yaml_file.write_text(malformed_yaml, encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TelescopeConfig.from_yaml(yaml_file)

        assert not isinstance(exc_info.value, yaml.YAMLError)
        assert not isinstance(exc_info.value, ValidationError)

    @pytest.mark.parametrize(
        "non_mapping_yaml",
        [
            "- item_one\n- item_two\n",
            '"just_a_scalar_string"\n',
            "42\n",
            "true\n",
            "\n",
        ],
    )
    def test_from_yaml_raises_domain_value_error_on_non_mapping_roots(
        self,
        tmp_path: Path,
        non_mapping_yaml: str,
    ) -> None:
        """Verify YAML lists, scalars, and empty documents raise a clean domain ValueError."""
        yaml_file: Path = tmp_path / "non_mapping.yaml"
        yaml_file.write_text(non_mapping_yaml, encoding="utf-8")

        with pytest.raises(ValueError, match="mapping") as exc_info:
            TelescopeConfig.from_yaml(yaml_file)

        assert not isinstance(exc_info.value, yaml.YAMLError)
        assert not isinstance(exc_info.value, TypeError)
        assert not isinstance(exc_info.value, ValidationError)


# ---------------------------------------------------------------------------
# 4. API DoS Bounds, NaN/Inf Smuggling & CORS Hardening
# ---------------------------------------------------------------------------


class TestAPIDoSNaNSmugglingAndCORSHardening:
    """Verify pagination DoS caps, non-finite float rejection, and strict CORS settings."""

    @pytest.mark.parametrize(
        "query_string",
        [
            "limit=0",
            "limit=-1",
            "limit=501",
            "limit=1000",
            "limit=100000",
            "offset=-1",
            "offset=-50",
            "min_cnn_score=NaN",
            "min_cnn_score=nan",
            "min_cnn_score=Infinity",
            "min_cnn_score=inf",
            "min_cnn_score=-Infinity",
            "min_cnn_score=-inf",
        ],
    )
    def test_candidates_endpoint_rejects_dos_pagination_and_nan_inf_smuggling(
        self,
        query_string: str,
    ) -> None:
        """Verify GET /api/v1/candidates rejects out-of-range pagination and NaN/Inf with 422."""
        app = create_app(DuckDBConfig(database=":memory:"))
        with TestClient(app) as client:
            response = client.get(
                f"/api/v1/candidates?{query_string}", headers=_api_headers(client)
            )
            assert response.status_code == 422, (
                f"Expected HTTP 422 for query '{query_string}', got {response.status_code}"
            )

    @pytest.mark.parametrize("bad_score", ["NaN", "Infinity", "-Infinity"])
    def test_detection_summary_rejects_nan_and_infinity_query_params(
        self,
        bad_score: str,
    ) -> None:
        """Verify GET /api/v1/detections/summary also rejects NaN/Infinity query filters."""
        app = create_app(DuckDBConfig(database=":memory:"))
        with TestClient(app) as client:
            response = client.get(
                f"/api/v1/detections/summary?min_cnn_score={bad_score}",
                headers=_api_headers(client),
            )
            assert response.status_code == 422

    def test_candidates_endpoint_allows_boundary_limit_500(self) -> None:
        """Verify valid upper boundary limit=500 and offset=0 succeed with HTTP 200."""
        app = create_app(DuckDBConfig(database=":memory:"))
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/candidates?limit=500&offset=0",
                headers=_api_headers(client),
            )
            assert response.status_code == 200

    def test_cors_middleware_never_pairs_credentials_with_wildcards(self) -> None:
        """Verify CORSMiddleware forbids wildcard origins, methods, and headers with credentials."""
        app = create_app(DuckDBConfig(database=":memory:"))
        cors_middlewares = [
            mw for mw in app.user_middleware if mw.cls is CORSMiddleware
        ]
        assert len(cors_middlewares) == 1
        cors_kwargs = cors_middlewares[0].kwargs

        if cors_kwargs.get("allow_credentials") is True:
            allow_origins = list(cors_kwargs.get("allow_origins", ()))
            allow_methods = list(cors_kwargs.get("allow_methods", ()))
            allow_headers = list(cors_kwargs.get("allow_headers", ()))

            assert len(allow_origins) > 0
            assert "*" not in allow_origins, (
                "CORSMiddleware must not pair allow_credentials=True with '*' origin."
            )
            assert "*" not in allow_methods, (
                "CORSMiddleware must not pair allow_credentials=True with '*' methods."
            )
            assert "*" not in allow_headers, (
                "CORSMiddleware must not pair allow_credentials=True with '*' headers."
            )

        with pytest.raises(ValueError, match="wildcard|origin|\\*"):
            create_app(
                DuckDBConfig(database=":memory:"),
                allowed_origins=["*"],
            )


# ---------------------------------------------------------------------------
# 5. AST Architectural Firewall
# ---------------------------------------------------------------------------


def _resolve_relative_import(
    file_path: Path,
    src_root: Path,
    level: int,
    module: str | None,
) -> str:
    """Resolve a relative ImportFrom node to its fully qualified package path."""
    rel_parts = list(file_path.relative_to(src_root).with_suffix("").parts)
    package_parts = rel_parts[:-1]
    ascend = level - 1
    base_parts = package_parts[: max(len(package_parts) - ascend, 0)]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def _matches_forbidden_target(imported_name: str, forbidden_prefixes: tuple[str, ...]) -> bool:
    """Return True if imported_name equals or is a submodule of any forbidden prefix."""
    return any(
        imported_name == prefix or imported_name.startswith(f"{prefix}.")
        for prefix in forbidden_prefixes
    )


class TestASTArchitecturalFirewall:
    """Enforce zero persistence, orchestration, or API imports in physics and ml layers."""

    FORBIDDEN_PREFIXES: tuple[str, ...] = (
        "duckdb",
        "sqlalchemy",
        "fastapi",
        "prefect",
        "LUMIA.api",
        "LUMIA.data_pipeline",
    )

    @pytest.mark.parametrize("subpackage", ["physics", "ml"])
    def test_domain_packages_contain_zero_forbidden_imports(
        self,
        subpackage: str,
    ) -> None:
        """Parse every .py file under src/LUMIA/{physics,ml} and forbid infrastructure imports."""
        src_root: Path = _REPO_ROOT / "src"
        pkg_dir: Path = src_root / "LUMIA" / subpackage
        assert pkg_dir.is_dir(), f"Expected package directory {pkg_dir} to exist."

        py_files: list[Path] = sorted(pkg_dir.rglob("*.py"))
        assert len(py_files) > 0, f"Expected Python modules inside {pkg_dir}."

        violations: list[str] = []
        for py_file in py_files:
            source_text: str = py_file.read_text(encoding="utf-8")
            tree: ast.Module = ast.parse(source_text, filename=str(py_file))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _matches_forbidden_target(alias.name, self.FORBIDDEN_PREFIXES):
                            violations.append(
                                f"{py_file.relative_to(_REPO_ROOT)}:{node.lineno} "
                                f"-> import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        resolved_mod = _resolve_relative_import(
                            py_file, src_root, node.level, node.module
                        )
                    else:
                        resolved_mod = node.module or ""

                    if _matches_forbidden_target(resolved_mod, self.FORBIDDEN_PREFIXES):
                        violations.append(
                            f"{py_file.relative_to(_REPO_ROOT)}:{node.lineno} "
                            f"-> from {resolved_mod} import ..."
                        )
                    for alias in node.names:
                        candidate_full = (
                            f"{resolved_mod}.{alias.name}" if resolved_mod else alias.name
                        )
                        if _matches_forbidden_target(
                            candidate_full, self.FORBIDDEN_PREFIXES
                        ):
                            violations.append(
                                f"{py_file.relative_to(_REPO_ROOT)}:{node.lineno} "
                                f"-> from {resolved_mod} import {alias.name}"
                            )

        assert violations == [], (
            f"Architectural firewall violated in LUMIA.{subpackage}:\n"
            + "\n".join(violations)
        )
