"""Unit tests for Module 10 FastAPI Web-UI endpoints, DuckDB queries, and schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from LUMIA.api.main import create_app
from LUMIA.api.schemas import (
    CandidateCutoutSequenceResponse,
    CandidateDetection,
    CandidateListResponse,
    CatalogQueryResponse,
    DetectionSummaryResponse,
    JobStatusResponse,
    JobSubmissionRequest,
    PipelineStatusResponse,
)
from LUMIA.data_pipeline.database import (
    CatalogDatabase,
    CatalogQueryParams,
    DuckDBConfig,
    StellarCatalogRecord,
)
from LUMIA.data_pipeline.repositories import CandidateDetectionRepository

if TYPE_CHECKING:
    from pathlib import Path


def _seed_candidates() -> list[CandidateDetection]:
    """Build deterministic asteroid candidate detections for DuckDB seeding."""
    return [
        CandidateDetection(
            candidate_id="cand-001",
            field_id="ZTF_0412",
            ra_deg=180.125,
            dec_deg=15.340,
            bjd_tdb=2460755.75658,
            mag=20.4,
            snr=12.5,
            cnn_score=0.96,
            astrometric_residual_arcsec=0.11,
        ),
        CandidateDetection(
            candidate_id="cand-002",
            field_id="ZTF_0412",
            ra_deg=180.440,
            dec_deg=15.610,
            bjd_tdb=2460755.75662,
            mag=21.2,
            snr=7.5,
            cnn_score=0.84,
            astrometric_residual_arcsec=0.18,
        ),
        CandidateDetection(
            candidate_id="cand-003",
            field_id="ZTF_0412",
            ra_deg=180.890,
            dec_deg=15.120,
            bjd_tdb=2460755.75650,
            mag=21.9,
            snr=4.8,
            cnn_score=0.42,
            astrometric_residual_arcsec=0.35,
        ),
        CandidateDetection(
            candidate_id="cand-004",
            field_id="ZTF_0999",
            ra_deg=210.220,
            dec_deg=-12.450,
            bjd_tdb=2460755.81200,
            mag=19.6,
            snr=15.2,
            cnn_score=0.71,
            astrometric_residual_arcsec=None,
        ),
    ]


def _seed_stars() -> list[StellarCatalogRecord]:
    """Build deterministic stellar catalog records for API query serialization tests."""
    return [
        StellarCatalogRecord(source_id=5001, ra_deg=180.10, dec_deg=15.20, mag_g=14.2),
        StellarCatalogRecord(source_id=5002, ra_deg=180.50, dec_deg=15.50, mag_g=16.1),
        StellarCatalogRecord(source_id=5003, ra_deg=215.00, dec_deg=-10.00, mag_g=13.0),
    ]


def _auth_headers(client: TestClient, role: str = "reviewer") -> dict[str, str]:
    state = client.app.state.lumia_state
    state.auth.register_operator(
        username=f"operator-{role}",
        email=f"{role}@observatory.test",
        password="correct-horse-battery",
        role=role,
    )
    response = client.post(
        "/api/v1/auth/login",
        json={
            "username": f"operator-{role}",
            "password": "correct-horse-battery",
        },
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_pipeline_status_endpoint_starts_and_returns_health() -> None:
    """Verify FastAPI app starts cleanly with in-memory DuckDB and serves pipeline status."""
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")
        assert response.status_code == 200

        status_model: PipelineStatusResponse = PipelineStatusResponse.model_validate(
            response.json()
        )
        assert status_model.status == "operational"
        assert status_model.duckdb_connected is True
        assert status_model.total_jobs == 0
        assert status_model.theme_mode == "dark"
        assert status_model.accent_color_hex in {"#3b185f", "#581c87"}


def test_detection_summary_and_candidate_list_from_duckdb(tmp_path: Path) -> None:
    """Verify detection summary statistics and filtered candidate lists from DuckDB."""
    db_file: Path = tmp_path / "web_ui_candidates.duckdb"
    candidates: list[CandidateDetection] = _seed_candidates()

    with CatalogDatabase(DuckDBConfig(database=db_file, read_only=False)) as db:
        repo = CandidateDetectionRepository(db)
        repo.ensure_table()
        inserted: int = repo.insert_many(candidates)
        assert inserted == 4

    app = create_app(db_config=DuckDBConfig(database=db_file, read_only=True))
    with TestClient(app) as client:
        headers = _auth_headers(client)
        summary_resp = client.get(
            "/api/v1/detections/summary",
            params={"min_cnn_score": 0.80},
            headers=headers,
        )
        assert summary_resp.status_code == 200
        summary: DetectionSummaryResponse = DetectionSummaryResponse.model_validate(
            summary_resp.json()
        )
        assert summary.total_candidates == 4
        assert summary.high_confidence_count == 2
        assert summary.distinct_fields == 2
        assert summary.mean_snr == pytest.approx((12.5 + 7.5 + 4.8 + 15.2) / 4.0)
        assert summary.brightest_mag == pytest.approx(19.6)

        cand_resp = client.get(
            "/api/v1/candidates",
            params={"min_cnn_score": 0.80, "field_id": "ZTF_0412"},
            headers=headers,
        )
        assert cand_resp.status_code == 200
        cand_list: CandidateListResponse = CandidateListResponse.model_validate(
            cand_resp.json()
        )
        assert cand_list.total_returned == 2
        assert [c.candidate_id for c in cand_list.candidates] == ["cand-001", "cand-002"]
        assert cand_list.candidates[0].cnn_score == pytest.approx(0.96)

        unlinked_resp = client.get(
            "/api/v1/candidates",
            params={"field_id": "ZTF_0999"},
            headers=headers,
        )
        assert unlinked_resp.status_code == 200
        unlinked_list = CandidateListResponse.model_validate(unlinked_resp.json())
        assert len(unlinked_list.candidates) == 1
        assert unlinked_list.candidates[0].astrometric_residual_arcsec is None


def test_pydantic_job_submission_and_catalog_query_serialization(tmp_path: Path) -> None:
    """Verify Pydantic request/response serialization for job requests and catalog queries."""
    db_file: Path = tmp_path / "web_ui_jobs.duckdb"
    with CatalogDatabase(DuckDBConfig(database=db_file, read_only=False)) as db:
        db.create_stellar_catalog_table()
        db.insert_stellar_records(_seed_stars())
        CandidateDetectionRepository(db).ensure_table()

    app = create_app(db_config=DuckDBConfig(database=db_file, read_only=True))
    with TestClient(app) as client:
        headers = _auth_headers(client, "astronomer")
        job_req = JobSubmissionRequest(
            job_id="ui-job-01",
            target_field="ZTF_0412",
            query_params=CatalogQueryParams(
                ra_min_deg=180.0,
                ra_max_deg=181.0,
                dec_min_deg=15.0,
                dec_max_deg=16.0,
                max_mag=17.0,
            ),
        )
        post_resp = client.post(
            "/api/v1/jobs",
            json=job_req.model_dump(mode="json"),
            headers=headers,
        )
        assert post_resp.status_code == 201, post_resp.text
        job_status: JobStatusResponse = JobStatusResponse.model_validate_json(
            post_resp.content
        )
        assert job_status.job_id == "ui-job-01"
        assert job_status.target_field == "ZTF_0412"
        assert job_status.matched_catalog_sources == 2
        assert job_status.brightest_catalog_mag_g == pytest.approx(14.2)
        assert job_status.status == "queued"

        get_job_resp = client.get("/api/v1/jobs/ui-job-01", headers=headers)
        assert get_job_resp.status_code == 200
        assert JobStatusResponse.model_validate_json(get_job_resp.content) == job_status

        missing_resp = client.get("/api/v1/jobs/non-existent-job", headers=headers)
        assert missing_resp.status_code == 404

        query_params = CatalogQueryParams(
            ra_min_deg=180.0,
            ra_max_deg=181.0,
            dec_min_deg=15.0,
            dec_max_deg=16.0,
            max_mag=17.0,
        )
        cat_resp = client.post(
            "/api/v1/catalog/query",
            json=query_params.model_dump(mode="json"),
            headers=headers,
        )
        assert cat_resp.status_code == 200
        cat_model: CatalogQueryResponse = CatalogQueryResponse.model_validate(
            cat_resp.json()
        )
        assert cat_model.matched_count == 2
        assert [item.source_id for item in cat_model.items] == [5001, 5002]


def test_api_validation_rejects_invalid_requests() -> None:
    """Verify malformed query parameters and job payloads return HTTP 422."""
    app = create_app()
    with TestClient(app) as client:
        headers = _auth_headers(client, "astronomer")
        bad_summary = client.get(
            "/api/v1/detections/summary",
            params={"min_cnn_score": 1.5},
            headers=headers,
        )
        assert bad_summary.status_code == 422

        bad_job = client.post(
            "/api/v1/jobs",
            json={"job_id": "", "target_field": "ZTF_0412"},
            headers=headers,
        )
        assert bad_job.status_code == 422


def test_candidate_cutout_sequence_and_cors_headers(tmp_path: Path) -> None:
    """Verify multi-epoch extracted 31x31 cutouts for blink-loop UI, 404 when unseeded, and CORS."""
    import numpy as np

    from LUMIA.data_pipeline.repositories import (
        CandidateCutoutRepository,
        CandidateCutoutStampRecord,
    )
    from LUMIA.physics.cutout_extraction import (
        CutoutConfig,
        SNRPeakCandidate,
        slice_cutout_triplets,
    )

    db_file: Path = tmp_path / "web_ui_cutouts.duckdb"
    rng = np.random.default_rng(77)
    ref_frame = rng.normal(100.0, 1.0, size=(64, 64)).astype(np.float32)

    stamp_records: list[CandidateCutoutStampRecord] = []
    for epoch_idx in range(4):
        sci_frame = ref_frame.copy()
        tx = 28 + epoch_idx * 2
        ty = 34 - epoch_idx
        sci_frame[ty, tx] += 45.0
        diff_frame = sci_frame - ref_frame

        batch = slice_cutout_triplets(
            sci_frame,
            ref_frame,
            diff_frame,
            [
                SNRPeakCandidate(
                    candidate_id="cand-001",
                    x_px=tx,
                    y_px=ty,
                    x_centroid_px=float(tx),
                    y_centroid_px=float(ty),
                    snr=12.5,
                )
            ],
            CutoutConfig(stamp_size=31, boundary_mode="pad"),
        )
        stamp_records.append(
            CandidateCutoutStampRecord.from_triplet(
                candidate_id="cand-001",
                epoch_index=epoch_idx,
                bjd_tdb=2460755.75658 + epoch_idx * 0.01042,
                mag=20.4 + 0.02 * epoch_idx,
                mag_err=0.08,
                target_x_px=15.0,
                target_y_px=15.0,
                triplet=batch.triplets[0],
            )
        )

    with CatalogDatabase(DuckDBConfig(database=db_file, read_only=False)) as db:
        repo = CandidateDetectionRepository(db)
        repo.ensure_table()
        repo.insert_many(_seed_candidates())

        cutout_repo = CandidateCutoutRepository(db)
        cutout_repo.ensure_table()
        assert cutout_repo.upsert_many(stamp_records) == 4

    app = create_app(db_config=DuckDBConfig(database=db_file, read_only=True))
    with TestClient(app) as client:
        headers = _auth_headers(client)
        cors_resp = client.options(
            "/api/v1/status",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert cors_resp.status_code == 200
        assert "access-control-allow-origin" in cors_resp.headers

        cutout_resp = client.get("/api/v1/candidates/cand-001/cutouts", headers=headers)
        assert cutout_resp.status_code == 200
        seq: CandidateCutoutSequenceResponse = (
            CandidateCutoutSequenceResponse.model_validate(cutout_resp.json())
        )
        assert seq.candidate_id == "cand-001"
        assert seq.cutout_size == 31
        assert len(seq.epochs) == 4
        assert len(seq.epochs[0].pixels) == 31
        assert len(seq.epochs[0].pixels[0]) == 31
        assert seq.epochs[0].reference_pixels is not None
        assert seq.epochs[0].difference_pixels is not None
        assert len(seq.epochs[0].difference_pixels) == 31
        assert seq.epochs[0].difference_pixels[15][15] == pytest.approx(45.0, abs=1e-3)

        # A persisted candidate without extracted stamps returns 404.
        unseeded_cutout = client.get(
            "/api/v1/candidates/cand-002/cutouts", headers=headers
        )
        assert unseeded_cutout.status_code == 404

        # An extracted 51x51 sequence serves 51x51 stamps from the API cache.
        from LUMIA.api.schemas import CandidateCutoutEpoch

        cached_seq = CandidateCutoutSequenceResponse(
            candidate_id="cand-002",
            field_id="ZTF_0412",
            cutout_size=51,
            epochs=[
                CandidateCutoutEpoch(
                    epoch_index=0,
                    bjd_tdb=2460755.75662,
                    mag=21.2,
                    mag_err=0.11,
                    target_x_px=25.0,
                    target_y_px=25.0,
                    pixels=[[2.5] * 51 for _ in range(51)],
                    reference_pixels=[[1.0] * 51 for _ in range(51)],
                    difference_pixels=[[1.5] * 51 for _ in range(51)],
                )
            ],
        )
        app.state.lumia_state.register_candidate_cutouts(cached_seq)
        cached_resp = client.get(
            "/api/v1/candidates/cand-002/cutouts", headers=headers
        )
        assert cached_resp.status_code == 200
        parsed_cached = CandidateCutoutSequenceResponse.model_validate(
            cached_resp.json()
        )
        assert parsed_cached.cutout_size == 51
        assert len(parsed_cached.epochs[0].pixels) == 51

        missing_cutout = client.get(
            "/api/v1/candidates/unknown-cand/cutouts", headers=headers
        )
        assert missing_cutout.status_code == 404


def test_api_layer_contains_no_raw_sql_mock_cutouts_or_import_time_app_creation() -> None:
    """P0-1 & P0-4 guard: routes.py has no SQL or synthetic Gaussian mock cutout code."""
    import importlib
    import inspect

    import LUMIA.api.main as api_main
    import LUMIA.api.routes as api_routes

    reloaded_main = importlib.reload(api_main)
    assert "app" not in reloaded_main.__dict__

    routes_src = inspect.getsource(api_routes)
    for forbidden in (
        "CREATE TABLE",
        "INSERT INTO",
        "SELECT ",
        "FROM candidate_detections",
        "query_numpy",
        "default_rng",
        "static_stars",
        "math.sin",
    ):
        assert forbidden not in routes_src, f"Forbidden leak in routes.py: {forbidden}"

    # Lazily accessing api_main.app (for ASGI servers) constructs a FastAPI instance on demand
    lazy_app = reloaded_main.app
    assert lazy_app.title == "Project LUMIA API"


def test_startup_seeds_candidates_and_cutouts() -> None:
    """Verify that a fresh API startup seeds dummy candidates and multi-epoch cutouts."""
    app = create_app()
    with TestClient(app) as client:
        headers = _auth_headers(client)

        cand_resp = client.get("/api/v1/candidates", headers=headers)
        assert cand_resp.status_code == 200
        candidates = cand_resp.json()["candidates"]
        assert len(candidates) >= 5
        sim_ids = [c["candidate_id"] for c in candidates]
        assert "sim-neo-001" in sim_ids

        cutout_resp = client.get("/api/v1/candidates/sim-neo-001/cutouts", headers=headers)
        assert cutout_resp.status_code == 200
        cutout_data = cutout_resp.json()
        assert cutout_data["cutout_size"] == 31
        assert len(cutout_data["epochs"]) == 4
        first_epoch = cutout_data["epochs"][0]
        assert len(first_epoch["pixels"]) == 31
        assert len(first_epoch["reference_pixels"]) == 31
        assert len(first_epoch["difference_pixels"]) == 31
