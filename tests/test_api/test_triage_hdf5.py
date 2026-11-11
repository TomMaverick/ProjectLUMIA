"""Hermetic tests for Tomo-e-Gozen HDF5 triage loading and API queue endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from LUMIA.api.main import create_app
from LUMIA.api.schemas import (
    CandidateCutoutSequenceResponse,
    CandidateDetection,
    CandidateTriageResponse,
    CatalogQueryParams,
    JobSubmissionRequest,
    TriageQueueResponse,
)
from LUMIA.data_pipeline import DEFAULT_TOMOE_HDF5_PATH, TomoeHDF5TriageLoader


def _auth_headers(client: TestClient, role: str = "reviewer") -> dict[str, str]:
    """Register and log in operator to obtain authenticated bearer token."""
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
    assert response.status_code == 200, f"Login failed: {response.text}"
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


class TestTomoeHDF5TriageLoader:
    """Test suite verifying fast indexed slicing from Tomo-e-Gozen ground-truth HDF5."""

    def test_loader_availability_and_metadata(self) -> None:
        """Verify the default Tomo-e dataset is available and inspect its split statistics."""
        loader = TomoeHDF5TriageLoader(DEFAULT_TOMOE_HDF5_PATH)
        assert loader.is_available() is True

        info = loader.get_dataset_info()
        assert info["available"] is True
        assert "test" in info["splits"]
        test_split = info["splits"]["test"]
        assert test_split["total_samples"] == 256003
        assert test_split["real_count"] == 292
        assert test_split["bogus_count"] == 255711

    def test_load_triage_sample_balanced_batch(self) -> None:
        """Verify bulk slicing produces valid CandidateDetection and CandidateCutoutStampRecord lists."""
        loader = TomoeHDF5TriageLoader(DEFAULT_TOMOE_HDF5_PATH)
        candidates, stamps = loader.load_triage_sample(
            split="test",
            num_real=15,
            num_bogus=15,
            field_id="TOMOE_01",
        )

        assert len(candidates) == 30
        assert len(stamps) == 30

        for cand, stamp in zip(candidates, stamps):
            assert isinstance(cand, CandidateDetection)
            assert cand.candidate_id == stamp.candidate_id
            assert cand.field_id == "TOMOE_01"
            assert 0.0 <= cand.ra_deg < 360.0
            assert -90.0 <= cand.dec_deg <= 90.0
            assert cand.snr is not None and cand.snr > 0.0
            assert cand.cnn_score is not None and 0.0 <= cand.cnn_score <= 1.0

            assert stamp.cutout_size == 31
            assert len(stamp.science_pixels) == 31
            assert len(stamp.science_pixels[0]) == 31
            assert stamp.reference_pixels is not None
            assert len(stamp.reference_pixels) == 31
            assert stamp.difference_pixels is not None
            assert len(stamp.difference_pixels) == 31

    def test_load_candidate_stamp_by_id(self) -> None:
        """Verify individual candidate stamp lookups succeed via cached split indexing."""
        loader = TomoeHDF5TriageLoader(DEFAULT_TOMOE_HDF5_PATH)
        stamp = loader.load_candidate_stamp("test_real_0", split="test")

        assert stamp is not None
        assert stamp.candidate_id == "test_real_0"
        assert stamp.cutout_size == 31
        assert len(stamp.science_pixels) == 31
        assert len(stamp.reference_pixels) == 31
        assert len(stamp.difference_pixels) == 31

        # Non-existent candidate returns None without error
        missing = loader.load_candidate_stamp("non_existent_candidate_xyz")
        assert missing is None


class TestTriageAPIEndpoints:
    """Integration test suite verifying /api/v1/triage/queue and cutout endpoints."""

    def test_get_triage_queue_returns_tomoe_candidates(self) -> None:
        """Verify GET /api/v1/triage/queue returns real Tomo-e candidates in expected schema."""
        app = create_app()
        with TestClient(app) as client:
            headers = _auth_headers(client, role="reviewer")
            response = client.get("/api/v1/triage/queue?split=test&limit=20&offset=0", headers=headers)

            assert response.status_code == 200
            data = response.json()
            queue = TriageQueueResponse.model_validate(data)

            assert queue.source == "hdf5_tomoe"
            assert queue.split == "test"
            assert queue.total_returned == 20
            assert len(queue.candidates) == 20
            assert len(queue.items) == 20

            # Verify candidate identifiers and fields
            first_cand = queue.candidates[0]
            assert "test_" in first_cand.candidate_id
            assert 0.0 <= first_cand.ra_deg < 360.0
            assert first_cand.snr is not None

    def test_get_triage_queue_with_include_cutouts(self) -> None:
        """Verify GET /api/v1/triage/queue with include_cutouts=true delivers 31x31 triplets."""
        app = create_app()
        with TestClient(app) as client:
            headers = _auth_headers(client, role="reviewer")
            response = client.get(
                "/api/v1/triage/queue?split=test&limit=5&include_cutouts=true",
                headers=headers,
            )

            assert response.status_code == 200
            data = response.json()
            queue = TriageQueueResponse.model_validate(data)

            assert queue.total_returned == 5
            for item in queue.items:
                assert item.cutouts is not None
                assert item.cutouts.cutout_size == 31
                assert len(item.cutouts.epochs) == 1
                epoch = item.cutouts.epochs[0]
                assert len(epoch.pixels) == 31
                assert epoch.reference_pixels is not None and len(epoch.reference_pixels) == 31
                assert epoch.difference_pixels is not None and len(epoch.difference_pixels) == 31

    def test_get_candidate_cutouts_for_tomoe_record(self) -> None:
        """Verify GET /api/v1/candidates/{id}/cutouts serves Tomo-e 3-channel cutouts."""
        app = create_app()
        with TestClient(app) as client:
            headers = _auth_headers(client, role="reviewer")
            response = client.get("/api/v1/candidates/test_real_0/cutouts", headers=headers)

            assert response.status_code == 200
            data = response.json()
            seq = CandidateCutoutSequenceResponse.model_validate(data)

            assert seq.candidate_id == "test_real_0"
            assert seq.cutout_size == 31
            assert len(seq.epochs) == 1
            epoch = seq.epochs[0]
            assert len(epoch.pixels) == 31
            assert len(epoch.reference_pixels) == 31
            assert len(epoch.difference_pixels) == 31

    def test_get_triage_queue_database_source_fallback(self) -> None:
        """Verify source=database explicitly pulls from DuckDB candidate table."""
        app = create_app()
        with TestClient(app) as client:
            headers = _auth_headers(client, role="reviewer")
            response = client.get(
                "/api/v1/triage/queue?source=database&limit=10&offset=0",
                headers=headers,
            )

            assert response.status_code == 200
            data = response.json()
            queue = TriageQueueResponse.model_validate(data)

            assert queue.source == "database"
            assert queue.total_returned > 0
            cand_ids = [c.candidate_id for c in queue.candidates]
            assert "sim-neo-001" in cand_ids

    def test_triage_submission_for_tomoe_candidate(self) -> None:
        """Verify operator can submit triage decisions for a Tomo-e candidate."""
        app = create_app()
        with TestClient(app) as client:
            headers = _auth_headers(client, role="admin")

            # Create backing pipeline job for the candidate's job_id
            job_req = JobSubmissionRequest(
                job_id="job-20261108-01",
                target_field="TOMOE_01",
                query_params=CatalogQueryParams(
                    ra_min_deg=0.0,
                    ra_max_deg=1.0,
                    dec_min_deg=0.0,
                    dec_max_deg=1.0,
                    max_mag=21.0,
                ),
            )
            job_resp = client.post(
                "/api/v1/jobs",
                json=job_req.model_dump(mode="json"),
                headers=headers,
            )
            assert job_resp.status_code == 201

            payload = {
                "disposition": "confirmed_real",
                "notes": "Verified fast-moving asteroid streak on difference cutout.",
            }
            response = client.post(
                "/api/v1/candidates/test_real_0/triage",
                json=payload,
                headers=headers,
            )

            assert response.status_code == 200
            triage_res = CandidateTriageResponse.model_validate_json(response.content)
            assert triage_res.candidate_id == "test_real_0"
            assert triage_res.disposition == "confirmed_real"
