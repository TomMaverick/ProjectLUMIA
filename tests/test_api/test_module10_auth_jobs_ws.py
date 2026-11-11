"""Integration coverage for Module 10 authentication, persistence, RBAC, and streaming."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from LUMIA.api.main import create_app
from LUMIA.api.schemas import WEBSOCKET_ENVELOPE_ADAPTER
from LUMIA.core.schemas import CandidateDetection

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(jwt_secret="test-signing-key-with-at-least-32-bytes")
    with TestClient(app) as test_client:
        yield test_client


def _login(client: TestClient, role: str, suffix: str) -> str:
    client.app.state.lumia_state.auth.register_operator(
        username=f"operator-{suffix}",
        email=f"{suffix}@observatory.test",
        password="correct-horse-battery",
        role=role,
    )
    response = client.post(
        "/api/v1/auth/login",
        json={"username": f"operator-{suffix}", "password": "correct-horse-battery"},
    )
    assert response.status_code == 200
    return str(response.json()["access_token"])


def _job(job_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "target_field": "ZTF_0412",
        "query_params": {
            "ra_min_deg": 180.0,
            "ra_max_deg": 181.0,
            "dec_min_deg": 15.0,
            "dec_max_deg": 16.0,
            "max_mag": 17.0,
        },
    }


def test_startup_cors_and_public_status(client: TestClient) -> None:
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"
    assert "postgres_connected" in response.json()

    cors = client.options(
        "/api/v1/status",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert cors.status_code == 200
    assert cors.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_login_refresh_rotation_and_reuse_revokes_sessions(client: TestClient) -> None:
    client.app.state.lumia_state.auth.register_operator(
        username="reviewer",
        email="reviewer@observatory.test",
        password="correct-horse-battery",
        role="reviewer",
    )
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "reviewer", "password": "correct-horse-battery"},
    )
    assert login.status_code == 200
    cookie_header = login.headers["set-cookie"].lower()
    assert "httponly" in cookie_header
    assert "secure" in cookie_header
    assert "samesite=strict" in cookie_header
    old_refresh = login.cookies["lumia_refresh_token"]

    rotated = client.post(
        "/api/v1/auth/refresh",
        cookies={"lumia_refresh_token": old_refresh},
    )
    assert rotated.status_code == 200
    new_refresh = rotated.cookies["lumia_refresh_token"]
    assert new_refresh != old_refresh

    replay = client.post(
        "/api/v1/auth/refresh",
        cookies={"lumia_refresh_token": old_refresh},
    )
    assert replay.status_code == 401
    revoked_session = client.post(
        "/api/v1/auth/refresh",
        cookies={"lumia_refresh_token": new_refresh},
    )
    assert revoked_session.status_code == 401


def test_rbac_jobs_inspection_and_cancellation(client: TestClient) -> None:
    unauthenticated = client.get("/api/v1/jobs")
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["error_code"] == "AUTH_REQUIRED"

    reviewer_token = _login(client, "reviewer", "rbac-reviewer")
    headers = {"Authorization": f"Bearer {reviewer_token}"}
    denied = client.post("/api/v1/jobs", json=_job("reviewer-job"), headers=headers)
    assert denied.status_code == 403
    assert denied.json()["error_code"] == "INSUFFICIENT_ROLE"

    astronomer_token = _login(client, "astronomer", "rbac-astronomer")
    headers = {"Authorization": f"Bearer {astronomer_token}"}
    submitted = client.post("/api/v1/jobs", json=_job("api-job-01"), headers=headers)
    assert submitted.status_code == 201
    assert submitted.json()["status"] == "queued"
    listed = client.get(
        "/api/v1/jobs",
        params={
            "status": "queued",
            "observation_night_utc": submitted.json()["observation_night_utc"],
        },
        headers=headers,
    )
    assert [job["job_id"] for job in listed.json()["jobs"]] == ["api-job-01"]

    inspected = client.get("/api/v1/jobs/api-job-01", headers=headers)
    assert inspected.status_code == 200
    assert inspected.json()["job_id"] == "api-job-01"

    cancelled = client.post("/api/v1/jobs/api-job-01/cancel", headers=headers)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert client.post("/api/v1/jobs/api-job-01/cancel", headers=headers).status_code == 409


def test_candidate_pagination_cutouts_and_acid_triage(client: TestClient) -> None:
    reviewer_token = _login(client, "reviewer", "candidate-reviewer")
    headers = {"Authorization": f"Bearer {reviewer_token}"}
    state = client.app.state.lumia_state
    candidates = [
        CandidateDetection(
            candidate_id=f"cand-{i:03}",
            field_id="ZTF_0412",
            ra_deg=180.0 + i / 100,
            dec_deg=15.0,
            bjd_tdb=2460755.5,
            mag=20.0 + i,
            snr=10.0 - i,
            cnn_score=0.9 - i / 10,
        )
        for i in range(3)
    ]
    state.candidates.insert_many(candidates)
    assert client.get("/api/v1/detections/summary", headers=headers).status_code == 200

    page = client.get(
        "/api/v1/candidates?limit=1&offset=1&field_id=ZTF_0412",
        headers=headers,
    )
    assert page.status_code == 200
    assert len(page.json()["candidates"]) == 1
    assert page.json()["candidates"][0]["candidate_id"] == "cand-001"

    missing_cutout = client.get("/api/v1/candidates/cand-000/cutouts", headers=headers)
    assert missing_cutout.status_code == 404

    no_candidate_token = _login(client, "service", "candidate-service")
    denied = client.post(
        "/api/v1/candidates/cand-000/triage",
        json={"disposition": "confirmed_real"},
        headers={"Authorization": f"Bearer {no_candidate_token}"},
    )
    assert denied.status_code == 403

    astronomer_token = _login(client, "astronomer", "candidate-astronomer")
    assert client.post(
        "/api/v1/jobs",
        json=_job("cand-000"),
        headers={"Authorization": f"Bearer {astronomer_token}"},
    ).status_code == 201
    triage = client.post(
        "/api/v1/candidates/cand-000/triage",
        json={"disposition": "confirmed_real", "notes": "Linear motion."},
        headers={"Authorization": f"Bearer {reviewer_token}"},
    )
    assert triage.status_code == 200
    assert triage.json()["previous_disposition"] == "unreviewed"
    assert triage.json()["queued_for_hdf5_export"] is True
    second_triage = client.post(
        "/api/v1/candidates/cand-000/triage",
        json={"disposition": "follow_up"},
        headers={"Authorization": f"Bearer {reviewer_token}"},
    )
    assert second_triage.status_code == 200
    assert second_triage.json()["previous_disposition"] == "confirmed_real"
    filtered = client.get(
        "/api/v1/candidates?disposition=follow_up",
        headers={"Authorization": f"Bearer {reviewer_token}"},
    )
    assert [item["candidate_id"] for item in filtered.json()["candidates"]] == ["cand-000"]
    detail = client.get(
        "/api/v1/candidates/cand-000",
        headers={"Authorization": f"Bearer {reviewer_token}"},
    )
    assert detail.json()["disposition"] == "follow_up"


def test_ws_ticket_snapshot_single_use_and_bounded_backpressure(client: TestClient) -> None:
    token = _login(client, "service", "ws-service")
    headers = {"Authorization": f"Bearer {token}"}
    profile = client.get("/api/v1/auth/me", headers=headers)
    assert profile.status_code == 200
    assert profile.json()["role"] == "service"
    assert client.post("/api/v1/jobs", json=_job("stream-job"), headers=headers).status_code == 201
    ticket_response = client.post("/api/v1/auth/ws-ticket", headers=headers)
    assert ticket_response.status_code == 201
    ticket = ticket_response.json()["ticket"]

    with client.websocket_connect(f"/api/v1/ws/jobs?ticket={ticket}") as websocket:
        snapshot = websocket.receive_json()
        assert snapshot["schema_version"] == "1.0"
        assert snapshot["event_type"] == "snapshot"
        assert snapshot["sequence_no"] == 1
        assert snapshot["payload"]["active_jobs"][0]["job_id"] == "stream-job"
        assert WEBSOCKET_ENVELOPE_ADAPTER.validate_json(
            json.dumps(snapshot, allow_nan=False)
        ).event_type == "snapshot"
        client.app.state.lumia_state.event_hub.publish_state(
            {
                "event_type": "job.state_changed",
                "payload": {
                    "job_id": "stream-job",
                    "target_field_id": "ZTF_0412",
                    "previous_status": "queued",
                    "status": "running",
                },
            }
        )
        state_change = websocket.receive_json()
        assert state_change["event_type"] == "job.state_changed"
        assert state_change["sequence_no"] == 2
        assert WEBSOCKET_ENVELOPE_ADAPTER.validate_json(
            json.dumps(state_change, allow_nan=False)
        ).event_type == "job.state_changed"

    with pytest.raises(WebSocketDisconnect) as disconnect, client.websocket_connect(
        f"/api/v1/ws/jobs?ticket={ticket}"
    ):
        pass
    assert disconnect.value.code == 4401

    hub = client.app.state.lumia_state.event_hub
    subscription = hub.subscribe()
    for index in range(hub.queue_capacity):
        hub.publish_log({"event_type": "task.log_appended", "payload": {"index": index}})
    hub.publish_log({"event_type": "task.log_appended", "payload": {"index": 999}})
    assert subscription.dropped_log_frames == 1
    for index in range(hub.queue_capacity + 1):
        hub.publish_state(
            {"event_type": "job.state_changed", "payload": {"job_id": f"x-{index}"}}
        )
    assert subscription.closed


def test_login_and_identifier_validation(client: TestClient) -> None:
    assert client.post(
        "/api/v1/auth/login",
        json={"username": "missing", "password": "not-a-valid-password"},
    ).status_code == 401
    assert client.get("/api/v1/jobs/not%20ascii").status_code == 401
    token = _login(client, "reviewer", "identifier-validation")
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/jobs/not%20ascii", headers=headers).status_code == 422
    assert client.get(
        "/api/v1/detections/summary?min_cnn_score=nan", headers=headers
    ).status_code == 422
    assert client.post(
        "/api/v1/auth/login",
        json={
            "username": "reviewer-identifier-validation",
            "password": "correct-horse-battery",
            "extra": "forbidden",
        },
    ).status_code == 422


def test_wildcard_cors_is_rejected() -> None:
    with pytest.raises(ValueError, match="Wildcard"):
        create_app(allowed_origins=["*"])


def test_production_requires_a_256_bit_jwt_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMIA_ENV", "production")
    monkeypatch.delenv("LUMIA_JWT_SECRET", raising=False)
    with pytest.raises(ValueError, match="LUMIA_JWT_SECRET"):
        create_app()
    with pytest.raises(ValueError, match="32 bytes"):
        create_app(jwt_secret="too-short")


def test_login_rate_limit_is_enforced(client: TestClient) -> None:
    for _ in range(5):
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "unknown", "password": "wrong-password"},
        )
        assert response.status_code == 401
    limited = client.post(
        "/api/v1/auth/login",
        json={"username": "unknown", "password": "wrong-password"},
    )
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"
