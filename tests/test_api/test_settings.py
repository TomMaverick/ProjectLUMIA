"""Hermetic unit tests for /api/v1/settings endpoint and archive storage path configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from LUMIA.api.main import create_app
from LUMIA.core.concurrency import ResourceBudgetConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_get_settings_returns_default_archive_path() -> None:
    """GET /api/v1/settings returns baseline configuration including archive_storage_path."""
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/v1/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "archive_storage_path" in data
        assert data["archive_storage_path"] == "data/archives/"
        assert data["num_cpu_threads"] >= 1
        assert data["max_gpu_memory_gb"] > 0.0
        assert data["enable_openmp_threading_control"] is True


def test_post_settings_updates_archive_path_and_creates_dir(tmp_path: Path) -> None:
    """POST /api/v1/settings updates archive_storage_path and provisions the directory on disk."""
    custom_archive_dir = tmp_path / "custom_telescope_fits"
    assert not custom_archive_dir.exists()

    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/settings",
            json={"archive_storage_path": str(custom_archive_dir)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["archive_storage_path"] == str(custom_archive_dir)

        # Directory must be automatically created
        assert custom_archive_dir.is_dir()

        # Follow-up GET returns the updated path
        get_resp = client.get("/api/v1/settings")
        assert get_resp.status_code == 200
        assert get_resp.json()["archive_storage_path"] == str(custom_archive_dir)


def test_post_settings_rejects_empty_archive_path() -> None:
    """POST /api/v1/settings rejects whitespace or empty archive_storage_path."""
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/settings",
            json={"archive_storage_path": "   "},
        )
        assert resp.status_code == 422


def test_app_startup_creates_configured_archive_dir(tmp_path: Path) -> None:
    """create_app creates archive directory automatically on startup."""
    startup_dir = tmp_path / "startup_provisioned_archive"
    assert not startup_dir.exists()

    budget = ResourceBudgetConfig(archive_storage_path=str(startup_dir))
    create_app(resource_budget=budget)

    assert startup_dir.is_dir()
