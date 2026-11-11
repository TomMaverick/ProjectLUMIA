"""Unit tests for Module 2 Prefect DAG orchestration and DuckDB catalog queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import prefect.settings as prefect_settings
import pytest
from prefect import Flow, Task
from pydantic import BaseModel, ValidationError

from LUMIA.data_pipeline.database import (
    CatalogDatabase,
    CatalogQueryParams,
    DuckDBConfig,
    StellarCatalogRecord,
)
from LUMIA.data_pipeline.orchestration import (
    PipelineJobRequest,
    PipelineJobResult,
    query_catalog_sources_task,
    run_catalog_pipeline_flow,
    validate_job_request_task,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(scope="module")
def isolated_prefect_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Provide an isolated local SQLite Prefect database with clean server teardown."""
    from prefect.server.api.server import SubprocessASGIServer

    prefect_dir: Path = tmp_path_factory.mktemp("prefect_test_db")
    db_url: str = "sqlite+aiosqlite:///" + (prefect_dir / "prefect.db").as_posix()
    with prefect_settings.temporary_settings(
        updates={
            prefect_settings.PREFECT_SERVER_DATABASE_CONNECTION_URL: db_url,
            prefect_settings.PREFECT_SERVER_ANALYTICS_ENABLED: False,
            prefect_settings.PREFECT_LOGGING_TO_API_ENABLED: False,
        }
    ):
        try:
            yield
        finally:
            for server in list(SubprocessASGIServer._instances.values()):
                if server.running:
                    server.stop()
            SubprocessASGIServer._instances.clear()


def _mock_stellar_catalog() -> list[StellarCatalogRecord]:
    """Return deterministic mock stellar catalog records across sky coordinates."""
    return [
        StellarCatalogRecord(source_id=1001, ra_deg=180.10, dec_deg=15.20, mag_g=13.4),
        StellarCatalogRecord(source_id=1002, ra_deg=180.25, dec_deg=15.35, mag_g=15.8),
        StellarCatalogRecord(source_id=1003, ra_deg=180.40, dec_deg=15.50, mag_g=19.2),
        StellarCatalogRecord(source_id=1004, ra_deg=210.00, dec_deg=-22.10, mag_g=12.1),
    ]


def test_duckdb_in_memory_stellar_catalog_parametrized_query() -> None:
    """Verify in-memory DuckDB table creation, insertion, and parametrized NumPy query."""
    config = DuckDBConfig(database=":memory:")
    records: list[StellarCatalogRecord] = _mock_stellar_catalog()

    with CatalogDatabase(config) as db:
        db.create_stellar_catalog_table()
        inserted: int = db.insert_stellar_records(records)
        assert inserted == 4

        query_params = CatalogQueryParams(
            ra_min_deg=180.0,
            ra_max_deg=181.0,
            dec_min_deg=15.0,
            dec_max_deg=16.0,
            max_mag=16.5,
        )
        result: np.ndarray = db.query_stellar_catalog(query_params)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        np.testing.assert_array_equal(result["source_id"], np.array([1001, 1002], dtype=np.int64))
        np.testing.assert_allclose(result["ra_deg"], np.array([180.10, 180.25]))
        np.testing.assert_allclose(result["dec_deg"], np.array([15.20, 15.35]))
        np.testing.assert_allclose(result["mag_g"], np.array([13.4, 15.8]))

        raw_result: np.ndarray = db.query_numpy(
            "SELECT source_id, mag_g FROM stellar_catalog WHERE mag_g < ? ORDER BY source_id ASC",
            params=[14.0],
        )
        assert isinstance(raw_result, np.ndarray)
        np.testing.assert_array_equal(
            raw_result["source_id"],
            np.array([1001, 1004], dtype=np.int64),
        )


def test_duckdb_file_based_persistence_and_read_only_query(tmp_path: Path) -> None:
    """Verify file-backed DuckDB creation, persistence across sessions, and read-only access."""
    db_file: Path = tmp_path / "mock_catalog.duckdb"
    records: list[StellarCatalogRecord] = _mock_stellar_catalog()

    with CatalogDatabase(DuckDBConfig(database=db_file, read_only=False)) as writer_db:
        writer_db.create_stellar_catalog_table()
        writer_db.insert_stellar_records(records)

    assert db_file.exists()

    with CatalogDatabase(DuckDBConfig(database=db_file, read_only=True)) as reader_db:
        result: np.ndarray = reader_db.query_numpy(
            "SELECT source_id, ra_deg, dec_deg, mag_g FROM stellar_catalog WHERE source_id = ?",
            params=[1004],
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert int(result["source_id"][0]) == 1004
        assert float(result["mag_g"][0]) == pytest.approx(12.1)


def test_database_pydantic_validation_errors() -> None:
    """Verify invalid catalog coordinates, inverted bounds, or bad DB configs fail validation."""
    with pytest.raises(ValidationError):
        StellarCatalogRecord(source_id=1, ra_deg=365.0, dec_deg=0.0, mag_g=15.0)

    with pytest.raises(ValidationError):
        CatalogQueryParams(
            ra_min_deg=180.0,
            ra_max_deg=181.0,
            dec_min_deg=20.0,
            dec_max_deg=10.0,
            max_mag=18.0,
        )

    with pytest.raises(ValidationError):
        DuckDBConfig(database=":memory:", read_only=True)


def test_prefect_flow_and_task_execution(
    isolated_prefect_env: None,
    tmp_path: Path,
) -> None:
    """Verify Prefect flow and tasks execute end-to-end with Pydantic-validated models."""
    assert isinstance(validate_job_request_task, Task)
    assert isinstance(query_catalog_sources_task, Task)
    assert isinstance(run_catalog_pipeline_flow, Flow)
    assert validate_job_request_task.retries >= 1
    assert query_catalog_sources_task.retries >= 2
    assert isinstance(query_catalog_sources_task.retry_delay_seconds, list)
    assert query_catalog_sources_task.retry_delay_seconds[1] > query_catalog_sources_task.retry_delay_seconds[0]

    db_file: Path = tmp_path / "pipeline_catalog.duckdb"
    records: list[StellarCatalogRecord] = _mock_stellar_catalog()
    with CatalogDatabase(DuckDBConfig(database=db_file)) as db:
        db.create_stellar_catalog_table()
        db.insert_stellar_records(records)

    job_request = PipelineJobRequest(
        job_id="job-001",
        target_field="ZTF_FIELD_0412",
        query_params=CatalogQueryParams(
            ra_min_deg=180.0,
            ra_max_deg=181.0,
            dec_min_deg=15.0,
            dec_max_deg=16.0,
            max_mag=17.0,
        ),
        db_config=DuckDBConfig(database=db_file, read_only=True),
    )

    job_result: PipelineJobResult = run_catalog_pipeline_flow(request=job_request)

    assert isinstance(job_result, BaseModel)
    assert isinstance(job_result, PipelineJobResult)
    assert job_result.job_id == "job-001"
    assert job_result.target_field == "ZTF_FIELD_0412"
    assert job_result.matched_source_count == 2
    assert job_result.brightest_mag_g == pytest.approx(13.4)
    assert job_result.status == "completed"


def test_prefect_flow_rejects_invalid_payload(isolated_prefect_env: None) -> None:
    """Verify invalid raw dictionary payloads raise a Pydantic ValidationError in the flow."""
    invalid_payload: dict[str, object] = {
        "job_id": "",
        "target_field": "ZTF_FIELD_0412",
    }
    with pytest.raises(ValidationError):
        run_catalog_pipeline_flow(request=invalid_payload)
