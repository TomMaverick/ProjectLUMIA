"""Prefect DAG flows and tasks for Module 2 pipeline orchestration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from prefect import flow, task
from prefect.tasks import exponential_backoff
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from LUMIA.data_pipeline.database import (
    CatalogDatabase,
    CatalogQueryParams,
    DuckDBConfig,
)

if TYPE_CHECKING:
    from prefect.client.schemas.objects import TaskRun
    from prefect.states import State
    from prefect.tasks import Task

logger = logging.getLogger(__name__)

_ORCHESTRATION_MODEL_CONFIG: ConfigDict = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)


class PipelineJobRequest(BaseModel):
    """Validated input contract for a catalog ingestion/query pipeline job."""

    model_config = _ORCHESTRATION_MODEL_CONFIG

    job_id: str = Field(
        ...,
        min_length=1,
        description="Unique pipeline job identifier.",
    )
    target_field: str = Field(
        ...,
        min_length=1,
        description="Target sky field or exposure group identifier.",
    )
    query_params: CatalogQueryParams = Field(
        ...,
        description="Validated spatial and photometric catalog filter parameters.",
    )
    db_config: DuckDBConfig = Field(
        default_factory=DuckDBConfig,
        description="DuckDB analytical database connection settings.",
    )


class CatalogQuerySummary(BaseModel):
    """Validated summary produced by the catalog query task."""

    model_config = _ORCHESTRATION_MODEL_CONFIG

    matched_source_count: int = Field(..., ge=0)
    brightest_mag_g: float | None = Field(default=None)


class PipelineJobResult(BaseModel):
    """Validated final result emitted by the catalog pipeline flow."""

    model_config = _ORCHESTRATION_MODEL_CONFIG

    job_id: str = Field(..., min_length=1)
    target_field: str = Field(..., min_length=1)
    matched_source_count: int = Field(..., ge=0)
    brightest_mag_g: float | None = Field(default=None)
    status: Literal["completed"] = "completed"


def log_task_failure(
    failed_task: Task[..., Any],
    task_run: TaskRun,
    state: State[Any],
) -> None:
    """Log structured task failure context for pipeline diagnostics."""
    logger.error(
        "Prefect task '%s' (run_id=%s) failed with state '%s': %s",
        failed_task.name,
        task_run.id,
        state.name,
        state.message,
    )


def _should_retry_non_validation_error(
    failed_task: Task[..., Any],
    task_run: TaskRun,
    state: State[Any],
) -> bool:
    """Retry transient failures while failing fast on deterministic ValidationError."""
    del failed_task, task_run
    exc: Any = state.result(raise_on_failure=False)
    return not isinstance(exc, ValidationError)


@task(
    name="validate-pipeline-job-request",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=1.0),
    retry_condition_fn=_should_retry_non_validation_error,
    on_failure=[log_task_failure],
)
def validate_job_request_task(
    request: PipelineJobRequest | dict[str, Any],
) -> PipelineJobRequest:
    """Validate and normalize an incoming pipeline job request into a Pydantic model."""
    if isinstance(request, PipelineJobRequest):
        return request
    return PipelineJobRequest.model_validate(request)


@task(
    name="query-catalog-sources",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=1.0),
    retry_condition_fn=_should_retry_non_validation_error,
    on_failure=[log_task_failure],
)
def query_catalog_sources_task(
    db_config: DuckDBConfig,
    query_params: CatalogQueryParams,
) -> CatalogQuerySummary:
    """Query DuckDB stellar catalog sources and return a validated summary."""
    with CatalogDatabase(db_config) as db:
        sources: np.ndarray = db.query_stellar_catalog(query_params)

    count: int = int(sources.shape[0])
    brightest: float | None = (
        float(np.min(sources["mag_g"])) if count > 0 else None
    )
    return CatalogQuerySummary(
        matched_source_count=count,
        brightest_mag_g=brightest,
    )


@flow(name="lumia-catalog-pipeline-flow")
def run_catalog_pipeline_flow(
    request: PipelineJobRequest | dict[str, Any],
) -> PipelineJobResult:
    """Execute the validated Prefect DAG flow for stellar catalog querying."""
    validated_request: PipelineJobRequest = validate_job_request_task(request)
    summary: CatalogQuerySummary = query_catalog_sources_task(
        db_config=validated_request.db_config,
        query_params=validated_request.query_params,
    )
    return PipelineJobResult(
        job_id=validated_request.job_id,
        target_field=validated_request.target_field,
        matched_source_count=summary.matched_source_count,
        brightest_mag_g=summary.brightest_mag_g,
        status="completed",
    )
