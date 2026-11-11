"""Module 2 data engineering, DuckDB catalog/candidate repositories, and Prefect orchestration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from LUMIA.data_pipeline.database import (
    CatalogDatabase,
    CatalogQueryParams,
    DuckDBConfig,
    StellarCatalogRecord,
)
from LUMIA.data_pipeline.oltp_repositories import (
    CandidateTriageRepository,
    CandidateTriageStateRecord,
    OLTPRepositorySet,
    OperatorAuthRepository,
    OperatorProfile,
    PipelineJobRecord,
    PipelineJobRepository,
    RepositoryConflictError,
    TriageRecord,
    create_oltp_repositories,
    refresh_token_hash,
)
from LUMIA.data_pipeline.ground_truth_orchestrator import (
    GroundTruthEpochResult,
    GroundTruthPipeline,
)
from LUMIA.data_pipeline.hdf5_storage import HDF5GroundTruthWriter
from LUMIA.data_pipeline.repositories import (
    CandidateCutoutRepository,
    CandidateCutoutStampRecord,
    CandidateDetectionRepository,
    CandidateDetectionSummary,
    CandidateListQuery,
    CutoutStampSize,
)
from LUMIA.data_pipeline.schema_oltp import OLTPDatabaseConfig
from LUMIA.data_pipeline.triage_loader import (
    DEFAULT_TOMOE_HDF5_PATH,
    TomoeHDF5TriageLoader,
)

if TYPE_CHECKING:
    from LUMIA.data_pipeline.orchestration import (
        CatalogQuerySummary,
        PipelineJobRequest,
        PipelineJobResult,
        log_task_failure,
        query_catalog_sources_task,
        run_catalog_pipeline_flow,
        validate_job_request_task,
    )

_ORCHESTRATION_EXPORTS: frozenset[str] = frozenset(
    {
        "CatalogQuerySummary",
        "PipelineJobRequest",
        "PipelineJobResult",
        "log_task_failure",
        "query_catalog_sources_task",
        "run_catalog_pipeline_flow",
        "validate_job_request_task",
    }
)

__all__ = [
    "CandidateCutoutRepository",
    "CandidateCutoutStampRecord",
    "CandidateDetectionRepository",
    "CandidateDetectionSummary",
    "CandidateTriageRepository",
    "CandidateTriageStateRecord",
    "CandidateListQuery",
    "CatalogDatabase",
    "CatalogQueryParams",
    "CatalogQuerySummary",
    "CutoutStampSize",
    "DEFAULT_TOMOE_HDF5_PATH",
    "DuckDBConfig",
    "GroundTruthEpochResult",
    "GroundTruthPipeline",
    "HDF5GroundTruthWriter",
    "TomoeHDF5TriageLoader",
    "PipelineJobRequest",
    "PipelineJobRecord",
    "PipelineJobRepository",
    "PipelineJobResult",
    "OperatorAuthRepository",
    "OperatorProfile",
    "OLTPDatabaseConfig",
    "OLTPRepositorySet",
    "StellarCatalogRecord",
    "TriageRecord",
    "RepositoryConflictError",
    "create_oltp_repositories",
    "refresh_token_hash",
    "log_task_failure",
    "query_catalog_sources_task",
    "run_catalog_pipeline_flow",
    "validate_job_request_task",
]


def __getattr__(name: str) -> object:
    """Lazily resolve Prefect orchestration symbols without eager import overhead."""
    if name in _ORCHESTRATION_EXPORTS:
        orch_mod = importlib.import_module("LUMIA.data_pipeline.orchestration")
        value: object = getattr(orch_mod, name)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return module attributes including lazily exposed orchestration symbols."""
    return sorted({*globals().keys(), *_ORCHESTRATION_EXPORTS})
