# LUMIA API

## Overview
LUMIA API implements Module 10 of Project LUMIA: an asynchronous REST and WebSocket gateway engineered with FastAPI. It delivers operator authentication, candidate catalog exploration, candidate triage disposition recording, Prefect pipeline job monitoring, runtime configuration updates, and real-time WebSocket telemetry streaming to the frontend user interface.

## Interface
- `create_app`: Application factory constructing the FastAPI service with configured CORS rules, standard exception handlers, and lifespan-managed database repositories.
- `APIState`: Lifespan state container encapsulating DuckDB OLAP repositories, PostgreSQL OLTP repositories, and JWT cryptographic configuration.
- `router`: Central APIRouter mounting REST endpoints for authentication, candidates, pipeline jobs, candidate triage, catalog queries, and runtime settings.
- `EventHub`: In-memory event hub managing bounded fan-out subscriptions, log priorities, and JSON telemetry frame broadcasting.
- `LoginRequest` / `AccessTokenResponse`: Authentication payload contracts for operator credentials and JWT tokens.
- `CandidateListResponse` / `CandidateDetailResponse`: Responses delivering moving-object candidate detections and triage history.
- `CandidateCutoutSequenceResponse` / `CandidateCutoutEpoch`: Multi-epoch cutout triplet sequence contract containing pixel scale ($''/\text{px}$), FOV physical size, filter band, and survey identifier.
- `CandidateTriageRequest` / `CandidateTriageResponse`: Request/response schemas for recording human operator candidate triage decisions (`confirmed_real`, `flagged_bogus`, `follow_up`).
- `JobSubmissionRequest` / `JobStatusResponse`: Contracts governing pipeline execution jobs and lifecycle states.
- `PipelineStatusResponse`: Health-check and backend connection diagnostic report schema.
- `SettingsResponse` / `SettingsUpdateRequest`: Runtime CPU/GPU resource budgets and archive storage configuration schemas.
- `SnapshotEnvelope` / `JobStateChangedEnvelope` / `TaskStateChangedEnvelope` / `JobMetricsUpdatedEnvelope` / `TaskLogAppendedEnvelope`: Strongly-typed WebSocket message envelopes.

## Invariants
- `create_app`: In production (`LUMIA_ENV=production`), `LUMIA_JWT_SECRET` is strictly required and must contain at least 32 bytes. Wildcard `*` CORS origins are rejected when `allow_credentials=True`.
- All requests and responses adhere to strict Pydantic v2 schemas; unknown attributes raise immediate HTTP 400 validation errors.
- Passwords are verified exclusively using Argon2id; bearer tokens must be valid HS256 JWTs with valid expiration timestamps.

## Side Effects
- Establishes and terminates database connections across DuckDB (in-memory or on-disk) and PostgreSQL connection pools.
- Writes operator authentication records, triage disposition decisions, and pipeline status logs to PostgreSQL.
- Broadcasts real-time events and telemetry snapshots across active WebSocket client connections.
- Updates global CPU and GPU resource limits in `LUMIA.core.concurrency`.

## Verification
- Run: `uv run pytest tests/test_api`
