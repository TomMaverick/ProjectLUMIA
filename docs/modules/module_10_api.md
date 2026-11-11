# Module 10: FastAPI REST, JWT Authentication & WebSocket Streaming Specification

* **Module Number:** Module 10 (Backend API Sub-Specification; complements `docs/modules/module_10_web_ui.md`)
* **Governance Alignment:** `ADR-0001` (Module Boundaries), `ADR-0002` (Strict Pydantic v2 Contracts), `ADR-0003` (Polyglot Persistence Separation), `docs/database_schema_blueprint.md`, `docs/backlog.md` (Section P2.7)
* **Target Package Scope:** `src/LUMIA/api/` (`main.py`, `routes.py`, `schemas.py`, `auth.py`, `ws.py`) and `tests/test_api/`

---

## 1. Overview & Architectural Governance

Module 10 exposes the external HTTP/1.1 REST (OpenAPI 3.1) and WebSocket interface powering the Project LUMIA dark-mode React/TypeScript observatory dashboard (`#3b185f` to `#581c87` purple accent palette), interactive multi-epoch `31x31` / `51x51` blink-loop inspection, real-time Prefect pipeline telemetry, and human-in-the-loop Active Learning candidate triage.

To maintain strict compliance with Project LUMIA architectural governance:

1. **Zero Raw SQL & Zero Ephemeral State (`ADR-0001`, `ADR-0003`):**
   * `src/LUMIA/api/` must **never** execute raw SQL strings, instantiate `duckdb` or `sqlalchemy` sessions directly, or store mutable job state in process-local Python dictionaries.
   * All persistence operations are delegated through FastAPI lifespan-injected **Module 2** (`src/LUMIA/data_pipeline/`) repositories and coordinators:
     * **PostgreSQL OLTP Repositories:** `OperatorAuthRepository` (`operators`, `auth_refresh_sessions`), `PipelineJobRepository` (`pipeline_jobs`, `pipeline_tasks`, `pipeline_task_logs`), and `CandidateTriageRepository` (`candidate_triage_state`, `candidate_triage_events`).
     * **DuckDB / Apache Parquet OLAP Repositories:** `CatalogDatabase` (`stellar_catalog`), `CandidateDetectionRepository` (`candidate_detections`), `CandidateCutoutRepository` (`candidate_cutout_stamps`), and `TrackletRepository` (`tracklets`, `v_tracklet_triage_summary`).

2. **Strict Pydantic v2 Serialization & Input Validation (`ADR-0002`):**
   * Every OpenAPI request body, response payload, JWT claim model, and WebSocket message envelope must inherit from the shared `LumiaBaseModel` configured with `strict=True`, `extra="forbid"`, `frozen=True`, and `allow_inf_nan=False`.
   * All floating-point query parameters and JSON fields strictly reject `NaN`, `+Inf`, and `-Inf` (`422 Unprocessable Entity` on ingress; guaranteed RFC 8259 JSON compliance on egress).
   * All path identifiers (`job_id`, `candidate_id`, `field_id`, `tracklet_id`) are constrained to `1..128` ASCII characters matching `^[A-Za-z0-9_.-]+$`.

3. **CORS & Transport Security:**
   * Cross-Origin Resource Sharing (`CORSMiddleware`) requires an explicit origin allowlist (defaulting in local development to `http://localhost:5173` and `http://127.0.0.1:5173`). Configuring wildcard `*` while `allow_credentials=True` is strictly prohibited and raises a startup configuration error.

---

## 2. JWT Authentication & Role-Based Access Control (RBAC)

Authentication combines stateless, short-lived JSON Web Token (JWT) access tokens for low-latency REST validation with stateful, revocable refresh sessions persisted in PostgreSQL (`operators` and `auth_refresh_sessions` in `docs/database_schema_blueprint.md` Section 2.3.A).

### 2.1 Cryptographic & Token Lifecycle Parameters

| Parameter | Access Token | Refresh Token | WebSocket Handshake Ticket |
| :--- | :--- | :--- | :--- |
| **Purpose** | Authorize REST API calls | Obtain rotated token pairs without re-entering credentials | Single-use browser WebSocket authentication without exposing bearer JWTs in URLs |
| **Signing / Format** | JWS (`HS256` with >= 256-bit secret or `EdDSA` asymmetric key) | JWS (`HS256` / `EdDSA`) with unique `jti` persisted as `SHA-256` hex hash in `auth_refresh_sessions.token_jti_hash` | Cryptographically random 256-bit opaque token or short-lived JWS (`token_use = "ws_ticket"`) |
| **Time-to-Live (TTL)** | `900 seconds` (`15 minutes`) | `604,800 seconds` (`7 days`) | `30 seconds` (single-use) |
| **Transport** | `Authorization: Bearer <access_token>` header | `HttpOnly; Secure; SameSite=Strict; Path=/api/v1/auth` cookie (`lumia_refresh_token`) | Query parameter `?ticket=<ws_ticket>` or `Sec-WebSocket-Protocol` bearer subprotocol for non-browser clients |
| **Password Hashing** | N/A | `Argon2id` (`operators.password_hash`) verified in constant time inside `OperatorAuthRepository` | N/A |

### 2.2 Canonical JWT Access Token Claims

```json
{
  "iss": "lumia-api",
  "aud": "lumia-web-ui",
  "sub": "8d4c9a12-5b7e-4f01-9c3a-1e8b7d6f5a40",
  "username": "obs_operator_01",
  "role": "astronomer",
  "token_use": "access",
  "jti": "c9f4e210-31a8-4d92-8b77-0f12a34b56c7",
  "iat": 1794140000,
  "nbf": 1794140000,
  "exp": 1794140900
}
```

### 2.3 Authentication, Refresh Rotation & Reuse-Detection Flow

```mermaid
sequenceDiagram
    autonumber
    participant UI as React Web-UI
    participant API as Module 10 FastAPI
    participant Repo as Module 2 OperatorAuthRepository
    participant PG as PostgreSQL (operators / auth_refresh_sessions)

    UI->>API: POST /api/v1/auth/login {username, password}
    API->>Repo: verify_credentials(username, password)
    Repo->>PG: SELECT operator + Argon2id verify (is_active = TRUE)
    PG-->>Repo: OperatorAccount (role = astronomer)
    API->>Repo: create_refresh_session(operator_id, jti_sha256, expires_at_utc, ip, ua)
    Repo->>PG: INSERT INTO auth_refresh_sessions
    API-->>UI: 200 OK {access_token, expires_in: 900, operator} + Set-Cookie(lumia_refresh_token)

    UI->>API: POST /api/v1/candidates/{id}/triage (Authorization: Bearer <access_token>)
    API->>API: Verify JWT signature, exp, aud, token_use="access" & role RBAC
    API-->>UI: 200 OK CandidateTriageResponse

    Note over UI,API: Access token nears expiration (15 min)
    UI->>API: POST /api/v1/auth/refresh (Cookie: lumia_refresh_token)
    API->>Repo: rotate_refresh_session(old_jti_sha256, new_jti_sha256)
    Repo->>PG: Verify revoked_at_utc IS NULL & expires_at_utc > NOW(); set revoked_at_utc=NOW(); INSERT new row
    API-->>UI: 200 OK {access_token, expires_in: 900} + Set-Cookie(rotated lumia_refresh_token)

    UI->>API: POST /api/v1/auth/ws-ticket (Authorization: Bearer <access_token>)
    API-->>UI: 201 Created {ticket, expires_in: 30}
    UI->>API: WS /api/v1/ws/jobs?ticket=<ticket>
    API->>API: Consume single-use ticket -> websocket.accept()
    API-->>UI: event_type = "snapshot" (initial job states)
```

* **Refresh Replay Protection:** If `POST /api/v1/auth/refresh` presents a refresh `jti` whose row in `auth_refresh_sessions` already has `revoked_at_utc IS NOT NULL`, `OperatorAuthRepository` immediately revokes **all** active sessions for that `operator_id` (`UPDATE auth_refresh_sessions SET revoked_at_utc = NOW() WHERE operator_id = :id AND revoked_at_utc IS NULL`) to contain potential token theft and returns `401 Unauthorized`.
* **Account Deactivation Enforcement:** During token refresh and WebSocket ticket issuance, `operators.is_active` is re-verified in PostgreSQL; deactivated operators are denied immediately.

### 2.4 Role-Based Access Control (RBAC) Matrix

Roles align strictly with the `operators.role` `CHECK` constraint in `docs/database_schema_blueprint.md` (`'admin'`, `'astronomer'`, `'reviewer'`, `'service'`):

| Capability / Endpoint Group | Unauthenticated | `reviewer` | `astronomer` | `service` | `admin` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `GET /api/v1/status` (Health & UI Theme) | Allowed | Allowed | Allowed | Allowed | Allowed |
| `POST /api/v1/auth/login`, `/refresh` | Allowed | Allowed | Allowed | Allowed | Allowed |
| `GET /api/v1/detections/summary`, `/candidates*`, `/jobs*`, `/catalog/query` | Denied (`401`) | Allowed | Allowed | Allowed | Allowed |
| `POST /api/v1/candidates/{candidate_id}/triage` (Active Learning Disposition) | Denied (`401`) | Allowed | Allowed | Denied (`403`) | Allowed |
| `POST /api/v1/jobs`, `POST /api/v1/jobs/{job_id}/cancel` | Denied (`401`) | Denied (`403`) | Allowed | Allowed | Allowed |
| `WS /api/v1/ws/jobs` (Real-Time Job & Task Stream) | Denied (`4401`) | Allowed | Allowed | Allowed | Allowed |
| Operator Management (`/api/v1/admin/operators*`) | Denied (`401`) | Denied (`403`) | Denied (`403`) | Denied (`403`) | Allowed |

---

## 3. OpenAPI v1 Route Specification (`/api/v1`)

All endpoints are mounted under `/api/v1` with `Content-Type: application/json` (except `204 No Content` responses).

### 3.1 Endpoint Summary Table

| Method | OpenAPI Path | Auth / Minimum Role | Backing Module 2 Repository | Success Status & Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/status` | Public | `CatalogDatabase`, `PipelineJobRepository` | `200 OK` — `PipelineStatusResponse` |
| `POST` | `/api/v1/auth/login` | Public (Rate-Limited) | `OperatorAuthRepository` | `200 OK` — `AccessTokenResponse` |
| `POST` | `/api/v1/auth/refresh` | Refresh Cookie / Token | `OperatorAuthRepository` | `200 OK` — `AccessTokenResponse` |
| `POST` | `/api/v1/auth/logout` | Any Authenticated Role | `OperatorAuthRepository` | `204 No Content` |
| `GET` | `/api/v1/auth/me` | Any Authenticated Role | `OperatorAuthRepository` | `200 OK` — `OperatorProfileResponse` |
| `POST` | `/api/v1/auth/ws-ticket` | Any Authenticated Role | In-memory TTL store / `OperatorAuthRepository` | `201 Created` — `WebSocketTicketResponse` |
| `POST` | `/api/v1/jobs` | `astronomer`, `service`, `admin` | `PipelineJobRepository`, `CatalogDatabase` | `201 Created` — `JobStatusResponse` |
| `GET` | `/api/v1/jobs` | `reviewer`+ | `PipelineJobRepository` | `200 OK` — `JobListResponse` |
| `GET` | `/api/v1/jobs/{job_id}` | `reviewer`+ | `PipelineJobRepository` | `200 OK` — `JobStatusResponse` |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | `astronomer`, `admin` | `PipelineJobRepository` | `200 OK` — `JobStatusResponse` |
| `GET` | `/api/v1/detections/summary` | `reviewer`+ | `CandidateDetectionRepository` | `200 OK` — `DetectionSummaryResponse` |
| `GET` | `/api/v1/candidates` | `reviewer`+ | `CandidateDetectionRepository`, `CandidateTriageRepository` | `200 OK` — `CandidateListResponse` |
| `GET` | `/api/v1/candidates/{candidate_id}` | `reviewer`+ | `CandidateDetectionRepository`, `CandidateTriageRepository` | `200 OK` — `CandidateDetailResponse` |
| `GET` | `/api/v1/candidates/{candidate_id}/cutouts` | `reviewer`+ | `CandidateCutoutRepository`, `CandidateDetectionRepository` | `200 OK` — `CandidateCutoutSequenceResponse` |
| `POST` | `/api/v1/candidates/{candidate_id}/triage` | `reviewer`, `astronomer`, `admin` | `CandidateTriageRepository`, `CandidateDetectionRepository` | `200 OK` — `CandidateTriageResponse` |
| `POST` | `/api/v1/catalog/query` | `reviewer`+ | `CatalogDatabase` | `200 OK` — `CatalogQueryResponse` |

---

### 3.2 Detailed Endpoint Contracts

#### A. System Health & Observatory Theme
* **`GET /api/v1/status`**
  * **Description:** Verifies liveness/readiness across both DuckDB OLAP and PostgreSQL OLTP engines, reports total tracked jobs, and supplies strict dark-mode observatory UI configuration (`theme_mode: "dark"`, `accent_color_hex: "#581c87"`).
  * **Response (`200 OK` — `PipelineStatusResponse`):**
    ```json
    {
      "status": "operational",
      "duckdb_connected": true,
      "postgres_connected": true,
      "total_jobs": 42,
      "theme_mode": "dark",
      "accent_color_hex": "#581c87"
    }
    ```

#### B. Operator Authentication Endpoints
* **`POST /api/v1/auth/login`**
  * **Request Body (`LoginRequest`):**
    * `username`: `string` (`1..64` chars, required)
    * `password`: `SecretStr` (`8..256` chars, required; redacted from all logs/tracebacks)
  * **Response (`200 OK` — `AccessTokenResponse`):** Sets `HttpOnly` refresh cookie and returns:
    ```json
    {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "token_type": "bearer",
      "expires_in": 900,
      "operator": {
        "operator_id": "8d4c9a12-5b7e-4f01-9c3a-1e8b7d6f5a40",
        "username": "obs_operator_01",
        "email": "operator01@observatory.org",
        "role": "astronomer"
      }
    }
    ```
  * **Error Codes:** `401 Unauthorized` (invalid username, wrong password, or `is_active = false`), `422 Unprocessable Entity` (strict type/schema mismatch), `429 Too Many Requests` (exceeded login rate limit per IP/account).

* **`POST /api/v1/auth/refresh`**
  * **Input:** `lumia_refresh_token` `HttpOnly` cookie (or `RefreshTokenRequest` body for headless `service` roles).
  * **Response (`200 OK` — `AccessTokenResponse`):** Issues a new 15-minute access token and rotates the refresh session in PostgreSQL.
  * **Error Codes:** `401 Unauthorized` (missing, expired, or previously revoked refresh session).

* **`POST /api/v1/auth/logout`**
  * **Behavior:** Sets `revoked_at_utc = NOW()` on the caller's current `auth_refresh_sessions` record and clears the `lumia_refresh_token` cookie (`Max-Age=0`).
  * **Response:** `204 No Content`.

* **`GET /api/v1/auth/me`**
  * **Response (`200 OK` — `OperatorProfileResponse`):** Returns `operator_id`, `username`, `email`, `role`, and `is_active`.

* **`POST /api/v1/auth/ws-ticket`**
  * **Behavior:** Issues a single-use, 30-second ticket bound to `operator_id` and `role` for authenticating a subsequent `WS /api/v1/ws/jobs` connection without placing bearer JWTs into query URLs.
  * **Response (`201 Created` — `WebSocketTicketResponse`):**
    ```json
    {
      "ticket": "wst_9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c",
      "expires_in": 30
    }
    ```

#### C. Pipeline Job Submission & Inspection
* **`POST /api/v1/jobs`**
  * **Request Body (`JobSubmissionRequest`):**
    * `job_id`: `string` (`^[A-Za-z0-9_.-]{1,128}$`)
    * `target_field`: `string` (`^[A-Za-z0-9_.-]{1,128}$`)
    * `observation_night_utc`: `date` (`YYYY-MM-DD`, defaults to current UTC date if omitted in configuration schema or explicitly required in v1 contract)
    * `profile_id`: `string | null` (optional reference to `telescope_profiles.profile_id`)
    * `query_params`: `CatalogQueryParams` (`ra_min_deg` in `[0.0, 360.0)`, `ra_max_deg` in `[0.0, 360.0]`, `dec_min_deg` in `[-90.0, 90.0]`, `dec_max_deg` in `[-90.0, 90.0]` with `dec_min_deg <= dec_max_deg`, `max_mag` finite `float`; supports `ra_min_deg > ra_max_deg` for `0° / 360°` meridian wrap-around)
  * **Behavior:** Persists a new `'queued'` record into PostgreSQL `pipeline_jobs` (linked to `submitted_by_operator_id` from the JWT `sub` claim), evaluates DuckDB reference catalog coverage, triggers asynchronous Prefect flow execution via Module 2, and emits a `job.state_changed` event to `/api/v1/ws/jobs`.
  * **Response (`201 Created` — `JobStatusResponse`):** Returns job metadata, `matched_catalog_sources`, `brightest_catalog_mag_g`, `status` (`"queued" | "running" | "completed" | "failed" | "cancelled"`), and timestamps.
  * **Error Codes:** `401`, `403`, `409 Conflict` (duplicate `job_id`), `422 Unprocessable Entity`.

* **`GET /api/v1/jobs`**
  * **Query Parameters:** `status` (optional filter), `observation_night_utc` (optional `YYYY-MM-DD`), `limit` (`1..200`, default `50`), `offset` (`>= 0`, default `0`).
  * **Response (`200 OK` — `JobListResponse`):** `{ "total_returned": int, "jobs": [JobStatusResponse, ...] }`.

* **`GET /api/v1/jobs/{job_id}`** and **`POST /api/v1/jobs/{job_id}/cancel`**
  * **Path Parameter:** `job_id` (`SafeIdentifierPath`).
  * **Error Codes:** `404 Not Found` if `job_id` does not exist; `409 Conflict` on `/cancel` if job is already in terminal state (`completed`, `failed`, `cancelled`).

#### D. Candidate Detections, Cutout Blink-Loops & Active Learning Triage
* **`GET /api/v1/detections/summary`**
  * **Query Parameter:** `min_cnn_score` (`float` in `[0.0, 1.0]`, default `0.8`, `allow_inf_nan=False`).
  * **Response (`200 OK` — `DetectionSummaryResponse`):** `total_candidates`, `high_confidence_count`, `distinct_fields`, `mean_snr`, `brightest_mag`.

* **`GET /api/v1/candidates`**
  * **Query Parameters:**
    * `min_cnn_score`: `float` in `[0.0, 1.0]`, default `0.0`
    * `field_id`: `string | null` (`1..128` chars)
    * `disposition`: `"unreviewed" | "confirmed_real" | "follow_up" | "flagged_bogus" | null`
    * `limit`: `int` in `[1, 500]`, default `100`
    * `offset`: `int >= 0`, default `0`
  * **Response (`200 OK` — `CandidateListResponse`):** Ordered deterministically by `(cnn_score DESC, snr DESC, candidate_id ASC)`.

* **`GET /api/v1/candidates/{candidate_id}`**
  * **Response (`200 OK` — `CandidateDetailResponse`):** Composes canonical `CandidateDetection` fields (`candidate_id`, `field_id`, `ra_deg`, `dec_deg`, `bjd_tdb`, `mag`, `mag_err`, `snr`, `cnn_score`, `model_version`, `astrometric_uncertainty_arcsec`, `astrometric_residual_arcsec`, `tracklet_id`) with current PostgreSQL `CandidateTriageState` (`disposition`, `reviewed_by_operator_id`, `updated_at_utc`).

* **`GET /api/v1/candidates/{candidate_id}/cutouts`**
  * **Response (`200 OK` — `CandidateCutoutSequenceResponse`):** Serves odd-dimension (`cutout_size` in `{31, 51}`) multi-epoch `pixels` (Science), `reference_pixels` (Reference), and `difference_pixels` (ZOGY Difference) nested `float` matrices alongside per-epoch `bjd_tdb`, `mag`, `mag_err`, `target_x_px`, and `target_y_px` for interactive blink-loop rendering.

* **`POST /api/v1/candidates/{candidate_id}/triage`** *(Active Learning Feedback Endpoint)*
  * **Path Parameter:** `candidate_id` (`^[A-Za-z0-9_.-]{1,128}$`).
  * **Request Body (`CandidateTriageRequest`):**
    ```json
    {
      "disposition": "confirmed_real",
      "notes": "Clean 4-epoch linear motion across ZOGY difference stamps; zero dipole residual."
    }
    ```
    * `disposition`: `"confirmed_real" | "flagged_bogus" | "follow_up" | "unreviewed"` (strictly typed literal).
    * `notes`: `string | null` (`max_length = 2000`, default `null`).
  * **Transactional Execution via Module 2:**
    1. Confirms `candidate_id` exists in `CandidateDetectionRepository.get_by_id(candidate_id)` (retrieves `field_id`, `job_id`, `cnn_score`, and `model_version`); if missing, raises `404 Not Found`.
    2. Calls `CandidateTriageRepository.record_disposition(...)` which, inside a **single PostgreSQL ACID transaction**:
       * Reads existing `candidate_triage_state.disposition` (defaulting `previous_disposition` to `"unreviewed"` on first review).
       * Upserts `candidate_triage_state` setting `disposition = new_disposition`, `reviewed_by_operator_id = jwt.sub`, `cnn_score_at_review`, `model_version_at_review`, `updated_at_utc = NOW()`, and resets `hdf5_export_batch_id = NULL` whenever `disposition` changes so **Module 8** (`docs/modules/module_08_ground_truth_mocking.md`) exports the updated label on the next Active Learning cycle.
       * Appends an immutable audit row to `candidate_triage_events` (`triage_event_id`, `candidate_id`, `operator_id`, `previous_disposition`, `new_disposition`, `notes`, `decided_at_utc`).
  * **Response (`200 OK` — `CandidateTriageResponse`):**
    ```json
    {
      "triage_event_id": 1084,
      "candidate_id": "job-20261108-01_e0_c0042",
      "job_id": "job-20261108-01",
      "field_id": "ZTF_0412",
      "previous_disposition": "unreviewed",
      "disposition": "confirmed_real",
      "reviewed_by_operator_id": "8d4c9a12-5b7e-4f01-9c3a-1e8b7d6f5a40",
      "cnn_score_at_review": 0.942,
      "model_version_at_review": "resnet_lite_s31_v2",
      "queued_for_hdf5_export": true,
      "decided_at_utc": "2026-11-08T12:15:30.412Z"
    }
    ```

#### E. Reference Stellar Catalog Sky-Box Query
* **`POST /api/v1/catalog/query`**
  * **Request Body:** `CatalogQueryParams` (`ra_min_deg`, `ra_max_deg`, `dec_min_deg`, `dec_max_deg`, `max_mag`).
  * **Response (`200 OK` — `CatalogQueryResponse`):** `{ "query": CatalogQueryParams, "matched_count": int, "items": [CatalogStarItem, ...] }`.

### 3.3 Standardized Error Response Envelope

All `4xx` and `5xx` HTTP errors (outside FastAPI's native `422` Pydantic validation array, or unified alongside it) conform to deterministic JSON structures without leaking internal SQL tracebacks or filesystem paths:

| HTTP Status | `error_code` | Trigger Condition |
| :--- | :--- | :--- |
| `400 Bad Request` | `BAD_REQUEST` | Malformed JSON syntax or invalid protocol framing. |
| `401 Unauthorized` | `AUTH_REQUIRED` / `TOKEN_EXPIRED` / `INVALID_CREDENTIALS` | Missing/expired/invalid JWT bearer token or revoked refresh session (`WWW-Authenticate: Bearer` header included). |
| `403 Forbidden` | `INSUFFICIENT_ROLE` | Authenticated operator role lacks permission for the target action (e.g., `reviewer` attempting `POST /api/v1/jobs`). |
| `404 Not Found` | `RESOURCE_NOT_FOUND` | Requested `job_id` or `candidate_id` does not exist in Module 2 repositories. |
| `409 Conflict` | `STATE_CONFLICT` | Duplicate `job_id` submission or invalid job lifecycle transition. |
| `422 Unprocessable Entity` | `VALIDATION_ERROR` | Strict Pydantic v2 validation failure (wrong scalar type, unknown field via `extra="forbid"`, `NaN`/`Inf` float, or out-of-range coordinate). |
| `429 Too Many Requests` | `RATE_LIMITED` | Authentication brute-force protection threshold exceeded. |
| `503 Service Unavailable` | `STORAGE_UNAVAILABLE` | PostgreSQL or DuckDB repository pool failed readiness check. |

---

## 4. WebSocket Job Streaming Specification (`/api/v1/ws/jobs`)

### 4.1 Endpoint, Handshake & Subscription Lifecycle

* **Endpoint URI:** `ws://<host>/api/v1/ws/jobs` (`wss://` in TLS production)
* **Query Parameters:**
  * `ticket`: Required for browser clients (`wst_...` obtained from `POST /api/v1/auth/ws-ticket`); headless `service` clients may alternatively pass `Authorization: Bearer <access_token>` header.
  * `job_id`: Optional filter (`^[A-Za-z0-9_.-]{1,128}$`). When omitted, the client subscribes to status transitions across all jobs visible to the authenticated operator.
* **Connection Lifecycle & Close Codes:**
  1. **Pre-Accept Authentication:** Validate `ticket` (atomically marking it consumed) or `Bearer` header. If invalid or expired, reject or close immediately with application code **`4401` (`Unauthorized`)** (or **`4403` (`Forbidden`)** if `is_active = false`).
  2. **Initial Snapshot Synchronization:** Immediately upon `websocket.accept()`, the server emits a single `snapshot` frame containing the current state of matching active/recent jobs so reconnecting clients never miss transitions that occurred during network reconnects.
  3. **Event Streaming & Heartbeat:** Server pushes typed state/metric/log frames as Module 2 Prefect tasks update PostgreSQL and DuckDB, interleaved with a `heartbeat` frame every `25 seconds` when idle.
  4. **Bounded Backpressure:** Each connected WebSocket client owns a bounded outbound `asyncio.Queue(maxsize=256)`. If a slow client falls behind capacity, transient `task.log_appended` frames are dropped first and `dropped_log_frames` is incremented on the next `heartbeat`; if state frames overflow, the connection is closed with code **`1013` (`Try Again Later`)** so the client reconnects cleanly via `snapshot` without stalling pipeline workers.

### 4.2 Canonical Server-to-Client Envelope Structure

Every server-to-client WebSocket frame is a UTF-8 JSON object conforming to a discriminated union keyed on `event_type`, strictly forbidding `NaN`/`Inf` floats:

| Envelope Field | JSON Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `schema_version` | `string` | Literal `"1.0"` | Wire protocol version for forward-compatible UI parsing. |
| `event_id` | `string` | UUIDv4 | Unique identifier for deduplication across UI state stores. |
| `event_type` | `string` | `"snapshot" \| "job.state_changed" \| "task.state_changed" \| "task.log_appended" \| "job.metrics_updated" \| "heartbeat" \| "error"` | Discriminator determining the exact schema of `payload`. |
| `sequence_no` | `integer` | `>= 1` | Strictly monotonically increasing counter scoped to the WebSocket connection. |
| `emitted_at_utc` | `string` | ISO-8601 UTC (`...Z`) | Server emission timestamp in UTC. |
| `payload` | `object` | Strict Pydantic v2 model per `event_type` | Event-specific body detailed in Section 4.3. |

---

### 4.3 WebSocket Event Payload Schemas

#### 1. Initial Synchronization Frame (`event_type = "snapshot"`)
Emitted immediately after connection acceptance (and whenever the client dynamically changes its subscription filter):
```json
{
  "schema_version": "1.0",
  "event_id": "3c8e7a10-91f2-4b33-a8d1-55e67f890123",
  "event_type": "snapshot",
  "sequence_no": 1,
  "emitted_at_utc": "2026-11-08T12:20:00.105Z",
  "payload": {
    "subscribed_job_id": null,
    "active_jobs": [
      {
        "job_id": "job-20261108-01",
        "target_field_id": "ZTF_0412",
        "observation_night_utc": "2026-11-08",
        "status": "running",
        "matched_catalog_sources": 18420,
        "brightest_catalog_mag_g": 9.41,
        "completed_tasks": 2,
        "total_tasks": 5,
        "submitted_at_utc": "2026-11-08T12:18:10.000Z",
        "started_at_utc": "2026-11-08T12:18:12.240Z",
        "completed_at_utc": null
      }
    ]
  }
}
```

#### 2. Job Lifecycle Transition (`event_type = "job.state_changed"`)
Emitted whenever `pipeline_jobs.status` transitions (`queued` -> `running` -> `completed` | `failed` | `cancelled`):
```json
{
  "schema_version": "1.0",
  "event_id": "7f21b904-12c4-4e89-9a01-88c76d54e321",
  "event_type": "job.state_changed",
  "sequence_no": 2,
  "emitted_at_utc": "2026-11-08T12:21:44.890Z",
  "payload": {
    "job_id": "job-20261108-01",
    "target_field_id": "ZTF_0412",
    "previous_status": "running",
    "status": "completed",
    "prefect_flow_run_id": "a1b2c3d4-e5f6-4789-8012-3456789abcde",
    "matched_catalog_sources": 18420,
    "parquet_partition_uri": "data/olap/candidate_detections/observation_night=2026-11-08/field_id=ZTF_0412/job_id=job-20261108-01.parquet",
    "error_summary": null,
    "started_at_utc": "2026-11-08T12:18:12.240Z",
    "completed_at_utc": "2026-11-08T12:21:44.875Z"
  }
}
```

#### 3. Module Task State & Retry Update (`event_type = "task.state_changed"`)
Maps directly to PostgreSQL `pipeline_tasks` transitions across Modules 1–10:
```json
{
  "schema_version": "1.0",
  "event_id": "9a04d122-66b1-4c77-8e55-1234567890ab",
  "event_type": "task.state_changed",
  "sequence_no": 3,
  "emitted_at_utc": "2026-11-08T12:19:50.312Z",
  "payload": {
    "task_run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "job_id": "job-20261108-01",
    "task_name": "classify_cutout_triplets_gpu",
    "module_number": 7,
    "status": "running",
    "attempt_count": 1,
    "max_retries": 3,
    "next_retry_at_utc": null,
    "started_at_utc": "2026-11-08T12:19:50.300Z",
    "finished_at_utc": null
  }
}
```

#### 4. Analytical Detection & Tracklet Metrics Update (`event_type = "job.metrics_updated"`)
Pushes live DuckDB analytical counters to update dashboard KPI cards without polling `GET /api/v1/detections/summary`:
```json
{
  "schema_version": "1.0",
  "event_id": "4e19c330-88d2-4f10-b722-998877665544",
  "event_type": "job.metrics_updated",
  "sequence_no": 4,
  "emitted_at_utc": "2026-11-08T12:21:40.005Z",
  "payload": {
    "job_id": "job-20261108-01",
    "field_id": "ZTF_0412",
    "extracted_candidates": 312,
    "cnn_passed_candidates": 29,
    "linked_tracklets": 4,
    "brightest_candidate_mag": 18.65,
    "mean_candidate_snr": 9.4
  }
}
```

#### 5. Structured Task Log Entry (`event_type = "task.log_appended"`)
Streams structured entries recorded in `pipeline_task_logs` (`severity` in `"DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"`):
```json
{
  "schema_version": "1.0",
  "event_id": "5b77e811-23a9-4d04-9f11-223344556677",
  "event_type": "task.log_appended",
  "sequence_no": 5,
  "emitted_at_utc": "2026-11-08T12:19:52.118Z",
  "payload": {
    "log_id": 90412,
    "job_id": "job-20261108-01",
    "task_run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "attempt_number": 1,
    "severity": "INFO",
    "event_code": "CNN_BATCH_SCORED",
    "message": "Scored 312 31x31 ZOGY triplets in 3 mini-batches (29 >= threshold 0.80).",
    "logged_at_utc": "2026-11-08T12:19:52.110Z"
  }
}
```

#### 6. Keep-Alive Heartbeat (`event_type = "heartbeat"`)
```json
{
  "schema_version": "1.0",
  "event_id": "1d22f344-55a6-4b78-9c01-fedcba987654",
  "event_type": "heartbeat",
  "sequence_no": 6,
  "emitted_at_utc": "2026-11-08T12:20:25.105Z",
  "payload": {
    "server_status": "operational",
    "dropped_log_frames": 0
  }
}
```

### 4.4 Client-to-Server Control Frames

Connected WebSocket clients may send strictly validated JSON control frames (any unrecognized keys or malformed payloads return an `event_type = "error"` frame with `error_code = "INVALID_WS_COMMAND"`):

| `action` | Payload Fields | Behavior |
| :--- | :--- | :--- |
| `"subscribe"` | `{ "action": "subscribe", "job_id": "job-20261108-01", "min_log_severity": "INFO" }` | Scopes stream to a specific `job_id` (or `null` for all jobs) and sets minimum log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`), triggering a fresh `snapshot` frame. |
| `"ping"` | `{ "action": "ping", "client_sent_at_utc": "2026-11-08T12:20:30.000Z" }` | Immediately answered with a server `heartbeat` frame for UI round-trip latency display. |

## 5. Runtime Configuration

Production deployments must set `LUMIA_ENV=production` and provide `LUMIA_JWT_SECRET` with at least 256 bits of entropy. `LUMIA_OLTP_DATABASE_URL` selects the SQLAlchemy PostgreSQL connection (for example, `postgresql+psycopg://...`); isolated tests and local development default to SQLite. Refresh cookies are `HttpOnly`, `Secure`, and `SameSite=Strict`.
