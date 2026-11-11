"""FastAPI v1 routes delegating all persistence to injected Module 2 repositories."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, date, datetime
from typing import Annotated, Any, Literal
from uuid import UUID  # noqa: TC003

from pathlib import Path as FilePath

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response, status

from LUMIA.api.auth import (
    ACCESS_TTL_SECONDS,
    REFRESH_TTL_SECONDS,
    AccessTokenResponse,
    JWTClaims,
    JWTCodec,
    LoginRateLimiter,
    LoginRequest,
    OperatorProfileResponse,
    WebSocketTicketResponse,
    require_roles,
)
from LUMIA.api.schemas import (
    CandidateCutoutEpoch,
    CandidateCutoutSequenceResponse,
    CandidateDetailResponse,
    CandidateListResponse,
    CandidateTriageRequest,
    CandidateTriageResponse,
    CatalogQueryResponse,
    CatalogStarItem,
    DetectionSummaryResponse,
    JobListResponse,
    JobStatusResponse,
    JobSubmissionRequest,
    PipelineStatusResponse,
    SettingsResponse,
    SettingsUpdateRequest,
    TriageQueueItem,
    TriageQueueResponse,
)
from LUMIA.api.ws import EventHub
from LUMIA.api.ws import router as websocket_router
from LUMIA.core.concurrency import (
    ResourceBudgetConfig,
    configure_global_runtime,
    get_active_resource_budget,
)
from LUMIA.core.schemas import CandidateDetection  # noqa: TC001
from LUMIA.data_pipeline import (
    CatalogDatabase,
    CatalogQueryParams,
    DuckDBConfig,
    OLTPDatabaseConfig,
    OLTPRepositorySet,
    PipelineJobRecord,
    RepositoryConflictError,
    StellarCatalogRecord,
    TomoeHDF5TriageLoader,
    create_oltp_repositories,
    refresh_token_hash,
)
from LUMIA.data_pipeline.repositories import (
    CandidateCutoutRepository,
    CandidateDetectionRepository,
    CandidateListQuery,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["lumia-web-ui"])
SafeIdentifierPath = Annotated[
    str,
    Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]{1,128}$"),
]
REFRESH_COOKIE = "lumia_refresh_token"


class APIState:
    """Lifespan-owned Module 2 repositories and volatile WebSocket ticket/event state."""

    def __init__(
        self,
        db_config: DuckDBConfig | None = None,
        oltp_config: OLTPDatabaseConfig | None = None,
        jwt_secret: str | bytes | None = None,
    ) -> None:
        self._db_config = db_config if db_config is not None else DuckDBConfig()
        self._lock = threading.Lock()
        self._db = CatalogDatabase(self._db_config)
        self._oltp_repositories: OLTPRepositorySet = create_oltp_repositories(oltp_config)
        self.auth = self._oltp_repositories.auth
        self.jobs = self._oltp_repositories.jobs
        self.triage = self._oltp_repositories.triage
        self.jwt_codec = JWTCodec(
            jwt_secret if jwt_secret is not None else b"local-development-secret-not-for-production"
        )
        self.event_hub = EventHub()
        self.login_limiter = LoginRateLimiter()
        from LUMIA.api.auth import WebSocketTicketStore

        self.ws_tickets = WebSocketTicketStore()
        self.candidates = CandidateDetectionRepository(self._db)
        self._cutouts = CandidateCutoutRepository(self._db)
        self._cutout_cache: dict[str, CandidateCutoutSequenceResponse] = {}
        self.triage_loader = TomoeHDF5TriageLoader()
        if not self._db_config.read_only:
            self._db.create_stellar_catalog_table()
            self.candidates.ensure_table()
            self._cutouts.ensure_table()
            self._seed_mock_data()

    def _seed_mock_data(self) -> None:
        """Seed initial realistic dummy candidates and cutout sequences if table is empty."""
        with self._lock:
            existing = self.candidates.list_candidates(CandidateListQuery(limit=10))
            if not existing:
                seed_candidates = [
                    CandidateDetection(
                        candidate_id="sim-neo-001",
                        job_id="job-20261108-01",
                        field_id="LUMIA_F0412",
                        ra_deg=184.6125,
                        dec_deg=14.882,
                        bjd_tdb=2460755.74125,
                        mag=20.18,
                        snr=15.4,
                        cnn_score=0.97,
                        model_version="resnet_lite_s31_v2",
                        astrometric_uncertainty_arcsec=0.08,
                        astrometric_residual_arcsec=0.09,
                        survey_id="ZTF",
                        filter_band="r",
                    ),
                    CandidateDetection(
                        candidate_id="sim-mba-002",
                        job_id="job-20261108-01",
                        field_id="LUMIA_F0412",
                        ra_deg=184.9042,
                        dec_deg=15.106,
                        bjd_tdb=2460755.74125,
                        mag=21.05,
                        snr=9.8,
                        cnn_score=0.88,
                        model_version="resnet_lite_s31_v2",
                        astrometric_uncertainty_arcsec=0.11,
                        astrometric_residual_arcsec=0.14,
                        survey_id="Pan-STARRS",
                        filter_band="g",
                    ),
                    CandidateDetection(
                        candidate_id="sim-tno-003",
                        job_id="job-20261108-02",
                        field_id="LUMIA_F0419",
                        ra_deg=186.118,
                        dec_deg=12.451,
                        bjd_tdb=2460755.7621,
                        mag=21.84,
                        snr=6.9,
                        cnn_score=0.74,
                        model_version="resnet_lite_s31_v2",
                        astrometric_uncertainty_arcsec=0.14,
                        astrometric_residual_arcsec=0.19,
                        survey_id="Rubin",
                        filter_band="i",
                    ),
                    CandidateDetection(
                        candidate_id="sim-art-004",
                        job_id="job-20261108-02",
                        field_id="LUMIA_F0419",
                        ra_deg=186.395,
                        dec_deg=12.602,
                        bjd_tdb=2460755.7621,
                        mag=22.31,
                        snr=4.8,
                        cnn_score=0.28,
                        model_version="resnet_lite_s31_v2",
                        astrometric_uncertainty_arcsec=0.22,
                        astrometric_residual_arcsec=0.42,
                        survey_id="ZTF",
                        filter_band="g",
                    ),
                    CandidateDetection(
                        candidate_id="sim-ast-005",
                        job_id="job-20261108-01",
                        field_id="LUMIA_F0412",
                        ra_deg=185.1204,
                        dec_deg=14.995,
                        bjd_tdb=2460755.7425,
                        mag=19.85,
                        snr=18.2,
                        cnn_score=0.99,
                        model_version="resnet_lite_s31_v2",
                        astrometric_uncertainty_arcsec=0.07,
                        astrometric_residual_arcsec=0.06,
                        survey_id="JWST",
                        filter_band="F090W",
                        light_travel_time_sec=4.98,
                    ),
                ]
                self.candidates.insert_many(seed_candidates)
                existing = seed_candidates

            if existing:
                self._cutouts.seed_synthetic_cutouts_if_empty(existing)

            if self.triage_loader.is_available():
                self._seed_tomoe_triage_data()

    def _seed_tomoe_triage_data(self, num_real: int = 50, num_bogus: int = 50) -> None:
        """Seed representative Tomo-e-Gozen ground-truth candidates and cutouts."""
        if not self.triage_loader.is_available():
            return
        if self.candidates.get_by_id("test_real_0") is not None:
            return
        try:
            tomoe_cands, tomoe_stamps = self.triage_loader.load_triage_sample(
                split="test",
                num_real=num_real,
                num_bogus=num_bogus,
            )
            if tomoe_cands:
                existing_cands = {
                    c.candidate_id
                    for c in self.candidates.list_candidates(CandidateListQuery(limit=500))
                }
                fresh = [c for c in tomoe_cands if c.candidate_id not in existing_cands]
                if fresh:
                    self.candidates.insert_many(fresh)
            if tomoe_stamps:
                self._cutouts.upsert_many(tomoe_stamps)
        except Exception:
            logger.exception("Failed to seed Tomo-e-Gozen triage data.")

    def close(self) -> None:
        with self._lock:
            self._db.close()
        self._oltp_repositories.close()

    def check_health(self) -> PipelineStatusResponse:
        with self._lock:
            duckdb_connected = self._db.ping()
        postgres_connected = self._oltp_repositories.check_connection()
        return PipelineStatusResponse(
            status="operational" if duckdb_connected and postgres_connected else "degraded",
            duckdb_connected=duckdb_connected,
            postgres_connected=postgres_connected,
            total_jobs=self.jobs.count_jobs() if postgres_connected else 0,
        )

    def get_detection_summary(self, min_cnn_score: float) -> DetectionSummaryResponse:
        with self._lock:
            summary = self.candidates.summarize(min_cnn_score=min_cnn_score)
        return DetectionSummaryResponse(**summary.model_dump())

    def list_candidates(
        self,
        *,
        min_cnn_score: float,
        field_id: str | None,
        disposition: str | None,
        limit: int,
        offset: int,
    ) -> CandidateListResponse:
        if disposition is None:
            query = CandidateListQuery(
                min_cnn_score=min_cnn_score,
                field_id=field_id,
                limit=limit,
                offset=offset,
            )
            with self._lock:
                items = self.candidates.list_candidates(query)
        else:
            items = []
            scan_offset = 0
            batch_size = 10000
            while len(items) < offset + limit:
                query = CandidateListQuery(
                    min_cnn_score=min_cnn_score,
                    field_id=field_id,
                    limit=batch_size,
                    offset=scan_offset,
                )
                with self._lock:
                    batch = self.candidates.list_candidates(query)
                if not batch:
                    break
                decisions = self.triage.get_dispositions(
                    [item.candidate_id for item in batch]
                )
                items.extend(
                    item
                    for item in batch
                    if decisions.get(item.candidate_id, "unreviewed") == disposition
                )
                scan_offset += len(batch)
                if len(batch) < batch_size:
                    break
            items = items[offset : offset + limit]
        return CandidateListResponse(total_returned=len(items), candidates=items)

    def submit_job(
        self,
        job_request: JobSubmissionRequest,
        operator_id: UUID,
    ) -> JobStatusResponse:
        with self._lock:
            stars: list[StellarCatalogRecord] = self._db.query_stellar_records(
                job_request.query_params
            )
        count = len(stars)
        brightest = min((star.mag_g for star in stars), default=None)
        record = self.jobs.create_job(
            job_id=job_request.job_id,
            operator_id=operator_id,
            target_field_id=job_request.target_field,
            observation_night_utc=job_request.observation_night_utc,
            ra_min_deg=job_request.query_params.ra_min_deg,
            ra_max_deg=job_request.query_params.ra_max_deg,
            dec_min_deg=job_request.query_params.dec_min_deg,
            dec_max_deg=job_request.query_params.dec_max_deg,
            max_mag=job_request.query_params.max_mag,
            profile_id=job_request.profile_id,
            matched_catalog_sources=count,
            brightest_catalog_mag_g=brightest,
        )
        frame: dict[str, object] = {
            "event_type": "job.state_changed",
            "payload": {
                "job_id": record.job_id,
                "target_field_id": record.target_field_id,
                "status": record.status,
            },
        }
        self.event_hub.publish_state(frame)
        return _job_response(record)

    def get_candidate(self, candidate_id: str) -> CandidateDetection | None:
        with self._lock:
            return self.candidates.get_by_id(candidate_id)

    def register_candidate_cutouts(self, sequence: CandidateCutoutSequenceResponse) -> None:
        with self._lock:
            self._cutout_cache[sequence.candidate_id] = sequence

    def get_candidate_cutouts(
        self, candidate_id: str
    ) -> CandidateCutoutSequenceResponse | None:
        with self._lock:
            cached = self._cutout_cache.get(candidate_id)
            if cached is not None:
                return cached
            candidate = self.candidates.get_by_id(candidate_id)
            if candidate is None:
                return None
            records = self._cutouts.get_sequence(candidate_id)
            if not records and self.triage_loader.is_available():
                stamp = self.triage_loader.load_candidate_stamp(candidate_id)
                if stamp is not None:
                    records = [stamp]
        if not records:
            return None
        stamp_size = records[0].cutout_size
        if any(record.cutout_size != stamp_size for record in records):
            raise ValueError(f"Inconsistent cutout dimensions for candidate {candidate_id!r}.")
        survey = getattr(candidate, "survey_id", None) or (records[0].survey_id if records else "ZTF")
        return CandidateCutoutSequenceResponse(
            candidate_id=candidate_id,
            field_id=candidate.field_id,
            cutout_size=stamp_size,
            survey_id=survey,
            epochs=[
                CandidateCutoutEpoch(
                    epoch_index=record.epoch_index,
                    bjd_tdb=record.bjd_tdb,
                    mag=record.mag,
                    mag_err=record.mag_err,
                    target_x_px=record.target_x_px,
                    target_y_px=record.target_y_px,
                    pixels=record.science_pixels,
                    reference_pixels=record.reference_pixels,
                    difference_pixels=record.difference_pixels,
                    survey_id=record.survey_id,
                    filter_band=record.filter_band,
                    pixel_scale_arcsec=record.pixel_scale_arcsec,
                    fov_arcsec=record.fov_arcsec,
                )
                for record in records
            ],
        )

    def get_triage_queue(
        self,
        *,
        split: Literal["test", "train"] = "test",
        limit: int = 100,
        offset: int = 0,
        include_cutouts: bool = False,
        source: Literal["auto", "hdf5", "database"] = "auto",
    ) -> TriageQueueResponse:
        use_hdf5 = (source in ("auto", "hdf5")) and self.triage_loader.is_available()

        if use_hdf5:
            num_real = max(50, (offset + limit) // 2 + 1)
            num_bogus = max(50, (offset + limit) // 2 + 1)
            cands, stamps = self.triage_loader.load_triage_sample(
                split=split,
                num_real=num_real,
                num_bogus=num_bogus,
            )
            with self._lock:
                for cand, stamp in zip(cands, stamps):
                    if cand.candidate_id not in self._cutout_cache:
                        survey = getattr(cand, "survey_id", "ZTF") or "ZTF"
                        filter_b = getattr(cand, "filter_band", "r") or "r"
                        scale = getattr(stamp, "pixel_scale_arcsec", 1.012)
                        fov = getattr(stamp, "fov_arcsec", float(stamp.cutout_size * scale))
                        self._cutout_cache[cand.candidate_id] = CandidateCutoutSequenceResponse(
                            candidate_id=cand.candidate_id,
                            field_id=cand.field_id,
                            cutout_size=stamp.cutout_size,
                            survey_id=survey,
                            epochs=[
                                CandidateCutoutEpoch(
                                    epoch_index=stamp.epoch_index,
                                    bjd_tdb=stamp.bjd_tdb,
                                    mag=stamp.mag,
                                    mag_err=stamp.mag_err,
                                    target_x_px=stamp.target_x_px,
                                    target_y_px=stamp.target_y_px,
                                    pixels=stamp.science_pixels,
                                    reference_pixels=stamp.reference_pixels,
                                    difference_pixels=stamp.difference_pixels,
                                    survey_id=survey,
                                    filter_band=filter_b,
                                    pixel_scale_arcsec=scale,
                                    fov_arcsec=fov,
                                )
                            ],
                        )

            selected_cands = cands[offset : offset + limit]
            stamp_map = {s.candidate_id: s for s in stamps}

            items: list[TriageQueueItem] = []
            for cand in selected_cands:
                stamp = stamp_map.get(cand.candidate_id)
                cutouts_resp: CandidateCutoutSequenceResponse | None = None
                if include_cutouts and stamp is not None:
                    survey = getattr(cand, "survey_id", "ZTF") or "ZTF"
                    filter_b = getattr(cand, "filter_band", "r") or "r"
                    scale = getattr(stamp, "pixel_scale_arcsec", 1.012)
                    fov = getattr(stamp, "fov_arcsec", float(stamp.cutout_size * scale))
                    cutouts_resp = CandidateCutoutSequenceResponse(
                        candidate_id=cand.candidate_id,
                        field_id=cand.field_id,
                        cutout_size=stamp.cutout_size,
                        survey_id=survey,
                        epochs=[
                            CandidateCutoutEpoch(
                                epoch_index=stamp.epoch_index,
                                bjd_tdb=stamp.bjd_tdb,
                                mag=stamp.mag,
                                mag_err=stamp.mag_err,
                                target_x_px=stamp.target_x_px,
                                target_y_px=stamp.target_y_px,
                                pixels=stamp.science_pixels,
                                reference_pixels=stamp.reference_pixels,
                                difference_pixels=stamp.difference_pixels,
                                survey_id=survey,
                                filter_band=filter_b,
                                pixel_scale_arcsec=scale,
                                fov_arcsec=fov,
                            )
                        ],
                    )
                gt_label = (
                    1 if "real" in cand.candidate_id or (cand.cnn_score or 0) >= 0.5 else 0
                )
                items.append(
                    TriageQueueItem(
                        candidate=cand,
                        cutouts=cutouts_resp,
                        ground_truth_label=gt_label,
                    )
                )

            return TriageQueueResponse(
                total_returned=len(selected_cands),
                source="hdf5_tomoe",
                split=split,
                candidates=selected_cands,
                items=items,
            )

        with self._lock:
            db_cands = self.candidates.list_candidates(
                CandidateListQuery(limit=limit, offset=offset)
            )

        items = []
        for cand in db_cands:
            cutouts_resp = None
            if include_cutouts:
                cutouts_resp = self.get_candidate_cutouts(cand.candidate_id)
            items.append(
                TriageQueueItem(
                    candidate=cand,
                    cutouts=cutouts_resp,
                    ground_truth_label=1 if (cand.cnn_score or 0) >= 0.5 else 0,
                )
            )

        return TriageQueueResponse(
            total_returned=len(db_cands),
            source="database",
            split=split,
            candidates=db_cands,
            items=items,
        )

    def query_catalog(self, params: CatalogQueryParams) -> CatalogQueryResponse:
        with self._lock:
            stars = self._db.query_stellar_records(params)
        return CatalogQueryResponse(
            query=params,
            matched_count=len(stars),
            items=[
                CatalogStarItem(
                    source_id=star.source_id,
                    ra_deg=star.ra_deg,
                    dec_deg=star.dec_deg,
                    mag_g=star.mag_g,
                )
                for star in stars
            ],
        )


def _job_response(job: PipelineJobRecord) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        target_field=job.target_field_id,
        observation_night_utc=job.observation_night_utc,
        matched_catalog_sources=job.matched_catalog_sources,
        brightest_catalog_mag_g=job.brightest_catalog_mag_g,
        status=job.status,
        submitted_at_utc=job.submitted_at_utc,
        started_at_utc=job.started_at_utc,
        completed_at_utc=job.completed_at_utc,
        profile_id=job.profile_id,
    )


def get_api_state(request: Request) -> APIState:
    state: Any = getattr(request.app.state, "lumia_state", None)
    if not isinstance(state, APIState):
        raise RuntimeError("APIState has not been initialized on FastAPI application lifespan.")
    return state


StateDep = Annotated[APIState, Depends(get_api_state)]
ReviewerDep = Depends(require_roles("reviewer", "astronomer", "service", "admin"))
JobSubmitterDep = Depends(require_roles("astronomer", "service", "admin"))
JobCancellerDep = Depends(require_roles("astronomer", "admin"))
TriageOperatorDep = Depends(require_roles("reviewer", "astronomer", "admin"))
AnyRoleDep = Depends(require_roles("reviewer", "astronomer", "service", "admin"))


@router.get("/status", response_model=PipelineStatusResponse)
def read_pipeline_status(state: StateDep) -> PipelineStatusResponse:
    return state.check_health()


@router.post("/auth/login", response_model=AccessTokenResponse)
def login(payload: LoginRequest, request: Request, response: Response) -> AccessTokenResponse:
    state: APIState = request.app.state.lumia_state
    remote_ip = request.client.host if request.client else "unknown"
    ip_key = f"ip:{remote_ip}"
    account_key = f"account:{payload.username.casefold()}"
    if not state.login_limiter.allowed(ip_key) or not state.login_limiter.allowed(account_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Login rate limit exceeded.",
            headers={"Retry-After": "60"},
        )
    profile = state.auth.verify_credentials(payload.username, payload.password.get_secret_value())
    if profile is None:
        state.login_limiter.record_failure(ip_key)
        state.login_limiter.record_failure(account_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    state.login_limiter.clear(ip_key)
    state.login_limiter.clear(account_key)
    access_token, _ = state.jwt_codec.issue(
        profile, "access", ttl_seconds=ACCESS_TTL_SECONDS
    )
    refresh_token, refresh_claims = state.jwt_codec.issue(
        profile, "refresh", ttl_seconds=REFRESH_TTL_SECONDS
    )
    expiry = datetime.fromtimestamp(refresh_claims.exp, tz=UTC)
    created = state.auth.create_refresh_session(
        profile.operator_id,
        refresh_token_hash(str(refresh_claims.jti)),
        expiry,
        request.client.host if request.client else None,
        request.headers.get("user-agent"),
    )
    if not created:
        raise HTTPException(status_code=401, detail="Operator account is unavailable.")
    response.set_cookie(
        REFRESH_COOKIE,
        refresh_token,
        max_age=REFRESH_TTL_SECONDS,
        httponly=True,
        secure=True,
        samesite="strict",
        path="/api/v1/auth",
    )
    return AccessTokenResponse(
        access_token=access_token,
        operator=OperatorProfileResponse.from_profile(profile),
    )


@router.post("/auth/refresh", response_model=AccessTokenResponse)
def refresh_tokens(request: Request, response: Response) -> AccessTokenResponse:
    state: APIState = request.app.state.lumia_state
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Refresh token required.")
    try:
        old_claims = state.jwt_codec.decode(token, expected_use="refresh")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.") from exc
    new_refresh_token, new_claims = state.jwt_codec.issue(
        state.auth.get_active_operator(old_claims.sub)
        or _inactive_profile(old_claims),
        "refresh",
        ttl_seconds=REFRESH_TTL_SECONDS,
    )
    profile = state.auth.rotate_refresh_session(
        refresh_token_hash(str(old_claims.jti)),
        refresh_token_hash(str(new_claims.jti)),
        datetime.fromtimestamp(new_claims.exp, tz=UTC),
        request.client.host if request.client else None,
        request.headers.get("user-agent"),
    )
    if profile is None:
        response.delete_cookie(REFRESH_COOKIE, path="/api/v1/auth")
        raise HTTPException(status_code=401, detail="Refresh session is revoked or unavailable.")
    access_token, _ = state.jwt_codec.issue(profile, "access", ttl_seconds=ACCESS_TTL_SECONDS)
    response.set_cookie(
        REFRESH_COOKIE,
        new_refresh_token,
        max_age=REFRESH_TTL_SECONDS,
        httponly=True,
        secure=True,
        samesite="strict",
        path="/api/v1/auth",
    )
    return AccessTokenResponse(
        access_token=access_token,
        operator=OperatorProfileResponse.from_profile(profile),
    )


def _inactive_profile(claims: Any) -> Any:
    """Supply claim-bound identity to repository rotation, which rejects inactive users."""
    from LUMIA.data_pipeline.oltp_repositories import OperatorProfile

    return OperatorProfile(
        operator_id=claims.sub,
        username=claims.username,
        email="inactive@invalid",
        role=claims.role,
        is_active=False,
    )


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    response: Response,
    _: Any = AnyRoleDep,
) -> Response:
    state: APIState = request.app.state.lumia_state
    token = request.cookies.get(REFRESH_COOKIE)
    if token:
        try:
            claims = state.jwt_codec.decode(token, expected_use="refresh")
        except ValueError:
            pass
        else:
            state.auth.revoke_refresh_session(refresh_token_hash(str(claims.jti)))
    response.delete_cookie(REFRESH_COOKIE, path="/api/v1/auth")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/auth/me", response_model=OperatorProfileResponse)
def read_current_operator(
    request: Request,
    claims: Any = AnyRoleDep,
) -> OperatorProfileResponse:
    profile = request.app.state.lumia_state.auth.get_active_operator(claims.sub)
    if profile is None:
        raise HTTPException(status_code=401, detail="Operator account is unavailable.")
    return OperatorProfileResponse.from_profile(profile)


@router.post(
    "/auth/ws-ticket",
    response_model=WebSocketTicketResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_ws_ticket(
    request: Request,
    claims: Any = AnyRoleDep,
) -> WebSocketTicketResponse:
    profile = request.app.state.lumia_state.auth.get_active_operator(claims.sub)
    if profile is None:
        raise HTTPException(status_code=401, detail="Operator account is unavailable.")
    ticket = request.app.state.lumia_state.ws_tickets.issue(profile)
    return WebSocketTicketResponse(ticket=ticket)


@router.get("/detections/summary", response_model=DetectionSummaryResponse)
def read_detection_summary(
    state: StateDep,
    _: Any = ReviewerDep,
    min_cnn_score: Annotated[float, Query(ge=0.0, le=1.0, allow_inf_nan=False)] = 0.8,
) -> DetectionSummaryResponse:
    return state.get_detection_summary(min_cnn_score=min_cnn_score)


@router.get("/candidates", response_model=CandidateListResponse)
def read_candidates(
    state: StateDep,
    _: Any = ReviewerDep,
    min_cnn_score: Annotated[float, Query(ge=0.0, le=1.0, allow_inf_nan=False)] = 0.0,
    field_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]{1,128}$"),
    ] = None,
    disposition: str | None = Query(
        default=None,
        pattern=r"^(unreviewed|confirmed_real|follow_up|flagged_bogus)$",
    ),
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CandidateListResponse:
    return state.list_candidates(
        min_cnn_score=min_cnn_score,
        field_id=field_id,
        disposition=disposition,
        limit=limit,
        offset=offset,
    )


@router.get("/triage/queue", response_model=TriageQueueResponse)
def read_triage_queue(
    state: StateDep,
    _: Any = ReviewerDep,
    split: Literal["test", "train"] = "test",
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_cutouts: bool = False,
    source: Literal["auto", "hdf5", "database"] = "auto",
) -> TriageQueueResponse:
    return state.get_triage_queue(
        split=split,
        limit=limit,
        offset=offset,
        include_cutouts=include_cutouts,
        source=source,
    )


@router.get("/candidates/{candidate_id}", response_model=CandidateDetailResponse)
def read_candidate_detail(
    candidate_id: SafeIdentifierPath,
    state: StateDep,
    _: Any = ReviewerDep,
) -> CandidateDetailResponse:
    candidate = state.get_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    triage = state.triage.get_state(candidate_id)
    return CandidateDetailResponse(
        **candidate.model_dump(),
        disposition=triage.disposition if triage is not None else "unreviewed",
        reviewed_by_operator_id=(
            triage.reviewed_by_operator_id if triage is not None else None
        ),
        updated_at_utc=triage.updated_at_utc if triage is not None else None,
    )


@router.get("/candidates/{candidate_id}/cutouts", response_model=CandidateCutoutSequenceResponse)
def read_candidate_cutouts(
    candidate_id: SafeIdentifierPath,
    state: StateDep,
    _: Any = ReviewerDep,
) -> CandidateCutoutSequenceResponse:
    sequence = state.get_candidate_cutouts(candidate_id)
    if sequence is None:
        raise HTTPException(status_code=404, detail="Candidate cutouts not found.")
    return sequence


@router.post(
    "/candidates/{candidate_id}/triage",
    response_model=CandidateTriageResponse,
)
def triage_candidate(
    candidate_id: SafeIdentifierPath,
    payload: CandidateTriageRequest,
    request: Request,
    claims: Any = TriageOperatorDep,
) -> CandidateTriageResponse:
    state: APIState = request.app.state.lumia_state
    candidate = state.get_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    job_id = candidate.job_id or candidate.candidate_id
    if state.jobs.get_job(job_id) is None:
        raise HTTPException(status_code=409, detail="Candidate has no persisted pipeline job.")
    try:
        result = state.triage.record_disposition(
            candidate_id=candidate_id,
            job_id=job_id,
            field_id=candidate.field_id,
            operator_id=claims.sub,
            disposition=payload.disposition,
            notes=payload.notes,
            cnn_score=candidate.cnn_score,
            model_version=candidate.model_version,
        )
    except RepositoryConflictError as exc:
        raise HTTPException(status_code=409, detail="Triage transaction conflicted.") from exc
    return CandidateTriageResponse(**result.model_dump())


@router.post("/jobs", response_model=JobStatusResponse, status_code=status.HTTP_201_CREATED)
def submit_pipeline_job(
    payload: JobSubmissionRequest,
    request: Request,
    claims: JWTClaims = JobSubmitterDep,
) -> JobStatusResponse:
    state: APIState = request.app.state.lumia_state
    try:
        return state.submit_job(payload, claims.sub)
    except RepositoryConflictError as exc:
        raise HTTPException(status_code=409, detail="Job identifier already exists.") from exc


@router.get("/jobs", response_model=JobListResponse)
def list_pipeline_jobs(
    state: StateDep,
    _: Any = ReviewerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    job_status: Annotated[
        str | None,
        Query(alias="status", pattern=r"^(queued|running|completed|failed|cancelled)$"),
    ] = None,
    observation_night_utc: date | None = None,
) -> JobListResponse:
    jobs = state.jobs.list_jobs(
        limit=limit,
        offset=offset,
        status_filter=job_status,
        observation_night_utc=observation_night_utc,
    )
    return JobListResponse(total_returned=len(jobs), jobs=[_job_response(job) for job in jobs])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def read_pipeline_job(
    job_id: SafeIdentifierPath,
    state: StateDep,
    _: Any = ReviewerDep,
) -> JobStatusResponse:
    job = state.jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
def cancel_pipeline_job(
    job_id: SafeIdentifierPath,
    request: Request,
    claims: Any = JobCancellerDep,
) -> JobStatusResponse:
    state: APIState = request.app.state.lumia_state
    job = state.jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in {"completed", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Job is already terminal.")
    cancelled = state.jobs.cancel_job(job_id)
    if cancelled is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    state.event_hub.publish_state(
        {
            "event_type": "job.state_changed",
            "payload": {
                "job_id": cancelled.job_id,
                "target_field_id": cancelled.target_field_id,
                "status": cancelled.status,
                "previous_status": job.status,
            },
        }
    )
    return _job_response(cancelled)


@router.post("/catalog/query", response_model=CatalogQueryResponse)
def execute_catalog_query(
    params: CatalogQueryParams,
    state: StateDep,
    _: Any = ReviewerDep,
) -> CatalogQueryResponse:
    return state.query_catalog(params)


@router.get("/settings", response_model=SettingsResponse)
def get_global_settings(request: Request) -> SettingsResponse:
    budget: ResourceBudgetConfig = (
        getattr(request.app.state, "resource_budget", None)
        or get_active_resource_budget()
        or ResourceBudgetConfig.from_env()
    )
    return SettingsResponse(
        archive_storage_path=budget.archive_storage_path,
        num_cpu_threads=budget.num_cpu_threads,
        max_gpu_memory_gb=budget.max_gpu_memory_gb,
        enable_openmp_threading_control=budget.enable_openmp_threading_control,
    )


@router.post("/settings", response_model=SettingsResponse)
@router.put("/settings", response_model=SettingsResponse)
def update_global_settings(
    payload: SettingsUpdateRequest,
    request: Request,
) -> SettingsResponse:
    current_budget: ResourceBudgetConfig = (
        getattr(request.app.state, "resource_budget", None)
        or get_active_resource_budget()
        or ResourceBudgetConfig.from_env()
    )

    updates: dict[str, Any] = {}
    if payload.archive_storage_path is not None:
        path_str = payload.archive_storage_path.strip()
        if not path_str:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="archive_storage_path cannot be empty.",
            )
        try:
            FilePath(path_str).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot create or access archive directory '{path_str}': {exc}",
            ) from exc
        updates["archive_storage_path"] = path_str

    if payload.num_cpu_threads is not None:
        updates["num_cpu_threads"] = payload.num_cpu_threads
    if payload.max_gpu_memory_gb is not None:
        updates["max_gpu_memory_gb"] = payload.max_gpu_memory_gb
    if payload.enable_openmp_threading_control is not None:
        updates["enable_openmp_threading_control"] = payload.enable_openmp_threading_control

    new_budget = current_budget.model_copy(update=updates)
    configure_global_runtime(new_budget)
    request.app.state.resource_budget = new_budget

    return SettingsResponse(
        archive_storage_path=new_budget.archive_storage_path,
        num_cpu_threads=new_budget.num_cpu_threads,
        max_gpu_memory_gb=new_budget.max_gpu_memory_gb,
        enable_openmp_threading_control=new_budget.enable_openmp_threading_control,
    )


router.include_router(websocket_router)

__all__ = ["APIState", "SafeIdentifierPath", "get_api_state", "router"]
