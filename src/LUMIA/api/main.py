"""FastAPI application factory and lifespan-managed Module 2 repositories."""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from LUMIA.api.routes import APIState, router
from LUMIA.api.schemas import APIErrorCode, APIErrorResponse
from LUMIA.core.concurrency import ResourceBudgetConfig, configure_global_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from LUMIA.data_pipeline import DuckDBConfig, OLTPDatabaseConfig

__all__ = ["create_app"]
_lazy_default_app: FastAPI | None = None


def create_app(
    db_config: DuckDBConfig | None = None,
    *,
    oltp_config: OLTPDatabaseConfig | None = None,
    allowed_origins: list[str] | None = None,
    jwt_secret: str | bytes | None = None,
    resource_budget: ResourceBudgetConfig | None = None,
) -> FastAPI:
    """Construct FastAPI with lifespan-injected Module 2 OLAP and OLTP repositories."""
    active_budget = configure_global_runtime(resource_budget)

    env_secret = os.environ.get("LUMIA_JWT_SECRET")
    if (
        jwt_secret is None
        and not env_secret
        and os.environ.get("LUMIA_ENV", "").lower() == "production"
    ):
        raise ValueError("LUMIA_JWT_SECRET must be configured in production.")
    secret = jwt_secret if jwt_secret is not None else env_secret or secrets.token_bytes(32)
    secret_bytes = secret.encode("utf-8") if isinstance(secret, str) else secret
    if len(secret_bytes) < 32:
        raise ValueError("LUMIA_JWT_SECRET must contain at least 32 bytes.")
    resolved_oltp_config = oltp_config
    oltp_url = os.environ.get("LUMIA_OLTP_DATABASE_URL")
    if resolved_oltp_config is None and oltp_url:
        from LUMIA.data_pipeline import OLTPDatabaseConfig as _OLTPDatabaseConfig

        resolved_oltp_config = _OLTPDatabaseConfig(url=oltp_url)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.resource_budget = active_budget
        api_state = APIState(
            db_config=db_config,
            oltp_config=resolved_oltp_config,
            jwt_secret=secret,
        )
        application.state.lumia_state = api_state
        try:
            yield
        finally:
            api_state.close()

    application = FastAPI(
        title="Project LUMIA API",
        description="JWT-secured REST and WebSocket API for Project LUMIA.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.resource_budget = active_budget

    @application.exception_handler(HTTPException)
    async def api_http_error_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        del request
        error_code_map: dict[int, APIErrorCode] = {
            400: "BAD_REQUEST",
            401: "AUTH_REQUIRED",
            403: "INSUFFICIENT_ROLE",
            404: "RESOURCE_NOT_FOUND",
            409: "STATE_CONFLICT",
            429: "RATE_LIMITED",
            503: "STORAGE_UNAVAILABLE",
        }
        error_code: APIErrorCode = error_code_map.get(exc.status_code, "BAD_REQUEST")
        payload = APIErrorResponse(
            error_code=error_code,
            detail=str(exc.detail),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
            headers=exc.headers,
        )

    origins = (
        list(allowed_origins)
        if allowed_origins is not None
        else ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    env_origins = os.environ.get("LUMIA_ALLOWED_ORIGINS")
    if env_origins:
        for o in env_origins.split(","):
            cleaned = o.strip()
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
    if any(origin.strip() == "*" for origin in origins):
        raise ValueError(
            "Wildcard '*' CORS origin is forbidden when allow_credentials=True; "
            "provide explicit allowed origins."
        )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type"],
    )
    application.include_router(router)
    return application


def __getattr__(name: str) -> FastAPI:
    """Lazily construct the default ASGI app only when requested."""
    global _lazy_default_app
    if name == "app":
        if _lazy_default_app is None:
            _lazy_default_app = create_app()
        return _lazy_default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
