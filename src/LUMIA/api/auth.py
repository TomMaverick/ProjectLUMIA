"""JWT encoding, strict claims, and authentication dependencies for Module 10."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, Request, status
from pydantic import Field, SecretStr

from LUMIA.core.schemas import LumiaBaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from LUMIA.data_pipeline.oltp_repositories import OperatorProfile

ACCESS_TTL_SECONDS = 900
REFRESH_TTL_SECONDS = 604800
WS_TICKET_TTL_SECONDS = 30


class LoginRequest(LumiaBaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: SecretStr = Field(min_length=8, max_length=256)


class OperatorProfileResponse(LumiaBaseModel):
    operator_id: UUID
    username: str
    email: str
    role: Literal["admin", "astronomer", "reviewer", "service"]
    is_active: bool

    @classmethod
    def from_profile(cls, profile: OperatorProfile) -> OperatorProfileResponse:
        return cls(**profile.model_dump())


class AccessTokenResponse(LumiaBaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = ACCESS_TTL_SECONDS
    operator: OperatorProfileResponse


class WebSocketTicketResponse(LumiaBaseModel):
    ticket: str
    expires_in: int = WS_TICKET_TTL_SECONDS


class JWTClaims(LumiaBaseModel):
    iss: Literal["lumia-api"]
    aud: Literal["lumia-web-ui"]
    sub: UUID
    username: str
    role: Literal["admin", "astronomer", "reviewer", "service"]
    token_use: Literal["access", "refresh"]
    jti: UUID
    iat: int
    nbf: int
    exp: int


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class JWTCodec:
    """Minimal HS256 JWS codec with fixed issuer/audience and strict claim validation."""

    def __init__(self, secret: str | bytes) -> None:
        key = secret.encode("utf-8") if isinstance(secret, str) else secret
        if len(key) < 32:
            raise ValueError("JWT signing secret must contain at least 32 bytes.")
        self._key = key

    def issue(
        self,
        profile: OperatorProfile,
        token_use: Literal["access", "refresh"],
        *,
        ttl_seconds: int,
        now: int | None = None,
    ) -> tuple[str, JWTClaims]:
        issued_at = int(time.time()) if now is None else now
        claims = JWTClaims(
            iss="lumia-api",
            aud="lumia-web-ui",
            sub=profile.operator_id,
            username=profile.username,
            role=profile.role,
            token_use=token_use,
            jti=uuid4(),
            iat=issued_at,
            nbf=issued_at,
            exp=issued_at + ttl_seconds,
        )
        header = _b64encode(b'{"alg":"HS256","typ":"JWT"}')
        payload = _b64encode(
            json.dumps(
                claims.model_dump(mode="json"),
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        signing_input = f"{header}.{payload}".encode("ascii")
        signature = _b64encode(hmac.new(self._key, signing_input, hashlib.sha256).digest())
        return f"{header}.{payload}.{signature}", claims

    def decode(
        self,
        token: str,
        *,
        expected_use: Literal["access", "refresh"],
        now: int | None = None,
    ) -> JWTClaims:
        try:
            encoded_header, encoded_payload, encoded_signature = token.split(".")
            header = json.loads(_b64decode(encoded_header))
            if header != {"alg": "HS256", "typ": "JWT"}:
                raise ValueError("Unexpected token algorithm or type.")
            signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
            expected_signature = hmac.new(self._key, signing_input, hashlib.sha256).digest()
            if not hmac.compare_digest(_b64decode(encoded_signature), expected_signature):
                raise ValueError("Invalid token signature.")
            claims = JWTClaims.model_validate_json(_b64decode(encoded_payload))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid bearer token.") from exc
        current_time = int(time.time()) if now is None else now
        if (
            claims.token_use != expected_use
            or claims.iss != "lumia-api"
            or claims.aud != "lumia-web-ui"
            or claims.nbf > current_time
            or claims.exp <= current_time
        ):
            raise ValueError("Expired or invalid bearer token.")
        return claims


class WebSocketTicketStore:
    """Thread-safe, expiring, single-use opaque WebSocket handshake tickets."""

    def __init__(self) -> None:
        self._tickets: dict[str, tuple[OperatorProfile, float]] = {}
        self._lock = threading.Lock()

    def issue(self, profile: OperatorProfile) -> str:
        now = time.monotonic()
        ticket = f"wst_{secrets.token_urlsafe(32)}"
        with self._lock:
            self._tickets = {
                key: value for key, value in self._tickets.items() if value[1] > now
            }
            self._tickets[ticket] = (profile, now + WS_TICKET_TTL_SECONDS)
        return ticket

    def consume(self, ticket: str) -> OperatorProfile | None:
        with self._lock:
            value = self._tickets.pop(ticket, None)
        if value is None or value[1] <= time.monotonic():
            return None
        return value[0]


class LoginRateLimiter:
    """Bounded in-process sliding-window protection for credential guessing."""

    def __init__(self, *, limit: int = 5, window_seconds: int = 60) -> None:
        self._limit = limit
        self._window = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allowed(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            attempts = self._failures[key]
            while attempts and attempts[0] <= now - self._window:
                attempts.popleft()
            return len(attempts) < self._limit

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            attempts = self._failures[key]
            while attempts and attempts[0] <= now - self._window:
                attempts.popleft()
            attempts.append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


def get_bearer_claims(request: Request) -> JWTClaims:
    import os

    auth_mode = os.environ.get("LUMIA_AUTH_MODE", "jwt").lower()
    env = os.environ.get("LUMIA_ENV", "").lower()
    if auth_mode == "bypass" and env != "production":
        now = int(time.time())
        return JWTClaims(
            iss="lumia-api",
            aud="lumia-web-ui",
            sub=UUID("00000000-0000-0000-0000-000000000001"),
            username="Local Bypass",
            role="admin",
            token_use="access",
            jti=uuid4(),
            iat=now,
            nbf=now,
            exp=now + 86400,
        )

    state = request.app.state.lumia_state
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims: JWTClaims = state.jwt_codec.decode(token, expected_use="access")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    profile = state.auth.get_active_operator(claims.sub)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Operator account is inactive or unavailable.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return claims


ClaimsDep = Annotated[JWTClaims, Depends(get_bearer_claims)]


def require_roles(*roles: str) -> Callable[[JWTClaims], JWTClaims]:
    """Build a dependency enforcing the allowed authenticated operator roles."""

    def dependency(claims: ClaimsDep) -> JWTClaims:
        if claims.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role for this operation.",
            )
        return claims

    return dependency


__all__ = [
    "ACCESS_TTL_SECONDS",
    "REFRESH_TTL_SECONDS",
    "WS_TICKET_TTL_SECONDS",
    "AccessTokenResponse",
    "JWTClaims",
    "JWTCodec",
    "LoginRateLimiter",
    "LoginRequest",
    "OperatorProfileResponse",
    "WebSocketTicketResponse",
    "WebSocketTicketStore",
    "get_bearer_claims",
    "require_roles",
]
