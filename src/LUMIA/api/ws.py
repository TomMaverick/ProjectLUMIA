"""Authenticated WebSocket job snapshots and bounded per-client event queues."""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import Field, ValidationError

from LUMIA.api.schemas import (
    ErrorEnvelope,
    ErrorPayload,
    HeartbeatEnvelope,
    HeartbeatPayload,
    JobMetricsUpdatedEnvelope,
    JobMetricsUpdatedPayload,
    JobSnapshotItem,
    JobStateChangedEnvelope,
    JobStateChangedPayload,
    SnapshotEnvelope,
    SnapshotPayload,
    TaskLogAppendedEnvelope,
    TaskLogAppendedPayload,
    TaskStateChangedEnvelope,
    TaskStateChangedPayload,
    WebSocketEnvelope,
)
from LUMIA.core.schemas import LumiaBaseModel

if TYPE_CHECKING:
    from LUMIA.data_pipeline.oltp_repositories import OperatorProfile

router = APIRouter(tags=["lumia-web-ui"])
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class SubscribeCommand(LumiaBaseModel):
    action: Literal["subscribe"]
    job_id: str | None = Field(default=None, min_length=1, max_length=128)
    min_log_severity: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class PingCommand(LumiaBaseModel):
    action: Literal["ping"]
    client_sent_at_utc: str | None = None


class _Subscription:
    def __init__(self, capacity: int) -> None:
        self.queue: queue.Queue[tuple[dict[str, object], bool]] = queue.Queue(capacity)
        self.dropped_log_frames = 0
        self.closed = False


class EventHub:
    """Bounded fan-out with expendable task logs taking priority over state frames."""

    def __init__(self, queue_capacity: int = 256) -> None:
        self.queue_capacity = queue_capacity
        self._subscriptions: set[_Subscription] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> _Subscription:
        subscription = _Subscription(self.queue_capacity)
        with self._lock:
            self._subscriptions.add(subscription)
        return subscription

    def unsubscribe(self, subscription: _Subscription) -> None:
        with self._lock:
            self._subscriptions.discard(subscription)

    def publish_log(self, event: dict[str, object]) -> None:
        self._publish(event, is_log=True)

    def publish_state(self, event: dict[str, object]) -> None:
        self._publish(event, is_log=False)

    def _publish(self, event: dict[str, object], *, is_log: bool) -> None:
        with self._lock:
            subscribers = tuple(self._subscriptions)
        for subscriber in subscribers:
            if subscriber.closed:
                continue
            try:
                subscriber.queue.put_nowait((event, is_log))
                continue
            except queue.Full:
                pass
            if is_log:
                subscriber.dropped_log_frames += 1
                continue
            retained: list[tuple[dict[str, object], bool]] = []
            while True:
                try:
                    old_event = subscriber.queue.get_nowait()
                except queue.Empty:
                    break
                if old_event[1]:
                    subscriber.dropped_log_frames += 1
                else:
                    retained.append(old_event)
            for old_event in retained:
                subscriber.queue.put_nowait(old_event)
            try:
                subscriber.queue.put_nowait((event, False))
            except queue.Full:
                subscriber.closed = True


async def _next_item(
    subscription: _Subscription,
    timeout: float,
) -> tuple[dict[str, object], bool] | None:
    try:
        return await asyncio.to_thread(subscription.queue.get, True, timeout)
    except queue.Empty:
        return None


def _snapshot(application: Any, job_id: str | None) -> SnapshotEnvelope:
    state = application.state.lumia_state
    jobs = state.jobs.active_jobs(job_id)
    return SnapshotEnvelope(
        sequence_no=1,
        payload=SnapshotPayload(
            subscribed_job_id=job_id,
            active_jobs=[
                JobSnapshotItem(
                    job_id=job.job_id,
                    target_field_id=job.target_field_id,
                    observation_night_utc=job.observation_night_utc,
                    status=job.status,
                    matched_catalog_sources=job.matched_catalog_sources,
                    brightest_catalog_mag_g=job.brightest_catalog_mag_g,
                    submitted_at_utc=job.submitted_at_utc,
                    started_at_utc=job.started_at_utc,
                    completed_at_utc=job.completed_at_utc,
                )
                for job in jobs
            ],
        ),
    )


def _ws_profile(websocket: WebSocket) -> tuple[OperatorProfile | None, int]:
    import os
    from uuid import UUID
    from LUMIA.data_pipeline.oltp_repositories import OperatorProfile

    auth_mode = os.environ.get("LUMIA_AUTH_MODE", "jwt").lower()
    env = os.environ.get("LUMIA_ENV", "").lower()
    if auth_mode == "bypass" and env != "production":
        return (
            OperatorProfile(
                operator_id=UUID("00000000-0000-0000-0000-000000000001"),
                username="Local Bypass",
                email="local-bypass@lumia.local",
                role="admin",
                is_active=True,
            ),
            200,
        )

    state = websocket.app.state.lumia_state
    ticket = websocket.query_params.get("ticket")
    if ticket:
        ticket_profile = state.ws_tickets.consume(ticket)
        if ticket_profile is None:
            return None, 4401
        profile = state.auth.get_active_operator(ticket_profile.operator_id)
        return (profile, 4401 if profile is not None else 4403)
    scheme, _, token = websocket.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None, 4401
    try:
        claims = state.jwt_codec.decode(token, expected_use="access")
    except ValueError:
        return None, 4401
    profile = state.auth.get_active_operator(claims.sub)
    return (profile, 4401 if profile is not None else 4403)


@router.websocket("/ws/jobs")
async def stream_jobs(websocket: WebSocket) -> None:
    """Authenticate and stream a snapshot followed by bounded job lifecycle updates."""
    job_id = websocket.query_params.get("job_id")
    if job_id is not None and _SAFE_IDENTIFIER.fullmatch(job_id) is None:
        await websocket.close(code=4401)
        return
    profile, failure_code = _ws_profile(websocket)
    if profile is None:
        await websocket.close(code=failure_code)
        return
    await websocket.accept()
    sequence = 1
    await websocket.send_json(_snapshot(websocket.app, job_id).model_dump(mode="json"))
    subscription = websocket.app.state.lumia_state.event_hub.subscribe()
    receive_task = asyncio.create_task(websocket.receive_text())
    last_heartbeat = time.monotonic()
    min_log_severity = "INFO"
    severity_rank = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    try:
        while not subscription.closed:
            item = await _next_item(subscription, 0.25)
            if receive_task.done():
                command_text = receive_task.result()
                try:
                    command = json.loads(command_text)
                    if command.get("action") == "subscribe":
                        parsed = SubscribeCommand.model_validate(command)
                        if (
                            parsed.job_id is not None
                            and _SAFE_IDENTIFIER.fullmatch(parsed.job_id) is None
                        ):
                            raise ValueError("Invalid job identifier.")
                        job_id = parsed.job_id
                        min_log_severity = parsed.min_log_severity
                        sequence += 1
                        await websocket.send_json(
                            _snapshot(websocket.app, job_id)
                            .model_copy(update={"sequence_no": sequence})
                            .model_dump(mode="json")
                        )
                    elif command.get("action") == "ping":
                        PingCommand.model_validate(command)
                        sequence += 1
                        await websocket.send_json(
                            HeartbeatEnvelope(
                                sequence_no=sequence,
                                payload=HeartbeatPayload(
                                    dropped_log_frames=subscription.dropped_log_frames
                                ),
                            ).model_dump(mode="json")
                        )
                        subscription.dropped_log_frames = 0
                    else:
                        raise ValueError("Unknown WebSocket action.")
                except (ValueError, ValidationError, AttributeError):
                    sequence += 1
                    await websocket.send_json(
                        ErrorEnvelope(
                            sequence_no=sequence,
                            payload=ErrorPayload(
                                error_code="INVALID_WS_COMMAND",
                                message="Malformed or unsupported WebSocket command.",
                            ),
                        ).model_dump(mode="json")
                    )
                receive_task = asyncio.create_task(websocket.receive_text())
            if item is None:
                if time.monotonic() - last_heartbeat >= 25.0:
                    sequence += 1
                    heartbeat = HeartbeatEnvelope(
                        sequence_no=sequence,
                        payload=HeartbeatPayload(
                            dropped_log_frames=subscription.dropped_log_frames
                        ),
                    )
                    subscription.dropped_log_frames = 0
                    await websocket.send_json(heartbeat.model_dump(mode="json"))
                    last_heartbeat = time.monotonic()
                continue
            event, is_log = item
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            payload_job_id = payload.get("job_id")
            if job_id is not None and payload_job_id != job_id:
                continue
            if is_log and severity_rank.get(str(payload.get("severity")), -1) < severity_rank[
                min_log_severity
            ]:
                continue
            sequence += 1
            frame: WebSocketEnvelope
            if event_type == "job.state_changed":
                frame = JobStateChangedEnvelope(
                    sequence_no=sequence,
                    payload=JobStateChangedPayload.model_validate(payload),
                )
            elif event_type == "task.state_changed":
                frame = TaskStateChangedEnvelope(
                    sequence_no=sequence,
                    payload=TaskStateChangedPayload.model_validate(payload),
                )
            elif event_type == "task.log_appended":
                frame = TaskLogAppendedEnvelope(
                    sequence_no=sequence,
                    payload=TaskLogAppendedPayload.model_validate(payload),
                )
            elif event_type == "job.metrics_updated":
                frame = JobMetricsUpdatedEnvelope(
                    sequence_no=sequence,
                    payload=JobMetricsUpdatedPayload.model_validate(payload),
                )
            else:
                frame = ErrorEnvelope(
                    sequence_no=sequence,
                    payload=ErrorPayload(
                        error_code="UNKNOWN_EVENT",
                        message="An unsupported server event was discarded.",
                    ),
                )
            await websocket.send_json(frame.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        receive_task.cancel()
        websocket.app.state.lumia_state.event_hub.unsubscribe(subscription)
    if subscription.closed:
        await websocket.close(code=1013)


__all__ = ["EventHub", "router"]
