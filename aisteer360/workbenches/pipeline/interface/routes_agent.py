"""Agent-facing endpoints under `/api/agent/sessions/{id}/*` for the pipeline workbench."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Query, Request

from aisteer360.workbenches.common.interface.auth import AgentScopedSession, get_db
from aisteer360.workbenches.common.interface.db import (
    Database,
    STATUS_CANCELLED,
    STATUS_CLAIMED,
    STATUS_CLOSED,
    STATUS_CLOSING,
    STATUS_FAILED,
    STATUS_READY,
)
from aisteer360.workbenches.common.interface.relay import ProgressRelay, RequestRelay

from .schemas import (
    SessionClaimResponse,
    SessionErrorPost,
    SessionPollResponse,
    SessionReadyPost,
    SessionResultPost,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["session-agent"], prefix="/agent")


def _progress(request: Request) -> ProgressRelay:
    return request.app.state.relay


def _requests(request: Request) -> RequestRelay:
    return request.app.state.request_relay


@router.post("/sessions/{session_id}/claim", response_model=SessionClaimResponse)
async def claim(
    session_id: str,
    session: AgentScopedSession,
    db: Database = Depends(get_db),
) -> SessionClaimResponse:
    if session.status not in (STATUS_CLOSED, STATUS_FAILED, STATUS_CANCELLED):
        await db.claim_session(session_id)
    provider_keys = await db.get_secrets(session.owner_token_hash)
    return SessionClaimResponse(
        session_id=session_id,
        model_name_or_path=session.model_name,
        config=session.config,
        provider_keys=provider_keys,
        idle_timeout_s=session.idle_timeout_s,
    )


@router.post("/sessions/{session_id}/ready")
async def ready(
    session_id: str,
    body: SessionReadyPost,
    request: Request,
    session: AgentScopedSession,  # noqa: ARG001 - used for auth
    db: Database = Depends(get_db),
) -> dict[str, str]:
    await db.update_session_status(
        session_id, status=STATUS_READY, model_info=body.model_info,
    )
    await _progress(request).publish(
        session_id, {"event": "ready", "model_info": body.model_info},
    )
    return {"status": "ok"}


@router.get("/sessions/{session_id}/poll", response_model=SessionPollResponse)
async def poll(
    session_id: str,
    session: AgentScopedSession,
    request: Request,
    timeout: float = Query(30.0, ge=1.0, le=120.0),
) -> SessionPollResponse:
    """Long-poll for the next inference request or close signal.

    Returns immediately with a close signal when the browser has marked the session as `closing`.
    Otherwise blocks up to `timeout` seconds for a queued request. On timeout returns an empty
    response so the agent can heartbeat and re-poll.
    """
    if session.status in (STATUS_CLOSING, STATUS_CLOSED, STATUS_FAILED, STATUS_CANCELLED):
        return SessionPollResponse(close=True)
    req = await _requests(request).poll(session_id, timeout_s=timeout)
    if req is None:
        return SessionPollResponse()
    return SessionPollResponse(request=req)


@router.post("/sessions/{session_id}/result")
async def result(
    session_id: str,
    body: SessionResultPost,
    request: Request,
    session: AgentScopedSession,  # noqa: ARG001
) -> dict[str, str]:
    event: dict[str, object] = {
        "event": "inference_result",
        "request_id": body.request_id,
        "generated_text": body.generated_text,
        "elapsed_ms": body.elapsed_ms,
        "pipeline_hash": body.pipeline_hash,
    }
    if body.error:
        event["error"] = body.error
    await _progress(request).publish(session_id, event)
    return {"status": "ok"}


@router.post("/sessions/{session_id}/heartbeat")
async def heartbeat(
    session_id: str,
    session: AgentScopedSession,  # noqa: ARG001 - heartbeat refreshed in dependency
) -> dict[str, str]:
    return {"status": "ok"}


@router.post("/sessions/{session_id}/error")
async def error(
    session_id: str,
    body: SessionErrorPost,
    request: Request,
    session: AgentScopedSession,  # noqa: ARG001
    db: Database = Depends(get_db),
) -> dict[str, str]:
    await db.update_session_status(session_id, status=STATUS_FAILED, error=body.message)
    await _progress(request).publish(
        session_id, {"event": "phase", "phase": STATUS_FAILED, "error": body.message},
    )
    return {"status": "ok"}


@router.post("/sessions/{session_id}/close")
async def close(
    session_id: str,
    request: Request,
    session: AgentScopedSession,  # noqa: ARG001
    db: Database = Depends(get_db),
) -> dict[str, str]:
    """Agent-initiated close (e.g. idle timeout fired)."""
    await db.update_session_status(session_id, status=STATUS_CLOSED)
    await _progress(request).publish(
        session_id, {"event": "phase", "phase": STATUS_CLOSED},
    )
    await _requests(request).clear(session_id)
    return {"status": "ok"}
