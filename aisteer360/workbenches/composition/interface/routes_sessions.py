"""Browser-facing session endpoints scoped by owner token hash."""
from __future__ import annotations

import logging
import shlex
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from aisteer360.workbenches.common.interface.auth import (
    OwnerScopedSession,
    OwnerTokenHash,
    get_db,
)
from aisteer360.workbenches.common.interface.db import (
    Database,
    STATUS_CANCELLED,
    STATUS_CLOSED,
    STATUS_CLOSING,
    STATUS_FAILED,
    hash_agent_token,
    mint_agent_token,
    mint_session_id,
)
from aisteer360.workbenches.common.interface.dispatch import dispatch_local, dispatch_ssh
from aisteer360.workbenches.common.interface.relay import ProgressRelay, RequestRelay

from .schemas import (
    AgentCommand,
    InferenceAcceptedResponse,
    InferenceRequest,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDetail,
    SessionListResponse,
    SessionSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])

_REMOTE_AGENT_MODULE = "aisteer360.workbenches.composition.agent"
_LOG_LABEL = "aisteer360-compose"


def _public_server_url(request: Request) -> str:
    override = getattr(request.app.state, "public_server_url", None)
    if override:
        return override.rstrip("/")
    return str(request.base_url).rstrip("/")


def _agent_command(request: Request, session_id: str, agent_token: str) -> AgentCommand:
    name = getattr(request.app.state, "agent_command_name", "aisteer360-compose-agent")
    server = _public_server_url(request)
    parts = [name, "--server", server, "--session-id", session_id, "--agent-token", agent_token]
    command = " ".join(shlex.quote(p) for p in parts)
    return AgentCommand(
        command=command, server=server, session_id=session_id, agent_token=agent_token,
    )


async def _dispatch_agent(
    request: Request,
    db: Database,
    *,
    session_id: str,
    cmd: AgentCommand,
    owner_hash: str,
) -> tuple[str, str | None]:
    name = getattr(request.app.state, "agent_command_name", "aisteer360-compose-agent")
    agent_argv = [
        name,
        "--server", cmd.server,
        "--session-id", cmd.session_id,
        "--agent-token", cmd.agent_token,
    ]
    compute = await db.get_compute_config(owner_hash)
    mode = compute.get("mode") if compute else None
    solo = getattr(request.app.state, "solo_mode", False)

    if mode == "ssh":
        try:
            dispatch_ssh(
                compute, agent_argv,
                remote_module=_REMOTE_AGENT_MODULE,
                log_label=_LOG_LABEL,
            )
            return "ssh", None
        except Exception as exc:
            logger.warning("SSH dispatch failed for %s: %s", session_id, exc)
            return "failed", str(exc)
    if mode == "local" or solo:
        try:
            proc = dispatch_local(agent_argv)
            request.app.state.local_agents[session_id] = proc
            return "local", None
        except Exception as exc:
            logger.warning("Local dispatch failed for %s: %s", session_id, exc)
            return "failed", str(exc)
    return "manual", None


def _session_detail(session) -> SessionDetail:
    summary = session.to_summary()
    return SessionDetail(**summary, config=session.config)


# ── create / list / get ──────────────────────────────────────────

@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(
    body: SessionCreateRequest,
    request: Request,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> SessionCreateResponse:
    if not body.model_name_or_path.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model_name_or_path is required.")

    session_id = mint_session_id()
    agent_token = mint_agent_token()
    config = {
        "model_name_or_path": body.model_name_or_path.strip(),
        "hf_model_kwargs": dict(body.hf_model_kwargs),
        "device_map": body.device_map,
    }
    session = await db.create_session(
        session_id=session_id,
        model_name=config["model_name_or_path"],
        config=config,
        owner_token_hash=owner_hash,
        agent_token_hash=hash_agent_token(agent_token),
        idle_timeout_s=body.idle_timeout_s,
    )
    cmd = _agent_command(request, session_id, agent_token)
    dispatch_status, dispatch_error = await _dispatch_agent(
        request, db, session_id=session_id, cmd=cmd, owner_hash=owner_hash,
    )
    return SessionCreateResponse(
        session=_session_detail(session),
        agent_token=agent_token,
        agent_command=cmd,
        dispatch_status=dispatch_status,
        dispatch_error=dispatch_error,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> SessionListResponse:
    rows = await db.list_sessions_for_owner(owner_hash)
    return SessionListResponse(sessions=[SessionSummary(**r.to_summary()) for r in rows])


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session_detail(session: OwnerScopedSession) -> SessionDetail:
    return _session_detail(session)


# ── inference ────────────────────────────────────────────────────

@router.post(
    "/sessions/{session_id}/infer",
    response_model=InferenceAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_inference(
    session_id: str,
    body: InferenceRequest,
    request: Request,
    session: OwnerScopedSession,  # noqa: ARG001 - used for auth
) -> InferenceAcceptedResponse:
    if session.status in (STATUS_CLOSED, STATUS_FAILED, STATUS_CANCELLED, STATUS_CLOSING):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Session is {session.status}; cannot accept new inference requests.",
        )
    relay: RequestRelay = request.app.state.request_relay
    payload: dict[str, Any] = body.model_dump()
    if payload.get("request_id") is None:
        payload.pop("request_id", None)
    request_id = await relay.push(session_id, payload)
    return InferenceAcceptedResponse(request_id=request_id)


@router.post("/sessions/{session_id}/cancel-infer")
async def cancel_inference(
    session_id: str,
    request: Request,
    session: OwnerScopedSession,  # noqa: ARG001
) -> dict[str, str]:
    relay: RequestRelay = request.app.state.request_relay
    await relay.clear(session_id)
    return {"status": "ok"}


# ── lifecycle ────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/close")
async def close_session(
    session_id: str,
    request: Request,
    session: OwnerScopedSession,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    if session.status in (STATUS_CLOSED, STATUS_FAILED, STATUS_CANCELLED):
        return {"status": session.status}
    await db.update_session_status(session_id, status=STATUS_CLOSING)
    progress: ProgressRelay = request.app.state.relay
    await progress.publish(session_id, {"event": "phase", "phase": STATUS_CLOSING})
    return {"status": "closing"}


@router.delete("/sessions/{session_id}")
async def force_kill_session(
    session_id: str,
    request: Request,
    session: OwnerScopedSession,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    agents = getattr(request.app.state, "local_agents", {})
    proc = agents.pop(session_id, None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        logger.info("Terminated local session agent for %s (pid %d)", session_id, proc.pid)
    if session.status not in (STATUS_CLOSED, STATUS_FAILED, STATUS_CANCELLED):
        await db.update_session_status(
            session_id, status=STATUS_CANCELLED, error="terminated by owner",
        )
    relay: RequestRelay = request.app.state.request_relay
    await relay.clear(session_id)
    return {"status": "killed"}
