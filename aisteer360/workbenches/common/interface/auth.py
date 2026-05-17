"""Token-based auth dependencies for browser and agent routes.

Browser traffic carries `Authorization: Bearer dt-<hex>`; the server stores `sha256(token)` as
`owner_token_hash` on every run row. Lookup is by hash, so the SPA can generate its token locally
on first load without any registration step.

Agent traffic carries `Authorization: Bearer sk-run-<hex>`; the server stores `bcrypt(token)` per
run row and validates with `bcrypt.checkpw`. Each POST also refreshes `last_heartbeat`.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, WebSocket, status

from .db import Database, Run, Session, sha256_hex, verify_agent_token

logger = logging.getLogger(__name__)

AUTH_SCHEME = "bearer"


def _extract_bearer(header: str | None) -> str | None:
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != AUTH_SCHEME:
        return None
    token = parts[1].strip()
    return token or None


def get_db(request: Request) -> Database:
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise RuntimeError("Database is not initialised on app.state.db")
    return db


async def owner_token_hash(request: Request) -> str:
    """Extract and hash the browser's bearer token. Raises 401 if missing."""
    raw = _extract_bearer(request.headers.get("authorization"))
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing owner token.")
    return sha256_hex(raw)


OwnerTokenHash = Annotated[str, Depends(owner_token_hash)]


async def owner_scoped_run(
    run_id: str,
    db: Annotated[Database, Depends(get_db)],
    owner_hash: OwnerTokenHash,
) -> Run:
    """Load a run and verify its `owner_token_hash` matches the caller. 404 when not owned."""
    run = await db.get_run(run_id)
    if not run or run.owner_token_hash != owner_hash:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found.")
    return run


OwnerScopedRun = Annotated[Run, Depends(owner_scoped_run)]


async def agent_scoped_run(
    run_id: str,
    request: Request,
    db: Annotated[Database, Depends(get_db)],
) -> Run:
    """Validate a run-scoped agent token and refresh heartbeat."""
    raw = _extract_bearer(request.headers.get("authorization"))
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing agent token.")
    run = await db.get_run(run_id)
    if not run or not verify_agent_token(raw, run.agent_token_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid agent token.")
    await db.heartbeat(run_id)
    # re-fetch so callers see the updated heartbeat
    refreshed = await db.get_run(run_id)
    return refreshed or run


AgentScopedRun = Annotated[Run, Depends(agent_scoped_run)]


async def owner_scoped_session(
    session_id: str,
    db: Annotated[Database, Depends(get_db)],
    owner_hash: OwnerTokenHash,
) -> Session:
    """Load a session and verify its `owner_token_hash` matches the caller. 404 when not owned."""
    session = await db.get_session(session_id)
    if not session or session.owner_token_hash != owner_hash:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")
    return session


OwnerScopedSession = Annotated[Session, Depends(owner_scoped_session)]


async def agent_scoped_session(
    session_id: str,
    request: Request,
    db: Annotated[Database, Depends(get_db)],
) -> Session:
    """Validate a session-scoped agent token and refresh heartbeat."""
    raw = _extract_bearer(request.headers.get("authorization"))
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing agent token.")
    session = await db.get_session(session_id)
    if not session or not verify_agent_token(raw, session.agent_token_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid agent token.")
    await db.session_heartbeat(session_id)
    refreshed = await db.get_session(session_id)
    return refreshed or session


AgentScopedSession = Annotated[Session, Depends(agent_scoped_session)]


async def authorise_ws_owner(
    run_id: str,
    websocket: WebSocket,
    db: Database,
) -> Run | None:
    """Validate a browser WebSocket connection against the owner token query param.

    Returns the run on success. On failure, closes the socket with code 4401 and returns None.
    """
    token = websocket.query_params.get("token") or _extract_bearer(
        websocket.headers.get("authorization")
    )
    if not token:
        await websocket.close(code=4401, reason="missing token")
        return None
    run = await db.get_run(run_id)
    if not run or run.owner_token_hash != sha256_hex(token):
        await websocket.close(code=4401, reason="unauthorized")
        return None
    return run


async def authorise_ws_session_owner(
    session_id: str,
    websocket: WebSocket,
    db: Database,
) -> Session | None:
    """Validate a browser WebSocket connection against the session owner token query param.

    Returns the session on success. On failure, closes the socket with code 4401 and returns None.
    """
    token = websocket.query_params.get("token") or _extract_bearer(
        websocket.headers.get("authorization")
    )
    if not token:
        await websocket.close(code=4401, reason="missing token")
        return None
    session = await db.get_session(session_id)
    if not session or session.owner_token_hash != sha256_hex(token):
        await websocket.close(code=4401, reason="unauthorized")
        return None
    return session


__all__ = [
    "get_db",
    "owner_token_hash",
    "OwnerTokenHash",
    "owner_scoped_run",
    "OwnerScopedRun",
    "agent_scoped_run",
    "AgentScopedRun",
    "owner_scoped_session",
    "OwnerScopedSession",
    "agent_scoped_session",
    "AgentScopedSession",
    "authorise_ws_owner",
    "authorise_ws_session_owner",
]
