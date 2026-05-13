"""Agent-facing endpoints under `/api/agent/runs/{id}/*`."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from .auth import AgentScopedRun, get_db
from .db import (
    ACTIVE_STATUSES,
    STATUS_CANCELLED,
    STATUS_CLAIMED,
    STATUS_COMPLETED,
    STATUS_CREATED,
    STATUS_FAILED,
    STATUS_RUNNING,
    Database,
)
from .relay import ProgressRelay
from .schemas import (
    CancelCheckResponse,
    ClaimResponse,
    ErrorPost,
    LogPost,
    ModelInfoPost,
    ProgressPost,
    StageCompleteRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"], prefix="/agent")

_LOG_TAIL_LIMIT = 500
_ARTIFACT_FILENAMES = {
    "pairs": "pairs.jsonl",
    "svec": None,  # filename depends on run.behavior
    "calibration_result": "calibration_result.json",
    "calibration_checkpoint": "calibration_checkpoint.json",
    "run_meta": "run_meta.json",
}


def _relay(request: Request) -> ProgressRelay:
    return request.app.state.relay


# ── lifecycle ────────────────────────────────────────────────────

@router.post("/runs/{run_id}/claim", response_model=ClaimResponse)
async def claim(
    run_id: str,
    run: AgentScopedRun,
    db: Database = Depends(get_db),
) -> ClaimResponse:
    if run.status in (STATUS_COMPLETED, STATUS_CANCELLED, STATUS_FAILED):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Run is already {run.status}; cannot claim.",
        )
    if run.status == STATUS_CREATED or run.is_stale():
        await db.claim(run_id)
    return ClaimResponse(run_id=run_id, run_dir=run.run_dir, config=run.config)


@router.get("/runs/{run_id}/config")
async def agent_config(run: AgentScopedRun) -> dict:
    return run.config


@router.get("/runs/{run_id}/cancel-check", response_model=CancelCheckResponse)
async def cancel_check(run: AgentScopedRun) -> CancelCheckResponse:
    return CancelCheckResponse(cancel_requested=run.cancel_requested)


# ── progress + model info ────────────────────────────────────────

@router.post("/runs/{run_id}/progress")
async def post_progress(
    run_id: str,
    body: ProgressPost,
    request: Request,
    run: AgentScopedRun,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    progress = {
        "phase": body.phase,
        "completed": body.completed,
        "total": body.total,
        **body.payload,
    }
    await db.update_progress(run_id, progress)
    if run.status != STATUS_RUNNING:
        await db.update_status(run_id, status=STATUS_RUNNING, phase=body.phase)
    else:
        await db.update_status(run_id, phase=body.phase)
    await _relay(request).publish(run_id, {"event": "progress", **progress})
    return {"status": "ok"}


@router.post("/runs/{run_id}/model-info")
async def post_model_info(
    run_id: str,
    body: ModelInfoPost,
    request: Request,
    run: AgentScopedRun,  # noqa: ARG001 - used for auth
    db: Database = Depends(get_db),
) -> dict[str, str]:
    info = body.model_dump(exclude_none=False)
    await db.update_model_info(run_id, info)
    await _relay(request).publish(run_id, {"event": "model_info", **info})
    return {"status": "ok"}


# ── stage transitions ────────────────────────────────────────────

@router.post("/runs/{run_id}/stage/{stage}/start")
async def stage_start(
    run_id: str,
    stage: str,
    request: Request,
    run: AgentScopedRun,  # noqa: ARG001
    db: Database = Depends(get_db),
) -> dict[str, str]:
    await db.update_status(run_id, status=STATUS_RUNNING, phase=stage)
    await _relay(request).publish(run_id, {"event": "phase", "phase": stage})
    return {"status": "ok"}


@router.post("/runs/{run_id}/stage/{stage}/complete")
async def stage_complete(
    run_id: str,
    stage: str,
    body: StageCompleteRequest,
    request: Request,
    run: AgentScopedRun,  # noqa: ARG001
    db: Database = Depends(get_db),  # noqa: ARG001
) -> dict[str, str]:
    await _relay(request).publish(
        run_id, {"event": "stage_complete", "stage": stage, "notes": body.notes}
    )
    return {"status": "ok"}


@router.post("/runs/{run_id}/complete")
async def run_complete(
    run_id: str,
    request: Request,
    run: AgentScopedRun,  # noqa: ARG001
    db: Database = Depends(get_db),
) -> dict[str, str]:
    await db.update_status(
        run_id,
        status=STATUS_COMPLETED,
        completed_at=time.time(),
    )
    await _relay(request).publish(run_id, {"event": "phase", "phase": STATUS_COMPLETED})
    return {"status": "ok"}


@router.post("/runs/{run_id}/error")
async def run_error(
    run_id: str,
    body: ErrorPost,
    request: Request,
    run: AgentScopedRun,  # noqa: ARG001
    db: Database = Depends(get_db),
) -> dict[str, str]:
    final = STATUS_CANCELLED if run.cancel_requested else STATUS_FAILED
    await db.update_status(
        run_id,
        status=final,
        error=body.message,
        completed_at=time.time(),
    )
    await _relay(request).publish(
        run_id, {"event": "phase", "phase": final, "error": body.message}
    )
    return {"status": "ok"}


# ── artifacts ────────────────────────────────────────────────────

@router.post("/runs/{run_id}/artifacts/{name}")
async def upload_artifact(
    run_id: str,  # noqa: ARG001 - used via dep
    name: str,
    run: AgentScopedRun,
    file: UploadFile,
) -> dict[str, str]:
    if name not in _ARTIFACT_FILENAMES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown artifact '{name}'.")
    run_dir = Path(run.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    filename = _ARTIFACT_FILENAMES[name]
    if name == "svec":
        filename = f"{run.behavior}.svec"
    dest = run_dir / filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "ok", "path": str(dest)}


# ── log tail (optional, bounded) ─────────────────────────────────

@router.post("/runs/{run_id}/logs")
async def post_logs(
    run_id: str,
    body: LogPost,
    request: Request,
    run: AgentScopedRun,  # noqa: ARG001
) -> dict[str, str]:
    logs: dict[str, list[str]] = getattr(request.app.state, "run_logs", {})
    if not logs:
        request.app.state.run_logs = logs
    buf = logs.setdefault(run_id, [])
    buf.extend(body.lines)
    if len(buf) > _LOG_TAIL_LIMIT:
        del buf[: len(buf) - _LOG_TAIL_LIMIT]
    await _relay(request).publish(run_id, {"event": "log", "lines": body.lines})
    return {"status": "ok"}
