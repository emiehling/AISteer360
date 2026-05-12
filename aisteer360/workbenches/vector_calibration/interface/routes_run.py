"""Pipeline execution endpoints."""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from .schemas import RunRequest, RunStatusResponse
from .state import RunPhase, ServerState
from .ws import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["run"])


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        return 0
    return count


def _read_first_jsonl_record(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return None
    except OSError:
        return None
    return None


def _pipeline_task(
    state: ServerState,
    manager: ConnectionManager,
    stages: list[str],
    resume_run_dir: str | None = None,
):
    """Coroutine that runs the requested stages in background threads."""

    async def _run():
        async with state.run_lock:
            state.reset_cancel()
            state.run_status.phase = RunPhase.IDLE
            state.run_status.progress = {}
            state.run_status.error = None
            state.run_status.started_at = time.time()
            state.run_status.finished_at = None

            try:
                if state.builder is None:
                    state.rebuild_builder()

                if "generation" in stages and not state.config.generation.behavior.strip():
                    raise ValueError(
                        "Behavior label is required for generation. Set it in the 'Dimension label' field."
                    )

                if "generation" in stages:
                    state.run_status.phase = RunPhase.GENERATION
                    manager.broadcast("phase", {"phase": "generation"})

                    def gen_progress(done: int, total: int) -> None:
                        state.run_status.progress = {
                            "completed": done,
                            "total": total,
                        }
                        manager.broadcast(
                            "progress",
                            {
                                "phase": "generation",
                                "completed": done,
                                "total": total,
                            },
                        )

                    run_dir_override: Path | None = None
                    if resume_run_dir:
                        candidate = state.save_dir / resume_run_dir
                        if not candidate.is_dir():
                            raise ValueError(
                                f"Resume run directory '{resume_run_dir}' not found under {state.save_dir}."
                            )
                        run_dir_override = candidate

                    result = await asyncio.to_thread(
                        state.builder.run_generation,
                        on_progress=gen_progress,
                        cancel_check=lambda: state.is_cancel_requested,
                        run_dir=run_dir_override,
                    )
                    state.generation_result = result

                    if state.is_cancel_requested:
                        state.run_status.phase = RunPhase.CANCELLED
                        state.run_status.finished_at = time.time()
                        manager.broadcast("phase", {"phase": "cancelled"})
                        return

                if "extraction" in stages:
                    state.run_status.phase = RunPhase.EXTRACTION
                    manager.broadcast("phase", {"phase": "extraction"})

                    pairs = (
                        state.generation_result.pairs
                        if state.generation_result
                        else None
                    )
                    sv = await asyncio.to_thread(
                        state.builder.run_extraction, pairs=pairs
                    )
                    state.steering_vector = sv
                    state.model_info = state.extract_model_info()

                if state.is_cancel_requested:
                    state.run_status.phase = RunPhase.CANCELLED
                    state.run_status.finished_at = time.time()
                    manager.broadcast("phase", {"phase": "cancelled"})
                    return

                if "calibration" in stages:
                    state.run_status.phase = RunPhase.CALIBRATION
                    manager.broadcast("phase", {"phase": "calibration"})

                    def cal_progress(data: dict) -> None:
                        state.run_status.progress = data
                        manager.broadcast(
                            "progress",
                            {"phase": "calibration", **data},
                        )
                        if "current_cell" in data:
                            manager.broadcast(
                                "cell_complete", dict(data["current_cell"])
                            )

                    cal_result = await asyncio.to_thread(
                        state.builder.run_calibration,
                        steering_vector=state.steering_vector,
                        on_progress=cal_progress,
                    )
                    state.calibration_result = cal_result

                state.run_status.phase = RunPhase.COMPLETE
                state.run_status.finished_at = time.time()
                manager.broadcast("phase", {"phase": "complete"})

            except Exception as exc:
                logger.exception("Pipeline run failed.")
                state.run_status.phase = RunPhase.ERROR
                state.run_status.error = str(exc)
                state.run_status.finished_at = time.time()
                manager.broadcast(
                    "phase", {"phase": "error", "error": str(exc)}
                )

    return _run


@router.post("/run")
async def start_run(request: Request, body: RunRequest = RunRequest()):
    """Start a pipeline run.

    Returns 409 if a run is already active.  Execution happens in a
    background task; progress is streamed over the WebSocket.
    """
    state: ServerState = request.app.state.server
    manager: ConnectionManager = request.app.state.ws_manager

    if state.run_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"error": "A run is already in progress."},
        )

    asyncio.create_task(
        _pipeline_task(state, manager, body.stages, resume_run_dir=body.resume_run_dir)()
    )
    return {"status": "started", "stages": body.stages}


@router.get("/runs")
def list_runs(request: Request, behavior: str = Query("", max_length=200)) -> dict:
    """List existing run subdirectories matching a behavior label.

    For each matching run directory under `save_dir`, returns its pair count, the expected total (from the sidecar
    `run_meta.json` when present), and the parameters used so the frontend can decide whether resume is compatible.
    """
    state: ServerState = request.app.state.server
    save_dir = state.save_dir
    if not save_dir.exists():
        return {"runs": []}

    behavior_clean = (behavior or "").strip()
    prefix = f"{behavior_clean}_" if behavior_clean else ""

    runs = []
    for entry in save_dir.iterdir():
        if not entry.is_dir():
            continue
        if prefix and not entry.name.startswith(prefix):
            continue
        pairs_path = entry / "pairs.jsonl"
        meta_path = entry / "run_meta.json"
        pairs_count = _count_jsonl_lines(pairs_path)

        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}

        run_behavior = meta.get("behavior")
        if not run_behavior:
            first = _read_first_jsonl_record(pairs_path)
            if first:
                run_behavior = first.get("behavior")

        if behavior_clean and run_behavior and run_behavior != behavior_clean:
            continue

        has_svec = any(entry.glob("*.svec"))
        has_calibration = (entry / "calibration_result.json").exists()

        created = meta.get("created")
        if not created:
            try:
                created = datetime.datetime.fromtimestamp(
                    entry.stat().st_mtime, tz=datetime.UTC
                ).isoformat()
            except OSError:
                created = None

        runs.append({
            "run_dir": entry.name,
            "pairs_count": pairs_count,
            "behavior": run_behavior,
            "generator_model": meta.get("generator_model"),
            "positive_prompt": meta.get("positive_prompt"),
            "negative_prompt": meta.get("negative_prompt"),
            "created": created,
            "has_svec": has_svec,
            "has_calibration": has_calibration,
        })

    runs.sort(key=lambda r: r.get("created") or "", reverse=True)
    return {"runs": runs}


@router.post("/run/cancel")
async def cancel_run(request: Request) -> dict[str, str]:
    """Request cancellation of the active run.

    The cancellation flag is checked between stages; GPU work already in
    flight for the current stage is not aborted.
    """
    state: ServerState = request.app.state.server
    if state.run_status.phase in (
        RunPhase.GENERATION,
        RunPhase.EXTRACTION,
        RunPhase.CALIBRATION,
    ):
        state.request_cancel()
        return {"status": "cancel_requested"}
    return {"status": "no_run_active"}


@router.get("/run/status", response_model=RunStatusResponse)
def get_run_status(request: Request) -> RunStatusResponse:
    """Return the current run status."""
    state: ServerState = request.app.state.server
    rs = state.run_status
    return RunStatusResponse(
        phase=rs.phase.value,
        progress=rs.progress,
        error=rs.error,
        started_at=rs.started_at,
        finished_at=rs.finished_at,
        wall_time_s=state.run_wall_time(),
    )
