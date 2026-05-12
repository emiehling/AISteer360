"""Pipeline execution endpoints."""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .schemas import RunRequest, RunStatusResponse
from .state import RunPhase, ServerState
from .ws import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["run"])


def _pipeline_task(state: ServerState, manager: ConnectionManager, stages: list[str]):
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

                    result = await asyncio.to_thread(
                        state.builder.run_generation, on_progress=gen_progress
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

    asyncio.create_task(_pipeline_task(state, manager, body.stages)())
    return {"status": "started", "stages": body.stages}


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
