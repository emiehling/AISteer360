"""Browser-facing run endpoints scoped by owner token hash."""
from __future__ import annotations

import json
import logging
import shlex
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from aisteer360.workbenches.vector_calibration.results import CalibrationResult, CellResult

from .auth import OwnerScopedRun, OwnerTokenHash, get_db
from .db import (
    ACTIVE_STATUSES,
    STATUS_CANCELLED,
    STATUS_CREATED,
    STATUS_CLAIMED,
    Database,
    build_run_id,
    ensure_run_dir,
    hash_agent_token,
    mint_agent_token,
)
from .schemas import (
    AgentCommand,
    CellDetailResponse,
    FullConfigSchema,
    HeatmapResponse,
    RegenerateTokenResponse,
    RunCreateRequest,
    RunCreateResponse,
    RunDetail,
    RunListResponse,
    RunSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])


# ── helpers ──────────────────────────────────────────────────────

def _public_server_url(request: Request) -> str:
    override = getattr(request.app.state, "public_server_url", None)
    if override:
        return override.rstrip("/")
    return str(request.base_url).rstrip("/")


def _agent_command(request: Request, run_id: str, agent_token: str) -> AgentCommand:
    name = getattr(request.app.state, "agent_command_name", "aisteer360-agent")
    server = _public_server_url(request)
    parts = [name, "--server", server, "--run-id", run_id, "--agent-token", agent_token]
    command = " ".join(shlex.quote(p) for p in parts)
    return AgentCommand(
        command=command,
        server=server,
        run_id=run_id,
        agent_token=agent_token,
    )


def _load_calibration_result(run_dir: Path) -> CalibrationResult:
    path = run_dir / "calibration_result.json"
    if path.exists():
        return CalibrationResult.load(path)
    checkpoint = run_dir / "calibration_checkpoint.json"
    if checkpoint.exists():
        return _result_from_checkpoint(checkpoint)
    raise HTTPException(status.HTTP_404_NOT_FOUND, "No calibration result available yet.")


def _result_from_checkpoint(path: Path) -> CalibrationResult:
    """Build a partial CalibrationResult from an in-progress checkpoint file."""
    data = json.loads(path.read_text())
    cells = [CellResult(**d) for d in data]
    layers = sorted({c.layer for c in cells})
    multipliers = sorted({c.multiplier for c in cells})
    coherent = [c for c in cells if c.coherent]
    peak = max(coherent, key=lambda c: c.score_delta) if coherent else None
    return CalibrationResult(
        cells=cells,
        baseline_score=float("nan"),
        baseline_perplexity=float("nan"),
        peak_cell=peak,
        grid_shape=(len(layers), len(multipliers)),
        layers=layers,
        multipliers=multipliers,
        config={},
    )


# ── create / list / read ─────────────────────────────────────────

@router.post("/runs", response_model=RunCreateResponse)
async def create_run(
    body: RunCreateRequest,
    request: Request,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> RunCreateResponse:
    """Create a new run and mint its agent token."""
    cfg = body.config
    behavior = cfg.generation.behavior.strip() or "run"
    run_id = build_run_id(behavior)
    data_root: Path = request.app.state.data_root
    run_dir = ensure_run_dir(data_root, run_id)

    # persist save_dir hint so the workbench lands files in the right place
    cfg_dump = cfg.model_dump()
    cfg_dump["save_dir"] = str(run_dir.parent)

    agent_token = mint_agent_token()
    run = await db.create_run(
        run_id=run_id,
        behavior=behavior,
        steered_model=cfg.steered_model,
        config=cfg_dump,
        owner_token_hash=owner_hash,
        agent_token_hash=hash_agent_token(agent_token),
        run_dir=run_dir,
    )
    return RunCreateResponse(
        run=RunDetail(**run.to_detail()),
        agent_token=agent_token,
        agent_command=_agent_command(request, run_id, agent_token),
    )


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> RunListResponse:
    rows = await db.list_runs_for_owner(owner_hash)
    return RunListResponse(runs=[RunSummary(**r.to_summary()) for r in rows])


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run: OwnerScopedRun) -> RunDetail:
    return RunDetail(**run.to_detail())


@router.put("/runs/{run_id}/config")
async def update_config(
    run_id: str,
    body: FullConfigSchema,
    run: OwnerScopedRun,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    if run.status not in (STATUS_CREATED, STATUS_CLAIMED):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Cannot update config while status={run.status}.",
        )
    cfg_dump = body.model_dump()
    cfg_dump["save_dir"] = str(Path(run.run_dir).parent)
    await db.update_config(run_id, cfg_dump)
    return {"status": "ok"}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    run: OwnerScopedRun,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    if run.status in ACTIVE_STATUSES:
        await db.set_cancel(run_id, True)
        return {"status": "cancel_requested"}
    if run.status == STATUS_CREATED:
        await db.update_status(run_id, status=STATUS_CANCELLED)
        await db.set_cancel(run_id, True)
        return {"status": "cancelled"}
    return {"status": "no_active_run"}


@router.post("/runs/{run_id}/regenerate-token", response_model=RegenerateTokenResponse)
async def regenerate_agent_token(
    run_id: str,
    request: Request,
    run: OwnerScopedRun,
    db: Database = Depends(get_db),
) -> RegenerateTokenResponse:
    if run.status != STATUS_CREATED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Cannot regenerate token while status={run.status}.",
        )
    token = await db.regenerate_agent_token(run_id)
    return RegenerateTokenResponse(
        agent_token=token,
        agent_command=_agent_command(request, run_id, token),
    )


# ── results ──────────────────────────────────────────────────────

@router.get("/runs/{run_id}/results/heatmap", response_model=HeatmapResponse)
async def get_heatmap(run: OwnerScopedRun) -> HeatmapResponse:
    cal = _load_calibration_result(Path(run.run_dir))
    peak_data = None
    if cal.peak_cell is not None:
        peak_data = {
            k: v for k, v in asdict(cal.peak_cell).items() if k != "generations"
        }
    return HeatmapResponse(
        layers=cal.layers,
        multipliers=cal.multipliers,
        grids=cal.to_heatmap_grid(),
        baseline_score=cal.baseline_score,
        baseline_perplexity=cal.baseline_perplexity,
        peak=peak_data,
    )


@router.get(
    "/runs/{run_id}/results/cell/{layer}/{multiplier}",
    response_model=CellDetailResponse,
)
async def get_cell_detail(
    layer: int,
    multiplier: float,
    run: OwnerScopedRun,
) -> CellDetailResponse:
    cal = _load_calibration_result(Path(run.run_dir))
    for cell in cal.cells:
        if cell.layer == layer and abs(cell.multiplier - multiplier) < 1e-6:
            return CellDetailResponse(**asdict(cell))
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        f"No cell at layer={layer}, multiplier={multiplier}",
    )


# ── artifacts ────────────────────────────────────────────────────

@router.get("/runs/{run_id}/artifacts/pairs")
async def download_pairs(run: OwnerScopedRun) -> FileResponse:
    path = Path(run.run_dir) / "pairs.jsonl"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No pairs file.")
    return FileResponse(path, filename="pairs.jsonl", media_type="application/x-jsonlines")


@router.get("/runs/{run_id}/artifacts/svec")
async def download_svec(run: OwnerScopedRun) -> FileResponse:
    path = Path(run.run_dir) / f"{run.behavior}.svec"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No svec file.")
    return FileResponse(path, filename=f"{run.behavior}.svec", media_type="application/json")


@router.get("/runs/{run_id}/artifacts/result")
async def download_result(run: OwnerScopedRun) -> FileResponse:
    path = Path(run.run_dir) / "calibration_result.json"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No calibration result.")
    return FileResponse(
        path, filename="calibration_result.json", media_type="application/json"
    )
