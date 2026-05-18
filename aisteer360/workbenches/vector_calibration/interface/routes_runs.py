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

from aisteer360.workbenches.common.interface.auth import OwnerScopedRun, OwnerTokenHash, get_db
from aisteer360.workbenches.common.interface.catalog import load_catalog
from aisteer360.workbenches.common.interface.db import (
    ACTIVE_STATUSES,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_CREATED,
    STATUS_CLAIMED,
    STATUS_FAILED,
    Database,
    build_run_id,
    ensure_run_dir,
    hash_agent_token,
    mint_agent_token,
)
from aisteer360.workbenches.common.interface.dispatch import dispatch_local, dispatch_ssh
from .schemas import (
    AgentCommand,
    CellDetailResponse,
    FullConfigSchema,
    HeatmapResponse,
    RegenerateTokenResponse,
    RunContinueRequest,
    RunCreateRequest,
    RunCreateResponse,
    RunDetail,
    RunListResponse,
    RunSummary,
    Stage,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])


# ── helpers ──────────────────────────────────────────────────────

_CATALOG_PROVIDER_TO_WIRE = {
    "hf": "hf",
    "anthropic": "anthropic",
    "openai": "openai",
    "openai_compatible": "openai",
}


def _check_catalog_entry(
    entries: dict, model_id: str, declared_provider: str | None, role: str
) -> None:
    """Validate one model id against the catalog. `declared_provider` may be None for target."""
    entry = entries.get(model_id)
    if entry is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{role.capitalize()} model '{model_id}' is not in the model catalog. "
            "Add it via Settings → Model Catalog.",
        )
    if declared_provider is not None:
        wire = _CATALOG_PROVIDER_TO_WIRE.get(entry.provider)
        if wire != declared_provider:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"{role.capitalize()} model '{model_id}' is registered with provider "
                f"'{entry.provider}' but the run requested '{declared_provider}'. "
                "Update the catalog entry or pick a different model.",
            )
    if role not in entry.roles:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Model '{model_id}' is not enabled for the {role} role in the catalog.",
        )


def _validate_for_stages(cfg: FullConfigSchema, stages: list[Stage]) -> None:
    """Validate that the config has every field required by the requested stages."""
    needs = set(stages)
    entries = {e.model_id: e for e in load_catalog()}

    if "generation" in needs:
        if not (cfg.generation.behavior or "").strip():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Behavior label is required.")
        _check_catalog_entry(
            entries, cfg.generation.generator_model, cfg.generation.generator_provider, "inference"
        )

    if needs & {"extraction", "calibration"}:
        if not (cfg.steered_model or "").strip():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Target model is required for extraction and calibration.",
            )
        _check_catalog_entry(entries, cfg.steered_model, None, "target")

    if "calibration" in needs:
        _check_catalog_entry(
            entries, cfg.calibration.judge.model, cfg.calibration.judge.provider, "inference"
        )


def _check_prerequisites(run_dir: Path, stages: list[Stage], behavior: str) -> None:
    """Verify that on-disk artifacts needed by `stages` exist (unless an earlier stage produces them)."""
    needs = set(stages)
    if "extraction" in needs and "generation" not in needs:
        if not (run_dir / "pairs.jsonl").exists():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Cannot run extraction: pairs.jsonl not found in run directory. "
                "Run generation first or upload a pairs file.",
            )
    if "calibration" in needs and "extraction" not in needs:
        svec = run_dir / f"{behavior}.svec"
        if not svec.exists():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Cannot run calibration: {svec.name} not found in run directory. "
                "Run extraction first.",
            )


def _write_pairs_data(run_dir: Path, pairs_data: str | None) -> None:
    """Write user-uploaded pairs jsonl content to the run directory."""
    if not pairs_data:
        return
    text = pairs_data if pairs_data.endswith("\n") else pairs_data + "\n"
    (run_dir / "pairs.jsonl").write_text(text, encoding="utf-8")


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


async def _dispatch_agent(
    request: Request,
    db: Database,
    *,
    run_id: str,
    cmd: AgentCommand,
    owner_hash: str,
) -> tuple[str, str | None]:
    """Try to start the agent process. Returns (dispatch_status, dispatch_error)."""
    name = getattr(request.app.state, "agent_command_name", "aisteer360-agent")
    agent_argv = [
        name,
        "--server", cmd.server,
        "--run-id", cmd.run_id,
        "--agent-token", cmd.agent_token,
    ]
    compute = await db.get_compute_config(owner_hash)
    mode = compute.get("mode") if compute else None
    solo = getattr(request.app.state, "solo_mode", False)

    if mode == "ssh":
        try:
            dispatch_ssh(compute, agent_argv)
            return "ssh", None
        except Exception as exc:
            logger.warning("SSH dispatch failed for %s: %s", run_id, exc)
            return "failed", str(exc)
    if mode == "local" or solo:
        try:
            proc = dispatch_local(agent_argv)
            request.app.state.local_agents[run_id] = proc
            return "local", None
        except Exception as exc:
            logger.warning("Local dispatch failed for %s: %s", run_id, exc)
            return "failed", str(exc)
    return "manual", None


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


# ── data root ────────────────────────────────────────────────────

@router.get("/data-root")
async def get_data_root(request: Request, _: OwnerTokenHash) -> dict:
    root = str(request.app.state.data_root.resolve())
    return {"data_root": root}


# ── create / list / read ─────────────────────────────────────────

@router.post("/runs", response_model=RunCreateResponse)
async def create_run(
    body: RunCreateRequest,
    request: Request,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> RunCreateResponse:
    """Create a new run and mint its agent token."""
    if "generation" not in body.stages and not body.pairs_data:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "New runs must include the generation stage (or upload a pairs file to skip it). "
            "To run extraction or calibration on an existing run, use the continue endpoint.",
        )
    cfg = body.config
    _validate_for_stages(cfg, body.stages)
    behavior = cfg.generation.behavior.strip() or "run"
    run_id = build_run_id(behavior)
    data_root: Path = request.app.state.data_root
    run_dir = ensure_run_dir(data_root, run_id)
    _write_pairs_data(run_dir, body.pairs_data)

    # persist save_dir hint so the workbench lands files in the right place
    cfg_dump = cfg.model_dump()
    cfg_dump["save_dir"] = str(run_dir.parent)

    agent_token = mint_agent_token()
    run = await db.create_run(
        run_id=run_id,
        behavior=behavior,
        steered_model=cfg.steered_model,
        config=cfg_dump,
        stages=list(body.stages),
        owner_token_hash=owner_hash,
        agent_token_hash=hash_agent_token(agent_token),
        run_dir=run_dir,
    )

    cmd = _agent_command(request, run_id, agent_token)
    dispatch_status, dispatch_error = await _dispatch_agent(
        request, db, run_id=run_id, cmd=cmd, owner_hash=owner_hash,
    )
    return RunCreateResponse(
        run=RunDetail(**run.to_detail()),
        agent_token=agent_token,
        agent_command=cmd,
        dispatch_status=dispatch_status,
        dispatch_error=dispatch_error,
    )


@router.post("/runs/{run_id}/continue", response_model=RunCreateResponse)
async def continue_run(
    run_id: str,
    body: RunContinueRequest,
    request: Request,
    run: OwnerScopedRun,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> RunCreateResponse:
    """Re-dispatch a terminal run with a new config and stage selection."""
    if run.status not in (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Cannot continue a run while status={run.status}.",
        )

    cfg = body.config
    _validate_for_stages(cfg, body.stages)
    _write_pairs_data(Path(run.run_dir), body.pairs_data)
    _check_prerequisites(Path(run.run_dir), body.stages, cfg.generation.behavior)

    cfg_dump = cfg.model_dump()
    cfg_dump["save_dir"] = str(Path(run.run_dir).parent)
    await db.update_config(run_id, cfg_dump)
    await db.update_stages(run_id, list(body.stages))
    await db.reset_for_continue(run_id)

    # clear stale calibration artifacts so the agent starts a fresh sweep with the new grid
    if "calibration" in body.stages:
        run_path = Path(run.run_dir)
        for stale_file in ("calibration_checkpoint.json", "calibration_result.json"):
            p = run_path / stale_file
            if p.exists():
                p.unlink()

    agent_token = await db.regenerate_agent_token(run_id)
    cmd = _agent_command(request, run_id, agent_token)
    dispatch_status, dispatch_error = await _dispatch_agent(
        request, db, run_id=run_id, cmd=cmd, owner_hash=owner_hash,
    )
    updated = await db.get_run(run_id)
    assert updated is not None
    return RunCreateResponse(
        run=RunDetail(**updated.to_detail()),
        agent_token=agent_token,
        agent_command=cmd,
        dispatch_status=dispatch_status,
        dispatch_error=dispatch_error,
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
    if run.status not in (STATUS_CREATED, STATUS_CLAIMED, STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED):
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
    request: Request,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    agents = getattr(request.app.state, "local_agents", {})
    proc = agents.pop(run_id, None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        logger.info("Terminated local agent for %s (pid %d)", run_id, proc.pid)

    if run.status in ACTIVE_STATUSES:
        await db.set_cancel(run_id, True)
        if proc is not None:
            await db.update_status(run_id, status=STATUS_CANCELLED)
            return {"status": "cancelled"}
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
    run_dir = Path(run.run_dir)

    # checkpoint preserves generations; the result file strips them to keep JSON small
    checkpoint = run_dir / "calibration_checkpoint.json"
    if checkpoint.exists():
        try:
            data = json.loads(checkpoint.read_text())
            for d in data:
                if d.get("layer") == layer and abs(d.get("multiplier", 0) - multiplier) < 1e-6:
                    return CellDetailResponse(**{
                        k: v for k, v in d.items() if k != "n_generations"
                    })
        except (json.JSONDecodeError, KeyError):
            pass

    cal = _load_calibration_result(run_dir)
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