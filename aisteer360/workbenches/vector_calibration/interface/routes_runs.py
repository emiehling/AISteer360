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
from .catalog import load_catalog
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
from .dispatch import dispatch_local, dispatch_ssh, test_ssh
from .schemas import (
    AgentCommand,
    CellDetailResponse,
    ComputeConfig,
    ComputeConfigResponse,
    ComputeTestResponse,
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

_CATALOG_PROVIDER_TO_WIRE = {
    "hf": "hf",
    "anthropic": "anthropic",
    "openai": "openai",
    "openai_compatible": "openai",
}


def _validate_models_against_catalog(cfg: FullConfigSchema) -> None:
    """Check generator and judge models against the user's catalog.

    Both roles are picked from a curated catalog in the UI, so a model id that is missing or
    mismatched against its declared provider almost always indicates a stale saved config (the
    silent fallback in the UI used to coerce unknown ids to provider 'hf', which then sent
    requests for API-only models to Hugging Face).
    """
    entries = {e.model_id: e for e in load_catalog()}

    def _check(model_id: str, declared_provider: str, role: str) -> None:
        entry = entries.get(model_id)
        if entry is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"{role.capitalize()} model '{model_id}' is not in the model catalog. "
                "Add it via Settings → Model Catalog.",
            )
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

    _check(cfg.generation.generator_model, cfg.generation.generator_provider, "generator")
    _check(cfg.calibration.judge.model, cfg.calibration.judge.provider, "judge")


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
    cfg = body.config
    _validate_models_against_catalog(cfg)
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

    cmd = _agent_command(request, run_id, agent_token)
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

    dispatch_status = "manual"
    dispatch_error: str | None = None

    if mode == "ssh":
        try:
            dispatch_ssh(compute, agent_argv)
            dispatch_status = "ssh"
        except Exception as exc:
            logger.warning("SSH dispatch failed for %s: %s", run_id, exc)
            dispatch_status = "failed"
            dispatch_error = str(exc)
    elif mode == "local" or solo:
        try:
            proc = dispatch_local(agent_argv)
            request.app.state.local_agents[run_id] = proc
            dispatch_status = "local"
        except Exception as exc:
            logger.warning("Local dispatch failed for %s: %s", run_id, exc)
            dispatch_status = "failed"
            dispatch_error = str(exc)

    return RunCreateResponse(
        run=RunDetail(**run.to_detail()),
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


# ── compute config ───────────────────────────────────────────────

@router.get("/compute/config", response_model=ComputeConfigResponse)
async def get_compute(
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> ComputeConfigResponse:
    config = await db.get_compute_config(owner_hash)
    if config is None:
        return ComputeConfigResponse(mode="local")
    return ComputeConfigResponse(
        mode=config.get("mode", "local"),
        host=config.get("host"),
        port=config.get("port", 22),
        username=config.get("username"),
        auth_method=config.get("auth_method"),
        credential_set=bool(config.get("credential")),
        python_path=config.get("python_path") or "python3",
    )


@router.put("/compute/config")
async def put_compute(
    body: ComputeConfig,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    payload = body.model_dump()
    # treat unset credential as "leave unchanged" by dropping the key entirely
    if payload.get("credential") is None:
        payload.pop("credential", None)
    await db.upsert_compute_config(owner_hash, payload)
    return {"status": "ok"}


@router.post("/compute/test", response_model=ComputeTestResponse)
async def test_compute(
    body: ComputeConfig,
    request: Request,
    owner_hash: OwnerTokenHash,
    db: Database = Depends(get_db),
) -> ComputeTestResponse:
    payload = body.model_dump()
    # if the user typed a host but left credential blank, fall back to the stored one
    if not payload.get("credential"):
        existing = await db.get_compute_config(owner_hash)
        if existing and existing.get("credential"):
            payload["credential"] = existing["credential"]
    server_url = _public_server_url(request)
    result = test_ssh(payload, server_url)
    return ComputeTestResponse(**result)
