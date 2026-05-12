"""Result data endpoints for the dashboard."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .schemas import CellDetailResponse, HeatmapResponse
from .state import ServerState

router = APIRouter(tags=["results"])


def _current_state(request: Request) -> ServerState:
    return request.app.state.server


@router.get("/results/heatmap", response_model=HeatmapResponse)
def get_heatmap(request: Request) -> HeatmapResponse:
    """Return the full heatmap grid for all view modes.

    The `grids` field contains 2D arrays for `score_delta`, `coherence`,
    and `perplexity`, ready for direct rendering.
    """
    state = _current_state(request)
    cal = state.calibration_result
    if cal is None:
        raise HTTPException(404, "No calibration result available.")

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
    "/results/cell/{layer}/{multiplier}", response_model=CellDetailResponse
)
def get_cell_detail(
    request: Request, layer: int, multiplier: float
) -> CellDetailResponse:
    """Return full detail (including generations) for one heatmap cell."""
    state = _current_state(request)
    cal = state.calibration_result
    if cal is None:
        raise HTTPException(404, "No calibration result available.")

    for cell in cal.cells:
        if cell.layer == layer and abs(cell.multiplier - multiplier) < 1e-6:
            return CellDetailResponse(**asdict(cell))

    raise HTTPException(
        404, f"No cell at layer={layer}, multiplier={multiplier}"
    )


@router.get("/artifacts/svec")
def download_svec(request: Request) -> FileResponse:
    """Download the steering vector as a .svec file."""
    state = _current_state(request)
    run_dir = state.run_dir
    if run_dir is None:
        raise HTTPException(404, "No active run.")
    behavior = state.config.generation.behavior
    path = run_dir / f"{behavior}.svec"
    if not path.exists():
        raise HTTPException(404, "No .svec file found.")
    return FileResponse(
        path, filename=f"{behavior}.svec", media_type="application/json"
    )


@router.get("/artifacts/pairs")
def download_pairs(request: Request) -> FileResponse:
    """Download the generated contrastive pairs."""
    state = _current_state(request)
    run_dir = state.run_dir
    if run_dir is None:
        raise HTTPException(404, "No active run.")
    path = run_dir / "pairs.jsonl"
    if not path.exists():
        raise HTTPException(404, "No pairs file found.")
    return FileResponse(
        path, filename="pairs.jsonl", media_type="application/x-jsonlines"
    )


@router.get("/artifacts/result")
def download_result(request: Request) -> FileResponse:
    """Download the full calibration result."""
    state = _current_state(request)
    run_dir = state.run_dir
    if run_dir is None:
        raise HTTPException(404, "No active run.")
    path = run_dir / "calibration_result.json"
    if not path.exists():
        raise HTTPException(404, "No calibration result found.")
    return FileResponse(
        path,
        filename="calibration_result.json",
        media_type="application/json",
    )
