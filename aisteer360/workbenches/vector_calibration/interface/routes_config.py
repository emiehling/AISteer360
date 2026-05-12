"""Configuration read/write endpoints."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from aisteer360.workbenches.vector_calibration import (
    CalibrationBuilderConfig,
    CalibrationConfig,
    ExtractionConfig,
    GenerationConfig,
    JudgeConfig,
    QualityGate,
    SweepGrid,
)

from .schemas import (
    CalibrationConfigSchema,
    ExtractionConfigSchema,
    FullConfigSchema,
    GenerationConfigSchema,
)
from .state import RunPhase, ServerState

router = APIRouter(tags=["config"])


def _current_state(request: Request) -> ServerState:
    return request.app.state.server


def _full_schema_from_config(cfg: CalibrationBuilderConfig) -> FullConfigSchema:
    cal = cfg.calibration
    return FullConfigSchema(
        steered_model=cfg.steered_model,
        generation=GenerationConfigSchema(**asdict(cfg.generation)),
        extraction=ExtractionConfigSchema(**asdict(cfg.extraction)),
        calibration=CalibrationConfigSchema(
            judge=asdict(cal.judge),
            sweep=asdict(cal.sweep),
            quality_gate=asdict(cal.quality_gate),
            transform=cal.transform,
            token_scope=cal.token_scope,
            max_new_tokens=cal.max_new_tokens,
            batch_size=cal.batch_size,
            eval_prompts=cal.eval_prompts,
            n_eval_prompts=cal.n_eval_prompts,
        ),
        hf_model_kwargs=cfg.hf_model_kwargs,
        device_map=cfg.device_map,
        save_dir=cfg.save_dir,
    )


def _generation_from_schema(s: GenerationConfigSchema) -> GenerationConfig:
    return GenerationConfig(**s.model_dump())


def _extraction_from_schema(s: ExtractionConfigSchema) -> ExtractionConfig:
    return ExtractionConfig(**s.model_dump())


def _calibration_from_schema(s: CalibrationConfigSchema) -> CalibrationConfig:
    return CalibrationConfig(
        judge=JudgeConfig(**s.judge.model_dump()),
        sweep=SweepGrid(**s.sweep.model_dump()),
        quality_gate=QualityGate(**s.quality_gate.model_dump()),
        transform=s.transform,
        token_scope=s.token_scope,
        max_new_tokens=s.max_new_tokens,
        batch_size=s.batch_size,
        eval_prompts=s.eval_prompts,
        n_eval_prompts=s.n_eval_prompts,
    )


def _guard_not_running(state: ServerState) -> None:
    if state.run_status.phase in (
        RunPhase.GENERATION,
        RunPhase.EXTRACTION,
        RunPhase.CALIBRATION,
    ):
        raise HTTPException(409, "Cannot update config while a run is in progress.")


@router.get("/config", response_model=FullConfigSchema)
def get_config(request: Request) -> FullConfigSchema:
    """Return the current full pipeline configuration."""
    return _full_schema_from_config(_current_state(request).config)


@router.put("/config")
def update_config(request: Request, body: FullConfigSchema) -> dict[str, str]:
    """Replace the full pipeline configuration and rebuild the builder."""
    state = _current_state(request)
    _guard_not_running(state)

    state.config = CalibrationBuilderConfig(
        steered_model=body.steered_model,
        generation=_generation_from_schema(body.generation),
        extraction=_extraction_from_schema(body.extraction),
        calibration=_calibration_from_schema(body.calibration),
        hf_model_kwargs=body.hf_model_kwargs,
        device_map=body.device_map,
        save_dir=body.save_dir or str(state.save_dir),
    )
    state.rebuild_builder()
    return {"status": "ok"}


@router.patch("/config/generation")
def patch_generation_config(
    request: Request, body: GenerationConfigSchema
) -> dict[str, str]:
    """Replace the generation-stage config."""
    state = _current_state(request)
    _guard_not_running(state)
    state.config.generation = _generation_from_schema(body)
    state.rebuild_builder()
    return {"status": "ok"}


@router.patch("/config/extraction")
def patch_extraction_config(
    request: Request, body: ExtractionConfigSchema
) -> dict[str, str]:
    """Replace the extraction-stage config."""
    state = _current_state(request)
    _guard_not_running(state)
    state.config.extraction = _extraction_from_schema(body)
    state.rebuild_builder()
    return {"status": "ok"}


@router.patch("/config/calibration")
def patch_calibration_config(
    request: Request, body: CalibrationConfigSchema
) -> dict[str, str]:
    """Replace the calibration-stage config."""
    state = _current_state(request)
    _guard_not_running(state)
    state.config.calibration = _calibration_from_schema(body)
    state.rebuild_builder()
    return {"status": "ok"}
