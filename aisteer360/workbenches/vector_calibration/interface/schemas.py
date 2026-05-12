"""Pydantic models for the calibration dashboard API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── config schemas ───────────────────────────────────────────────

class GenerationConfigSchema(BaseModel):
    generator_model: str
    behavior: str
    positive_prompt: str
    negative_prompt: str
    seed_prompts: list[str] | str | None = None
    max_new_tokens: int = 160
    temperature: float = 0.9
    top_p: float = 0.95
    batch_size: int = 8
    seed: int = 42


class ExtractionConfigSchema(BaseModel):
    method: Literal["mean_diff", "pca_pairwise"] = "mean_diff"
    accumulate: Literal["all", "last_token", "suffix-only"] = "last_token"
    normalize: bool = True
    center: bool = True
    per_layer_rescale: bool = False
    layers: list[int] | Literal["all"] = "all"
    batch_size: int = 8


class JudgeConfigSchema(BaseModel):
    model: str
    criteria: str = ""
    rating_scale: list[tuple[int, str]] | None = None
    scale: tuple[int, int] = (1, 5)
    batch_size: int = 32
    hf_model_kwargs: dict[str, Any] = Field(default_factory=dict)


class SweepGridSchema(BaseModel):
    multiplier_range: tuple[float, float] = (-3.0, 3.0)
    multiplier_step: float = 0.25
    layer_range: tuple[int, int] | None = None
    layer_step: int = 1


class QualityGateSchema(BaseModel):
    coherence_threshold: float = 0.95
    perplexity_max_ratio: float = 2.0


class CalibrationConfigSchema(BaseModel):
    judge: JudgeConfigSchema
    sweep: SweepGridSchema = Field(default_factory=SweepGridSchema)
    quality_gate: QualityGateSchema = Field(default_factory=QualityGateSchema)
    transform: Literal["additive", "norm_preserving"] = "additive"
    token_scope: Literal["all", "after_prompt", "last_k", "from_position"] = "all"
    max_new_tokens: int = 200
    batch_size: int = 32
    eval_prompts: list[str] | str | None = None
    n_eval_prompts: int = 30


class FullConfigSchema(BaseModel):
    steered_model: str
    generation: GenerationConfigSchema
    extraction: ExtractionConfigSchema
    calibration: CalibrationConfigSchema
    hf_model_kwargs: dict[str, Any] = Field(default_factory=dict)
    device_map: str = "auto"
    save_dir: str | None = None


# ── run schemas ──────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Which stages to execute."""
    stages: list[Literal["generation", "extraction", "calibration"]] = Field(
        default=["generation", "extraction", "calibration"],
        description="Ordered list of stages to run.",
    )
    resume_run_dir: str | None = Field(
        default=None,
        description="Name of an existing run subdirectory (under save_dir) to resume generation into.",
    )


class RunStatusResponse(BaseModel):
    phase: str
    progress: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    wall_time_s: float | None = None


# ── result schemas ───────────────────────────────────────────────

class HeatmapResponse(BaseModel):
    """Grid data for the heatmap panel."""
    layers: list[int]
    multipliers: list[float]
    grids: dict[str, list[list[float | None]]]
    baseline_score: float
    baseline_perplexity: float
    peak: dict[str, Any] | None = None


class CellDetailResponse(BaseModel):
    """Full detail for a selected heatmap cell."""
    layer: int
    multiplier: float
    score_mean: float
    score_delta: float
    coherence: float
    perplexity: float
    perplexity_delta: float
    coherent: bool
    generations: list[dict[str, Any]] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    model_name: str | None = None
    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    dtype: str | None = None
    device: str | None = None
    model_type: str | None = None
