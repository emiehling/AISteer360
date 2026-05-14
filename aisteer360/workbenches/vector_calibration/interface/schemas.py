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
    generator_provider: Literal["hf", "anthropic", "openai"] = "hf"
    generator_base_url: str | None = None


class ExtractionConfigSchema(BaseModel):
    method: Literal["mean_diff", "pca_pairwise"] = "mean_diff"
    accumulate: Literal["all", "last_token", "suffix-only"] = "last_token"
    normalize: bool = True
    center: bool = True
    per_layer_rescale: bool = False
    layers: list[int] | Literal["all"] = "all"
    batch_size: int = 8
    pair_split_ratio: float = Field(default=1.0, gt=0.0, le=1.0)


class JudgeConfigSchema(BaseModel):
    model: str
    criteria: str = ""
    rating_scale: list[tuple[int, str]] | None = None
    scale: tuple[int, int] = (1, 5)
    batch_size: int = 32
    hf_model_kwargs: dict[str, Any] = Field(default_factory=dict)
    provider: Literal["hf", "anthropic", "openai"] = "hf"
    base_url: str | None = None


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

Stage = Literal["generation", "extraction", "calibration"]


class RunCreateRequest(BaseModel):
    """Body for POST /api/runs."""
    config: FullConfigSchema
    stages: list[Stage]
    pairs_data: str | None = None


class RunContinueRequest(BaseModel):
    """Body for POST /api/runs/{id}/continue."""
    config: FullConfigSchema
    stages: list[Stage]
    pairs_data: str | None = None


class RunSummary(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    behavior: str
    steered_model: str
    status: str
    stages: list[Stage]
    phase: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    model_info: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    stale: bool = False
    created_at: float
    updated_at: float
    last_heartbeat: float | None = None
    claimed_at: float | None = None
    completed_at: float | None = None


class RunDetail(RunSummary):
    model_config = {"protected_namespaces": ()}

    config: FullConfigSchema
    run_dir: str
    has_pairs: bool = False


class AgentCommand(BaseModel):
    """Container for the command line the user runs locally to start an agent."""
    command: str
    server: str
    run_id: str
    agent_token: str


class RunCreateResponse(BaseModel):
    run: RunDetail
    agent_token: str
    agent_command: AgentCommand
    dispatch_status: Literal["local", "ssh", "manual", "failed"] = "manual"
    dispatch_error: str | None = None


class RunListResponse(BaseModel):
    runs: list[RunSummary]


class RegenerateTokenResponse(BaseModel):
    agent_token: str
    agent_command: AgentCommand


# ── compute config ───────────────────────────────────────────────

class ComputeConfig(BaseModel):
    mode: Literal["local", "ssh"] = "local"
    host: str | None = None
    port: int = 22
    username: str | None = None
    auth_method: Literal["key", "password"] | None = None
    credential: str | None = None
    python_path: str = "python3"


class ComputeConfigResponse(BaseModel):
    """Compute config returned to the browser. Credentials are never sent back in plaintext."""
    mode: Literal["local", "ssh"] = "local"
    host: str | None = None
    port: int = 22
    username: str | None = None
    auth_method: Literal["key", "password"] | None = None
    credential_set: bool = False
    python_path: str = "python3"


class ComputeTestResponse(BaseModel):
    ok: bool
    error: str | None = None
    device: str | None = None
    device_name: str | None = None
    device_count: int | None = None
    server_reachable: bool | None = None
    reachability_error: str | None = None


# ── agent-facing schemas ─────────────────────────────────────────

class ClaimResponse(BaseModel):
    run_id: str
    run_dir: str
    config: FullConfigSchema
    stages: list[Stage]
    provider_keys: dict[str, str | None] = Field(default_factory=dict)


class ProgressPost(BaseModel):
    phase: str
    completed: int | None = None
    total: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ModelInfoPost(BaseModel):
    model_config = {"protected_namespaces": ()}

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


class CancelCheckResponse(BaseModel):
    cancel_requested: bool


class StageCompleteRequest(BaseModel):
    notes: str | None = None


class ErrorPost(BaseModel):
    message: str


class LogPost(BaseModel):
    lines: list[str]


# ── result schemas ───────────────────────────────────────────────

class HeatmapResponse(BaseModel):
    layers: list[int]
    multipliers: list[float]
    grids: dict[str, list[list[float | None]]]
    baseline_score: float
    baseline_perplexity: float
    peak: dict[str, Any] | None = None


class CellDetailResponse(BaseModel):
    layer: int
    multiplier: float
    score_mean: float
    score_delta: float
    coherence: float
    perplexity: float
    perplexity_delta: float
    coherent: bool
    generations: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "GenerationConfigSchema",
    "ExtractionConfigSchema",
    "JudgeConfigSchema",
    "SweepGridSchema",
    "QualityGateSchema",
    "CalibrationConfigSchema",
    "FullConfigSchema",
    "Stage",
    "RunCreateRequest",
    "RunContinueRequest",
    "RunSummary",
    "RunDetail",
    "AgentCommand",
    "RunCreateResponse",
    "RunListResponse",
    "RegenerateTokenResponse",
    "ClaimResponse",
    "ProgressPost",
    "ModelInfoPost",
    "CancelCheckResponse",
    "StageCompleteRequest",
    "ErrorPost",
    "LogPost",
    "HeatmapResponse",
    "CellDetailResponse",
    "ComputeConfig",
    "ComputeConfigResponse",
    "ComputeTestResponse",
]
