"""Configuration dataclasses for the vector calibration workbench."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GenerationConfig:
    """How contrastive pairs are produced.

    The generator model receives seed prompts twice (once with the positive system prompt, once with the negative),
    producing paired responses that exhibit or lack the target behavior.

    Attributes:
        generator_model: HF model id or path for the model that writes the contrastive responses. May differ from
            the steered model.
        behavior: Short label for the target behavior (e.g. "warmth"). Used for naming artifacts; not passed to
            the model.
        positive_prompt: System instruction for the positive direction.
        negative_prompt: System instruction for the negative direction.
        seed_prompts: Seed user messages. If a string, treated as a path to a JSON or JSONL file of strings.
        n_pairs: Number of pairs to produce. If `len(seed_prompts) < n_pairs`, seeds are cycled; if greater, a
            random subset is sampled.
        max_new_tokens: Max tokens per response.
        temperature: Sampling temperature for the generator.
        top_p: Nucleus sampling threshold.
        batch_size: Batch size for generator inference.
        seed: Random seed for reproducibility.
    """

    generator_model: str
    behavior: str
    positive_prompt: str
    negative_prompt: str
    seed_prompts: list[str] | str | None = None
    n_pairs: int = 300
    max_new_tokens: int = 160
    temperature: float = 0.9
    top_p: float = 0.95
    batch_size: int = 8
    seed: int = 42


@dataclass
class ExtractionConfig:
    """How hidden states become a steering vector.

    Maps onto the existing `VectorTrainSpec` plus estimator selection, with extra post-processing knobs exposed from
    the dashboard.

    Attributes:
        method: Estimator name. `"mean_diff"` maps to `MeanDifferenceEstimator`; `"pca_pairwise"` maps to
            `ContrastiveDirectionEstimator`.
        accumulate: Token aggregation mode for hidden state extraction.
        normalize: L2-normalize each per-layer direction after fitting.
        center: Mean-center activations before computing directions.
        per_layer_rescale: Rescale each layer's direction by its explained variance (only meaningful for PCA).
        layers: Which layers to extract. `"all"` extracts every layer; a list of ints keeps only those layers.
        batch_size: Batch size for the extraction forward passes.
    """

    method: Literal["mean_diff", "pca_pairwise"] = "mean_diff"
    accumulate: Literal["all", "last_token", "suffix-only"] = "last_token"
    normalize: bool = True
    center: bool = True
    per_layer_rescale: bool = False
    layers: list[int] | Literal["all"] = "all"
    batch_size: int = 8


@dataclass
class JudgeConfig:
    """LLM-as-judge configuration for scoring steered outputs.

    Either `rating_scale` or `criteria` must be populated. When `rating_scale` is provided, the judge prompt is
    generated from the listed (label, description) rows and `scale` is inferred from the label extrema.

    Attributes:
        model: HF model id or path for the judge model.
        criteria: Prompt template. Should contain a `{response}` placeholder. Ignored when `rating_scale` is set.
        rating_scale: Ordered list of `(label, description)` rows, e.g. `[(0, "no warmth"), (1, "warm response")]`.
        scale: `(low, high)` integer range for the judge's score. Derived from `rating_scale` when present.
        batch_size: Batch size for judge inference.
        hf_model_kwargs: Extra kwargs for loading the judge model.
    """

    model: str
    criteria: str = ""
    rating_scale: list[tuple[int, str]] | None = None
    scale: tuple[int, int] = (1, 5)
    batch_size: int = 32
    hf_model_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rating_scale:
            labels = [int(row[0]) for row in self.rating_scale]
            self.scale = (min(labels), max(labels))


@dataclass
class SweepGrid:
    """Defines the (layer x multiplier) search space.

    Attributes:
        multiplier_range: `(start, end)` inclusive.
        multiplier_step: Step size between multiplier values.
        layer_range: `(start, end)` inclusive. Defaults to all layers in the steered model.
        layer_step: Step between layers (1 means every layer).
    """

    multiplier_range: tuple[float, float] = (-3.0, 3.0)
    multiplier_step: float = 0.25
    layer_range: tuple[int, int] | None = None
    layer_step: int = 1


@dataclass
class QualityGate:
    """Thresholds for marking a cell as coherent.

    Cells that fail the gate are recorded but flagged incoherent.

    Attributes:
        coherence_threshold: Minimum self-consistency score (0 to 1).
        perplexity_max_ratio: Maximum allowed perplexity as a multiple of the baseline perplexity.
    """

    coherence_threshold: float = 0.95
    perplexity_max_ratio: float = 2.0


@dataclass
class CalibrationConfig:
    """How the vector is applied and judged.

    Attributes:
        judge: Judge model configuration.
        sweep: Grid search space definition.
        quality_gate: Coherence / perplexity thresholds.
        transform: Which transform to use when applying the vector.
        token_scope: Which tokens to steer during calibration generation.
        max_new_tokens: Max tokens per calibration generation.
        batch_size: Batch size for steered generation.
        eval_prompts: Held-out prompts for calibration evaluation. If a string, treated as a path. If None, a
            subset of the generation seed prompts is reserved.
        n_eval_prompts: Number of eval prompts to use per cell.
    """

    judge: JudgeConfig
    sweep: SweepGrid = field(default_factory=SweepGrid)
    quality_gate: QualityGate = field(default_factory=QualityGate)
    transform: Literal["additive", "norm_preserving"] = "additive"
    token_scope: Literal["all", "after_prompt", "last_k", "from_position"] = "all"
    max_new_tokens: int = 200
    batch_size: int = 32
    eval_prompts: list[str] | str | None = None
    n_eval_prompts: int = 30


@dataclass
class CalibrationBuilderConfig:
    """Top-level configuration for the full builder pipeline.

    Attributes:
        steered_model: HF model id or path for the model being steered.
        generation: Stage 1 config.
        extraction: Stage 2 config.
        calibration: Stage 3 config.
        hf_model_kwargs: Extra kwargs for loading the steered model.
        device_map: Device placement for the steered model.
        save_dir: Directory for all artifacts (pairs, svec, calibration results).
    """

    steered_model: str
    generation: GenerationConfig
    extraction: ExtractionConfig
    calibration: CalibrationConfig
    hf_model_kwargs: dict = field(default_factory=dict)
    device_map: str = "auto"
    save_dir: str | None = None
