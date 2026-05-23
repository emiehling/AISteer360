"""Arguments for the PRewrite input control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class PRewriteArgs(BaseArgs):
    """Arguments for PRewrite (prompt rewriting via reinforcement learning).

    Required:
        initial_prompt: The user's hand-written under-optimized prompt. The rewriter is trained to improve it.
        rewriter_model_name_or_path: HF identifier or local path for the rewriter LM. Should be an instruct-style
            model capable of following the meta-prompt.
        training_data: List of training inputs. Each dict has keys:

            - `"input"`: str (the user query).
            - `"expected"`: str (optional reference for metric computation).

        feedback_metric: A `Metric` for computing reward. Called as `metric.compute(responses=[...], references=[...])`
            → dict. The first scalar value is extracted (same convention as `TaskLMScorer._extract_scalar`).

    Optional:
        mode: `"per_query"` (default) or `"static"`.

            - `per_query`: rewriter conditions on (meta_prompt, initial_prompt, query) during both training and
              serving. Adapter at serve time runs the rewriter on each user query. Memory = `ModelMemory`.
            - `static`: rewriter conditions on (meta_prompt, initial_prompt) only. At end of `steer()`, generate one
              rewrite and cache it. Adapter at serve time is template-fill with the cached instruction. Memory =
              `TextMemory` (rewriter discarded after training).

        meta_prompt: When None, uses `DEFAULT_PER_QUERY_META_PROMPT` (mode=per_query) or `DEFAULT_STATIC_META_PROMPT`
            (mode=static).
        n_steps: Number of PPO steps. Must be >= 1.
        batch_size: Rollouts per PPO step. Must be >= 1.
        mini_batch_size: PPO mini-batch size.
        ppo_epochs: PPO epochs per step.
        learning_rate: PPO optimizer learning rate.
        kl_coef: KL penalty coefficient against the frozen reference rewriter. Must be >= 0.
        rewriter_gen_kwargs: Forwarded to `rewriter.generate()`. Defaults: `max_new_tokens=128`, sampling during
            training, greedy at serve time.
        task_gen_kwargs: Forwarded to the task model's `generate()` for reward rollouts.
        use_peft: When True, train via LoRA.
        lora_kwargs: Forwarded to `peft.LoraConfig` (`r`, `lora_alpha`, `lora_dropout`, `target_modules`, ...).
        rewriter_load_kwargs: Forwarded to `AutoModelForCausalLM.from_pretrained` when loading the rewriter.
        output_dir: If set, TRL writes checkpoints here during training. None disables checkpointing.
        seed: RNG seed.
    """

    initial_prompt: str = ""
    rewriter_model_name_or_path: str = ""
    training_data: list[dict] = field(default_factory=list)
    feedback_metric: Any = None

    mode: Literal["per_query", "static"] = "per_query"
    meta_prompt: str | None = None

    n_steps: int = 100
    batch_size: int = 8
    mini_batch_size: int = 1
    ppo_epochs: int = 4
    learning_rate: float = 1.41e-5
    kl_coef: float = 0.1

    rewriter_gen_kwargs: dict | None = None
    task_gen_kwargs: dict | None = None

    use_peft: bool = False
    lora_kwargs: dict[str, Any] = field(default_factory=dict)

    rewriter_load_kwargs: dict | None = None

    output_dir: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.initial_prompt:
            raise ValueError("initial_prompt must be non-empty")
        if not self.rewriter_model_name_or_path:
            raise ValueError("rewriter_model_name_or_path must be non-empty")
        if not self.training_data:
            raise ValueError("training_data must be non-empty")
        if self.feedback_metric is None:
            raise ValueError("feedback_metric is required")
        if self.mode not in ("per_query", "static"):
            raise ValueError(f"mode must be 'per_query' or 'static'; got {self.mode!r}")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.mini_batch_size < 1:
            raise ValueError("mini_batch_size must be >= 1")
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be >= 1")
        if self.kl_coef < 0:
            raise ValueError("kl_coef must be >= 0")
        for i, row in enumerate(self.training_data):
            if "input" not in row:
                raise ValueError(f"training_data[{i}] missing 'input' key")
