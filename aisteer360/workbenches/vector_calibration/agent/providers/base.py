"""Provider interfaces and factory.

`GenerationProvider` powers contrastive pair generation (stage 1). `JudgeProvider` powers the
LLM-as-judge scoring step inside the calibration sweep (stage 3). Extraction (stage 2) always
runs against the local steered model — no provider is involved.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ProviderKeys:
    """Credentials / endpoint overrides supplied by the agent CLI or environment."""

    hf_token: str | None = None
    anthropic_key: str | None = None
    openai_key: str | None = None
    openai_base_url: str | None = None


class GenerationProvider(ABC):
    """Generate responses conditioned on a system prompt."""

    @abstractmethod
    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[str]:
        """Return one response per user prompt.

        Implementations must respect `cancel_check` if provided — when it returns True between
        requests, return the already-completed responses (callers discard the partial batch).
        """

    def close(self) -> None:
        """Release underlying resources (model weights, HTTP session, etc.)."""


class JudgeProvider(ABC):
    """Score generated responses against a template-driven rubric."""

    @abstractmethod
    def score(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        template: str,
        scale: tuple[float, float],
    ) -> dict[str, Any]:
        """Return `{"scores": [...], "mean_score": float, "raw_scores": [[...], ...]}`.

        `template` is the rubric template with a `{response}` placeholder (and optional
        `{prompt}`, `{lower_bound}`, `{upper_bound}`).
        """

    def close(self) -> None:  # noqa: B027 - intentional default
        """Release underlying resources."""


def build_from_config(
    config: dict,
    keys: ProviderKeys,
) -> tuple[GenerationProvider, JudgeProvider]:
    """Instantiate providers for a run's config.

    The config is the JSON form of `CalibrationBuilderConfig`. Reads
    `generation.generator_provider`, `calibration.judge.provider`, and the corresponding keys
    from `ProviderKeys`.
    """
    gen_cfg = config.get("generation", {})
    cal_cfg = config.get("calibration", {})
    judge_cfg = cal_cfg.get("judge", {})

    gen_provider_name = gen_cfg.get("generator_provider") or "hf"
    judge_provider_name = judge_cfg.get("provider") or "hf"

    gen = _make_generation_provider(gen_provider_name, gen_cfg, keys)
    judge = _make_judge_provider(judge_provider_name, judge_cfg, keys)
    return gen, judge


def _make_generation_provider(
    name: str,
    gen_cfg: dict,
    keys: ProviderKeys,
) -> GenerationProvider:
    if name == "hf":
        from .hf_local import HFGenerationProvider
        return HFGenerationProvider(
            model_id=gen_cfg["generator_model"],
            hf_token=keys.hf_token,
        )
    if name == "anthropic":
        from .anthropic_api import AnthropicGenerationProvider
        if not keys.anthropic_key:
            raise ValueError("Anthropic generator selected but --anthropic-key is missing.")
        return AnthropicGenerationProvider(
            model_id=gen_cfg["generator_model"],
            api_key=keys.anthropic_key,
        )
    if name == "openai":
        from .openai_api import OpenAIGenerationProvider
        if not keys.openai_key:
            raise ValueError("OpenAI generator selected but --openai-key is missing.")
        return OpenAIGenerationProvider(
            model_id=gen_cfg["generator_model"],
            api_key=keys.openai_key,
            base_url=gen_cfg.get("generator_base_url") or keys.openai_base_url,
        )
    raise ValueError(f"Unknown generator provider '{name}'.")


def _make_judge_provider(
    name: str,
    judge_cfg: dict,
    keys: ProviderKeys,
) -> JudgeProvider:
    if name == "hf":
        from .hf_local import HFJudgeProvider
        return HFJudgeProvider(config=judge_cfg, hf_token=keys.hf_token)
    if name == "anthropic":
        from .anthropic_api import AnthropicJudgeProvider
        if not keys.anthropic_key:
            raise ValueError("Anthropic judge selected but --anthropic-key is missing.")
        return AnthropicJudgeProvider(
            model_id=judge_cfg["model"],
            api_key=keys.anthropic_key,
        )
    if name == "openai":
        from .openai_api import OpenAIJudgeProvider
        if not keys.openai_key:
            raise ValueError("OpenAI judge selected but --openai-key is missing.")
        return OpenAIJudgeProvider(
            model_id=judge_cfg["model"],
            api_key=keys.openai_key,
            base_url=judge_cfg.get("base_url") or keys.openai_base_url,
        )
    raise ValueError(f"Unknown judge provider '{name}'.")
