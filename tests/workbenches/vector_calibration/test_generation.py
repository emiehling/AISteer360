"""Tests for ContrastivePairGenerator's provider-vs-cancellation handling."""
from __future__ import annotations

from typing import Callable

import pytest

from aisteer360.workbenches.vector_calibration.agent.providers.base import GenerationProvider
from aisteer360.workbenches.vector_calibration.configs import GenerationConfig
from aisteer360.workbenches.vector_calibration.generation import ContrastivePairGenerator


class _StubProvider(GenerationProvider):
    """Provider whose `generate_batch` returns whatever the test configured."""

    def __init__(self, outputs: list[str]):
        self._outputs = outputs

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
        return list(self._outputs)


def _make_config(seed_prompts: list[str]) -> GenerationConfig:
    return GenerationConfig(
        generator_model="unused",
        behavior="warmth",
        positive_prompt="be warm",
        negative_prompt="be cold",
        seed_prompts=seed_prompts,
        batch_size=2,
    )


def test_generate_raises_on_provider_failure() -> None:
    provider = _StubProvider(outputs=[])
    cfg = _make_config(["hi", "hello", "hey"])
    generator = ContrastivePairGenerator(cfg, provider=provider)

    with pytest.raises(RuntimeError, match="returned no results"):
        generator.generate(cancel_check=lambda: False)


def test_generate_breaks_on_cancel_with_empty_batch() -> None:
    provider = _StubProvider(outputs=[])
    cfg = _make_config(["hi", "hello", "hey"])
    generator = ContrastivePairGenerator(cfg, provider=provider)

    result = generator.generate(cancel_check=lambda: True)

    assert result.seed_prompts_used == []
    assert len(result.pairs.positives) == 1
    assert result.pairs.positives == [""]
    assert result.pairs.negatives == [""]
