"""Anthropic provider tests with a mocked SDK."""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, outputs: list[str]):
        self._outputs = outputs
        self._i = 0
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs) -> _FakeMessage:
        self.last_kwargs = kwargs
        text = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return _FakeMessage(text)


class _FakeAnthropicClient:
    def __init__(self, outputs: list[str]):
        self.messages = _FakeMessages(outputs)


@pytest.fixture
def anthropic_stub(monkeypatch):
    """Install a stub 'anthropic' module into sys.modules."""
    mod = types.ModuleType("anthropic")
    holder = {"client": None}

    def Anthropic(api_key: str) -> _FakeAnthropicClient:
        holder["client"] = _FakeAnthropicClient(holder.get("outputs") or ["ok"])
        return holder["client"]

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return holder


def test_generation_batch_hits_api(anthropic_stub) -> None:
    anthropic_stub["outputs"] = ["warm-A", "warm-B"]
    from aisteer360.workbenches.vector_calibration.agent.providers.anthropic_api import (
        AnthropicGenerationProvider,
    )
    p = AnthropicGenerationProvider(model_id="claude-test", api_key="k")
    out = p.generate_batch(
        "be warm",
        ["hi", "hello"],
        max_new_tokens=64,
        temperature=0.5,
        top_p=0.9,
    )
    assert set(out) == {"warm-A", "warm-B"}
    last = anthropic_stub["client"].messages.last_kwargs
    assert last["model"] == "claude-test"
    assert last["max_tokens"] == 64
    assert last["temperature"] == 0.5


def test_judge_parses_score(anthropic_stub) -> None:
    anthropic_stub["outputs"] = ['```json\n{"score": 4.2}\n```']
    from aisteer360.workbenches.vector_calibration.agent.providers.anthropic_api import (
        AnthropicJudgeProvider,
    )
    p = AnthropicJudgeProvider(model_id="claude-test", api_key="k")
    result = p.score(
        prompts=["why?"],
        responses=["because"],
        template="Rate the response 1-5. {response}",
        scale=(1, 5),
    )
    assert result["scores"] == [4.2]
    assert result["mean_score"] == 4.2


def test_judge_clamps_to_scale(anthropic_stub) -> None:
    anthropic_stub["outputs"] = ['```json\n{"score": 10}\n```']
    from aisteer360.workbenches.vector_calibration.agent.providers.anthropic_api import (
        AnthropicJudgeProvider,
    )
    p = AnthropicJudgeProvider(model_id="claude-test", api_key="k")
    result = p.score(
        prompts=["a"],
        responses=["b"],
        template="Rate 1-5. {response}",
        scale=(1, 5),
    )
    assert result["scores"] == [5.0]
