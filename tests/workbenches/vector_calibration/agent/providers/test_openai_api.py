"""OpenAI-compatible provider tests with a mocked SDK."""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _FakeCompletions:
    def __init__(self, outputs: list[str]):
        self._outputs = outputs
        self._i = 0
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs) -> _Resp:
        self.last_kwargs = kwargs
        text = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return _Resp(text)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, outputs: list[str]):
        self.chat = _FakeChat(_FakeCompletions(outputs))


@pytest.fixture
def openai_stub(monkeypatch):
    mod = types.ModuleType("openai")
    holder = {"client": None, "init_kwargs": None}

    def OpenAI(**kwargs) -> _FakeOpenAIClient:
        holder["init_kwargs"] = kwargs
        holder["client"] = _FakeOpenAIClient(holder.get("outputs") or ["ok"])
        return holder["client"]

    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return holder


def test_generation_forwards_kwargs(openai_stub) -> None:
    openai_stub["outputs"] = ["A", "B"]
    from aisteer360.workbenches.vector_calibration.agent.providers.openai_api import (
        OpenAIGenerationProvider,
    )
    p = OpenAIGenerationProvider(model_id="gpt-x", api_key="sk-...", base_url="http://vllm")
    out = p.generate_batch(
        "system",
        ["q1", "q2"],
        max_new_tokens=32,
        temperature=0.2,
        top_p=0.95,
    )
    assert set(out) == {"A", "B"}
    assert openai_stub["init_kwargs"] == {"api_key": "sk-...", "base_url": "http://vllm"}
    last = openai_stub["client"].chat.completions.last_kwargs
    assert last["model"] == "gpt-x"
    assert last["max_tokens"] == 32
    assert last["temperature"] == 0.2


def test_judge_round_trip(openai_stub) -> None:
    openai_stub["outputs"] = ['{"score": 3}']
    from aisteer360.workbenches.vector_calibration.agent.providers.openai_api import (
        OpenAIJudgeProvider,
    )
    p = OpenAIJudgeProvider(model_id="gpt-x", api_key="sk-...")
    result = p.score(
        prompts=["p"],
        responses=["r"],
        template="Rate 1-5. {response}",
        scale=(1, 5),
    )
    assert result["scores"] == [3.0]
    assert result["mean_score"] == 3.0
