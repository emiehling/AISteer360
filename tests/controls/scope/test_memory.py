"""Tests for `Rule` and `RuleStreamMemory`."""
from __future__ import annotations

import os
import time

import pytest

from aisteer360.algorithms.input_control.common.memory.base import Memory
from aisteer360.algorithms.input_control.scope.memory import Rule, RuleStreamMemory


def _make_rule(text: str = "be polite", stream: str = "strategic", confidence: float = 0.9) -> Rule:
    return Rule(
        text=text,
        confidence=confidence,
        stream=stream,
        created_at=time.time(),
        metadata={"synthesis_mode": "unified"},
    )


def test_rule_construction():
    rule = Rule(
        text="prefer concise answers",
        confidence=0.8,
        stream="strategic",
        created_at=123.0,
    )
    assert rule.text == "prefer concise answers"
    assert rule.confidence == 0.8
    assert rule.stream == "strategic"
    assert rule.created_at == 123.0
    assert rule.metadata == {}


def test_rule_stream_memory_defaults():
    memory = RuleStreamMemory()
    assert memory.strategic == []
    assert memory.tactical == []
    assert memory.model_type == "rule_stream"


def test_rule_stream_memory_round_trip(tmp_path):
    memory = RuleStreamMemory()
    memory.strategic.append(_make_rule("alpha", "strategic", 0.95))
    memory.tactical.append(_make_rule("beta", "tactical", 0.4))

    path = str(tmp_path / "memory.rsm")
    memory.save(path)
    loaded = RuleStreamMemory.load(path)

    assert len(loaded.strategic) == 1
    assert len(loaded.tactical) == 1
    assert loaded.strategic[0].text == "alpha"
    assert loaded.strategic[0].confidence == 0.95
    assert loaded.strategic[0].stream == "strategic"
    assert loaded.strategic[0].created_at == memory.strategic[0].created_at
    assert loaded.strategic[0].metadata == {"synthesis_mode": "unified"}
    assert loaded.tactical[0].text == "beta"
    assert loaded.model_type == "rule_stream"


def test_rule_stream_memory_extension_appended(tmp_path):
    memory = RuleStreamMemory()
    base_path = str(tmp_path / "no_extension")
    memory.save(base_path)
    assert os.path.exists(base_path + ".rsm")


def test_rule_stream_memory_load_rejects_wrong_type(tmp_path):
    path = tmp_path / "wrong.rsm"
    path.write_text('{"model_type": "text", "strategic": [], "tactical": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="model_type"):
        RuleStreamMemory.load(str(path))


def test_rule_stream_memory_reset_tactical():
    memory = RuleStreamMemory()
    memory.strategic.append(_make_rule("keep me", "strategic"))
    memory.tactical.append(_make_rule("drop me", "tactical"))

    memory.reset_tactical()

    assert len(memory.strategic) == 1
    assert memory.strategic[0].text == "keep me"
    assert memory.tactical == []


def test_rule_stream_memory_all_rules_ordering():
    memory = RuleStreamMemory()
    s = _make_rule("strategic_one", "strategic")
    t = _make_rule("tactical_one", "tactical")
    memory.strategic.append(s)
    memory.tactical.append(t)

    rules = memory.all_rules()

    assert rules == [s, t]


def test_rule_stream_memory_satisfies_memory_protocol():
    assert isinstance(RuleStreamMemory(), Memory)
