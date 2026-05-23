"""Tests for the SCOPE meta-agents."""
from __future__ import annotations

import time

from aisteer360.algorithms.input_control.scope.memory import Rule, RuleStreamMemory
from aisteer360.algorithms.input_control.scope.meta_agents import (
    GuidelineClassifier,
    GuidelineGenerator,
    GuidelineSelector,
    MemoryOptimizer,
)

from tests.controls.scope._stubs import StubReflectionLM, make_classifier_lm


def _empty_memory() -> RuleStreamMemory:
    return RuleStreamMemory()


def _strategic_rule(text: str) -> Rule:
    return Rule(text=text, confidence=0.9, stream="strategic", created_at=time.time())


def test_generator_returns_n_candidates():
    lm = StubReflectionLM(["alpha", "beta", "gamma"])
    generator = GuidelineGenerator(lm=lm, n_candidates=3)
    candidates = generator.synthesize(
        input_text="user input",
        response_text="model response",
        current_memory=_empty_memory(),
    )
    assert candidates == ["alpha", "beta", "gamma"]
    assert len(lm.prompts) == 3


def test_generator_empty_response_handling():
    lm = StubReflectionLM(["", "useful guideline", "   "])
    generator = GuidelineGenerator(lm=lm, n_candidates=3)
    candidates = generator.synthesize(
        input_text="user input",
        response_text="model response",
        current_memory=_empty_memory(),
    )
    assert candidates == ["useful guideline"]


def test_selector_picks_one_candidate():
    lm = StubReflectionLM(["1"])
    selector = GuidelineSelector(lm=lm)
    chosen = selector.select(
        candidates=["a", "b", "c"],
        current_memory=_empty_memory(),
        input_text="input",
        response_text="response",
    )
    assert chosen == "b"


def test_selector_falls_back_to_first_on_parse_failure():
    lm = StubReflectionLM(["I cannot decide right now."])
    selector = GuidelineSelector(lm=lm)
    chosen = selector.select(
        candidates=["x", "y"],
        current_memory=_empty_memory(),
        input_text="input",
        response_text="response",
    )
    assert chosen == "x"


def test_classifier_returns_stream_and_confidence():
    classifier = GuidelineClassifier(lm=make_classifier_lm("strategic", 0.9))
    stream, confidence = classifier.classify("be polite", _empty_memory())
    assert stream == "strategic"
    assert confidence == 0.9


def test_classifier_falls_back_to_tactical_on_parse_failure():
    lm = StubReflectionLM(["totally unparseable response, no JSON here"])
    classifier = GuidelineClassifier(lm=lm)
    stream, confidence = classifier.classify("guideline text", _empty_memory())
    assert stream == "tactical"
    assert confidence == 0.0


def test_optimizer_consolidates():
    rules = [_strategic_rule("be polite"), _strategic_rule("always be polite")]
    lm = StubReflectionLM([
        "- be polite\n- always be polite",  # conflict pass leaves both
        "- be polite",                        # subsumption keeps only the general
        "- be polite",                        # consolidation pass-through
    ])
    optimizer = MemoryOptimizer(lm=lm)
    consolidated = optimizer.consolidate(rules)
    assert len(consolidated) == 1
    assert consolidated[0].text == "be polite"


def test_optimizer_returns_at_most_input_length():
    rules = [_strategic_rule("a"), _strategic_rule("b"), _strategic_rule("c")]
    lm = StubReflectionLM([
        "- a\n- b\n- c\n- d\n- e",  # conflict step expands to 5
        "- a\n- b\n- c\n- d\n- e",
        "- a\n- b\n- c\n- d\n- e",
    ])
    optimizer = MemoryOptimizer(lm=lm)
    consolidated = optimizer.consolidate(rules)
    assert len(consolidated) <= len(rules)
