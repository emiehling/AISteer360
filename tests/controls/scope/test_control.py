"""Unit tests for the SCOPE control."""
from __future__ import annotations

import time

import torch

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.scope import SCOPE, SCOPEArgs
from aisteer360.algorithms.input_control.scope.memory import Rule, RuleStreamMemory

from tests.controls.scope._stubs import StubReflectionLM


class _CharTokenizer:
    """Char-level tokenizer with no chat template — `f'{system}\\n\\n{user}'` join."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(
            chr(int(i)) for i in ids
            if not (skip_special_tokens and i in (self.pad_token_id, self.eos_token_id))
        )


def _strategic_rule(text: str, confidence: float = 0.95) -> Rule:
    return Rule(text=text, confidence=confidence, stream="strategic", created_at=time.time())


def _tactical_rule(text: str, confidence: float = 0.4) -> Rule:
    return Rule(text=text, confidence=confidence, stream="tactical", created_at=time.time())


def _build_scope(
    *,
    reflection_lm,
    seed_rules=None,
    n_candidates: int = 1,
    confidence_threshold: float = 0.85,
    strategic_max_size: int = 10,
    tactical_max_size: int | None = None,
    base_prompt: str | None = None,
    trigger_predicate=None,
) -> SCOPE:
    args = SCOPEArgs(
        reflection_lm=reflection_lm,
        seed_rules=seed_rules,
        n_candidates=n_candidates,
        confidence_threshold=confidence_threshold,
        strategic_max_size=strategic_max_size,
        tactical_max_size=tactical_max_size,
        base_prompt=base_prompt,
        trigger_predicate=trigger_predicate,
    )
    return SCOPE(args)


def _classifier_response(stream: str, confidence: float) -> str:
    return '{"category": "%s", "confidence": %s}' % (stream, confidence)


def test_scope_is_stateful():
    assert SCOPE.is_stateful is True


def test_scope_steer_initializes_memory():
    scope = _build_scope(reflection_lm=lambda p: "")
    scope.steer(model=None, tokenizer=_CharTokenizer())
    assert isinstance(scope.memory, RuleStreamMemory)
    assert scope.memory.strategic == []
    assert scope.memory.tactical == []


def test_scope_steer_loads_seed_rules():
    seeds = [
        _strategic_rule("strategic seed"),
        _tactical_rule("tactical seed"),
    ]
    scope = _build_scope(reflection_lm=lambda p: "", seed_rules=seeds)
    scope.steer(model=None, tokenizer=_CharTokenizer())
    assert len(scope.memory.strategic) == 1
    assert scope.memory.strategic[0].text == "strategic seed"
    assert len(scope.memory.tactical) == 1
    assert scope.memory.tactical[0].text == "tactical seed"


def test_scope_adapt_with_empty_memory():
    scope = _build_scope(reflection_lm=lambda p: "")
    tok = _CharTokenizer()
    scope.steer(model=None, tokenizer=tok)
    ids = tok.encode("hello")
    out = scope.adapt(ids)
    assert out == ids


def test_scope_adapt_includes_rules_in_system_prompt():
    seeds = [_strategic_rule("be polite"), _tactical_rule("use bullet points")]
    scope = _build_scope(reflection_lm=lambda p: "", seed_rules=seeds)
    tok = _CharTokenizer()
    scope.steer(model=None, tokenizer=tok)

    ids = tok.encode("user input")
    adapted = scope.adapt(ids)
    decoded = tok.decode(adapted)
    assert "be polite" in decoded
    assert "use bullet points" in decoded
    assert "user input" in decoded


def test_scope_observe_adds_rule():
    lm = StubReflectionLM([
        "be concise",                                # generator
        _classifier_response("strategic", 0.95),     # classifier (selector skipped, n=1)
    ])
    scope = _build_scope(reflection_lm=lm, n_candidates=1)
    tok = _CharTokenizer()
    scope.steer(model=None, tokenizer=tok)

    input_ids = torch.tensor([tok.encode("question?")], dtype=torch.long)
    output = Output(output_ids=torch.tensor([tok.encode("answer.")], dtype=torch.long))
    scope.observe(input_ids=input_ids, output=output)

    assert len(scope.memory.strategic) == 1
    assert scope.memory.strategic[0].text == "be concise"
    assert scope.memory.strategic[0].confidence == 0.95


def test_scope_observe_routes_low_confidence_to_tactical():
    lm = StubReflectionLM([
        "be concise",                                # generator
        _classifier_response("strategic", 0.5),      # classifier confidence < threshold
    ])
    scope = _build_scope(reflection_lm=lm, n_candidates=1, confidence_threshold=0.85)
    tok = _CharTokenizer()
    scope.steer(model=None, tokenizer=tok)

    input_ids = torch.tensor([tok.encode("q")], dtype=torch.long)
    output = Output(output_ids=torch.tensor([tok.encode("a")], dtype=torch.long))
    scope.observe(input_ids=input_ids, output=output)

    assert scope.memory.strategic == []
    assert len(scope.memory.tactical) == 1
    assert scope.memory.tactical[0].stream == "tactical"


def test_scope_observe_skipped_when_predicate_false():
    lm = StubReflectionLM(["should not be called"])
    scope = _build_scope(
        reflection_lm=lm,
        n_candidates=1,
        trigger_predicate=lambda i, o: False,
    )
    tok = _CharTokenizer()
    scope.steer(model=None, tokenizer=tok)

    input_ids = torch.tensor([tok.encode("q")], dtype=torch.long)
    output = Output(output_ids=torch.tensor([tok.encode("a")], dtype=torch.long))
    scope.observe(input_ids=input_ids, output=output)

    assert scope.memory.strategic == []
    assert scope.memory.tactical == []
    assert lm.prompts == []


def test_scope_observe_triggers_optimizer_at_capacity():
    seeds = [
        _strategic_rule("rule a"),
        _strategic_rule("rule b"),
    ]
    lm = StubReflectionLM([
        "rule c",                                    # generator
        _classifier_response("strategic", 0.95),     # classifier -> strategic, push triggers optimizer
        "- rule a\n- rule b\n- rule c",              # optimizer: conflict
        "- rule a\n- rule b",                         # optimizer: subsumption (collapses one)
        "- rule a\n- rule b",                         # optimizer: consolidation
    ])
    scope = _build_scope(reflection_lm=lm, n_candidates=1, seed_rules=seeds, strategic_max_size=2)
    tok = _CharTokenizer()
    scope.steer(model=None, tokenizer=tok)

    input_ids = torch.tensor([tok.encode("q")], dtype=torch.long)
    output = Output(output_ids=torch.tensor([tok.encode("a")], dtype=torch.long))
    scope.observe(input_ids=input_ids, output=output)

    assert len(scope.memory.strategic) <= 2


def test_scope_reset_session_clears_tactical():
    seeds = [_strategic_rule("keep me"), _tactical_rule("drop me")]
    scope = _build_scope(reflection_lm=lambda p: "", seed_rules=seeds)
    scope.steer(model=None, tokenizer=_CharTokenizer())

    scope.reset_session()

    assert len(scope.memory.strategic) == 1
    assert scope.memory.strategic[0].text == "keep me"
    assert scope.memory.tactical == []


def test_scope_cleanup_releases_reflection_lm():
    scope = _build_scope(reflection_lm=lambda p: "")
    scope.steer(model=None, tokenizer=_CharTokenizer())
    assert scope._reflection_lm is not None

    scope.cleanup()
    assert scope._reflection_lm is None
    assert scope._generator is None
    # idempotent
    scope.cleanup()
    assert scope._reflection_lm is None
