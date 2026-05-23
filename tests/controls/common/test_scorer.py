"""Tests for the Scorer Protocol and TaskLMScorer."""
from __future__ import annotations

import torch

from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.scorers import Scorer, TaskLMScorer
from aisteer360.algorithms.input_control.common.trace import Trace

from ._stubs import StubMetric, StubModel, StubTokenizer, make_data


class _ProtocolScorer:
    def score(self, candidates, data):
        return [[] for _ in candidates]


def test_scorer_protocol_check():
    assert isinstance(_ProtocolScorer(), Scorer)


def test_task_lm_scorer_basic():
    model = StubModel(suffix_ids=[ord("X"), ord("Y")])
    tokenizer = StubTokenizer()
    metric = StubMetric()

    def adapter(input_ids, memory):
        return input_ids

    scorer = TaskLMScorer(model=model, tokenizer=tokenizer, adapter=adapter, metric=metric)

    data = make_data(["abc", "de"])
    candidates = [Candidate(memory=None), Candidate(memory={"x": 1})]

    results = scorer.score(candidates, data)

    assert len(results) == 2
    for traces in results:
        assert len(traces) == 2
        for t in traces:
            assert isinstance(t, Trace)
            # response is "XY" (suffix), length 2 from StubMetric
            assert t.score == 2.0
            assert t.metadata["raw_response"] == "XY"


def test_task_lm_scorer_passes_memory_via_adapter():
    model = StubModel()
    tokenizer = StubTokenizer()
    metric = StubMetric()

    received: list = []

    def adapter(input_ids, memory):
        received.append(memory)
        return input_ids

    scorer = TaskLMScorer(model=model, tokenizer=tokenizer, adapter=adapter, metric=metric)

    mem_a = {"name": "A"}
    mem_b = {"name": "B"}
    data = make_data(["x"])
    scorer.score([Candidate(memory=mem_a), Candidate(memory=mem_b)], data)

    # one call per (candidate, example) -- two candidates, one example
    assert received == [mem_a, mem_b]
