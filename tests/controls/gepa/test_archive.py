"""Tests for PerInstanceParetoArchive."""
from __future__ import annotations

from collections import Counter

import torch

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace
from aisteer360.algorithms.input_control.gepa.archive import PerInstanceParetoArchive


def _trace(score: float, example: dict) -> Trace:
    t = torch.tensor([[1]])
    return Trace(
        input_ids=t,
        steered_input_ids=t,
        output=Output(output_ids=t),
        score=score,
        metadata={"task_example": example},
    )


def _data(ids: list[str]) -> list[dict]:
    return [{"id": i} for i in ids]


def test_per_instance_pareto_basic_frontier():
    arc = PerInstanceParetoArchive()
    data = _data(["i1", "i2", "i3"])
    c1 = Candidate(memory=None, id="c1")
    c2 = Candidate(memory=None, id="c2")
    c3 = Candidate(memory=None, id="c3")
    # rock-paper-scissors: each best on a different instance
    arc.ingest(
        [c1, c2, c3],
        [
            [_trace(1.0, data[0]), _trace(0.0, data[1]), _trace(0.0, data[2])],
            [_trace(0.0, data[0]), _trace(1.0, data[1]), _trace(0.0, data[2])],
            [_trace(0.0, data[0]), _trace(0.0, data[1]), _trace(1.0, data[2])],
        ],
    )
    member_ids = {c.id for c in arc.members()}
    assert member_ids == {"c1", "c2", "c3"}


def test_per_instance_pareto_dominated_dropped():
    arc = PerInstanceParetoArchive()
    data = _data(["i1", "i2"])
    c1 = Candidate(memory=None, id="c1")  # 0.9, 0.9
    c2 = Candidate(memory=None, id="c2")  # 0.5, 0.5  -- dominated
    arc.ingest(
        [c1, c2],
        [
            [_trace(0.9, data[0]), _trace(0.9, data[1])],
            [_trace(0.5, data[0]), _trace(0.5, data[1])],
        ],
    )
    member_ids = {c.id for c in arc.members()}
    assert member_ids == {"c1"}


def test_per_instance_pareto_selection_weighted():
    arc = PerInstanceParetoArchive(rng_seed=42)
    data = _data(["i1", "i2", "i3", "i4"])
    # c1 covers 3 instances, c2 covers 1 -> 3:1 expected weighting
    c1 = Candidate(memory=None, id="c1")
    c2 = Candidate(memory=None, id="c2")
    arc.ingest(
        [c1, c2],
        [
            [_trace(1.0, data[0]), _trace(1.0, data[1]), _trace(1.0, data[2]), _trace(0.0, data[3])],
            [_trace(0.0, data[0]), _trace(0.0, data[1]), _trace(0.0, data[2]), _trace(1.0, data[3])],
        ],
    )
    counts = Counter(arc.select_for_mutation().id for _ in range(2000))
    ratio = counts["c1"] / counts["c2"]
    # expected ~3:1; allow generous tolerance
    assert 2.3 < ratio < 4.0


def test_per_instance_pareto_best_max_mean():
    arc = PerInstanceParetoArchive()
    data = _data(["i1", "i2"])
    # rock-paper-scissors ensuring all on frontier; distinct means
    c1 = Candidate(memory=None, id="c1")  # 1.0, 0.2 -> mean 0.6
    c2 = Candidate(memory=None, id="c2")  # 0.3, 1.0 -> mean 0.65 (top mean)
    arc.ingest(
        [c1, c2],
        [
            [_trace(1.0, data[0]), _trace(0.2, data[1])],
            [_trace(0.3, data[0]), _trace(1.0, data[1])],
        ],
    )
    member_ids = {c.id for c in arc.members()}
    assert member_ids == {"c1", "c2"}
    assert arc.best().id == "c2"


def test_per_instance_pareto_uses_instance_id():
    arc = PerInstanceParetoArchive()
    # examples without explicit id default to str(idx); with explicit id, use it
    ex_with_id = [{"id": "alpha"}, {"id": "beta"}]
    c1 = Candidate(memory=None, id="c1")
    arc.ingest([c1], [[_trace(1.0, ex_with_id[0]), _trace(0.5, ex_with_id[1])]])
    score_map = arc._scores["c1"]
    assert "alpha" in score_map
    assert "beta" in score_map
    assert score_map["alpha"] == 1.0
    assert score_map["beta"] == 0.5


def test_per_instance_pareto_ties():
    arc = PerInstanceParetoArchive()
    data = _data(["i1"])
    c1 = Candidate(memory=None, id="c1")
    c2 = Candidate(memory=None, id="c2")
    arc.ingest(
        [c1, c2],
        [
            [_trace(1.0, data[0])],
            [_trace(1.0, data[0])],
        ],
    )
    member_ids = {c.id for c in arc.members()}
    assert member_ids == {"c1", "c2"}
    # both should have coverage 1
    assert arc._coverage["c1"] == 1
    assert arc._coverage["c2"] == 1
