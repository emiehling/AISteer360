"""Tests for the Archive Protocol, LatestArchive, and ParetoArchive."""
from __future__ import annotations

import pytest
import torch

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.common.archives import Archive, LatestArchive, ParetoArchive
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.trace import Trace


def _trace(score) -> Trace:
    t = torch.tensor([[1]])
    return Trace(
        input_ids=t,
        steered_input_ids=t,
        output=Output(output_ids=torch.tensor([[2]])),
        score=score,
    )


def test_archive_protocol_check():
    assert isinstance(LatestArchive(), Archive)
    assert isinstance(ParetoArchive(score_keys=["a"]), Archive)


def test_latest_archive_overwrites():
    arc = LatestArchive()
    c1 = Candidate(memory="m1", id="c1")
    c2 = Candidate(memory="m2", id="c2")
    arc.ingest([c1], [[_trace(0.1)]])
    arc.ingest([c2], [[_trace(0.9)]])
    best = arc.best()
    assert best.id == "c2"
    assert arc.traces_for(c1) == []
    assert len(arc.traces_for(c2)) == 1


def test_latest_archive_empty_raises():
    arc = LatestArchive()
    with pytest.raises(RuntimeError):
        arc.select_for_mutation()


def test_pareto_archive_keeps_non_dominated():
    arc = ParetoArchive(score_keys=["a", "b"])
    c1 = Candidate(memory=None, id="c1")  # {a:1, b:1}
    c2 = Candidate(memory=None, id="c2")  # {a:0, b:2}
    c3 = Candidate(memory=None, id="c3")  # {a:0, b:0}  -- dominated by both
    arc.ingest(
        [c1, c2, c3],
        [
            [_trace({"a": 1.0, "b": 1.0})],
            [_trace({"a": 0.0, "b": 2.0})],
            [_trace({"a": 0.0, "b": 0.0})],
        ],
    )
    member_ids = {c.id for c in arc.members()}
    assert member_ids == {"c1", "c2"}


def test_pareto_archive_best_max_sum():
    arc = ParetoArchive(score_keys=["a", "b"])
    c1 = Candidate(memory=None, id="c1")  # {a:2, b:0} sum 2
    c2 = Candidate(memory=None, id="c2")  # {a:0, b:2} sum 2
    c3 = Candidate(memory=None, id="c3")  # {a:1.5, b:1.6} sum 3.1
    arc.ingest(
        [c1, c2, c3],
        [
            [_trace({"a": 2.0, "b": 0.0})],
            [_trace({"a": 0.0, "b": 2.0})],
            [_trace({"a": 1.5, "b": 1.6})],
        ],
    )
    member_ids = {c.id for c in arc.members()}
    assert member_ids == {"c1", "c2", "c3"}
    assert arc.best().id == "c3"


def test_pareto_archive_requires_dict_score():
    arc = ParetoArchive(score_keys=["a"])
    c = Candidate(memory=None)
    with pytest.raises(TypeError):
        arc.ingest([c], [[_trace(0.5)]])


def test_pareto_archive_select_is_seeded():
    def build_arc(seed):
        arc = ParetoArchive(score_keys=["a", "b"], rng_seed=seed)
        c1 = Candidate(memory=None, id="c1")
        c2 = Candidate(memory=None, id="c2")
        c3 = Candidate(memory=None, id="c3")
        arc.ingest(
            [c1, c2, c3],
            [
                [_trace({"a": 1.0, "b": 0.0})],
                [_trace({"a": 0.0, "b": 1.0})],
                [_trace({"a": 0.5, "b": 0.5})],
            ],
        )
        return arc

    arc1 = build_arc(42)
    arc2 = build_arc(42)
    seq1 = [arc1.select_for_mutation().id for _ in range(10)]
    seq2 = [arc2.select_for_mutation().id for _ in range(10)]
    assert seq1 == seq2
