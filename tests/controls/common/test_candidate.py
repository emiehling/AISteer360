"""Tests for the Candidate value type."""
from aisteer360.algorithms.input_control.common.candidate import Candidate


def test_candidate_auto_id():
    a = Candidate(memory={"x": 1})
    b = Candidate(memory={"x": 1})
    assert a.id != b.id


def test_candidate_explicit_id():
    c = Candidate(memory={"x": 1}, id="foo")
    assert c.id == "foo"


def test_candidate_metadata_default():
    a = Candidate(memory=None)
    b = Candidate(memory=None)
    assert a.metadata == {}
    a.metadata["k"] = 1
    assert b.metadata == {}  # not shared between instances
