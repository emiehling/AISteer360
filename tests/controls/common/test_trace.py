"""Tests for the Trace value type."""
import torch

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.common.trace import Trace


def _make_output() -> Output:
    return Output(output_ids=torch.tensor([[7, 8, 9]], dtype=torch.long))


def test_trace_construction():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    steered = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    output = _make_output()
    trace = Trace(
        input_ids=input_ids,
        steered_input_ids=steered,
        output=output,
        score=0.5,
    )
    assert torch.equal(trace.input_ids, input_ids)
    assert torch.equal(trace.steered_input_ids, steered)
    assert trace.output is output
    assert trace.score == 0.5
    assert trace.metadata == {}


def test_trace_score_can_be_scalar():
    trace = Trace(
        input_ids=torch.tensor([[1]]),
        steered_input_ids=torch.tensor([[1]]),
        output=_make_output(),
        score=0.5,
    )
    assert trace.score == 0.5


def test_trace_score_can_be_dict():
    trace = Trace(
        input_ids=torch.tensor([[1]]),
        steered_input_ids=torch.tensor([[1]]),
        output=_make_output(),
        score={"a": 0.5, "b": 0.7},
    )
    assert trace.score == {"a": 0.5, "b": 0.7}


def test_trace_feedback_default_none():
    trace = Trace(
        input_ids=torch.tensor([[1]]),
        steered_input_ids=torch.tensor([[1]]),
        output=_make_output(),
        score=0.5,
    )
    assert trace.feedback is None


def test_trace_feedback_explicit():
    trace = Trace(
        input_ids=torch.tensor([[1]]),
        steered_input_ids=torch.tensor([[1]]),
        output=_make_output(),
        score=0.5,
        feedback="critique",
    )
    assert trace.feedback == "critique"
