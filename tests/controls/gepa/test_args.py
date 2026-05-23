"""Tests for GEPAArgs validation."""
from __future__ import annotations

import pytest

from aisteer360.algorithms.input_control.gepa.args import GEPAArgs
from aisteer360.algorithms.input_control.gepa.feedback_metric import ScoreOnlyFeedbackMetric

from tests.controls.common._stubs import StubMetric


def _base_kwargs(**overrides):
    defaults = dict(
        seed_instruction="hi",
        feedback_metric=ScoreOnlyFeedbackMetric(StubMetric()),
        reflection_lm=lambda p: p,
        max_metric_calls=10,
        train_data=[{"input_ids": [1, 2, 3]}],
    )
    defaults.update(overrides)
    return defaults


def test_args_validation_seed_instruction():
    with pytest.raises(ValueError):
        GEPAArgs(**_base_kwargs(seed_instruction=""))


def test_args_validation_max_metric_calls():
    with pytest.raises(ValueError):
        GEPAArgs(**_base_kwargs(max_metric_calls=0))


def test_args_validation_minibatch_size():
    with pytest.raises(ValueError):
        GEPAArgs(**_base_kwargs(reflection_minibatch_size=0))


def test_args_validation_train_data():
    with pytest.raises(ValueError):
        GEPAArgs(**_base_kwargs(train_data=[]))


def test_args_validation_merge_invocations():
    with pytest.raises(ValueError):
        GEPAArgs(**_base_kwargs(max_merge_invocations=-1))


def test_args_val_data_optional():
    args = GEPAArgs(**_base_kwargs(val_data=None))
    assert args.val_data is None
