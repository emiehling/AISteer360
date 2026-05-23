"""Tests for `SCOPEArgs` validation."""
from __future__ import annotations

import pytest

from aisteer360.algorithms.input_control.scope.args import SCOPEArgs


def _stub_lm(prompt: str) -> str:
    return ""


def test_args_validation_n_candidates():
    with pytest.raises(ValueError, match="n_candidates"):
        SCOPEArgs(reflection_lm=_stub_lm, n_candidates=0)


def test_args_validation_confidence_threshold():
    with pytest.raises(ValueError, match="confidence_threshold"):
        SCOPEArgs(reflection_lm=_stub_lm, confidence_threshold=1.5)
    with pytest.raises(ValueError, match="confidence_threshold"):
        SCOPEArgs(reflection_lm=_stub_lm, confidence_threshold=-0.1)


def test_args_validation_strategic_max_size():
    with pytest.raises(ValueError, match="strategic_max_size"):
        SCOPEArgs(reflection_lm=_stub_lm, strategic_max_size=0)


def test_args_validation_tactical_max_size_optional():
    args = SCOPEArgs(reflection_lm=_stub_lm, tactical_max_size=None)
    assert args.tactical_max_size is None
    with pytest.raises(ValueError, match="tactical_max_size"):
        SCOPEArgs(reflection_lm=_stub_lm, tactical_max_size=0)


def test_args_no_training_data_required():
    args = SCOPEArgs(reflection_lm=_stub_lm)
    assert args.reflection_lm is _stub_lm
    assert args.seed_rules is None
