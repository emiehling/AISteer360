"""Tests for `EPRArgs` validation."""
from __future__ import annotations

import pytest

from aisteer360.algorithms.input_control.epr.args import EPRArgs


def _row(input_: str = "x", output_: str = "y") -> dict:
    return {"input": input_, "output": output_}


def test_args_validation_empty_corpus():
    with pytest.raises(ValueError, match="corpus"):
        EPRArgs(corpus=[])


def test_args_validation_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        EPRArgs(corpus=[_row()], mode="invalid")  # type: ignore[arg-type]


def test_args_validation_n_demonstrations():
    with pytest.raises(ValueError, match="n_demonstrations"):
        EPRArgs(corpus=[_row()], n_demonstrations=0)


def test_args_validation_candidate_set_size_too_small():
    with pytest.raises(ValueError, match="candidate_set_size"):
        EPRArgs(
            corpus=[_row()],
            candidate_set_size=2,
            n_positives=2,
            n_negatives=2,
        )


def test_args_validation_n_positives_negatives_minimum():
    with pytest.raises(ValueError, match="n_positives|n_negatives"):
        EPRArgs(corpus=[_row()], n_positives=0)


def test_args_validation_row_schema():
    bad_row = {"input": "x"}  # missing "output"
    with pytest.raises(ValueError, match=r"corpus\[0\]"):
        EPRArgs(corpus=[bad_row])


def test_args_default_mode_is_epr():
    args = EPRArgs(corpus=[_row()])
    assert args.mode == "epr"
