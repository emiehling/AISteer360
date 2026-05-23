"""Tests for `CPOArgs` validation."""
from __future__ import annotations

import pytest

from aisteer360.algorithms.input_control.cpo.args import CPOArgs


def _row(query: str = "q", prompt: str = "p", outcome: float = 0.5) -> dict:
    return {"query": query, "prompt": prompt, "outcome": outcome}


def test_args_validation_empty_training_data():
    with pytest.raises(ValueError, match="training_data"):
        CPOArgs(training_data=[], prompt_pool=["x"])


def test_args_validation_empty_pool():
    with pytest.raises(ValueError, match="prompt_pool"):
        CPOArgs(training_data=[_row()], prompt_pool=[])


def test_args_validation_n_folds():
    with pytest.raises(ValueError, match="n_folds"):
        CPOArgs(training_data=[_row()], prompt_pool=["x"], n_folds=1)


def test_args_validation_dim_reduction():
    with pytest.raises(ValueError, match="embedding_dim_reduction"):
        CPOArgs(training_data=[_row()], prompt_pool=["x"], embedding_dim_reduction=0)


def test_args_validation_temperature():
    with pytest.raises(ValueError, match="selection_temperature"):
        CPOArgs(training_data=[_row()], prompt_pool=["x"], selection_temperature=-0.1)


def test_args_validation_row_schema():
    bad_row = {"query": "q", "prompt": "p"}  # missing "outcome"
    with pytest.raises(ValueError, match=r"training_data\[0\].*outcome"):
        CPOArgs(training_data=[bad_row], prompt_pool=["x"])
