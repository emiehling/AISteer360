"""Tests for `agent.config_loader.from_server_config`."""
from __future__ import annotations

from aisteer360.workbenches.vector_calibration.agent.config_loader import from_server_config
from aisteer360.workbenches.vector_calibration.configs import CalibrationBuilderConfig

from ..conftest import minimal_config


def test_roundtrip_minimal() -> None:
    cfg = from_server_config(minimal_config())
    assert isinstance(cfg, CalibrationBuilderConfig)
    assert cfg.steered_model == "test/model"
    assert cfg.generation.behavior == "warmth"
    assert cfg.calibration.judge.model == "test/model"


def test_drops_unknown_fields() -> None:
    raw = minimal_config()
    raw["generation"]["something_unexpected"] = "ignore me"
    raw["future_top_level_key"] = 123
    cfg = from_server_config(raw)
    assert cfg.generation.behavior == "warmth"


def test_tuple_coercion_for_sweep() -> None:
    raw = minimal_config()
    raw["calibration"]["sweep"] = {
        "multiplier_range": [-2.0, 2.0],
        "multiplier_step": 0.5,
        "layer_range": [0, 7],
        "layer_step": 1,
    }
    cfg = from_server_config(raw)
    assert cfg.calibration.sweep.multiplier_range == (-2.0, 2.0)
    assert cfg.calibration.sweep.layer_range == (0, 7)
