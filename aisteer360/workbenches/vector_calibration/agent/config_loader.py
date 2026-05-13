"""Rehydrate a `CalibrationBuilderConfig` from the JSON blob returned by the server."""
from __future__ import annotations

import logging

from aisteer360.workbenches.vector_calibration.configs import (
    CalibrationBuilderConfig,
    CalibrationConfig,
    ExtractionConfig,
    GenerationConfig,
    JudgeConfig,
    QualityGate,
    SweepGrid,
)

logger = logging.getLogger(__name__)


def _coerce_generation(raw: dict) -> GenerationConfig:
    known = {f.name for f in GenerationConfig.__dataclass_fields__.values()}
    return GenerationConfig(**{k: v for k, v in raw.items() if k in known})


def _coerce_extraction(raw: dict) -> ExtractionConfig:
    known = {f.name for f in ExtractionConfig.__dataclass_fields__.values()}
    return ExtractionConfig(**{k: v for k, v in raw.items() if k in known})


def _coerce_judge(raw: dict) -> JudgeConfig:
    known = {f.name for f in JudgeConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known}
    rs = filtered.get("rating_scale")
    if rs is not None:
        filtered["rating_scale"] = [(int(r[0]), str(r[1])) for r in rs]
    scale = filtered.get("scale")
    if scale is not None:
        filtered["scale"] = (int(scale[0]), int(scale[1]))
    return JudgeConfig(**filtered)


def _coerce_calibration(raw: dict) -> CalibrationConfig:
    known = {f.name for f in CalibrationConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known}
    filtered["judge"] = _coerce_judge(raw["judge"])
    if "sweep" in filtered:
        sweep_fields = {f.name for f in SweepGrid.__dataclass_fields__.values()}
        sweep_raw = {k: v for k, v in filtered["sweep"].items() if k in sweep_fields}
        if "multiplier_range" in sweep_raw and sweep_raw["multiplier_range"] is not None:
            sweep_raw["multiplier_range"] = tuple(sweep_raw["multiplier_range"])
        if "layer_range" in sweep_raw and sweep_raw["layer_range"] is not None:
            sweep_raw["layer_range"] = tuple(sweep_raw["layer_range"])
        filtered["sweep"] = SweepGrid(**sweep_raw)
    if "quality_gate" in filtered:
        qg_fields = {f.name for f in QualityGate.__dataclass_fields__.values()}
        filtered["quality_gate"] = QualityGate(
            **{k: v for k, v in filtered["quality_gate"].items() if k in qg_fields}
        )
    return CalibrationConfig(**filtered)


def from_server_config(raw: dict) -> CalibrationBuilderConfig:
    """Materialise a `CalibrationBuilderConfig` from a server-returned dict."""
    return CalibrationBuilderConfig(
        steered_model=raw["steered_model"],
        generation=_coerce_generation(raw["generation"]),
        extraction=_coerce_extraction(raw["extraction"]),
        calibration=_coerce_calibration(raw["calibration"]),
        hf_model_kwargs=raw.get("hf_model_kwargs", {}) or {},
        device_map=raw.get("device_map") or "auto",
        save_dir=raw.get("save_dir"),
    )
