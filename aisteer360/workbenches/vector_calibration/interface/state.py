"""Mutable server state shared across all routes."""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.workbenches.vector_calibration import (
    CalibrationBuilderConfig,
    CalibrationResult,
    GenerationResult,
    VectorCalibrationWorkbench,
)

logger = logging.getLogger(__name__)


class RunPhase(str, enum.Enum):
    IDLE = "idle"
    GENERATION = "generation"
    EXTRACTION = "extraction"
    CALIBRATION = "calibration"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class RunStatus:
    """Snapshot of the current or last-completed run."""
    phase: RunPhase = RunPhase.IDLE
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


class ServerState:
    """Holds config, builder, models, and results in memory."""

    def __init__(self, config: CalibrationBuilderConfig, save_dir: str | Path):
        self.config = config
        self.save_dir = Path(save_dir)

        self.builder: VectorCalibrationWorkbench | None = None
        self.run_status = RunStatus()
        self.run_lock = asyncio.Lock()

        self.generation_result: GenerationResult | None = None
        self.steering_vector: SteeringVector | None = None
        self.calibration_result: CalibrationResult | None = None

        self.model_info: dict[str, Any] = {}

        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    @property
    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def reset_cancel(self) -> None:
        self._cancel_requested = False

    def rebuild_builder(self) -> None:
        """Reconstruct the builder from current config (e.g. after a config change)."""
        if self.builder is not None:
            self.builder.cleanup()
        self.config.save_dir = str(self.save_dir)
        self.builder = VectorCalibrationWorkbench(self.config)

    def extract_model_info(self) -> dict[str, Any]:
        """Read metadata from the loaded steered model."""
        if self.builder is None or self.builder._model is None:
            return {}
        model = self.builder._model
        config = model.config
        return {
            "model_name": self.config.steered_model,
            "num_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
            "intermediate_size": getattr(config, "intermediate_size", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "dtype": str(model.dtype),
            "device": str(model.device),
            "model_type": getattr(config, "model_type", None),
        }

    def run_wall_time(self) -> float | None:
        """Elapsed wall time for the current or last run in seconds."""
        if self.run_status.started_at is None:
            return None
        end = self.run_status.finished_at or time.time()
        return round(end - self.run_status.started_at, 1)
