"""Vector calibration workbench.

Orchestrates three stages for producing calibrated steering vectors:

  1. Generation of contrastive pairs via an LLM.
  2. Extraction of a steering vector from paired hidden states.
  3. Calibration of the vector via a grid sweep over (layer, multiplier).
"""
from .calibration import CalibrationSweep
from .configs import (
    CalibrationBuilderConfig,
    CalibrationConfig,
    ExtractionConfig,
    GenerationConfig,
    JudgeConfig,
    QualityGate,
    SweepGrid,
)
from .extraction import SteeringVectorExtractor
from .generation import ContrastivePairGenerator
from .results import CalibrationResult, CellResult, GenerationResult
from .workbench import VectorCalibrationWorkbench

__all__ = [
    "VectorCalibrationWorkbench",
    "ContrastivePairGenerator",
    "SteeringVectorExtractor",
    "CalibrationSweep",
    "CalibrationBuilderConfig",
    "GenerationConfig",
    "ExtractionConfig",
    "CalibrationConfig",
    "JudgeConfig",
    "SweepGrid",
    "QualityGate",
    "GenerationResult",
    "CalibrationResult",
    "CellResult",
]
