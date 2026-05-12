"""Launch the calibration dashboard server.

Usage:
    python -m aisteer360.workbenches.vector_calibration.interface \\
        --model ibm-granite/granite-3.3-2b-instruct \\
        --behavior warmth \\
        --port 8360
"""
from __future__ import annotations

import argparse
import logging

import uvicorn

from aisteer360.workbenches.vector_calibration import (
    CalibrationBuilderConfig,
    CalibrationConfig,
    ExtractionConfig,
    GenerationConfig,
    JudgeConfig,
)

from .app import create_app

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration dashboard server"
    )
    parser.add_argument(
        "--model", required=True, help="Steered model name or path"
    )
    parser.add_argument(
        "--behavior", default="warmth", help="Target behavior label"
    )
    parser.add_argument(
        "--save-dir", default="./runs", help="Artifact directory"
    )
    parser.add_argument("--port", type=int, default=8360)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--log-level", default="info", help="Uvicorn log level"
    )
    args = parser.parse_args()

    config = CalibrationBuilderConfig(
        steered_model=args.model,
        generation=GenerationConfig(
            generator_model=args.model,
            behavior=args.behavior,
            positive_prompt="",
            negative_prompt="",
        ),
        extraction=ExtractionConfig(),
        calibration=CalibrationConfig(
            judge=JudgeConfig(model=args.model, criteria=""),
        ),
        save_dir=args.save_dir,
    )

    app = create_app(config, save_dir=args.save_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
