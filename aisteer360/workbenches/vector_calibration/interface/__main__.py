"""Launch the calibration dashboard server.

Usage:
    python -m aisteer360.workbenches.vector_calibration.interface \\
        --model ibm-granite/granite-3.3-2b-instruct \\
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
from .catalog import ALL_ROLES, CatalogEntry, load_catalog, save_catalog

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration dashboard server"
    )
    parser.add_argument(
        "--model", required=True, help="Steered model name or path"
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
            behavior="",
            positive_prompt="",
            negative_prompt="",
        ),
        extraction=ExtractionConfig(),
        calibration=CalibrationConfig(
            judge=JudgeConfig(model=args.model, criteria=""),
        ),
        save_dir=args.save_dir,
    )

    catalog = load_catalog()
    if not any(e.model_id == args.model for e in catalog):
        catalog.append(CatalogEntry(
            label=args.model.split("/")[-1],
            model_id=args.model,
            provider="hf",
            roles=list(ALL_ROLES),
        ))
        save_catalog(catalog)

    app = create_app(config, save_dir=args.save_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
