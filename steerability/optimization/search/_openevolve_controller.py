"""The OpenEvolve controller process of `run_openevolve`.

Run as `python -m steerability.optimization.search._openevolve_controller <run_dir>` with the evaluation service's
address in `STEERABILITY_EVAL_ADDRESS` and its authkey, in hex, in `STEERABILITY_EVAL_KEY`. The controller runs
`openevolve.api.run_evolution` on the files in `run_dir`, with a population strategy that asks the service for the
archive's estimates:

- `admit` keeps a program without a parent (an initial program, which OpenEvolve requires to be admitted) and
  otherwise keeps a program only when the archive has accepted it with a positive fitness.
- `replace_cell` replaces an incumbent that the archive has not accepted, or one whose fitness is lower than the
  candidate's.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from steerability.optimization.evaluation.service import call

logger = logging.getLogger(__name__)

ADDRESS_ENV = "STEERABILITY_EVAL_ADDRESS"
KEY_ENV = "STEERABILITY_EVAL_KEY"

Estimate = Callable[[str], Mapping[str, Any]]


def _safe(estimate: Estimate, code: str) -> Mapping[str, Any]:
    try:
        return estimate(code)
    except Exception:
        logger.warning("The evaluation service did not answer an estimate.", exc_info=True)
        return {"known": False, "accepted": False, "fitness": 0.0}


def admit_program(candidate: Any, estimate: Estimate) -> bool:
    """Whether to admit a program: always without a parent, else when the archive accepted it with positive fitness.

    Args:
        candidate: An OpenEvolve `ProgramState` (`parent_id` and `code` are read).
        estimate: A function returning the archive's estimate of a program text.

    Returns:
        The decision.
    """
    if candidate.parent_id is None:
        return True
    reply = _safe(estimate, candidate.code)
    return bool(reply.get("accepted")) and float(reply.get("fitness", 0.0)) > 0.0


def replace_program(candidate: Any, incumbent: Any, estimate: Estimate) -> bool:
    """Whether a program replaces a cell's incumbent: when the incumbent is not accepted or has a lower fitness.

    Args:
        candidate: The new program's `ProgramState`.
        incumbent: The cell's current `ProgramState`.
        estimate: A function returning the archive's estimate of a program text.

    Returns:
        The decision.
    """
    incumbent_reply = _safe(estimate, incumbent.code)
    if not incumbent_reply.get("accepted"):
        return True
    candidate_reply = _safe(estimate, candidate.code)
    return float(candidate_reply.get("fitness", 0.0)) > float(incumbent_reply.get("fitness", 0.0))


def latest_checkpoint(output_dir: str | Path) -> str | None:
    """The checkpoint directory with the highest iteration under `<output_dir>/checkpoints`, or None."""
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return None
    found = []
    for path in root.iterdir():
        match = re.fullmatch(r"checkpoint_(\d+)", path.name)
        if match is not None and path.is_dir():
            found.append((int(match.group(1)), path))
    return str(max(found)[1]) if found else None


def main(argv: list[str] | None = None) -> int:
    """Run OpenEvolve on a run directory written by `write_run_files`.

    Returns:
        The exit code.
    """
    import yaml

    from steerability.optimization.search.openevolve import require_openevolve

    logging.basicConfig(level=logging.INFO)
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        logger.error("usage: python -m steerability.optimization.search._openevolve_controller <run_dir>")
        return 2
    require_openevolve()
    from openevolve.api import run_evolution
    from openevolve.controller import EvolutionFailedError
    from openevolve.population import PopulationStrategy

    run_dir = Path(arguments[0])
    address = os.environ[ADDRESS_ENV]
    authkey = bytes.fromhex(os.environ[KEY_ENV])

    def estimate(code: str) -> Mapping[str, Any]:
        return call(address, authkey, {"op": "estimate", "program": code})

    strategy = PopulationStrategy(
        admit=lambda snapshot, candidate, island: admit_program(candidate, estimate),
        replace_cell=lambda snapshot, candidate, incumbent, island: replace_program(candidate, incumbent, estimate),
    )
    config_path = run_dir / "config.yaml"
    iterations = int(yaml.safe_load(config_path.read_text(encoding="utf-8"))["max_iterations"])
    output_dir = run_dir / "openevolve"
    arguments_of_run = {
        "initial_program": str(run_dir / "initial_program.txt"),
        "evaluator": str(run_dir / "evaluator.py"),
        "config": str(config_path),
        "iterations": iterations,
        "output_dir": str(output_dir),
        "cleanup": False,
        "population_strategy": strategy,
    }
    checkpoint = latest_checkpoint(output_dir)
    try:
        run_evolution(**arguments_of_run, checkpoint_path=checkpoint)
    except EvolutionFailedError:
        raise
    except Exception:
        if checkpoint is None:
            raise
        logger.warning("Resuming from checkpoint %s failed; starting without it (results are memoized in the store).",
                       checkpoint, exc_info=True)
        run_evolution(**arguments_of_run, checkpoint_path=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
