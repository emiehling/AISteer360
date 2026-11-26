"""The OpenEvolve controller process of `OpenEvolveEngine`.

Run as `python -m steerability.optimization.search._openevolve_controller <run_dir>` with the evaluation service's
address in `STEERABILITY_EVAL_ADDRESS` and its authkey, in hex, in `STEERABILITY_EVAL_KEY`. The controller runs
`openevolve.api.run_evolution` on the files in `run_dir`, with a population strategy that asks the service for the
archive's estimates:

- `admit` keeps a program without a parent (an initial program, which OpenEvolve requires to be admitted) and
  otherwise keeps a program only when the archive accepted it with a positive fitness, it belongs to its island's
  structure (`islands.json`), and its candidate is new on the island (`admit_program`).
- `replace_cell` replaces an incumbent that the archive has not accepted, or one whose fitness is lower than the
  candidate's.
- `migration_due` is always false, since neighboring islands can search different structures.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steerability.optimization.evaluation.service import call

logger = logging.getLogger(__name__)

ADDRESS_ENV = "STEERABILITY_EVAL_ADDRESS"
KEY_ENV = "STEERABILITY_EVAL_KEY"

Estimate = Callable[[str], Mapping[str, Any]]

_UNKNOWN = {"known": False, "accepted": False, "fitness": 0.0, "candidate_id": None, "structure": None}


def _safe(estimate: Estimate, code: str) -> Mapping[str, Any]:
    try:
        return estimate(code)
    except Exception:
        logger.warning("The evaluation service did not answer an estimate.", exc_info=True)
        return dict(_UNKNOWN)


@dataclass
class Admitted:
    """The candidates admitted to each island, and the OpenEvolve programs whose candidates are counted.

    Attributes:
        candidates: Mapping from island index to the ids of the candidates admitted to it.
        programs: The ids of the OpenEvolve programs whose candidates are in `candidates`.
    """

    candidates: dict[int, set[str]] = field(default_factory=dict)
    programs: set[str] = field(default_factory=set)


def admit_program(
    *,
    snapshot: Any,
    candidate: Any,
    island: int,
    structures: Sequence[str],
    admitted: Admitted,
    estimate: Estimate,
) -> bool:
    """Whether to admit a program to an island.

    A program without a parent is admitted, since OpenEvolve requires it for the first program. Any other program is
    admitted when all of these are true, checked in this order:

    1. Its metrics have `combined_score`. When OpenEvolve's stage-2 timeout fires while the service is still
       evaluating, OpenEvolve keeps the stage-1 metrics only, and the fitness it would store is wrong. The result is
       recorded in the archive, and a later proposal of the candidate is a lookup with the correct score.
    2. The archive accepted its candidate with a positive fitness.
    3. Its candidate's structure is the island's.
    4. Its candidate is new on the island. OpenEvolve rejects a rewrite only when it is identical to its parent,
       and without this rule a known candidate (which costs no model work) would be admitted again.

    The candidate of every admitted program is recorded in `admitted`. The island's programs that `admitted` does
    not count yet (after a resume from a checkpoint) are counted first, with one estimate each.

    Args:
        snapshot: OpenEvolve's `PopulationSnapshot` (`islands` and `programs` are read).
        candidate: The program's `ProgramState` (`id`, `parent_id`, `code`, and `metrics` are read).
        island: The index of the island the program is added to.
        structures: The structure of each island, by island index.
        admitted: The record of admitted candidates, which the call updates.
        estimate: A function returning the archive's estimate of a program text.

    Returns:
        The decision.
    """
    known = admitted.candidates.setdefault(island, set())
    members = snapshot.islands[island] if island < len(snapshot.islands) else ()
    for program_id in members:
        if program_id in admitted.programs:
            continue
        admitted.programs.add(program_id)
        program = snapshot.programs.get(program_id)
        candidate_id = None if program is None else _safe(estimate=estimate, code=program.code).get("candidate_id")
        if candidate_id is not None:
            known.add(candidate_id)
    if candidate.parent_id is None:
        candidate_id = _safe(estimate=estimate, code=candidate.code).get("candidate_id")
        if candidate_id is not None:
            known.add(candidate_id)
        admitted.programs.add(candidate.id)
        return True
    if "combined_score" not in candidate.metrics:
        return False
    reply = _safe(estimate=estimate, code=candidate.code)
    if not reply.get("accepted") or float(reply.get("fitness", 0.0)) <= 0.0:
        return False
    if island >= len(structures) or reply.get("structure") != structures[island]:
        return False
    candidate_id = reply.get("candidate_id")
    if candidate_id is None or candidate_id in known:
        return False
    known.add(candidate_id)
    admitted.programs.add(candidate.id)
    return True


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

    from steerability.optimization.search.openevolve import ISLANDS_FILE, OUTPUT_DIR, require_openevolve

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
    records = json.loads((run_dir / ISLANDS_FILE).read_text(encoding="utf-8"))
    structures = [record["structure"] for record in sorted(records, key=lambda record: record["island"])]
    admitted = Admitted()

    def estimate(code: str) -> Mapping[str, Any]:
        return call(address, authkey, {"op": "estimate", "program": code})

    strategy = PopulationStrategy(
        admit=lambda snapshot, candidate, island: admit_program(
            snapshot=snapshot,
            candidate=candidate,
            island=island,
            structures=structures,
            admitted=admitted,
            estimate=estimate,
        ),
        replace_cell=lambda snapshot, candidate, incumbent, island: replace_program(
            candidate=candidate,
            incumbent=incumbent,
            estimate=estimate,
        ),
        migration_due=lambda snapshot: False,
    )
    config_path = run_dir / "config.yaml"
    iterations = int(yaml.safe_load(config_path.read_text(encoding="utf-8"))["max_iterations"])
    output_dir = run_dir / OUTPUT_DIR
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
