"""An OpenEvolve search over a study's candidates.

OpenEvolve evolves program texts with a language model. Each program here is one candidate:

```
# steerability-candidate/1 study=<study id>
# EVOLVE-BLOCK-START
{"structure": "<structure name>", "values": {"<axis path>": <value>, ...}}
# EVOLVE-BLOCK-END
```

Since the header line is outside the block, `enforce_evolve_blocks` keeps it fixed. OpenEvolve runs in a subprocess
(`_openevolve_controller`), since it starts worker processes that must not be forked from a process holding a GPU
model. Its evaluator calls an `EvaluationService` over the study's session, and its population strategy admits and
replaces programs by the archive's latest results rather than by OpenEvolve's stored metrics. An initial program is
therefore always admitted, and a program whose evaluation timed out is kept out. Re-scored results replace the stale
ones.

This module and the controller are the only modules that import `openevolve` (the `optimize` extra).
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steerability.optimization.evaluation.service import EvaluationService
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.search._openevolve_controller import ADDRESS_ENV, KEY_ENV
from steerability.optimization.search.description import (
    candidate_document,
    describe_axes,
    describe_structure,
    describe_task,
    parse_document,
)
from steerability.optimization.study import Candidate
from steerability.utils.optional import require

if TYPE_CHECKING:
    from steerability.algorithms.core.store import Store
    from steerability.algorithms.core.sweeps import PipelineFactory
    from steerability.optimization.evaluation.archive import Archive
    from steerability.optimization.study import Study

logger = logging.getLogger(__name__)

PROGRAM_SCHEMA = "steerability-candidate/1"
BLOCK_START = "# EVOLVE-BLOCK-START"
BLOCK_END = "# EVOLVE-BLOCK-END"
CONTROLLER = "steerability.optimization.search._openevolve_controller"
POLL_SECONDS = 5.0
TERMINATE_SECONDS = 60.0

EVALUATOR_SOURCE = '''import json
import os
from multiprocessing.connection import Client

from openevolve.evaluation_result import EvaluationResult


def _call(op, program_path):
    with open(program_path, encoding="utf-8") as handle:
        program = handle.read()
    address = os.environ["STEERABILITY_EVAL_ADDRESS"]
    authkey = bytes.fromhex(os.environ["STEERABILITY_EVAL_KEY"])
    with Client(address, authkey=authkey) as connection:
        connection.send_bytes(json.dumps({"op": op, "program": program}).encode("utf-8"))
        reply = json.loads(connection.recv_bytes().decode("utf-8"))
    return EvaluationResult(metrics=reply["metrics"], artifacts=reply.get("artifacts", {}))


def evaluate_stage1(program_path):
    return _call("stage1", program_path)


def evaluate_stage2(program_path):
    return _call("stage2", program_path)


def evaluate(program_path):
    return _call("full", program_path)
'''


# OpenEvolve's default user template for full rewrites, with a task section that describes a steering
# configuration in place of code
USER_TEMPLATE = '''# Current Program Information
- Fitness: {fitness_score}
- Feature coordinates: {feature_coords}
- Focus areas: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Propose a new program with a higher FITNESS SCORE.
The system maintains diversity across these dimensions: {feature_dimensions}
Different programs with similar fitness but different features are valuable.

A program here is a steering configuration, not code. Change only the JSON object between the block markers, and keep
the header line and the two marker lines exactly as they are.

Reply with the complete program in one fenced block:

```{language}
# the complete program
```
'''


def program_text(study: Study, candidate: Candidate) -> str:
    """The program text of a candidate: the header line and the candidate's document between the block markers."""
    document = candidate_document(candidate=candidate)
    return f"# {PROGRAM_SCHEMA} study={study.id}\n{BLOCK_START}\n{document}\n{BLOCK_END}\n"


def parse(program: str) -> Candidate:
    """The candidate of a program: the single JSON document between the block markers (`parse_document`).

    Whether the candidate is valid is `Study.bind`'s decision.

    Args:
        program: The program text.

    Returns:
        The candidate.

    Raises:
        ValueError: If the program does not have exactly one block, or the block is not one candidate document.
    """
    lines = program.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == BLOCK_START]
    ends = [index for index, line in enumerate(lines) if line.strip() == BLOCK_END]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError(f"A program needs exactly one block between {BLOCK_START!r} and {BLOCK_END!r}.")
    return parse_document(text="\n".join(lines[starts[0] + 1:ends[0]]))


def system_message(*, study: Study, structure: str, goal: str) -> str:
    """The system message of an island that searches one structure.

    The message states the task and its goal (`describe_task`), the structure's controls and solved setting
    (`describe_structure`), and each of its proposer axes (`describe_axes`). It ends with the reply format and the
    structure's default candidate as an example reply. It describes no other structure, since a model that is told
    about the other structures proposes them, and an island admits its own structure only. It contains no benchmark
    item text.

    Args:
        study: The study.
        structure: The island's structure.
        goal: What the behavior score measures (see `describe_task`).

    Returns:
        The message.

    Raises:
        ValueError: If `goal` is not a non-empty string.
        RecipeError: If the structure is unknown.
    """
    paragraphs = [describe_task(goal=goal), describe_structure(study=study, structure=structure)]
    axes = describe_axes(study=study, structure=structure)
    paragraphs.append(f"Its settings, keyed by path, are:\n{axes}" if axes else "It has no settings to propose.")
    paragraphs.append(
        f"Reply with the complete program. Keep the header line and the `{BLOCK_START}` and `{BLOCK_END}` lines "
        f"unchanged, and put exactly one JSON object between them, with the keys \"structure\" (always "
        f"\"{structure}\") and \"values\" (an object with one entry per setting of the structure)."
    )
    example = program_text(study=study, candidate=study.default_candidate(structure=structure)).rstrip("\n")
    paragraphs.append(f"For instance, a valid reply for this structure is:\n\n```text\n{example}\n```")
    return "\n\n".join(paragraphs)


def write_run_files(
    *,
    study: Study,
    run_dir: str | Path,
    goal: str,
    llm: Mapping[str, Any],
    iterations: int,
    stage_timeout_seconds: int,
) -> dict[str, Path]:
    """Write the programs, the evaluator, and the configuration of an OpenEvolve run.

    The initial program is the default candidate of the first structure, and every further structure gets an island
    that begins from its own default candidate. The configuration has one island per structure with the island's own
    system message (`system_message`), OpenEvolve's built-in feature dimensions, full rewrites inside the enforced
    block, and a two-stage cascade with one evaluation at a time. Its user template (`USER_TEMPLATE`, in
    `<run_dir>/templates/full_rewrite_user.txt`) asks for a steering configuration rather than code, and the hint to
    simplify a long program is turned off (`suggest_simplification_after_chars: null`).

    Args:
        study: The study.
        run_dir: The run directory.
        goal: What the behavior score measures (see `describe_task`).
        llm: OpenEvolve's `llm` configuration, written unchanged.
        iterations: The number of iterations.
        stage_timeout_seconds: OpenEvolve's timeout of each evaluation stage.

    Returns:
        Mapping with the paths of `"initial_program"`, `"evaluator"`, `"config"`, `"user_template"`, and each
        further island's program (`"island_<i>"`).

    Raises:
        ValueError: If `goal` is not a non-empty string.
    """
    import yaml

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    structures = list(study.space)
    paths: dict[str, Path] = {"initial_program": run_dir / "initial_program.txt"}
    first = study.default_candidate(structure=structures[0])
    paths["initial_program"].write_text(program_text(study=study, candidate=first), encoding="utf-8")
    initial_programs: list[str | None] = [None]
    for index, structure in enumerate(structures[1:], start=1):
        path = run_dir / f"island_{index}_program.txt"
        candidate = study.default_candidate(structure=structure)
        path.write_text(program_text(study=study, candidate=candidate), encoding="utf-8")
        paths[f"island_{index}"] = path
        initial_programs.append(str(path.resolve()))
    paths["evaluator"] = run_dir / "evaluator.py"
    paths["evaluator"].write_text(EVALUATOR_SOURCE, encoding="utf-8")
    templates = run_dir / "templates"
    templates.mkdir(exist_ok=True)
    paths["user_template"] = templates / "full_rewrite_user.txt"
    paths["user_template"].write_text(USER_TEMPLATE, encoding="utf-8")
    messages = [system_message(study=study, structure=structure, goal=goal) for structure in structures]
    config = {
        "max_iterations": int(iterations),
        "random_seed": study.seed,
        "language": "text",
        "enforce_evolve_blocks": True,
        "diff_based_evolution": False,
        "llm": json.loads(json.dumps(dict(llm))),
        "prompt": {
            "system_message": messages[0],
            "island_system_messages": messages,
            "template_dir": str(templates.resolve()),
            "suggest_simplification_after_chars": None,
        },
        "database": {
            "num_islands": len(structures),
            "island_initial_programs": initial_programs,
            "feature_dimensions": ["complexity", "diversity"],
            "feature_bins": 6,
        },
        "evaluator": {
            "timeout": int(stage_timeout_seconds),
            "cascade_evaluation": True,
            "cascade_thresholds": [0.5],
            "parallel_evaluations": 1,
        },
    }
    paths["config"] = run_dir / "config.yaml"
    paths["config"].write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return paths


def require_openevolve() -> None:
    """Check that OpenEvolve is installed and its `run_evolution` takes a `population_strategy`.

    Raises:
        ModuleNotFoundError: If OpenEvolve is not installed (the message names the `optimize` extra).
        RuntimeError: If the installed `run_evolution` has no `population_strategy` parameter.
    """
    require("openevolve")
    from openevolve.api import run_evolution

    if "population_strategy" not in inspect.signature(run_evolution).parameters:
        raise RuntimeError(
            "The installed OpenEvolve's run_evolution has no population_strategy parameter; install OpenEvolve 0.4.1 "
            "or later (pip install \"steerability[optimize]\")."
        )


def run_openevolve(
    study: Study,
    *,
    store: Store,
    run_dir: str | Path,
    goal: str,
    llm: Mapping[str, Any],
    iterations: int,
    max_generations: int | None = None,
    stage_timeout_seconds: int = 3600,
    factory: PipelineFactory | None = None,
    allow_infeasible: bool = False,
) -> Archive:
    """Search a study with OpenEvolve.

    The run files are written to `run_dir`, a session is started, and an `EvaluationService` serves it with a
    deadline of 90% of `stage_timeout_seconds` per operation. The service's deadline, rather than OpenEvolve's
    timeout, stops a long evaluation. OpenEvolve runs in a subprocess and resumes from the latest checkpoint under
    `<run_dir>/openevolve` when there is one. The subprocess is terminated once the session is exhausted.

    Args:
        study: The study.
        store: The store.
        run_dir: The run directory.
        goal: What the behavior score measures (see `describe_task`).
        llm: OpenEvolve's `llm` configuration (models, API base, and so on), written unchanged.
        iterations: The number of OpenEvolve iterations.
        max_generations: The number of generations the session may run, or None.
        stage_timeout_seconds: OpenEvolve's timeout of each evaluation stage.
        factory: The pipeline factory, or None to build one.
        allow_infeasible: Whether to accept a study that fails the feasibility check.

    Returns:
        The study's archive.

    Raises:
        ModuleNotFoundError: If OpenEvolve is not installed.
        RuntimeError: If the installed OpenEvolve has no population strategy hooks.
        ValueError: If `goal` is not a non-empty string.
    """
    require_openevolve()
    run_dir = Path(run_dir)
    write_run_files(
        study=study,
        run_dir=run_dir,
        goal=goal,
        llm=llm,
        iterations=iterations,
        stage_timeout_seconds=stage_timeout_seconds,
    )
    session = StudySession(study, store=store, factory=factory, max_generations=max_generations,
                           allow_infeasible=allow_infeasible)
    service: EvaluationService | None = None
    process: subprocess.Popen | None = None
    try:
        session.start()
        service = EvaluationService(session, proposer="openevolve", parse=parse,
                                    stage_deadline_seconds=0.9 * stage_timeout_seconds)
        address, authkey = service.start()
        environment = {**os.environ, ADDRESS_ENV: address, KEY_ENV: authkey.hex()}
        process = subprocess.Popen([sys.executable, "-m", CONTROLLER, str(run_dir)], env=environment)
        while process.poll() is None:
            if session.exhausted():
                logger.info("The session is exhausted; stopping OpenEvolve.")
                process.terminate()
                try:
                    process.wait(timeout=TERMINATE_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(POLL_SECONDS)
        process.wait()
        if process.returncode != 0 and not session.exhausted():
            logger.warning("OpenEvolve exited with code %s.", process.returncode)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if service is not None:
            service.stop()
            logger.info("OpenEvolve sent %d programs: %d did not parse and %d did not bind.", service.programs,
                        service.unparsed, service.unbound)
        session.close()
    return session.archive


__all__ = [
    "EVALUATOR_SOURCE",
    "USER_TEMPLATE",
    "parse",
    "program_text",
    "require_openevolve",
    "run_openevolve",
    "system_message",
    "write_run_files",
]
