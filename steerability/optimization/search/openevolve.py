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
replaces programs by the archive's latest results rather than by OpenEvolve's stored metrics. A seed is therefore
always admitted, and a program whose evaluation timed out is kept out. Re-scored results replace the stale
ones.

This module and the controller are the only modules that import `openevolve` (the `optimize` extra).
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.recipe import Choice, Interval, Text
from steerability.optimization.evaluation.result import jsonable
from steerability.optimization.evaluation.service import EvaluationService
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.search._openevolve_controller import ADDRESS_ENV, KEY_ENV
from steerability.optimization.study import Candidate
from steerability.utils.optional import require

if TYPE_CHECKING:
    from steerability.algorithms.core.recipe import Axis
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
    """The program text of a candidate."""
    body = json.dumps({"structure": candidate.structure, "values": jsonable(dict(candidate.values))}, sort_keys=True)
    return f"# {PROGRAM_SCHEMA} study={study.id}\n{BLOCK_START}\n{body}\n{BLOCK_END}\n"


def parse(program: str) -> Candidate:
    """The candidate of a program: the single JSON object between the block markers.

    Whether the candidate is valid is `Study.bind`'s decision.

    Args:
        program: The program text.

    Returns:
        The candidate.

    Raises:
        ValueError: If the program does not have exactly one block, the block is not exactly one JSON object, or
            the object has keys other than `"structure"` and `"values"` (or values of the wrong type).
    """
    lines = program.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == BLOCK_START]
    ends = [index for index, line in enumerate(lines) if line.strip() == BLOCK_END]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError(f"A program needs exactly one block between {BLOCK_START!r} and {BLOCK_END!r}.")
    text = "\n".join(lines[starts[0] + 1:ends[0]]).strip()
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The block does not start with a JSON object ({error}).") from error
    if text[end:].strip():
        raise ValueError("The block contains more than one JSON value.")
    if not isinstance(value, dict):
        raise ValueError(f"The block's JSON value is a {type(value).__name__}, not an object.")
    if set(value) != {"structure", "values"}:
        raise ValueError(
            f"The block's object must have exactly the keys 'structure' and 'values'; got {sorted(value)}."
        )
    if not isinstance(value["structure"], str) or not isinstance(value["values"], dict):
        raise ValueError("'structure' must be a string and 'values' an object.")
    return Candidate(value["structure"], value["values"])


def _field_help(method: str, argument: str) -> str:
    """The help text of an argument of a registered method, from its `Args` field or its docstring."""
    from steerability.algorithms.core.registry import resolve_method_key

    try:
        args_cls = resolve_method_key(method).control_cls.Args
    except Exception:
        return ""
    if args_cls is None:
        return ""
    for item in dataclasses.fields(args_cls):
        if item.name == argument and item.metadata.get("help"):
            return " ".join(str(item.metadata["help"]).split())
    docstring = inspect.getdoc(args_cls) or ""
    match = re.search(r"^Attributes:\s*$(.*?)(?=^\S|\Z)", docstring, re.M | re.S)
    if match is None:
        return ""
    lines: list[str] = []
    for line in match.group(1).splitlines():
        found = re.match(r"^\s{4}(\w+):\s*(.*)$", line)
        if found is not None:
            if lines:
                break
            if found.group(1) == argument:
                lines.append(found.group(2))
        elif lines and line.strip():
            lines.append(line.strip())
    return " ".join(" ".join(lines).split())


def _describe_range(member: Any) -> str:
    if isinstance(member, Choice):
        return "one of " + ", ".join(json.dumps(jsonable(value)) for value in member.values)
    if isinstance(member, Interval):
        kind = "an integer" if member.integer else "a number"
        scale = " (log scale)" if member.log else ""
        return f"{kind} from {member.low:g} to {member.high:g}{scale}"
    if isinstance(member, Text):
        return f"a string of at most {member.max_chars} characters"
    return repr(member)


def _describe_axis(study: Study, structure: str, axis: Axis) -> list[str]:
    template = study.space[structure]
    methods = dict(zip(template.names, (item.method for item in template.entries)))
    lines = []
    if axis.key.startswith("@"):
        members = ", ".join(f"`{path}`" for path in axis.members)
        if axis.continuous:
            lines.append(f"- `{axis.key}`: a fraction from 0 to 1 that moves {members} together.")
        else:
            lines.append(f"- `{axis.key}`: a position index from 0 to {axis.count - 1} that moves {members} together.")
    for path, member in axis.members.items():
        name, _, rest = path.partition(".")
        argument = re.split(r"[.\[]", rest, maxsplit=1)[0]
        help_text = _field_help(methods.get(name, ""), argument)
        line = f"- `{path}`: {_describe_range(member)}."
        if axis.key.startswith("@"):
            line = f"  - `{path}` (member of `{axis.key}`): {_describe_range(member)}."
        lines.append(line + (f" {help_text}" if help_text else ""))
    return lines


def system_message(study: Study, structure: str | None = None) -> str:
    """The system message of the search, or of one island's structure.

    The message states the task, lists the structures, and, for an island, lists each proposer axis of its structure
    with its path, its range, and the help text of its `Args` field, followed by the structure's seed program as an
    example reply. It contains no benchmark item text.

    Args:
        study: The study.
        structure: The island's structure, or None for the message shared by every island.

    Returns:
        The message.
    """
    paragraphs = [
        "You propose steering configurations for a language model. Each program is one candidate: a structure (a "
        "fixed arrangement of steering controls) and a value for each of the structure's free settings. A candidate "
        "is scored by how much it raises a behavior score above the unsteered model while the model's accuracy on "
        "capability benchmarks stays within a fixed margin of the unsteered model's; a candidate outside the margin "
        "scores 0. Each structure may also have one setting that the evaluator solves for each candidate, which a "
        "program does not set.",
    ]
    lines = ["The structures are:"]
    for name, template in study.space.items():
        controls = ", ".join(item.method.partition("/")[2] for item in template.entries if item.enabled)
        solved = study.solved_axis(name)
        solved_note = f"; the evaluator solves `{solved.key}`" if solved is not None else ""
        lines.append(f"- `{name}`: {controls or 'no controls'}{solved_note}.")
    paragraphs.append("\n".join(lines))
    if structure is not None:
        axes = study.proposer_axes(structure)
        if axes:
            lines = [f"This island searches the structure `{structure}`. Its settings, keyed by path, are:"]
            for axis in axes.values():
                lines.extend(_describe_axis(study, structure, axis))
        else:
            lines = [f"This island searches the structure `{structure}`, which has no settings to propose."]
        paragraphs.append("\n".join(lines))
    paragraphs.append(
        f"Reply with the complete program. Keep the header line and the `{BLOCK_START}` and `{BLOCK_END}` lines "
        "unchanged, and put exactly one JSON object between them, with the keys \"structure\" and \"values\" "
        "(an object with one entry per setting of the structure)."
    )
    if structure is not None:
        example = program_text(study, study.seed_candidate(structure)).rstrip("\n")
        paragraphs.append(f"For instance, a valid reply for this structure is:\n\n```text\n{example}\n```")
    return "\n\n".join(paragraphs)


def write_run_files(study: Study, run_dir: str | Path, *, llm: Mapping[str, Any], iterations: int,
                    stage_timeout_seconds: int) -> dict[str, Path]:
    """Write the programs, the evaluator, and the configuration of an OpenEvolve run.

    The initial program is the seed candidate of the first structure, and every further structure gets an island
    seeded with its own seed candidate. The configuration has one island per structure, OpenEvolve's built-in
    feature dimensions, full rewrites inside the enforced block, and a two-stage cascade with one evaluation at a
    time. Its user template (`USER_TEMPLATE`, in `<run_dir>/templates/full_rewrite_user.txt`) asks for a steering
    configuration rather than code, and the hint to simplify a long program is turned off
    (`suggest_simplification_after_chars: null`).

    Args:
        study: The study.
        run_dir: The run directory.
        llm: OpenEvolve's `llm` configuration, written unchanged.
        iterations: The number of iterations.
        stage_timeout_seconds: OpenEvolve's timeout of each evaluation stage.

    Returns:
        Mapping with the paths of `"initial_program"`, `"evaluator"`, `"config"`, `"user_template"`, and each
        island seed (`"island_<i>"`).
    """
    import yaml

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    structures = list(study.space)
    paths: dict[str, Path] = {"initial_program": run_dir / "initial_program.txt"}
    paths["initial_program"].write_text(program_text(study, study.seed_candidate(structures[0])), encoding="utf-8")
    seeds: list[str | None] = [None]
    for index, structure in enumerate(structures[1:], start=1):
        path = run_dir / f"island_{index}_program.txt"
        path.write_text(program_text(study, study.seed_candidate(structure)), encoding="utf-8")
        paths[f"island_{index}"] = path
        seeds.append(str(path.resolve()))
    paths["evaluator"] = run_dir / "evaluator.py"
    paths["evaluator"].write_text(EVALUATOR_SOURCE, encoding="utf-8")
    templates = run_dir / "templates"
    templates.mkdir(exist_ok=True)
    paths["user_template"] = templates / "full_rewrite_user.txt"
    paths["user_template"].write_text(USER_TEMPLATE, encoding="utf-8")
    config = {
        "max_iterations": int(iterations),
        "random_seed": study.seed,
        "language": "text",
        "enforce_evolve_blocks": True,
        "diff_based_evolution": False,
        "llm": json.loads(json.dumps(dict(llm))),
        "prompt": {
            "system_message": system_message(study),
            "island_system_messages": [system_message(study, structure) for structure in structures],
            "template_dir": str(templates.resolve()),
            "suggest_simplification_after_chars": None,
        },
        "database": {
            "num_islands": len(structures),
            "island_initial_programs": seeds,
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
    """
    require_openevolve()
    run_dir = Path(run_dir)
    write_run_files(study, run_dir, llm=llm, iterations=iterations, stage_timeout_seconds=stage_timeout_seconds)
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
