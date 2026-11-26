"""An OpenEvolve search engine over a study's candidates.

OpenEvolve evolves program texts with a language model. Each program here is one candidate:

```
# steerability-candidate/1 study=<study id>
# EVOLVE-BLOCK-START
{"structure": "<structure name>", "values": {"<axis path>": <value>, ...}}
# EVOLVE-BLOCK-END
```

Since the header line is outside the block, `enforce_evolve_blocks` keeps it fixed. Each island of the run searches
one structure and begins from one member of the engine's initial population, which is taken from the archive. Its
system message describes that structure only, and its population admits that structure only. OpenEvolve runs in a
subprocess (`_openevolve_controller`), since it starts worker processes that must not be forked from a process
holding a GPU model. Its evaluator calls an `EvaluationService` over the run's session, and its population strategy
admits and replaces programs by the archive's latest results rather than by OpenEvolve's stored metrics.

This module and the controller are the only modules that import `openevolve` (the `optimize` extra).
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steerability.optimization.evaluation.service import EvaluationService
from steerability.optimization.search._openevolve_controller import ADDRESS_ENV, KEY_ENV, latest_checkpoint
from steerability.optimization.search.description import (
    candidate_document,
    describe_axes,
    describe_structure,
    describe_task,
    parse_document,
)
from steerability.optimization.search.engine import SearchEngine, select_structures
from steerability.optimization.search.population import initial_population
from steerability.optimization.study import Candidate
from steerability.utils.optional import require

if TYPE_CHECKING:
    from steerability.optimization.evaluation.session import StudySession
    from steerability.optimization.study import Study

logger = logging.getLogger(__name__)

PROGRAM_SCHEMA = "steerability-candidate/1"
BLOCK_START = "# EVOLVE-BLOCK-START"
BLOCK_END = "# EVOLVE-BLOCK-END"
CONTROLLER = "steerability.optimization.search._openevolve_controller"
OUTPUT_DIR = "openevolve"
ISLANDS_FILE = "islands.json"
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
# configuration in place of code and the levels that the population keeps apart
USER_TEMPLATE = '''# Current Program Information
- Fitness: {fitness_score}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Propose a new program with a higher FITNESS SCORE.

A program here is a steering configuration, not code. Change only the JSON object between the block markers, and keep
the header line and the two marker lines exactly as they are.

Each accepted candidate has a level, the share of the capability margin it uses (from 0 to 1). The search keeps the
best candidate at each level. A candidate at a level that has no good candidate yet is worth proposing even when its
fitness is lower. The recent rejections and the programs that could not be evaluated, when listed above, show what
not to propose again.

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
    islands: Sequence[Candidate],
    goal: str,
    llm: Mapping[str, Any],
    iterations: int,
    stage_timeout_seconds: int,
    level_bins: int,
    elite_size: int,
    checkpoint_interval: int,
) -> dict[str, Path]:
    """Write the programs, the island assignment, the evaluator, and the configuration of an OpenEvolve run.

    Island `i` begins from the candidate `islands[i]` and searches its structure. The files are the initial program
    (island 0, `initial_program.txt`), one `island_<i>_program.txt` per further island, `islands.json` (one
    `{"island": i, "structure": ...}` record per island), the evaluator, the user template (`USER_TEMPLATE`, in
    `templates/full_rewrite_user.txt`), and `config.yaml`. The configuration has one island per entry of `islands`
    with the island's own system message (`system_message`), full rewrites inside the enforced block, and a two-stage
    cascade with one evaluation at a time. The population keeps one program per level bin and island
    (`feature_dimensions: ["level"]`), its elite archive has `elite_size` programs, and every program's artifacts
    reach the prompts (`max_snapshot_artifacts: null`). The hint to simplify a long program is turned off
    (`suggest_simplification_after_chars: null`).

    Args:
        study: The study.
        run_dir: The run directory.
        islands: The candidate each island begins from, at least one.
        goal: What the behavior score measures (see `describe_task`).
        llm: OpenEvolve's `llm` configuration, written unchanged.
        iterations: The number of iterations.
        stage_timeout_seconds: OpenEvolve's timeout of each evaluation stage.
        level_bins: The number of level bins of an island's population.
        elite_size: The size of OpenEvolve's elite archive (`database.archive_size`).
        checkpoint_interval: The number of iterations between checkpoints.

    Returns:
        Mapping with the paths of `"initial_program"`, each further island's program (`"island_<i>"`),
        `"islands"`, `"evaluator"`, `"user_template"`, and `"config"`.

    Raises:
        ValueError: If `islands` is empty or `goal` is not a non-empty string.
    """
    import yaml

    islands = list(islands)
    if not islands:
        raise ValueError("An OpenEvolve run needs at least one island.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {"initial_program": run_dir / "initial_program.txt"}
    paths["initial_program"].write_text(program_text(study=study, candidate=islands[0]), encoding="utf-8")
    initial_programs: list[str | None] = [None]
    for index, candidate in enumerate(islands[1:], start=1):
        path = run_dir / f"island_{index}_program.txt"
        path.write_text(program_text(study=study, candidate=candidate), encoding="utf-8")
        paths[f"island_{index}"] = path
        initial_programs.append(str(path.resolve()))
    paths["islands"] = run_dir / ISLANDS_FILE
    records = [{"island": index, "structure": candidate.structure} for index, candidate in enumerate(islands)]
    paths["islands"].write_text(json.dumps(records, indent=2), encoding="utf-8")
    paths["evaluator"] = run_dir / "evaluator.py"
    paths["evaluator"].write_text(EVALUATOR_SOURCE, encoding="utf-8")
    templates = run_dir / "templates"
    templates.mkdir(exist_ok=True)
    paths["user_template"] = templates / "full_rewrite_user.txt"
    paths["user_template"].write_text(USER_TEMPLATE, encoding="utf-8")
    messages = [system_message(study=study, structure=candidate.structure, goal=goal) for candidate in islands]
    config = {
        "max_iterations": int(iterations),
        "checkpoint_interval": int(checkpoint_interval),
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
            "num_islands": len(islands),
            "island_initial_programs": initial_programs,
            "feature_dimensions": ["level"],
            "feature_bins": {"level": int(level_bins)},
            "archive_size": int(elite_size),
            "max_snapshot_artifacts": None,
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


class OpenEvolveEngine(SearchEngine):
    """A search engine that runs OpenEvolve, which evolves the candidates of a study with a language model.

    Each island of the run searches one structure and begins from one member of the initial population
    (`initial_population`), which is drawn from the best results of the initial candidates and of every earlier
    phase. The island's system message states the goal and describes its structure only, and the island admits
    accepted candidates of that structure only (each once), with no migration between islands. The population keeps
    the best candidate per level bin, as the study spreads its finalists over levels. A final evaluation reply lists
    the recent rejections of the candidate's structure and the programs that could not be evaluated, which reach the
    model through the artifacts of the programs it is shown.

    A structure without proposer axes has one candidate, which is evaluated once and gets no island. A run directory
    that contains an OpenEvolve checkpoint resumes from it with the island assignment of its `islands.json`. The
    other run files are rewritten, and a changed `goal`, `llm`, or `iterations` takes effect.

    Each evaluation operation has a deadline of 90% of `stage_timeout_seconds`, which stops a long evaluation before
    OpenEvolve's own timeout does. A training or merge of a structural control runs inside a stage and is not
    interrupted by a deadline. For a phase over a structure that trains, `stage_timeout_seconds` must therefore cover
    one training plus screening, or the phase runs with `structural_fits=0` on the products an earlier phase trained.
    When OpenEvolve gives up on a call first, the result is still recorded, and a later proposal of the candidate is
    a lookup. The numeric defaults follow OpenEvolve's own short-run examples and were not measured on a study.

    Args:
        run_dir: The run directory (programs, configuration, and OpenEvolve's output under `<run_dir>/openevolve`).
        llm: OpenEvolve's `llm` configuration (models, API base, and so on), written unchanged.
        iterations: The number of OpenEvolve iterations.
        goal: What the behavior score measures, written by the study's author without benchmark item text (see
            `describe_task`).
        structures: The structures to search, or None for every structure of the study.
        islands_per_structure: The largest number of islands per structure, each beginning from another member of
            the initial population.
        stage_timeout_seconds: OpenEvolve's timeout of each evaluation stage.
        rejection_digest: The largest number of recent rejections listed in a reply's artifacts.
        level_bins: The number of level bins of an island's population.
        elite_size: The size of OpenEvolve's elite archive.
        checkpoint_interval: The number of iterations between OpenEvolve checkpoints.

    Raises:
        ValueError: If `goal` is not a non-empty string, or a count is not a positive integer (a non-negative one for
            `rejection_digest`).
        TypeError: If `llm` is not a mapping.
    """

    name = "openevolve"

    def __init__(
        self,
        *,
        run_dir: str | Path,
        llm: Mapping[str, Any],
        iterations: int,
        goal: str,
        structures: Sequence[str] | None = None,
        islands_per_structure: int = 1,
        stage_timeout_seconds: int = 3600,
        rejection_digest: int = 5,
        level_bins: int = 6,
        elite_size: int = 15,
        checkpoint_interval: int = 10,
    ) -> None:
        describe_task(goal=goal)
        counts = {
            "iterations": iterations,
            "islands_per_structure": islands_per_structure,
            "stage_timeout_seconds": stage_timeout_seconds,
            "level_bins": level_bins,
            "elite_size": elite_size,
            "checkpoint_interval": checkpoint_interval,
            "rejection_digest": rejection_digest,
        }
        for key, value in counts.items():
            low = 0 if key == "rejection_digest" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                kind = "non-negative" if low == 0 else "positive"
                raise ValueError(f"{key} must be a {kind} integer; got {value!r}.")
        if not isinstance(llm, Mapping):
            raise TypeError(f"llm must be a mapping (OpenEvolve's llm configuration); got {type(llm).__name__}.")
        self.run_dir = Path(run_dir)
        self.llm = dict(llm)
        self.iterations = iterations
        self.goal = goal
        self.structures = None if structures is None else tuple(structures)
        self.islands_per_structure = islands_per_structure
        self.stage_timeout_seconds = stage_timeout_seconds
        self.rejection_digest = rejection_digest
        self.level_bins = level_bins
        self.elite_size = elite_size
        self.checkpoint_interval = checkpoint_interval

    def run(
        self,
        *,
        session: StudySession,
        initial_candidates: Sequence[Candidate] = (),
    ) -> None:
        """Search the session's study with OpenEvolve until the session is exhausted or OpenEvolve finishes.

        The steps are:

        1. OpenEvolve is required (`require_openevolve`).
        2. The default candidate of each structure in scope without proposer axes is evaluated, and the structure
           gets no island. Without a structure left, the run ends.
        3. With a checkpoint under `<run_dir>/openevolve`, the islands of `<run_dir>/islands.json` are kept and the
           initial population of this run is not used. Otherwise each island begins from a member of the initial
           population, and every member is evaluated through the session first (most are recorded already, which
           costs nothing). OpenEvolve's evaluation of its initial programs is then a lookup. The run ends when the
           session is then exhausted.
        4. A warning is logged for each structure in scope whose discrete proposer axes have fewer combinations than
           its share of the iterations, since a grid in an earlier phase covers it.
        5. The run files are written, an `EvaluationService` serves the session, and the controller runs OpenEvolve
           in a subprocess, which is terminated once the session is exhausted. The service is stopped and its
           counters are logged, also when a step raises an error.

        Args:
            session: The started session of the run.
            initial_candidates: The candidates given to `run_study`, already evaluated.

        Raises:
            ModuleNotFoundError: If OpenEvolve is not installed.
            RuntimeError: If the installed OpenEvolve has no population strategy hooks.
            ValueError: If the run directory contains a checkpoint without `islands.json`, or `islands.json` gives a
                structure that is not in the study's space.
        """
        require_openevolve()
        study = session.study
        in_scope: list[str] = []
        fixed: list[Candidate] = []
        for structure in select_structures(study=study, structures=self.structures):
            if study.proposer_axes(structure=structure):
                in_scope.append(structure)
            else:
                fixed.append(study.default_candidate(structure=structure))
        self._evaluate(session=session, candidates=fixed)
        if not in_scope or session.exhausted():
            return
        islands = self._resumed_islands(study=study)
        if islands is None:
            population = initial_population(
                study=study,
                view=session.archive.view(),
                initial_candidates=initial_candidates,
                per_structure=self.islands_per_structure,
                structures=in_scope,
            )
            islands = [candidate for structure in in_scope for candidate in population[structure]]
            self._evaluate(session=session, candidates=islands)
            if session.exhausted():
                return
        self._warn_small_structures(study=study, structures=in_scope, islands=len(islands))
        write_run_files(
            study=study,
            run_dir=self.run_dir,
            islands=islands,
            goal=self.goal,
            llm=self.llm,
            iterations=self.iterations,
            stage_timeout_seconds=self.stage_timeout_seconds,
            level_bins=self.level_bins,
            elite_size=self.elite_size,
            checkpoint_interval=self.checkpoint_interval,
        )
        self._search(session=session)

    def _evaluate(self, *, session: StudySession, candidates: Sequence[Candidate]) -> None:
        """Prepare the fits of candidates together and evaluate each through the session."""
        if not candidates:
            return
        session.prepare(candidates=candidates)
        for candidate in candidates:
            session.evaluate(candidate=candidate, proposer=self.name)

    def _resumed_islands(self, *, study: Study) -> list[Candidate] | None:
        """The islands of a run directory that contains a checkpoint, or None for a new run.

        Each island keeps its structure from `islands.json` and its program file's candidate (the structure's
        default candidate when the file is missing or of another structure).
        """
        checkpoint = latest_checkpoint(output_dir=self.run_dir / OUTPUT_DIR)
        if checkpoint is None:
            return None
        path = self.run_dir / ISLANDS_FILE
        if not path.exists():
            raise ValueError(
                f"{self.run_dir} contains an OpenEvolve checkpoint ({checkpoint}) and no {ISLANDS_FILE}, which records "
                "the structure of each island; use a new run directory."
            )
        islands: list[Candidate] = []
        for record in json.loads(path.read_text(encoding="utf-8")):
            index, structure = int(record["island"]), record["structure"]
            if structure not in study.space:
                raise ValueError(
                    f"{path} gives island {index} the structure {structure!r}, which is not in the study's space; "
                    f"its structures are {list(study.space)}."
                )
            program_path = self.run_dir / ("initial_program.txt" if index == 0 else f"island_{index}_program.txt")
            try:
                candidate = parse(program=program_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                candidate = None
            if candidate is None or candidate.structure != structure:
                candidate = study.default_candidate(structure=structure)
            islands.append(candidate)
        logger.info(
            "OpenEvolve resumes from %s with the %d island(s) of %s; the initial population of this run is not used.",
            checkpoint,
            len(islands),
            path,
        )
        return islands

    def _warn_small_structures(self, *, study: Study, structures: Sequence[str], islands: int) -> None:
        share = self.iterations / islands
        for structure in structures:
            axes = study.proposer_axes(structure=structure).values()
            if any(axis.continuous or axis.has_text for axis in axes):
                continue
            combinations = math.prod(axis.count for axis in axes)
            if combinations < share:
                logger.warning(
                    "Structure %r has %d combinations of its discrete settings, fewer than its share of %.0f "
                    "OpenEvolve iterations; GridProposer(study=study, structures=[%r]) in an earlier phase "
                    "evaluates each of them once.",
                    structure,
                    combinations,
                    share,
                    structure,
                )

    def _search(self, *, session: StudySession) -> None:
        """Serve the session, run the controller until it ends or the session is exhausted, and stop the service."""
        service = EvaluationService(
            session=session,
            proposer=self.name,
            parse=parse,
            stage_deadline_seconds=0.9 * self.stage_timeout_seconds,
            digest=self.rejection_digest,
        )
        process: subprocess.Popen | None = None
        try:
            address, authkey = service.start()
            environment = {**os.environ, ADDRESS_ENV: address, KEY_ENV: authkey.hex()}
            process = subprocess.Popen([sys.executable, "-m", CONTROLLER, str(self.run_dir)], env=environment)
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
            service.stop()
            logger.info("OpenEvolve sent %d programs: %d did not parse and %d did not bind.", service.programs,
                        service.unparsed, service.unbound)


__all__ = [
    "EVALUATOR_SOURCE",
    "OpenEvolveEngine",
    "USER_TEMPLATE",
    "parse",
    "program_text",
    "require_openevolve",
    "system_message",
    "write_run_files",
]
