"""Tests for the OpenEvolve adapter, without OpenEvolve (`search/openevolve.py`, `search/_openevolve_controller.py`)."""
import importlib.util
import json
import sys
import threading
import types
from dataclasses import dataclass, field
from multiprocessing.connection import Listener
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")
yaml = pytest.importorskip("yaml")

from steerability.algorithms.core.store import Store
from steerability.optimization.search._openevolve_controller import admit_program, latest_checkpoint, replace_program
from steerability.optimization.search.openevolve import (
    BLOCK_END,
    BLOCK_START,
    parse,
    program_text,
    require_openevolve,
    system_message,
    write_run_files,
)
from steerability.optimization.study import Candidate
from tests.optimization.helpers import Truth, build_study, caa_template, prompt_template

GOAL = "The behavior score measures how formal the reply reads."
TEMPLATE_PLACEHOLDERS = ("metrics", "fitness_score", "feature_coords", "feature_dimensions", "improvement_areas",
                         "evolution_history", "current_program", "language", "artifacts")


def _quiet(recipe, item_id):
    return Truth()


@pytest.fixture()
def study(tmp_path):
    store = Store.open(tmp_path / "store")
    return build_study(store, _quiet, space={"prompt": prompt_template(), "caa": caa_template()})


def _block(body: str) -> str:
    return f"# header\n{BLOCK_START}\n{body}\n{BLOCK_END}\n"


def test_parse_reads_the_program_of_a_candidate(study):
    candidate = Candidate(structure="prompt", values={"prompt.text": "be kind"})
    program = program_text(study=study, candidate=candidate)
    assert program.splitlines()[0] == f"# steerability-candidate/1 study={study.id}"
    assert parse(program=program) == candidate
    assert parse(program=_block('  {"structure": "caa", "values": {}}  ')) == Candidate(structure="caa", values={})


@pytest.mark.parametrize("program, message", [
    ('{"structure": "caa", "values": {}}', "exactly one block"),
    (_block("{}") + _block("{}"), "exactly one block"),
    (f"{BLOCK_END}\n{BLOCK_START}\n", "exactly one block"),
    (_block("structure = caa"), "JSON object"),
    (_block('{"structure": "caa", "values": {}, "notes": "x"}'), "exactly the keys"),
])
def test_parse_rejects_malformed_programs(program, message):
    with pytest.raises(ValueError, match=message):
        parse(program=program)


def test_the_run_files(study, tmp_path):
    llm = {"models": [{"name": "model", "api_base": "http://localhost:1"}], "temperature": 0.7}
    paths = write_run_files(
        study=study,
        run_dir=tmp_path / "run",
        goal=GOAL,
        llm=llm,
        iterations=7,
        stage_timeout_seconds=120,
    )
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    assert {key: config[key] for key in ("max_iterations", "random_seed", "language", "enforce_evolve_blocks",
                                         "diff_based_evolution", "llm")} == {
        "max_iterations": 7, "random_seed": study.seed, "language": "text", "enforce_evolve_blocks": True,
        "diff_based_evolution": False, "llm": llm,
    }
    messages = [system_message(study=study, structure=structure, goal=GOAL) for structure in ("prompt", "caa")]
    assert config["prompt"]["island_system_messages"] == messages
    assert config["prompt"]["system_message"] == messages[0]
    database = config["database"]
    assert database["num_islands"] == 2 and database["island_initial_programs"][0] is None
    assert database["feature_dimensions"] == ["complexity", "diversity"] and database["feature_bins"] == 6
    assert config["evaluator"] == {"timeout": 120, "cascade_evaluation": True, "cascade_thresholds": [0.5],
                                   "parallel_evaluations": 1}
    first = parse(program=paths["initial_program"].read_text(encoding="utf-8"))
    assert first == study.default_candidate(structure="prompt")
    island = Path(database["island_initial_programs"][1])
    assert island.is_absolute()
    assert parse(program=island.read_text(encoding="utf-8")) == study.default_candidate(structure="caa")
    assert config["prompt"]["suggest_simplification_after_chars"] is None
    template_dir = Path(config["prompt"]["template_dir"])
    assert template_dir.is_absolute() and paths["user_template"] == template_dir / "full_rewrite_user.txt"
    template = paths["user_template"].read_text(encoding="utf-8")
    assert "internal implementation" not in template
    template.format(**{name: "" for name in TEMPLATE_PLACEHOLDERS})


def test_openevolve_builds_the_user_prompt_from_the_run_files(study, tmp_path):
    pytest.importorskip("openevolve")
    from openevolve.config import Config
    from openevolve.prompt.sampler import PromptSampler

    llm = {"models": [{"name": "model", "api_base": "http://localhost:1"}], "temperature": 0.7}
    paths = write_run_files(
        study=study,
        run_dir=tmp_path / "run",
        goal=GOAL,
        llm=llm,
        iterations=7,
        stage_timeout_seconds=120,
    )
    config = Config.from_yaml(paths["config"])
    assert config.prompt.suggest_simplification_after_chars is None
    assert Path(config.prompt.template_dir) == paths["user_template"].parent
    program = program_text(study, Candidate("prompt", {"prompt.text": "Write formally. " * 40}))
    assert len(program) > 500
    prompt = PromptSampler(config.prompt).build_prompt(
        current_program=program, program_metrics={"combined_score": 0.1}, language="text",
        diff_based_evolution=False, feature_dimensions=["complexity", "diversity"],
    )
    assert "Propose a new program with a higher FITNESS SCORE." in prompt["user"]
    assert "A program here is a steering configuration, not code." in prompt["user"]
    assert "Consider simplifying" not in prompt["user"]


def test_a_system_message_describes_its_own_structure_only(study):
    message = system_message(study=study, structure="prompt", goal=GOAL)
    assert GOAL in message and "`prompt.text`" in message and "System-prompt text set or merged" in message
    assert "exactly one JSON object" in message and '(always "prompt")' in message
    example = program_text(study=study, candidate=study.default_candidate(structure="prompt")).rstrip("\n")
    assert example in message
    assert "`caa" not in message
    assert "`prompt" not in system_message(study=study, structure="caa", goal=GOAL)
    for benchmark in study.benchmarks:
        assert not any(item_id in message for item_id in benchmark.item_ids())
    with pytest.raises(ValueError, match="goal"):
        system_message(study=study, structure="prompt", goal="")


@dataclass
class _Result:
    metrics: dict
    artifacts: dict = field(default_factory=dict)


def test_the_generated_evaluator_returns_the_service_reply(study, tmp_path, monkeypatch):
    package = types.ModuleType("openevolve")
    module = types.ModuleType("openevolve.evaluation_result")
    module.EvaluationResult = _Result
    monkeypatch.setitem(sys.modules, "openevolve", package)
    monkeypatch.setitem(sys.modules, "openevolve.evaluation_result", module)
    reply = {"metrics": {"combined_score": 0.25, "accepted": 1.0}, "artifacts": {"status": "accepted"}}
    authkey = b"k" * 32
    listener = Listener(authkey=authkey)
    requests = []

    def serve():
        for _ in range(3):
            with listener.accept() as connection:
                requests.append(json.loads(connection.recv_bytes().decode("utf-8")))
                connection.send_bytes(json.dumps(reply).encode("utf-8"))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    paths = write_run_files(
        study=study,
        run_dir=tmp_path / "run",
        goal=GOAL,
        llm={},
        iterations=1,
        stage_timeout_seconds=60,
    )
    monkeypatch.setenv("STEERABILITY_EVAL_ADDRESS", listener.address)
    monkeypatch.setenv("STEERABILITY_EVAL_KEY", authkey.hex())
    spec = importlib.util.spec_from_file_location("generated_evaluator", paths["evaluator"])
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    program_path = str(paths["initial_program"])
    results = [evaluator.evaluate_stage1(program_path), evaluator.evaluate_stage2(program_path),
               evaluator.evaluate(program_path)]
    thread.join(timeout=10)
    listener.close()
    assert all(result.metrics == reply["metrics"] and result.artifacts == reply["artifacts"] for result in results)
    assert [request["op"] for request in requests] == ["stage1", "stage2", "full"]
    assert requests[0]["program"] == paths["initial_program"].read_text(encoding="utf-8")


def test_admit_and_replace_follow_the_archive_not_the_metrics():
    seed = SimpleNamespace(parent_id=None, code="seed", metrics={})
    child = SimpleNamespace(parent_id="parent", code="child", metrics={"combined_score": 1.0})
    unknown = {"known": False, "accepted": False, "fitness": 0.0}
    assert admit_program(seed, lambda code: unknown)
    assert not admit_program(child, lambda code: unknown)
    assert not admit_program(child, lambda code: {"known": True, "accepted": True, "fitness": 0.0})
    assert admit_program(child, lambda code: {"known": True, "accepted": True, "fitness": 0.1})

    def failing(code):
        raise ConnectionError("down")

    assert not admit_program(child, failing)
    incumbent = SimpleNamespace(parent_id="parent", code="incumbent", metrics={"combined_score": 9.0})

    def estimates(incumbent_estimate):
        return lambda code: {"child": {"accepted": True, "fitness": 0.2}, "incumbent": incumbent_estimate}[code]

    assert replace_program(child, incumbent, estimates({"accepted": False, "fitness": 0.0}))
    assert not replace_program(child, incumbent, estimates({"accepted": True, "fitness": 0.3}))
    assert replace_program(child, incumbent, estimates({"accepted": True, "fitness": 0.1}))


def test_latest_checkpoint(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    for name in ("checkpoint_3", "checkpoint_12", "notes"):
        (tmp_path / "checkpoints" / name).mkdir(parents=True)
    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoints" / "checkpoint_12")


def test_openevolve_is_an_optional_dependency():
    if importlib.util.find_spec("openevolve") is not None:
        pytest.skip("openevolve is installed")
    with pytest.raises(ModuleNotFoundError, match="optimize"):
        require_openevolve()
