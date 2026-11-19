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


def _quiet(recipe, item_id):
    return Truth()


@pytest.fixture()
def study(tmp_path):
    store = Store.open(tmp_path / "store")
    return build_study(store, _quiet, space={"prompt": prompt_template(), "caa": caa_template()})


def _block(body: str) -> str:
    return f"# header\n{BLOCK_START}\n{body}\n{BLOCK_END}\n"


def test_parse_reads_the_program_of_a_candidate(study):
    candidate = Candidate("prompt", {"prompt.text": "be kind"})
    program = program_text(study, candidate)
    assert program.splitlines()[0] == f"# steerability-candidate/1 study={study.id}"
    assert parse(program) == candidate
    assert parse(_block('  {"structure": "caa", "values": {}}  ')) == Candidate("caa", {})


@pytest.mark.parametrize("program, message", [
    ('{"structure": "caa", "values": {}}', "exactly one block"),
    (_block("{}") + _block("{}"), "exactly one block"),
    (f"{BLOCK_END}\n{BLOCK_START}\n", "exactly one block"),
    (_block("structure = caa"), "JSON object"),
    (_block('{"structure": "caa", "values": {}} {"structure": "caa", "values": {}}'), "more than one"),
    (_block("[1, 2]"), "not an object"),
    (_block('{"structure": "caa", "values": {}, "notes": "x"}'), "exactly the keys"),
    (_block('{"structure": 1, "values": {}}'), "must be a string"),
])
def test_parse_rejects_malformed_programs(program, message):
    with pytest.raises(ValueError, match=message):
        parse(program)


def test_the_run_files(study, tmp_path):
    llm = {"models": [{"name": "model", "api_base": "http://localhost:1"}], "temperature": 0.7}
    paths = write_run_files(study, tmp_path / "run", llm=llm, iterations=7, stage_timeout_seconds=120)
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    assert {key: config[key] for key in ("max_iterations", "random_seed", "language", "enforce_evolve_blocks",
                                         "diff_based_evolution", "llm")} == {
        "max_iterations": 7, "random_seed": study.seed, "language": "text", "enforce_evolve_blocks": True,
        "diff_based_evolution": False, "llm": llm,
    }
    assert config["prompt"]["system_message"] == system_message(study)
    assert config["prompt"]["island_system_messages"] == [system_message(study, "prompt"),
                                                          system_message(study, "caa")]
    database = config["database"]
    assert database["num_islands"] == 2 and database["island_initial_programs"][0] is None
    assert database["feature_dimensions"] == ["complexity", "diversity"] and database["feature_bins"] == 6
    assert config["evaluator"] == {"timeout": 120, "cascade_evaluation": True, "cascade_thresholds": [0.5],
                                   "parallel_evaluations": 1}
    assert parse(paths["initial_program"].read_text(encoding="utf-8")) == study.seed_candidate("prompt")
    seed = Path(database["island_initial_programs"][1])
    assert seed.is_absolute() and parse(seed.read_text(encoding="utf-8")) == study.seed_candidate("caa")


def test_system_messages_describe_the_axes_and_contain_no_item_text(study):
    island = system_message(study, "prompt")
    assert "`prompt.text`" in island and "one of" in island and "System-prompt text set or merged" in island
    assert "exactly one JSON object" in island
    assert program_text(study, study.seed_candidate("prompt")).rstrip("\n") in island
    assert "the evaluator solves `caa.multiplier`" in system_message(study)
    assert "has no settings to propose" in system_message(study, "caa")
    for benchmark in study.benchmarks:
        assert not any(item_id in island for item_id in benchmark.item_ids())


def test_the_help_text_falls_back_to_the_args_docstring(tmp_path):
    store = Store.open(tmp_path / "store")
    searched = build_study(store, _quiet, space={"caa": caa_template()}, searched={"caa.multiplier"})
    message = system_message(searched, "caa")
    assert "a number from 0 to 3" in message and "Scaling factor for the steering vector" in message


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
    paths = write_run_files(study, tmp_path / "run", llm={}, iterations=1, stage_timeout_seconds=60)
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
