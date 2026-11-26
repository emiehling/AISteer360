"""Tests for the OpenEvolve engine, without OpenEvolve (`search/openevolve.py`, `search/_openevolve_controller.py`)."""
import importlib.util
import json
import logging
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

from steerability.algorithms.core.recipe import Choice, Template, Text, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.search import openevolve as openevolve_engine
from steerability.optimization.search._openevolve_controller import (
    Admitted,
    admit_program,
    latest_checkpoint,
    replace_program,
)
from steerability.optimization.search.openevolve import (
    BLOCK_END,
    BLOCK_START,
    USER_TEMPLATE,
    OpenEvolveEngine,
    parse,
    program_text,
    require_openevolve,
    system_message,
    write_run_files,
)
from steerability.optimization.study import Candidate
from tests.optimization.helpers import (
    StubFactory,
    Truth,
    argument,
    build_study,
    caa_template,
    multiplier,
    prompt_template,
)

GOAL = "The behavior score measures how formal the reply reads."
LLM = {"models": [{"name": "model", "api_base": "http://localhost:1"}], "temperature": 0.7}
TEMPLATE_PLACEHOLDERS = ("metrics", "fitness_score", "feature_coords", "feature_dimensions", "improvement_areas",
                         "evolution_history", "current_program", "language", "artifacts")


def _quiet(recipe, item_id):
    return Truth()


def _truth(recipe, item_id):
    value = multiplier(recipe)
    if value is not None:
        return Truth(loss=0.0055 * value * value, gain=0.05 * value)
    return Truth(gain=0.2 if argument(recipe, "input_control/system_prompt", "text") == "be kind" else 0.0)


def _space():
    return {"prompt": prompt_template(), "caa": caa_template(layer_id=Choice(values=(1, 2)))}


@pytest.fixture()
def study(tmp_path):
    return build_study(store=Store.open(root=tmp_path / "store"), truth=_quiet, space=_space())


def _prompt(text):
    return Candidate(structure="prompt", values={"prompt.text": text})


def _layer(layer_id):
    return Candidate(structure="caa", values={"caa.layer_id": layer_id})


def _block(body: str) -> str:
    return f"# header\n{BLOCK_START}\n{body}\n{BLOCK_END}\n"


def _write(study, run_dir, islands, **options):
    settings = {"goal": GOAL, "llm": LLM, "iterations": 7, "stage_timeout_seconds": 120, "level_bins": 6,
                "elite_size": 15, "checkpoint_interval": 10, **options}
    return write_run_files(study=study, run_dir=run_dir, islands=islands, **settings)


def test_parse_reads_the_program_of_a_candidate(study):
    candidate = _prompt("be kind")
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


def test_the_run_files_follow_the_islands(study, tmp_path):
    islands = [_prompt("be brief"), _prompt("be kind"), _layer(2)]
    paths = _write(study, tmp_path / "run", islands)
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    assert {key: config[key] for key in ("max_iterations", "checkpoint_interval", "random_seed", "language",
                                         "enforce_evolve_blocks", "diff_based_evolution", "llm")} == {
        "max_iterations": 7, "checkpoint_interval": 10, "random_seed": study.seed, "language": "text",
        "enforce_evolve_blocks": True, "diff_based_evolution": False, "llm": LLM,
    }
    messages = [system_message(study=study, structure=candidate.structure, goal=GOAL) for candidate in islands]
    assert config["prompt"]["island_system_messages"] == messages
    assert config["prompt"]["system_message"] == messages[0]
    assert all(GOAL in message for message in messages)
    database = config["database"]
    initial = database.pop("island_initial_programs")
    assert database == {"num_islands": 3, "feature_dimensions": ["level"], "feature_bins": {"level": 6},
                        "archive_size": 15, "max_snapshot_artifacts": None}
    assert initial[0] is None and all(Path(path).is_absolute() for path in initial[1:])
    programs = [paths["initial_program"], *(Path(path) for path in initial[1:])]
    assert [parse(program=path.read_text(encoding="utf-8")) for path in programs] == islands
    assert json.loads(paths["islands"].read_text(encoding="utf-8")) == [
        {"island": 0, "structure": "prompt"}, {"island": 1, "structure": "prompt"}, {"island": 2, "structure": "caa"},
    ]
    assert config["evaluator"] == {"timeout": 120, "cascade_evaluation": True, "cascade_thresholds": [0.5],
                                   "parallel_evaluations": 1}
    assert config["prompt"]["suggest_simplification_after_chars"] is None
    template_dir = Path(config["prompt"]["template_dir"])
    assert template_dir.is_absolute() and paths["user_template"] == template_dir / "full_rewrite_user.txt"
    assert paths["user_template"].read_text(encoding="utf-8") == USER_TEMPLATE


def test_the_user_template_states_the_levels_and_leaves_out_the_feature_grid():
    task = " ".join(USER_TEMPLATE.split())
    assert "Each accepted candidate has a level, the share of the capability margin it uses (from 0 to 1)." in task
    assert "The search keeps the best candidate at each level." in task
    assert "worth proposing even when its fitness is lower" in task
    for name in ("fitness_score", "artifacts", "evolution_history", "current_program", "language"):
        assert "{" + name + "}" in USER_TEMPLATE
    for name in ("feature_coords", "improvement_areas", "feature_dimensions", "metrics"):
        assert "{" + name + "}" not in USER_TEMPLATE
    USER_TEMPLATE.format(**{name: "" for name in TEMPLATE_PLACEHOLDERS})


def test_openevolve_builds_the_island_prompts_from_the_run_files(study, tmp_path):
    pytest.importorskip("openevolve")
    from openevolve.config import Config
    from openevolve.prompt.sampler import PromptSampler

    paths = _write(study, tmp_path / "run", [_prompt("be brief"), _layer(1)])
    config = Config.from_yaml(paths["config"])
    assert config.prompt.suggest_simplification_after_chars is None
    assert Path(config.prompt.template_dir) == paths["user_template"].parent
    program = program_text(study=study, candidate=_layer(2))
    prompt = PromptSampler(config.prompt).build_prompt(
        current_program=program,
        program_metrics={"combined_score": 0.1, "level": 0.4},
        language="text",
        diff_based_evolution=False,
        feature_dimensions=["level"],
        island_id=1,
    )
    assert prompt["system"] == system_message(study=study, structure="caa", goal=GOAL)
    assert "Propose a new program with a higher FITNESS SCORE." in prompt["user"]
    assert "A program here is a steering configuration, not code." in prompt["user"]
    assert "Each accepted candidate has a level" in prompt["user"]
    assert "Consider simplifying" not in prompt["user"] and program in prompt["user"]


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


@pytest.fixture()
def offline(monkeypatch):
    """The engine with OpenEvolve required and launched as no-ops; each call of the search records the session."""
    searches = []
    monkeypatch.setattr(openevolve_engine, "require_openevolve", lambda: None)
    monkeypatch.setattr(OpenEvolveEngine, "_search", lambda self, *, session: searches.append(session))
    return searches


def _session(store, space, truth=_truth):
    study = build_study(store=store, truth=truth, space=space)
    session = StudySession(study=study, store=store, factory=StubFactory(protocol=study.protocol))
    session.start()
    return session


def _engine(run_dir, **options):
    return OpenEvolveEngine(run_dir=run_dir, llm=LLM, iterations=options.pop("iterations", 8), goal=GOAL, **options)


def _islands(run_dir):
    return [record["structure"] for record in json.loads((run_dir / "islands.json").read_text(encoding="utf-8"))]


def _program(run_dir, index):
    name = "initial_program.txt" if index == 0 else f"island_{index}_program.txt"
    return parse(program=(run_dir / name).read_text(encoding="utf-8"))


class TestEngine:
    def test_islands_begin_from_the_best_accepted_candidate_of_their_structure(self, tmp_path, offline):
        session = _session(Store.open(root=tmp_path / "store"), _space())
        assert session.evaluate(candidate=_prompt("be kind")).status == "accepted"
        run_dir = tmp_path / "run"
        _engine(run_dir).run(session=session)
        assert offline == [session]
        assert _islands(run_dir) == ["prompt", "caa"]
        assert _program(run_dir, 0) == _prompt("be kind")
        assert _program(run_dir, 1) == session.study.default_candidate(structure="caa")
        assert session.archive.latest(candidate_id=session.study.candidate_id(candidate=_layer(1))) is not None

    def test_several_islands_of_a_structure_begin_from_distinct_candidates(self, tmp_path, offline):
        session = _session(Store.open(root=tmp_path / "store"), {"prompt": prompt_template()})
        run_dir = tmp_path / "run"
        initial = [_prompt("be formal"), _prompt("be brief")]
        _engine(run_dir, islands_per_structure=2).run(session=session, initial_candidates=initial)
        assert _islands(run_dir) == ["prompt", "prompt"]
        assert [_program(run_dir, index) for index in (0, 1)] == initial

    def test_a_structure_without_proposer_axes_gets_no_island_and_is_evaluated_once(self, tmp_path, offline):
        space = {**_space(), "fixed": caa_template()}
        session = _session(Store.open(root=tmp_path / "store"), space)
        run_dir = tmp_path / "run"
        _engine(run_dir).run(session=session)
        assert "fixed" not in _islands(run_dir)
        default = session.study.default_candidate(structure="fixed")
        rows = session.store.execute(
            sql="SELECT proposer FROM candidates WHERE candidate_id = ?",
            parameters=(session.study.candidate_id(candidate=default),),
        ).fetchall()
        assert rows == [("openevolve",)]

    def test_a_scope_of_structures_without_proposer_axes_does_not_launch_openevolve(self, tmp_path, offline):
        session = _session(Store.open(root=tmp_path / "store"), {**_space(), "fixed": caa_template()})
        _engine(tmp_path / "run", structures=["fixed"]).run(session=session)
        assert offline == [] and not (tmp_path / "run").exists()

    def test_a_resumed_run_keeps_the_island_assignment(self, tmp_path, offline, caplog):
        session = _session(Store.open(root=tmp_path / "store"), _space())
        run_dir = tmp_path / "run"
        (run_dir / "openevolve" / "checkpoints" / "checkpoint_3").mkdir(parents=True)
        records = [{"island": 0, "structure": "caa"}, {"island": 1, "structure": "prompt"}]
        (run_dir / "islands.json").write_text(json.dumps(records), encoding="utf-8")
        with caplog.at_level(logging.INFO, logger="steerability.optimization.search.openevolve"):
            _engine(run_dir, iterations=12).run(session=session, initial_candidates=[_prompt("be kind")])
        assert "is not used" in caplog.text
        assert _islands(run_dir) == ["caa", "prompt"]
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        assert config["max_iterations"] == 12 and config["database"]["num_islands"] == 2
        expected = system_message(study=session.study, structure="caa", goal=GOAL)
        assert config["prompt"]["island_system_messages"][0] == expected
        assert session.archive.latest(candidate_id=session.study.candidate_id(candidate=_prompt("be kind"))) is None

    def test_a_checkpoint_without_islands_raises(self, tmp_path, offline):
        session = _session(Store.open(root=tmp_path / "store"), _space())
        run_dir = tmp_path / "run"
        (run_dir / "openevolve" / "checkpoints" / "checkpoint_3").mkdir(parents=True)
        with pytest.raises(ValueError, match="islands.json"):
            _engine(run_dir).run(session=session)
        assert offline == []

    def test_small_discrete_structures_get_a_warning_that_suggests_a_grid(self, tmp_path, offline, caplog):
        text = Template(entries=(entry(SystemPrompt, name="free", text=Text(max_chars=200)),))
        session = _session(Store.open(root=tmp_path / "store"), {**_space(), "free": text}, truth=_quiet)
        with caplog.at_level(logging.WARNING, logger="steerability.optimization.search.openevolve"):
            _engine(tmp_path / "run", iterations=30).run(session=session)
        warnings = [record.getMessage() for record in caplog.records
                    if record.name == openevolve_engine.__name__ and record.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert "'caa'" in warnings[1] and "GridProposer(study=study, structures=['caa'])" in warnings[1]
        assert not any("'free'" in warning for warning in warnings)

    def test_the_engine_refuses_an_empty_goal_and_invalid_counts(self, tmp_path):
        with pytest.raises(ValueError, match="goal"):
            OpenEvolveEngine(run_dir=tmp_path, llm=LLM, iterations=8, goal=" ")
        with pytest.raises(ValueError, match="iterations"):
            OpenEvolveEngine(run_dir=tmp_path, llm=LLM, iterations=0, goal=GOAL)
        with pytest.raises(ValueError, match="rejection_digest"):
            OpenEvolveEngine(run_dir=tmp_path, llm=LLM, iterations=8, goal=GOAL, rejection_digest=-1)


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
    paths = _write(study, tmp_path / "run", [_prompt("be brief")], llm={}, iterations=1)
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


ESTIMATES = {
    "kind": {"known": True, "accepted": True, "fitness": 0.2, "candidate_id": "kind", "structure": "prompt"},
    "brief": {"known": True, "accepted": True, "fitness": 0.1, "candidate_id": "brief", "structure": "prompt"},
    "zero": {"known": True, "accepted": True, "fitness": 0.0, "candidate_id": "zero", "structure": "prompt"},
    "rejected": {"known": True, "accepted": False, "fitness": 0.0, "candidate_id": "rejected", "structure": "prompt"},
    "layer": {"known": True, "accepted": True, "fitness": 0.3, "candidate_id": "layer", "structure": "caa"},
}
STRUCTURES = ["prompt", "caa"]


def _state(program_id, code, *, parent_id="parent", metrics=None):
    return SimpleNamespace(
        id=program_id,
        code=code,
        parent_id=parent_id,
        metrics={"combined_score": 0.2} if metrics is None else metrics,
    )


def _snapshot(*islands, programs=()):
    return SimpleNamespace(
        islands=tuple(frozenset(island) for island in islands),
        programs={program.id: program for program in programs},
    )


class TestAdmission:
    @pytest.fixture()
    def estimates(self):
        calls = []

        def estimate(code):
            calls.append(code)
            return ESTIMATES[code]

        estimate.calls = calls
        return estimate

    def _admit(self, estimate, admitted, candidate, island, snapshot=None):
        return admit_program(
            snapshot=snapshot or _snapshot(set(), set()),
            candidate=candidate,
            island=island,
            structures=STRUCTURES,
            admitted=admitted,
            estimate=estimate,
        )

    def test_a_program_without_a_parent_is_admitted_and_recorded(self, estimates):
        admitted = Admitted()
        assert self._admit(estimates, admitted, _state("p0", "rejected", parent_id=None), 0)
        assert admitted.candidates[0] == {"rejected"} and admitted.programs == {"p0"}

    def test_each_rule_rejects_in_order(self, estimates):
        admitted = Admitted()
        timed_out = _state("p1", "kind", metrics={"stage1_passed": 1.0, "stage2_passed": 0.0, "timeout": True})
        assert not self._admit(estimates, admitted, timed_out, 0)
        assert estimates.calls == []
        assert not self._admit(estimates, admitted, _state("p2", "rejected"), 0)
        assert not self._admit(estimates, admitted, _state("p3", "zero"), 0)
        assert not self._admit(estimates, admitted, _state("p4", "layer"), 0)
        assert self._admit(estimates, admitted, _state("p5", "kind"), 0)
        assert not self._admit(estimates, admitted, _state("p6", "kind"), 0)
        assert self._admit(estimates, admitted, _state("p7", "layer"), 1)
        assert admitted.candidates == {0: {"kind"}, 1: {"layer"}} and admitted.programs == {"p5", "p7"}

    def test_after_a_resume_the_island_set_is_rebuilt_once_from_the_snapshot(self, estimates):
        admitted = Admitted()
        snapshot = _snapshot({"old"}, set(), programs=[_state("old", "kind")])
        assert not self._admit(estimates, admitted, _state("new", "kind"), 0, snapshot)
        assert estimates.calls == ["kind", "kind"]
        assert self._admit(estimates, admitted, _state("other", "brief"), 0, snapshot)
        assert estimates.calls == ["kind", "kind", "brief"]

    def test_a_failing_service_admits_nothing_but_programs_without_a_parent(self):
        def failing(code):
            raise ConnectionError("down")

        admitted = Admitted()
        assert not self._admit(failing, admitted, _state("p1", "kind"), 0)
        assert self._admit(failing, admitted, _state("p0", "kind", parent_id=None), 0)


def test_replace_follows_the_archive_not_the_metrics():
    child = SimpleNamespace(parent_id="parent", code="child", metrics={"combined_score": 1.0})
    incumbent = SimpleNamespace(parent_id="parent", code="incumbent", metrics={"combined_score": 9.0})

    def estimates(incumbent_estimate):
        return lambda code: {"child": {"accepted": True, "fitness": 0.2}, "incumbent": incumbent_estimate}[code]

    unaccepted, higher, lower = ({"accepted": accepted, "fitness": fitness}
                                 for accepted, fitness in ((False, 0.0), (True, 0.3), (True, 0.1)))
    assert replace_program(candidate=child, incumbent=incumbent, estimate=estimates(unaccepted))
    assert not replace_program(candidate=child, incumbent=incumbent, estimate=estimates(higher))
    assert replace_program(candidate=child, incumbent=incumbent, estimate=estimates(lower))


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
