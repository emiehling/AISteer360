"""Tests for `SteeringEval` and its frames (`evaluation/runner.py`) on the tiny CI model."""
import json
import textwrap

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.identity import versioned
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import BASELINE, Choice, Recipe, Template, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.evaluation import batching
from steerability.evaluation.items import InspectBenchmark, PromptBenchmark
from steerability.evaluation.runner import SteeringEval, runs_frame, samples_frame, summarize_runs

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPTS = {"a": "the cat", "b": "the cat sat on", "c": "the dog sat", "d": "a cat"}


@versioned("1")
def length_score(response, row):
    return float(len(response))


def _protocol(model: str = MODEL) -> Protocol:
    return Protocol.create(model, generate_defaults={"temperature": 0, "max_tokens": 3})


def _benchmark() -> PromptBenchmark:
    return PromptBenchmark("lengths", PROMPTS, length_score)


def _pipelines() -> dict:
    return {
        "baseline": BASELINE,
        "prompted": Template((entry(SystemPrompt, text=Choice(("be brief", "be kind"))),)),
    }


@pytest.fixture()
def generations(monkeypatch):
    """Count the prompts that direct generation sends to a pipeline."""
    prompts = []
    original = batching.RequestAdmission.run_batch

    def counting(self, batch):
        for member in batch:
            prompt = member.prompt
            prompts.append(prompt if isinstance(prompt, str) else prompt[-1]["content"])
        return original(self, batch)

    monkeypatch.setattr(batching.RequestAdmission, "run_batch", counting)
    return prompts


class TestRun:
    def test_template_sweep_frames(self, tmp_path):
        evaluation = SteeringEval(_protocol(), _pipelines(), [_benchmark()], num_trials=2, device_map=None,
                                  progress=False)
        evaluation.run(tmp_path / "run")
        samples = evaluation.samples_frame(include_text=True)
        assert len(samples) == 3 * 2 * len(PROMPTS)
        assert {"pipeline", "recipe_id", "trial_id", "benchmark", "item_id", "status", "tokens", "value",
                "input", "target", "completion", "system_prompt.text"} <= set(samples.columns)
        assert set(samples["system_prompt.text"].dropna()) == {"be brief", "be kind"}
        runs = evaluation.runs_frame({"length": "length_score"})
        assert list(runs.columns[:6]) == ["pipeline", "recipe_id", "trial_id", "n", "errors", "length"]
        assert len(runs) == 3 * 2
        assert (runs["n"] == len(PROMPTS)).all() and (runs["errors"] == 0).all()
        summary = summarize_runs(runs, ["length"])
        assert len(summary) == 3 and (summary["n_trials"] == 2).all()
        assert runs_frame(tmp_path / "run", {"length": "length_score"}).equals(runs)

        points = json.loads((tmp_path / "run" / "points.json").read_text())
        assert [record["pipeline"] for record in points["points"]] == ["baseline", "prompted", "prompted"]
        protocol = json.loads((tmp_path / "run" / "protocol.json").read_text())
        assert protocol["id"] == _protocol().id and protocol["observed"]["model_fingerprint"]

    def test_a_run_builds_one_factory_and_prepares_once(self, tmp_path, monkeypatch):
        calls = []
        original = PipelineFactory.prepare

        def counting(self, recipes):
            calls.append(list(recipes))
            return original(self, recipes)

        monkeypatch.setattr(PipelineFactory, "prepare", counting)
        evaluation = SteeringEval(_protocol(), _pipelines(), [_benchmark()], device_map=None, progress=False)
        evaluation.run(tmp_path / "run")
        assert len(calls) == 1
        assert evaluation.factory.stats.pipelines == 3

    def test_resume_generates_nothing_and_a_larger_selection_only_the_new_items(self, tmp_path, generations):
        pipelines = {"baseline": BASELINE}
        SteeringEval(_protocol(), pipelines, [_benchmark()], items={"lengths": ["a", "b"]}, device_map=None,
                     progress=False).run(tmp_path / "run")
        assert sorted(generations) == ["the cat", "the cat sat on"]
        generations.clear()
        SteeringEval(_protocol(), pipelines, [_benchmark()], items={"lengths": ["a", "b"]}, device_map=None,
                     progress=False).run(tmp_path / "run")
        assert generations == []
        larger = SteeringEval(_protocol(), pipelines, [_benchmark()], items={"lengths": ["a", "b", "c"]},
                              device_map=None, progress=False)
        larger.run(tmp_path / "run")
        assert generations == ["the dog sat"]
        assert len(larger.samples_frame()) == 3

    def test_a_save_dir_of_another_base_model_is_refused(self, tmp_path):
        SteeringEval(_protocol(), {"baseline": BASELINE}, [_benchmark()], items={"lengths": ["a"]}, device_map=None,
                     progress=False).run(tmp_path / "run")
        other = SteeringEval(_protocol("other/model"), {"baseline": BASELINE}, [_benchmark()], device_map=None,
                             progress=False)
        with pytest.raises(ValueError, match="another protocol|written under protocol"):
            other.run(tmp_path / "run")

    def test_a_shared_store(self, tmp_path, generations):
        store = Store.open(tmp_path / "shared")
        for save_dir in ("one", "two"):
            SteeringEval(_protocol(), {"baseline": BASELINE}, [_benchmark()], items={"lengths": ["a"]},
                         device_map=None, progress=False).run(tmp_path / save_dir, store=store)
        assert generations == ["the cat"]
        assert not (tmp_path / "one" / "store").exists()

    def test_frames_require_a_run(self):
        with pytest.raises(RuntimeError, match="run"):
            SteeringEval(_protocol(), {"baseline": BASELINE}, [_benchmark()]).runs_frame()

    def test_unsupported_points_raise_before_any_model_work(self, tmp_path, monkeypatch):
        spec = BackendSpec(kind="vllm-serve", model=MODEL, options={"base_url": "http://x:8000", "api_key": "secret"})
        protocol = Protocol.create(MODEL, backend=spec)
        recipe = Recipe((entry(SystemPrompt, text="x"),))
        evaluation = SteeringEval(protocol, {"p": recipe}, [_benchmark()], progress=False)
        monkeypatch.setattr(PipelineFactory, "prepare", lambda self, recipes: pytest.fail("model work ran"))
        monkeypatch.setattr(
            "steerability.evaluation.runner.preflight", lambda points, protocol: ["p [x] C (generate): no"]
        )
        with pytest.raises(RuntimeError, match="Unsupported"):
            evaluation.run(tmp_path / "run")
        written = (tmp_path / "run" / "protocol.json").read_text()
        assert json.loads(written)["id"] == protocol.id
        assert "secret" not in written
        assert "secret" not in json.dumps(protocol.identity())


DICT_TASK = '''
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import generate


@scorer(metrics={"*": [mean()]})
def shape():
    async def score(state, target):
        text = state.output.completion
        return Score(value={"length": float(len(text)), "has_target": float(target.text in text)})
    return score


@task
def tiny():
    samples = [Sample(id=i, input=f"the cat {i}", target="the", metadata={"question_id": i // 2}) for i in range(4)]
    return Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=shape())
'''


def test_samples_frame_reads_dict_scores_metadata_and_text(tmp_path):
    path = tmp_path / "dict_task.py"
    path.write_text(textwrap.dedent(DICT_TASK))
    benchmark = InspectBenchmark(str(path), scorer="shape/length", group_key="question_id",
                                 generate={"max_tokens": 3, "temperature": 0})
    evaluation = SteeringEval(_protocol(), {"baseline": BASELINE}, [benchmark], device_map=None, progress=False)
    evaluation.run(tmp_path / "run")
    samples = samples_frame(tmp_path / "run", {"length": "shape/length", "hit": "shape/has_target"},
                            metadata_keys=["question_id"], include_text=True)
    assert list(samples["item_id"]) == ["0", "1", "2", "3"]
    assert list(samples["question_id"]) == [0, 0, 1, 1]
    assert samples["length"].notna().all() and set(samples["hit"]) <= {0.0, 1.0}
    assert samples["completion"].notna().all() and samples["input"].notna().all()
    runs = runs_frame(tmp_path / "run", {"length": "shape/length"})
    summary = summarize_runs(runs, ["length"])

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from steerability.evaluation import plotting

    plotting.plot_comparison_bars(summary, metric_cols=["length_mean"], group_col="pipeline")
