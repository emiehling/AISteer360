"""Tests for direct generation of Inspect tasks: capture, the parity probe, and the direct path."""
import textwrap

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import BASELINE
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import Steered
from steerability.algorithms.state_control.pasta.control import PASTA
from steerability.evaluation.items import InspectBenchmark, evaluate_items
from steerability.evaluation.provider import ProviderOptions
from tests.evaluation.conftest import CHAT_TEMPLATE
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

ONE_CALL = '''
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import exact, includes
from inspect_ai.solver import generate


LONG = " ".join(["the cat sat on the mat"] * 8)


@task
def tiny():
    samples = [Sample(id=f"s{i}", input=f"{LONG} {i}", target="the") for i in range(4)]
    return Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=[includes(), exact()])
'''

REWRITING = '''
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import solver


@solver
def rewrite():
    async def solve(state, generate):
        state = await generate(state)
        state.output.completion = "rewritten " + state.output.completion
        return state
    return solve


@task
def tiny():
    samples = [Sample(id=f"s{i}", input=f"the cat {i}", target="the") for i in range(4)]
    return Task(dataset=MemoryDataset(samples), solver=[rewrite()], scorer=includes())
'''

TWO_CALLS = '''
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def tiny():
    samples = [Sample(id=f"s{i}", input=f"the cat {i}", target="the") for i in range(4)]
    return Task(dataset=MemoryDataset(samples), solver=[generate(), generate()], scorer=includes())
'''

RUNTIME_KWARGS = '''
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes

from steerability.evaluation.solvers import runtime_kwargs_solver

WORDS = ["cat", "dog", "mat", "sat", "ran", "fast"]


@task
def tiny():
    samples = [
        Sample(id=word, input=f"the {word} on", target="the", metadata={"runtime_kwargs": {"substrings": [word]}})
        for word in WORDS
    ]
    return Task(dataset=MemoryDataset(samples), solver=[runtime_kwargs_solver()], scorer=includes())
'''


class _RecordingPasta(PASTA):
    """PASTA recording the `substrings` values `get_hooks` receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received_substrings: list = []

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        self.received_substrings.append(runtime_kwargs.get("substrings"))
        return super().get_hooks(input_ids, runtime_kwargs, **kwargs)


def _protocol() -> Protocol:
    return Protocol.create(
        "tiny", generate_defaults={"temperature": 0, "max_tokens": 3}, provider=ProviderOptions(max_batch_size=4),
    )


def _steered(controls=(), chat: bool = False) -> Steered:
    tokenizer = wordlevel_tokenizer()
    if chat:
        tokenizer.chat_template = CHAT_TEMPLATE
    pipeline = SteeringPipeline(controls=list(controls), model=tiny_llama(), tokenizer=tokenizer)
    pipeline.steer()
    return Steered(pipeline, BASELINE, "resolved", _protocol())


def _prepared(tmp_path, source: str) -> tuple[InspectBenchmark, Store]:
    path = tmp_path / "task.py"
    path.write_text(textwrap.dedent(source))
    store = Store.open(tmp_path / "store")
    benchmark = InspectBenchmark(str(path), scorer="includes")
    benchmark.prepare(store)
    return benchmark, store


@pytest.fixture()
def inspect_runs(monkeypatch):
    """Count the `inspect_ai.eval` calls made after the fixture is requested."""
    import inspect_ai

    calls = []
    original = inspect_ai.eval

    def counting(*args, **kwargs):
        calls.append(kwargs.get("model"))
        return original(*args, **kwargs)

    monkeypatch.setattr(inspect_ai, "eval", counting)
    return calls


def test_the_probe_records_direct_and_later_rows_have_rollout_keys(tmp_path, inspect_runs):
    steered = _steered()
    benchmark, store = _prepared(tmp_path, ONE_CALL)
    assert benchmark.capturable
    request = store.get_items(benchmark.id)[0].payload["request"]
    assert request["messages"][-1]["content"].startswith("the cat sat on the mat the cat")
    assert benchmark.mode(steered.protocol) is None
    first = evaluate_items(steered, benchmark, ["s0", "s1"], store=store)
    assert benchmark.mode(steered.protocol) == "direct", benchmark.mode_detail(steered.protocol)
    assert all(row.status == "ok" and row.rollout_key is not None for row in first.values())
    assert all(set(row.scores) == {"includes", "exact"} for row in first.values())

    inspect_runs.clear()
    later = evaluate_items(steered, benchmark, ["s2", "s3"], store=store)
    assert inspect_runs == []
    assert all(row.status == "ok" and row.rollout_key is not None and row.generated for row in later.values())


def test_a_sampling_protocol_without_concurrent_generation_runs_through_inspect(tmp_path):
    steered = _steered()
    sampling = Protocol.create("tiny", generate_defaults={"temperature": 0.8, "max_tokens": 3}, seed=3)
    steered = Steered(steered.pipeline, BASELINE, "resolved", sampling)
    assert not steered.pipeline.supports_concurrent_generate
    benchmark, store = _prepared(tmp_path, ONE_CALL)
    rows = evaluate_items(steered, benchmark, ["s0", "s1"], store=store)
    assert all(row.status == "ok" and row.rollout_key is None for row in rows.values())
    assert benchmark.mode(sampling) is None


def test_a_solver_that_rewrites_the_output_records_inspect(tmp_path):
    steered = _steered()
    benchmark, store = _prepared(tmp_path, REWRITING)
    assert benchmark.capturable
    rows = evaluate_items(steered, benchmark, ["s0", "s1"], store=store)
    assert benchmark.mode(steered.protocol) == "inspect"
    assert "completions differ" in benchmark.mode_detail(steered.protocol)
    assert all(row.rollout_key is None and row.completion.startswith("rewritten") for row in rows.values())


def test_a_task_with_two_model_calls_is_not_capturable(tmp_path):
    steered = _steered()
    benchmark, store = _prepared(tmp_path, TWO_CALLS)
    assert not benchmark.capturable
    rows = evaluate_items(steered, benchmark, ["s0"], store=store)
    assert rows["s0"].status == "ok" and rows["s0"].rollout_key is None
    assert benchmark.mode(steered.protocol) is None


def test_runtime_kwargs_reach_the_pipeline_on_the_direct_path(tmp_path, inspect_runs):
    pasta = _RecordingPasta(head_config=[0], alpha=1.5, scale_position="include")
    steered = _steered([pasta], chat=True)
    benchmark, store = _prepared(tmp_path, RUNTIME_KWARGS)
    evaluate_items(steered, benchmark, ["cat", "dog"], store=store)
    assert benchmark.mode(steered.protocol) == "direct", benchmark.mode_detail(steered.protocol)

    pasta.received_substrings.clear()
    inspect_runs.clear()
    rows = evaluate_items(steered, benchmark, ["mat", "sat", "ran", "fast"], store=store)
    assert inspect_runs == []
    assert all(row.rollout_key is not None for row in rows.values())
    received = [group for call in pasta.received_substrings for group in call]
    assert sorted(received) == [["fast"], ["mat"], ["ran"], ["sat"]]


def test_a_scored_completion_is_read_back_from_the_store(tmp_path):
    benchmark, store = _prepared(tmp_path, ONE_CALL)
    value, scores = benchmark.score("s0", "the end")
    assert value == 1.0 and set(scores) == {"includes", "exact"}
    assert benchmark.score("s0", "the end") == (value, scores)
