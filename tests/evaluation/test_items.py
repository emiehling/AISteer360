"""Tests for item-level evaluation (`evaluation/items.py`) on tiny hub-free models."""
import textwrap

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.identity import versioned
from steerability.algorithms.core.protocol import Protocol, ProviderOptions
from steerability.algorithms.core.recipe import BASELINE
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import Steered
from steerability.evaluation import batching
from steerability.evaluation.items import (
    InspectBenchmark,
    ItemResult,
    PromptBenchmark,
    Request,
    evaluate_items,
    generate_requests,
)
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

PROMPTS = {"a": "the cat", "b": "the cat sat on", "c": "the dog sat", "d": "a cat"}


@versioned("1")
def length_score(response, row):
    return float(len(response))


@pytest.fixture()
def steered():
    pipeline = SteeringPipeline(controls=[], model=tiny_llama(), tokenizer=wordlevel_tokenizer())
    pipeline.steer()
    protocol = Protocol.create(
        "tiny", generate_defaults={"temperature": 0, "max_tokens": 3}, provider=ProviderOptions(max_batch_size=2),
    )
    return Steered(pipeline, BASELINE, "resolved", protocol)


@pytest.fixture()
def batches(monkeypatch):
    """Record the prompts of every batched pipeline call that direct generation issues."""
    calls = []
    original = batching.RequestAdmission.run_batch

    def recording(self, batch):
        calls.append([member.prompt for member in batch])
        return original(self, batch)

    monkeypatch.setattr(batching.RequestAdmission, "run_batch", recording)
    return calls


def _benchmark(**kwargs) -> PromptBenchmark:
    return PromptBenchmark("lengths", PROMPTS, length_score, targets={"a": "x"}, **kwargs)


class TestPromptBenchmark:
    def test_a_second_call_generates_nothing(self, tmp_path, steered, batches):
        store = Store.open(tmp_path)
        benchmark = _benchmark()
        benchmark.prepare(store)
        first = evaluate_items(steered, benchmark, list(PROMPTS), store=store)
        assert all(row.generated and row.status == "ok" for row in first.values())
        calls = len(batches)
        second = evaluate_items(steered, benchmark, list(PROMPTS), store=store)
        assert len(batches) == calls
        assert not any(row.generated for row in second.values())
        assert {key: row.value for key, row in second.items()} == {key: row.value for key, row in first.items()}
        assert all(row.rollout_key is not None and row.completion is not None for row in second.values())

    def test_a_superset_generates_only_the_new_items(self, tmp_path, steered, batches):
        store = Store.open(tmp_path)
        benchmark = _benchmark()
        benchmark.prepare(store)
        evaluate_items(steered, benchmark, ["a", "b"], store=store)
        batches.clear()
        rows = evaluate_items(steered, benchmark, ["a", "b", "c"], store=store)
        assert [prompt for call in batches for prompt in call] == [PROMPTS["c"]]
        assert [item for item, row in rows.items() if row.generated] == ["c"]

    def test_a_later_replicate_generates_again_in_another_batch_order(self, tmp_path, steered, batches):
        store = Store.open(tmp_path)
        benchmark = _benchmark()
        benchmark.prepare(store)
        evaluate_items(steered, benchmark, list(PROMPTS), store=store, replicate=0)
        first_order = [prompt for call in batches for prompt in call]
        batches.clear()
        rows = evaluate_items(steered, benchmark, list(PROMPTS), store=store, replicate=1)
        assert all(row.generated for row in rows.values())
        assert first_order == sorted(PROMPTS.values(), key=lambda prompt: (len(prompt), prompt))
        assert [prompt for call in batches for prompt in call] != first_order

    def test_a_new_scorer_version_writes_new_evidence_and_generates_nothing(
        self, tmp_path, steered, batches, monkeypatch
    ):
        store = Store.open(tmp_path)
        first = _benchmark()
        first.prepare(store)
        evaluate_items(steered, first, list(PROMPTS), store=store)
        calls = len(batches)
        monkeypatch.setattr(length_score, "__steerability_version__", "2")
        second = _benchmark()
        assert second.id != first.id
        second.prepare(store)
        rows = evaluate_items(steered, second, list(PROMPTS), store=store)
        assert len(batches) == calls
        assert len(store.get_evidence(steered.protocol.id, "resolved", second.id)) == len(PROMPTS)
        assert not any(row.generated for row in rows.values())

    def test_items_with_one_completion_and_target_but_different_inputs_are_scored_separately(self, tmp_path):
        seen = []

        @versioned("1")
        def counting(response, row):
            seen.append(row["input"])
            return 1.0

        counting.__qualname__ = "length_score"  # a module-level name, so the scorer has an identity
        store = Store.open(tmp_path)
        benchmark = PromptBenchmark("pair", {"x": "first prompt", "y": "second prompt"}, counting)
        benchmark.score("x", "same", store=store)
        benchmark.score("y", "same", store=store)
        benchmark.score("x", "same", store=store)
        assert seen == ["first prompt", "second prompt"]

    def test_an_unversioned_scorer_is_refused(self):
        def unversioned(response, row):
            return 0.0

        with pytest.raises(ValueError, match="version"):
            PromptBenchmark("x", {"a": "b"}, unversioned)

    def test_item_ids_require_prepare(self):
        with pytest.raises(RuntimeError, match="prepare"):
            _benchmark().item_ids()

    def test_groups(self, tmp_path):
        store = Store.open(tmp_path)
        benchmark = _benchmark(groups={"a": "g1", "b": "g1"})
        benchmark.prepare(store)
        assert benchmark.item_ids() == tuple(PROMPTS)
        assert benchmark.groups() == {"a": "g1", "b": "g1"}


class _OmittingBenchmark:
    """A benchmark whose `run` never completes the item `"missing"`."""

    name = "omitting"
    id = "omitting-id"

    def __init__(self):
        self.calls: list[list[str]] = []

    def prepare(self, store):
        pass

    def item_ids(self):
        return ("ok", "missing")

    def groups(self):
        return {}

    def run(self, steered, item_ids, *, store, replicate):
        self.calls.append(list(item_ids))
        return {
            item_id: ItemResult(item_id, 1.0, "ok", 1, "text", {"s": 1.0}, generated=True)
            for item_id in item_ids if item_id != "missing"
        }


def test_an_omitted_item_is_retried_then_written_as_an_error(tmp_path, steered):
    store = Store.open(tmp_path)
    benchmark = _OmittingBenchmark()
    rows = evaluate_items(steered, benchmark, ["ok", "missing"], store=store, retries=2)
    assert benchmark.calls == [["ok", "missing"], ["missing"], ["missing"]]
    assert rows["ok"].status == "ok" and rows["ok"].completion == "text"
    assert (rows["missing"].status, rows["missing"].value, rows["missing"].completion) == ("error", 0.0, None)
    assert evaluate_items(steered, benchmark, ["ok", "missing"], store=store)["missing"].status == "error"
    assert len(benchmark.calls) == 3


def test_generate_requests_uses_stored_rollouts(tmp_path, steered, batches):
    store = Store.open(tmp_path)
    requests = [Request("a", "the cat"), Request("b", [{"role": "user", "content": "the dog"}])]
    first = generate_requests(steered, requests, store=store, benchmark_generate={}, replicate=0)
    second = generate_requests(steered, requests, store=store, benchmark_generate={}, replicate=0)
    assert all(generation.generated for generation in first.values())
    assert not any(generation.generated for generation in second.values())
    assert {key: g.completion for key, g in first.items()} == {key: g.completion for key, g in second.items()}
    sampled = generate_requests(
        steered, requests, store=store, benchmark_generate={"temperature": 0.9}, replicate=0,
    )
    assert {g.rollout_key for g in sampled.values()}.isdisjoint({g.rollout_key for g in first.values()})


def test_one_request_set_is_batched_alike_for_two_recipes(tmp_path, steered, batches):
    store = Store.open(tmp_path)
    requests = [Request(item_id, prompt) for item_id, prompt in PROMPTS.items()]
    generate_requests(steered, requests, store=store, benchmark_generate={}, replicate=0)
    first = list(batches)
    batches.clear()
    other = Steered(steered.pipeline, BASELINE, "other-resolved", steered.protocol)
    generated = generate_requests(other, requests, store=store, benchmark_generate={}, replicate=0)
    assert all(generation.generated for generation in generated.values())
    assert batches == first and len(first) == 2


TASK_FILE = '''
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import exact, includes
from inspect_ai.solver import generate


@task
def tiny():
    samples = [Sample(id=f"s{i}", input=f"the cat {i}", target="the", metadata={"index": i}) for i in range(4)]
    return Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=[includes(), exact()])
'''


class TestInspectBenchmark:
    def test_rows_for_the_requested_samples_with_every_score(self, tmp_path, steered):
        path = tmp_path / "tiny_task.py"
        path.write_text(textwrap.dedent(TASK_FILE))
        store = Store.open(tmp_path / "store")
        benchmark = InspectBenchmark(str(path), scorer="includes", generate={"max_tokens": 3, "temperature": 0})
        benchmark.prepare(store)
        assert benchmark.item_ids() == ("s0", "s1", "s2", "s3")
        rows = evaluate_items(steered, benchmark, ["s1", "s3"], store=store)
        assert sorted(rows) == ["s1", "s3"]
        for row in rows.values():
            assert row.status == "ok"
            assert set(row.scores) == {"includes", "exact"}
            assert row.completion is not None
        again = evaluate_items(steered, benchmark, ["s1", "s3"], store=store)
        assert not any(row.generated for row in again.values())
        assert {key: row.completion for key, row in again.items()} == {key: row.completion for key, row in rows.items()}

    def test_sample_ids_with_colons_and_glob_characters(self, tmp_path, steered):
        path = tmp_path / "tiny_task.py"
        path.write_text(textwrap.dedent(TASK_FILE).replace('id=f"s{i}"', 'id=f"q:{i}[x]*"'))
        store = Store.open(tmp_path / "store")
        benchmark = InspectBenchmark(str(path), scorer="includes")
        benchmark.prepare(store)
        rows = evaluate_items(steered, benchmark, ["q:1[x]*", "q:3[x]*"], store=store, retries=0)
        assert sorted(rows) == ["q:1[x]*", "q:3[x]*"]
        assert all(row.status == "ok" for row in rows.values())

    def test_editing_the_task_file_changes_the_id(self, tmp_path):
        path = tmp_path / "tiny_task.py"
        path.write_text(textwrap.dedent(TASK_FILE))
        first = InspectBenchmark(str(path)).id
        path.write_text(textwrap.dedent(TASK_FILE) + "\n# edited\n")
        assert InspectBenchmark(str(path)).id != first

    def test_group_key(self, tmp_path):
        path = tmp_path / "tiny_task.py"
        path.write_text(textwrap.dedent(TASK_FILE))
        store = Store.open(tmp_path / "store")
        benchmark = InspectBenchmark(str(path), group_key="index")
        benchmark.prepare(store)
        assert benchmark.groups() == {"s0": "0", "s1": "1", "s2": "2", "s3": "3"}
