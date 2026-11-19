"""Tests for the persistent store (`algorithms/core/store.py`)."""
import json
import subprocess
import sys
import textwrap

import pytest

from steerability.algorithms.core.protocol import Protocol, ProtocolFacts, ProtocolMismatchError
from steerability.algorithms.core.recipe import Recipe, entry
from steerability.algorithms.core.store import (
    EvidenceRow,
    FitRow,
    ItemRow,
    RolloutRow,
    ScoreRow,
    Store,
    StoreIntegrityError,
)
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt

PROTOCOL = Protocol.create("hf-internal-testing/tiny-random-LlamaForCausalLM")


def _evidence(item_id: str, value: float, replicate: int = 0) -> EvidenceRow:
    return EvidenceRow(
        protocol_id="p", resolved_id="r", benchmark_id="b", item_id=item_id, replicate=replicate, value=value,
        status="ok", tokens=3, rollout_key=None, detail={"scores": {"s": value}}, created_at="t",
    )


def test_layout(tmp_path):
    store = Store.open(tmp_path / "store")
    assert json.loads((tmp_path / "store" / "store.json").read_text()) == {"schema": "store/1"}
    assert (tmp_path / "store" / "index.sqlite").exists()
    assert store.artifacts.root == tmp_path / "store" / "artifacts"
    assert store.logs.is_dir()
    store.close()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "store.json").write_text('{"schema": "store/0"}')
    with pytest.raises(StoreIntegrityError, match="store/0"):
        Store.open(tmp_path / "other")


class TestWriteOnce:
    def test_put_returns_the_stored_row(self, tmp_path):
        store = Store.open(tmp_path)
        first = store.put_rollout(RolloutRow("k", "p", "r", "first", "stop", 2, "t1"))
        second = store.put_rollout(RolloutRow("k", "p", "r", "second", "length", 9, "t2"))
        assert first == second
        assert second.completion == "first"
        assert store.put_score(ScoreRow("s", "scorer", 1.0, None)) == store.put_score(ScoreRow("s", "scorer", 0.0, {}))
        assert store.put_evidence(_evidence("a", 1.0)).value == 1.0
        assert store.put_evidence(_evidence("a", 0.0)).value == 1.0
        fit = FitRow("f", "c", "base", "d", '{"x": 1}', 1.5, "t")
        assert store.put_fit(fit) == store.put_fit(FitRow("f", "c", "base", "d", '{"x": 2}', 9.0, "t2")) == fit

    def test_a_second_different_resolution_raises(self, tmp_path):
        store = Store.open(tmp_path)
        assert store.put_resolution("p", "recipe", "resolved-1") == "resolved-1"
        assert store.put_resolution("p", "recipe", "resolved-1") == "resolved-1"
        with pytest.raises(StoreIntegrityError, match="resolves to resolved-1"):
            store.put_resolution("p", "recipe", "resolved-2")

    def test_recipes_round_trip(self, tmp_path):
        store = Store.open(tmp_path)
        recipe = Recipe((entry(SystemPrompt, text="be brief"),))
        assert store.put_recipe(recipe) == recipe.canonical()
        assert store.get_recipe(recipe.id) == recipe.canonical()
        assert store.get_recipe("missing") is None

    def test_items_and_evidence_queries(self, tmp_path):
        store = Store.open(tmp_path)
        store.put_items([ItemRow("b", "1", "g", {"id": 1, "input": "x"}), ItemRow("b", "2", None, {"id": 2})])
        store.put_items([ItemRow("b", "1", "g", {"id": 1, "input": "changed"})])
        assert [row.payload for row in store.get_items("b")] == [{"id": 1, "input": "x"}, {"id": 2}]
        assert store.has_items("b") and not store.has_items("c")
        for item_id in ("1", "2"):
            store.put_evidence(_evidence(item_id, 1.0))
        store.put_evidence(_evidence("1", 0.5, replicate=1))
        assert sorted(store.get_evidence("p", "r", "b", replicate=0)) == ["1", "2"]
        assert list(store.get_evidence("p", "r", "b", ["1"], replicate=1)) == ["1"]
        assert [(row.item_id, row.replicate) for row in store.evidence_rows("p", "r", "b")] == [
            ("1", 0), ("2", 0), ("1", 1),
        ]


class TestProtocols:
    def test_facts_are_recorded_once_and_compared(self, tmp_path):
        store = Store.open(tmp_path)
        row = store.put_protocol(PROTOCOL)
        assert row.identity == PROTOCOL.identity()
        assert row.observed is None
        facts = ProtocolFacts(model_fingerprint="a", torch_dtype="float32")
        assert store.put_protocol(PROTOCOL, facts).observed == facts.to_dict()
        assert store.put_protocol(PROTOCOL, ProtocolFacts(model_fingerprint="a")).observed == facts.to_dict()
        with pytest.raises(ProtocolMismatchError, match="model_fingerprint"):
            store.put_protocol(PROTOCOL, ProtocolFacts(model_fingerprint="b"))

    def test_extension_schema_is_idempotent(self, tmp_path):
        store = Store.open(tmp_path)
        statement = "CREATE TABLE IF NOT EXISTS extra (key TEXT PRIMARY KEY)"
        store.ensure_schema("extra/1", [statement])
        store.ensure_schema("extra/1", [statement])


def test_two_processes_writing_concurrently_lose_no_rows(tmp_path):
    root = tmp_path / "shared"
    Store.open(root).close()
    script = textwrap.dedent(f"""
        import sys
        from steerability.algorithms.core.store import EvidenceRow, Store
        store = Store.open({str(root)!r})
        worker = sys.argv[1]
        for index in range(200):
            store.put_evidence(EvidenceRow("p", "r", "b", f"{{worker}}-{{index}}", 0, 1.0, "ok", None, None, None, "t"))
    """)
    processes = [subprocess.Popen([sys.executable, "-c", script, str(worker)]) for worker in range(3)]
    assert [process.wait(timeout=300) for process in processes] == [0, 0, 0]
    assert len(Store.open(root).get_evidence("p", "r", "b")) == 600
