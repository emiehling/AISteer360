"""Tests for the baseline, usable items, paired tables, and batches (`optimization/measurement/evidence.py`)."""
import dataclasses
import logging
import random

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.identity import derive_seed
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.store import Store
from steerability.evaluation.items import evaluate_items
from steerability.optimization.measurement.evidence import (
    batch,
    degenerate,
    ensure_baseline,
    noise_floor,
    paired,
    sd_floor,
    usable,
)
from steerability.optimization.study import StudyError, StudyInfeasibleError
from tests.optimization.helpers import (
    StubFactory,
    SyntheticBenchmark,
    Truth,
    benchmarks,
    build_study,
    caa_template,
    multiplier,
)


def _quiet(recipe, item_id):
    return Truth()


@pytest.fixture()
def store(tmp_path):
    return Store.open(tmp_path / "store")


def _started(store, truth=_quiet, **kwargs):
    study = build_study(store, truth, **kwargs)
    factory = StubFactory(study.protocol)
    ensure_baseline(study, factory, store)
    return study, factory


class TestBaseline:
    def test_the_baseline_measures_every_item_and_the_search_items_again(self, store):
        study, factory = _started(store)
        capability = study.constraints[0].benchmark
        split = study.pools[capability.id]
        measured = {replicate: set(items) for _, items, replicate in capability.calls}
        assert measured == {0: set(split.search) | set(split.sealed), 1: set(split.search)}
        assert noise_floor(study, store, capability) == 0.0
        assert sd_floor(study, store, capability) == pytest.approx(0.02 ** 0.5)
        calls = len(capability.calls)
        ensure_baseline(study, factory, store)
        assert len(capability.calls) == calls

    def test_a_baseline_error_excludes_its_item_from_every_batch(self, store):
        probe = build_study(Store.open(store.root.parent / "probe"), _quiet)
        dropped = probe.pools[probe.constraints[0].benchmark.id].search[0]
        capability, behavior = benchmarks(_quiet, capability_omit={dropped})
        study, _ = _started(store, capability=capability, behavior=behavior)
        assert dropped not in usable(study, store, capability, "search")
        assert len(usable(study, store, capability, "search")) == len(study.pools[capability.id].search) - 1
        for round_index in range(4):
            drawn = batch(study, store, capability, round_index)
            assert dropped not in drawn.ladder + drawn.fresh

    def test_too_many_baseline_errors_raise(self, store):
        probe = build_study(Store.open(store.root.parent / "probe"), _quiet)
        search = probe.pools[probe.constraints[0].benchmark.id].search
        capability, behavior = benchmarks(_quiet, capability_omit=set(search[:30]))
        with pytest.raises(StudyError, match="error rows"):
            _started(store, capability=capability, behavior=behavior)

    def test_a_baseline_value_outside_the_unit_interval_raises(self, store):
        capability, behavior = benchmarks(_quiet)
        behavior.base = 1.5
        with pytest.raises(StudyError, match=r"\[0, 1\]"):
            _started(store, capability=capability, behavior=behavior)

    def test_the_noise_floor_is_the_share_of_items_that_differ_between_replicates(self, store):
        class Noisy(SyntheticBenchmark):
            def run(self, steered, item_ids, *, store, replicate):
                results = super().run(steered, item_ids, store=store, replicate=replicate)
                if replicate == 1:
                    for item_id in list(results)[:50]:
                        results[item_id] = dataclasses.replace(results[item_id], value=0.0)
                return results

        capability = Noisy("capability", 3000, kind="capability", truth=_quiet, base=0.8)
        _, behavior = benchmarks(_quiet)
        study, _ = _started(store, capability=capability, behavior=behavior)
        assert noise_floor(study, store, capability) == pytest.approx(50 / 1000)

    def test_a_sampling_protocol_measures_four_replicates_of_both_pools(self, store):
        study = build_study(store, _quiet)
        study = dataclasses.replace(study, protocol=Protocol.create("synthetic-model"))
        ensure_baseline(study, StubFactory(study.protocol), store)
        capability = study.constraints[0].benchmark
        assert sorted({replicate for _, _, replicate in capability.calls}) == [0, 1, 2, 3]


class TestPaired:
    def test_a_sampling_protocol_pairs_against_the_mean_baseline_value(self, store):
        class Drifting(SyntheticBenchmark):
            def run(self, steered, item_ids, *, store, replicate):
                results = super().run(steered, item_ids, store=store, replicate=replicate)
                if not steered.recipe.entries:
                    for item_id in results:
                        results[item_id] = dataclasses.replace(results[item_id],
                                                               value=results[item_id].value + 0.04 * replicate)
                return results

        capability = Drifting("capability", 3000, kind="capability", truth=_quiet, base=0.8)
        _, behavior = benchmarks(_quiet)
        study = build_study(store, _quiet, capability=capability, behavior=behavior)
        study = dataclasses.replace(study, protocol=Protocol.create("synthetic-model"))
        factory = StubFactory(study.protocol)
        ensure_baseline(study, factory, store)
        items = list(study.pools[capability.id].search[:20])
        recipe = caa_template().bind({"caa.multiplier": 1.0}).recipe()
        with factory.steered(recipe) as steered:
            evaluate_items(steered, capability, items, store=store)
            table = paired(study, store, steered.resolved_id, capability, items)
        assert table.mean == pytest.approx(-0.04 * (0 + 1 + 2 + 3) / 4)

    def test_differences_are_paired_against_the_baseline(self, store):
        def truth(recipe, item_id):
            return Truth(loss=0.1 * multiplier(recipe), gain=0.05)

        study, factory = _started(store, truth)
        capability = study.constraints[0].benchmark
        recipe = caa_template().bind({"caa.multiplier": 1.0}).recipe()
        items = list(study.pools[capability.id].search[:40])
        with factory.steered(recipe) as steered:
            evaluate_items(steered, capability, items, store=store)
            table = paired(study, store, steered.resolved_id, capability, items)
        assert table.n == 40 and table.mean == pytest.approx(-0.1)
        with pytest.raises(StudyError, match="no evidence row"):
            paired(study, store, steered.resolved_id, capability, [study.pools[capability.id].search[50]])

    def test_an_error_row_counts_with_its_value_of_zero(self, store):
        study, factory = _started(store)
        capability = study.constraints[0].benchmark
        items = list(study.pools[capability.id].search[:10])
        capability.omit = {items[0]}
        recipe = caa_template().bind({"caa.multiplier": 1.0}).recipe()
        with factory.steered(recipe) as steered:
            evaluate_items(steered, capability, items, store=store)
            table = paired(study, store, steered.resolved_id, capability, items)
        assert table.total == pytest.approx(-0.8)


class TestBatches:
    def test_blocks_are_disjoint_seeded_draws_of_the_usable_pool(self, store):
        study, _ = _started(store)
        capability = study.constraints[0].benchmark
        drawn = batch(study, store, capability, 0)
        assert (len(drawn.ladder), len(drawn.fresh)) == (64, 400)
        assert not set(drawn.ladder) & set(drawn.fresh)
        expected = random.Random(derive_seed(study.seed, "batch", capability.id, 0)).sample(
            sorted(usable(study, store, capability, "search")), 464)
        assert list(drawn.ladder + drawn.fresh) == expected
        assert batch(study, store, capability, 0) == drawn
        assert batch(study, store, capability, 1) != drawn
        assert not drawn.overlapping

    def test_the_small_pool_rule_applies_only_under_allow_infeasible(self, store, caplog):
        study, _ = _started(store, capability_sealed=2800)
        capability = study.constraints[0].benchmark
        with pytest.raises(StudyInfeasibleError, match="fewer than"):
            batch(study, store, capability, 0)
        with caplog.at_level(logging.WARNING):
            drawn = batch(study, store, capability, 0, allow_infeasible=True)
        assert "chooses and judges on the same items" in caplog.text
        assert drawn.overlapping
        assert drawn.ladder == drawn.fresh[:64]
        assert sorted(drawn.fresh) == sorted(usable(study, store, capability, "search"))


def test_degenerate_completions():
    assert degenerate("") and degenerate("   ") and degenerate(None)
    assert degenerate("a a a a a a a a a")
    assert not degenerate("one two three four five six")
    assert not degenerate("two words")
