"""Tests for the baseline, paired tables, sizes, feasibility, and batches (`optimization/measurement/evidence.py`)."""
import dataclasses
import logging
import random

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.identity import derive_seed
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.store import Store
from steerability.evaluation.items import evaluate_items
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.measurement.evidence import (
    NOISE_PROBE_ITEMS,
    baseline_variance,
    batch,
    block_sizes,
    degenerate,
    design_variance,
    ensure_baseline,
    feasibility,
    noise_floor,
    paired,
    usable,
)
from steerability.optimization.measurement.pools import split_items
from steerability.optimization.measurement.stats import floor_variance
from steerability.optimization.study import Constraint, Study, StudyError
from tests.optimization.helpers import (
    StubFactory,
    SyntheticBenchmark,
    Truth,
    benchmarks,
    build_study,
    caa_template,
    multiplier,
    protocol,
    two_level,
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


class Noisy(SyntheticBenchmark):
    """A benchmark whose `flipped` items score 0.0 instead of their baseline value at replicate 1 of the baseline."""

    flipped: frozenset = frozenset()

    def run(self, steered, item_ids, *, store, replicate):
        results = super().run(steered, item_ids, store=store, replicate=replicate)
        if replicate == 1 and not steered.recipe.entries:
            for item_id in set(results) & self.flipped:
                results[item_id] = dataclasses.replace(results[item_id], value=0.0)
        return results


def _noisy_study(store, *, every: int):
    capability = Noisy("capability", 3000, kind="capability", truth=_quiet, base=1.0)
    _, behavior = benchmarks(_quiet)
    study = build_study(store, _quiet, capability=capability, behavior=behavior)
    capability.flipped = frozenset(study.pools[capability.id].search[::every])
    return study, capability


class TestBaseline:
    def test_the_baseline_measures_every_item_and_the_search_items_again(self, store):
        study, factory = _started(store)
        capability = study.constraints[0].benchmark
        split = study.pools[capability.id]
        measured = {}
        for _, items, replicate in capability.calls:
            measured.setdefault(replicate, set()).update(items)
        assert measured == {0: set(split.search) | set(split.sealed), 1: set(split.search)}
        assert noise_floor(study, store, capability) == 0.0
        calls = len(capability.calls)
        ensure_baseline(study, factory, store)
        assert len(capability.calls) == calls

    def test_the_sealed_pool_is_its_own_request_set(self, store):
        study, _ = _started(store)
        capability = study.constraints[0].benchmark
        split = study.pools[capability.id]
        sealed_calls = [items for _, items, _ in capability.calls if set(items) & set(split.sealed)]
        assert sealed_calls == [split.sealed]

    def test_the_probe_comes_first_and_its_rows_are_reused(self, store):
        study, _ = _started(store)
        for benchmark in study.benchmarks:
            probe = study.pools[benchmark.id].search[:NOISE_PROBE_ITEMS]
            assert [(items, replicate) for _, items, replicate in benchmark.calls[:2]] == [(probe, 0), (probe, 1)]
            later = {item for _, items, _ in benchmark.calls[2:] for item in items}
            assert not later & set(probe)

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

    def test_a_batch_dependent_baseline_raises_after_the_probe(self, store):
        study, capability = _noisy_study(store, every=20)
        with pytest.raises(StudyError, match="different batch orders"):
            ensure_baseline(study, StubFactory(study.protocol), store)
        assert sum(len(items) for _, items, _ in capability.calls) <= 2 * NOISE_PROBE_ITEMS
        assert store.execute("SELECT COUNT(*) FROM baselines").fetchone()[0] == 0

    def test_allow_infeasible_does_not_waive_the_noise_limit(self, store):
        study, _ = _noisy_study(store, every=20)
        session = StudySession(study, store=store, factory=StubFactory(study.protocol), allow_infeasible=True)
        with pytest.raises(StudyError, match="different batch orders"):
            session.start()

    def test_the_noise_floor_is_the_mean_absolute_difference_between_the_replicates(self, store):
        study, capability = _noisy_study(store, every=200)
        ensure_baseline(study, StubFactory(study.protocol), store)
        assert noise_floor(study, store, capability) == pytest.approx(0.005)

    def test_a_sampling_protocol_measures_four_replicates_of_each_pool_separately(self, store):
        study = build_study(store, _quiet)
        study = dataclasses.replace(study, protocol=Protocol.create("synthetic-model",
                                                                    generate_defaults={"temperature": 1.0}))
        ensure_baseline(study, StubFactory(study.protocol), store)
        capability = study.constraints[0].benchmark
        split = study.pools[capability.id]
        calls = [(items, replicate) for _, items, replicate in capability.calls]
        assert calls == [(pool, replicate) for replicate in range(4) for pool in (split.search, split.sealed)]

    def test_a_constraint_that_cannot_fail_raises(self, store):
        capability = SyntheticBenchmark("capability", 3000, kind="capability", truth=_quiet, base=0.03)
        _, behavior = benchmarks(_quiet)
        with pytest.raises(StudyError, match="cannot fail"):
            _started(store, capability=capability, behavior=behavior)


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
        study = dataclasses.replace(study, protocol=Protocol.create("synthetic-model",
                                                                    generate_defaults={"temperature": 1.0}))
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
        assert table.total == pytest.approx(-capability.baseline(items[0]))


class TestDesignVariance:
    def test_a_zero_one_baseline_gives_p_times_one_minus_p(self, store):
        capability = SyntheticBenchmark("capability", 3000, kind="capability", truth=_quiet,
                                        base=two_level("capability", 3000, high=1.0, low=0.0))
        _, behavior = benchmarks(_quiet)
        study, _ = _started(store, capability=capability, behavior=behavior)
        values = [capability.baseline(item) for item in usable(study, store, capability, "search")]
        share = sum(values) / len(values)
        assert baseline_variance(study, store, capability) == pytest.approx(share * (1 - share))
        assert design_variance(study, store, capability) == pytest.approx(share * (1 - share))

    def test_the_floor_without_a_baseline_and_for_a_constant_one(self, store):
        capability = SyntheticBenchmark("capability", 3000, kind="capability", truth=_quiet, base=0.8)
        _, behavior = benchmarks(_quiet)
        study = build_study(store, _quiet, capability=capability, behavior=behavior)
        assert baseline_variance(study, store, capability) is None
        assert design_variance(study, store, capability) == pytest.approx(floor_variance(0.05))
        ensure_baseline(study, StubFactory(study.protocol), store)
        assert baseline_variance(study, store, capability) == pytest.approx(0.0, abs=1e-12)
        assert design_variance(study, store, capability) == pytest.approx(floor_variance(0.05))
        assert block_sizes(study, store, capability) == (50, 200)

    def test_errored_items_are_left_out(self, store):
        base = two_level("capability", 3000, high=1.0, low=0.0)
        capability = SyntheticBenchmark("capability", 3000, kind="capability", truth=_quiet, base=base)
        _, behavior = benchmarks(_quiet)
        study = build_study(store, _quiet, capability=capability, behavior=behavior)
        search = study.pools[capability.id].search
        dropped = set(sorted(item for item in search if base[item] == 1.0)[:15])
        capability.omit = dropped
        ensure_baseline(study, StubFactory(study.protocol), store)
        assert set(usable(study, store, capability, "search")) == set(search) - dropped
        values = [base[item] for item in search if item not in dropped]
        share = sum(values) / len(values)
        assert baseline_variance(study, store, capability) == pytest.approx(share * (1 - share))

    def test_the_objective_has_no_design_variance(self, store):
        study, _ = _started(store)
        with pytest.raises(ValueError, match="not a constraint"):
            design_variance(study, store, study.objective)

    def test_block_sizes_follow_the_design_variance(self, store):
        study, _ = _started(store)
        capability = study.constraints[0].benchmark
        assert design_variance(study, store, capability) == pytest.approx(0.09, abs=0.002)
        assert block_sizes(study, store, capability) == (100, 400)
        assert block_sizes(study, store, study.objective) == (32, 64)


class TestFeasibility:
    @staticmethod
    def _gsm8k(store):
        """A 0/1 GSM8K-sized benchmark in which exactly 1,733 of the 3,792 search items score 1."""
        capability = SyntheticBenchmark("gsm8k", 8792, kind="capability", truth=_quiet, base=0.0)
        behavior = SyntheticBenchmark("formality", 1000, kind="behavior", truth=_quiet, base=0.3)
        for benchmark in (capability, behavior):
            benchmark.prepare(store)
        pools = {capability.id: split_items(capability, sealed=5000, seed=0),
                 behavior.id: split_items(behavior, sealed=200, seed=0)}
        search = pools[capability.id].search
        assert len(search) == 3792
        capability.base = {**{item: 0.0 for item in capability.items}, **{item: 1.0 for item in search[:1733]}}

        def study(margin):
            built = Study(name="gsm8k", protocol=protocol(), space={"caa": caa_template()}, objective=behavior,
                          constraints=(Constraint(capability, margin),), pools=pools)
            ensure_baseline(built, StubFactory(built.protocol), store)
            return built

        return study

    def test_the_gsm8k_example(self, store):
        study = self._gsm8k(store)
        at_three = feasibility(study(0.03), store)
        assert len(at_three) == 3
        room, band, alternative = at_three
        assert "6,750" in room and "3,792" in room and "2,700" in room and "675" in room
        assert "2.16 points" in band and "3-point margin" in band
        assert "4.0 points" in alternative and "0.14" in alternative
        assert feasibility(study(0.04), store) == []
        assert feasibility(study(0.05), store) == []

    def test_the_band_rule_alone(self, store):
        study, _ = _started(store, capability_sealed=200)
        problems = feasibility(study, store)
        assert any("guard band" in problem for problem in problems)
        assert not any("two rounds" in problem for problem in problems)

    def test_the_room_rule_alone(self, store):
        study, _ = _started(store, capability_sealed=2100)
        problems = feasibility(study, store)
        assert any("two rounds of blocks need 1,000 search items" in problem for problem in problems)
        assert not any("guard band" in problem for problem in problems)

    def test_the_objective_rule(self, store):
        study, _ = _started(store, behavior_sealed=250)
        assert feasibility(study, store) == [
            "behavior: two rounds of behavior blocks need 192 search items and the pool has 150"
        ]

    def test_before_the_baseline_the_floor_and_raw_sizes_are_used(self, store):
        study = build_study(store, _quiet, margin=0.005)
        problems = feasibility(study, store)
        assert "two rounds of blocks need 5,000 search items" in problems[0]


class TestBatches:
    def test_blocks_are_disjoint_seeded_draws_of_the_usable_pool(self, store):
        study, _ = _started(store)
        capability = study.constraints[0].benchmark
        drawn = batch(study, store, capability, 0)
        assert (len(drawn.ladder), len(drawn.fresh)) == (100, 400)
        assert not set(drawn.ladder) & set(drawn.fresh)
        expected = random.Random(derive_seed(study.seed, "batch", capability.id, 0)).sample(
            sorted(usable(study, store, capability, "search")), 500)
        assert list(drawn.ladder + drawn.fresh) == expected
        assert batch(study, store, capability, 0) == drawn
        assert batch(study, store, capability, 1) != drawn
        assert not drawn.overlapping

    def test_a_small_pool_gets_overlapping_blocks_with_a_warning(self, store, caplog):
        study, _ = _started(store, capability_sealed=2800)
        capability = study.constraints[0].benchmark
        with caplog.at_level(logging.WARNING):
            drawn = batch(study, store, capability, 0)
        assert "chooses and judges on the same items" in caplog.text
        assert drawn.overlapping
        assert drawn.ladder == drawn.fresh[:100]
        assert sorted(drawn.fresh) == sorted(usable(study, store, capability, "search"))


def test_degenerate_completions():
    assert degenerate("") and degenerate("   ") and degenerate(None)
    assert degenerate("a a a a a a a a a")
    assert not degenerate("one two three four five six")
    assert not degenerate("two words")
