"""Evaluating candidates on a synthetic store: stages, calibration, confirmation, and reports.

Differences against the baseline are exact in the synthetic world. With a margin of 0.05, 2,000 held-out items, the
two-level capability baseline (design variance close to 0.09), and the default power settings, the looks are
(32, 128, 400), and a point is confirmed when its loss is at most 0.0253 (second look) or 0.0319 (third look). On a
solved axis outside `study.monotone`, a ladder position is kept when its loss is at most 0.0420. The quadratic truth
puts the refined position 2.5 at a loss of 0.03125 and the position 3 at 0.045.
"""
import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.recipe import Choice
from steerability.algorithms.core.store import Store
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.measurement.evidence import batch, point_items
from steerability.optimization.study import Candidate
from tests.optimization.helpers import SMALL, StubFactory, Truth, build_study, caa_template, multiplier, prompt_template

DISCRETE = Choice((0.0, 1.0, 2.0, 3.0), ordered=True)
CAA = Candidate("caa", {})


def _quadratic(recipe, item_id):
    value = multiplier(recipe)
    if value is None:
        return Truth(gain=0.1)
    return Truth(loss=0.005 * value * value, gain=0.05 * value)


def _by_position(losses, gains, tokens=None):
    def truth(recipe, item_id):
        value = multiplier(recipe)
        if value is None:
            return Truth(gain=0.1)
        index = int(value)
        return Truth(loss=losses[index], gain=gains[index], tokens=(tokens or {}).get(index, 10))

    return truth


def _item_truth(gains, loss_of):
    """A truth with the gains of `_by_position` and an item-dependent loss `loss_of(index, item_id)`."""
    def truth(recipe, item_id):
        value = multiplier(recipe)
        if value is None:
            return Truth(gain=0.1)
        return Truth(loss=loss_of(int(value), item_id), gain=gains[int(value)])

    return truth


def _resolved(value, multiplier_range=DISCRETE):
    return f"resolved-{caa_template(multiplier_range).bind({'caa.multiplier': value}).recipe().id}"


def _session(store, truth, **kwargs) -> StudySession:
    study = build_study(store, truth, **kwargs)
    session = StudySession(study, store=store, factory=StubFactory(study.protocol))
    session.start()
    return session


@pytest.fixture()
def store(tmp_path):
    return Store.open(tmp_path / "store")


def _rows(store, table):
    return store.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class TestStages:
    def test_zero_loss_and_a_gain_is_accepted(self, store):
        session = _session(store, lambda recipe, item_id: Truth(gain=0.1), space={"prompt": prompt_template()})
        result = session.evaluate(Candidate("prompt", {"prompt.text": "be kind"}))
        assert (result.status, result.stage, result.reason) == ("accepted", 5, None)
        assert result.fitness == pytest.approx(0.1)
        assert result.operating.capability["capability"].mean == pytest.approx(0.0, abs=1e-12)
        assert {"gain", "gain_lower", "level", "cost_ratio", "degenerate_increase", "loss.capability",
                "position"} <= set(result.summary)
        assert result.generations > 0
        assert _rows(store, "results") == 1

    def test_no_gain_at_any_probe_is_rejected_at_stage_two(self, store):
        session = _session(store, _by_position([0.0] * 4, [0.0] * 4), space={"caa": caa_template(DISCRETE)})
        result = session.evaluate(CAA)
        assert (result.status, result.stage, result.reason, result.fitness) == ("rejected", 2, "no_behavior_gain", 0.0)
        assert _rows(store, "results") == 1

    def test_a_gain_that_collapses_only_at_the_strongest_position_passes_stage_two(self, store):
        truth = _by_position([0.0] * 4, [0.1, 0.1, 0.1, 0.0])
        result = _session(store, truth, space={"caa": caa_template(DISCRETE)}).evaluate(CAA)
        assert result.status == "accepted" and result.stage == 5

    def test_every_probe_clearly_over_the_margin_is_rejected_at_stage_three(self, store):
        session = _session(store, _by_position([0.2] * 4, [0.1] * 4), space={"caa": caa_template(DISCRETE)})
        result = session.evaluate(CAA)
        assert (result.status, result.stage, result.reason) == ("rejected", 3, "capability_fail")
        assert result.summary["loss.capability"] == pytest.approx(0.2)

    def test_inside_the_margin_but_outside_the_band_is_rejected_with_fitness_zero(self, store):
        session = _session(store, lambda recipe, item_id: Truth(loss=0.04, gain=0.2),
                           space={"prompt": prompt_template()})
        result = session.evaluate(Candidate("prompt", {"prompt.text": "be kind"}))
        assert (result.status, result.stage, result.reason, result.fitness) == ("rejected", 4, "not_confirmed", 0.0)
        assert result.summary["loss.capability"] == pytest.approx(0.04)

    def test_a_degenerate_point_is_not_eligible(self, store):
        session = _session(store, lambda recipe, item_id: Truth(gain=0.1, completion=""),
                           space={"prompt": prompt_template()})
        result = session.evaluate(Candidate("prompt", {"prompt.text": "be kind"}))
        assert (result.status, result.reason) == ("rejected", "degenerate")

    def test_a_candidate_that_does_not_bind_is_invalid_and_not_recorded(self, store):
        session = _session(store, _quadratic, space={"prompt": prompt_template()})
        partial = session.screen(Candidate("prompt", {"prompt.text": "be loud"}))
        assert partial.candidate_id is None and "outside" in partial.error
        result = session.evaluate(Candidate("prompt", {"prompt.text": "be loud"}))
        assert (result.status, result.reason, result.candidate_id) == ("rejected", "invalid", "")
        assert _rows(store, "results") == 0 and _rows(store, "candidates") == 0

    def test_a_second_evaluation_does_no_work(self, store):
        session = _session(store, _quadratic, monotone={"caa.multiplier"})
        first = session.evaluate(CAA)
        capability = session.study.constraints[0].benchmark
        calls, steered = len(capability.calls), len(session.factory.steered_recipes)
        assert session.evaluate(CAA) == first
        assert (len(capability.calls), len(session.factory.steered_recipes)) == (calls, steered)


class TestCalibration:
    def test_a_monotone_axis_is_refined_and_the_strongest_confirmable_position_is_picked(self, store):
        result = _session(store, _quadratic, monotone={"caa.multiplier"}).evaluate(CAA)
        assert result.status == "accepted"
        assert result.operating.position["caa.multiplier"] == pytest.approx(2.5)
        assert [point.position["caa.multiplier"] for point in result.ladder] == pytest.approx([0, 1, 2, 2.5, 3])
        ranks = {point.position["caa.multiplier"]: point.rank for point in result.ladder}
        assert ranks[2.5] == 0 and ranks[3.0] is None
        assert result.fitness == pytest.approx(0.125)

    def test_without_monotone_each_position_is_judged_on_its_own(self, store, tmp_path):
        truth = _by_position([0.0, 0.09, 0.09, 0.0], [0.0, 0.1, 0.2, 0.3])
        result = _session(store, truth, space={"caa": caa_template(DISCRETE)}).evaluate(CAA)
        assert result.operating.position == {"caa.multiplier": 3.0}
        monotone = _session(Store.open(tmp_path / "monotone"), truth, space={"caa": caa_template(DISCRETE)},
                            monotone={"caa.multiplier"})
        assert monotone.evaluate(CAA).operating.position == {"caa.multiplier": 0.0}

    def test_the_best_confirmable_position_is_picked(self, store):
        truth = _by_position([0.0, 0.09, 0.04, 0.0], [0.0, 0.1, 0.4, 0.3])
        result = _session(store, truth, space={"caa": caa_template(DISCRETE)}).evaluate(CAA)
        assert result.operating.position == {"caa.multiplier": 3.0}
        ranks = {point.position["caa.multiplier"]: point.rank for point in result.ladder}
        assert (ranks[2.0], ranks[3.0], ranks[1.0]) == (0, 1, None)

    def test_a_position_over_the_cost_limit_is_skipped_for_the_next_ranked_one(self, store):
        truth = _by_position([0.0] * 4, [0.0, 0.1, 0.2, 0.3], tokens={3: 30})
        result = _session(store, truth, space={"caa": caa_template(DISCRETE)}).evaluate(CAA)
        assert result.operating.position == {"caa.multiplier": 2.0}
        assert {point.position["caa.multiplier"]: point.rank for point in result.ladder}[3.0] is None
        assert result.operating.cost_ratio == pytest.approx(1.0)

    def test_smaller_samples_are_prefixes_of_their_block(self, store):
        session = _session(store, _quadratic, monotone={"caa.multiplier"})
        session.evaluate(CAA)
        behavior = session.study.objective
        ladder = session.batches(0)[behavior.id].ladder
        first = caa_template().bind({"caa.multiplier": 0.0}).recipe()
        calls = [items for resolved, items, _ in behavior.calls if resolved == f"resolved-{first.id}"]
        assert calls[:2] == [ladder[:SMALL.quick_behavior_items], ladder[SMALL.quick_behavior_items:]]

    def test_no_kept_position_is_rejected_with_its_ladder(self, store):
        truth = _by_position([0.046] * 4, [0.1] * 4)
        result = _session(store, truth, space={"caa": caa_template(DISCRETE)}).evaluate(CAA)
        assert (result.status, result.stage, result.reason) == ("rejected", 4, "no_feasible_point")
        assert len(result.ladder) == 4 and all(point.rank is None for point in result.ladder)



class TestMonotoneCalibration:
    def test_a_ladder_estimate_alone_does_not_reject_a_candidate(self, tmp_path):
        gains = [0.1, 0.2, 0.3, 0.4]
        results, sessions = {}, {}
        for name, monotone in (("ladder", frozenset()), ("monotone", frozenset({"caa.multiplier"}))):
            session = _session(Store.open(tmp_path / name), _by_position([0.0] * 4, gains),
                               space={"caa": caa_template(DISCRETE)}, monotone=monotone)
            capability = session.study.constraints[0].benchmark
            ladder = set(session.batches(0)[capability.id].ladder)
            truth = _item_truth(gains, lambda index, item_id: 0.046 if item_id in ladder else 0.0)
            capability.set_truth(truth)
            session.study.objective.set_truth(truth)
            results[name], sessions[name] = session.evaluate(CAA), session
        assert (results["ladder"].status, results["ladder"].reason) == ("rejected", "no_feasible_point")
        session = sessions["ladder"]
        fresh = {item for benchmark in session.study.benchmarks for item in session.batches(0)[benchmark.id].fresh}
        for benchmark in session.study.benchmarks:
            assert not any(benchmark.measured(_resolved(float(value))) & fresh for value in range(4))
        assert results["monotone"].status == "accepted"
        assert results["monotone"].operating.position == {"caa.multiplier": 3.0}

    def test_positions_above_a_clearly_over_one_are_not_measured(self, store):
        session = _session(store, _by_position([0.0, 0.0, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4]),
                           space={"caa": caa_template(DISCRETE)}, monotone={"caa.multiplier"})
        result = session.evaluate(CAA)
        capability, behavior = session.study.constraints[0].benchmark, session.study.objective
        first_look = set(session.batches(0)[capability.id].ladder[:session.looks(capability)[0]])
        assert not capability.measured(_resolved(3.0)) and not behavior.measured(_resolved(3.0))
        assert capability.measured(_resolved(2.0)) == first_look and not behavior.measured(_resolved(2.0))
        assert len(result.ladder) == 3
        assert result.status == "accepted" and result.operating.position == {"caa.multiplier": 1.0}

    def test_bisection_finds_the_strongest_confirmable_position(self, store):
        positions = Choice(tuple(float(index) for index in range(8)), ordered=True)
        truth = _by_position([0.006 * index for index in range(8)], [0.05 * (index + 1) for index in range(8)])
        session = _session(store, truth, space={"caa": caa_template(positions)}, monotone={"caa.multiplier"})
        result = session.evaluate(CAA)
        assert result.operating.position == {"caa.multiplier": 5.0}
        capability = session.study.constraints[0].benchmark
        fresh = set(session.batches(0)[capability.id].fresh)
        for value in (0, 1, 2, 4, 7):
            assert not capability.measured(_resolved(float(value), positions)) & fresh

    def test_the_search_stops_at_the_best_gain(self, store):
        session = _session(store, _by_position([0.0] * 4, [0.1, 0.4, 0.2, 0.3]),
                           space={"caa": caa_template(DISCRETE)}, monotone={"caa.multiplier"})
        result = session.evaluate(CAA)
        assert result.operating.position == {"caa.multiplier": 1.0}
        capability = session.study.constraints[0].benchmark
        fresh = set(session.batches(0)[capability.id].fresh)
        for value in (2.0, 3.0):
            assert not capability.measured(_resolved(value)) & fresh

    def test_a_position_confirmed_on_the_second_look_is_picked(self, store):
        gains = [0.1, 0.2, 0.3, 0.4]
        session = _session(store, _by_position([0.0] * 4, gains), space={"caa": caa_template(DISCRETE)},
                           monotone={"caa.multiplier"})
        capability = session.study.constraints[0].benchmark
        second_look = session.looks(capability)[1]
        late = set(session.batches(0)[capability.id].fresh[second_look:])
        truth = _item_truth(gains, lambda index, item_id: 0.3 if index == 3 and item_id in late else 0.0)
        capability.set_truth(truth)
        session.study.objective.set_truth(truth)
        result = session.evaluate(CAA)
        assert result.operating.position == {"caa.multiplier": 3.0}
        assert result.operating.capability["capability"].n == second_look

class TestReporting:
    def test_fitness_is_reported_on_the_same_number_of_items_with_and_without_a_solved_axis(self, store):
        session = _session(store, _quadratic, space={"caa": caa_template(), "prompt": prompt_template()},
                           monotone={"caa.multiplier"})
        solved = session.evaluate(CAA)
        plain = session.evaluate(Candidate("prompt", {"prompt.text": "be kind"}))
        assert solved.operating.behavior.n == plain.operating.behavior.n == SMALL.behavior_items

    def test_operating_tables_use_no_ladder_item_also_after_a_rescore(self, store):
        session = _session(store, _quadratic, space={"caa": caa_template(), "prompt": prompt_template("abc")},
                           monotone={"caa.multiplier"})
        for candidate in (CAA, *(Candidate("prompt", {"prompt.text": text}) for text in "abc")):
            session.evaluate(candidate)
        assert session.round_index == 1
        study = session.study
        latest = session.archive.latest(study.candidate_id(CAA))
        assert (latest.version, latest.round_index, latest.status) == (2, 0, "accepted")
        capability, behavior = study.constraints[0].benchmark, study.objective
        for benchmark, table in ((capability, latest.operating.capability["capability"]),
                                 (behavior, latest.operating.behavior)):
            items = point_items(study, store, latest.operating.resolved_id, benchmark, session.rounds, 0)
            assert not set(items) & set(session.rounds[0][benchmark.id].ladder)
            assert table.n == len(items)
        assert latest.operating.capability["capability"].n > session.looks(capability)[2]
        assert latest.operating.behavior.n > SMALL.behavior_items

    def test_rescore_falls_to_the_next_ranked_position(self, store):
        session = _session(store, _quadratic, space={"caa": caa_template(), "prompt": prompt_template("abc")},
                           monotone={"caa.multiplier"})
        study = session.study
        capability, behavior = study.constraints[0].benchmark, study.objective
        first, second = session.batches(0)[capability.id], batch(study, store, capability, 1)
        poisoned = set(second.fresh) - set(first.ladder) - set(first.fresh)

        def truth(recipe, item_id):
            value = multiplier(recipe)
            if value == 2.5 and item_id in poisoned:
                return Truth(loss=0.3, gain=0.125)
            return _quadratic(recipe, item_id)

        capability.set_truth(truth)
        behavior.set_truth(truth)
        for candidate in (CAA, *(Candidate("prompt", {"prompt.text": text}) for text in "abc")):
            session.evaluate(candidate)
        versions = session.archive.versions(study.candidate_id(CAA))
        assert versions[0].operating.position["caa.multiplier"] == pytest.approx(2.5)
        assert len(versions) == 2 and versions[1].status == "accepted"
        assert versions[1].operating.position == {"caa.multiplier": 2.0}
        assert versions[1].fitness == pytest.approx(0.1)
