"""Tests for the statistics of paired differences (`optimization/measurement/stats.py`)."""
import math

import numpy as np
import pytest

from steerability.optimization.measurement.stats import (
    Paired,
    certify_fixed_order,
    confirm_step,
    feasible,
    floor_variance,
    guard_band,
    isotonic_increasing,
    ladder_size,
    last_look,
    look_sizes,
    provisional_positions,
    rank_positions,
    release_band,
)

ALPHA = 0.05
POWER = 0.80
SEALED = 5000


def _diffs(rng, size, length, loss, change_rate=0.10):
    """Per-item differences in {-1, 0, 1} with the given change rate and mean `-loss`."""
    draws = rng.random((size, length))
    down = (change_rate + loss) / 2
    up = (change_rate - loss) / 2
    return np.where(draws < down, -1.0, np.where(draws < down + up, 1.0, 0.0))


def _losses(rng, size, length, loss):
    """Per-item differences in {-1, 0} whose only changes are losses, with mean `-loss`."""
    return np.where(rng.random((size, length)) < loss, -1.0, 0.0)


def _confirmed_share(margin, variance, loss, *, runs=20000, seed=0):
    looks = look_sizes(last_look(margin=margin, variance=variance))
    rng = np.random.default_rng(seed)
    confirmed = 0
    for start in range(0, runs, 2000):
        size = min(2000, runs - start)
        diffs = _diffs(rng, size, looks[-1], loss, change_rate=variance)
        totals = np.cumsum(diffs, axis=1)
        changed = np.cumsum(diffs != 0, axis=1)
        for run in range(size):
            for look, count in enumerate(looks):
                table = Paired(count, float(totals[run, count - 1]), float(changed[run, count - 1]),
                               int(changed[run, count - 1]))
                decision = confirm_step(table, look, margin, n_sealed=SEALED, alpha=ALPHA, power=POWER)
                if decision != "continue":
                    confirmed += decision == "accept"
                    break
    return confirmed / runs


def _sealed_pass_rate(margin, n_sealed):
    """The exact pass rate of the sealed test for a candidate on the margin whose changes are all losses."""
    from scipy.stats import binom

    losses = np.arange(n_sealed + 1)
    passes = np.array([Paired(n_sealed, -float(k), float(k), int(k)).passes(margin, ALPHA) for k in losses])
    return float(binom.pmf(losses, n_sealed, margin)[passes].sum())


class TestReferenceValues:
    def test_guard_band(self):
        se_sealed = math.sqrt(0.1 / 5000)
        assert guard_band(math.sqrt(0.1 / 2400), se_sealed, alpha=ALPHA, power=POWER) == pytest.approx(0.0140, abs=5e-5)
        assert guard_band(math.sqrt(0.1 / 1100), se_sealed, alpha=ALPHA, power=POWER) == pytest.approx(0.0162, abs=5e-5)
        assert guard_band(math.sqrt(0.1 / 1100), se_sealed, alpha=ALPHA, power=POWER,
                          benchmarks=3) == pytest.approx(0.0228, abs=5e-5)

    def test_floor_variance(self):
        assert floor_variance(0.03) == pytest.approx(0.0291)
        assert floor_variance(0.05) == pytest.approx(0.0475)

    def test_last_look(self):
        assert last_look(margin=0.03, variance=0.10) == 1100
        assert last_look(margin=0.02, variance=0.10) == 2400
        assert last_look(margin=0.05, variance=0.25) == 1000
        assert last_look(margin=0.05, variance=0.0475) == 200

    def test_look_sizes(self):
        expected = {
            1100: (128, 512, 1100), 2400: (128, 1024, 2400), 1000: (64, 256, 1000), 400: (32, 128, 400),
            200: (32, 64, 200), 100: (32, 64, 100), 60: (32, 60, 60),
        }
        assert {last: look_sizes(last) for last in expected} == expected
        with pytest.raises(ValueError, match="at least one item"):
            look_sizes(0)

    def test_ladder_size(self):
        assert (ladder_size(1000), ladder_size(100)) == (250, 32)

    def test_the_floor_in_the_sealed_test(self):
        assert not Paired(65, 0.0, 0.0, 0).passes(0.03, ALPHA)
        bound = Paired(200, 0.0, 0.0, 0).lower(ALPHA, floor=floor_variance(0.03))
        assert bound == pytest.approx(-0.0198, abs=5e-5)
        assert Paired(200, 0.0, 0.0, 0).passes(0.03, ALPHA)

    def test_isotonic_example(self):
        fitted = isotonic_increasing([0.1, 0.3, 0.2, 0.25, 0.6, 0.5], [1, 2, 1, 3, 1, 1])
        assert fitted == pytest.approx([0.1, 0.258333, 0.258333, 0.258333, 0.55, 0.55], abs=1e-5)

    def test_isotonic_requires_positive_weights(self):
        with pytest.raises(ValueError, match="positive"):
            isotonic_increasing([0.1, 0.2], [1, 0])


class TestPaired:
    def test_stored_form_round_trips(self):
        table = Paired.from_diffs([0.0, -1.0, 1.0, -1.0, 0.25])
        assert Paired.from_list(table.as_list()) == table
        assert table.n == 5 and table.changed == 4

    def test_pooling_adds_the_statistics(self):
        first, second = Paired.from_diffs([1.0, 0.0]), Paired.from_diffs([-1.0, -1.0, 0.5])
        assert first + second == Paired.from_diffs([1.0, 0.0, -1.0, -1.0, 0.5])

    def test_variance_and_the_floor_of_the_standard_error(self):
        table = Paired.from_diffs([0.0] * 100)
        assert table.variance == 0.0 and table.se() == 0.0
        assert table.se(floor=0.0291) == pytest.approx(math.sqrt(0.0291 / 100))
        assert Paired.from_diffs([-1.0, 0.0, 0.0, 0.0]).variance == pytest.approx(0.25 * 0.75)
        assert table.passes(0.03, ALPHA)
        assert not Paired.from_diffs([-1.0] * 10 + [0.0] * 90).passes(0.03, ALPHA)

    def test_empty_tables(self):
        empty = Paired(0, 0.0, 0.0, 0)
        assert empty.mean == 0.0 and empty.change_rate == 0.0 and empty.variance == 0.0
        assert empty.se() == math.inf and empty.lower(ALPHA) == -math.inf and empty.upper(ALPHA) == math.inf


class TestBands:
    def test_no_guard_gives_a_zero_band(self):
        table = Paired.from_diffs([-1.0] * 5 + [0.0] * 95)
        assert release_band(table, 0.05, n_sealed=SEALED, alpha=ALPHA, power=POWER, guard=False) == 0.0
        assert feasible(table, 0.05, n_sealed=SEALED, alpha=ALPHA, power=POWER, guard=False)
        assert not feasible(table, 0.05, n_sealed=SEALED, alpha=ALPHA, power=POWER)

    def test_the_band_plans_for_a_larger_confirmation(self):
        table = Paired.from_diffs([-1.0] * 5 + [0.0] * 95)
        small = release_band(table, 0.05, n_sealed=SEALED, alpha=ALPHA, power=POWER)
        planned = release_band(table, 0.05, n_sealed=SEALED, alpha=ALPHA, power=POWER, n_search=1000)
        assert planned < small

    def test_the_band_takes_the_variance_as_at_least_the_floor(self):
        constant = Paired.from_diffs([-0.01] * 400)
        at_floor = guard_band(math.sqrt(0.0475 / 400), math.sqrt(0.0475 / SEALED), alpha=ALPHA, power=POWER)
        assert release_band(constant, 0.05, n_sealed=SEALED, alpha=ALPHA, power=POWER) == pytest.approx(at_floor)
        wide = Paired.from_diffs([-1.0, 1.0] * 200)
        assert release_band(wide, 0.05, n_sealed=SEALED, alpha=ALPHA, power=POWER) > at_floor

    def test_look_zero_only_rejects(self):
        inside = Paired.from_diffs([0.0] * 400)
        assert confirm_step(inside, 0, 0.03, n_sealed=SEALED, alpha=ALPHA, power=POWER) == "continue"
        assert confirm_step(inside, 1, 0.03, n_sealed=SEALED, alpha=ALPHA, power=POWER) == "accept"
        over = Paired.from_diffs([-1.0] * 40 + [0.0] * 60)
        assert confirm_step(over, 0, 0.03, n_sealed=SEALED, alpha=ALPHA, power=POWER) == "reject"


class TestLadderDecisions:
    def test_monotone_positions_stop_at_the_first_outside(self):
        losses = [Paired.from_diffs([-x] * 256) for x in (0.0, 0.01, 0.02, 0.09, 0.03)]
        kept = provisional_positions(losses, 0.05, n_target=400, n_sealed=SEALED, monotone=True, alpha=ALPHA,
                                     power=0.5)
        assert kept == [0, 1, 2]
        each = provisional_positions(losses, 0.05, n_target=400, n_sealed=SEALED, monotone=False, alpha=ALPHA,
                                     power=0.5)
        assert each == [0, 1, 2, 4]

    def test_ranking_orders_by_the_gain_lower_bound(self):
        gains = [Paired.from_diffs([value] * 64) for value in (0.1, 0.3, 0.2)]
        assert rank_positions([0, 1, 2], gains, 0.10) == [1, 2, 0]


@pytest.mark.parametrize("margin, variance, expected", [
    (0.03, 0.10, (0.94, 0.55, 0.11)),
    (0.05, 0.25, (0.95, 0.56, 0.13)),
])
def test_confirmation_simulation(margin, variance, expected):
    shares = [_confirmed_share(margin, variance, loss, seed=index)
              for index, loss in enumerate((0.0, margin / 2, margin))]
    for share, reference in zip(shares, expected):
        assert share == pytest.approx(reference, abs=0.015)


@pytest.mark.parametrize("margin, pools, expected", [
    (0.03, (65, 221, 1000, 5000), (0.000, 0.037, 0.052, 0.051)),
    (0.05, (65, 221, 1000, 5000), (0.036, 0.033, 0.043, 0.047)),
])
def test_the_sealed_test_on_the_margin_with_losses_only(margin, pools, expected):
    rates = [_sealed_pass_rate(margin, n_sealed) for n_sealed in pools]
    assert rates == pytest.approx(expected, abs=0.0015)


def test_the_sealed_test_stays_near_its_level_over_margins_and_pools():
    rates = [_sealed_pass_rate(margin, n_sealed) for margin in (0.02, 0.03, 0.05, 0.08)
             for n_sealed in (65, 221, 1000, 5000)]
    assert max(rates) <= 0.06


def test_isotonic_regression_matches_scikit_learn():
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(3)
    for size in (1, 2, 7, 40):
        values = rng.normal(size=size)
        weights = rng.uniform(0.1, 3.0, size=size)
        reference = IsotonicRegression().fit_transform(np.arange(size), values, sample_weight=weights)
        assert np.allclose(isotonic_increasing(values, weights), reference, atol=1e-12)


@pytest.mark.parametrize("changes", ["both ways", "losses only"])
def test_fixed_order_certification_of_finalists_on_the_margin(changes):
    rng = np.random.default_rng(11)
    runs, margin = 4000, 0.03
    certified_any = 0
    for start in range(0, runs, 200):
        if changes == "both ways":
            diffs = _diffs(rng, 200 * 5, SEALED, margin)
        else:
            diffs = _losses(rng, 200 * 5, SEALED, margin)
        tables = [Paired.from_diffs(row) for row in diffs]
        for run in range(200):
            finalists = [[tables[run * 5 + index]] for index in range(5)]
            certified_any += certify_fixed_order(finalists, [margin], ALPHA) > 0
    assert certified_any / runs <= 0.065
