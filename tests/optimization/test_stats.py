"""Tests for the statistics of paired differences (`optimization/measurement/stats.py`)."""
import math

import numpy as np
import pytest

from steerability.optimization.measurement.stats import (
    Paired,
    certify_fixed_order,
    confirm_looks,
    confirm_step,
    feasible,
    guard_band,
    isotonic_increasing,
    provisional_positions,
    rank_positions,
    release_band,
)

ALPHA = 0.05
POWER = 0.80
CHANGE_RATE = 0.10
SEALED = 5000


def _diffs(rng, size, length, loss):
    """Per-item differences in {-1, 0, 1} with the given change rate and mean `-loss`."""
    draws = rng.random((size, length))
    down = (CHANGE_RATE + loss) / 2
    up = (CHANGE_RATE - loss) / 2
    return np.where(draws < down, -1.0, np.where(draws < down + up, 1.0, 0.0))


def _confirmed_share(margin, loss, *, runs=20000, seed=0):
    looks = confirm_looks(margin)
    rng = np.random.default_rng(seed)
    confirmed = 0
    for start in range(0, runs, 2000):
        size = min(2000, runs - start)
        diffs = _diffs(rng, size, looks[-1], loss)
        totals = np.cumsum(diffs, axis=1)
        changed = np.cumsum(diffs != 0, axis=1)
        for run in range(size):
            for look, count in enumerate(looks):
                table = Paired(count, float(totals[run, count - 1]), float(changed[run, count - 1]),
                               int(changed[run, count - 1]))
                decision = confirm_step(table, look, margin, n_sealed=SEALED, sd_floor=math.sqrt(0.02), alpha=ALPHA,
                                        power=POWER)
                if decision != "continue":
                    confirmed += decision == "accept"
                    break
    return confirmed / runs


class TestReferenceValues:
    def test_guard_band(self):
        se_sealed = math.sqrt(0.1 / 5000)
        assert guard_band(math.sqrt(0.1 / 2400), se_sealed, alpha=ALPHA, power=POWER) == pytest.approx(0.0140, abs=5e-5)
        assert guard_band(math.sqrt(0.1 / 1100), se_sealed, alpha=ALPHA, power=POWER) == pytest.approx(0.0162, abs=5e-5)
        assert guard_band(math.sqrt(0.1 / 1100), se_sealed, alpha=ALPHA, power=POWER,
                          benchmarks=3) == pytest.approx(0.0228, abs=5e-5)

    def test_confirm_looks(self):
        assert confirm_looks(0.04) == (64, 256, 600)
        assert confirm_looks(0.03) == (128, 512, 1100)
        assert confirm_looks(0.0300004) == (128, 512, 1100)

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

    def test_smoothing_gives_a_table_without_changes_a_positive_standard_error(self):
        table = Paired.from_diffs([0.0] * 100)
        assert table.se() == 0.0
        assert table.se(smooth=True) > 0.0
        assert table.passes(0.03, ALPHA)
        assert not Paired.from_diffs([-1.0] * 10 + [0.0] * 90).passes(0.03, ALPHA)

    def test_empty_tables(self):
        empty = Paired(0, 0.0, 0.0, 0)
        assert empty.mean == 0.0 and empty.change_rate == 0.0
        assert empty.se() == math.inf and empty.lower(ALPHA) == -math.inf and empty.upper(ALPHA) == math.inf


class TestBands:
    def test_no_guard_gives_a_zero_band(self):
        table = Paired.from_diffs([-1.0] * 5 + [0.0] * 95)
        assert release_band(table, n_sealed=SEALED, sd_floor=0.2, alpha=ALPHA, power=POWER, guard=False) == 0.0
        assert feasible(table, 0.05, n_sealed=SEALED, sd_floor=0.2, alpha=ALPHA, power=POWER, guard=False)
        assert not feasible(table, 0.05, n_sealed=SEALED, sd_floor=0.2, alpha=ALPHA, power=POWER)

    def test_the_band_plans_for_a_larger_confirmation(self):
        table = Paired.from_diffs([-1.0] * 5 + [0.0] * 95)
        small = release_band(table, n_sealed=SEALED, sd_floor=0.2, alpha=ALPHA, power=POWER)
        planned = release_band(table, n_sealed=SEALED, sd_floor=0.2, alpha=ALPHA, power=POWER, n_search=1000)
        assert planned < small

    def test_look_zero_only_rejects(self):
        inside = Paired.from_diffs([0.0] * 400)
        assert confirm_step(inside, 0, 0.03, n_sealed=SEALED, sd_floor=0.1, alpha=ALPHA, power=POWER) == "continue"
        assert confirm_step(inside, 1, 0.03, n_sealed=SEALED, sd_floor=0.1, alpha=ALPHA, power=POWER) == "accept"
        over = Paired.from_diffs([-1.0] * 40 + [0.0] * 60)
        assert confirm_step(over, 0, 0.03, n_sealed=SEALED, sd_floor=0.1, alpha=ALPHA, power=POWER) == "reject"


class TestLadderDecisions:
    def test_monotone_positions_stop_at_the_first_outside(self):
        losses = [Paired.from_diffs([-x] * 256) for x in (0.0, 0.01, 0.02, 0.09, 0.03)]
        kept = provisional_positions(losses, 0.05, n_target=400, n_sealed=SEALED, sd_floor=0.1, monotone=True,
                                     alpha=ALPHA, power=0.5)
        assert kept == [0, 1, 2]
        each = provisional_positions(losses, 0.05, n_target=400, n_sealed=SEALED, sd_floor=0.1, monotone=False,
                                     alpha=ALPHA, power=0.5)
        assert each == [0, 1, 2, 4]

    def test_ranking_orders_by_the_gain_lower_bound(self):
        gains = [Paired.from_diffs([value] * 64) for value in (0.1, 0.3, 0.2)]
        assert rank_positions([0, 1, 2], gains, 0.10) == [1, 2, 0]


@pytest.mark.parametrize("margin, expected", [
    (0.03, (0.94, 0.55, 0.11)),
    (0.02, (0.86, 0.38, 0.06)),
])
def test_confirmation_simulation(margin, expected):
    shares = [_confirmed_share(margin, loss, seed=index) for index, loss in enumerate((0.0, margin / 2, margin))]
    for share, reference in zip(shares, expected):
        assert share == pytest.approx(reference, abs=0.015)


def test_isotonic_regression_matches_scikit_learn():
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(3)
    for size in (1, 2, 7, 40):
        values = rng.normal(size=size)
        weights = rng.uniform(0.1, 3.0, size=size)
        reference = IsotonicRegression().fit_transform(np.arange(size), values, sample_weight=weights)
        assert np.allclose(isotonic_increasing(values, weights), reference, atol=1e-12)


def test_fixed_order_certification_of_finalists_on_the_margin():
    rng = np.random.default_rng(11)
    runs, margin = 4000, 0.03
    certified_any = 0
    for start in range(0, runs, 200):
        diffs = _diffs(rng, 200 * 5, SEALED, margin)
        tables = [Paired.from_diffs(row) for row in diffs]
        for run in range(200):
            finalists = [[tables[run * 5 + index]] for index in range(5)]
            certified_any += certify_fixed_order(finalists, [margin], ALPHA) > 0
    assert certified_any / runs <= 0.065
