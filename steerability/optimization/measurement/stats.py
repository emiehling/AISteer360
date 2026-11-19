"""Statistics of paired differences between a candidate and the unsteered baseline.

Every estimate in a study is a table of per-item differences `d = candidate score - baseline score`, held as its
sufficient statistics (`Paired`). A loss on a capability benchmark is `-paired.mean` and a gain on the behavior
benchmark is `paired.mean`. The functions here are pure. `alpha` and `power` have no defaults, since every caller
passes them from the study's `Settings`.

The decision rules are the following:

- **Certification (`Paired.passes`)**: the smoothed one-sided lower bound of the difference is above `-margin`.
- **Guard band (`guard_band`, `release_band`)**: how far inside the margin a search estimate must sit so that the
  sealed test passes with probability `power`, covering the noise of both the search estimate and the sealed test.
- **Confirmation (`confirm_step`)**: one look of a sequential check that accepts a point whose loss is inside the
  margin by the band (`feasible`) and rejects a point whose loss is above the margin (`clearly_over`).
- **Ladder decisions (`provisional_positions`, `rank_positions`)**: which positions of a solved axis are worth
  confirming, and in which order.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np

_Z = NormalDist().inv_cdf

CONFIRM_LOOKS: dict[float, tuple[int, int, int]] = {
    0.05: (64, 192, 400),
    0.03: (128, 512, 1100),
    0.02: (128, 1024, 2400),
}


@dataclass(frozen=True, slots=True)
class Paired:
    """The sufficient statistics of per-item differences `d = candidate score - baseline score`.

    `as_list()` is the stored (JSON) form, and `+` pools two tables of disjoint items.

    Attributes:
        n: The number of items.
        total: The sum of the differences.
        total_sq: The sum of the squared differences.
        changed: The number of items whose difference is not zero.
    """

    n: int
    total: float
    total_sq: float
    changed: int

    @classmethod
    def from_diffs(cls, diffs: Sequence[float]) -> Paired:
        """The table of a sequence of differences."""
        values = np.asarray(diffs, dtype=float)
        return cls(int(values.size), float(values.sum()), float((values * values).sum()), int(np.count_nonzero(values)))

    @classmethod
    def from_list(cls, values: Sequence[float]) -> Paired:
        """The table of its stored form `[n, total, total_sq, changed]`."""
        n, total, total_sq, changed = values
        return cls(int(n), float(total), float(total_sq), int(changed))

    def as_list(self) -> list[float]:
        """The stored form `[n, total, total_sq, changed]`."""
        return [self.n, self.total, self.total_sq, self.changed]

    def __add__(self, other: Paired) -> Paired:
        return Paired(self.n + other.n, self.total + other.total, self.total_sq + other.total_sq,
                      self.changed + other.changed)

    @property
    def mean(self) -> float:
        """The mean difference (0.0 for an empty table)."""
        return self.total / self.n if self.n else 0.0

    @property
    def change_rate(self) -> float:
        """The share of items whose difference is not zero (0.0 for an empty table)."""
        return self.changed / self.n if self.n else 0.0

    def se(self, *, smooth: bool = False) -> float:
        """The standard error of the mean difference.

        The smoothed form adds two items to the count and one to the sum of squares. A table with no changed items
        then still has a positive standard error.

        Args:
            smooth: Whether to smooth.

        Returns:
            The standard error, or infinity for an empty table.
        """
        n, total_sq = (self.n + 2, self.total_sq + 1.0) if smooth else (self.n, self.total_sq)
        if n == 0:
            return math.inf
        mean = self.total / n
        return math.sqrt(max(total_sq / n - mean * mean, 0.0) / n)

    def lower(self, alpha: float, *, smooth: bool = False) -> float:
        """The one-sided lower confidence bound of the mean difference at level `alpha`.

        Args:
            alpha: The error rate of the bound.
            smooth: Whether to use the smoothed mean and standard error (two added items).

        Returns:
            The bound, or minus infinity for an empty table.
        """
        n = self.n + 2 if smooth else self.n
        if n == 0:
            return -math.inf
        return self.total / n - _Z(1 - alpha) * self.se(smooth=smooth)

    def upper(self, alpha: float) -> float:
        """The one-sided upper confidence bound of the mean difference at level `alpha` (infinity when empty)."""
        if self.n == 0:
            return math.inf
        return self.mean + _Z(1 - alpha) * self.se()

    def passes(self, margin: float, alpha: float) -> bool:
        """The certification test, i.e., whether the smoothed lower bound of the difference is above `-margin`.

        Args:
            margin: The allowed loss, as an absolute difference of scores.
            alpha: The error rate of the test.

        Returns:
            Whether the table passes.
        """
        return self.lower(alpha, smooth=True) > -margin


def confirm_looks(margin: float) -> tuple[int, int, int]:
    """The cumulative item counts of the three looks of a confirmation at a margin.

    Margins of 0.05, 0.03, and 0.02 use the tabulated counts in `CONFIRM_LOOKS`. Any other margin uses a last look
    of `0.96 / margin**2` items rounded up to a hundred, a middle look at the largest power of two below 47% of it
    (at least 64), and a first look of a quarter of the middle one, between 32 and 128.

    Args:
        margin: The margin.

    Returns:
        The three counts, in increasing order.
    """
    key = round(margin, 4)
    if key in CONFIRM_LOOKS:
        return CONFIRM_LOOKS[key]
    last = int(math.ceil(0.96 / margin ** 2 / 100.0)) * 100
    middle = max(64, 2 ** int(math.floor(math.log2(0.47 * last))))
    first = max(32, min(128, middle // 4))
    return first, middle, last


def guard_band(se_search: float, se_sealed: float, *, alpha: float, power: float, benchmarks: int = 1) -> float:
    """How far inside the margin a search estimate must sit to pass the sealed test with the given power.

    A point whose search estimate sits exactly at `margin - band` passes the sealed test with probability `power`.
    With several constraint benchmarks, each one is given the power `power ** (1 / benchmarks)`.

    Args:
        se_search: The standard error of the search estimate.
        se_sealed: The standard error of the sealed test.
        alpha: The error rate of the sealed test.
        power: The probability that the point passes the sealed test.
        benchmarks: The number of constraint benchmarks the point must pass.

    Returns:
        The band, as an absolute difference of scores.
    """
    per_benchmark = power ** (1.0 / benchmarks)
    return _Z(1 - alpha) * se_sealed + _Z(per_benchmark) * math.hypot(se_search, se_sealed)


def release_band(
    paired: Paired,
    *,
    n_sealed: int,
    sd_floor: float,
    alpha: float,
    power: float,
    benchmarks: int = 1,
    n_search: int | None = None,
    guard: bool = True,
) -> float:
    """The guard band of a table.

    The per-item standard deviation is the table's own, floored at `sd_floor`. The search estimate's standard error
    uses the larger of the table's size and `n_search`, which plans for a later confirmation of that size.

    Args:
        paired: The table.
        n_sealed: The number of sealed items the point is certified on.
        sd_floor: The lower limit of the per-item standard deviation.
        alpha: The error rate of the sealed test.
        power: The probability that the point passes the sealed test.
        benchmarks: The number of constraint benchmarks.
        n_search: The size of a planned confirmation, or None.
        guard: Whether the band applies; False gives a band of 0.

    Returns:
        The band.
    """
    if not guard:
        return 0.0
    per_item_sd = max(paired.se() * math.sqrt(paired.n), sd_floor)
    n = max(paired.n, n_search or 0)
    return guard_band(per_item_sd / math.sqrt(n), per_item_sd / math.sqrt(n_sealed), alpha=alpha, power=power,
                      benchmarks=benchmarks)


def feasible(
    paired: Paired,
    margin: float,
    *,
    n_sealed: int,
    sd_floor: float,
    alpha: float,
    power: float,
    benchmarks: int = 1,
    guard: bool = True,
) -> bool:
    """Whether the estimated loss of a table is inside the margin by at least its guard band.

    Args:
        paired: The table.
        margin: The margin.
        n_sealed: The number of sealed items.
        sd_floor: The lower limit of the per-item standard deviation.
        alpha: The error rate of the sealed test.
        power: The probability that the point passes the sealed test.
        benchmarks: The number of constraint benchmarks.
        guard: Whether the band applies.

    Returns:
        Whether `-paired.mean <= margin - band`.
    """
    band = release_band(paired, n_sealed=n_sealed, sd_floor=sd_floor, alpha=alpha, power=power, benchmarks=benchmarks,
                        guard=guard)
    return -paired.mean <= margin - band


def clearly_over(paired: Paired, margin: float, alpha: float) -> bool:
    """Whether the loss of a table exceeds the margin, i.e., the upper bound of the difference is below `-margin`."""
    return paired.upper(alpha) < -margin


def confirm_step(
    paired: Paired,
    look: int,
    margin: float,
    *,
    n_sealed: int,
    sd_floor: float,
    alpha: float,
    power: float,
    benchmarks: int = 1,
    guard: bool = True,
) -> Literal["reject", "accept", "continue"]:
    """One look of the confirmation of a point on one constraint.

    A table that is `clearly_over` the margin is rejected at any look. From the second look (`look >= 1`) on, a
    `feasible` table is accepted. `"continue"` after the last look means the point is not confirmed.

    Args:
        paired: The point's table after this look.
        look: The index of the look (0, 1, or 2).
        margin: The margin.
        n_sealed: The number of sealed items.
        sd_floor: The lower limit of the per-item standard deviation.
        alpha: The error rate.
        power: The probability that an accepted point passes the sealed test.
        benchmarks: The number of constraint benchmarks.
        guard: Whether the band applies.

    Returns:
        `"reject"`, `"accept"`, or `"continue"`.
    """
    if clearly_over(paired, margin, alpha):
        return "reject"
    if look >= 1 and feasible(paired, margin, n_sealed=n_sealed, sd_floor=sd_floor, alpha=alpha, power=power,
                              benchmarks=benchmarks, guard=guard):
        return "accept"
    return "continue"


def certify_fixed_order(finalists: Sequence[Sequence[Paired]], margins: Sequence[float], alpha: float) -> int:
    """The number of leading finalists that pass the certification test on every constraint.

    Testing stops at the first finalist that fails.

    Args:
        finalists: Per finalist, in test order, one sealed table per constraint.
        margins: The margin of each constraint.
        alpha: The error rate of each test.

    Returns:
        The number of certified finalists.
    """
    for index, tables in enumerate(finalists):
        if not all(table.passes(margin, alpha) for table, margin in zip(tables, margins, strict=True)):
            return index
    return len(finalists)


def isotonic_increasing(values: Sequence[float], weights: Sequence[float]) -> np.ndarray:
    """The weighted least-squares fit to `values` that never decreases (pool adjacent violators).

    Args:
        values: The values, in order.
        weights: Their positive weights.

    Returns:
        The fitted values.

    Raises:
        ValueError: If a weight is not positive.
    """
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights, strict=True):
        if weight <= 0:
            raise ValueError("isotonic_increasing requires positive weights.")
        blocks.append([float(value), float(weight), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            value_right, weight_right, count_right = blocks.pop()
            value_left, weight_left, count_left = blocks.pop()
            pooled = (value_left * weight_left + value_right * weight_right) / (weight_left + weight_right)
            blocks.append([pooled, weight_left + weight_right, count_left + count_right])
    fitted: list[float] = []
    for value, _, count in blocks:
        fitted.extend([value] * int(count))
    return np.asarray(fitted)


def last_feasible(
    losses: Sequence[Paired],
    margin: float,
    *,
    n_target: int,
    n_sealed: int,
    sd_floor: float,
    alpha: float,
    power: float,
    benchmarks: int = 1,
    guard: bool = True,
) -> int | None:
    """The index of the last position whose isotonic loss is inside the margin by its band.

    The losses (`-paired.mean`) are first fitted with `isotonic_increasing`, weighted by item counts. Positions are
    scanned from the first, and the scan stops at the first position outside the margin by its band.

    Args:
        losses: One capability table per position, in order.
        margin: The margin.
        n_target: The size of the planned confirmation.
        n_sealed: The number of sealed items.
        sd_floor: The lower limit of the per-item standard deviation.
        alpha: The error rate.
        power: The power of the band.
        benchmarks: The number of constraint benchmarks.
        guard: Whether the band applies.

    Returns:
        The index, or None when the first position is outside.
    """
    fitted = isotonic_increasing([-paired.mean for paired in losses], [paired.n for paired in losses])
    best = None
    for index, paired in enumerate(losses):
        band = release_band(paired, n_sealed=n_sealed, sd_floor=sd_floor, alpha=alpha, power=power,
                            benchmarks=benchmarks, n_search=n_target, guard=guard)
        if fitted[index] > margin - band:
            break
        best = index
    return best


def provisional_positions(
    losses: Sequence[Paired],
    margin: float,
    *,
    n_target: int,
    n_sealed: int,
    sd_floor: float,
    monotone: bool,
    alpha: float,
    power: float,
    benchmarks: int = 1,
    guard: bool = True,
) -> list[int]:
    """The ladder positions worth confirming on one constraint, judged on ladder-block tables.

    With `monotone=True` the positions up to `last_feasible` are kept. Otherwise each position is kept when its own
    loss is inside the margin by its band.

    Args:
        losses: One capability table per position, in order.
        margin: The margin.
        n_target: The size of the planned confirmation.
        n_sealed: The number of sealed items.
        sd_floor: The lower limit of the per-item standard deviation.
        monotone: Whether the loss is assumed not to fall as the position rises.
        alpha: The error rate.
        power: The power of the band.
        benchmarks: The number of constraint benchmarks.
        guard: Whether the band applies.

    Returns:
        The kept position indices, in increasing order.
    """
    if monotone:
        last = last_feasible(losses, margin, n_target=n_target, n_sealed=n_sealed, sd_floor=sd_floor, alpha=alpha,
                             power=power, benchmarks=benchmarks, guard=guard)
        return [] if last is None else list(range(last + 1))
    kept = []
    for index, paired in enumerate(losses):
        band = release_band(paired, n_sealed=n_sealed, sd_floor=sd_floor, alpha=alpha, power=power,
                            benchmarks=benchmarks, n_search=n_target, guard=guard)
        if -paired.mean <= margin - band:
            kept.append(index)
    return kept


def rank_positions(positions: Sequence[int], gains: Sequence[Paired], alpha: float) -> list[int]:
    """Positions ordered by the lower bound of their behavior gain, best first.

    Args:
        positions: The position indices to order.
        gains: One behavior table per position (indexed by position).
        alpha: The error rate of the lower bound.

    Returns:
        The ordered indices.
    """
    return sorted(positions, key=lambda index: gains[index].lower(alpha), reverse=True)


__all__ = [
    "CONFIRM_LOOKS",
    "Paired",
    "certify_fixed_order",
    "clearly_over",
    "confirm_looks",
    "confirm_step",
    "feasible",
    "guard_band",
    "isotonic_increasing",
    "last_feasible",
    "provisional_positions",
    "rank_positions",
    "release_band",
]
