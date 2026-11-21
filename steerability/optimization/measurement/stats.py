"""Statistics of paired differences between a candidate and the unsteered baseline.

Every estimate in a study is a table of per-item differences `d = candidate score - baseline score`, stored as its
sufficient statistics (`Paired`). A loss on a capability benchmark is `-paired.mean` and a gain on the behavior
benchmark is `paired.mean`. The functions here are pure. `alpha` and `power` have no defaults, since every caller
passes them from the study's `Settings`.

The per-item variance of a capability table is taken as at least `floor_variance(margin)`, the variance of a 0/1
score whose only changes are losses that add up to the margin. A table built from few changed items would otherwise
understate its variance. The decision rules are the following:

- **Certification (`Paired.passes`)**: the one-sided lower bound of the difference, at the floor variance, is above
  `-margin`.
- **Guard band (`guard_band`, `release_band`)**: how far inside the margin a search estimate must sit so that the
  sealed test passes with probability `power`, covering the noise of both the search estimate and the sealed test.
- **Confirmation (`confirm_step`)**: one look of a sequential check that accepts a point whose loss is inside the
  margin by the band (`feasible`) and rejects a point whose loss is above the margin (`clearly_over`).
- **Ladder decisions (`provisional_positions`, `rank_positions`)**: which positions of a solved axis outside
  `study.monotone` are worth confirming, and the order in which the positions of a solved axis are confirmed.

The sizes of a confirmation follow from the margin and the per-item variance a study is designed for. The last look
(`last_look`) gives the estimate a standard error of `margin / sqrt(LOOK_PRECISION)`, and the earlier looks
(`look_sizes`) and the ladder block (`ladder_size`) are set from it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np

_Z = NormalDist().inv_cdf

LOOK_PRECISION = 9.6  # the last look has a standard error of margin / sqrt(LOOK_PRECISION)
LOOK_STEP = 100  # last looks are whole multiples of this many items
BAND_SHARE = 0.6  # the largest share of a margin that the guard band of a feasible study takes


def floor_variance(margin: float) -> float:
    """The smallest per-item variance the statistics assume at a margin, `margin * (1 - margin)`.

    It is the variance of a 0/1 score whose only changes are losses that add up to the margin, which is the least
    favorable case for a test on few items. A table that shows less is treated as having this much.

    Args:
        margin: The margin.

    Returns:
        The variance.
    """
    return margin * (1.0 - margin)


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
    def variance(self) -> float:
        """The per-item (population) variance of the differences (0.0 for an empty table)."""
        if self.n == 0:
            return 0.0
        mean = self.total / self.n
        return max(self.total_sq / self.n - mean * mean, 0.0)

    @property
    def change_rate(self) -> float:
        """The share of items whose difference is not zero (0.0 for an empty table)."""
        return self.changed / self.n if self.n else 0.0

    def se(self, *, floor: float = 0.0) -> float:
        """The standard error of the mean difference, with the per-item variance taken as at least `floor`.

        Args:
            floor: The smallest per-item variance.

        Returns:
            The standard error, or infinity for an empty table.
        """
        if self.n == 0:
            return math.inf
        return math.sqrt(max(self.variance, floor) / self.n)

    def lower(self, alpha: float, *, floor: float = 0.0) -> float:
        """The one-sided lower confidence bound of the mean difference at level `alpha`.

        Args:
            alpha: The error rate of the bound.
            floor: The smallest per-item variance of its standard error.

        Returns:
            The bound, or minus infinity for an empty table.
        """
        if self.n == 0:
            return -math.inf
        return self.mean - _Z(1 - alpha) * self.se(floor=floor)

    def upper(self, alpha: float) -> float:
        """The one-sided upper confidence bound of the mean difference at level `alpha` (infinity when empty)."""
        if self.n == 0:
            return math.inf
        return self.mean + _Z(1 - alpha) * self.se()

    def passes(self, margin: float, alpha: float) -> bool:
        """The certification test, i.e., whether the lower bound at `floor_variance(margin)` is above `-margin`.

        Args:
            margin: The allowed loss, as an absolute difference of scores.
            alpha: The error rate of the test.

        Returns:
            Whether the table passes.
        """
        return self.lower(alpha, floor=floor_variance(margin)) > -margin


def last_look(*, margin: float, variance: float) -> int:
    """The number of items a confirmation needs at a margin and per-item variance.

    The count is `LOOK_PRECISION * variance / margin**2` rounded up to a whole multiple of `LOOK_STEP`. Both arguments
    are keyword-only, since they are small floats of similar size.

    Args:
        margin: The margin.
        variance: The per-item variance of the differences.

    Returns:
        The size of the last look.
    """
    return int(math.ceil(round(LOOK_PRECISION * variance / margin ** 2 / LOOK_STEP, 9))) * LOOK_STEP


def look_sizes(last: int) -> tuple[int, int, int]:
    """The cumulative item counts of the three looks of a confirmation whose last look has `last` items.

    The middle look is the largest power of two below 47% of the last one (at least 64), and the first look is a
    quarter of the middle one, between 32 and 128. Neither is larger than the look after it.

    Args:
        last: The size of the last look.

    Returns:
        The three counts, in increasing order.

    Raises:
        ValueError: If `last` is smaller than 1.
    """
    if last < 1:
        raise ValueError(f"A confirmation needs at least one item; got {last}.")
    middle = min(last, max(64, 2 ** int(math.floor(math.log2(0.47 * last)))))
    first = min(middle, max(32, min(128, middle // 4)))
    return first, middle, last


def ladder_size(last: int) -> int:
    """The capability items per ladder position, a quarter of the last look and at least 32."""
    return max(32, last // 4)


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
    margin: float,
    *,
    n_sealed: int,
    alpha: float,
    power: float,
    benchmarks: int = 1,
    n_search: int | None = None,
    guard: bool = True,
) -> float:
    """The guard band of a table.

    The per-item variance is the table's own, and at least `floor_variance(margin)`. The search estimate's standard
    error uses the larger of the table's size and `n_search`, which plans for a later confirmation of that size.

    Args:
        paired: The table.
        margin: The margin.
        n_sealed: The number of sealed items the point is certified on.
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
    per_item_sd = math.sqrt(max(paired.variance, floor_variance(margin)))
    n = max(paired.n, n_search or 0)
    return guard_band(per_item_sd / math.sqrt(n), per_item_sd / math.sqrt(n_sealed), alpha=alpha, power=power,
                      benchmarks=benchmarks)


def feasible(
    paired: Paired,
    margin: float,
    *,
    n_sealed: int,
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
        alpha: The error rate of the sealed test.
        power: The probability that the point passes the sealed test.
        benchmarks: The number of constraint benchmarks.
        guard: Whether the band applies.

    Returns:
        Whether `-paired.mean <= margin - band`.
    """
    band = release_band(paired, margin, n_sealed=n_sealed, alpha=alpha, power=power, benchmarks=benchmarks,
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
        alpha: The error rate.
        power: The probability that an accepted point passes the sealed test.
        benchmarks: The number of constraint benchmarks.
        guard: Whether the band applies.

    Returns:
        `"reject"`, `"accept"`, or `"continue"`.
    """
    if clearly_over(paired, margin, alpha):
        return "reject"
    if look >= 1 and feasible(paired, margin, n_sealed=n_sealed, alpha=alpha, power=power, benchmarks=benchmarks,
                              guard=guard):
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


def provisional_positions(
    losses: Sequence[Paired],
    margin: float,
    *,
    n_target: int,
    n_sealed: int,
    alpha: float,
    power: float,
    benchmarks: int = 1,
    guard: bool = True,
) -> list[int]:
    """The ladder positions worth confirming on one constraint, judged on ladder-block tables.

    Each position is kept when its own loss is inside the margin by its band.

    Args:
        losses: One capability table per position, in order.
        margin: The margin.
        n_target: The size of the planned confirmation.
        n_sealed: The number of sealed items.
        alpha: The error rate.
        power: The power of the band.
        benchmarks: The number of constraint benchmarks.
        guard: Whether the band applies.

    Returns:
        The kept position indices, in increasing order.
    """
    kept = []
    for index, paired in enumerate(losses):
        band = release_band(paired, margin, n_sealed=n_sealed, alpha=alpha, power=power, benchmarks=benchmarks,
                            n_search=n_target, guard=guard)
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
    "BAND_SHARE",
    "LOOK_PRECISION",
    "LOOK_STEP",
    "Paired",
    "certify_fixed_order",
    "clearly_over",
    "confirm_step",
    "feasible",
    "floor_variance",
    "guard_band",
    "ladder_size",
    "last_look",
    "look_sizes",
    "provisional_positions",
    "rank_positions",
    "release_band",
]
