"""The evidence a study's decisions are made on, and the sizes and feasibility that follow from its baseline.

The unsteered baseline (`BASELINE`) is measured once per study on every search and sealed item, and a second time on
the search items in another batch order (`ensure_baseline`). The mean absolute difference between the two runs is
the benchmark's noise floor (`noise_floor`). Under greedy decoding a noise floor above `GREEDY_NOISE_LIMIT` stops the
study, since an evidence row is keyed by recipe and item and describes the recipe only when generation does not
depend on the rest of the batch. An item whose baseline row is an error row is excluded from its pool for the whole
study (`usable`), since an errored baseline scores 0 and would make every candidate look better on that item.

Every estimate is a table of paired differences against the baseline (`paired`). The sizes of a constraint's blocks
follow from the variance of the baseline's scores (`design_variance`, `block_sizes`), and `feasibility` checks that
the pools are large enough for them. Items are drawn into batches per round (`batch`), each with a ladder block (on
which choices are made) and a fresh block (on which points are confirmed and reported). A point's tables use
fresh-block items only (`point_items`).
"""
from __future__ import annotations

import json
import logging
import math
import random
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from steerability.algorithms.core.identity import derive_seed
from steerability.algorithms.core.recipe import BASELINE
from steerability.optimization.measurement.stats import (
    BAND_SHARE,
    Paired,
    floor_variance,
    guard_band,
    ladder_size,
    last_look,
)
from steerability.optimization.study import StudyError, dumps, ensure_tables

if TYPE_CHECKING:
    from steerability.algorithms.core.store import EvidenceRow, Store
    from steerability.algorithms.core.sweeps import PipelineFactory, Steered
    from steerability.evaluation.items import Benchmark, ItemResult
    from steerability.optimization.study import Settings, Study

logger = logging.getLogger(__name__)

MAX_EXCLUDED_SHARE = 0.02
GREEDY_NOISE_LIMIT = 0.01
NOISE_PROBE_ITEMS = 400
SAMPLED_BASELINE_REPLICATES = 4
_MARGIN_STEP = 0.001
_VARIANCE_STEP = 0.001
_LARGEST_MARGIN = 0.5

_BASELINES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def sampled(study: Study, benchmark: Benchmark) -> bool:
    """Whether a benchmark samples under the study's protocol (temperature other than 0)."""
    merged = {**study.protocol.generate_defaults, **(getattr(benchmark, "generate", None) or {})}
    return merged.get("temperature") != 0


def _baseline_row(store: Store, study: Study, benchmark: Benchmark) -> tuple[str, float, list[str]] | None:
    ensure_tables(store)
    row = store.execute(
        "SELECT resolved_id, noise_floor, excluded FROM baselines WHERE study_id = ? AND benchmark_id = ?",
        (study.id, benchmark.id),
    ).fetchone()
    return None if row is None else (row[0], float(row[1]), json.loads(row[2]))


def _require_baseline(store: Store, study: Study, benchmark: Benchmark) -> tuple[str, float, list[str]]:
    row = _baseline_row(store, study, benchmark)
    if row is None:
        raise StudyError(f"Study {study.name!r} has no baseline for {benchmark.name!r}; run ensure_baseline first.")
    return row


def _check_unit_interval(benchmark: Benchmark, replicate: int, results: Mapping[str, ItemResult]) -> None:
    outside = [item for item, result in results.items() if result.status == "ok" and not 0.0 <= result.value <= 1.0]
    if outside:
        raise StudyError(
            f"{benchmark.name}: baseline values must be in [0, 1]; items {outside[:5]} at replicate {replicate} are "
            "outside."
        )


def _mean_absolute_difference(rows: Mapping[int, Mapping[str, ItemResult]], items: Sequence[str]) -> float:
    if not items:
        return 0.0
    return sum(abs(rows[0][item].value - rows[1][item].value) for item in items) / len(items)


def _check_noise(benchmark: Benchmark, floor: float, items: int) -> None:
    if floor > GREEDY_NOISE_LIMIT:
        raise StudyError(
            f"{benchmark.name}: the baseline's scores differ by {floor:.4f} on average between two runs in different "
            f"batch orders over {items:,} search items (for 0/1 scores, that share of items changed). The limit under "
            f"greedy decoding is {GREEDY_NOISE_LIMIT}. A study needs generation that does not depend on the batch, "
            "since stored rows are keyed by recipe and item. Load the model in float32 "
            "(`hf_model_kwargs={\"dtype\": \"float32\"}` on the Hugging Face backend, or `engine_kwargs={\"dtype\": "
            "\"float32\"}` in the options of a vLLM backend spec), or set `ProviderOptions(max_batch_size=1)`. A "
            "scorer that is not deterministic gives the same symptom."
        )


def _probe_noise(study: Study, store: Store, steered: Steered, benchmark: Benchmark) -> None:
    """Measure replicates 0 and 1 on the first search items and apply the greedy noise limit to them."""
    from steerability.evaluation.items import evaluate_items

    probe = list(study.pools[benchmark.id].search[:NOISE_PROBE_ITEMS])
    if not probe:
        return
    rows = {replicate: evaluate_items(steered, benchmark, probe, store=store, replicate=replicate)
            for replicate in (0, 1)}
    for replicate, results in rows.items():
        _check_unit_interval(benchmark, replicate, results)
    items = [item for item in probe if rows[0][item].status == "ok" and rows[1][item].status == "ok"]
    _check_noise(benchmark, _mean_absolute_difference(rows, items), len(items))


def _measure_baseline(study: Study, store: Store, steered: Steered, benchmark: Benchmark) -> None:
    """Measure the baseline of one benchmark, check it, and record its `baselines` row."""
    from steerability.evaluation.items import evaluate_items

    split = study.pools[benchmark.id]
    sampling = sampled(study, benchmark)
    replicates = tuple(range(SAMPLED_BASELINE_REPLICATES)) if sampling else (0, 1)
    rows: dict[int, dict[str, ItemResult]] = {}
    for replicate in replicates:
        rows[replicate] = {}
        pools = (split.search, split.sealed) if sampling or replicate == 0 else (split.search,)
        for pool in pools:
            if pool:
                rows[replicate].update(evaluate_items(steered, benchmark, list(pool), store=store,
                                                      replicate=replicate))
        _check_unit_interval(benchmark, replicate, rows[replicate])
    excluded = sorted({item for results in rows.values() for item, result in results.items()
                       if result.status == "error"})
    excluded_set = set(excluded)
    for role, pool in (("search", split.search), ("sealed", split.sealed)):
        if not pool:
            continue
        share = len(excluded_set.intersection(pool)) / len(pool)
        if share > MAX_EXCLUDED_SHARE:
            raise StudyError(
                f"{benchmark.name}: {100 * share:.1f}% of the {role} pool has baseline error rows (the limit is "
                f"{100 * MAX_EXCLUDED_SHARE:.0f}%); since error rows are evidence, rerun on a new store once the "
                "cause is fixed."
            )
    usable_search = [item for item in split.search if item not in excluded_set]
    floor = _mean_absolute_difference(rows, usable_search)
    if not sampling:
        _check_noise(benchmark, floor, len(usable_search))
    constraint = study.constraint_for(benchmark)
    if constraint is not None and usable_search:
        paired_replicates = replicates if sampling else (0,)
        mean = sum(
            sum(rows[replicate][item].value for replicate in paired_replicates) / len(paired_replicates)
            for item in usable_search
        ) / len(usable_search)
        if mean <= constraint.margin:
            raise StudyError(
                f"{benchmark.name}: the baseline's mean score on the usable search items is {mean:.4f}, at most the "
                f"margin of {constraint.margin}. No candidate can lose more than the margin, and the constraint "
                "cannot fail."
            )
    store.execute(
        """INSERT OR IGNORE INTO baselines (study_id, benchmark_id, resolved_id, noise_floor, excluded)
            VALUES (?, ?, ?, ?, ?)""",
        (study.id, benchmark.id, steered.resolved_id, floor, dumps(excluded)),
    )
    logger.info("Baseline of %s: noise floor %.4f, %d excluded item(s).", benchmark.name, floor, len(excluded))


def ensure_baseline(study: Study, factory: PipelineFactory, store: Store) -> None:
    """Measure the unsteered baseline of every benchmark of a study, once.

    Each benchmark is prepared, and `BASELINE` is evaluated at replicate 0 on the search pool and, in a separate
    call, on the sealed pool, then at replicate 1 on the search pool. The sealed pool is its own request set, which
    is the one a finalist's sealed call uses. Under a sampling protocol, replicates 0 to 3 are each evaluated on the
    two pools in separate calls. One `baselines` row per benchmark records the baseline's resolved id, the noise
    floor (the mean absolute difference between the values at replicates 0 and 1 over the usable search items), and
    the excluded items (those with an error row in any replicate). A benchmark that already has a row is skipped.

    Under greedy decoding, the first `NOISE_PROBE_ITEMS` search items of every pending benchmark are measured at
    replicates 0 and 1 before anything else, and the rest of the baseline reuses these rows. A noise floor above
    `GREEDY_NOISE_LIMIT`, on the probe items or on the whole search pool, raises `StudyError`. No argument waives
    this, and no `baselines` row is written.

    Args:
        study: The study.
        factory: The factory the baseline is steered with.
        store: The store.

    Raises:
        StudyError: If a baseline value is outside [0, 1], more than 2% of a pool is excluded, the noise floor of a
            greedy benchmark is above `GREEDY_NOISE_LIMIT`, or the baseline's mean score on a constraint is at most
            its margin.
    """
    ensure_tables(store)
    for benchmark in study.benchmarks:
        benchmark.prepare(store)
    pending = [benchmark for benchmark in study.benchmarks if _baseline_row(store, study, benchmark) is None]
    if not pending:
        return
    with factory.steered(BASELINE) as steered:
        for benchmark in pending:
            if not sampled(study, benchmark):
                _probe_noise(study, store, steered, benchmark)
        for benchmark in pending:
            _measure_baseline(study, store, steered, benchmark)


def usable(study: Study, store: Store, benchmark: Benchmark, role: Literal["search", "sealed"]) -> tuple[str, ...]:
    """A benchmark's search or sealed pool without the items excluded by the baseline.

    Raises:
        StudyError: If the benchmark has no baseline.
    """
    _, _, excluded = _require_baseline(store, study, benchmark)
    excluded_set = set(excluded)
    split = study.pools[benchmark.id]
    pool = split.search if role == "search" else split.sealed
    return tuple(item for item in pool if item not in excluded_set)


def noise_floor(study: Study, store: Store, benchmark: Benchmark) -> float:
    """The mean absolute difference between the baseline values at replicates 0 and 1 over the usable search items.

    For 0/1 scores it is the share of items whose score changed between the two runs.

    Raises:
        StudyError: If the benchmark has no baseline.
    """
    return _require_baseline(store, study, benchmark)[1]


def baseline_resolved_id(study: Study, store: Store, benchmark: Benchmark) -> str:
    """The resolved id the baseline of a benchmark was measured under."""
    return _require_baseline(store, study, benchmark)[0]


def completion_of(store: Store, row: EvidenceRow) -> str | None:
    """The completion of an evidence row (None on an error row)."""
    if row.status == "error":
        return None
    detail = row.detail or {}
    if detail.get("completion") is not None:
        return detail["completion"]
    if row.rollout_key is not None:
        rollout = store.get_rollout(row.rollout_key)
        return None if rollout is None else rollout.completion
    return None


def degenerate(completion: str | None) -> bool:
    """Whether a completion is degenerate, i.e., empty or with more than half of its word 4-grams repeated.

    A 4-gram counts as repeated at each occurrence after its first.
    """
    if completion is None or not completion.strip():
        return True
    words = completion.split()
    grams = [tuple(words[index:index + 4]) for index in range(len(words) - 3)]
    if not grams:
        return False
    return len(grams) - len(set(grams)) > 0.5 * len(grams)


@dataclass
class _BaselineTable:
    """The baseline values, tokens, and degenerate indicators of a benchmark, averaged over the paired replicates."""

    store: Store
    rows: dict[str, list[EvidenceRow]]
    _degenerate: dict[str, float] = field(default_factory=dict)

    def _rows(self, item_id: str) -> list[EvidenceRow]:
        rows = self.rows.get(item_id)
        if not rows:
            raise StudyError(f"Item {item_id!r} has no baseline row; it is not in the study's pools.")
        return rows

    def values(self, item_id: str) -> list[float]:
        return [row.value for row in self._rows(item_id)]

    def value(self, item_id: str) -> float:
        rows = self._rows(item_id)
        return sum(row.value for row in rows) / len(rows)

    def tokens(self, item_id: str) -> float | None:
        counts = [row.tokens for row in self._rows(item_id) if row.tokens is not None]
        return sum(counts) / len(counts) if counts else None

    def degenerate(self, item_id: str) -> float:
        if item_id not in self._degenerate:
            rows = self._rows(item_id)
            self._degenerate[item_id] = sum(degenerate(completion_of(self.store, row)) for row in rows) / len(rows)
        return self._degenerate[item_id]


def _baseline_table(study: Study, store: Store, benchmark: Benchmark) -> _BaselineTable:
    tables = _BASELINES.setdefault(store, {})
    key = (study.id, benchmark.id)
    if key not in tables:
        resolved, _, _ = _require_baseline(store, study, benchmark)
        replicates = range(SAMPLED_BASELINE_REPLICATES) if sampled(study, benchmark) else range(1)
        rows: dict[str, list[EvidenceRow]] = {}
        for row in store.evidence_rows(study.protocol.id, resolved, benchmark.id):
            if row.replicate in replicates:
                rows.setdefault(row.item_id, []).append(row)
        tables[key] = _BaselineTable(store, rows)
    return tables[key]


def baseline_variance(study: Study, store: Store, benchmark: Benchmark) -> float | None:
    """The population variance of a benchmark's baseline values over its usable search items.

    The values are pooled over the replicates that `paired` reads (replicate 0 under greedy decoding, replicates 0 to
    3 under sampling). For 0/1 scores with mean `p` it is `p * (1 - p)`.

    Returns:
        The variance, or None when the benchmark has no baseline.
    """
    if _baseline_row(store, study, benchmark) is None:
        return None
    table = _baseline_table(study, store, benchmark)
    values = [value for item_id in usable(study, store, benchmark, "search") for value in table.values(item_id)]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return max(sum(value * value for value in values) / len(values) - mean * mean, 0.0)


def design_variance(study: Study, store: Store, benchmark: Benchmark) -> float:
    """The per-item variance of paired differences that the study sizes a constraint for.

    It is the baseline's variance (`baseline_variance`) and at least `floor_variance(margin)`, or the floor alone
    when the benchmark has no baseline. It is used only to size the looks and to decide feasibility. Acceptance and
    the sealed test use each table's own variance (at least the floor). An inaccurate design variance therefore
    changes the power and the number of generations of a study, and it leaves the error rate of certification
    unchanged.

    Args:
        study: The study.
        store: The store.
        benchmark: A constraint benchmark of the study.

    Returns:
        The design variance.

    Raises:
        ValueError: If the benchmark is not a constraint of the study.
    """
    constraint = study.constraint_for(benchmark)
    if constraint is None:
        raise ValueError(f"Benchmark {benchmark.name!r} is not a constraint of study {study.name!r}.")
    measured = baseline_variance(study, store, benchmark)
    return max(measured or 0.0, floor_variance(constraint.margin))


def point_rows(study: Study, store: Store, resolved_id: str, benchmark: Benchmark,
               item_ids: Sequence[str] | None = None) -> dict[str, EvidenceRow]:
    """A resolved recipe's evidence rows on a benchmark at replicate 0, by item id (all when `item_ids` is None)."""
    return store.get_evidence(study.protocol.id, resolved_id, benchmark.id, item_ids, replicate=0)


def paired(study: Study, store: Store, resolved_id: str, benchmark: Benchmark, item_ids: Sequence[str]) -> Paired:
    """The table of differences between a resolved recipe and the baseline on items of a benchmark.

    Each difference is the recipe's value at replicate 0 minus the baseline's value at replicate 0 (or its mean over
    replicates 0 to 3 under a sampling protocol). A recipe row with `status="error"` counts with its value of 0.

    Args:
        study: The study.
        store: The store.
        resolved_id: The resolved id of the recipe.
        benchmark: The benchmark.
        item_ids: The items.

    Returns:
        The table.

    Raises:
        StudyError: If an item has no row for the recipe or no baseline row.
    """
    baseline = _baseline_table(study, store, benchmark)
    rows = point_rows(study, store, resolved_id, benchmark, item_ids)
    diffs = []
    for item_id in item_ids:
        row = rows.get(item_id)
        if row is None:
            raise StudyError(f"{benchmark.name}: item {item_id!r} has no evidence row for {resolved_id}.")
        diffs.append(row.value - baseline.value(item_id))
    return Paired.from_diffs(diffs)


def cost_ratio(study: Study, store: Store, resolved_id: str, items: Mapping[str, Sequence[str]]) -> float:
    """The ratio of a recipe's generated tokens per item to the baseline's, over the same items.

    Items whose row (or baseline row) has no token count are left out. The baseline total is floored at one token.

    Args:
        study: The study.
        store: The store.
        resolved_id: The resolved id of the recipe.
        items: Mapping from benchmark id to the items.

    Returns:
        The ratio.
    """
    candidate_total = baseline_total = 0.0
    by_id = {benchmark.id: benchmark for benchmark in study.benchmarks}
    for benchmark_id, item_ids in items.items():
        benchmark = by_id[benchmark_id]
        baseline = _baseline_table(study, store, benchmark)
        rows = point_rows(study, store, resolved_id, benchmark, item_ids)
        for item_id in item_ids:
            row = rows.get(item_id)
            base_tokens = baseline.tokens(item_id)
            if row is None or row.tokens is None or base_tokens is None:
                continue
            candidate_total += row.tokens
            baseline_total += base_tokens
    return candidate_total / max(baseline_total, 1.0)


def degenerate_increase(study: Study, store: Store, resolved_id: str, item_ids: Sequence[str]) -> float:
    """The degenerate rate of a recipe's objective completions minus the baseline's, over the same items.

    An error row counts as degenerate.
    """
    if not item_ids:
        return 0.0
    benchmark = study.objective
    baseline = _baseline_table(study, store, benchmark)
    rows = point_rows(study, store, resolved_id, benchmark, item_ids)
    candidate_rate = sum(
        degenerate(completion_of(store, rows[item_id])) if item_id in rows else True for item_id in item_ids
    ) / len(item_ids)
    baseline_rate = sum(baseline.degenerate(item_id) for item_id in item_ids) / len(item_ids)
    return candidate_rate - baseline_rate


@dataclass(frozen=True)
class Batch:
    """The items of one benchmark drawn for one round.

    Every smaller sample of a block is a prefix of it, and a later step reuses the earlier rows.

    Attributes:
        ladder: The items a choice may be made on.
        fresh: The items a point is confirmed and reported on.
    """

    ladder: tuple[str, ...]
    fresh: tuple[str, ...]

    @property
    def overlapping(self) -> bool:
        """Whether the two blocks share items, which happens only under the small-pool rule."""
        return bool(set(self.ladder) & set(self.fresh))

    def to_json(self) -> dict:
        return {"ladder": list(self.ladder), "fresh": list(self.fresh)}

    @classmethod
    def from_json(cls, data: Mapping) -> Batch:
        return cls(tuple(data["ladder"]), tuple(data["fresh"]))


def block_sizes(study: Study, store: Store, benchmark: Benchmark) -> tuple[int, int]:
    """The ladder and fresh block sizes of a benchmark of the study.

    A constraint's fresh block is the last look at its design variance, `last_look(margin=margin,
    variance=design_variance(...))`, and its ladder block is `ladder_size` of that. The objective uses
    `settings.behavior_ladder_items` and `settings.behavior_items`.

    Raises:
        ValueError: If the benchmark is not part of the study.
    """
    constraint = study.constraint_for(benchmark)
    if constraint is not None:
        last = last_look(margin=constraint.margin, variance=design_variance(study, store, benchmark))
        return ladder_size(last), last
    if benchmark.id == study.objective.id:
        return study.settings.behavior_ladder_items, study.settings.behavior_items
    raise ValueError(f"Benchmark {benchmark.name!r} is not part of study {study.name!r}.")


def batch(study: Study, store: Store, benchmark: Benchmark, round_index: int) -> Batch:
    """The items of a benchmark for a round, drawn from its usable search pool.

    The draw is `random.Random(derive_seed(study.seed, "batch", benchmark.id, round_index)).sample(sorted(pool),
    ladder size + fresh size)`; the first part is the ladder block and the rest is the fresh block. When the pool
    has fewer items than the two blocks together, each block is a prefix of one shuffled copy of the pool and a
    warning is logged. The blocks then overlap, and the study chooses and judges on the same items.

    Args:
        study: The study.
        store: The store.
        benchmark: The benchmark.
        round_index: The round.

    Returns:
        The batch.
    """
    pool = sorted(usable(study, store, benchmark, "search"))
    ladder, fresh = block_sizes(study, store, benchmark)
    rng = random.Random(derive_seed(study.seed, "batch", benchmark.id, round_index))
    if len(pool) >= ladder + fresh:
        drawn = rng.sample(pool, ladder + fresh)
        return Batch(tuple(drawn[:ladder]), tuple(drawn[ladder:]))
    logger.warning(
        "%s: the usable search pool (%d items) is smaller than the ladder and fresh blocks (%d + %d); both blocks are "
        "prefixes of one shuffled copy of the pool, and the study chooses and judges on the same items.",
        benchmark.name, len(pool), ladder, fresh,
    )
    shuffled = rng.sample(pool, len(pool))
    return Batch(tuple(shuffled[:ladder]), tuple(shuffled[:fresh]))


def round_batches(study: Study, store: Store, round_index: int) -> dict[str, Batch]:
    """The batches of every benchmark of the study for a round, by benchmark id."""
    return {benchmark.id: batch(study, store, benchmark, round_index) for benchmark in study.benchmarks}


def record_round(store: Store, study: Study, round_index: int, batches: Mapping[str, Batch]) -> None:
    """Record the batches of a round in the `rounds` table (once)."""
    ensure_tables(store)
    store.execute(
        "INSERT OR IGNORE INTO rounds (study_id, round_index, batches) VALUES (?, ?, ?)",
        (study.id, round_index, dumps({benchmark_id: item.to_json() for benchmark_id, item in batches.items()})),
    )


def load_rounds(store: Store, study: Study) -> dict[int, dict[str, Batch]]:
    """The recorded batches of every round of a study, by round index and benchmark id."""
    ensure_tables(store)
    rows = store.execute(
        "SELECT round_index, batches FROM rounds WHERE study_id = ? ORDER BY round_index", (study.id,),
    ).fetchall()
    return {
        int(index): {benchmark_id: Batch.from_json(item) for benchmark_id, item in json.loads(text).items()}
        for index, text in rows
    }


def point_items(study: Study, store: Store, resolved_id: str, benchmark: Benchmark,
                rounds: Mapping[int, Mapping[str, Batch]], calibration_round: int) -> tuple[str, ...]:
    """The items a point's table on a benchmark is built from.

    These are the items of any round's fresh block that have an evidence row for the point, except the items of the
    ladder block of the round its candidate was calibrated in (when the blocks of that round overlap under the
    small-pool rule, nothing is excluded). Items keep the order of the rounds and blocks.

    Args:
        study: The study.
        store: The store.
        resolved_id: The resolved id of the point's recipe.
        benchmark: The benchmark.
        rounds: The batches of every round, by round index and benchmark id.
        calibration_round: The round the candidate was calibrated in.

    Returns:
        The item ids.
    """
    calibration = rounds[calibration_round][benchmark.id]
    excluded = set() if calibration.overlapping else set(calibration.ladder)
    measured = point_rows(study, store, resolved_id, benchmark)
    items: list[str] = []
    seen: set[str] = set()
    for round_index in sorted(rounds):
        for item_id in rounds[round_index][benchmark.id].fresh:
            if item_id in measured and item_id not in excluded and item_id not in seen:
                seen.add(item_id)
                items.append(item_id)
    return tuple(items)


def _pool_counts(study: Study, store: Store, benchmark: Benchmark) -> tuple[int, int]:
    """The search and sealed pool sizes: usable counts with a baseline, raw sizes without one."""
    split = study.pools.get(benchmark.id)
    if split is None:
        return 0, 0
    if _baseline_row(store, study, benchmark) is None:
        return len(split.search), len(split.sealed)
    return len(usable(study, store, benchmark, "search")), len(usable(study, store, benchmark, "sealed"))


def _size_problems(name: str, margin: float, variance: float, n_search: int, n_sealed: int, settings: Settings,
                   constraints: int) -> list[str]:
    """The room and band rules of one constraint at a margin and design variance."""
    last = last_look(margin=margin, variance=variance)
    ladder = ladder_size(last)
    problems = []
    if 2 * (ladder + last) > n_search:
        problems.append(
            f"{name}: two rounds of blocks need {2 * (ladder + last):,} search items (last look {last:,}, ladder "
            f"block {ladder:,}) and the pool has {n_search:,}"
        )
    if settings.guard_band:
        band = math.inf
        if n_sealed:
            band = guard_band(math.sqrt(variance / last), math.sqrt(variance / n_sealed), alpha=settings.alpha,
                              power=settings.power, benchmarks=constraints)
        if band > BAND_SHARE * margin:
            problems.append(
                f"{name}: the guard band is {100 * band:.2f} points, more than {100 * BAND_SHARE:.0f}% of the "
                f"{100 * margin:g}-point margin (design variance {variance:.4f}, last look {last:,}, {n_sealed:,} "
                "sealed items)"
            )
    return problems


def _feasible_alternatives(name: str, margin: float, measured: float | None, n_search: int, n_sealed: int,
                           settings: Settings, constraints: int) -> str | None:
    """What would be feasible for one constraint on its pools: the smallest margin and the largest variance."""

    def feasible_at(at_margin: float, variance: float) -> bool:
        return not _size_problems(name, at_margin, variance, n_search, n_sealed, settings, constraints)

    smallest = None
    step = 1
    while (at_margin := round(margin + step * _MARGIN_STEP, 6)) < _LARGEST_MARGIN:
        if feasible_at(at_margin, max(measured or 0.0, floor_variance(at_margin))):
            smallest = at_margin
            break
        step += 1
    largest = None
    floor = floor_variance(margin)
    variance = math.floor(max(measured or 0.0, floor) / _VARIANCE_STEP) * _VARIANCE_STEP
    while variance >= floor:
        if feasible_at(margin, variance):
            largest = variance
            break
        variance = round(variance - _VARIANCE_STEP, 6)
    parts = []
    if smallest is not None:
        parts.append(f"from a margin of {100 * smallest:.1f} points")
    if largest is not None:
        parts.append(f"at the present margin with a design variance of at most {largest:.3f}")
    if not parts:
        return None
    return f"{name}: on these pools the study is feasible " + ", or ".join(parts)


def feasibility(study: Study, store: Store) -> list[str]:
    """The reasons a study lacks power on its pools, with their numbers; empty when it has none.

    Pool sizes are usable counts for a benchmark with a baseline and raw sizes for one without, and the variance is
    `design_variance`. Each constraint, with `(ladder, last) = block_sizes(...)`, must pass two rules:

    - **Room**: `2 * (ladder + last) <= n_search`, i.e., two rounds of blocks fit the search pool.
    - **Band**: the guard band at the last look and the sealed pool size is at most `BAND_SHARE` of the margin
      (skipped when `settings.guard_band` is False).

    The objective needs `2 * (settings.behavior_ladder_items + settings.behavior_items)` search items. Each failed rule
    gives one message. After the messages of a failed constraint, one more message gives what would be feasible on
    these pools, i.e., the smallest margin (in steps of 0.001, below 0.5) at which both rules hold, and the largest
    design variance (in steps of 0.001, at least the floor) at which they hold at the present margin.

    Args:
        study: The study.
        store: The store, with the study's baseline when it is measured.

    Returns:
        One message per problem.
    """
    settings = study.settings
    constraints = len(study.constraints)
    problems: list[str] = []
    for constraint in study.constraints:
        benchmark = constraint.benchmark
        n_search, n_sealed = _pool_counts(study, store, benchmark)
        measured = baseline_variance(study, store, benchmark)
        variance = max(measured or 0.0, floor_variance(constraint.margin))
        found = _size_problems(benchmark.name, constraint.margin, variance, n_search, n_sealed, settings, constraints)
        if found:
            problems.extend(found)
            alternative = _feasible_alternatives(benchmark.name, constraint.margin, measured, n_search, n_sealed,
                                                 settings, constraints)
            if alternative is not None:
                problems.append(alternative)
    objective = study.objective
    n_search, _ = _pool_counts(study, store, objective)
    needed = 2 * (settings.behavior_ladder_items + settings.behavior_items)
    if needed > n_search:
        problems.append(
            f"{objective.name}: two rounds of behavior blocks need {needed:,} search items and the pool has "
            f"{n_search:,}"
        )
    return problems


__all__ = [
    "Batch",
    "GREEDY_NOISE_LIMIT",
    "MAX_EXCLUDED_SHARE",
    "NOISE_PROBE_ITEMS",
    "batch",
    "baseline_resolved_id",
    "baseline_variance",
    "block_sizes",
    "completion_of",
    "cost_ratio",
    "degenerate",
    "degenerate_increase",
    "design_variance",
    "ensure_baseline",
    "feasibility",
    "load_rounds",
    "noise_floor",
    "paired",
    "point_items",
    "point_rows",
    "record_round",
    "round_batches",
    "sampled",
    "usable",
]
