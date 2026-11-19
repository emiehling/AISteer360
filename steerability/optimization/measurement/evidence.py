"""The evidence a study's decisions are made on.

The unsteered baseline (`BASELINE`) is measured once per study on every search and sealed item, and a second time on
the search items in another batch order, which gives the noise floor of each benchmark (`noise_floor`). An item
whose baseline row is an error row is excluded from its pool for the whole study (`usable`), since an errored
baseline scores 0 and would make every candidate look better on that item.

Every estimate is a table of paired differences against the baseline (`paired`). Items are drawn into batches per
round (`batch`), each with a ladder block (on which choices are made) and a fresh block (on which points are
confirmed and reported). A point's tables use fresh-block items only (`point_items`).
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
from steerability.optimization.measurement.stats import Paired, confirm_looks
from steerability.optimization.study import StudyError, StudyInfeasibleError, dumps, ensure_tables

if TYPE_CHECKING:
    from steerability.algorithms.core.store import EvidenceRow, Store
    from steerability.algorithms.core.sweeps import PipelineFactory
    from steerability.evaluation.items import Benchmark
    from steerability.optimization.study import Study

logger = logging.getLogger(__name__)

MAX_EXCLUDED_SHARE = 0.02
SAMPLED_BASELINE_REPLICATES = 4
SD_FLOOR_MINIMUM = 0.02

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


def ensure_baseline(study: Study, factory: PipelineFactory, store: Store) -> None:
    """Measure the unsteered baseline of every benchmark of a study, once.

    Each benchmark is prepared, and `BASELINE` is evaluated on every search and sealed item at replicate 0 and on the
    search items again at replicate 1. Under a sampling protocol it is evaluated at replicates 0 to 3 on both pools.
    One `baselines` row per benchmark records the baseline's resolved id, the noise floor (the share of usable search
    items whose value differs between replicates 0 and 1), and the excluded items (those with an error row in any
    replicate). A benchmark that already has a row is skipped.

    Args:
        study: The study.
        factory: The factory the baseline is steered with.
        store: The store.

    Raises:
        StudyError: If a baseline value is outside [0, 1], or more than 2% of a pool is excluded.
    """
    from steerability.evaluation.items import evaluate_items

    ensure_tables(store)
    for benchmark in study.benchmarks:
        benchmark.prepare(store)
    pending = [benchmark for benchmark in study.benchmarks if _baseline_row(store, study, benchmark) is None]
    if not pending:
        return
    with factory.steered(BASELINE) as steered:
        for benchmark in pending:
            split = study.pools[benchmark.id]
            everything = [*split.search, *split.sealed]
            if sampled(study, benchmark):
                plan = {replicate: everything for replicate in range(SAMPLED_BASELINE_REPLICATES)}
            else:
                plan = {0: everything, 1: list(split.search)}
            rows = {
                replicate: evaluate_items(steered, benchmark, items, store=store, replicate=replicate)
                for replicate, items in plan.items()
            }
            for replicate, results in rows.items():
                outside = [item for item, result in results.items()
                           if result.status == "ok" and not 0.0 <= result.value <= 1.0]
                if outside:
                    raise StudyError(
                        f"{benchmark.name}: baseline values must be in [0, 1]; items {outside[:5]} at replicate "
                        f"{replicate} are outside."
                    )
            excluded = sorted({item for results in rows.values() for item, result in results.items()
                               if result.status == "error"})
            excluded_set = set(excluded)
            for role, pool in (("search", split.search), ("sealed", split.sealed)):
                if not pool:
                    continue
                share = len(excluded_set.intersection(pool)) / len(pool)
                if share > MAX_EXCLUDED_SHARE:
                    raise StudyError(
                        f"{benchmark.name}: {100 * share:.1f}% of the {role} pool has baseline error rows (the "
                        f"limit is {100 * MAX_EXCLUDED_SHARE:.0f}%); since error rows are evidence, rerun on a new "
                        "store once the cause is fixed."
                    )
            usable_search = [item for item in split.search if item not in excluded_set]
            changed = sum(1 for item in usable_search if rows[0][item].value != rows[1][item].value)
            floor = changed / len(usable_search) if usable_search else 0.0
            store.execute(
                """INSERT OR IGNORE INTO baselines (study_id, benchmark_id, resolved_id, noise_floor, excluded)
                    VALUES (?, ?, ?, ?, ?)""",
                (study.id, benchmark.id, steered.resolved_id, floor, dumps(excluded)),
            )
            logger.info("Baseline of %s: noise floor %.4f, %d excluded item(s).", benchmark.name, floor, len(excluded))


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
    """The share of usable search items whose baseline value differs between replicates 0 and 1.

    Raises:
        StudyError: If the benchmark has no baseline.
    """
    return _require_baseline(store, study, benchmark)[1]


def sd_floor(study: Study, store: Store, benchmark: Benchmark) -> float:
    """The lower limit of the per-item standard deviation of a benchmark's differences.

    Returns:
        `sqrt(max(noise_floor, 0.02))`.

    Raises:
        StudyError: If the benchmark has no baseline.
    """
    return math.sqrt(max(noise_floor(study, store, benchmark), SD_FLOOR_MINIMUM))


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


def block_sizes(study: Study, benchmark: Benchmark) -> tuple[int, int]:
    """The ladder and fresh block sizes of a benchmark of the study.

    A constraint uses `settings.ladder_items` and `confirm_looks(margin)[2]`, and the objective uses
    `settings.behavior_ladder_items` and `settings.behavior_items`.

    Raises:
        ValueError: If the benchmark is not part of the study.
    """
    constraint = study.constraint_for(benchmark)
    if constraint is not None:
        return study.settings.ladder_items, confirm_looks(constraint.margin)[2]
    if benchmark.id == study.objective.id:
        return study.settings.behavior_ladder_items, study.settings.behavior_items
    raise ValueError(f"Benchmark {benchmark.name!r} is not part of study {study.name!r}.")


def batch(study: Study, store: Store, benchmark: Benchmark, round_index: int, *,
          allow_infeasible: bool = False) -> Batch:
    """The items of a benchmark for a round, drawn from its usable search pool.

    The draw is `random.Random(derive_seed(study.seed, "batch", benchmark.id, round_index)).sample(sorted(pool),
    ladder size + fresh size)`; the first part is the ladder block and the rest is the fresh block. When the pool
    has fewer items than the two blocks together (accepted only under `allow_infeasible`), each block is a prefix
    of one shuffled copy of the pool. The blocks then overlap, and the study chooses and judges on the same items.

    Args:
        study: The study.
        store: The store.
        benchmark: The benchmark.
        round_index: The round.
        allow_infeasible: Whether a pool smaller than the two blocks is accepted.

    Returns:
        The batch.

    Raises:
        StudyInfeasibleError: If the pool is smaller than the two blocks and `allow_infeasible` is False.
    """
    pool = sorted(usable(study, store, benchmark, "search"))
    ladder_size, fresh_size = block_sizes(study, benchmark)
    rng = random.Random(derive_seed(study.seed, "batch", benchmark.id, round_index))
    if len(pool) >= ladder_size + fresh_size:
        drawn = rng.sample(pool, ladder_size + fresh_size)
        return Batch(tuple(drawn[:ladder_size]), tuple(drawn[ladder_size:]))
    if not allow_infeasible:
        raise StudyInfeasibleError(
            f"{benchmark.name}: the usable search pool has {len(pool)} items, fewer than the ladder and fresh blocks "
            f"({ladder_size} + {fresh_size})."
        )
    logger.warning(
        "%s: the usable search pool (%d items) is smaller than the ladder and fresh blocks (%d + %d); both blocks are "
        "prefixes of one shuffled copy of the pool, and the study chooses and judges on the same items.",
        benchmark.name, len(pool), ladder_size, fresh_size,
    )
    shuffled = rng.sample(pool, len(pool))
    return Batch(tuple(shuffled[:ladder_size]), tuple(shuffled[:fresh_size]))


def round_batches(study: Study, store: Store, round_index: int, *, allow_infeasible: bool = False) -> dict[str, Batch]:
    """The batches of every benchmark of the study for a round, by benchmark id."""
    return {
        benchmark.id: batch(study, store, benchmark, round_index, allow_infeasible=allow_infeasible)
        for benchmark in study.benchmarks
    }


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


__all__ = [
    "Batch",
    "batch",
    "baseline_resolved_id",
    "block_sizes",
    "completion_of",
    "cost_ratio",
    "degenerate",
    "degenerate_increase",
    "ensure_baseline",
    "load_rounds",
    "noise_floor",
    "paired",
    "point_items",
    "point_rows",
    "record_round",
    "round_batches",
    "sampled",
    "sd_floor",
    "usable",
]
