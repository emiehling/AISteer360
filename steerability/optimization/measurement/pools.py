"""The item pools of a study and the ledger of sealed uses.

Each benchmark's items are split into a search pool, whose aggregates guide the search, and a sealed pool, which is
read only by certification. The roles are:

| Role | Used for | What may reach a proposer |
|---|---|---|
| Fit | Fitting artifacts | Everything, through the artifact |
| Search | Estimates inside the loop | Aggregate numbers only |
| Sealed | Certification | Nothing |

Certification records a `sealed_uses` row for each finalist and benchmark before it reads that benchmark's sealed
items. A later study can then tell whether its sealed items were already used.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from steerability.algorithms.core.identity import content_id, derive_seed

if TYPE_CHECKING:
    from steerability.algorithms.core.store import Store
    from steerability.evaluation.items import Benchmark
    from steerability.optimization.study import Study

_EXAMPLES = 5


@dataclass(frozen=True)
class PoolSplit:
    """The split of one benchmark's items into a search pool and a sealed pool.

    Attributes:
        search: The ids of the search items.
        sealed: The ids of the sealed items.
    """

    search: tuple[str, ...]
    sealed: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "search", tuple(str(item) for item in self.search))
        object.__setattr__(self, "sealed", tuple(str(item) for item in self.sealed))


def split_items(benchmark: Benchmark, *, sealed: int, seed: int) -> PoolSplit:
    """Split a prepared benchmark's items into a search pool and a sealed pool, keeping groups whole.

    The benchmark's groups are sorted (an ungrouped item is its own group) and shuffled with
    `random.Random(derive_seed(seed, "split", benchmark.id))`. Whole groups are then assigned to the sealed pool until
    it contains at least `sealed` items, and the rest is the search pool. Both pools keep the benchmark's item order.

    Args:
        benchmark: The benchmark, prepared on a store.
        sealed: The smallest number of sealed items.
        seed: The split seed.

    Returns:
        The split.

    Raises:
        ValueError: If `sealed` is negative or exceeds the number of items.
    """
    item_ids = list(benchmark.item_ids())
    if sealed < 0 or sealed > len(item_ids):
        raise ValueError(f"sealed must be between 0 and the {len(item_ids)} items of {benchmark.name!r}; got {sealed}.")
    groups = benchmark.groups()
    members: dict[str, list[str]] = {}
    for item_id in item_ids:
        key = f"group:{groups[item_id]}" if item_id in groups else f"item:{item_id}"
        members.setdefault(key, []).append(item_id)
    keys = sorted(members)
    random.Random(derive_seed(seed, "split", benchmark.id)).shuffle(keys)
    sealed_items: set[str] = set()
    for key in keys:
        if len(sealed_items) >= sealed:
            break
        sealed_items.update(members[key])
    return PoolSplit(
        search=tuple(item_id for item_id in item_ids if item_id not in sealed_items),
        sealed=tuple(item_id for item_id in item_ids if item_id in sealed_items),
    )


def _examples(items: Sequence[str] | set[str]) -> str:
    listed = sorted(items)
    suffix = ", ..." if len(listed) > _EXAMPLES else ""
    return ", ".join(repr(item) for item in listed[:_EXAMPLES]) + suffix


def check_pools(study: Study) -> None:
    """Check that the pools of a study separate their items.

    Every benchmark of the study (objective and constraints) needs a split of its own items with a search pool that is
    not empty, and a constraint also needs a sealed pool that is not empty. The two pools of one benchmark must share
    no item id and no group id, neither pool may repeat an item, and neither may contain an item of
    `study.fit_item_ids`. The benchmarks must be prepared.

    Args:
        study: The study.

    Raises:
        ValueError: If a check fails; the message gives the benchmark and example item ids.
    """
    benchmark_ids = {benchmark.id for benchmark in study.benchmarks}
    unknown_pools = sorted(set(study.pools) - benchmark_ids)
    if unknown_pools:
        raise ValueError(f"study.pools has splits for {unknown_pools}, which are ids of no benchmark of the study.")
    unknown_fits = sorted(set(study.fit_item_ids) - benchmark_ids)
    if unknown_fits:
        raise ValueError(f"study.fit_item_ids has keys {unknown_fits}, which are ids of no benchmark of the study.")
    for benchmark in study.benchmarks:
        split = study.pools.get(benchmark.id)
        if split is None:
            raise ValueError(f"Benchmark {benchmark.name!r} ({benchmark.id}) has no pool split in study.pools.")
        if not split.search:
            raise ValueError(f"{benchmark.name}: the search pool is empty.")
        if not split.sealed and study.constraint_for(benchmark) is not None:
            raise ValueError(f"{benchmark.name}: the sealed pool of a constraint is empty.")
        known = set(benchmark.item_ids())
        for role, items in (("search", split.search), ("sealed", split.sealed)):
            repeated = {item for item, count in Counter(items).items() if count > 1}
            if repeated:
                raise ValueError(f"{benchmark.name}: the {role} pool repeats items {_examples(repeated)}.")
            unknown = set(items) - known
            if unknown:
                raise ValueError(f"{benchmark.name}: the {role} pool has items that are not in the benchmark: "
                                 f"{_examples(unknown)}.")
        shared = set(split.search) & set(split.sealed)
        if shared:
            raise ValueError(f"{benchmark.name}: {len(shared)} items are in both pools: {_examples(shared)}.")
        groups = benchmark.groups()
        search_groups = {groups[item] for item in split.search if item in groups}
        sealed_groups = {groups[item] for item in split.sealed if item in groups}
        shared_groups = search_groups & sealed_groups
        if shared_groups:
            raise ValueError(f"{benchmark.name}: {len(shared_groups)} groups are in both pools: "
                             f"{_examples(shared_groups)}.")
        fit_items = study.fit_item_ids.get(benchmark.id, frozenset()) & (set(split.search) | set(split.sealed))
        if fit_items:
            raise ValueError(f"{benchmark.name}: {len(fit_items)} fit items are in a pool: {_examples(fit_items)}.")


def sealed_id(benchmark: Benchmark, split: PoolSplit) -> str:
    """The id of a benchmark's sealed pool, `content_id({"benchmark": name, "items": sorted sealed ids})`."""
    return content_id({"benchmark": benchmark.name, "items": sorted(split.sealed)})


def record_sealed_use(store: Store, study: Study, benchmark: Benchmark, resolved_id: str) -> None:
    """Record that a resolved recipe is about to be measured on a benchmark's sealed items (once).

    Args:
        store: The store.
        study: The study.
        benchmark: The benchmark whose sealed items are read.
        resolved_id: The resolved id of the measured recipe.
    """
    from steerability.optimization.study import ensure_tables, now

    ensure_tables(store)
    identity = sealed_id(benchmark, study.pools[benchmark.id])
    exists = store.execute(
        "SELECT 1 FROM sealed_uses WHERE study_id = ? AND benchmark_name = ? AND sealed_id = ? AND resolved_id = ?",
        (study.id, benchmark.name, identity, resolved_id),
    ).fetchone()
    if exists is None:
        store.execute(
            """INSERT INTO sealed_uses (study_id, benchmark_name, sealed_id, resolved_id, used_at)
                VALUES (?, ?, ?, ?, ?)""",
            (study.id, benchmark.name, identity, resolved_id, now()),
        )


def shared_sealed_items(store: Store, study: Study) -> dict[str, list[str]]:
    """The other studies whose recorded sealed uses overlap this study's sealed items, by benchmark name.

    A study counts when it has a `sealed_uses` row for a benchmark of the same name and its recorded sealed pool for
    a benchmark of that name shares an item with this study's sealed pool.

    Args:
        store: The store.
        study: The study.

    Returns:
        Mapping from benchmark name to the ids of the other studies; empty when nothing is shared.
    """
    from steerability.optimization.study import ensure_tables, load_definition

    ensure_tables(store)
    mine = {
        benchmark.name: set(study.pools[benchmark.id].sealed)
        for benchmark in study.benchmarks if benchmark.id in study.pools
    }
    rows = store.execute(
        "SELECT DISTINCT study_id, benchmark_name FROM sealed_uses WHERE study_id != ? ORDER BY study_id",
        (study.id,),
    ).fetchall()
    shared: dict[str, list[str]] = {}
    for other_id, name in rows:
        if name not in mine:
            continue
        definition = load_definition(store, other_id)
        if definition is None:
            continue
        named = [definition["objective"], *definition["constraints"]]
        other_items: set[str] = set()
        for benchmark in named:
            if benchmark["name"] == name and benchmark["id"] in definition["pools"]:
                other_items.update(definition["pools"][benchmark["id"]]["sealed"])
        if mine[name] & other_items:
            shared.setdefault(name, []).append(other_id)
    return shared


__all__ = [
    "PoolSplit",
    "check_pools",
    "record_sealed_use",
    "sealed_id",
    "shared_sealed_items",
    "split_items",
]
