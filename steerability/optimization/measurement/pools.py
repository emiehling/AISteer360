"""The item pools of a study and the ledger of held-out uses.

Each benchmark's items are split into a search pool, whose aggregates guide the search, and a held-out pool, which is
read only by certification. The roles are:

| Role | Used for | What may reach a proposer |
|---|---|---|
| Fit | Fitting artifacts | Everything, through the artifact |
| Search | Estimates inside the loop | Aggregate numbers only |
| Held out | Certification | Nothing |

Certification records a `held_out_uses` row for each finalist and benchmark before it reads that benchmark's held-out
items. A later study can then tell whether its held-out items were already used.
"""
from __future__ import annotations

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
    """The split of one benchmark's items into a search pool and a held-out pool.

    Attributes:
        search: The ids of the search items.
        held_out: The ids of the held-out items.
    """

    search: tuple[str, ...]
    held_out: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "search", tuple(str(item) for item in self.search))
        object.__setattr__(self, "held_out", tuple(str(item) for item in self.held_out))


def split_items(benchmark: Benchmark, *, held_out: int, seed: int) -> PoolSplit:
    """Split a prepared benchmark's items into a search pool and a held-out pool, keeping groups whole.

    Each group (an ungrouped item is its own group) is ranked by `derive_seed(seed, "split", key)`, where the key is
    the group's name or the item's id. Whole groups are then assigned to the held-out pool in rank order until it
    contains at least `held_out` items, and the rest is the search pool. Both pools keep the benchmark's item order.
    The split depends only on the seed, the item ids, and the groups (not on the benchmark's id), and an item that is
    added or removed moves no other item between the pools.

    Args:
        benchmark: The benchmark, prepared on a store.
        held_out: The smallest number of held-out items.
        seed: The split seed.

    Returns:
        The split.

    Raises:
        ValueError: If `held_out` is negative or exceeds the number of items.
    """
    item_ids = list(benchmark.item_ids())
    if held_out < 0 or held_out > len(item_ids):
        raise ValueError(f"held_out must be between 0 and the {len(item_ids)} items of {benchmark.name!r}; "
                         f"got {held_out}.")
    groups = benchmark.groups()
    members: dict[str, list[str]] = {}
    for item_id in item_ids:
        key = f"group:{groups[item_id]}" if item_id in groups else f"item:{item_id}"
        members.setdefault(key, []).append(item_id)
    keys = sorted(members, key=lambda key: (derive_seed(seed, "split", key), key))
    held_out_items: set[str] = set()
    for key in keys:
        if len(held_out_items) >= held_out:
            break
        held_out_items.update(members[key])
    return PoolSplit(
        search=tuple(item_id for item_id in item_ids if item_id not in held_out_items),
        held_out=tuple(item_id for item_id in item_ids if item_id in held_out_items),
    )


def _examples(items: Sequence[str] | set[str]) -> str:
    listed = sorted(items)
    suffix = ", ..." if len(listed) > _EXAMPLES else ""
    return ", ".join(repr(item) for item in listed[:_EXAMPLES]) + suffix


def check_pools(study: Study) -> None:
    """Check that the pools of a study separate their items.

    Every benchmark of the study (objective and constraints) needs a split of its own items with a search pool that is
    not empty, and a constraint also needs a held-out pool that is not empty. The two pools of one benchmark must share
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
        if not split.held_out and study.constraint_for(benchmark) is not None:
            raise ValueError(f"{benchmark.name}: the held-out pool of a constraint is empty.")
        known = set(benchmark.item_ids())
        for role, items in (("search", split.search), ("held_out", split.held_out)):
            repeated = {item for item, count in Counter(items).items() if count > 1}
            if repeated:
                raise ValueError(f"{benchmark.name}: the {role} pool repeats items {_examples(repeated)}.")
            unknown = set(items) - known
            if unknown:
                raise ValueError(f"{benchmark.name}: the {role} pool has items that are not in the benchmark: "
                                 f"{_examples(unknown)}.")
        shared = set(split.search) & set(split.held_out)
        if shared:
            raise ValueError(f"{benchmark.name}: {len(shared)} items are in both pools: {_examples(shared)}.")
        groups = benchmark.groups()
        search_groups = {groups[item] for item in split.search if item in groups}
        held_out_groups = {groups[item] for item in split.held_out if item in groups}
        shared_groups = search_groups & held_out_groups
        if shared_groups:
            raise ValueError(f"{benchmark.name}: {len(shared_groups)} groups are in both pools: "
                             f"{_examples(shared_groups)}.")
        fit_items = study.fit_item_ids.get(benchmark.id, frozenset()) & (set(split.search) | set(split.held_out))
        if fit_items:
            raise ValueError(f"{benchmark.name}: {len(fit_items)} fit items are in a pool: {_examples(fit_items)}.")


def held_out_id(benchmark: Benchmark, split: PoolSplit) -> str:
    """The id of a benchmark's held-out pool, `content_id({"benchmark": name, "items": sorted held-out ids})`."""
    return content_id({"benchmark": benchmark.name, "items": sorted(split.held_out)})


def record_held_out_use(store: Store, study: Study, benchmark: Benchmark, resolved_id: str) -> None:
    """Record that a resolved recipe is about to be measured on a benchmark's held-out items (once).

    Args:
        store: The store.
        study: The study.
        benchmark: The benchmark whose held-out items are read.
        resolved_id: The resolved id of the measured recipe.
    """
    from steerability.optimization.study import ensure_tables, now

    ensure_tables(store)
    identity = held_out_id(benchmark, study.pools[benchmark.id])
    exists = store.execute(
        "SELECT 1 FROM held_out_uses WHERE study_id = ? AND benchmark_name = ? AND held_out_id = ? AND resolved_id = ?",
        (study.id, benchmark.name, identity, resolved_id),
    ).fetchone()
    if exists is None:
        store.execute(
            """INSERT INTO held_out_uses (study_id, benchmark_name, held_out_id, resolved_id, used_at)
                VALUES (?, ?, ?, ?, ?)""",
            (study.id, benchmark.name, identity, resolved_id, now()),
        )


def shared_held_out_items(store: Store, study: Study) -> dict[str, list[str]]:
    """The other studies whose recorded held-out uses overlap this study's held-out items, by benchmark name.

    A study counts when it has a `held_out_uses` row for a benchmark of the same name and its recorded held-out pool for
    a benchmark of that name shares an item with this study's held-out pool.

    Args:
        store: The store.
        study: The study.

    Returns:
        Mapping from benchmark name to the ids of the other studies; empty when nothing is shared.
    """
    from steerability.optimization.study import ensure_tables, load_definition

    ensure_tables(store)
    mine = {
        benchmark.name: set(study.pools[benchmark.id].held_out)
        for benchmark in study.benchmarks if benchmark.id in study.pools
    }
    rows = store.execute(
        "SELECT DISTINCT study_id, benchmark_name FROM held_out_uses WHERE study_id != ? ORDER BY study_id",
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
                other_items.update(definition["pools"][benchmark["id"]]["held_out"])
        if mine[name] & other_items:
            shared.setdefault(name, []).append(other_id)
    return shared


__all__ = [
    "PoolSplit",
    "check_pools",
    "record_held_out_use",
    "held_out_id",
    "shared_held_out_items",
    "split_items",
]
