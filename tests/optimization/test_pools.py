"""Tests for the item pools and the held-out ledger (`optimization/measurement/pools.py`)."""
import dataclasses

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.store import Store
from steerability.optimization.measurement.pools import (
    PoolSplit,
    check_pools,
    held_out_id,
    record_held_out_use,
    shared_held_out_items,
    split_items,
)
from steerability.optimization.study import record_study
from tests.optimization.helpers import SyntheticBenchmark, Truth, build_study


def _quiet(recipe, item_id):
    return Truth()


def _benchmark(store, *, groups=None, size=200):
    benchmark = SyntheticBenchmark("bench", size, kind="capability", truth=_quiet, base=0.8, groups=groups)
    benchmark.prepare(store)
    return benchmark


@pytest.fixture()
def store(tmp_path):
    return Store.open(tmp_path / "store")


def test_splits_are_deterministic_by_seed_and_disjoint(store):
    benchmark = _benchmark(store)
    first = split_items(benchmark, held_out=50, seed=1)
    assert first == split_items(benchmark, held_out=50, seed=1)
    assert first != split_items(benchmark, held_out=50, seed=2)
    assert not set(first.search) & set(first.held_out)
    assert sorted(first.search + first.held_out) == sorted(benchmark.item_ids())
    assert len(first.held_out) == 50


def test_splits_keep_groups_whole(store):
    groups = {f"bench-{index:05d}": f"group-{index // 3}" for index in range(200)}
    benchmark = _benchmark(store, groups=groups)
    split = split_items(benchmark, held_out=50, seed=0)
    assert not {groups[item] for item in split.held_out} & {groups[item] for item in split.search}
    assert 50 <= len(split.held_out) <= 52


def test_a_held_out_size_beyond_the_items_is_refused(store):
    with pytest.raises(ValueError, match="held_out"):
        split_items(_benchmark(store), held_out=201, seed=0)


def test_check_pools_rejects_overlap_groups_fit_items_and_empty_pools(store):
    study = build_study(store, _quiet)
    check_pools(study)
    capability = study.constraints[0].benchmark
    split = study.pools[capability.id]
    overlapping = dataclasses.replace(
        study, pools={**study.pools, capability.id: PoolSplit(split.search + split.held_out[:1], split.held_out)},
    )
    with pytest.raises(ValueError, match="both pools"):
        check_pools(overlapping)
    fitted = dataclasses.replace(study, fit_item_ids={capability.id: {split.search[0]}})
    with pytest.raises(ValueError, match="fit items"):
        check_pools(fitted)
    unknown = dataclasses.replace(study, pools={**study.pools, capability.id: PoolSplit(("missing",), split.held_out)})
    with pytest.raises(ValueError, match="not in the benchmark"):
        check_pools(unknown)
    missing = dataclasses.replace(study, pools={key: value for key, value in study.pools.items()
                                                if key != capability.id})
    with pytest.raises(ValueError, match="no pool split"):
        check_pools(missing)
    no_search = dataclasses.replace(study, pools={**study.pools, capability.id: PoolSplit((), split.held_out)})
    with pytest.raises(ValueError, match="search pool is empty"):
        check_pools(no_search)
    no_held_out = dataclasses.replace(study, pools={**study.pools, capability.id: PoolSplit(split.search, ())})
    with pytest.raises(ValueError, match="held-out pool of a constraint is empty"):
        check_pools(no_held_out)
    objective = study.objective
    open_objective = dataclasses.replace(
        study, pools={**study.pools, objective.id: PoolSplit(study.pools[objective.id].search, ())},
    )
    check_pools(open_objective)


def test_check_pools_rejects_a_group_in_both_pools(store):
    groups = {f"capability-{index:05d}": "shared" for index in range(2)}
    study = build_study(store, _quiet)
    capability = study.constraints[0].benchmark
    grouped = SyntheticBenchmark("capability", 3000, kind="capability", truth=_quiet, base=0.8, groups=groups)
    grouped.prepare(store)
    split = PoolSplit(("capability-00000",) + tuple(f"capability-{index:05d}" for index in range(2, 1000)),
                      ("capability-00001",) + tuple(f"capability-{index:05d}" for index in range(1000, 3000)))
    pools = {study.objective.id: study.pools[study.objective.id], grouped.id: split}
    regrouped = dataclasses.replace(
        study, constraints=(dataclasses.replace(study.constraints[0], benchmark=grouped),), pools=pools,
    )
    assert capability.id != grouped.id
    with pytest.raises(ValueError, match="groups are in both pools"):
        check_pools(regrouped)


def test_the_ledger_finds_another_study_that_used_the_held_out_items(store):
    first = build_study(store, _quiet, name="first")
    second = build_study(store, _quiet, name="second")
    for study in (first, second):
        record_study(store, study)
    capability = first.constraints[0].benchmark
    assert shared_held_out_items(store, second) == {}
    record_held_out_use(store, first, capability, "resolved-a")
    record_held_out_use(store, first, capability, "resolved-a")
    assert shared_held_out_items(store, second) == {"capability": [first.id]}
    assert shared_held_out_items(store, first) == {}
    rows = store.execute("SELECT COUNT(*) FROM held_out_uses WHERE study_id = ?", (first.id,)).fetchone()[0]
    assert rows == 1
    assert held_out_id(capability, first.pools[capability.id]) == held_out_id(capability, second.pools[capability.id])
