"""Registry auto-discovery is lossless across the `core/` promotion (doc 01 §7).

The crawl root is derived from the `algorithms/` package, not the registry module's own location,
so moving `registry.py` into `aisteer360/core/` must not change the discovered method set. This
snapshots the current `(category, name)` set and asserts a re-crawl reproduces it exactly; the count
is derived, never hard-coded, so new methods survive the check.
"""
from aisteer360.core.registry import REGISTRY, _crawl_methods


def _discovered() -> set[tuple[str, str]]:
    return {
        (method.category, method.name)
        for bucket in REGISTRY.values()
        for method in bucket.values()
    }


def test_registry_discovers_all_categories():
    discovered = _discovered()
    categories = {category for category, _ in discovered}
    assert categories == {"input", "structural", "state", "output"}
    # a representative from each category must be present
    assert ("state", "cast") in discovered
    assert ("input", "few_shot") in discovered
    assert ("output", "deal") in discovered
    assert ("structural", "dpo") in discovered


def test_registry_crawl_is_idempotent():
    before = _discovered()
    _crawl_methods()  # re-crawl must not lose or duplicate methods
    after = _discovered()
    assert before == after
    assert len(after) >= 20  # sanity floor; exact count not pinned so new methods survive
