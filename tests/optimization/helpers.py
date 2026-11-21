"""A synthetic world for the optimization tests: benchmarks with known truth and a factory that steers nothing.

`SyntheticBenchmark` scores every item from a `Truth` chosen per (recipe, item) by the test, so a capability item's
value is its baseline value minus `loss` and a behavior item's value is its baseline value plus `gain` (plus an
optional per-item spread). Differences against the baseline are then exact, which makes every statistical decision
deterministic. The capability benchmark of `benchmarks` has a two-level baseline (1.0 for an item with
`unit(item_id) < 0.5`, else 0.4), which gives it a baseline variance close to 0.09. `StubFactory` yields `Steered`
objects whose resolved id is derived from the recipe id, and `StubPipeline` records the bundles written.

With the default pools (1,000 search and 2,000 sealed capability items, 200 behavior search items) and the 5-point
margin, a constraint has blocks of (100, 400) items and looks of (32, 128, 400). A candidate that shifts every item
equally is confirmed when its loss is at most 0.0253 (second look) or 0.0319 (third look), a ladder position of a
solved axis outside `study.monotone` is kept when its loss is at most 0.0420, and the sealed test passes up to a loss
of 0.0420.
"""
import hashlib
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steerability.algorithms.core.identity import content_id
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Choice, Interval, Recipe, Template, entry
from steerability.algorithms.core.store import ItemRow, Store
from steerability.algorithms.core.sweeps import Steered
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA
from steerability.evaluation.items import ItemResult
from steerability.optimization.measurement.pools import split_items
from steerability.optimization.study import Constraint, Settings, Study

PAIRS = {"positives": ["the cat sat on", "the cat ran", "a cat"], "negatives": ["the dog sat", "dog ran", "a dog"]}
OTHER = {"positives": ["the mat", "on the mat"], "negatives": ["the dog", "a dog ran"]}
RAW = {"method": "mean_diff", "prompt_format": "raw"}

SMALL = Settings(
    round_candidates=4, ladder_points=4, behavior_ladder_items=32, behavior_items=64, quick_behavior_items=16,
    rescore_top=2, finalists=3,
)
MARGIN = 0.05
CAPABILITY_ITEMS = 3000
CAPABILITY_SEALED = 2000
BEHAVIOR_ITEMS = 400
BEHAVIOR_SEALED = 200


@dataclass(frozen=True)
class Truth:
    """What a recipe does to one item.

    Attributes:
        loss: The capability loss on the item.
        gain: The behavior gain on the item.
        spread: The size of a per-item behavior variation (half the items get `+spread`, half `-spread`).
        tokens: The tokens generated for the item.
        completion: The completion of the item.
    """

    loss: float = 0.0
    gain: float = 0.0
    spread: float = 0.0
    tokens: int = 10
    completion: str = "one two three four five six seven eight"


def unit(item_id: str) -> float:
    """A deterministic uniform value in [0, 1) for an item id."""
    return int(hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32


def argument(recipe: Recipe, method: str, name: str, default: Any = None) -> Any:
    """The value of an argument of the first entry of a method in a recipe, or `default`."""
    for item in recipe.entries:
        if item.method == method:
            return item.args.get(name, default)
    return default


def multiplier(recipe: Recipe) -> float | None:
    """The CAA multiplier of a recipe, or None without a CAA entry."""
    value = argument(recipe, "state_control/caa", "multiplier")
    return None if value is None else float(value)


TruthFunction = Callable[[Recipe, str], Truth]


class SyntheticBenchmark:
    """A benchmark whose item values follow a truth function of (recipe, item).

    Args:
        name: The benchmark's name.
        size: The number of items.
        kind: `"capability"` or `"behavior"`.
        truth: The truth function; BASELINE gets `Truth()` unless the function says otherwise.
        base: The baseline value of every item, or a mapping from item id to its baseline value.
        groups: Mapping from item id to group id.
        omit: Items `run` always omits (they get error rows).
    """

    def __init__(self, name: str, size: int, *, kind: str, truth: TruthFunction, base: float | Mapping[str, float],
                 groups: Mapping[str, str] | None = None, omit: set[str] | None = None) -> None:
        self.name = name
        self.items = [f"{name}-{index:05d}" for index in range(size)]
        self.kind = kind
        self._truth = truth
        self.base = base
        self.groups_by_item = dict(groups or {})
        self.omit = set(omit or ())
        self.calls: list[tuple[str, tuple[str, ...], int]] = []
        self._hook: Callable[[Steered, list[str]], None] | None = None
        self._id = content_id({"name": name, "items": self.items, "kind": kind, "base": base,
                               "groups": self.groups_by_item})
        self._store: Store | None = None

    @property
    def id(self) -> str:
        return self._id

    def prepare(self, store: Store) -> None:
        store.put_items(ItemRow(self.id, item, self.groups_by_item.get(item), {
            "id": item, "input": item, "target": None, "metadata": {},
        }) for item in self.items)
        self._store = store

    def item_ids(self) -> tuple[str, ...]:
        return tuple(self.items)

    def groups(self) -> Mapping[str, str]:
        return dict(self.groups_by_item)

    def baseline(self, item_id: str) -> float:
        """The baseline value of an item."""
        return float(self.base[item_id]) if isinstance(self.base, Mapping) else float(self.base)

    def value(self, recipe: Recipe, item_id: str) -> tuple[float, Truth]:
        truth = Truth() if not recipe.entries else self._truth(recipe, item_id)
        if self.kind == "capability":
            return self.baseline(item_id) - truth.loss, truth
        spread = truth.spread if unit(item_id) < 0.5 else -truth.spread
        return self.baseline(item_id) + truth.gain + spread, truth

    def run(self, steered: Steered, item_ids, *, store: Store, replicate: int) -> dict[str, ItemResult]:
        item_ids = list(item_ids)
        if self._hook is not None:
            self._hook(steered, item_ids)
        self.calls.append((steered.resolved_id, tuple(item_ids), replicate))
        results = {}
        for item_id in item_ids:
            if item_id in self.omit:
                continue
            value, truth = self.value(steered.recipe, item_id)
            results[item_id] = ItemResult(
                item_id=item_id, value=value, status="ok", tokens=truth.tokens, completion=truth.completion,
                scores={"score": value}, rollout_key=None, generated=True,
            )
        return results

    def set_truth(self, truth: TruthFunction) -> None:
        """Replace the truth function."""
        self._truth = truth

    def set_hook(self, hook: Callable[[Steered, list[str]], None] | None) -> None:
        """Call `hook(steered, item_ids)` at the start of every `run`."""
        self._hook = hook

    def measured(self, resolved_id: str | None = None) -> set[str]:
        """The items `run` was called on (for one resolved id, when given)."""
        return {item for resolved, items, _ in self.calls if resolved_id in (None, resolved) for item in items}


@dataclass
class StubBundle:
    """The bundle a `StubPipeline` writes."""

    options: dict
    saved: list[Path] = field(default_factory=list)

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text("bundle", encoding="utf-8")
        self.saved.append(path)
        return path


class StubPipeline:
    """A pipeline that records the bundles written from it."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.bundles: list[StubBundle] = []

    def to_spipe(self, **options) -> StubBundle:
        bundle = StubBundle(dict(options))
        self.bundles.append(bundle)
        return bundle


class StubFactory:
    """A factory whose steered pipelines run nothing; each recipe resolves to `"resolved-<recipe id>"`."""

    def __init__(self, protocol: Protocol) -> None:
        self.protocol = protocol
        self.prepared: list[list[str]] = []
        self.steered_recipes: list[str] = []
        self.pipelines: list[StubPipeline] = []
        self.released = False

    def prepare(self, recipes) -> None:
        self.prepared.append([recipe.id for recipe in recipes])

    @contextmanager
    def steered(self, recipe: Recipe):
        self.steered_recipes.append(recipe.id)
        pipeline = StubPipeline(recipe)
        self.pipelines.append(pipeline)
        yield Steered(pipeline, recipe, f"resolved-{recipe.id}", self.protocol)

    def release(self) -> None:
        self.released = True


def caa_template(multiplier_range=None, *, layer_id: Any = 1, data: Any = None, **args) -> Template:
    """A CAA template with the multiplier (by default `Interval(0, 3)`) as its free argument."""
    return Template((entry(
        CAA, data=PAIRS if data is None else data, layer_id=layer_id, train_spec=RAW,
        multiplier=Interval(0.0, 3.0) if multiplier_range is None else multiplier_range, **args,
    ),))


def prompt_template(texts=("be brief", "be formal", "be kind")) -> Template:
    """A system-prompt template (entry name `prompt`) whose text is an unordered choice, at `prompt.text`."""
    return Template((entry(SystemPrompt, name="prompt", text=Choice(tuple(texts))),))


def protocol() -> Protocol:
    return Protocol.create("synthetic-model", generate_defaults={"temperature": 0})


def two_level(name: str, size: int, *, high: float = 1.0, low: float = 0.4) -> dict[str, float]:
    """The two-level baseline of a synthetic benchmark's items: `high` when `unit(item_id) < 0.5`, else `low`."""
    items = (f"{name}-{index:05d}" for index in range(size))
    return {item: high if unit(item) < 0.5 else low for item in items}


def benchmarks(truth: TruthFunction, *, capability_omit=None, behavior_omit=None,
               capability_items: int = CAPABILITY_ITEMS, behavior_items: int = BEHAVIOR_ITEMS):
    """A capability benchmark (the two-level baseline) and a behavior benchmark (base 0.3) under one truth function."""
    capability = SyntheticBenchmark("capability", capability_items, kind="capability", truth=truth,
                                    base=two_level("capability", capability_items), omit=capability_omit)
    behavior = SyntheticBenchmark("behavior", behavior_items, kind="behavior", truth=truth, base=0.3,
                                  omit=behavior_omit)
    return capability, behavior


def build_study(store: Store, truth: TruthFunction, *, space: Mapping[str, Template] | None = None,
                settings: Settings = SMALL, margin: float = MARGIN, name: str = "synthetic",
                monotone=frozenset(), searched=frozenset(), capability=None, behavior=None,
                capability_sealed: int = CAPABILITY_SEALED, behavior_sealed: int = BEHAVIOR_SEALED,
                fit_item_ids=None) -> Study:
    """A study over a synthetic capability constraint and a synthetic behavior objective."""
    if capability is None or behavior is None:
        capability, behavior = benchmarks(truth)
    capability.prepare(store)
    behavior.prepare(store)
    pools = {
        capability.id: split_items(capability, sealed=capability_sealed, seed=0),
        behavior.id: split_items(behavior, sealed=behavior_sealed, seed=0),
    }
    return Study(
        name=name, protocol=protocol(), space=space if space is not None else {"caa": caa_template()},
        objective=behavior, constraints=(Constraint(capability, margin),), pools=pools, settings=settings,
        monotone=frozenset(monotone), searched=frozenset(searched), fit_item_ids=fit_item_ids or {},
    )
