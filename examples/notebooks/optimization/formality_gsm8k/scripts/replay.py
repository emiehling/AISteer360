"""Replays of whole studies, with no model, in a world built from the truth grid where the truth is known.

The `export` command reads the truth grid's shard stores through their `grid_shard_*.json` indexes and writes one
JSON file with, for both benchmarks, the values and tokens of the baseline (replicates 0 and 1) and of every setting
on every grid item, and for the objective whether each completion is degenerate (`evidence.degenerate`).

The `run` command runs one replay per seed. A replay draws its pools with replacement from the grid's items (3,792
search and 5,000 held-out GSM8K items, 1,500 search and 600 held-out formality items). Each drawn item gets a new id
that maps to its grid item. The truth of a setting is its mean over all grid items, and the pools are independent of it
and of each other. The benchmarks serve the exported values and tokens with a stand-in completion (empty when the real
one was degenerate), and the factory steers nothing. The notebook `formality_gsm8k.ipynb` reads the export with
`load_export` and the truth with `truth`, and builds the world of one replay with `build_world`.

The study of a replay has one structure, the grid's CAA entry with `layer_id` as an unordered `Choice` and
`multiplier` as an ordered `Choice`. The multiplier is in `study.monotone`, or in `study.searched` with
`--searched`. The margin is 5 points, and the settings are `Settings(round_candidates=5)` unless
`--round-candidates` is given (and `ladder_power` is `--ladder-power` when it is given). Each replay runs `run_study`
on a new store, with one search phase, and then `certify`. The phase's engine is a `GridProposer`, or a
`RandomProposer` seeded with the replay's seed with `--proposer random`, which visits the layers in no fixed order when
`--max-generations` stops a search early.

The report gives the following measures over the seeds:

- the share of replays that certify a setting whose true loss is over the margin
- the share of replays whose first finalist passes
- the best certified true gain as a share of the best setting truly inside the margin
- per layer, the chosen multipliers and the largest truly feasible multiplier
- the held-out pass rate of every accepted point (the confirmation re-run)
- each pick's true gain as a share of the best true gain inside `margin - band` (the calibration re-run)
- the best true gain among accepted points truly inside the margin (solved against searched, at an equal
  `--max-generations`)
- the bias of the winner's loss estimate in standard errors (round length)
- the mean number of recorded candidates and the mean generations per candidate
- the false-rejection share, i.e., the share of rejected candidates among the recoverable ones (those whose layer
  has a multiplier with a true loss of at most `margin - band` and a true gain of at least `min_gain`), and the
  number of rejections per reason

The `compare` command reads a report of today's rule and the reports of challengers (run on the same seeds), listed
from the smallest change to the largest. A challenger qualifies when it passes three gates and shows a benefit.
The gates are validity (at most the one-sided 95% binomial limit of replays at `alpha` certify a setting over the
margin), the promise (an accepted point's held-out pass share of at least `min(power, today's share - 0.02)`), and
value (the one-sided 95% bound of the per-seed difference in the best certified gain share is above -0.05). The
benefits are a better value (that bound is above 0), or at most 0.85 of today's generations per candidate together
with a positive bound of the per-seed difference in false-rejection share (today's minus the challenger's). The
choice starts from the qualifying challenger with the highest mean value, and the first qualifying challenger in
the listed order that it does not beat (the bound of their per-seed value difference is at most 0) replaces it.
Without a qualifying challenger, the choice is today's rule.

usage:
    python examples/notebooks/optimization/formality_gsm8k/scripts/replay.py export \
        <store> [<store> ...] --output grid.json
    python examples/notebooks/optimization/formality_gsm8k/scripts/replay.py run \
        grid.json --seeds 200 [--workers 8] [--searched] [--ladder-power P]
    python examples/notebooks/optimization/formality_gsm8k/scripts/replay.py compare \
        today.json challenger.json [challenger.json ...]
"""
import argparse
import dataclasses
import json
import logging
import math
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from formality import PAIRS
from truth_grid import shards

from steerability.algorithms.core.identity import content_id
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Choice, Template, entry
from steerability.algorithms.core.store import ItemRow, Store
from steerability.algorithms.core.sweeps import Steered
from steerability.algorithms.state_control.caa.control import CAA
from steerability.evaluation.items import ItemResult
from steerability.optimization import Constraint, PoolSplit, Settings, Study
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.measurement.evidence import (
    block_sizes,
    completion_of,
    degenerate,
    design_variance,
    usable,
)
from steerability.optimization.measurement.stats import Paired, guard_band
from steerability.optimization.release.certify import certify
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.proposers import GridProposer, RandomProposer

logger = logging.getLogger("replay")

MARGIN = 0.05
POOLS = {"capability": (3792, 5000), "behavior": (1500, 600)}
STAND_IN = "a stand-in completion of the replayed item"
BOUND_Z = 1.645  # one-sided 95% bound of a mean over seeds
VALIDITY_QUANTILE = 0.95
PROMISE_SLACK = 0.02
VALUE_TOLERANCE = 0.05
COST_SHARE = 0.85
_DATA: dict | None = None


def _rows(store: Store, protocol_id: str, resolved_id: str, benchmark_id: str, items, replicate: int = 0):
    rows = store.get_evidence(protocol_id, resolved_id, benchmark_id, items, replicate)
    return [rows[item] for item in items]


def _columns(store: Store, rows, with_degenerate: bool) -> dict:
    columns = {"values": [row.value for row in rows], "tokens": [row.tokens or 0 for row in rows]}
    if with_degenerate:
        columns["degenerate"] = [bool(degenerate(completion_of(store, row))) for row in rows]
    return columns


def export(arguments) -> None:
    """Write the grid's baseline and setting values, tokens, and degenerate flags for both benchmarks to one file."""
    found = shards(arguments.stores)
    store, index = next((store, index) for store, index in found if "baseline" in index)
    protocol_id, baseline_id = index["protocol_id"], index["baseline"]
    names = {"capability": (index["benchmark_id"], index["items"], False),
             "behavior": (index["objective_id"], index["objective_items"], True)}
    data: dict = {"settings": [], "benchmarks": {}}
    for name, (benchmark_id, items, flags) in names.items():
        data["benchmarks"][name] = {
            "items": items,
            "baseline": [_columns(store, _rows(store, protocol_id, baseline_id, benchmark_id, items, replicate),
                                  flags) for replicate in (0, 1)],
            "settings": [],
        }
    for shard_store, shard_index in found:
        for record in shard_index["points"].values():
            data["settings"].append(dict(record["position"]))
            for name, (benchmark_id, items, flags) in names.items():
                rows = _rows(shard_store, protocol_id, record["resolved_id"], benchmark_id, items)
                data["benchmarks"][name]["settings"].append(_columns(shard_store, rows, flags))
    Path(arguments.output).write_text(json.dumps(data), encoding="utf-8")
    logger.info("%d settings written to %s", len(data["settings"]), arguments.output)


def _template(layers, multipliers) -> Template:
    return Template((entry(
        CAA, data=PAIRS, train_spec={"method": "mean_diff", "accumulate": "all"}, layer_id=Choice(tuple(layers)),
        multiplier=Choice(tuple(multipliers), ordered=True),
    ),))


class ReplayBenchmark:
    """A benchmark that serves the exported values of the grid item each of its items maps to.

    Args:
        name: The benchmark's name.
        grid_items: Mapping from item id to its grid item index.
        exported: The benchmark's exported columns.
        settings: Mapping from recipe id to the setting index of the export.
    """

    def __init__(self, name: str, grid_items: dict[str, int], exported: dict, settings: dict[str, int]) -> None:
        self.name = name
        self._grid_items = grid_items
        self._baseline = [{key: np.asarray(value) for key, value in replicate.items()}
                          for replicate in exported["baseline"]]
        self._settings = [{key: np.asarray(value) for key, value in setting.items()}
                          for setting in exported["settings"]]
        self._recipes = settings
        self._id = content_id({"name": name, "items": sorted(grid_items.items())})

    @property
    def id(self) -> str:
        return self._id

    def prepare(self, store: Store) -> None:
        store.put_items(ItemRow(self.id, item, None, {"id": item, "input": item, "target": None, "metadata": {}})
                        for item in self._grid_items)

    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._grid_items)

    def groups(self) -> dict:
        return {}

    def run(self, steered, item_ids, *, store, replicate) -> dict[str, ItemResult]:
        columns = self._baseline[replicate] if not steered.recipe.entries else self._settings[
            self._recipes[steered.recipe.id]]
        results = {}
        for item_id in item_ids:
            index = self._grid_items[item_id]
            value = float(columns["values"][index])
            flagged = "degenerate" in columns and bool(columns["degenerate"][index])
            results[item_id] = ItemResult(
                item_id=item_id, value=value, status="ok", tokens=int(columns["tokens"][index]),
                completion="" if flagged else STAND_IN, scores={"score": value}, rollout_key=None, generated=True,
            )
        return results


class _Bundle:
    def save(self, path) -> Path:
        Path(path).write_text("replay", encoding="utf-8")
        return Path(path)


class _Pipeline:
    def to_spipe(self, **options) -> _Bundle:
        return _Bundle()


class ReplayFactory:
    """A factory whose steered pipelines run nothing; each recipe resolves to `"resolved-<recipe id>"`."""

    def __init__(self, protocol: Protocol) -> None:
        self.protocol = protocol

    def prepare(self, recipes) -> None:
        pass

    @contextmanager
    def steered(self, recipe):
        yield Steered(_Pipeline(), recipe, f"resolved-{recipe.id}", self.protocol)

    def release(self) -> None:
        pass


def load_export(path: str | Path) -> dict:
    """Read an export written by the `export` command."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load(path: str) -> None:
    global _DATA
    _DATA = load_export(path)


def _differences(data: dict, name: str) -> np.ndarray:
    exported = data["benchmarks"][name]
    baseline = float(np.mean(exported["baseline"][0]["values"]))
    return np.asarray([float(np.mean(setting["values"])) - baseline for setting in exported["settings"]])


def truth(data: dict) -> list[dict]:
    """The true loss and gain of every setting of an export, as means over all grid items.

    Args:
        data: The export, as returned by `load_export`.

    Returns:
        One row per setting, in export order. Its keys are:

            - `"layer"`, `"multiplier"`: The setting.
            - `"loss"`: The baseline's GSM8K accuracy minus the setting's accuracy.
            - `"gain"`: The setting's mean formality score minus the baseline's mean.
    """
    loss, gain = -_differences(data, "capability"), _differences(data, "behavior")
    return [{"layer": setting["caa.layer_id"], "multiplier": setting["caa.multiplier"], "loss": float(loss[index]),
             "gain": float(gain[index])} for index, setting in enumerate(data["settings"])]


@dataclass(frozen=True)
class World:
    """The benchmarks, pools, protocol, and factory of one replay.

    Attributes:
        gsm8k: The capability benchmark (named `"capability"`).
        formality: The objective (named `"behavior"`).
        pools: The pool split of each benchmark, by benchmark id.
        protocol: The protocol of the replay (greedy decoding and no model).
        factory: The pipeline factory, whose steered pipelines run nothing.
        template: The grid's CAA template, with `layer_id` unordered and `multiplier` ordered.
        recipes: Mapping from recipe id to the setting index of the export.
        held_out: The grid item index of every held-out item of `gsm8k`, in pool order.
    """

    gsm8k: ReplayBenchmark
    formality: ReplayBenchmark
    pools: dict[str, PoolSplit]
    protocol: Protocol
    factory: ReplayFactory
    template: Template
    recipes: dict[str, int]
    held_out: np.ndarray


def build_world(data: dict, *, seed: int) -> World:
    """Draw the pools of one replay from the grid's items and build its benchmarks, protocol, and factory.

    Args:
        data: The export, as returned by `load_export`.
        seed: The seed of the draws.

    Returns:
        The world of the replay.
    """
    settings_list = data["settings"]
    layers = sorted({setting["caa.layer_id"] for setting in settings_list})
    multipliers = sorted({setting["caa.multiplier"] for setting in settings_list})
    template = _template(layers, multipliers)
    recipes = {template.bind({"caa.layer_id": setting["caa.layer_id"], "caa.multiplier": setting["caa.multiplier"]})
               .recipe().id: index for index, setting in enumerate(settings_list)}
    rng = np.random.default_rng(seed)
    benchmarks, pools, held_out_draws = {}, {}, {}
    for name, (n_search, n_held_out) in POOLS.items():
        size = len(data["benchmarks"][name]["items"])
        search, held_out = rng.integers(0, size, n_search), rng.integers(0, size, n_held_out)
        held_out_draws[name] = held_out
        grid_items = {f"{name}-s{index:05d}": int(item) for index, item in enumerate(search)}
        grid_items.update({f"{name}-x{index:05d}": int(item) for index, item in enumerate(held_out)})
        benchmark = ReplayBenchmark(name, grid_items, data["benchmarks"][name], recipes)
        benchmarks[name] = benchmark
        pools[benchmark.id] = PoolSplit(tuple(grid_items)[:n_search], tuple(grid_items)[n_search:])
    protocol = Protocol.create("replay-world", generate_defaults={"temperature": 0}, seed=0)
    return World(
        gsm8k=benchmarks["capability"], formality=benchmarks["behavior"], pools=pools, protocol=protocol,
        factory=ReplayFactory(protocol), template=template, recipes=recipes, held_out=held_out_draws["capability"],
    )


def _held_out_passes(data: dict, held_out: np.ndarray, setting: int, alpha: float) -> bool:
    exported = data["benchmarks"]["capability"]
    diffs = np.asarray(exported["settings"][setting]["values"])[held_out] - np.asarray(
        exported["baseline"][0]["values"])[held_out]
    return Paired.from_diffs(diffs).passes(MARGIN, alpha)


def _rejection_fields(archive, settings_list: list[dict], loss: np.ndarray, gain: np.ndarray, *, room: float,
                      min_gain: float) -> dict:
    """The recorded candidates, the recoverable ones, those of them rejected, and the rejections per reason."""
    recoverable_layers = {setting["caa.layer_id"] for index, setting in enumerate(settings_list)
                          if loss[index] <= room and gain[index] >= min_gain}
    results = archive.results()
    recoverable = [result for result in results
                   if archive.candidate(result.candidate_id).values["caa.layer_id"] in recoverable_layers]
    return {
        "candidates": len(results),
        "recoverable": len(recoverable),
        "false_rejections": sum(result.status == "rejected" for result in recoverable),
        "rejections": dict(Counter(result.reason for result in results if result.status == "rejected")),
    }


def replay(seed: int, options: dict) -> dict:
    """Run one study in the replay world and return its measures."""
    data = _DATA
    settings_list = data["settings"]
    world = build_world(data, seed=seed)
    recipes = world.recipes
    searched = frozenset({"caa.multiplier"}) if options["searched"] else frozenset()
    settings = Settings(round_candidates=options["round_candidates"])
    if options["ladder_power"] is not None:
        settings = dataclasses.replace(settings, ladder_power=options["ladder_power"])
    study = Study(
        name=f"replay-{seed}", protocol=world.protocol, space={"caa": world.template}, objective=world.formality,
        constraints=(Constraint(world.gsm8k, MARGIN),), pools=world.pools, settings=settings, searched=searched,
        monotone=frozenset() if options["searched"] else frozenset({"caa.multiplier"}),
    )
    loss, gain = -_differences(data, "capability"), _differences(data, "behavior")
    factory = world.factory
    with tempfile.TemporaryDirectory(dir=options["scratch"]) as scratch:
        store = Store.open(Path(scratch) / "store")
        if options["proposer"] == "random":
            proposer = RandomProposer(study=study, seed=seed)
        else:
            proposer = GridProposer(study=study)
        archive = run_study(
            study=study,
            phases=[SearchPhase(engine=proposer)],
            store=store,
            factory=factory,
            max_generations=options["max_generations"],
        )
        generations = sum(version.generations for result in archive.results()
                          for version in archive.versions(result.candidate_id))
        with StudySession(study=study, store=store, factory=factory) as session:
            session.start()
            capability = world.gsm8k
            variance = design_variance(study, store, capability)
            last = block_sizes(study, store, capability)[1]
            n_held_out = len(usable(study, store, capability, "held_out"))
            band = guard_band(math.sqrt(variance / last), math.sqrt(variance / n_held_out), alpha=study.settings.alpha,
                              power=study.settings.power)
            finalists = session.plan_finalists()
            certification = certify(session=session, bundle_dir=Path(scratch) / "bundles")
        points = archive.points()
        rejection_fields = None if options["searched"] else _rejection_fields(
            archive, settings_list, loss, gain, room=MARGIN - band, min_gain=study.settings.min_gain)
        store.close()

    def setting_of(resolved_id: str) -> int:
        return recipes[resolved_id.removeprefix("resolved-")]

    inside = [index for index in range(len(settings_list)) if loss[index] <= MARGIN]
    best_inside = max((gain[index] for index in inside), default=0.0)
    certified = [setting_of(resolved) for resolved in certification.certified]
    accepted = [setting_of(result.operating.resolved_id) for result in points]
    alpha = study.settings.alpha
    picks = []
    if not options["searched"]:
        for result in points:
            pick = setting_of(result.operating.resolved_id)
            layer = settings_list[pick]["caa.layer_id"]
            room = [index for index, setting in enumerate(settings_list)
                    if setting["caa.layer_id"] == layer and loss[index] <= MARGIN - band]
            best = max((gain[index] for index in room), default=None)
            picks.append({"layer": layer, "multiplier": settings_list[pick]["caa.multiplier"],
                          "gain_share": None if not best or best <= 0 else float(gain[pick] / best)})
    winner = max(points, key=lambda result: (result.fitness, result.candidate_id)) if points else None
    winner_bias = None
    if winner is not None:
        table = winner.operating.capability[study.constraints[0].benchmark.name]
        winner_bias = float((-table.mean - loss[setting_of(winner.operating.resolved_id)]) / max(table.se(), 1e-12))
    return {
        "seed": seed, "band": band, "design_variance": variance, "last_look": last, "generations": generations,
        "finalists": len(finalists), "certified": len(certification.certified), "failed_at": certification.failed_at,
        "over_margin_certified": any(loss[index] > MARGIN for index in certified),
        "first_finalist_passed": None if not finalists else certification.failed_at != 0,
        "best_certified_gain_share": (max((gain[index] for index in certified), default=0.0) / best_inside
                                      if best_inside > 0 else None),
        "best_accepted_gain_inside": max((gain[index] for index in accepted if loss[index] <= MARGIN), default=0.0),
        "accepted_held_out_passes": [bool(_held_out_passes(data, world.held_out, index, alpha)) for index in accepted],
        "picks": picks,
        "winner_loss_bias_in_se": winner_bias,
        **(rejection_fields or dict.fromkeys(("candidates", "recoverable", "false_rejections", "rejections"))),
    }


def _mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def run(arguments) -> None:
    """Run the replays and write the report."""
    _load(arguments.data)
    data = _DATA
    options = {"searched": arguments.searched, "round_candidates": arguments.round_candidates,
               "max_generations": arguments.max_generations, "proposer": arguments.proposer,
               "ladder_power": arguments.ladder_power, "scratch": arguments.scratch}
    seeds = range(arguments.start_seed, arguments.start_seed + arguments.seeds)
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_load, initargs=(arguments.data,)) as executor:
        outcomes = list(executor.map(replay, seeds, [options] * len(seeds)))
    loss, gain = -_differences(data, "capability"), _differences(data, "behavior")
    settings_list = data["settings"]
    by_layer: dict = {}
    for outcome in outcomes:
        for pick in outcome["picks"]:
            by_layer.setdefault(str(pick["layer"]), []).append(pick["multiplier"])
    bands = [outcome["band"] for outcome in outcomes]
    layers = {}
    for layer in sorted({setting["caa.layer_id"] for setting in settings_list}):
        indices = [index for index, setting in enumerate(settings_list) if setting["caa.layer_id"] == layer]
        layers[str(layer)] = {
            "largest_truly_feasible": max((settings_list[index]["caa.multiplier"] for index in indices
                                           if loss[index] <= MARGIN), default=None),
            "largest_inside_room": max((settings_list[index]["caa.multiplier"] for index in indices
                                        if loss[index] <= MARGIN - float(np.mean(bands))), default=None),
            "chosen": sorted(by_layer.get(str(layer), [])),
        }
    passes = [passed for outcome in outcomes for passed in outcome["accepted_held_out_passes"]]
    counted = [outcome for outcome in outcomes if outcome["candidates"] is not None]
    recoverable = sum(outcome["recoverable"] for outcome in counted)
    rejections: Counter = Counter()
    for outcome in counted:
        rejections.update(outcome["rejections"])
    report = {
        "options": {key: value for key, value in options.items() if key != "scratch"},
        "seeds": len(outcomes), "margin": MARGIN, "band_mean": float(np.mean(bands)),
        "design_variance_mean": float(np.mean([outcome["design_variance"] for outcome in outcomes])),
        "truth": truth(data),
        "over_margin_certified_share": _mean(outcome["over_margin_certified"] for outcome in outcomes),
        "first_finalist_pass_share": _mean(outcome["first_finalist_passed"] for outcome in outcomes),
        "best_certified_gain_share": _mean(outcome["best_certified_gain_share"] for outcome in outcomes),
        "best_accepted_gain_inside": _mean(outcome["best_accepted_gain_inside"] for outcome in outcomes),
        "accepted_held_out_pass_share": _mean(passes),
        "pick_gain_share": _mean(pick["gain_share"] for outcome in outcomes for pick in outcome["picks"]),
        "winner_loss_bias_in_se": _mean(outcome["winner_loss_bias_in_se"] for outcome in outcomes),
        "generations_mean": _mean(outcome["generations"] for outcome in outcomes),
        "certified_mean": _mean(outcome["certified"] for outcome in outcomes),
        "finalists_mean": _mean(outcome["finalists"] for outcome in outcomes),
        "candidates_mean": _mean(outcome["candidates"] for outcome in counted),
        "generations_per_candidate": _mean(outcome["generations"] / outcome["candidates"] for outcome in counted
                                           if outcome["candidates"]),
        "false_rejection_share": (sum(outcome["false_rejections"] for outcome in counted) / recoverable
                                  if recoverable else None),
        "rejections": dict(sorted(rejections.items())) if counted else None,
        "layers": layers,
        "outcomes": outcomes,
    }
    Path(arguments.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("%d replays: over-margin certified %.3f, first finalist passed %s, report written to %s",
                len(outcomes), report["over_margin_certified_share"], report["first_finalist_pass_share"],
                arguments.output)


def _bound(differences: list[float]) -> float | None:
    """The one-sided 95% lower bound of the mean of per-seed differences, or None with fewer than two."""
    if len(differences) < 2:
        return None
    values = np.asarray(differences, dtype=float)
    return float(values.mean() - BOUND_Z * values.std(ddof=1) / math.sqrt(len(values)))


def _value_differences(first: dict, second: dict) -> list[float]:
    """The per-seed differences in `best_certified_gain_share`, first minus second."""
    return [first[seed]["best_certified_gain_share"] - second[seed]["best_certified_gain_share"] for seed in first
            if first[seed]["best_certified_gain_share"] is not None
            and second[seed]["best_certified_gain_share"] is not None]


def _false_rejection_differences(first: dict, second: dict) -> list[float]:
    """The per-seed differences in false-rejection share, first minus second, over seeds where both have some."""
    return [first[seed]["false_rejections"] / first[seed]["recoverable"]
            - second[seed]["false_rejections"] / second[seed]["recoverable"] for seed in first
            if first[seed]["recoverable"] and second[seed]["recoverable"]]


def _validity_limit(seeds: int, rate: float) -> int:
    """The one-sided 95% limit of a binomial count over `seeds` replays at a true rate of `rate`."""
    total = 0.0
    for count in range(seeds + 1):
        total += math.comb(seeds, count) * rate ** count * (1 - rate) ** (seeds - count)
        if total >= VALIDITY_QUANTILE:
            return count
    return seeds


def compare(arguments) -> None:
    """Compare the report of today's rule with the reports of challengers and apply the rule of choice."""
    names = [Path(path).stem for path in arguments.reports]
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in arguments.reports]
    by_seed = [{outcome["seed"]: outcome for outcome in report["outcomes"]} for report in reports]
    if any(set(seeds) != set(by_seed[0]) for seeds in by_seed[1:]):
        raise SystemExit("The reports were run on different seeds.")
    settings = Settings()
    limit = _validity_limit(len(by_seed[0]), settings.alpha)
    over = [sum(bool(outcome["over_margin_certified"]) for outcome in seeds.values()) for seeds in by_seed]
    logger.info("%-12s %8s %8s %8s %10s %10s %10s", "report", "over", "held_out", "value", "gen/cand", "false rej",
                "candidates")
    for name, report, count in zip(names, reports, over):
        logger.info("%-12s %8d %8.3f %8.3f %10.1f %10.3f %10.2f", name, count, report["accepted_held_out_pass_share"],
                    report["best_certified_gain_share"], report["generations_per_candidate"],
                    report["false_rejection_share"], report["candidates_mean"])
    today, today_seeds = reports[0], by_seed[0]
    if over[0] > limit:
        raise SystemExit(f"Today's rule ({names[0]}) certifies a setting over the margin in {over[0]} replays, above "
                         f"the limit of {limit}; this is a fault in certification.")
    promise = min(settings.power, today["accepted_held_out_pass_share"] - PROMISE_SLACK)
    qualifying = []
    for index in range(1, len(reports)):
        name, report, seeds = names[index], reports[index], by_seed[index]
        value_bound = _bound(_value_differences(seeds, today_seeds))
        rejection_bound = _bound(_false_rejection_differences(today_seeds, seeds))
        cost_share = report["generations_per_candidate"] / today["generations_per_candidate"]
        gates = {
            "validity": over[index] <= limit,
            "promise": report["accepted_held_out_pass_share"] >= promise,
            "value": value_bound is not None and value_bound > -VALUE_TOLERANCE,
        }
        benefits = {
            "better value": value_bound is not None and value_bound > 0,
            "cheaper, with fewer false rejections": (cost_share <= COST_SHARE and rejection_bound is not None
                                                      and rejection_bound > 0),
        }
        logger.info("%s: validity %d of %d replays over the margin (limit %d): %s", name, over[index],
                    len(seeds), limit, gates["validity"])
        logger.info("%s: promise held-out pass share %.3f (needs %.3f): %s", name,
                    report["accepted_held_out_pass_share"], promise, gates["promise"])
        logger.info("%s: value bound of the per-seed difference %s (needs > %.2f): %s", name,
                    "n/a" if value_bound is None else f"{value_bound:.4f}", -VALUE_TOLERANCE, gates["value"])
        logger.info("%s: benefit better value (bound > 0): %s", name, benefits["better value"])
        logger.info("%s: benefit cheaper: generations per candidate %.3f of today's (needs <= %.2f), false-rejection "
                    "bound %s (needs > 0): %s", name, cost_share, COST_SHARE,
                    "n/a" if rejection_bound is None else f"{rejection_bound:.4f}",
                    benefits["cheaper, with fewer false rejections"])
        if all(gates.values()) and any(benefits.values()):
            qualifying.append(index)
    if not qualifying:
        logger.info("choice: %s (no challenger qualifies)", names[0])
        return
    best = max(qualifying, key=lambda index: reports[index]["best_certified_gain_share"])
    choice = best
    for index in qualifying:
        bound = _bound(_value_differences(by_seed[best], by_seed[index]))
        if bound is None or bound <= 0:
            choice = index
            break
    logger.info("highest mean value among the qualifying challengers: %s", names[best])
    logger.info("choice: %s", names[choice])


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("stores", nargs="+")
    exporting.add_argument("--output", default="replay_grid.json")
    running = commands.add_parser("run")
    running.add_argument("data")
    running.add_argument("--seeds", type=int, default=200)
    running.add_argument("--start-seed", type=int, default=0)
    running.add_argument("--workers", type=int, default=1)
    running.add_argument("--searched", action="store_true", help="Make the multiplier a proposer axis.")
    running.add_argument("--round-candidates", type=int, default=5)
    running.add_argument("--max-generations", type=int, default=None)
    running.add_argument("--proposer", default="grid", choices=("grid", "random"))
    running.add_argument("--ladder-power", type=float, default=None, help="The settings' ladder_power.")
    running.add_argument("--scratch", default=None, help="The directory of the replays' temporary stores.")
    running.add_argument("--output", default="replay_report.json")
    comparing = commands.add_parser("compare")
    comparing.add_argument("reports", nargs="+", help="Today's report, then the challengers' reports.")
    arguments = parser.parse_args()
    if arguments.command == "export":
        export(arguments)
    elif arguments.command == "run":
        run(arguments)
    else:
        if len(arguments.reports) < 2:
            parser.error("compare needs today's report and at least one challenger's report.")
        compare(arguments)


if __name__ == "__main__":
    main()
