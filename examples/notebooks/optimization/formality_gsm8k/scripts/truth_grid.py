"""The truth grid of the optimization experiments: every setting of a CAA grid on every search item of both benchmarks.

The `collect` command measures the unsteered baseline twice (replicates 0 and 1, in different batch orders) and every
(layer, multiplier) setting of one CAA vector once on every search item of the GSM8K pool split and of the formality
objective's pool split, and writes the settings-by-items score matrices into the store. No sealed item is used.
Settings can be split into shards that run in separate processes, each with its own store. One shard measures the
baseline and the others pass `--skip-baseline`, since a deterministic protocol gives every shard the same baseline.
`--benchmarks` selects the benchmarks a run measures, and stored rows are reused.

The `analyze` command reads the stores and reports the noise floor and the variance of each benchmark's baseline, the
design variance, last look, and guard band of GSM8K at `--margin`, and per setting the GSM8K loss, its paired
variance, and the formality gain with its per-item standard deviation. It also reports the ratio of the paired
variance to the baseline variance for settings within one point of `margin - band`, the departures of each layer's
loss curve from its isotonic fit, and the setup time of the pipeline factory against the generation time.

Run it from the repository root, since the GSM8K task is referenced by its relative path.

usage:
    python examples/notebooks/optimization/formality_gsm8k/scripts/truth_grid.py collect \
        <store> [--shard I --shards N] [--model M]
    python examples/notebooks/optimization/formality_gsm8k/scripts/truth_grid.py analyze \
        <store> [<store> ...] [--margin M] [--output F]
"""
import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from formality import OBJECTIVE_SEALED, PAIRS, formality_benchmark

from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.protocol import Protocol, ProviderOptions
from steerability.algorithms.core.recipe import BASELINE, Choice, Template, entry, expand
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.state_control.caa.control import CAA
from steerability.evaluation.items import InspectBenchmark, evaluate_items
from steerability.optimization import Settings, split_items
from steerability.optimization.measurement.stats import (
    Paired,
    floor_variance,
    guard_band,
    isotonic_increasing,
    last_look,
)

logger = logging.getLogger("truth_grid")

TASK = "examples/notebooks/optimization/gsm8k_all.py@gsm8k_all"  # relative to the repository root
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = (6, 8, 10, 12, 14, 16, 18, 20, 22, 23)
MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SEALED = 5000


def protocol(model: str = MODEL, *, backend: str = "huggingface", batch_size: int = 64,
             dtype: str | None = "float32") -> Protocol:
    """The protocol of the grid: greedy decoding with the weights in `dtype` and batches of `batch_size`."""
    spec = None
    if backend == "vllm":
        engine_kwargs = {"gpu_memory_utilization": 0.8, "max_model_len": 4096}
        if dtype:
            engine_kwargs["dtype"] = dtype
        spec = BackendSpec(kind="vllm", model=model, options={"hook_plugin": True, "engine_kwargs": engine_kwargs})
    return Protocol.create(model, backend=spec, provider=ProviderOptions(max_batch_size=batch_size),
                           hf_model_kwargs={"dtype": dtype} if dtype and backend != "vllm" else {},
                           generate_defaults={"temperature": 0}, seed=0)


def gsm8k_benchmark(max_tokens: int = 384) -> InspectBenchmark:
    """The GSM8K constraint over the train and test splits."""
    return InspectBenchmark(TASK, generate={"max_tokens": max_tokens})


def grid(layers=LAYERS, multipliers=MULTIPLIERS) -> Template:
    """The CAA grid: one vector (mean difference of the formality pairs) at every layer and multiplier."""
    return Template((entry(
        CAA, data=PAIRS, train_spec={"method": "mean_diff", "accumulate": "all"},
        layer_id=Choice(tuple(layers), ordered=True), multiplier=Choice(tuple(multipliers), ordered=True),
    ),))


def _measure(steered, benchmark, items, store, record: dict, prefix: str, replicate: int = 0) -> None:
    started = time.monotonic()
    rows = evaluate_items(steered, benchmark, items, store=store, replicate=replicate)
    generated = sum(row.generated for row in rows.values())
    if generated or f"{prefix}seconds" not in record:
        record[f"{prefix}seconds"] = time.monotonic() - started
        record[f"{prefix}generated"] = generated


def collect(arguments) -> None:
    """Measure the baseline and the settings of one shard on every search item of the selected benchmarks."""
    store = Store.open(arguments.store)
    chosen = {name.strip() for name in arguments.benchmarks.split(",") if name.strip()}
    gsm8k = gsm8k_benchmark(arguments.max_tokens)
    objective = formality_benchmark()
    gsm8k.prepare(store)
    objective.prepare(store)
    search = list(split_items(gsm8k, sealed=SEALED, seed=0).search)
    objective_search = list(split_items(objective, sealed=OBJECTIVE_SEALED, seed=0).search)
    if arguments.limit:
        search, objective_search = search[:arguments.limit], objective_search[:arguments.limit]
    measured = [(name, benchmark, items, prefix) for name, benchmark, items, prefix in (
        ("gsm8k", gsm8k, search, ""), ("formality", objective, objective_search, "objective_"),
    ) if name in chosen]
    grid_protocol = protocol(arguments.model, backend=arguments.backend, batch_size=arguments.batch_size,
                             dtype=arguments.dtype or None)
    layers = [int(value) for value in arguments.layers.split(",")] if arguments.layers else LAYERS
    multipliers = [float(value) for value in arguments.multipliers.split(",")] if arguments.multipliers else MULTIPLIERS
    points = list(expand(grid(layers, multipliers)))[arguments.shard::arguments.shards]
    factory = PipelineFactory(grid_protocol, store=store)
    index_path = Path(arguments.store) / f"grid_shard_{arguments.shard}.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"points": {}}
    try:
        if not arguments.skip_baseline:
            with factory.steered(BASELINE) as steered:
                baseline = index.setdefault("baseline_record", {})
                for _, benchmark, items, prefix in measured:
                    for replicate in (0, 1):
                        _measure(steered, benchmark, items, store, baseline, f"{prefix}{replicate}_", replicate)
                        logger.info("%s baseline replicate %d: %.1f s", benchmark.name, replicate,
                                    baseline[f"{prefix}{replicate}_seconds"])
                index["baseline"] = steered.resolved_id
        factory.prepare([point.recipe for point in points])
        for point in points:
            record = index["points"].setdefault(point.recipe.id, {"position": point.assignment})
            with factory.steered(point.recipe) as steered:
                record["resolved_id"] = steered.resolved_id
                for _, benchmark, items, prefix in measured:
                    _measure(steered, benchmark, items, store, record, prefix)
            logger.info("%s: %s", point.assignment, {key: value for key, value in record.items() if "seconds" in key})
            index_path.write_text(json.dumps(index, indent=2))
    finally:
        index.update({"benchmark_id": gsm8k.id, "protocol_id": grid_protocol.id, "items": search,
                      "objective_id": objective.id, "objective_items": objective_search,
                      "factory": vars(factory.stats)})
        index_path.write_text(json.dumps(index, indent=2))
        factory.release()


def values(store: Store, protocol_id: str, resolved_id: str, benchmark_id: str, items, replicate=0) -> np.ndarray:
    """The stored values of a resolved recipe on items, in item order."""
    rows = store.get_evidence(protocol_id, resolved_id, benchmark_id, items, replicate)
    return np.asarray([rows[item].value for item in items], dtype=float)


def shards(roots) -> list[tuple[Store, dict]]:
    """The stores and grid indexes of every shard under the given store directories."""
    found = [(Store.open(root), json.loads(path.read_text()))
             for root in roots for path in sorted(Path(root).glob("grid_shard_*.json"))]
    if not found:
        raise SystemExit("No grid_shard_*.json index found under the given stores.")
    first = found[0][1]
    for _, index in found:
        if (index["items"], index["protocol_id"], index["benchmark_id"]) != (
                first["items"], first["protocol_id"], first["benchmark_id"]):
            raise SystemExit("The shards differ in their items, protocol, or benchmark.")
    return found


def analyze(arguments) -> None:
    """Report the noise floors, the design variance and band, the per-setting tables, monotonicity, and setup time."""
    margin = arguments.margin
    settings = Settings()
    found = shards(arguments.stores)
    measured = [(store, index) for store, index in found if "baseline" in index]
    if not measured:
        raise SystemExit("No shard measured the baseline; collect one shard without --skip-baseline.")
    store, index = measured[0]
    protocol_id, baseline_id = index["protocol_id"], index["baseline"]
    items, benchmark_id = index["items"], index["benchmark_id"]
    objective_items, objective_id = index.get("objective_items"), index.get("objective_id")
    report: dict = {"margin": margin, "layers": {}, "settings": []}

    baseline = values(store, protocol_id, baseline_id, benchmark_id, items)
    second = values(store, protocol_id, baseline_id, benchmark_id, items, replicate=1)
    report["noise_floor"] = float(np.mean(np.abs(baseline - second)))
    report["baseline_accuracy"] = float(baseline.mean())
    report["baseline_variance"] = float(baseline.var())
    variance = max(report["baseline_variance"], floor_variance(margin))
    last = last_look(margin=margin, variance=variance)
    band = guard_band(math.sqrt(variance / last), math.sqrt(variance / SEALED), alpha=settings.alpha,
                      power=settings.power)
    report.update({"design_variance": variance, "last_look": last, "band": band})
    objective_baseline = None
    if objective_items:
        objective_baseline = values(store, protocol_id, baseline_id, objective_id, objective_items)
        objective_second = values(store, protocol_id, baseline_id, objective_id, objective_items, replicate=1)
        report["objective_noise_floor"] = float(np.mean(np.abs(objective_baseline - objective_second)))
        report["objective_baseline_mean"] = float(objective_baseline.mean())

    setup = generation = 0.0
    for shard_store, shard_index in found:
        setup += shard_index["factory"]["setup_seconds"]
        for record in shard_index["points"].values():
            table = Paired.from_diffs(values(shard_store, protocol_id, record["resolved_id"], benchmark_id, items)
                                      - baseline)
            generation += record.get("seconds", 0.0) + record.get("objective_seconds", 0.0)
            setting = {**record["position"], "loss": -table.mean, "se": table.se(), "variance": table.variance,
                       "change_rate": table.change_rate, "n": table.n}
            if objective_baseline is not None and "objective_seconds" in record:
                gains = (values(shard_store, protocol_id, record["resolved_id"], objective_id, objective_items)
                         - objective_baseline)
                setting.update({"gain": float(gains.mean()), "gain_sd": float(gains.std())})
            report["settings"].append(setting)

    near = [setting for setting in report["settings"] if abs(setting["loss"] - (margin - band)) <= 0.01]
    ratios = [setting["variance"] / report["baseline_variance"] for setting in near]
    report["near_margin_minus_band"] = {
        "count": len(near), "variance_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "variance_ratio_min": float(np.min(ratios)) if ratios else None,
        "variance_ratio_max": float(np.max(ratios)) if ratios else None,
    }
    inside = [setting for setting in report["settings"] if setting["loss"] <= margin - band and "gain" in setting]
    report["inside_room"] = {
        "count": len(inside),
        "above_min_gain": sum(setting["gain"] > settings.min_gain for setting in inside),
        "best_gain": max((setting["gain"] for setting in inside), default=None),
        "gain_sd_mean": float(np.mean([setting["gain_sd"] for setting in inside])) if inside else None,
    }

    departures = total = 0
    for layer in sorted({setting["caa.layer_id"] for setting in report["settings"]}):
        curve = sorted((s for s in report["settings"] if s["caa.layer_id"] == layer),
                       key=lambda setting: setting["caa.multiplier"])
        losses = [setting["loss"] for setting in curve]
        fitted = isotonic_increasing(losses, [setting["n"] for setting in curve])
        off = [abs(loss - fit) > 2 * max(setting["se"], 1e-12) for loss, fit, setting in zip(losses, fitted, curve)]
        departures += sum(off)
        total += len(off)
        largest = max((s["caa.multiplier"] for s in curve if s["loss"] <= margin - band), default=None)
        report["layers"][str(layer)] = {"losses": losses, "isotonic": fitted.tolist(), "departures": int(sum(off)),
                                        "largest_multiplier_inside_room": largest}
    report["monotone_departure_share"] = departures / total if total else None
    report["setup_seconds"] = setup
    report["generation_seconds"] = generation
    report["setup_share"] = setup / generation if generation else None
    Path(arguments.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("noise floor %s, design variance %.4f, band %.4f, monotone departures %s; report written to %s",
                report["noise_floor"], variance, band, report["monotone_departure_share"], arguments.output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    collecting = commands.add_parser("collect")
    collecting.add_argument("store")
    collecting.add_argument("--shard", type=int, default=0)
    collecting.add_argument("--shards", type=int, default=1)
    collecting.add_argument("--model", default=MODEL)
    collecting.add_argument("--backend", default="huggingface", choices=("huggingface", "vllm"))
    collecting.add_argument("--batch-size", type=int, default=64)
    collecting.add_argument("--max-tokens", type=int, default=384)
    collecting.add_argument("--limit", type=int, default=0, help="Use only the first items of each search pool.")
    collecting.add_argument("--dtype", default="float32", help="The weights dtype; an empty string loads the default.")
    collecting.add_argument("--benchmarks", default="gsm8k,formality", help="Comma-separated benchmarks to measure.")
    collecting.add_argument("--layers", default="", help="Comma-separated layers in place of the default grid.")
    collecting.add_argument("--multipliers", default="", help="Comma-separated multipliers in place of the default.")
    collecting.add_argument("--skip-baseline", action="store_true",
                            help="Measure only the settings; the analysis reads the baseline from another shard.")
    analyzing = commands.add_parser("analyze")
    analyzing.add_argument("stores", nargs="+")
    analyzing.add_argument("--margin", type=float, default=0.05)
    analyzing.add_argument("--output", default="truth_grid_report.json")
    arguments = parser.parse_args()
    if arguments.command == "collect":
        collect(arguments)
    else:
        analyze(arguments)


if __name__ == "__main__":
    main()
