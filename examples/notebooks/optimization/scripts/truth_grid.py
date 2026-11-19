"""The truth grid of the optimization experiments: every setting of a CAA grid on every GSM8K search item.

The `collect` command measures the unsteered baseline twice (replicates 0 and 1, in different batch orders) and every
(layer, multiplier) setting of one CAA vector once on every search item of the GSM8K pool split, and writes the
settings-by-items score matrix into the store. No sealed item is used. Settings can be split into shards that run in
separate processes, each with its own store. One shard measures the baseline and the others pass `--skip-baseline`,
since a deterministic protocol gives every shard the same baseline. The `analyze` command reads the stores and reports
the noise floor, the change rate of settings whose loss is near `margin - band`, the departures of each layer's loss
curve from its isotonic fit, and the setup time of the pipeline factory against the generation time.

Run it from the repository root, since the GSM8K task is referenced by its relative path.

usage:
    python examples/notebooks/optimization/scripts/truth_grid.py collect <store> [--shard I --shards N] [--model M]
    python examples/notebooks/optimization/scripts/truth_grid.py analyze <store> [<store> ...] [--margin M] [--output F]
"""
import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import BASELINE, Choice, Template, entry, expand
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.state_control.caa.control import CAA
from steerability.evaluation.items import InspectBenchmark, evaluate_items
from steerability.evaluation.provider import ProviderOptions
from steerability.optimization import split_items
from steerability.optimization.measurement.stats import Paired, confirm_looks, isotonic_increasing, release_band

logger = logging.getLogger("truth_grid")

TASK = "examples/notebooks/optimization/gsm8k_all.py@gsm8k_all"  # relative to the repository root
LAYERS = (6, 8, 10, 12, 14, 16, 18, 20, 22, 23)
MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SEALED = 5000

FORMAL = [
    "I regret to inform you that I will be delayed this evening.",
    "Please be advised that the meeting has been rescheduled to Thursday.",
    "I would be grateful if you could return the charger at your convenience.",
    "I am pleased to confirm that the arrangements remain unchanged.",
    "Kindly note that the event will commence at six o'clock.",
    "I wish to express my sincere appreciation for your assistance.",
    "It is with regret that I must decline your kind invitation.",
    "I would like to inform you that your parcel has been delivered.",
]
CASUAL = [
    "hey gonna be late tonight lol, sorry!!",
    "yo the meeting got moved to thursday btw",
    "can u bring back my charger whenever? thx!",
    "yep we're still on, super pumped!!",
    "party starts at 6, don't be late haha",
    "dude thanks so much, you're awesome!",
    "ugh can't make it, sorry guys",
    "hey ur package showed up at my place!",
]
TASKS = [
    "Tell a friend you will be late tonight.", "Tell a friend the meeting moved to Thursday.",
    "Ask a friend to return your charger.", "Confirm the plans are still on.", "Say when the party starts.",
    "Thank a friend for their help.", "Decline an invitation.", "Tell a friend their package arrived.",
]
PAIRS = {"prompts": TASKS, "positives": FORMAL, "negatives": CASUAL}


def _protocol(model: str, backend: str, batch_size: int, dtype: str | None) -> Protocol:
    spec = None
    if backend == "vllm":
        spec = BackendSpec(kind="vllm", model=model, options={
            "hook_plugin": True, "engine_kwargs": {"gpu_memory_utilization": 0.8, "max_model_len": 4096},
        })
    return Protocol.create(model, backend=spec, provider=ProviderOptions(max_batch_size=batch_size),
                           hf_model_kwargs={"dtype": dtype} if dtype else {}, generate_defaults={"temperature": 0},
                           seed=0)


def grid(layers=LAYERS, multipliers=MULTIPLIERS) -> Template:
    """The CAA grid: one vector (mean difference of the formality pairs) at every layer and multiplier."""
    return Template((entry(
        CAA, data=PAIRS, train_spec={"method": "mean_diff", "accumulate": "all"},
        layer_id=Choice(tuple(layers), ordered=True), multiplier=Choice(tuple(multipliers), ordered=True),
    ),))


def collect(arguments) -> None:
    """Measure the baseline and the settings of one shard on every search item."""
    store = Store.open(arguments.store)
    benchmark = InspectBenchmark(TASK, generate={"max_tokens": arguments.max_tokens})
    benchmark.prepare(store)
    search = list(split_items(benchmark, sealed=SEALED, seed=0).search)
    if arguments.limit:
        search = search[:arguments.limit]
    protocol = _protocol(arguments.model, arguments.backend, arguments.batch_size, arguments.dtype)
    layers = [int(value) for value in arguments.layers.split(",")] if arguments.layers else LAYERS
    multipliers = [float(value) for value in arguments.multipliers.split(",")] if arguments.multipliers else MULTIPLIERS
    points = list(expand(grid(layers, multipliers)))[arguments.shard::arguments.shards]
    factory = PipelineFactory(protocol, store=store)
    index_path = Path(arguments.store) / f"grid_shard_{arguments.shard}.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"points": {}}
    try:
        if not arguments.skip_baseline:
            with factory.steered(BASELINE) as steered:
                for replicate in (0, 1):
                    started = time.monotonic()
                    evaluate_items(steered, benchmark, search, store=store, replicate=replicate)
                    logger.info("baseline replicate %d: %.1f s", replicate, time.monotonic() - started)
                index["baseline"] = steered.resolved_id
        factory.prepare([point.recipe for point in points])
        for point in points:
            started = time.monotonic()
            with factory.steered(point.recipe) as steered:
                rows = evaluate_items(steered, benchmark, search, store=store)
            index["points"][point.recipe.id] = {
                "position": point.assignment, "resolved_id": steered.resolved_id,
                "seconds": time.monotonic() - started, "generated": sum(row.generated for row in rows.values()),
            }
            logger.info("%s: %.1f s", point.assignment, time.monotonic() - started)
            index_path.write_text(json.dumps(index, indent=2))
    finally:
        index.update({"benchmark_id": benchmark.id, "protocol_id": protocol.id, "items": search,
                      "factory": vars(factory.stats)})
        index_path.write_text(json.dumps(index, indent=2))
        factory.release()


def _values(store: Store, protocol_id: str, resolved_id: str, benchmark_id: str, items, replicate=0) -> np.ndarray:
    rows = store.get_evidence(protocol_id, resolved_id, benchmark_id, items, replicate)
    return np.asarray([rows[item].value for item in items], dtype=float)


def analyze(arguments) -> None:
    """Report the noise floor, change rates near the band, monotonicity departures, and setup time."""
    margin = arguments.margin
    look = confirm_looks(margin)[2]
    report: dict = {"margin": margin, "layers": {}, "settings": []}
    setup = generation = 0.0
    shards = [(Store.open(root), json.loads(path.read_text()))
              for root in arguments.stores for path in sorted(Path(root).glob("grid_shard_*.json"))]
    measured = [(store, index) for store, index in shards if "baseline" in index]
    if not measured:
        raise SystemExit("No shard measured the baseline; collect one shard without --skip-baseline.")
    store, index = measured[0]
    items, protocol_id, benchmark_id = index["items"], index["protocol_id"], index["benchmark_id"]
    baseline = _values(store, protocol_id, index["baseline"], benchmark_id, items)
    second = _values(store, protocol_id, index["baseline"], benchmark_id, items, replicate=1)
    report["noise_floor"] = float(np.mean(baseline != second))
    report["baseline_accuracy"] = float(baseline.mean())
    for store, index in shards:
        if (index["items"], index["protocol_id"], index["benchmark_id"]) != (items, protocol_id, benchmark_id):
            raise SystemExit("The shards differ in their items, protocol, or benchmark.")
        setup += index["factory"]["setup_seconds"]
        for record in index["points"].values():
            values = _values(store, protocol_id, record["resolved_id"], benchmark_id, items)
            table = Paired.from_diffs(values - baseline)
            generation += record["seconds"]
            report["settings"].append({**record["position"], "loss": -table.mean, "se": table.se(),
                                       "change_rate": table.change_rate, "n": table.n})
    floor = math.sqrt(max(report.get("noise_floor", 0.0), 0.02))
    band = release_band(Paired(look, 0.0, 0.1 * look, int(0.1 * look)), n_sealed=SEALED, sd_floor=floor, alpha=0.05,
                        power=0.80)
    near = [setting for setting in report["settings"] if abs(setting["loss"] - (margin - band)) <= 0.01]
    report["band"] = band
    report["change_rate_near_margin_minus_band"] = float(np.mean([s["change_rate"] for s in near])) if near else None
    departures = total = 0
    for layer in sorted({setting["caa.layer_id"] for setting in report["settings"]}):
        curve = sorted((s for s in report["settings"] if s["caa.layer_id"] == layer),
                       key=lambda setting: setting["caa.multiplier"])
        losses = [setting["loss"] for setting in curve]
        fitted = isotonic_increasing(losses, [setting["n"] for setting in curve])
        off = [abs(loss - fit) > 2 * max(setting["se"], 1e-12) for loss, fit, setting in zip(losses, fitted, curve)]
        departures += sum(off)
        total += len(off)
        report["layers"][str(layer)] = {"losses": losses, "isotonic": fitted.tolist(), "departures": int(sum(off))}
    report["monotone_departure_share"] = departures / total if total else None
    report["setup_seconds"] = setup
    report["generation_seconds"] = generation
    report["setup_share"] = setup / generation if generation else None
    Path(arguments.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("noise floor %s, monotone departures %s, setup share %s; report written to %s",
                report.get("noise_floor"), report["monotone_departure_share"], report["setup_share"], arguments.output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    collecting = commands.add_parser("collect")
    collecting.add_argument("store")
    collecting.add_argument("--shard", type=int, default=0)
    collecting.add_argument("--shards", type=int, default=1)
    collecting.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    collecting.add_argument("--backend", default="huggingface", choices=("huggingface", "vllm"))
    collecting.add_argument("--batch-size", type=int, default=64)
    collecting.add_argument("--max-tokens", type=int, default=384)
    collecting.add_argument("--limit", type=int, default=0, help="Use only the first items of the search pool.")
    collecting.add_argument("--dtype", default=None, help="The weights dtype, e.g., float32.")
    collecting.add_argument("--layers", default="", help="Comma-separated layers in place of the default grid.")
    collecting.add_argument("--multipliers", default="", help="Comma-separated multipliers in place of the default.")
    collecting.add_argument("--skip-baseline", action="store_true",
                            help="Measure only the settings; the analysis reads the baseline from another shard.")
    analyzing = commands.add_parser("analyze")
    analyzing.add_argument("stores", nargs="+")
    analyzing.add_argument("--margin", type=float, default=0.03)
    analyzing.add_argument("--output", default="truth_grid_report.json")
    arguments = parser.parse_args()
    if arguments.command == "collect":
        collect(arguments)
    else:
        analyze(arguments)


if __name__ == "__main__":
    main()
