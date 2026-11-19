"""`SteeringEval`: recipes x trials x benchmarks under one protocol, measured item by item.

A run expands the pipelines mapping into points (`algorithms/core/sweeps.py`), checks backend
support for every point before any model work, fits what the points need once (`PipelineFactory.
prepare`), and then measures every point on every benchmark and trial with `evaluate_items`. The
store keeps fits and per-item evidence, so a second run on the same store generates only what has
no evidence row, and a run with a larger selection generates only the new items.

`save_dir` records the run: `protocol.json` (the protocol's id, identity, and observed facts),
`points.json` (each point's pipeline, recipe id, resolved id, and assignment, the selections, and
the trial count), and the store under `store/` unless the run is given one. `samples_frame` and
`runs_frame` read a finished run from its `save_dir`, and `summarize_runs` aggregates the
per-trial frame into the summary form the plotting layer consumes.
"""
from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas

from steerability.algorithms.core.identity import canonical_json
from steerability.algorithms.core.protocol import Protocol, ProtocolFacts, ProtocolMismatchError
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory, Pipelines, expand_configurations, preflight

if TYPE_CHECKING:
    from steerability.evaluation.items import Benchmark

logger = logging.getLogger(__name__)

PROTOCOL_FILE = "protocol.json"
POINTS_FILE = "points.json"
STORE_DIR = "store"

_LETTER_GRADES = frozenset({"C", "I", "P", "N"})


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_save_dir(save_dir: Path, protocol: Protocol, store: Store) -> None:
    """Refuse a `save_dir` written under another protocol, and copy its recorded facts into the store.

    Raises:
        ValueError: If `save_dir/protocol.json` records a different protocol id.
        ProtocolMismatchError: If its observed facts conflict with the store's.
    """
    path = save_dir / PROTOCOL_FILE
    if not path.exists():
        return
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded.get("id") != protocol.id:
        raise ValueError(
            f"{save_dir} was written under protocol {recorded.get('id')}, not {protocol.id}; start a new save_dir "
            "for this protocol."
        )
    if recorded.get("observed"):
        store.put_protocol(protocol, ProtocolFacts.from_dict(recorded["observed"]))


def _write_protocol(save_dir: Path, protocol: Protocol, store: Store) -> None:
    """Write `save_dir/protocol.json` with the protocol's identity and the facts the store records for it."""
    stored = store.get_protocol(protocol.id)
    _write_json(save_dir / PROTOCOL_FILE, {
        "schema": "protocol/1", "id": protocol.id, "identity": protocol.identity(),
        "observed": stored.observed if stored is not None else None,
    })


class SteeringEval:
    """Evaluate recipes on benchmarks under one protocol, over trials.

    Args:
        protocol: The protocol every point runs under.
        pipelines: Mapping from pipeline name to recipes and templates (`BASELINE` for the
            unsteered baseline).
        benchmarks: The benchmarks; their names must be distinct.
        items: Mapping from benchmark name to the item ids to evaluate. A benchmark without an
            entry evaluates the first `limit` items, or all of them.
        limit: The number of items of a benchmark without an `items` entry, or None for all.
        num_trials: Trials per point; trial `t` is replicate `t` of every item.
        on_unsupported: `"raise"` (default) fails the run with one aggregate error on any
            unsupported point before any model work; `"skip"` runs the supported points.
        device_map: Device placement for in-process loads.
        trust_remote_code: Trust remote code when loading tokenizers.
        progress: Draw a progress bar over the (point, trial, benchmark) cells.

    Raises:
        ValueError: If `num_trials` is below 1, `on_unsupported` is unknown, there are no
            benchmarks, or two benchmarks share a name.
    """

    def __init__(
        self,
        protocol: Protocol,
        pipelines: Pipelines,
        benchmarks: Sequence[Benchmark],
        *,
        items: Mapping[str, Sequence[str]] | None = None,
        limit: int | None = None,
        num_trials: int = 1,
        on_unsupported: Literal["raise", "skip"] = "raise",
        device_map: str | dict | None = "auto",
        trust_remote_code: bool = False,
        progress: bool = True,
    ) -> None:
        if int(num_trials) < 1:
            raise ValueError(f"num_trials must be >= 1; got {num_trials}.")
        if on_unsupported not in ("raise", "skip"):
            raise ValueError(f"on_unsupported must be 'raise' or 'skip'; got {on_unsupported!r}.")
        benchmarks = list(benchmarks)
        if not benchmarks:
            raise ValueError("benchmarks must be non-empty.")
        names = [benchmark.name for benchmark in benchmarks]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Benchmark names must be distinct; duplicated: {duplicates}.")
        unknown = sorted(set(items or {}) - set(names))
        if unknown:
            raise ValueError(f"items refers to unknown benchmark(s) {unknown}.")
        if limit is not None and int(limit) < 1:
            raise ValueError(f"limit must be >= 1; got {limit}.")
        self.protocol = protocol
        self.pipelines = pipelines
        self.benchmarks = benchmarks
        self.items = {name: [str(item) for item in selection] for name, selection in (items or {}).items()}
        self.limit = limit
        self.num_trials = int(num_trials)
        self.on_unsupported = on_unsupported
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.progress = bool(progress)
        self.factory: PipelineFactory | None = None
        self._save_dir: Path | None = None
        self._scratch: tempfile.TemporaryDirectory | None = None

    def _selection(self, benchmark: Benchmark) -> list[str]:
        if benchmark.name in self.items:
            return list(self.items[benchmark.name])
        item_ids = list(benchmark.item_ids())
        return item_ids[: self.limit] if self.limit is not None else item_ids

    def run(self, save_dir: str | Path | None = None, *, store: Store | None = None) -> None:
        """Run every point x trial x benchmark cell, generating only what has no evidence.

        The store is `store` when given, so that sweeps and other callers share fits and evidence,
        else `save_dir/store`. With neither, the run's directory is temporary and is removed when
        this object is discarded.

        Args:
            save_dir: The run directory, or None.
            store: The store, or None.

        Raises:
            RuntimeError: If a point is unsupported and `on_unsupported="raise"`.
            ValueError: If `save_dir` was written under another protocol.
            ProtocolMismatchError: If the loaded model conflicts with the facts recorded for the
                protocol.
        """
        from tqdm.auto import tqdm

        from steerability.evaluation.items import evaluate_items

        if save_dir is None:
            self._scratch = tempfile.TemporaryDirectory(prefix="steering-eval-")
            save_dir = Path(self._scratch.name)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if store is None:
            store = Store.open(save_dir / STORE_DIR)
        _check_save_dir(save_dir, self.protocol, store)
        store.put_protocol(self.protocol)
        _write_protocol(save_dir, self.protocol, store)
        self._save_dir = save_dir

        points = list(expand_configurations(self.pipelines))
        failures: list[str] = []
        supported = []
        for name, point in points:
            messages = preflight([(name, point)], self.protocol)
            if messages:
                failures.extend(messages)
            else:
                supported.append((name, point))
        if failures:
            if self.on_unsupported == "raise":
                raise RuntimeError("Unsupported pipeline configuration(s):\n" + "\n".join(failures))
            for line in failures:
                logger.warning("Skipping unsupported configuration: %s", line)

        for benchmark in self.benchmarks:
            benchmark.prepare(store)
        selections = {benchmark.name: self._selection(benchmark) for benchmark in self.benchmarks}
        records: list[dict[str, Any]] = []

        factory = PipelineFactory(
            self.protocol, store=store, device_map=self.device_map, trust_remote_code=self.trust_remote_code,
        )
        self.factory = factory
        total = len(supported) * self.num_trials * len(self.benchmarks)
        bar = tqdm(total=total, disable=not self.progress, desc="steering eval", unit="cell", dynamic_ncols=True)
        try:
            pending = [(name, point) for name, point in supported if not self._complete(point, store, selections)]
            factory.prepare(dict.fromkeys(point.recipe for _, point in pending))
            for name, point in supported:
                resolved = store.get_resolution(self.protocol.id, point.recipe.id)
                if (name, point) in pending or resolved is None:
                    with factory.steered(point.recipe) as steered:
                        resolved = steered.resolved_id
                        for trial in range(self.num_trials):
                            for benchmark in self.benchmarks:
                                bar.set_postfix_str(f"{name} trial {trial} {benchmark.name}")
                                evaluate_items(
                                    steered, benchmark, selections[benchmark.name], store=store, replicate=trial,
                                )
                                bar.update(1)
                else:
                    bar.update(self.num_trials * len(self.benchmarks))
                records.append({
                    "pipeline": name, "recipe_id": point.recipe.id, "resolved_id": resolved,
                    "assignment": json.loads(canonical_json(dict(point.assignment))),
                })
        finally:
            bar.close()
            factory.release()

        _write_protocol(save_dir, self.protocol, store)
        _write_json(save_dir / POINTS_FILE, {
            "schema": "points/1",
            "protocol_id": self.protocol.id,
            "store": str(store.root.resolve()),
            "num_trials": self.num_trials,
            "benchmarks": [
                {"name": benchmark.name, "id": benchmark.id, "items": selections[benchmark.name]}
                for benchmark in self.benchmarks
            ],
            "points": records,
        })

    def _complete(self, point, store: Store, selections: Mapping[str, list[str]]) -> bool:
        """Whether every cell of a point already has evidence for every selected item."""
        resolved = store.get_resolution(self.protocol.id, point.recipe.id)
        if resolved is None:
            return False
        for benchmark in self.benchmarks:
            for trial in range(self.num_trials):
                rows = store.get_evidence(self.protocol.id, resolved, benchmark.id, selections[benchmark.name], trial)
                if len(rows) < len(selections[benchmark.name]):
                    return False
        return True

    def _require_run(self) -> Path:
        if self._save_dir is None:
            raise RuntimeError("The frames require a completed run(); call run() first.")
        return self._save_dir

    def samples_frame(
        self,
        scores: Mapping[str, str] | None = None,
        *,
        benchmark: str | None = None,
        metadata_keys: Sequence[str] = (),
        include_text: bool = False,
    ) -> pandas.DataFrame:
        """The last run's per-item frame; see the module-level `samples_frame`."""
        return samples_frame(
            self._require_run(), scores, benchmark=benchmark, metadata_keys=metadata_keys, include_text=include_text,
        )

    def runs_frame(self, metrics: Mapping[str, str] | None = None, *, benchmark: str | None = None) -> pandas.DataFrame:
        """The last run's per-trial frame; see the module-level `runs_frame`."""
        return runs_frame(self._require_run(), metrics, benchmark=benchmark)


def _converter():
    from steerability.utils.optional import require

    require("inspect_ai")
    from inspect_ai.scorer import value_to_float

    return value_to_float()


def _score_value(scores: Mapping[str, Any], key: str, converter) -> Any:
    """The value of a score key (`"<scorer>"` or `"<scorer>/<key>"`) in an item's scores.

    Scalar values (numbers, booleans, and the `C`/`I`/`P`/`N` letter grades) are converted with
    Inspect's `value_to_float`; other values are returned unchanged.

    Raises:
        KeyError: If the scorer or dict key is absent.
    """
    name, _, sub_key = key.partition("/")
    if name not in scores:
        raise KeyError(f"Score key {key!r} refers to scorer {name!r}, absent from an item; present: {sorted(scores)}.")
    value = scores[name]
    if sub_key:
        if not isinstance(value, Mapping) or sub_key not in value:
            available = sorted(value) if isinstance(value, Mapping) else value
            raise KeyError(
                f"Score key {key!r} refers to dict key {sub_key!r}, absent from scorer {name!r}; present: {available}."
            )
        value = value[sub_key]
    if isinstance(value, (bool, int, float)) or (isinstance(value, str) and value in _LETTER_GRADES):
        return converter(value)
    return value


def _read_run(save_dir: str | Path) -> tuple[dict, Store]:
    save_dir = Path(save_dir)
    points_path = save_dir / POINTS_FILE
    if not points_path.exists():
        raise FileNotFoundError(f"{save_dir} contains no {POINTS_FILE}; it is not a finished SteeringEval run.")
    points = json.loads(points_path.read_text(encoding="utf-8"))
    return points, Store.open(points["store"])


def _select_benchmarks(points: dict, benchmark: str | None, required: bool) -> list[dict]:
    benchmarks = points["benchmarks"]
    if benchmark is not None:
        selected = [entry for entry in benchmarks if entry["name"] == benchmark]
        if not selected:
            names = [b["name"] for b in benchmarks]
            raise ValueError(f"The run has no benchmark {benchmark!r}; benchmarks are {names}.")
        return selected
    if required and len(benchmarks) > 1:
        raise ValueError(f"The run spans several benchmarks {[b['name'] for b in benchmarks]}; pass benchmark=.")
    return benchmarks


def _assignment_columns(points: dict) -> list[str]:
    columns: list[str] = []
    for record in points["points"]:
        for path in record["assignment"]:
            if path not in columns:
                columns.append(path)
    return columns


def samples_frame(
    save_dir: str | Path,
    scores: Mapping[str, str] | None = None,
    *,
    benchmark: str | None = None,
    metadata_keys: Sequence[str] = (),
    include_text: bool = False,
) -> pandas.DataFrame:
    """One row per pipeline, recipe, trial, benchmark, and item of a finished run.

    Args:
        save_dir: The run's directory.
        scores: Mapping from output column to score key (`"<scorer>"` for a scalar score or
            `"<scorer>/<key>"` for one key of a dict-valued score), or None for a single `value`
            column with each item's value.
        benchmark: The benchmark to select by name; required when the run has several and
            `scores` is given.
        metadata_keys: Item metadata entries to add as columns (None when absent).
        include_text: Add the item's `input` and `target` and the `completion` as columns.

    Returns:
        A frame with columns `pipeline`, `recipe_id`, `trial_id`, `benchmark`, `item_id`, `status`,
        `tokens`, the score columns, the metadata columns, the text columns, and one column per
        assignment path, sorted by (pipeline, recipe_id, trial_id, benchmark, item order).

    Raises:
        FileNotFoundError: If `save_dir` is not a finished run.
        ValueError: If the benchmark selection is unknown or ambiguous.
        KeyError: If a score key is absent from an `ok` item.
    """
    points, store = _read_run(save_dir)
    benchmarks = _select_benchmarks(points, benchmark, required=scores is not None)
    converter = _converter() if scores else None
    assignment_columns = _assignment_columns(points)
    rows: list[dict[str, Any]] = []
    for entry in benchmarks:
        items = {row.item_id: row for row in store.get_items(entry["id"])}
        order = {item_id: index for index, item_id in enumerate(entry["items"])}
        for record in points["points"]:
            evidence = store.evidence_rows(points["protocol_id"], record["resolved_id"], entry["id"])
            for row in evidence:
                if row.item_id not in order or row.replicate >= points["num_trials"]:
                    continue
                detail = row.detail or {}
                out: dict[str, Any] = {
                    "pipeline": record["pipeline"], "recipe_id": record["recipe_id"], "trial_id": row.replicate,
                    "benchmark": entry["name"], "item_id": row.item_id, "status": row.status, "tokens": row.tokens,
                    "_order": order[row.item_id],
                }
                if scores is None:
                    out["value"] = row.value
                else:
                    for column, key in scores.items():
                        out[column] = _score_value(detail.get("scores", {}), key, converter) \
                            if row.status == "ok" else None
                payload = items[row.item_id].payload if row.item_id in items else {}
                metadata = payload.get("metadata") or {}
                for key in metadata_keys:
                    out[key] = metadata.get(key)
                if include_text:
                    out["input"] = payload.get("input")
                    out["target"] = payload.get("target")
                    completion = detail.get("completion")
                    if completion is None and row.rollout_key is not None:
                        rollout = store.get_rollout(row.rollout_key)
                        completion = rollout.completion if rollout is not None else None
                    out["completion"] = completion
                for path in assignment_columns:
                    out[path] = record["assignment"].get(path)
                rows.append(out)
    score_columns = ["value"] if scores is None else list(scores)
    columns = (
        ["pipeline", "recipe_id", "trial_id", "benchmark", "item_id", "status", "tokens"] + score_columns
        + list(metadata_keys) + (["input", "target", "completion"] if include_text else []) + assignment_columns
    )
    frame = pandas.DataFrame(rows, columns=columns + ["_order"])
    frame = frame.sort_values(["pipeline", "recipe_id", "trial_id", "benchmark", "_order"], ignore_index=True)
    return frame.drop(columns="_order")


def runs_frame(
    save_dir: str | Path,
    metrics: Mapping[str, str] | None = None,
    *,
    benchmark: str | None = None,
) -> pandas.DataFrame:
    """One row per pipeline, recipe, and trial of one benchmark of a finished run.

    Each metric column is the mean of its score key over the trial's `ok` items, the input
    `summarize_runs` and `evaluation/plotting.py` consume. A metric that is not a mean over items
    is computed from `samples_frame` and merged on `["pipeline", "recipe_id", "trial_id"]`.

    Args:
        save_dir: The run's directory.
        metrics: Mapping from output column to score key, or None for a single `value` column
            with the mean item value.
        benchmark: The benchmark to select by name; required when the run has several.

    Returns:
        A frame with columns `pipeline`, `recipe_id`, `trial_id`, `n`, `errors`, the metric
        columns, and one column per assignment path, sorted by (pipeline, recipe_id, trial_id).

    Raises:
        FileNotFoundError: If `save_dir` is not a finished run.
        ValueError: If the benchmark selection is unknown or ambiguous.
        KeyError: If a score key is absent from an `ok` item.
    """
    points, _ = _read_run(save_dir)
    selected = _select_benchmarks(points, benchmark, required=True)
    samples = samples_frame(save_dir, metrics, benchmark=selected[0]["name"])
    metric_columns = ["value"] if metrics is None else list(metrics)
    assignment_columns = _assignment_columns(points)
    keys = ["pipeline", "recipe_id", "trial_id"]
    rows: list[dict[str, Any]] = []
    for (pipeline, recipe_id, trial_id), group in samples.groupby(keys, sort=True):
        ok = group[group["status"] == "ok"]
        row: dict[str, Any] = {
            "pipeline": pipeline, "recipe_id": recipe_id, "trial_id": int(trial_id), "n": int(len(group)),
            "errors": int((group["status"] == "error").sum()),
        }
        for column in metric_columns:
            values = pandas.to_numeric(ok[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else float("nan")
        for path in assignment_columns:
            row[path] = group[path].iloc[0]
        rows.append(row)
    frame = pandas.DataFrame(rows, columns=keys + ["n", "errors"] + metric_columns + assignment_columns)
    for path in assignment_columns:
        converted = pandas.to_numeric(frame[path], errors="coerce")
        if not (converted.isna() & frame[path].notna()).any():
            frame[path] = converted
    return frame.sort_values(keys, ignore_index=True)


def summarize_runs(
    runs: pandas.DataFrame,
    metric_cols: Sequence[str],
    group_cols: Sequence[str] = ("pipeline", "recipe_id"),
    param_cols: Sequence[str] = (),
) -> pandas.DataFrame:
    """Aggregate per-trial rows into `{metric}_mean` / `{metric}_std` / `{metric}_sem` rows.

    Produces the summary-frame contract the plotting layer (`steerability.evaluation.plotting`)
    consumes; the plots read `{metric}_mean` and `{metric}_std`, and `{metric}_sem` (the
    standard error of the mean over trials) is included for tabular reporting. Adds `n_trials`
    (the non-null count of the first metric) and includes each `param_cols` entry with its
    first value per group; `param_cols` entries absent from `runs` are ignored. A
    single-trial group's std and sem are 0.0 rather than NaN, so the plotting layer draws a
    zero-length error bar.

    Args:
        runs: The wide per-trial frame from `runs_frame`.
        metric_cols: Metric column names to aggregate. Must be non-empty.
        group_cols: Grouping columns defining one configuration.
        param_cols: Per-configuration columns (e.g. a swept argument) to include.

    Returns:
        One row per group with the aggregated columns.

    Raises:
        ValueError: If `metric_cols` is empty.
    """
    if not metric_cols:
        raise ValueError("metric_cols must name at least one metric column.")
    group_cols = list(group_cols)
    param_cols = [col for col in param_cols if col in runs.columns]
    aggregations: dict[str, tuple[str, str]] = {}
    for metric_col in metric_cols:
        aggregations[f"{metric_col}_mean"] = (metric_col, "mean")
        aggregations[f"{metric_col}_std"] = (metric_col, "std")
        aggregations[f"{metric_col}_sem"] = (metric_col, "sem")
    aggregations["n_trials"] = (metric_cols[0], "count")
    for col in param_cols:
        aggregations[col] = (col, "first")
    summary = runs.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
    spread_cols = [f"{metric_col}_{stat}" for metric_col in metric_cols for stat in ("std", "sem")]
    summary[spread_cols] = summary[spread_cols].fillna(0.0)
    return summary


__all__ = ["SteeringEval", "runs_frame", "samples_frame", "summarize_runs"]
