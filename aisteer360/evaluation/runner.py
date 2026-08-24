"""Sweep runner: configurations x trials x suites over one base model.

`SteeringEval` owns what Inspect cannot: pipeline lifecycle and GPU discipline. Inspect treats
models as cheap concurrent handles to endpoints; a steered pipeline is a GPU-resident object that
must be built, steered, evaluated, and released sequentially, so the runner evaluates one pipeline
at a time and never passes more than one pipeline-backed model to one `eval` or `eval_set` call.

There is no results checkpoint: the `.eval` logs under `save_dir/inspect_logs/` are the store, and
`eval_set` resumes each (configuration, trial, suite) cell from them at sample granularity. `eval_set`
matches task identity only (task, task args, model name); the runner's seed, generate defaults, provider
options, fit, and backend are not part of it, so a changed protocol needs a new `save_dir` rather than a
re-run into the old one. Each result row's `provenance` entry records what actually ran.
"""
import datetime
import importlib.metadata
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import pandas
from tqdm.auto import tqdm

import aisteer360
from aisteer360.algorithms.core.execution.spec import BackendSpec
from aisteer360.algorithms.core.identity import derive_trial_seed
from aisteer360.algorithms.core.sweeps import PipelineFactory, expand_configurations, preflight
from aisteer360.utils.rendering import has_chat_template

if TYPE_CHECKING:
    from aisteer360.evaluation.provider import ProviderOptions
    from aisteer360.evaluation.suite import InspectSuite

logger = logging.getLogger(__name__)

_RESULTS_COLUMNS = (
    "config", "config_id", "trial", "seed", "suite", "task", "scorer", "metric", "value", "n", "log",
)


def _package_version(name: str) -> str | None:
    """Installed version of a package, or None when it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class SteeringEval:
    """Evaluate steering pipeline configurations on Inspect suites, over trials, sequentially.

    Configurations expand from `pipelines` (fixed controls, `ControlSpec` sweeps, and the empty
    baseline arm); a pre-flight support check runs over every configuration before any model or
    engine work. Each configuration is built and steered once, then every trial runs every suite
    against it before the pipeline is released; each suite run builds and discards its own
    provider, named by the configuration's `config_id`. Repetition is trial-based: with `seed`
    set, each (configuration, trial) derives one seed, attached to sampling dispatches whose
    config carries no seed of its own. Inspect epochs are not used.

    Attributes:
        pipelines: Mapping from pipeline name to `[]` (the unsteered baseline arm),
            `[Control, ...]`, or `[ControlSpec, ...]`.
        base_model_name_or_path: Hugging Face model ID or local path of the base model.
        suites: The `InspectSuite`s every configuration and trial runs.
        backend: Backend forwarded to the pipelines (a `BackendSpec` or a known kind name).
        fit: Fit venue policy forwarded to the pipelines.
        hf_model_kwargs: Load-time kwargs for in-process model loads.
        device_map: Device placement for in-process model loads.
        trust_remote_code: Trust remote code when loading tokenizers.
        num_trials: Trials per configuration; a completion target, not part of run identity.
        seed: Base seed deriving one seed per (configuration, trial), or None.
        provider_options: `ProviderOptions` forwarded to every suite run (static runtime kwargs,
            batching ceiling, reasoning split).
        generate_defaults: `GenerateConfig` defaults applied under each suite's overrides.
        on_unsupported: `"raise"` (default) fails the run with one aggregate error on any
            unsupported configuration; `"skip"` runs the supported ones with a warning.
        save_dir: Directory holding the `.eval` logs; when None, logs go to a
            fresh temporary directory and the run cannot be resumed.
        progress: Draw a `tqdm` bar over the (configuration, trial, suite) cells. The same
            information is logged at INFO regardless, so script users see it without the bar.
        display: Inspect's per-sample `display` mode, forwarded to every suite run (`"none"` by
            default, `"plain"` recommended in a sweep). Presentation only; not part of run
            identity.
    """

    def __init__(
        self,
        pipelines: dict[str, list],
        base_model_name_or_path: str | Path,
        suites: "Sequence[InspectSuite]",
        *,
        backend: BackendSpec | str | None = None,
        fit: Literal["auto", "in_process"] = "auto",
        hf_model_kwargs: dict | None = None,
        device_map: str | dict | None = "auto",
        trust_remote_code: bool = False,
        num_trials: int = 1,
        seed: int | None = None,
        provider_options: "ProviderOptions | None" = None,
        generate_defaults: Mapping[str, Any] | None = None,
        on_unsupported: Literal["raise", "skip"] = "raise",
        save_dir: str | Path | None = None,
        progress: bool = True,
        display: str = "none",
    ) -> None:
        if not isinstance(pipelines, dict):
            raise TypeError(f"pipelines must be a dict; got {type(pipelines).__name__}.")
        if int(num_trials) < 1:
            raise ValueError(f"num_trials must be >= 1; got {num_trials}.")
        if on_unsupported not in ("raise", "skip"):
            raise ValueError(f"on_unsupported must be 'raise' or 'skip'; got {on_unsupported!r}.")
        suites = list(suites)
        if not suites:
            raise ValueError("suites must be non-empty.")
        suite_names = [suite.name for suite in suites]
        duplicates = sorted({name for name in suite_names if suite_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Suite names must be distinct; duplicated: {duplicates}.")

        self.pipelines = pipelines
        self.base_model_name_or_path = base_model_name_or_path
        self.suites = suites
        self.backend = backend
        self.fit = fit
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.num_trials = int(num_trials)
        self.seed = seed
        self.provider_options = provider_options
        self.generate_defaults = dict(generate_defaults) if generate_defaults is not None else None
        self.on_unsupported = on_unsupported
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.progress = bool(progress)
        self.display = display
        self._results: dict[str, list[dict]] | None = None

    def run(self) -> dict[str, list[dict]]:
        """Run every configuration x trial x suite cell, sequentially, resuming from the logs.

        Returns:
            A mapping from pipeline name to a list of run dictionaries. Each run dictionary has
            keys:

                - `"trial_id"`: Integer trial index.
                - `"seed"`: The trial's derived seed, or None when no base seed was set.
                - `"config_id"`: The configuration's canonical identifier.
                - `"params"`: Mapping from spec name to resolved constructor kwargs, or an empty
                    dict for fixed and baseline configurations.
                - `"suites"`: Mapping from suite name to that suite's flattened results, with log
                    paths relative to `save_dir`.
                - `"provenance"`: Versions, backend kind, `prompt_path`, the shared-base
                    fingerprint when one exists, and a timestamp.

        Raises:
            RuntimeError: If any configuration is unsupported and `on_unsupported="raise"` (one
                aggregate error before any model or engine work), or a suite's `eval_set` fails
                after its retries.
        """
        points = list(expand_configurations(
            self.pipelines, base_model_name_or_path=self.base_model_name_or_path,
        ))

        failures: list[str] = []
        skipped: set[tuple[str, str]] = set()
        for point in points:
            messages = preflight(
                [point], base_model_name_or_path=self.base_model_name_or_path,
                backend=self.backend, fit=self.fit,
            )
            if messages:
                failures.extend(messages)
                skipped.add((point.pipeline_name, point.config_id))
        if failures:
            if self.on_unsupported == "raise":
                raise RuntimeError(
                    "Unsupported pipeline configuration(s):\n" + "\n".join(failures)
                )
            for line in failures:
                logger.warning("Skipping unsupported configuration: %s", line)

        if self.save_dir is not None:
            save_dir = self.save_dir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(tempfile.mkdtemp(prefix="steering-eval-"))
            logger.info("No save_dir was given; logs go to %s and the run cannot be resumed.", save_dir)

        versions = {
            "toolkit_version": getattr(aisteer360, "__version__", "unknown"),
            "inspect_ai_version": _package_version("inspect-ai"),
            "inspect_evals_version": _package_version("inspect-evals"),
        }
        factory = PipelineFactory(
            self.base_model_name_or_path,
            backend=self.backend,
            fit=self.fit,
            hf_model_kwargs=self.hf_model_kwargs,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        results: dict[str, list[dict]] = {name: [] for name in self.pipelines}
        active = [point for point in points if (point.pipeline_name, point.config_id) not in skipped]
        total_cells = len(active) * self.num_trials * len(self.suites)
        logger.info(
            "Evaluating %d configuration(s) x %d trial(s) x %d suite(s) = %d cell(s); logs under %s.",
            len(active), self.num_trials, len(self.suites), total_cells, save_dir,
        )
        bar = tqdm(
            total=total_cells, disable=not self.progress, desc="steering eval", unit="cell",
            dynamic_ncols=True,
        )
        cell_index = 0
        try:
            for point in active:
                label = f"{point.pipeline_name}/{point.config_id}"
                bar.set_postfix_str(f"steering {label}")
                steer_started = time.monotonic()
                with factory.steered(point.controls_factory()) as pipeline:
                    logger.info("Steered %s in %.0fs.", label, time.monotonic() - steer_started)
                    prompt_path = "messages" if has_chat_template(pipeline.tokenizer) else "text"
                    for trial_id in range(self.num_trials):
                        trial_seed = (
                            derive_trial_seed(self.seed, point.config_id, trial_id)
                            if self.seed is not None else None
                        )
                        suite_results: dict[str, dict] = {}
                        for suite in self.suites:
                            cell_index += 1
                            bar.set_postfix_str(f"{label} trial {trial_id} {suite.name}")
                            log_dir = (
                                save_dir / "inspect_logs" / point.config_id
                                / f"trial_{trial_id}" / suite.name
                            )
                            started = time.monotonic()
                            flattened = suite.run(
                                pipeline,
                                log_dir=log_dir,
                                options=self.provider_options,
                                base_seed=trial_seed,
                                model_name=point.config_id,
                                generate_defaults=self.generate_defaults,
                                display=self.display,
                            )
                            completed = sum(int(result["n"]) for result in flattened.values())
                            logger.info(
                                "Cell %d/%d %s trial=%d suite=%s: %d sample(s) in %.0fs.",
                                cell_index, total_cells, label, trial_id, suite.name, completed,
                                time.monotonic() - started,
                            )
                            bar.update(1)
                            cell = log_dir.relative_to(save_dir)
                            for task_result in flattened.values():
                                if not Path(task_result["log"]).is_absolute():
                                    task_result["log"] = str(cell / task_result["log"])
                            suite_results[suite.name] = flattened
                        results[point.pipeline_name].append({
                            "trial_id": trial_id,
                            "seed": trial_seed,
                            "config_id": point.config_id,
                            "params": {
                                name: dict(kwargs) for name, kwargs in (point.params or {}).items()
                            },
                            "suites": suite_results,
                            "provenance": {
                                **versions,
                                "backend": factory.backend_kind,
                                "prompt_path": prompt_path,
                                "shared_base_fingerprint": factory.shared_base_fingerprint,
                                "recorded_utc": datetime.datetime.now(
                                    datetime.timezone.utc
                                ).isoformat(),
                            },
                        })
        finally:
            bar.close()
            factory.release()
        self._results = results
        return results

    def results(self) -> pandas.DataFrame:
        """The last `run()`'s results as one row per (config, trial, suite, task, scorer/metric).

        `runs_frame` (module-level, or the `runs_frame` method for swept-parameter columns)
        pivots this frame to one row per (pipeline, trial), and `summarize_runs` aggregates
        trials into the `{metric}_mean` / `{metric}_std` / `{metric}_sem` summary rows the
        plotting layer (`aisteer360.evaluation.plotting`) consumes. Paired per-sample deltas
        remain a pandas exercise over the `.eval` logs.

        Returns:
            A `pandas.DataFrame` with columns `config`, `config_id`, `trial`, `seed`, `suite`,
            `task`, `scorer`, `metric`, `value`, `n`, and `log`.

        Raises:
            RuntimeError: If `run()` has not been called.
        """
        if self._results is None:
            raise RuntimeError("results() requires a completed run(); call run() first.")
        rows: list[dict[str, Any]] = []
        for config_name, runs in self._results.items():
            for run in runs:
                for suite_name, tasks in run["suites"].items():
                    for task_name, task_result in tasks.items():
                        for key, value in task_result["metrics"].items():
                            scorer_name, _, metric_name = key.partition("/")
                            rows.append({
                                "config": config_name,
                                "config_id": run["config_id"],
                                "trial": run["trial_id"],
                                "seed": run["seed"],
                                "suite": suite_name,
                                "task": task_name,
                                "scorer": scorer_name,
                                "metric": metric_name,
                                "value": value,
                                "n": task_result["n"],
                                "log": task_result["log"],
                            })
        return pandas.DataFrame(rows, columns=list(_RESULTS_COLUMNS))

    def runs_frame(
        self,
        metrics: Mapping[str, str],
        *,
        params: Mapping[str, tuple[str, str]] | None = None,
        suite: str | None = None,
        task: str | None = None,
    ) -> pandas.DataFrame:
        """The last `run()`'s per-trial metric values, one row per (pipeline, trial).

        Pivots `results()` through the module-level `runs_frame` and, when `params` names
        swept constructor arguments as `column -> (spec name, argument name)`, attaches each
        as a column keyed on `config_id`, read from the run records' resolved parameters.
        Rows of configurations that do not sweep the argument (the baseline arm, fixed
        pipelines) receive NaN. A column whose values are all numeric is returned with a
        numeric dtype; non-numeric values (strings, lists) are kept raw.

        Args:
            metrics: Mapping from output column name to a metric key, either
                `"scorer/metric"` (e.g. `"choice/accuracy"`) or a bare metric name when
                unambiguous. Must be non-empty.
            params: Optional mapping from output column name to `(spec name, argument name)`.
            suite: Suite to select; required when the results span several.
            task: Task to select; required when the results span several.

        Returns:
            The wide frame with columns `pipeline`, `config_id`, `trial_id`, `seed`, one
            column per `metrics` entry, and one column per `params` entry.

        Raises:
            RuntimeError: If `run()` has not been called.
            ValueError: If `metrics` is empty, or the suite/task selection is empty or
                ambiguous.
            KeyError: If a metric key matches nothing, or a bare metric name is ambiguous.
        """
        # the bare name resolves to the module-level runs_frame (name lookup skips class attributes)
        frame = runs_frame(self.results(), metrics, suite=suite, task=task)
        for column, (spec_name, argument) in (params or {}).items():
            mapped = frame["config_id"].map(self._sweep_param_map(spec_name, argument))
            converted = pandas.to_numeric(mapped, errors="coerce")
            # keep raw values for non-numeric arguments; otherwise take the numeric dtype
            frame[column] = mapped if (converted.isna() & mapped.notna()).any() else converted
        return frame

    def _sweep_param_map(self, spec_name: str, argument: str) -> dict[str, Any]:
        """`config_id -> value` for one swept constructor argument, from the run records."""
        mapping: dict[str, Any] = {}
        for runs in (self._results or {}).values():
            for run in runs:
                spec_params = (run.get("params") or {}).get(spec_name)
                if spec_params is not None and argument in spec_params:
                    mapping[run["config_id"]] = spec_params[argument]
        return mapping


def runs_frame(
    results: pandas.DataFrame,
    metrics: Mapping[str, str],
    *,
    suite: str | None = None,
    task: str | None = None,
) -> pandas.DataFrame:
    """One row per (pipeline, trial) with one column per requested metric.

    Pivots the tidy `SteeringEval.results()` frame into the wide per-trial form that
    `summarize_runs` and the plotting layer (`aisteer360.evaluation.plotting`) consume. The
    `config` column is renamed `pipeline` and `trial` is renamed `trial_id`.

    Args:
        results: The frame returned by `SteeringEval.results()`.
        metrics: Mapping from output column name to a metric key, either `"scorer/metric"`
            (e.g. `"choice/accuracy"`) or a bare metric name when unambiguous. Must be
            non-empty.
        suite: Suite to select; required when the frame holds several.
        task: Task to select; required when the frame holds several.

    Returns:
        The wide frame with columns `pipeline`, `config_id`, `trial_id`, `seed`, and one
        column per `metrics` entry, sorted by (pipeline, config_id, trial_id).

    Raises:
        ValueError: If `metrics` is empty, the selection is empty, the frame spans several
            suites or tasks without a selector, or a metric key selects duplicate
            (config, trial) rows.
        KeyError: If a metric key matches nothing, or a bare metric name is ambiguous.
    """
    if not metrics:
        raise ValueError("metrics must name at least one output column.")
    frame = results
    if suite is not None:
        frame = frame[frame["suite"] == suite]
    if task is not None:
        frame = frame[frame["task"] == task]
    if frame.empty:
        raise ValueError("No rows match the requested suite/task selection.")
    if suite is None and frame["suite"].nunique() > 1:
        raise ValueError(f"Results span several suites {sorted(frame['suite'].unique())}; pass suite=.")
    if task is None and frame["task"].nunique() > 1:
        raise ValueError(f"Results span several tasks {sorted(frame['task'].unique())}; pass task=.")

    frame = frame.assign(_key=frame["scorer"].astype(str) + "/" + frame["metric"].astype(str))
    index_cols = ["config", "config_id", "trial", "seed"]
    wide: pandas.DataFrame | None = None
    for column, key in metrics.items():
        selected = frame[frame["_key"] == key] if "/" in key else frame[frame["metric"] == key]
        if selected.empty:
            available = sorted(frame["_key"].unique())
            raise KeyError(f"Metric {key!r} not found in results; available: {available}.")
        if "/" not in key and selected["_key"].nunique() > 1:
            raise KeyError(
                f"Metric name {key!r} is ambiguous ({sorted(selected['_key'].unique())}); "
                "use the 'scorer/metric' form."
            )
        if selected.duplicated(subset=index_cols).any():
            raise ValueError(f"Metric {key!r} has duplicate (config, trial) rows; narrow the selection.")
        series = selected.set_index(index_cols)["value"].rename(column)
        wide = series.to_frame() if wide is None else wide.join(series, how="outer")

    return (
        wide.reset_index()
        .rename(columns={"config": "pipeline", "trial": "trial_id"})
        .sort_values(["pipeline", "config_id", "trial_id"], ignore_index=True)
    )


def summarize_runs(
    runs: pandas.DataFrame,
    metric_cols: Sequence[str],
    group_cols: Sequence[str] = ("pipeline", "config_id"),
    param_cols: Sequence[str] = (),
) -> pandas.DataFrame:
    """Aggregate per-trial rows into `{metric}_mean` / `{metric}_std` / `{metric}_sem` rows.

    Produces the summary-frame contract the plotting layer (`aisteer360.evaluation.plotting`)
    consumes; the plots read `{metric}_mean` and `{metric}_std`, and `{metric}_sem` (the
    standard error of the mean over trials) is carried for tabular reporting. Adds `n_trials`
    (the non-null count of the first metric) and carries each `param_cols` entry with its
    first value per group; `param_cols` entries absent from `runs` are ignored. A
    single-trial group's std and sem are 0.0 rather than NaN, so the plotting layer draws a
    zero-length error bar.

    Args:
        runs: The wide per-trial frame from `runs_frame`.
        metric_cols: Metric column names to aggregate. Must be non-empty.
        group_cols: Grouping columns defining one configuration.
        param_cols: Per-configuration columns (e.g. a swept argument) to carry through.

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
