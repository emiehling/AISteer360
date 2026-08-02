"""Benchmark runner for steering pipelines.

Provides a `Benchmark` class for evaluating one or more steering pipeline configurations on a single `UseCase`.
"""
import gc
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from aisteer360.algorithms.core.specs import ControlSpec
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.utils.tokenization import ensure_pad_token
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.data_utils import hash_params, to_jsonable

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "checkpoint.json"


def _config_id_for(params: dict[str, Any] | None) -> str:
    """Derive a stable config identifier from a params dict (or None/empty for baselines)."""
    return hash_params(params or {})


class Benchmark:
    """Benchmark functionality for comparing steering pipelines on a use case.

    A Benchmark runs one or more steering pipeline configurations on a given use case, optionally with multiple trials
    per configuration. Each trial reuses the same steered model and re-samples any generate-time randomness (e.g.,
    few-shot selection, sampling-based decoding, etc.).

    When ``save_dir`` is provided, results are checkpointed to disk after each completed configuration so that a run
    can be interrupted and resumed without re-generating completed work. On resume, configurations whose results are
    already present in the checkpoint are skipped entirely (no model loading or steering).

    Non-structural pipelines share one preloaded base model; structural pipelines load their own model from
    ``base_model_name_or_path``. The shared base is expected not to be mutated by a non-structural configuration. After
    each shared-base configuration finishes, a fingerprint tripwire checks the shared model for change and, on
    detecting one, warns naming the configuration and drops the shared model so the next configuration reloads a clean
    base. The tripwire samples a bounded subset of parameters, so it makes the no-mutation invariant observable rather
    than proven.

    Attributes:
        use_case: Use case that defines prompt construction, generation logic, and evaluation metrics.
        base_model_name_or_path: Hugging Face model ID or local path for the base causal language model.
        steering_pipelines: Mapping from pipeline name to a list of controls or `ControlSpec` objects; empty list
            denotes a baseline (no steering).
        runtime_overrides: Optional overrides passed through to `UseCase.generate` for runtime control parameters.
            Overrides are routed by control class name over the pipeline's supplied controls, so two instances of
            the same class in one pipeline share a single override entry.
        hf_model_kwargs: Extra kwargs forwarded to `AutoModelForCausalLM.from_pretrained`.
        gen_kwargs: Generation kwargs forwarded to :meth:`UseCase.generate`.
        device_map: Device placement strategy used when loading models.
        num_trials: Number of evaluation trials to run per concrete pipeline configuration.
        batch_size: Generation batch size forwarded as a keyword into ``UseCase.generate``.
        save_dir: Optional directory for incremental checkpoints. When set, completed configurations are written to a
            ``checkpoint.json`` file and the use case's ``export()`` is called after each pipeline finishes. Subsequent
             calls on already-completed configurations are skipped.
    """

    def __init__(
        self,
        use_case: UseCase,
        base_model_name_or_path: str | Path,
        steering_pipelines: dict[str, list[Any]],
        runtime_overrides: dict[str, dict[str, Any]] | None = None,
        hf_model_kwargs: dict | None = None,
        gen_kwargs: dict | None = None,
        device_map: str = "auto",
        num_trials: int = 1,
        batch_size: int = 8,
        save_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(use_case, UseCase):
            raise TypeError(f"use_case must be a UseCase instance; got {type(use_case).__name__}.")
        if not isinstance(steering_pipelines, dict):
            raise TypeError(f"steering_pipelines must be a dict; got {type(steering_pipelines).__name__}.")
        for name, pipeline in steering_pipelines.items():
            if pipeline is not None and not isinstance(pipeline, (list, tuple)):
                raise TypeError(
                    f"steering_pipelines[{name!r}] must be a list, tuple, or None; got {type(pipeline).__name__}."
                )
        self.num_trials = int(num_trials)
        if self.num_trials < 0:
            raise ValueError("num_trials must be >= 0.")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.use_case = use_case
        self.base_model_name_or_path = base_model_name_or_path
        self.steering_pipelines = steering_pipelines
        self.runtime_overrides = runtime_overrides
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}
        self.device_map = device_map
        self.save_dir = Path(save_dir) if save_dir is not None else None

        # lazy-init shared base model/tokenizer
        self._base_model: AutoModelForCausalLM | None = None
        self._base_tokenizer: AutoTokenizer | None = None
        self._base_fingerprint: str | None = None

    def _ensure_base_model(self) -> None:
        """Load the base model/tokenizer once (for reuse across pipelines)."""
        if self._base_model is not None and self._base_tokenizer is not None:
            return

        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path,
            device_map=self.device_map,
            **self.hf_model_kwargs,
        )
        self._base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self._base_tokenizer = ensure_pad_token(self._base_tokenizer)
        self._base_fingerprint = self._fingerprint_or_none(self._base_model)

    def _fingerprint_or_none(self, model) -> str | None:
        """Digest of the shared base model, or None (guard disabled) when fingerprinting fails."""
        if model is None:
            return None
        try:
            return model_fingerprint(model)
        except Exception:
            logger.debug("Model fingerprint unavailable; shared-model guard disabled.", exc_info=True)
            return None

    def _verify_shared_base_model(self, controls: Sequence[Any]) -> None:
        """Tripwire: detect shared-base mutation after a configuration, then quarantine.

        The fingerprint samples up to 8 parameters times 64 elements, so this makes the no-mutation
        invariant observable, not proven; trials after an early-trial mutation ran polluted before
        detection. Warn-and-quarantine (not raise) is deliberate, since aborting a long sweep for one
        misbehaving control is worse than reloading and flagging.

        Args:
            controls: The configuration's controls, named in the warning (or "baseline" when empty).
        """
        if self._base_model is None or self._base_fingerprint is None:
            return
        current = self._fingerprint_or_none(self._base_model)
        if current == self._base_fingerprint:
            return
        names = ", ".join(type(control).__name__ for control in controls) or "baseline"
        logger.warning(
            "Shared base model changed during configuration [%s] (fingerprint %s -> %s); its recorded "
            "results reflect the mutated weights. Dropping the shared model so the next configuration "
            "reloads a clean base.",
            names, self._base_fingerprint, current,
        )
        self._base_model = None
        self._base_tokenizer = None
        self._base_fingerprint = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _has_structural_control(controls: Sequence[Any]) -> bool:
        """Return True if any of the controls is a StructuralControl."""
        return any(
            isinstance(control, StructuralControl) and getattr(control, "enabled", True)
            for control in controls
        )

    def _load_checkpoint(self) -> dict[str, list[dict[str, Any]]]:
        """Load previously-saved profiles from disk, or return an empty dict."""
        if self.save_dir is None:
            return {}
        path = self.save_dir / _CHECKPOINT_FILENAME
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                profiles = json.load(f)
            n_runs = sum(len(runs) for runs in profiles.values())
            logger.info("Resumed from checkpoint: %d run(s) across %d pipeline(s)", n_runs, len(profiles))
            return profiles
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read checkpoint file; starting fresh.", exc_info=True)
            return {}

    def _save_checkpoint(self, profiles: dict[str, list[dict[str, Any]]]) -> None:
        """Atomically write current profiles to the checkpoint file."""
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        safe = to_jsonable(profiles)
        tmp = self.save_dir / f"{_CHECKPOINT_FILENAME}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False)
        tmp.rename(self.save_dir / _CHECKPOINT_FILENAME)

    @staticmethod
    def _runs_for_config(runs: list[dict[str, Any]], config_id: str) -> list[dict[str, Any]]:
        """Filter a list of runs to those matching a given config id."""
        return [r for r in runs if _config_id_for(r.get("params")) == config_id]

    def run(self) -> dict[str, list[dict[str, Any]]]:
        """Run the benchmark on all configured steering pipelines.

        Each pipeline configuration is expanded into one or more control settings (via `ControlSpecs` when present).
        For each configuration, the model is steered once and evaluated over `num_trials` trials.

        When ``save_dir`` was provided at construction time, completed configurations are persisted incrementally and
        the use case's ``export()`` method is called after each pipeline finishes. A subsequent call with the same
        ``save_dir`` automatically skips already-completed work.

        Returns:
            A mapping from pipeline name to a list of run dictionaries. Each run dictionary has keys:

                - `"trial_id"`: Integer trial index.
                - `"generations"`: Model generations returned by the use case.
                - `"evaluations"`: Metric results returned by the use case.
                - `"params"`: Mapping from spec name to constructor kwargs used for control, or an empty dict for
                    fixed/baseline pipelines.
        """
        profiles = self._load_checkpoint()

        for pipeline_name, pipeline in self.steering_pipelines.items():
            pipeline = pipeline or []

            logger.info("Running pipeline: %s", pipeline_name)

            has_specs = any(isinstance(control, ControlSpec) for control in pipeline)
            if has_specs and not all(isinstance(control, ControlSpec) for control in pipeline):
                raise TypeError(
                    f"Pipeline '{pipeline_name}' mixes ControlSpec and fixed controls. Either use only fixed controls "
                    "or only ControlSpecs. Wrap fixed configs in ControlSpec(vars=None) if needed."
                )

            existing_runs = profiles.get(pipeline_name, [])

            if not pipeline:  # baseline (no steering)
                runs = self._run_pipeline(controls=[], params=None, existing_runs=existing_runs)
            elif has_specs:
                runs = self._run_spec_pipeline(
                    pipeline_name, control_specs=pipeline, existing_runs=existing_runs, profiles=profiles,
                )
            else:
                runs = self._run_pipeline(controls=pipeline, params=None, existing_runs=existing_runs)

            profiles[pipeline_name] = runs
            logger.info("Pipeline %s complete", pipeline_name)

            self._save_checkpoint(profiles)
            self._try_export(profiles)

        return profiles

    def _run_pipeline(
        self,
        controls: list[Any],
        params: dict[str, dict[str, Any]] | None = None,
        existing_runs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a concrete steering pipeline configuration for all trials.

        This helper handles both baseline (no controls) and fixed-control pipelines. Structural steering is applied
        once; the use case is evaluated `num_trials` times (to capture generate-time variability).

        If the configuration is already present in `existing_runs` (from a prior checkpoint), its runs are returned
        immediately and the model is never loaded or steered.

        Args:
            controls: List of instantiated steering controls, or an empty list for the baseline (unsteered) model.
            params: Optional mapping from spec name to full constructor kwargs used to build the controls.
            existing_runs: Runs already loaded from a checkpoint for this pipeline.

        Returns:
            A list of run dictionaries, one per trial.
        """
        config_id = _config_id_for(params)

        # fast path: config already completed — skip model loading entirely
        cached = self._runs_for_config(existing_runs or [], config_id)
        if cached:
            logger.info("Skipping config=%s — already complete (%d run(s))", config_id, len(cached))
            return cached

        uses_shared_base = not self._has_structural_control(controls)
        pipeline: SteeringPipeline | None = None
        runs: list[dict[str, Any]] = []

        try:
            # build the pipeline once: structural pipelines load their own model; baseline and
            # non-structural pipelines share the preloaded base through an injected pipeline
            if not uses_shared_base:
                pipeline = SteeringPipeline(
                    model_name_or_path=self.base_model_name_or_path,
                    controls=list(controls),
                    device_map=self.device_map,
                    hf_model_kwargs=self.hf_model_kwargs,
                )
                pipeline.steer()
            else:
                self._ensure_base_model()  # only shared-base configurations load the shared base
                pipeline = SteeringPipeline(model_name_or_path=None, controls=list(controls), lazy_init=True)
                pipeline.model = self._base_model
                pipeline.tokenizer = self._base_tokenizer
                if self._base_model is not None:
                    pipeline.device = self._base_model.device
                pipeline.steer()

            tokenizer = pipeline.tokenizer
            model_or_pipeline: Any = pipeline

            # run trials
            for trial_id in range(self.num_trials):
                generations = self.use_case.generate(
                    model_or_pipeline=model_or_pipeline,
                    tokenizer=tokenizer,
                    gen_kwargs=self.gen_kwargs,
                    runtime_overrides=self.runtime_overrides,
                    batch_size=self.batch_size
                )
                scores = self.use_case.evaluate(generations)

                runs.append({
                    "trial_id": trial_id,
                    "generations": generations,
                    "evaluations": scores,
                    "params": params or {},
                })

            return runs

        finally:
            # cleanup controls that may hold GPU resources (e.g., reward models)
            if pipeline is not None:
                for control in (*pipeline.structural_controls, *pipeline.input_controls,
                                *pipeline.state_controls, *pipeline.output_controls):
                    cleanup_fn = getattr(control, "cleanup", None)
                    if callable(cleanup_fn):
                        try:
                            cleanup_fn()
                        except Exception:
                            logger.warning("Control cleanup failed", exc_info=True)
                del pipeline
            if uses_shared_base:
                self._verify_shared_base_model(controls)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_spec_pipeline(
        self,
        pipeline_name: str,
        control_specs: list[ControlSpec],
        existing_runs: list[dict[str, Any]] | None = None,
        profiles: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a pipeline whose controls are defined by `ControlSpec`s.

        This method:

        - Expands each `ControlSpec` into one or more local parameter choices
        - Takes the cartesian product across specs to form pipeline configurations
        - Evaluates each configuration using `_run_pipeline`

        Configurations already present in the checkpoint are skipped entirely (no model loading or steering).

        Args:
            pipeline_name: Name of the pipeline being evaluated; passed into the context for `ControlSpec`s.
            control_specs: `ControlSpec` objects describing the controls used in the given pipeline.
            existing_runs: Runs already loaded from a checkpoint for this pipeline.
            profiles: The full profiles dict, passed through for incremental checkpointing after each config.

        Returns:
            A flat list of run dictionaries across all configurations and trials.
            Each run dictionary includes:

                - "trial_id": Integer trial index
                - "generations": Model outputs for the given trial
                - "evaluations": Metric results for the given trial
                - "params": Mapping from spec name to full constructor kwargs for the given configuration
        """
        existing_runs = existing_runs or []

        # resolved spec names key the params dict (and thus config identity); duplicates would overwrite
        resolved_names = [spec.name or spec.control_cls.__name__ for spec in control_specs]
        duplicates = sorted({name for name in resolved_names if resolved_names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Pipeline '{pipeline_name}' has multiple ControlSpecs resolving to the same name(s): "
                f"{duplicates}. Give each spec a distinct `name=` so their parameters are tracked separately."
            )

        base_context = {
            "pipeline_name": pipeline_name,
            "base_model_name_or_path": self.base_model_name_or_path,
        }

        # collect points per spec
        spec_points: list[tuple[ControlSpec, list[dict[str, Any]]]] = []
        for spec in control_specs:
            points = list(spec.iter_points(base_context))
            if not points:
                points = [{}]
            spec_points.append((spec, points))

        if not spec_points:
            return self._run_pipeline(controls=[], params=None, existing_runs=existing_runs)

        spec_list, points_lists = zip(*spec_points)
        combos = itertools.product(*points_lists)

        runs: list[dict[str, Any]] = []

        for combo_id, combo in enumerate(combos):
            # pre-compute params so we can check the checkpoint before instantiating controls
            params: dict[str, dict[str, Any]] = {}
            global_context = {
                "pipeline_name": pipeline_name,
                "base_model_name_or_path": self.base_model_name_or_path,
                "combo_id": combo_id,
            }

            for spec, local_point in zip(spec_list, combo):
                spec_name = spec.name or spec.control_cls.__name__
                kwargs = spec.resolve_params(chosen=local_point, context=global_context)
                params[spec_name] = kwargs

            config_id = _config_id_for(params)

            # fast path: skip config entirely if already done
            cached = self._runs_for_config(existing_runs, config_id)
            if cached:
                logger.info("Skipping configuration %d (config=%s); already complete", combo_id + 1, config_id)
                runs.extend(cached)
                continue

            logger.info("Running configuration %d", combo_id + 1)

            # instantiate controls only when we actually need to run
            controls: list[Any] = []
            for spec, local_point in zip(spec_list, combo):
                spec_name = spec.name or spec.control_cls.__name__
                control = spec.control_cls(**params[spec_name])
                controls.append(control)

            config_runs = self._run_pipeline(controls=controls, params=params, existing_runs=existing_runs)
            runs.extend(config_runs)

            # checkpoint after each config so partial spec sweeps survive interruption
            if profiles is not None:
                profiles[pipeline_name] = runs
                self._save_checkpoint(profiles)

        return runs

    def _try_export(self, profiles: dict[str, list[dict[str, Any]]]) -> None:
        """Call the use case's export method; log and swallow failures."""
        if self.save_dir is None:
            return
        try:
            self.export(profiles, str(self.save_dir))
        except Exception:
            logger.warning("Incremental export failed; checkpoint is still intact.", exc_info=True)

    def export(self, profiles: dict[str, list[dict[str, Any]]], save_dir: str) -> None:
        """Export benchmark results to disk.

        Sanitizes the profiles to a JSON-friendly structure. When the use case overrides `export`, its
        method is called; otherwise the sanitized profiles are written to ``profiles.json`` under
        ``save_dir``. An `export` assigned as an instance attribute (rather than a class override) is
        not detected, so the default write runs.

        Args:
            profiles: The benchmark profiles to export.
            save_dir: Directory to export into; created if absent.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        safe_profiles = to_jsonable(profiles)
        if type(self.use_case).export is not UseCase.export:  # instance-attribute exports are not detected
            self.use_case.export(safe_profiles, save_dir)
            return
        with open(save_path / "profiles.json", "w", encoding="utf-8") as f:
            json.dump(safe_profiles, f, indent=4, ensure_ascii=False)
