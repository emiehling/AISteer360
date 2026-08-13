"""
Core steering pipeline for composing and applying multiple LLM control methods.
"""
import contextlib
import dataclasses
import gc
import logging
import warnings
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, overload

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)

from aisteer360.algorithms.core.execution.access import (
    ModelAccess,
    PlannedFit,
    PlannedStep,
    SteerPlan,
)
from aisteer360.algorithms.core.execution.backend import (
    SteeredSession,
    capabilities_for_spec,
    resolve_backend_class,
)
from aisteer360.algorithms.core.execution.contracts import (
    BackendCapabilities,
    Capability,
    SupportReport,
    UnsupportedOperationError,
    evaluate_support,
)
from aisteer360.algorithms.core.execution.session_utils import ScopedSession
from aisteer360.algorithms.core.execution.params import (
    GenerationParams,
    merge_lowered_params,
)
from aisteer360.algorithms.core.execution.payloads import (
    Artifact,
    ArtifactProvenance,
    ConstraintEntry,
    ConstraintSource,
    GenerationItem,
    HookEntry,
    InterventionEntry,
    PreparedPrompt,
    ProcessorSpecEntry,
    ScoringItem,
    StackEntry,
    StateControlEntry,
    remap_prompt_relative_scopes,
)
from aisteer360.algorithms.core.execution.spec import KNOWN_BACKEND_KINDS, BackendSpec
from aisteer360.algorithms.core.internals.fingerprint import is_absent_chat_template_fingerprint
from aisteer360.algorithms.core.output import (
    Output,
    infer_finish_reasons,
    truncate_at_stop_strings,
)
from aisteer360.algorithms.core.utils.controls import (
    merge_controls,
    warn_if_adapt_messages_bypassed,
)
from aisteer360.algorithms.core.utils.generation import (
    apply_adapt_messages_and_tokenize,
)
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.output_control.base import DecodingDriver, OutputControl
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.utils.tokenization import (
    ensure_pad_token,
    infer_attention_mask_from_ids,
    to_left_pad,
    warn_if_duplicate_bos,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SteeringPipeline:
    """Main steering pipeline for applying various control methods to Hugging Face causal language models.

    Enables application of structural, state, input, and output controls in a coordinated manner.
    Controls are applied in a fixed bottom-up order during steering, then used together during generation.

    Workflow:

    1. Instantiate with a base model checkpoint and/or control objects.
    2. Call `steer()` once to apply all controls in order (structural, input, state, output).
    3. Use `generate()` for inference with steering applied, accepting str, list[str], chat, or tensor input.

    Args:
        model_name_or_path (str or pathlib.Path, optional): HuggingFace model hub name or local directory.
            Required when `lazy_init=False`. Ignored when `lazy_init=True` and the structural
            control returns a model.
        controls (Sequence[StructuralControl | StateControl | InputControl | OutputControl], optional):
            Controls for the steering pipeline. Every category accepts any number of controls,
            applied in list order. The output category additionally accepts at most one enabled
            `DecodingDriver` (the decode loop does not compose). Omitted input/structural categories
            fall back to no-op controls; an omitted output category uses the default decoding driver
            (`model.generate`).
        tokenizer_name_or_path (str, optional): Tokenizer location. Defaults to `model_name_or_path`.
        device_map (str or dict[str, int], optional): Device map (passed to
            `transformers.AutoModelForCausalLM.from_pretrained`). Defaults to `"auto"`.
            Cannot be used together with `device` parameter.
        device (torch.device, str, optional): Device (passed to model's `.to()` method).
            When specified, `device_map` must remain at its default value of `"auto"`.
        hf_model_kwargs (dict, optional): Extra keyword arguments passed to
            `transformers.AutoModelForCausalLM.from_pretrained`.
        trust_remote_code (bool, optional): Trust remote code when loading the tokenizer. Defaults to
            `False`. To trust remote code for the model, pass `trust_remote_code=True` via `hf_model_kwargs`.
        lazy_init (bool, optional): If `True`, defers loading the base model until `steer()` time.
            Useful when a `StructuralControl` will itself load or create the final weights
            (e.g., MergeKit). When `False`, the model is loaded during `SteeringPipeline`
            construction. Defaults to `False`. On engine backends the base weights are never
            needed up front, so the flag is accepted and inert.
        backend (BackendSpec | str, optional): The pipeline's backend. Defaults to the
            in-process Hugging Face backend described by this pipeline's own construction
            arguments. A `"vllm"` spec boots an offline engine (requires the `vllm` extra) and
            a `"vllm-serve"` spec targets a running vLLM server; `check()` reports which
            enabled controls the backend supports, plus the steer plan, before anything
            executes.
        fit (str, optional): Fit venue policy. `"auto"` (default) fits through the backend's
            session where its capture surface serves the fit; `"in_process"` forces every fit
            onto a staged in-process model, for engine-independent numerics.

    Raises:
        RuntimeError: If `generate()` is called before `steer()`
        ValueError: If more than one enabled `DecodingDriver` is supplied or required arguments are missing

    Note:

    - Every category accepts multiple controls, applied in list order. Omitted input/structural
        categories use no-op defaults; an omitted output category uses the pipeline's default
        decoding driver.
    - Controls with a `tokenizer` attribute will have it auto-injected if not already set
    - On engine backends, `model` is non-None only while the staged in-process model exists
        during the steer phase; the stage is freed before the engine boots.

    For the state category, list order in `controls` defines the composition surface. List order sets
    `steer()` order, hook registration order, and execution order for hooks on the same module. PyTorch
    forward hooks chain, so a later hook receives the previous hook's returned output and pre-hooks chain
    likewise on inputs. Combinations that do not commute, such as ablate after add versus add after
    ablate, therefore produce order-sensitive results. A gated or condition-scoring control placed after
    another observes activations already edited at earlier layers by upstream list entries.

    For the input category, adaptation runs in two phases. On chat input, every control's
    `adapt_messages` runs in list order over the message batch (each non-None return feeds the next
    control); the result is templated and tokenized once, then every control whose `adapt_messages`
    returned None runs its token-level `adapt` in list order over the token stream. On text/tensor
    input there is no message phase; every control's `adapt` runs in list order. List order is
    authoritative within each phase, but the message phase structurally precedes the token phase:
    with `[TokenOnlyControl, MessageLevelControl]` on chat input, the message-level control's effect
    lands first even though it is listed second (tokens do not exist before templating). Place
    semantic rewriters (e.g. `PRewrite`, `CPO`, `GEPA`) before surface formatting (e.g. `FewShot`),
    since a rewriter trained on bare instructions degrades on exemplar-prepended input.

    For the structural category, `steer()` threads the model through the controls in list order:
    each control receives the previous control's returned model (and the possibly mutated tokenizer).
    Nothing implicit happens between stages (no adapter merging, no embedding-resize reconciliation);
    stage compatibility is the caller's responsibility.
    """

    # construction args
    model_name_or_path: str | Path | None = None
    controls: Sequence[StructuralControl | StateControl | InputControl | OutputControl] = ()
    tokenizer_name_or_path: str | None = None
    device_map: str | dict[str, int] | int | torch.device | None = "auto"
    device: torch.device | str | None = None
    hf_model_kwargs: dict = field(default_factory=dict)
    trust_remote_code: bool = False
    lazy_init: bool = False
    backend: BackendSpec | str | None = None
    fit: Literal["auto", "in_process"] = "auto"

    # lazy‑filled fields
    model: PreTrainedModel | None = field(init=False, default=None)
    tokenizer: AutoTokenizer | None = field(init=False, default=None)
    _support_report: SupportReport | None = field(init=False, default=None, repr=False)
    _backends: dict = field(init=False, default_factory=dict, repr=False)
    _structural_artifacts: tuple = field(init=False, default=(), repr=False)
    _lowered_state: dict = field(init=False, default_factory=dict, repr=False)

    structural_controls: list[StructuralControl] = field(init=False)
    input_controls: list[InputControl] = field(init=False)
    state_controls: list[StateControl] = field(init=False)
    output_controls: list[OutputControl] = field(init=False)

    _is_steered: bool = field(default=False, init=False, repr=False)
    _warned_tensor_with_adapt_messages: bool = field(default=False, init=False, repr=False)
    _warned_duplicate_bos: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:

        # sort/validate the supplied steering methods
        controls_merged = merge_controls(self.controls)
        self.structural_controls = controls_merged["structural_controls"]
        self.input_controls = controls_merged["input_controls"]
        self.state_controls = controls_merged["state_controls"]
        self.output_controls = controls_merged["output_controls"]

        if self.fit not in ("auto", "in_process"):
            raise ValueError(f"fit must be 'auto' or 'in_process'; got {self.fit!r}.")

        spec = self._resolve_backend_spec(self.backend)
        if spec.kind == "huggingface":
            # in-process backend: eager load unless lazy_init
            if not self.lazy_init:
                if self.model_name_or_path is None:
                    raise ValueError("`model_name_or_path` must be provided when lazy_init=False")
                self._load_in_process_model(self.model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path or self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                )
                self.tokenizer = ensure_pad_token(self.tokenizer)
            else:
                if isinstance(self.tokenizer_name_or_path, (str, Path)):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_name_or_path,
                        trust_remote_code=self.trust_remote_code
                    )
                    self.tokenizer = ensure_pad_token(self.tokenizer)
        else:
            # engine backend: the constructor never loads the model, and a client-side
            # tokenizer resolves at steer() so probe pipelines stay free of I/O
            if isinstance(self.tokenizer_name_or_path, (str, Path)):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                )
                self.tokenizer = ensure_pad_token(self.tokenizer)

        self._inject_tokenizer()

    def _resolve_client_tokenizer(self, spec: BackendSpec) -> None:
        """Resolve the client-side tokenizer for an engine backend, if not already set.

        The source is `tokenizer_name_or_path`, the spec's `tokenizer_name_or_path` option,
        the spec's model reference, or `model_name_or_path`, in that order. Leaves the
        tokenizer unset when no source is available or the source does not resolve, so the
        backend's own error (a missing optional dependency, a bad model reference) surfaces
        as the authoritative failure.
        """
        if self.tokenizer is not None:
            return
        source = (
            self.tokenizer_name_or_path
            or spec.get_option("tokenizer_name_or_path")
            or spec.model
            or (str(self.model_name_or_path) if self.model_name_or_path is not None else None)
        )
        if source is None:
            return
        try:
            self.tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(
                source, trust_remote_code=self.trust_remote_code,
            ))
        except Exception:
            logger.debug("Client tokenizer resolution from %r failed.", source, exc_info=True)
            return
        self._inject_tokenizer()

    def _load_in_process_model(self, model_ref: str | Path) -> None:
        """Load `model_ref` with the constructor's placement knobs and bind it as `model`.

        Raises:
            ValueError: If both `device` and a non-default `device_map` are set.
        """
        if self.device is not None and self.device_map != "auto":
            raise ValueError("Cannot specify both `device` and `device_map`.")

        if self.device is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                **self.hf_model_kwargs,
            )
            self.model = self.model.to(self.device)
            self.device = self.model.device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                device_map=self.device_map,
                **self.hf_model_kwargs,
            )
            self.device = self.model.device

    @property
    def supports_batching(self) -> bool:
        """Return True if all enabled controls in this pipeline are batch-safe.

        The default decoding driver is batch-safe, so an empty `output_controls` list is vacuously
        true and does not constrain batching.
        """
        controls = (
            *self.structural_controls,
            *self.input_controls,
            *self.state_controls,
            *self.output_controls,
        )
        return all(
            getattr(control, "supports_batching", False)
            for control in controls
            if control.enabled
        )

    def _inject_tokenizer(self) -> None:
        """Attach the pipeline tokenizer to every control exposing an unset `tokenizer`."""
        if self.tokenizer is None:
            return
        for control in (
            *self.structural_controls, *self.input_controls,
            *self.state_controls, *self.output_controls,
        ):
            if hasattr(control, "tokenizer") and getattr(control, "tokenizer", None) is None:
                control.tokenizer = self.tokenizer

    def _warn_on_runtime_kwargs_overlap(self) -> None:
        """Warn (UserWarning, once) when two or more enabled controls declare the same
        `RUNTIME_KWARGS_SCHEMA` variable name.

        All controls read from the single `runtime_kwargs` dict passed to `generate()`, so a shared
        name means one value feeds several controls. Sharing can be intentional, hence a warning
        rather than an error.
        """
        declared: dict[str, list[str]] = {}
        controls = (*self.structural_controls, *self.input_controls, *self.state_controls, *self.output_controls)
        for control in controls:
            if not control.enabled:
                continue
            for entry in getattr(control, "RUNTIME_KWARGS_SCHEMA", []):
                name = entry.get("name")
                if name:
                    declared.setdefault(name, []).append(type(control).__name__)
        overlaps = {name: owners for name, owners in declared.items() if len(owners) > 1}
        if overlaps:
            details = "; ".join(
                f"'{name}' is declared by {', '.join(owners)}" for name, owners in overlaps.items()
            )
            warnings.warn(
                f"Multiple controls declare the same runtime_kwargs variable: {details}. "
                "All controls read from the one runtime_kwargs dict, so these controls will share "
                "the same value at inference time.",
                UserWarning,
            )

    def _resolve_backend_spec(
        self, value: BackendSpec | str | None, param_name: str = "backend",
    ) -> BackendSpec:
        """Resolve a backend argument to a `BackendSpec`.

        None and `"huggingface"` resolve to the implicit in-process spec derived from this
        pipeline's construction arguments; another known kind name resolves to a bare spec of
        that kind carrying the pipeline's model reference; a `BackendSpec` passes through.

        Args:
            value: The backend argument to resolve.
            param_name: The caller's parameter name, used in the `TypeError` message so it names
                the argument the caller passed.

        Raises:
            TypeError: If `value` is neither None, a known kind name, nor a `BackendSpec`.
        """
        if isinstance(value, BackendSpec):
            return value
        model = str(self.model_name_or_path) if self.model_name_or_path is not None else None
        if value is None or value == "huggingface":
            return BackendSpec(
                kind="huggingface",
                model=model,
                options={
                    "hf_model_kwargs": self.hf_model_kwargs,
                    "device_map": self.device_map,
                    "trust_remote_code": self.trust_remote_code,
                    "tokenizer_name_or_path": self.tokenizer_name_or_path,
                },
            )
        if isinstance(value, str) and value in KNOWN_BACKEND_KINDS:
            return BackendSpec(kind=value, model=model)
        raise TypeError(
            f"{param_name} must be a BackendSpec or one of {', '.join(KNOWN_BACKEND_KINDS)}; got {value!r}."
        )

    def _backend_for(self, spec: BackendSpec):
        """The backend instance for `spec`, constructed on first use and cached by spec.

        The `"huggingface"` kind adopts this pipeline's live model and tokenizer (providers are
        re-read per access, so structural replacement stays visible); other kinds construct from
        the spec and receive the pipeline's structural artifacts to serve.
        """
        backend = self._backends.get(spec)
        if backend is None:
            backend_cls = resolve_backend_class(spec)
            if spec.kind == "huggingface":
                backend = backend_cls.adopt(spec, lambda: self.model, lambda: self.tokenizer)
            else:
                backend = backend_cls(spec, artifacts=self._structural_artifacts)
            self._backends[spec] = backend
        return backend

    def release_backends(self) -> None:
        """Release every backend this pipeline constructed and empty the cache.

        Subsequent operations construct fresh backends against the same specs. Lowered
        intervention entries and staged artifacts persist, so a released pipeline remains usable
        at the cost of re-booting engines on next use. Release is idempotent. Engine-owning
        backends shut down deterministically.
        """
        backends, self._backends = self._backends, {}
        for backend in backends.values():
            try:
                backend.release()
            except Exception:
                logger.warning("Backend release failed", exc_info=True)

    def __enter__(self) -> "SteeringPipeline":
        """Return the pipeline for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Release the pipeline's backends on exit; does not suppress exceptions."""
        self.release_backends()

    def check(self, backend: BackendSpec | str | None = None) -> SupportReport:
        """Evaluate every enabled control's backend requirements and compute the steer plan.

        Runs automatically at `steer()` (which raises on generate-phase failures) and is
        callable standalone against any backend. Disabled controls, including the pipeline's
        default identity controls, never gate a backend and do not appear in the report. The
        returned report's `plan` states, per enabled control and per fit artifact, where the
        steer phase will run each step; the plan is a pure function of the declarations and
        the spec, so the same configuration always yields the same verdicts and plan.

        Args:
            backend: Backend to evaluate against. Defaults to the pipeline's `backend`, then
                to the implicit in-process backend.

        Returns:
            The `SupportReport` with the steer plan and one failure per unsupported
            (control, phase) pair.
        """
        spec = self._resolve_backend_spec(backend if backend is not None else self.backend)
        capabilities = capabilities_for_spec(spec)
        controls = (*self.structural_controls, *self.input_controls, *self.state_controls, *self.output_controls)
        report = evaluate_support(controls, spec, capabilities)
        plan = self._compute_plan(controls, spec, capabilities)
        return dataclasses.replace(report, plan=plan)

    def _compute_plan(
        self,
        controls: Sequence[Any],
        spec: BackendSpec,
        capabilities: BackendCapabilities,
    ) -> SteerPlan:
        """The steer plan for `controls` on `spec`; pure, with no I/O and no weights.

        Venues on the Hugging Face backend are all `"live"`. On engine backends, `MODULE`
        steps stage, `FACTS` and `ROLLOUTS` steps run on the engine session, and `CAPTURE`
        steps run on the session when the spec statically advertises `HIDDEN_CAPTURE` and
        `fit == "auto"`, else on the stage. A fit's venue is its owning step's. A calibrated
        fit whose venue departs from its engine read venue (the `fit="in_process"` flag, or a
        spec whose capture surface is statically absent) contributes a notice.
        """
        in_process = spec.kind == "huggingface"
        capture_advertised = Capability.HIDDEN_CAPTURE in capabilities.atoms
        steps: list[PlannedStep] = []
        fits: list[PlannedFit] = []
        notices: list[str] = []
        for control in controls:
            if not getattr(control, "enabled", True):
                continue
            access = control.steer_access()
            if in_process:
                venue = "live"
            elif access >= ModelAccess.MODULE:
                venue = "stage"
            elif access == ModelAccess.CAPTURE:
                venue = "session" if (capture_advertised and self.fit == "auto") else "stage"
            else:
                venue = "session"
            name = type(control).__name__
            steps.append(PlannedStep(control=name, access=access, venue=venue))
            for artifact, artifact_class in control.steer_fits():
                fits.append(PlannedFit(
                    control=name, artifact=artifact, artifact_class=artifact_class, venue=venue,
                ))
                crossing = venue == "stage" and (self.fit == "in_process" or not capture_advertised)
                if artifact_class == "calibrated" and not in_process and crossing:
                    reason = (
                        "fit='in_process'" if self.fit == "in_process"
                        else "capture is unavailable on this backend"
                    )
                    notices.append(
                        f"{artifact} for {name} is scale-calibrated and will be read on "
                        f"backend kind '{spec.kind}', but is fitted in process ({reason}); "
                        "calibrated thresholds may shift across execution boundaries."
                    )
        return SteerPlan(
            steps=tuple(steps),
            fits=tuple(fits),
            stages=any(step.venue == "stage" for step in steps),
            notices=tuple(notices),
        )

    def steer(self, **steer_kwargs) -> None:
        """Apply all steering controls per the steer plan.

        Executes each control's steer() method in a fixed bottom-up order: structural -> input -> state -> output,
        and in list order within each category. If any control's steer() method returns a
        PreTrainedModel instance, it replaces the current model for subsequent controls, so
        structural controls thread the model through in list order.

        Before any control runs, `check()` evaluates the configured backend and raises on any
        generate-phase failure. Each control's `steer()` receives `session=`, a session scoped
        to its declared `steer_access()`, unless the caller supplied its own `session` keyword,
        and receives the live model only at `ModelAccess.MODULE`. On the Hugging Face backend
        every step runs against the live model in one phase. On engine backends the plan's
        stage-venued steps run first on a temporary in-process model that is freed before the
        engine boots (exported artifacts are the handoff), then the session-venued steps run
        through the engine session. The only channel between one control's steer and another's
        is the pipeline model, and every control that can touch it runs in the stage phase, so
        per-phase global order preserves the composition semantics of the single-phase order.

        A failed steer releases any backends it constructed before re-raising, so it does not
        leave an engine behind and a retried steer re-boots.

        Args:
            **steer_kwargs: Keyword arguments passed to all control steer() methods

        Warns:
            UserWarning: If two or more enabled controls declare the same `RUNTIME_KWARGS_SCHEMA`
                variable name, if a calibrated fit is fitted in process while its artifact is
                read on an engine, or if engine capture fails the steer-time smoke test and
                fitting degrades to a staged in-process model.

        Raises:
            RuntimeError: If called more than once, no model is available after steering, or
                the staged in-process model was retained past the steer stage.
            UnsupportedPipelineError: If any enabled control is unsupported at the generate
                phase on the configured backend.
            ModuleNotFoundError: If a configured backend kind requires an optional dependency
                that is not installed (e.g. the `vllm` extra).
        """
        if self._is_steered:
            return

        try:
            self._warn_on_runtime_kwargs_overlap()

            spec = self._resolve_backend_spec(self.backend)
            report = self.check()
            report.raise_for("generate")
            self._support_report = report

            if spec.kind == "huggingface":
                self._steer_in_process(spec, report.plan, steer_kwargs)
            else:
                self._resolve_client_tokenizer(spec)
                self._steer_on_engine(spec, report.plan, steer_kwargs)

            if self.tokenizer is None:
                repo = getattr(self.model, "name_or_path", None)
                source = repo or self._structural_out_path()
                if source is None:
                    raise RuntimeError("Failed to resolve tokenizer post‑steer.")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        source,
                        trust_remote_code=self.trust_remote_code,
                    )
                    self.tokenizer = ensure_pad_token(self.tokenizer)

                except Exception as exception:
                    raise RuntimeError("Failed to resolve tokenizer post‑steer.") from exception

            self._inject_tokenizer()

            # a spec-consuming backend gets every enabled control's interventions lowered now,
            # so inexpressible configurations fail before the first generate and artifacts are
            # staged once
            self._lower_state_controls(spec)
        except Exception:
            self.release_backends()
            raise

        # return steered pipeline
        self._is_steered = True

    def _enabled_controls(self) -> list:
        """Enabled controls in global steer order (structural, input, state, output)."""
        return [
            control
            for control in (
                *self.structural_controls, *self.input_controls,
                *self.state_controls, *self.output_controls,
            )
            if getattr(control, "enabled", True)
        ]

    def _run_control_steer(self, control, access: ModelAccess, venue_session, steer_kwargs) -> None:
        """Run one control's steer with a session scoped to `access` and the model gated by it.

        The live model travels only through the `model=` argument, and only at
        `ModelAccess.MODULE`. A caller-supplied `session` keyword overrides the scoped
        session. A returned `nn.Module` replaces the pipeline model for subsequent controls.
        """
        steer_fn = getattr(control, "steer", None)
        if not callable(steer_fn):
            return
        kwargs = steer_kwargs
        if "session" not in kwargs:
            scoped = ScopedSession(venue_session, type(control).__name__, access)
            kwargs = {**kwargs, "session": scoped}
        model = self.model if access >= ModelAccess.MODULE else None
        maybe_new_model = steer_fn(model, tokenizer=self.tokenizer, **kwargs)
        if isinstance(maybe_new_model, nn.Module):
            self.model = maybe_new_model

    def _steer_in_process(self, spec: BackendSpec, plan: SteerPlan, steer_kwargs: dict) -> None:
        """Run every enabled control's steer against the live model, in one phase."""
        backend = self._backend_for(spec)
        controls = self._enabled_controls()
        with backend.open_session() as session:
            for control, step in zip(controls, plan.steps):
                self._run_control_steer(control, step.access, session, steer_kwargs)

        self._structural_artifacts = self._collect_structural_artifacts(spec)

        if self.model is None:
            raise RuntimeError(
                "No model is available after steering. Either provide a base model (lazy_init=False) or ensure a "
                "`StructuralControl` returns one."
            )

    def _steer_on_engine(self, spec: BackendSpec, plan: SteerPlan, steer_kwargs: dict) -> None:
        """Run the staged steer: stage-venued steps on a temporary in-process model, freed
        before the engine boots, then session-venued steps through the engine session.

        When the plan assigned any fit to engine capture, one single-prompt capture smoke test
        runs before any session-venued steer; on failure the affected controls' venues revise
        to the stage (the engine is released first, so weights and engine never coexist) and
        the remaining session-venued steers run against a re-booted engine. No control's
        steer() ever runs twice.
        """
        controls = self._enabled_controls()
        steps = {id(control): step for control, step in zip(controls, plan.steps)}
        stage_controls = [c for c in controls if steps[id(c)].venue == "stage"]
        session_controls = [c for c in controls if steps[id(c)].venue == "session"]

        if plan.stages:
            for notice in plan.notices:
                warnings.warn(notice, UserWarning)
            self._run_stage(spec, stage_controls, steps, steer_kwargs)

        backend = self._backend_for(spec)
        session_fitters = {planned.control for planned in plan.fits if planned.venue == "session"}
        fit_controls = [c for c in session_controls if type(c).__name__ in session_fitters]

        session = backend.open_session()
        try:
            if fit_controls:
                error = self._capture_smoke_failure(session)
                if error is not None:
                    warnings.warn(
                        f"Hidden-state capture on backend kind '{spec.kind}' failed at steer "
                        f"({error}); fitting degrades to a staged in-process model. Set "
                        "fit='in_process' to plan this from the start.",
                        UserWarning,
                    )
                    session.close()
                    self.release_backends()
                    self._run_stage(spec, fit_controls, steps, steer_kwargs)
                    session_controls = [c for c in session_controls if c not in fit_controls]
                    backend = self._backend_for(spec)
                    session = backend.open_session()
            for control in session_controls:
                self._run_control_steer(control, steps[id(control)].access, session, steer_kwargs)
        finally:
            session.close()

    def _capture_smoke_failure(self, session) -> str | None:
        """Issue one single-prompt capture through `session`; the error text on failure."""
        tokenizer = getattr(session, "tokenizer", None) or self.tokenizer
        token_id = 0
        for attribute in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(tokenizer, attribute, None)
            if value is not None:
                token_id = int(value)
                break
        prompt = PreparedPrompt.from_token_ids(torch.tensor([[token_id]], dtype=torch.long))
        try:
            session.capture([prompt], layers=[0], mode="last_token", location="layer_output")
        except Exception as error:
            return str(error)
        return None

    def _run_stage(self, spec: BackendSpec, stage_controls, steps, steer_kwargs: dict) -> None:
        """Load the staged in-process model, run `stage_controls`' steers on it, collect
        structural artifacts, and free the stage.

        The stage is configured by the constructor's placement knobs and loads
        `spec.model` (or `model_name_or_path`). Structural returns thread through the stage,
        and the exported artifacts are the handoff to the engine.

        Raises:
            RuntimeError: If no model reference is available to load the stage from, or the
                staged model was retained past the stage by a control.
        """
        model_ref = spec.model or (
            str(self.model_name_or_path) if self.model_name_or_path is not None else None
        )
        if model_ref is None:
            raise RuntimeError(
                "The steer plan stages an in-process model, but neither the backend spec nor "
                "`model_name_or_path` names a model to load."
            )
        stage_spec = BackendSpec(
            kind="huggingface",
            model=model_ref,
            options={
                "hf_model_kwargs": self.hf_model_kwargs,
                "device_map": self.device_map,
                "trust_remote_code": self.trust_remote_code,
                "tokenizer_name_or_path": self.tokenizer_name_or_path,
            },
        )
        self._load_in_process_model(model_ref)
        stage_backend = resolve_backend_class(stage_spec).adopt(
            stage_spec, lambda: self.model, lambda: self.tokenizer,
        )
        try:
            with stage_backend.open_session() as stage_session:
                for control in stage_controls:
                    self._run_control_steer(
                        control, steps[id(control)].access, stage_session, steer_kwargs,
                    )
            if not self._structural_artifacts:
                self._structural_artifacts = self._collect_structural_artifacts(stage_spec)
        finally:
            stage_backend.release()
        self._free_stage()

    def _free_stage(self) -> None:
        """Free the staged in-process model and verify the weights are actually gone.

        Raises:
            RuntimeError: If a control retained the staged model past the stage; the message
                names the retaining controls where identifiable.
        """
        model = self.model
        if model is None:
            return
        ref = weakref.ref(model)
        self.model = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        referent = ref()
        if referent is None:
            return
        holders = self._find_model_holders(referent)
        names = ", ".join(holders) if holders else "an unidentified holder"
        raise RuntimeError(
            f"The staged in-process model was retained past the steer stage by: {names}. "
            "Controls supported at generate on this backend must not hold the pipeline model "
            "beyond steer(); release the reference in steer() or cleanup(), or require "
            "Capability.IN_PROCESS_TORCH at generate."
        )

    def _find_model_holders(self, referent) -> list[str]:
        """Controls holding `referent` in their instance attributes (one level) or in a bound
        intervention's transform or gate attributes."""

        def instance_values(obj):
            try:
                return list(vars(obj).values())
            except TypeError:
                slots = getattr(type(obj), "__slots__", ())
                return [getattr(obj, name, None) for name in slots]

        holders: list[str] = []
        for control in self._enabled_controls():
            found = any(value is referent for value in instance_values(control))
            if not found:
                for intervention in getattr(control, "interventions", ()) or ():
                    for slot in (
                        getattr(intervention, "transform", None),
                        getattr(intervention, "gate", None),
                    ):
                        if slot is not None and any(
                            value is referent for value in instance_values(slot)
                        ):
                            found = True
            if found:
                holders.append(type(control).__name__)
        return holders

    def _collect_structural_artifacts(self, spec: BackendSpec) -> tuple[Artifact, ...]:
        """Enabled structural controls' steer-time artifacts, provenance-stamped.

        Provenance carries the producing venue's spec hash (the stage spec on engine
        backends) and, when a live model is present, its fingerprint.
        """
        artifacts: list[Artifact] = []
        for control in self.structural_controls:
            if not control.enabled:
                continue
            exporter = getattr(control, "export_artifact", None)
            artifact = exporter() if callable(exporter) else None
            if artifact is not None:
                artifacts.append(artifact)
        if not artifacts:
            return ()

        model_fingerprint = None
        if self.model is not None:
            from aisteer360.algorithms.core.internals.fingerprint import (
                model_fingerprint as compute_model_fingerprint,
            )
            try:
                model_fingerprint = compute_model_fingerprint(self.model)
            except Exception:
                logger.debug("Model fingerprint unavailable for artifact provenance.")
        provenance = ArtifactProvenance(
            backend_spec_hash=spec.spec_hash,
            model_fingerprint=model_fingerprint,
        )
        return tuple(dataclasses.replace(artifact, provenance=provenance) for artifact in artifacts)

    def _structural_out_path(self) -> Path | None:
        """The last structural control's non-empty `args.out_path`, as a tokenizer-directory fallback.

        Scans `structural_controls` in reverse so the most recently produced artifact wins; controls
        without an `args` attribute or without `out_path` are skipped. When more than one control
        defines `out_path`, logs at info level which path won.
        """
        candidates = [
            (type(control).__name__, out_path)
            for control in self.structural_controls
            if (out_path := getattr(getattr(control, "args", None), "out_path", None))
        ]
        if not candidates:
            return None
        winner_name, winner_path = candidates[-1]
        if len(candidates) > 1:
            logger.info(
                "Multiple structural controls define out_path (%s); using %s from %s.",
                ", ".join(name for name, _ in candidates), winner_path, winner_name,
            )
        return Path(winner_path)

    def _prepare_inputs(
            self,
            input_ids: list[int] | torch.LongTensor,
            attention_mask: torch.Tensor | None,
            runtime_kwargs: dict | None,
            message_handled: frozenset[int] = frozenset(),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the token-level input-control chain and normalize input tensors.

        Runs each input control's `adapt` in list order (each control receives the previous
        control's output), then ensures both input_ids and attention_mask are properly shaped
        tensors on the correct device.

        Args:
            input_ids: Input token IDs as list or tensor [seq_len] or [batch, seq_len]
            attention_mask: Optional attention mask matching input_ids shape
            runtime_kwargs: Per-call parameters for input controls
            message_handled: `id()`s of input controls whose `adapt_messages` already performed the
                adaptation before tokenization for this call; their token-level `adapt` is skipped so
                no control is applied twice to the same prompt.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (steered_input_ids, attention_mask), both as 2D tensors on model device
        """
        runtime_kwargs = runtime_kwargs or {}
        device = self.model.device if self.model is not None else torch.device("cpu")

        # token-phase chain (controls already handled at message level are skipped)
        steered_input_ids = input_ids
        for control in self.input_controls:
            if id(control) in message_handled:
                continue
            steered_input_ids = control.adapt(
                steered_input_ids,
                runtime_kwargs=runtime_kwargs,
            )

        # normalize input_ids to 2D tensor
        if isinstance(steered_input_ids, list):
            steered_input_ids = torch.tensor(steered_input_ids, dtype=torch.long)
        if steered_input_ids.ndim == 1:
            steered_input_ids = steered_input_ids.unsqueeze(0)
        steered_input_ids = steered_input_ids.to(device)

        # normalize attention_mask
        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            # rebuild if length mismatch after input control transformation
            if attention_mask.shape[-1] != steered_input_ids.shape[-1]:
                attention_mask = None

        if attention_mask is None:
            if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
                attention_mask = infer_attention_mask_from_ids(steered_input_ids, self.tokenizer.pad_token_id)
            else:
                attention_mask = torch.ones_like(steered_input_ids, dtype=torch.long)

        attention_mask = attention_mask.to(dtype=steered_input_ids.dtype, device=device)

        self._warned_duplicate_bos = warn_if_duplicate_bos(
            steered_input_ids, attention_mask, self.tokenizer, self._warned_duplicate_bos
        )

        return steered_input_ids, attention_mask

    def _collect_state_entries(
            self,
            steered_input_ids: torch.Tensor,
            runtime_kwargs: dict | None,
            attention_mask: torch.Tensor | None = None,
            **kwargs,
    ) -> tuple[HookEntry, ...]:
        """Collect every enabled state control's hooks for the current logical generation.

        Hooks are per-generation artifacts built here, once per logical generation: they close
        over the prompt anchor, sized gate state, and a fresh position clock. They travel only
        as `HookEntry` contributions; the session that executes forwards owns registration, and
        controls are never mutated.

        Args:
            steered_input_ids: Input token IDs after input control transformation
            runtime_kwargs: Per-call parameters for state controls
            attention_mask: The prompt attention mask matching `steered_input_ids`. Forwarded to
                hook construction so condition scorers see the real (non-pad) prompt tokens
                rather than re-deriving a pad mask by token identity.
            **kwargs: Additional arguments passed to hook construction

        Returns:
            One `HookEntry` per enabled state control, in controls-list order.
        """
        spec = self._resolve_backend_spec(self.backend)
        capabilities = capabilities_for_spec(spec)
        if Capability.IN_PROCESS_TORCH not in capabilities.atoms:
            # spec-consuming backend: entries come from the steer-time lowering cache, filled
            # lazily for a control enabled after steer()
            entries = []
            for state_control in self.state_controls:
                if not state_control.enabled:
                    continue
                entry = self._lowered_state.get(id(state_control))
                if entry is None:
                    backend = self._backend_for(spec)
                    served = ((getattr(backend, "_discovery", None) or {}).get("model") or {})
                    payloads: dict = {}
                    entry = self._lower_control(
                        state_control, capabilities.intervention_kinds, served, payloads,
                    )
                    backend.stage_artifacts(payloads)
                    self._lowered_state[id(state_control)] = entry
                entries.append(entry)
            return tuple(entries)

        entries = []
        for state_control in self.state_controls:
            if not state_control.enabled:
                continue
            hooks = state_control.get_hooks(
                steered_input_ids, runtime_kwargs, attention_mask=attention_mask, model=self.model, **kwargs
            )
            entries.append(HookEntry(hooks=hooks))
        return tuple(entries)

    def _per_item_state_entries(
            self,
            steered_input_ids: torch.Tensor,
            steered_attention_mask: torch.Tensor,
            runtime_kwargs: dict | None,
            **kwargs,
    ) -> list[tuple[HookEntry, ...]]:
        """Per-row state entries computed by per-call control clones.

        Distinct per-item derived seeds force the in-process session onto its serial path, where
        each row runs its own forward. Hooks computed once on the batch hold batch-sized position
        and gate state, so each row instead gets hooks computed by a fresh clone on that row's
        prompt tensors.

        Args:
            steered_input_ids: Adapted prompt ids of shape `[batch, seq_len]`.
            steered_attention_mask: Attention mask matching `steered_input_ids`.
            runtime_kwargs: Per-call parameters for state controls.
            **kwargs: Additional arguments passed to `get_hooks()`.

        Returns:
            One tuple of `HookEntry` per row, each in controls-list order.
        """
        rows: list[tuple[HookEntry, ...]] = []
        for index in range(steered_input_ids.size(0)):
            entries: list[HookEntry] = []
            for state_control in self.state_controls:
                if not state_control.enabled:
                    continue
                clone = state_control.clone_for_call()
                hooks = clone.get_hooks(
                    steered_input_ids[index:index + 1],
                    runtime_kwargs,
                    attention_mask=steered_attention_mask[index:index + 1],
                    model=self.model,
                    **kwargs,
                )
                entries.append(HookEntry(hooks=hooks))
            rows.append(tuple(entries))
        return rows

    def _lower_state_controls(self, spec: BackendSpec) -> None:
        """Lower every enabled state control's interventions for a spec-consuming backend,
        cache the entries, and stage their artifacts.

        Runs at the end of `steer()` when the backend executes interventions as
        specs rather than in-process hooks. Specs are per-steer artifacts: the worker anchors
        positions per request server-side and the spec is prompt-independent by construction,
        so one lowering serves every subsequent generation. Each spec is verified against the
        backend's negotiated kinds (the intersection of the static tables and discovery), and
        a control's steering-artifact provenance is cross-checked against the served model's
        when the backend carries a discovery payload.

        Raises:
            UnsupportedOperationError: If an enabled control's configuration has no wire form
                (the failure names the control, the intervention, and the reason), or its spec
                requires a kind the backend does not advertise.
        """
        capabilities = capabilities_for_spec(spec)
        if Capability.IN_PROCESS_TORCH in capabilities.atoms:
            return
        if Capability.INTERVENTION_SPECS not in capabilities.atoms:
            return
        enabled = [c for c in self.state_controls if c.enabled]
        if not enabled:
            return

        backend = self._backend_for(spec)
        advertised = capabilities.intervention_kinds
        served_model = ((getattr(backend, "_discovery", None) or {}).get("model") or {})
        payloads: dict = {}
        for state_control in enabled:
            self._lowered_state[id(state_control)] = self._lower_control(
                state_control, advertised, served_model, payloads,
            )
        backend.stage_artifacts(payloads)

    def _lower_control(self, state_control, advertised, served_model, payloads) -> InterventionEntry:
        """Lower one control to an `InterventionEntry`, verifying kinds and provenance."""
        if served_model:
            self._warn_on_provenance_mismatch(state_control, served_model)
        exporter = getattr(state_control, "export_intervention_spec", None)
        spec = exporter() if callable(exporter) else None
        if spec is None:
            reason = self._lowering_failure_reason(state_control)
            raise UnsupportedOperationError(
                f"{type(state_control).__name__} has no intervention-spec form for this "
                f"configuration ({reason}); run this pipeline on the huggingface backend."
            )
        required = spec.required_kinds()
        if advertised is None or not advertised.contains(required):
            missing = sorted(
                (required.transforms - (advertised.transforms if advertised else frozenset()))
                | (required.modifiers - (advertised.modifiers if advertised else frozenset()))
                | (required.scopes - (advertised.scopes if advertised else frozenset()))
                | (required.gates - (advertised.gates if advertised else frozenset()))
            )
            raise UnsupportedOperationError(
                f"{type(state_control).__name__} requires intervention kind(s) "
                f"{', '.join(missing)} that the serving backend does not advertise; update the "
                "server's vllm_hook_plugins or run this pipeline on the huggingface backend."
            )
        payloads.update(spec.artifacts)
        return InterventionEntry(spec=spec)

    @staticmethod
    def _rollout_entries(state_entries, steered_input_ids, steered_attention_mask) -> tuple:
        """Rollout variants of the lowered entries for a driver on a spec-consuming backend.

        Prompt-relative scopes are rewritten to absolute positions at the generation's
        original prompt boundary. The rewrite needs one exact anchor per generation, and a
        rollout item cannot be traced back to a batch row, so uneven batches (rows whose true
        prompt lengths differ under padding) are refused. Conditional gates are refused too:
        a worker gate re-anchors its evidence at each rollout request's own prompt end, which
        would decide from generated text instead of the original prompt.

        Raises:
            UnsupportedOperationError: If the batch is uneven, a scope has no absolute rollout
                form, or an entry carries a conditional gate.
        """
        if steered_attention_mask is not None and not bool(steered_attention_mask.bool().all()):
            raise UnsupportedOperationError(
                "Driver rollouts on a spec-consuming backend need one exact prompt anchor per "
                "generation, and padded batch rows have per-row anchors; submit prompts of "
                "equal length, one prompt per call, or run this pipeline on the huggingface "
                "backend."
            )
        anchor = steered_input_ids.size(1)
        rollout_entries = []
        for entry in state_entries:
            if not isinstance(entry, InterventionEntry):
                rollout_entries.append(entry)
                continue
            if any(op.get("gate") is not None for op in entry.spec.to_wire()["ops"]):
                raise UnsupportedOperationError(
                    "Conditional gating has no rollout form on a spec-consuming backend: the "
                    "worker anchors gate evidence at each rollout request's own prompt end; "
                    "run gated controls under a decoding driver on the huggingface backend."
                )
            try:
                rewritten = remap_prompt_relative_scopes(entry.spec, anchor)
            except ValueError as error:
                raise UnsupportedOperationError(str(error)) from error
            rollout_entries.append(InterventionEntry(spec=rewritten))
        return tuple(rollout_entries)

    @staticmethod
    def _lowering_failure_reason(state_control) -> str:
        """Name the intervention (and hint) behind a lowering failure, for the raised error."""
        from aisteer360.algorithms.state_control._common.specs import (
            lower_interventions,
        )

        interventions = getattr(state_control, "interventions", ())
        num_layers = getattr(state_control, "_num_layers", None)
        if interventions and num_layers:
            for index, intervention in enumerate(interventions):
                if lower_interventions([intervention], num_layers=num_layers) is None:
                    core = type(intervention.transform).__name__
                    hint = getattr(state_control, "hook_only_hint", None)
                    detail = f"intervention {index} ({core}) has no wire form"
                    return f"{detail}; {hint}" if hint else detail
        hint = getattr(state_control, "hook_only_hint", None)
        return hint or "the configuration has no wire form"

    @staticmethod
    def _warn_on_provenance_mismatch(state_control, served_model: Mapping) -> None:
        """Warn when a control's steering-artifact fingerprints differ from the served model's.

        A served `chat_template_fingerprint` equal to the absent-template digest means the
        engine exposes no chat template; that key is skipped since a mismatch against it
        reflects exposure rather than divergence.
        """
        artifact = getattr(state_control, "_steering_vector", None)
        meta = getattr(artifact, "meta", None) or {}
        for key in ("config_fingerprint", "chat_template_fingerprint"):
            local = meta.get(key)
            remote = served_model.get(key)
            if not local or not remote or local == remote:
                continue
            if key == "chat_template_fingerprint" and is_absent_chat_template_fingerprint(remote):
                continue
            warnings.warn(
                f"{type(state_control).__name__}'s steering artifact records a {key} of "
                f"{local}, but the serving engine reports {remote}; the artifact was fitted "
                "on a different model or tokenizer configuration than the one serving it.",
                UserWarning,
            )

    def _processor_spec_contributions(
        self, runtime_kwargs: dict | None, inference_capabilities: BackendCapabilities,
    ) -> dict[int, "ProcessorSpecEntry"]:
        """Engine-hosted processor contributions from enabled output controls, keyed by `id()`.

        A control that returns a `ProcessorSpec` from `export_processor_spec` whose kind the
        backend serves is lowered for that call: the spec travels as a `ProcessorSpecEntry`
        and the control's live processor is not collected. The lowering choice is a ladder,
        highest supported rung first: normalized parameters, then engine-hosted specs, then
        live processors.
        """
        served = inference_capabilities.processor_kinds
        if served is None:
            return {}
        contributions: dict[int, ProcessorSpecEntry] = {}
        for control in self.output_controls:
            if not control.enabled:
                continue
            exporter = getattr(control, "export_processor_spec", None)
            spec = exporter(runtime_kwargs) if callable(exporter) else None
            if spec is not None and spec.kind in served.processors:
                contributions[id(control)] = ProcessorSpecEntry(spec=spec)
        return contributions

    def _constraint_contributions(self, runtime_kwargs: dict | None) -> dict[int, ConstraintSource]:
        """Declarative constraint sources from enabled output controls, keyed by `id()`.

        A control that returns a source from `export_constraint` is lowered for that call on
        backends hosting structured outputs natively: the source renders onto the engine's
        request parameters and the control's live processor is not collected.
        """
        contributions: dict[int, ConstraintSource] = {}
        for control in self.output_controls:
            if not control.enabled:
                continue
            exporter = getattr(control, "export_constraint", None)
            source = exporter(runtime_kwargs) if callable(exporter) else None
            if source is not None:
                contributions[id(control)] = source
        return contributions

    def _resolve_decoding_driver(self) -> DecodingDriver | None:
        """The sole enabled DecodingDriver, or None for the pipeline's default decode loop.

        merge_controls guarantees at most one enabled driver at construction; `enabled` is
        re-checked here so a driver disabled afterward falls back cleanly. The default loop
        (per-prompt items executed by the inference session) is pipeline infrastructure, not a
        phantom control.
        """
        for control in self.output_controls:
            if isinstance(control, DecodingDriver) and control.enabled:
                return control
        return None

    def _lowered_contributions(self, runtime_kwargs: dict | None) -> dict[int, Mapping]:
        """Sampling-expressible contributions from enabled output controls, keyed by `id()`.

        A control that returns a mapping from `export_generation_params` is lowered for this
        call: its contribution merges into the call's `GenerationParams` and its live processor
        and criteria hooks are not collected.
        """
        contributions: dict[int, Mapping] = {}
        for control in self.output_controls:
            if not control.enabled:
                continue
            exporter = getattr(control, "export_generation_params", None)
            contribution = exporter(runtime_kwargs) if callable(exporter) else None
            if contribution is not None:
                contributions[id(control)] = contribution
        return contributions

    def _collect_output_entries(
        self, input_ids, runtime_kwargs, attention_mask=None, for_scoring=False,
        skip_ids=frozenset(), **kwargs,
    ) -> tuple[StackEntry, ...]:
        """One `StackEntry` per contributing output control, in controls-list order.

        With `for_scoring=True`, only `include_in_scoring` controls contribute processors and
        criteria are skipped (there is no loop to stop). Controls whose `id()` is in `skip_ids`
        (lowered to generation parameters for this call) contribute nothing. Controls
        contributing neither processors nor criteria yield no entry.
        """
        entries: list[StackEntry] = []
        for control in self.output_controls:
            if not control.enabled or id(control) in skip_ids:
                continue
            if for_scoring and not getattr(control, "include_in_scoring", True):
                logger.info(
                    "compute_logprobs: skipping %s (include_in_scoring=False); scored logprobs will "
                    "not reflect this control's logits processors.",
                    type(control).__name__,
                )
                continue
            processors = control.get_logits_processors(
                input_ids, runtime_kwargs, attention_mask=attention_mask, **kwargs) or []
            criteria = [] if for_scoring else (control.get_stopping_criteria(
                input_ids, runtime_kwargs, attention_mask=attention_mask, **kwargs) or [])
            if processors or criteria:
                entries.append(StackEntry(
                    logits_processors=tuple(processors), stopping_criteria=tuple(criteria),
                ))
        return tuple(entries)

    def _compose_stacks(self, input_ids, runtime_kwargs, attention_mask, gen_kwargs,
                        skip_ids=frozenset(),
                        ) -> tuple[LogitsProcessorList, StoppingCriteriaList]:
        """Compose the controls' processors and criteria, then append caller extras popped from
        `gen_kwargs` (mutates `gen_kwargs`).

        Caller-supplied `logits_processor` / `stopping_criteria` entries append after the
        pipeline's own processors and criteria (per-call extras apply on top of the pipeline's
        standing configuration) and the keys are removed from `gen_kwargs`, so exactly one
        authoritative stack of each kind exists, travelling as an explicit parameter. gen_kwargs
        reaching the driver never contains processor or criteria objects, so drivers that copy or
        serialize their kwargs are safe by construction, and a driver that ignores the stacks
        visibly ignores named parameters.
        """
        entries = self._collect_output_entries(
            input_ids, runtime_kwargs, attention_mask=attention_mask, skip_ids=skip_ids, **gen_kwargs
        )
        processors = [p for entry in entries for p in entry.logits_processors]
        criteria = [c for entry in entries for c in entry.stopping_criteria]
        user_processors = gen_kwargs.pop("logits_processor", None) or []
        user_criteria = gen_kwargs.pop("stopping_criteria", None) or []
        return (
            LogitsProcessorList([*processors, *user_processors]),
            StoppingCriteriaList([*criteria, *user_criteria]),
        )

    def _apply_scoring_processors(self, logits, steered_input_ids, ref_output_ids,
                                  runtime_kwargs, attention_mask, is_encoder_decoder,
                                  **forward_kwargs) -> torch.Tensor:
        """Apply scoring-time logits processors position-by-position (teacher forcing).

        Processors receive the same `(prefix_ids, scores)` view as during generation. For causal
        models the prefix is `input ++ ref[:t]` when scoring `ref[t]`; for encoder-decoder models
        the prefix is the decoder ids `ref[:t+1]` when scoring `ref[t+1]` (matching the existing
        target alignment in both paths).
        """
        entries = self._collect_output_entries(
            steered_input_ids, runtime_kwargs, attention_mask=attention_mask,
            for_scoring=True, **forward_kwargs,
        )
        processors = [p for entry in entries for p in entry.logits_processors]
        if not processors:
            return logits
        stack = LogitsProcessorList(processors)
        with torch.no_grad():
            for t in range(logits.size(1)):
                prefix = (ref_output_ids[:, : t + 1] if is_encoder_decoder
                          else torch.cat([steered_input_ids, ref_output_ids[:, :t]], dim=1))
                logits[:, t, :] = stack(prefix, logits[:, t, :])
        return logits

    def _resolve_generate_source(
            self,
            inputs: Any,
            text: Any,
            messages: Any,
            input_ids: Any,
    ) -> tuple[Literal["text", "messages", "tokens"], Any]:
        """Select the single prompt source and its modality.

        Exactly one of positional `inputs`, `text=`, `messages=`, or `input_ids=` may be provided.
        Positional input is a convenience for text prompts (`str` or a `list` whose every element is
        a `str`) and routes to text; any other positional shape raises (E12). Because the check is a
        total `all(...)` over the list, a mixed list such as `["a", {"role": ...}]` fails here rather
        than downstream.

        Returns:
            tuple[kind, payload] where `kind` is `"text"`, `"messages"`, or `"tokens"` and `payload`
            is the value handed to the matching resolver.

        Raises:
            TypeError: If no source or more than one source is provided (E1/E2), or a positional
                input is neither a `str` nor a `list[str]` (E12).
        """
        provided = [
            name for name, value in (
                ("inputs", inputs), ("text", text), ("messages", messages), ("input_ids", input_ids),
            ) if value is not None
        ]
        if len(provided) == 0:
            raise TypeError(
                "generate() requires a prompt: pass positional text, or exactly one of text=, "
                "messages=, input_ids=."
            )
        if len(provided) > 1:
            names = ", ".join(provided)
            raise TypeError(
                f"generate() received multiple prompt sources ({names}); pass exactly one of "
                "positional inputs, text=, messages=, input_ids=."
            )

        if text is not None:
            return "text", text
        if messages is not None:
            return "messages", messages
        if input_ids is not None:
            return "tokens", input_ids

        # positional inputs: text convenience only
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(element, str) for element in inputs)
        ):
            return "text", inputs
        raise TypeError(
            "positional input to generate() must be a str or list of str; pass messages=... "
            "for chat or input_ids=... for token input."
        )

    def _resolve_text_prompt(self, text: Any) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
        """Validate and tokenize a text prompt (design §4.3.1).

        Args:
            text: A `str` (single) or a `list`/`tuple` whose elements are all `str` (batch).

        Returns:
            tuple[input_ids, attention_mask, is_single].

        Raises:
            TypeError: If `text` is a sequence containing a non-`str` element (E3).
            ValueError: If `text` is an empty sequence (E4).
        """
        is_single = isinstance(text, str)
        if is_single:
            normalized = [text]
        else:
            normalized = list(text)
            if len(normalized) == 0:
                raise ValueError("text= received an empty sequence.")
            for index, element in enumerate(normalized):
                if not isinstance(element, str):
                    raise TypeError(
                        f"text= must be a str or a sequence of str; element {index} is "
                        f"{type(element).__name__}."
                    )

        self._warned_tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
            self.input_controls, self._warned_tensor_with_adapt_messages
        )
        tokenized = self.tokenizer(normalized, return_tensors="pt", padding=True)
        return tokenized["input_ids"], tokenized.get("attention_mask"), is_single

    def _resolve_messages_prompt(
            self,
            messages: Any,
            runtime_kwargs: dict,
            chat_template_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, set[int], bool]:
        """Validate a chat prompt, then adapt and chat-template tokenize it (design §4.3.2).

        Accepts one conversation (a sequence of mappings) or a batch (a sequence of sequences of
        mappings). Message elements are validated as `collections.abc.Mapping`; role/content schema
        remains the responsibility of `apply_chat_template`.

        Args:
            messages: One conversation or a batch of conversations.
            runtime_kwargs: Per-call parameters forwarded to `adapt_messages`.
            chat_template_kwargs: Extra keyword arguments forwarded to `apply_chat_template` after the
                pipeline-owned kwargs. None or an empty mapping adds nothing.

        Returns:
            tuple[input_ids, attention_mask, message_handled, is_single], where `message_handled`
            holds `id()`s of controls that adapted at message level.

        Raises:
            ValueError: If the conversation or batch is empty (E5).
            TypeError: If a batch inner element is not a mapping (E6) or the outer sequence mixes
                element kinds (E7).
        """
        outer = list(messages)
        if len(outer) == 0:
            raise ValueError("messages= received an empty conversation or batch.")

        if all(isinstance(element, Mapping) for element in outer):
            is_single = True
            normalized = [list(outer)]
        elif all(isinstance(element, (list, tuple)) for element in outer):
            is_single = False
            normalized = []
            for i, chat in enumerate(outer):
                chat = list(chat)
                if len(chat) == 0:
                    raise ValueError("messages= received an empty conversation or batch.")
                for j, message in enumerate(chat):
                    if not isinstance(message, Mapping):
                        raise TypeError(
                            f"messages[{i}][{j}] must be a mapping (one chat message); got "
                            f"{type(message).__name__}."
                        )
                normalized.append(chat)
        else:
            raise TypeError(
                "messages= must be one conversation (a sequence of mappings) or a batch (a sequence "
                "of sequences of mappings); got mixed element types at the outer level."
            )

        input_ids, attention_mask, message_handled = apply_adapt_messages_and_tokenize(
            self.input_controls, self.tokenizer, normalized, runtime_kwargs,
            chat_template_kwargs=chat_template_kwargs,
        )
        return input_ids, attention_mask, message_handled, is_single

    def _resolve_token_prompt(
            self,
            input_ids: Any,
            attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
        """Validate a token prompt (tokens only; design §4.3.3).

        Args:
            input_ids: A 1-D/2-D `torch.Tensor`, a `list[int]`, or a `list[list[int]]`.
            attention_mask: Optional mask, passed through unchanged.

        Returns:
            tuple[input_ids, attention_mask, is_single].

        Raises:
            ValueError: If a tensor is neither 1-D nor 2-D (E8), or nested lists are ragged (E9).
            TypeError: If the value is not a token tensor or integer list (E10).
        """
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 1:
                resolved, is_single = input_ids.unsqueeze(0), True
            elif input_ids.ndim == 2:
                resolved, is_single = input_ids, False
            else:
                raise ValueError(f"input_ids tensor must be 1-D or 2-D; got {input_ids.ndim}-D.")
        elif isinstance(input_ids, list) and input_ids and all(isinstance(x, int) for x in input_ids):
            resolved, is_single = torch.tensor([input_ids], dtype=torch.long), True
        elif (
            isinstance(input_ids, list) and input_ids
            and all(isinstance(row, list) and row and all(isinstance(x, int) for x in row) for row in input_ids)
        ):
            try:
                resolved = torch.tensor(input_ids, dtype=torch.long)
            except ValueError as exception:
                raise ValueError(
                    "input_ids= nested lists must be rectangular (equal-length rows)."
                ) from exception
            is_single = False
        else:
            raise TypeError(
                f"input_ids= accepts a 1-D/2-D integer tensor, list[int], or list[list[int]]; got "
                f"{type(input_ids).__name__}. For text prompts use text= or positional input; for "
                "chat use messages=."
            )

        self._warned_tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
            self.input_controls, self._warned_tensor_with_adapt_messages
        )
        return resolved, attention_mask, is_single

    @overload
    def generate(
            self,
            inputs: str,
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> str: ...
    @overload
    def generate(
            self,
            inputs: list[str],
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> list[str]: ...
    @overload
    def generate(
            self,
            *,
            text: str,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> str: ...
    @overload
    def generate(
            self,
            *,
            text: Sequence[str],
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> list[str]: ...
    @overload
    def generate(
            self,
            *,
            messages: Sequence[Mapping],
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> str: ...
    @overload
    def generate(
            self,
            *,
            messages: Sequence[Sequence[Mapping]],
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> list[str]: ...
    @overload
    def generate(
            self,
            *,
            input_ids: torch.Tensor | list[int] | list[list[int]],
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> torch.Tensor: ...
    @overload
    def generate(
            self,
            inputs: Any = ...,
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            *,
            text: Any = ...,
            messages: Any = ...,
            input_ids: Any = ...,
            return_output: Literal[True],
            **gen_kwargs: Any,
    ) -> Output | list[Output]: ...

    def generate(
            self,
            inputs: str | list[str] | None = None,
            attention_mask: torch.Tensor | None = None,
            runtime_kwargs: dict | None = None,
            return_output: bool = False,
            *,
            text: str | Sequence[str] | None = None,
            messages: Sequence[Mapping] | Sequence[Sequence[Mapping]] | None = None,
            input_ids: torch.Tensor | list[int] | list[list[int]] | None = None,
            **gen_kwargs,
    ) -> str | list[str] | torch.Tensor | Output | list[Output]:
        """Generate with steering across text, chat, and token prompts.

        The prompt source is declared by keyword, with exactly one source per call:

        | Source | Tokenization | Default return type |
        | --- | --- | --- |
        | `text=` (`str`) | plain text | `str` |
        | `text=` (`Sequence[str]`) | batched plain text | `list[str]` |
        | `messages=` (one conversation) | `apply_chat_template` | `str` |
        | `messages=` (batch of conversations) | batched `apply_chat_template` | `list[str]` |
        | `input_ids=` (1-D tokens) | already tokenized; passed through | `torch.Tensor` |
        | `input_ids=` (2-D tokens) | already tokenized; passed through | `torch.Tensor` |

        Positional `str`/`list[str]` is accepted as a convenience for text prompts and behaves like
        `text=`; any other positional shape raises `TypeError`. With `return_output=True`, the return
        is always `Output` (single) or `list[Output]` (batched) regardless of source.

        Unlike `model.generate`, the returned token ids exclude the prompt by default. Do not slice
        the result by prompt length, since that discards generated tokens. Pass
        `return_full_sequence=True` to get HF-style prompt+continuation output. A stop string that
        also occurs inside a reasoning model's thinking segment cuts the decoded text there, so pass
        stop strings that cannot appear before the closing think tag when generating with thinking on.

        `attention_mask` is valid only with token input (`input_ids=`); it is derived automatically
        for `text=` and `messages=`, and passing it with either raises `TypeError`.
        The `adapt_messages` hook fires only on chat input; text and token input go straight to the
        token-level `adapt(input_ids, ...)` chain. For chat input, each input control whose
        `adapt_messages` returns a non-None result is not additionally run at token level, so every
        input control is applied exactly once per call.

        Args:
            inputs: Positional convenience for text prompts (`str` or `list[str]`), behaving like
                `text=`. Any other positional shape raises `TypeError`; use the keywords below.
            attention_mask: Attention mask, valid only with `input_ids=`.
            runtime_kwargs: Per-generation parameters for controls (e.g., `{"substrings": [...]}`).
            return_output: If True, return one or more `Output` objects instead of decoded text /
                token IDs.
            text: Text prompt as a `str` or a sequence of `str`.
            messages: Chat prompt as one conversation (a sequence of mappings) or a batch (a sequence
                of sequences of mappings).
            input_ids: Token prompt as a 1-D/2-D integer tensor, `list[int]`, or `list[list[int]]`.
            **gen_kwargs: Generation parameters in `model.generate` vocabulary, normalized
                through `GenerationParams` and executed by the inference backend's session
                (unlisted keys pass through in process and raise on API backends). Two keys are
                reserved and consumed here rather than forwarded to the backend:

                    - `return_full_sequence: bool` to include the prompt in the returned token IDs.
                    - `chat_template_kwargs: dict` forwarded to `apply_chat_template` after the
                        pipeline-owned template kwargs. Valid only with `messages=` (pairing it with
                        `text=`/`input_ids=` raises `TypeError`); it may not name any pipeline-owned
                        template kwarg (`return_tensors`, `padding`, `add_generation_prompt`,
                        `return_dict`, else `ValueError`). An empty mapping is a no-op. The toolkit
                        does not interpret the mapping; keys are model-family specific (e.g.
                        `enable_thinking`), and models whose templates expose no such switch are
                        unaffected.

        Returns:
            See dispatch table above.

        Raises:
            RuntimeError: If `steer()` has not yet been called.
            TypeError: If no prompt source or more than one is provided, if a source fails
                validation, if `attention_mask` is paired with `text=`/`messages=`, if
                `chat_template_kwargs` is paired with `text=`/`input_ids=`, or if
                `chat_template_kwargs` is not a mapping.
            ValueError: If a token tensor is not 1-D/2-D, nested token lists are ragged, a text/
                chat sequence is empty, or `chat_template_kwargs` names a pipeline-owned template
                argument.
        """
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.generate()`.")

        runtime_kwargs = runtime_kwargs or {}
        return_full_sequence = bool(gen_kwargs.pop("return_full_sequence", False))
        chat_template_kwargs = gen_kwargs.pop("chat_template_kwargs", None)

        if chat_template_kwargs is not None:
            if not isinstance(chat_template_kwargs, Mapping):
                raise TypeError(
                    "chat_template_kwargs must be a mapping of chat-template keyword arguments; got "
                    f"{type(chat_template_kwargs).__name__}."
                )
            reserved = {"return_tensors", "padding", "add_generation_prompt", "return_dict"}
            collisions = reserved & set(chat_template_kwargs)
            if collisions:
                names = ", ".join(sorted(collisions))
                raise ValueError(
                    f"chat_template_kwargs may not override pipeline-owned template arguments: {names}."
                )

        kind, payload = self._resolve_generate_source(inputs, text, messages, input_ids)

        # attention_mask pairing
        if attention_mask is not None and kind != "tokens":
            raise TypeError(
                "attention_mask is only valid with token input (input_ids=); it is derived "
                "automatically for text= and messages=."
            )

        # chat_template_kwargs pairing
        if chat_template_kwargs is not None and kind != "messages":
            raise TypeError(
                "chat_template_kwargs is only valid with chat input (messages=); text= and input_ids= "
                "are already templated or template-free."
            )

        # resolve the prompt tensors per modality
        message_handled: set[int] = set()
        if kind == "text":
            prompt_input_ids, prompt_attention_mask, is_single = self._resolve_text_prompt(payload)
        elif kind == "messages":
            prompt_input_ids, prompt_attention_mask, message_handled, is_single = (
                self._resolve_messages_prompt(payload, runtime_kwargs, chat_template_kwargs=chat_template_kwargs)
            )
        else:  # tokens
            prompt_input_ids, prompt_attention_mask, is_single = self._resolve_token_prompt(
                payload, attention_mask
            )

        return self._execute_generation(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            message_handled=message_handled,
            decode_text=(kind != "tokens"),
            is_single=is_single,
            runtime_kwargs=runtime_kwargs,
            return_output=return_output,
            return_full_sequence=return_full_sequence,
            gen_kwargs=gen_kwargs,
        )

    def _execute_generation(
            self,
            prompt_input_ids: torch.Tensor,
            prompt_attention_mask: torch.Tensor | None,
            message_handled: frozenset[int] | set[int],
            *,
            decode_text: bool,
            is_single: bool,
            runtime_kwargs: dict,
            return_output: bool,
            return_full_sequence: bool,
            gen_kwargs: dict,
    ) -> str | list[str] | torch.Tensor | Output | list[Output]:
        """Run the shared generation tail from prompt tensors through the shaped return.

        Applies the token-level input-control chain, merges sampling-expressible output
        controls into the call's `GenerationParams`, and configures state hooks. With the
        default decoding driver, each prompt row becomes a `GenerationItem` executed by the
        inference backend's session; an explicit `DecodingDriver` instead runs client-side
        under the state-control hook context with `session=` passed for its rollouts. The
        return is then shaped per modality: decoded continuation text truncates at the first
        stop string, and the prompt slice is removed by default (`return_full_sequence=False`
        returns continuation tokens only).

        Args:
            prompt_input_ids: Prompt token IDs [seq_len] or [batch, seq_len] before the token-level
                `adapt` chain.
            prompt_attention_mask: Attention mask matching `prompt_input_ids`, or None to derive it.
            message_handled: `id()`s of input controls already handled at message level; their
                token-level `adapt` is skipped.
            decode_text: If True, decode the returned IDs to `str`/`list[str]`; if False, return the
                token tensor.
            is_single: If True, the caller passed a single (non-batched) prompt, so unwrap the batch
                dimension in the return.
            runtime_kwargs: Per-call parameters forwarded to input, state, and output controls.
            return_output: If True, return `Output` (single) or `list[Output]` (batched) regardless
                of `decode_text`.
            return_full_sequence: If True, include the prompt in the returned token IDs.
            gen_kwargs: Generation parameters forwarded to the decoding driver.

        Returns:
            `str` or `list[str]` when `decode_text` is True, `torch.Tensor` when False, or
            `Output`/`list[Output]` when `return_output` is True.
        """
        # input controls (token-level adapt chain) + normalize
        steered_input_ids, steered_attention_mask = self._prepare_inputs(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            runtime_kwargs=runtime_kwargs,
            message_handled=frozenset(message_handled),
        )

        # sampling-expressible output controls lower to generation parameters for this call
        lowered = self._lowered_contributions(runtime_kwargs)
        skip_ids = frozenset(lowered)

        spec = self._resolve_backend_spec(self.backend)
        backend = self._backend_for(spec)
        decoding_driver = self._resolve_decoding_driver()
        inference_capabilities = capabilities_for_spec(spec)
        hooks_in_process = Capability.IN_PROCESS_TORCH in inference_capabilities.atoms
        has_enabled_state = any(control.enabled for control in self.state_controls)

        # state-control entry selection per backend: an in-process backend gets hooks built
        # once per logical generation; an intervention-capable backend gets exported specs. On
        # the in-process path, distinct per-item derived seeds run serially in the session, so
        # hooks are computed per row there rather than once on the batch.
        state_entry_rows: list[tuple[HookEntry, ...]] | None = None
        state_entries: tuple[StateControlEntry, ...] = ()
        if decoding_driver is not None:
            state_entries = self._collect_state_entries(
                steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask, **gen_kwargs
            )
        elif not hooks_in_process:
            if has_enabled_state:
                state_entries = self._collect_state_entries(
                    steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask, **gen_kwargs
                )
        elif (
            gen_kwargs.get("seed") is not None
            and steered_input_ids.size(0) > 1
            and has_enabled_state
        ):
            state_entry_rows = self._per_item_state_entries(
                steered_input_ids, steered_attention_mask, runtime_kwargs, **gen_kwargs
            )
        else:
            state_entries = self._collect_state_entries(
                steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask, **gen_kwargs
            )

        with backend.open_session() as session:
            if decoding_driver is not None:
                # client-side driver path: composed stacks, session-hosted hooks for the span
                # of the decode, rollouts through a SteeredSession
                logits_processors, stopping_criteria = self._compose_stacks(
                    steered_input_ids, runtime_kwargs, steered_attention_mask, gen_kwargs,
                    skip_ids=skip_ids,
                )
                params = GenerationParams.from_gen_kwargs(**gen_kwargs)
                for contribution in lowered.values():
                    params = merge_lowered_params(params, contribution)
                # in process, the session hosts this generation's hooks for the whole decode,
                # so rollouts through the session and auxiliary forwards on the live model are
                # steered alike and the SteeredSession injects nothing (ambient hooks already
                # cover its items); on spec-consuming backends the SteeredSession injects a
                # rollout variant of each lowered entry whose prompt-relative scopes are
                # rewritten to absolute positions at the generation's original prompt boundary
                rollout_entries: tuple = ()
                if state_entries and hooks_in_process:
                    applied = session.entries_applied(state_entries)
                else:
                    applied = contextlib.nullcontext()
                    if state_entries:
                        rollout_entries = self._rollout_entries(
                            state_entries, steered_input_ids, steered_attention_mask,
                        )
                with applied:
                    full_output_ids = decoding_driver.decode(
                        input_ids=steered_input_ids,
                        attention_mask=steered_attention_mask,
                        model=self.model,
                        logits_processors=logits_processors,
                        stopping_criteria=stopping_criteria,
                        runtime_kwargs=runtime_kwargs,
                        session=SteeredSession(session, rollout_entries),
                        **params.to_gen_kwargs(),
                    )
                prompt_len = steered_input_ids.size(1)
                new_tokens = full_output_ids[:, prompt_len:]
                reasons = infer_finish_reasons(
                    new_tokens,
                    {"max_new_tokens": params.max_new_tokens},
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stop_strings=params.stop_strings,
                    stop_token_ids=params.stop_token_ids,
                    tokenizer=self.tokenizer,
                )
            else:
                # default path: per-prompt items executed by the session; on backends hosting
                # structured outputs natively, declarative constraints lower in place of their
                # live processors
                constraint_sources: dict[int, ConstraintSource] = {}
                processor_specs: dict[int, ProcessorSpecEntry] = {}
                if not hooks_in_process:
                    constraint_sources = self._constraint_contributions(runtime_kwargs)
                    processor_specs = self._processor_spec_contributions(
                        runtime_kwargs, inference_capabilities,
                    )
                output_entries = self._collect_output_entries(
                    steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask,
                    skip_ids=skip_ids | frozenset(constraint_sources) | frozenset(processor_specs),
                    **gen_kwargs,
                )
                if constraint_sources or processor_specs:
                    output_entries = output_entries + tuple(
                        ConstraintEntry(source=source) for source in constraint_sources.values()
                    ) + tuple(processor_specs.values())
                user_processors = gen_kwargs.pop("logits_processor", None) or []
                user_criteria = gen_kwargs.pop("stopping_criteria", None) or []
                params = GenerationParams.from_gen_kwargs(**gen_kwargs)
                for contribution in lowered.values():
                    params = merge_lowered_params(params, contribution)
                if user_processors or user_criteria:
                    extra = dict(params.extra)
                    if user_processors:
                        extra["logits_processor"] = user_processors
                    if user_criteria:
                        extra["stopping_criteria"] = user_criteria
                    params = dataclasses.replace(params, extra=extra)

                items = [
                    GenerationItem(
                        prompt=PreparedPrompt.from_token_ids(
                            steered_input_ids[i:i + 1], steered_attention_mask[i:i + 1],
                        ),
                        state_entries=state_entry_rows[i] if state_entry_rows is not None else state_entries,
                        output_entries=output_entries,
                    )
                    for i in range(steered_input_ids.size(0))
                ]
                results = session.generate(items, params)
                new_tokens, reasons = self._assemble_item_outputs(results)
                num_candidates = params.n or 1
                if return_full_sequence:
                    repeated = steered_input_ids.repeat_interleave(num_candidates, dim=0)
                    full_output_ids = torch.cat([repeated, new_tokens.to(repeated.device)], dim=1)
                else:
                    full_output_ids = None

        returned_ids = full_output_ids if return_full_sequence else new_tokens

        # shape return per modality + flag
        num_candidates = params.n or 1
        if return_output:
            if is_single:
                return Output(
                    output_ids=new_tokens,
                    adapted_input_ids=steered_input_ids,
                    finish_reason=reasons[0],
                    finish_reasons=tuple(reasons),
                )
            return [
                Output(
                    output_ids=new_tokens[i:i + 1],
                    adapted_input_ids=steered_input_ids[i // num_candidates:i // num_candidates + 1],
                    finish_reason=reasons[i],
                    finish_reasons=(reasons[i],),
                )
                for i in range(new_tokens.size(0))
            ]

        if not decode_text:
            return returned_ids

        # text / chat → decode; decoded continuation text truncates at the first stop string
        decoded = self.tokenizer.batch_decode(
            returned_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if params.stop_strings and not return_full_sequence:
            decoded = [truncate_at_stop_strings(text, params.stop_strings) for text in decoded]
        if is_single:
            return decoded[0]
        return decoded

    def _assemble_item_outputs(self, results) -> tuple[torch.Tensor, list[str | None]]:
        """Stack item results into one `[batch * n, gen_len]` tensor plus flat per-row reasons.

        Rows pad to the longest continuation with the tokenizer's pad token, matching the
        right-padding a single batched `generate` call produces.
        """
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        rows: list[torch.Tensor] = []
        reasons: list[str | None] = []
        for result in results:
            output = result.output
            rows.append(output.output_ids)
            if output.finish_reasons is not None:
                reasons.extend(output.finish_reasons)
            else:
                reasons.extend([output.finish_reason] * output.output_ids.size(0))
        max_len = max((row.size(1) for row in rows), default=0)
        padded = [
            torch.nn.functional.pad(row, (0, max_len - row.size(1)), value=pad_token_id)
            for row in rows
        ]
        return torch.cat(padded, dim=0), reasons

    def compute_logprobs(
            self,
            input_ids: list[int] | torch.LongTensor,
            attention_mask: torch.Tensor | None = None,
            ref_output_ids: list[int] | torch.LongTensor = None,
            runtime_kwargs: dict | None = None,
            **forward_kwargs: Any,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities of ref_output_ids with structural, input, state, and output steering
        controls applied.

        Step-level output controls' logits processors are part of the steered next-token distribution, so they are
        applied here position-by-position (teacher-forced), for every enabled output control whose
        `include_in_scoring` is True. The exclusive mechanism has no scoring analogue: decoding drivers are not
        invoked and stopping criteria are not applied. A control opts out of scoring by setting
        `include_in_scoring=False` (e.g. processors too expensive to evaluate per reference position).

        Uses teacher forcing to compute log P(ref_t | steered_input, ref_1, ..., ref_{t-1}) for each
        token in the reference sequence.

        Decoder-only scoring executes through the inference backend's session as `ScoringItem`s.
        When all pipeline controls support batching, the items share one set of control entries
        and score in a single pass (inputs are left-padded internally for correct positional
        alignment); otherwise each item is prepared and scored sequentially. Encoder-decoder
        models score in process against the live model.

        Args:
            input_ids: Input token IDs as list or tensor [seq_len] or [batch, seq_len]
            attention_mask: Optional attention mask matching input_ids shape
            ref_output_ids: Reference tokens to score [ref_len] or [batch, ref_len]
            runtime_kwargs: Per-call parameters for controls (e.g., {"substrings": [...]})
            **forward_kwargs: Additional arguments passed to model forward pass

        Returns:
            torch.Tensor: Log probabilities of shape [batch, ref_len] for decoder-only models,
                or [batch, ref_len - 1] for encoder-decoder models (excludes first decoder token)

        Raises:
            RuntimeError: If steer() has not been called
            ValueError: If ref_output_ids is None
            UnsupportedPipelineError: If an enabled control is score-unsupported on the
                configured inference backend.
        """
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.compute_logprobs()`.")
        if ref_output_ids is None:
            raise ValueError("`ref_output_ids` is required for `compute_logprobs()`.")
        if self._support_report is not None:
            self._support_report.raise_for("score")

        runtime_kwargs = runtime_kwargs or {}

        is_encoder_decoder = (
            self.model is not None and getattr(self.model.config, "is_encoder_decoder", False)
        )
        if is_encoder_decoder:
            return self._compute_logprobs_encoder_decoder(
                input_ids, attention_mask, ref_output_ids, runtime_kwargs, **forward_kwargs,
            )

        device = self.model.device if self.model is not None else torch.device("cpu")

        # normalize ref_output_ids
        if isinstance(ref_output_ids, list):
            ref_output_ids = torch.tensor(ref_output_ids, dtype=torch.long)
        if ref_output_ids.ndim == 1:
            ref_output_ids = ref_output_ids.unsqueeze(0)
        ref_output_ids = ref_output_ids.to(device)
        ref_len = ref_output_ids.size(1)

        spec = self._resolve_backend_spec(self.backend)
        backend = self._backend_for(spec)
        score_params = GenerationParams(extra=forward_kwargs)
        inference_capabilities = capabilities_for_spec(spec)
        hooks_in_process = Capability.IN_PROCESS_TORCH in inference_capabilities.atoms
        has_enabled_state = any(control.enabled for control in self.state_controls)

        # batched path (all controls are batch-safe): one left-packed pass over shared entries
        if self.supports_batching:
            steered_input_ids, steered_attention_mask = self._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                runtime_kwargs=runtime_kwargs,
            )
            batch_size = steered_input_ids.size(0)
            if ref_output_ids.size(0) == 1 and batch_size > 1:
                ref_output_ids = ref_output_ids.expand(batch_size, -1)
            if ref_len == 0:
                return torch.zeros((batch_size, 0), device=device, dtype=torch.float32)

            # left-pad for correct positional alignment in causal models; with right-padding, pad
            # tokens between the real input and the appended ref tokens corrupt positional
            # encodings and the causal attention chain
            steered_input_ids, steered_attention_mask = to_left_pad(
                steered_input_ids, steered_attention_mask
            )
            state_entries = self._collect_state_entries(
                steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask,
                **forward_kwargs,
            )
            output_entries = self._collect_output_entries(
                steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask,
                for_scoring=True, **forward_kwargs,
            )
            items = [
                ScoringItem(
                    prompt=PreparedPrompt.from_token_ids(
                        steered_input_ids[i:i + 1], steered_attention_mask[i:i + 1],
                    ),
                    ref_output_ids=ref_output_ids[i:i + 1],
                    state_entries=state_entries,
                    output_entries=output_entries,
                )
                for i in range(batch_size)
            ]
            with backend.open_session() as session:
                return session.score(items, score_params)

        # sequential fallback (one or more controls do not support batching): per-item
        # preparation and entries, one scoring pass per item
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(device)

        num_inputs = input_ids.size(0)
        if ref_output_ids.size(0) == 1 and num_inputs > 1:
            ref_output_ids = ref_output_ids.expand(num_inputs, -1)
        if ref_len == 0:
            return torch.zeros((num_inputs, 0), device=device, dtype=torch.float32)

        all_logprobs = []
        with backend.open_session() as session:
            for i in range(num_inputs):
                single_attention_mask = attention_mask[i:i + 1] if attention_mask is not None else None
                steered_input_ids, steered_attention_mask = self._prepare_inputs(
                    input_ids=input_ids[i:i + 1],
                    attention_mask=single_attention_mask,
                    runtime_kwargs=runtime_kwargs,
                )
                state_entries = self._collect_state_entries(
                    steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask,
                    **forward_kwargs,
                )
                output_entries = self._collect_output_entries(
                    steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask,
                    for_scoring=True, **forward_kwargs,
                )
                item = ScoringItem(
                    prompt=PreparedPrompt.from_token_ids(steered_input_ids, steered_attention_mask),
                    ref_output_ids=ref_output_ids[i:i + 1],
                    state_entries=state_entries,
                    output_entries=output_entries,
                )
                all_logprobs.append(session.score([item], score_params))
        return torch.cat(all_logprobs, dim=0)

    def _compute_logprobs_encoder_decoder(
            self,
            input_ids: list[int] | torch.LongTensor,
            attention_mask: torch.Tensor | None,
            ref_output_ids: list[int] | torch.LongTensor,
            runtime_kwargs: dict,
            **forward_kwargs: Any,
    ) -> torch.Tensor:
        """Teacher-forced scoring for encoder-decoder models, run in process against the live
        model (a batched pass when every control is batch-safe, else a sequential fallback)."""
        device = self.model.device

        # normalize ref_output_ids
        if isinstance(ref_output_ids, list):
            ref_output_ids = torch.tensor(ref_output_ids, dtype=torch.long)
        if ref_output_ids.ndim == 1:
            ref_output_ids = ref_output_ids.unsqueeze(0)
        ref_output_ids = ref_output_ids.to(device)
        ref_len = ref_output_ids.size(1)

        # batched path (all controls are batch-safe)
        if self.supports_batching:
            # input controls
            steered_input_ids, attention_mask = self._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                runtime_kwargs=runtime_kwargs,
            )
            batch_size = steered_input_ids.size(0)

            # broadcast single ref sequence across batch
            if ref_output_ids.size(0) == 1 and batch_size > 1:
                ref_output_ids = ref_output_ids.expand(batch_size, -1)

            if ref_len == 0:
                return torch.zeros((batch_size, 0), device=device, dtype=torch.float32)

            # state controls, hosted by the in-process session for the span of the forward
            state_entries = self._collect_state_entries(
                steered_input_ids, runtime_kwargs, attention_mask=attention_mask, **forward_kwargs
            )
            backend = self._backend_for(self._resolve_backend_spec(self.backend))
            with backend.open_session() as session, session.entries_applied(state_entries):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=steered_input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=ref_output_ids,
                        **forward_kwargs,
                    )
                    # predicts ref[t+1] from ref[0:t]; logits[:, t, :] -> ref[t+1]
                    # logits[:, :-1, :] aligns with targets ref[:, 1:]
                    logits = outputs.logits[:, :-1, :]
                    target_ids = ref_output_ids[:, 1:]

                # apply output-control scoring processors under the steered distribution
                logits = self._apply_scoring_processors(
                    logits, steered_input_ids, ref_output_ids, runtime_kwargs,
                    attention_mask, True, **forward_kwargs,
                )

                # compute logprobs
                logprobs = torch.log_softmax(logits, dim=-1)
                return logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        # sequential fallback (one or more controls do not support batching)

        # normalize input_ids to 2D for indexing
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(device)

        num_inputs = input_ids.size(0)

        # broadcast single ref sequence across batch
        if ref_output_ids.size(0) == 1 and num_inputs > 1:
            ref_output_ids = ref_output_ids.expand(num_inputs, -1)

        if ref_len == 0:
            return torch.zeros((num_inputs, 0), device=device, dtype=torch.float32)

        all_logprobs = []

        for i in range(num_inputs):
            single_input_ids = input_ids[i:i + 1]
            single_attention_mask = attention_mask[i:i + 1] if attention_mask is not None else None
            single_ref = ref_output_ids[i:i + 1]

            # input controls
            steered_input_ids, steered_attention_mask = self._prepare_inputs(
                input_ids=single_input_ids,
                attention_mask=single_attention_mask,
                runtime_kwargs=runtime_kwargs,
            )

            # state controls, hosted by the in-process session for the span of the forward
            state_entries = self._collect_state_entries(
                steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask, **forward_kwargs
            )
            backend = self._backend_for(self._resolve_backend_spec(self.backend))
            with backend.open_session() as session, session.entries_applied(state_entries):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=steered_input_ids,
                        attention_mask=steered_attention_mask,
                        decoder_input_ids=single_ref,
                        **forward_kwargs,
                    )
                    logits = outputs.logits[:, :-1, :]
                    target_ids = single_ref[:, 1:]

                # apply output-control scoring processors under the steered distribution
                logits = self._apply_scoring_processors(
                    logits, steered_input_ids, single_ref, runtime_kwargs,
                    steered_attention_mask, True, **forward_kwargs,
                )

                # compute logprobs
                logprobs = torch.log_softmax(logits, dim=-1)
                token_logprobs = logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            all_logprobs.append(token_logprobs)

        return torch.cat(all_logprobs, dim=0)
