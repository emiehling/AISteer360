"""State control base classes.

This module provides the abstract base class for methods that register hooks into the model (e.g., to modify
intermediate representations during inference); does not change model weights.

Two base classes are provided:

- `StateControl`: Base class for all state control methods.
- `NoStateControl`: Identity (null) control; used when no state control is defined in steering pipeline.

State controls implement steering through runtime intervention in the model's forward pass, modifying internal states
(activations, attention patterns) to produce generations following y ~ p_θᵃ(x), where "p_θᵃ" is the model with state
controls.

Examples of state controls:

- Activation steering (e.g., adding direction vectors)
- Attention head manipulation and pruning
- Layer-wise activation editing
- Dynamic routing between components
- Representation engineering techniques

The base class provides automatic hook management through context managers (ensures cleanup and avoids memory leaks).

See Also:

- `aisteer360.algorithms.state_control`: Implementations of state control methods
- `aisteer360.core.steering_pipeline`: Integration with steering pipeline
"""
import copy
from abc import abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.base_control import BaseControl
from aisteer360.algorithms.core.execution.requirements import Requirements

PreHook = Callable[[nn.Module, tuple], tuple | torch.Tensor]
ForwardHook = Callable[[nn.Module, tuple, torch.Tensor], torch.Tensor]
BackwardHook = Callable[[nn.Module, tuple, tuple], tuple]
HookSpec = dict[str, str | PreHook | ForwardHook | BackwardHook]


class StateControl(BaseControl):
    """Abstract base class for state control steering methods.

    Modifies internal model states during forward passes via hooks.

    A control instance holds per-generation state on `self` (e.g. position offsets, cached masks,
    gate decisions) and therefore supports one in-flight generation at a time; do not share a single
    control instance across concurrently running pipelines.

    Methods:
        get_hooks(input_ids, runtime_kwargs, **kwargs) -> dict: Create hook specs (required)
        steer(model, tokenizer, **kwargs) -> None: One-time preparation (optional)
        reset() -> None: Reset logic (optional)
        register_hooks(model) -> None: Attach hooks to model (provided)
        remove_hooks() -> None: Remove all registered hooks (provided)
    """

    Args: type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = False
    _model_ref: PreTrainedModel | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hooks: dict[str, list[HookSpec]] = {"pre": [], "forward": [], "backward": []}
        self.registered: list[torch.utils.hooks.RemovableHandle] = []

    @abstractmethod
    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **kwargs,
    ) -> dict[str, list[HookSpec]]:
        """Create hook specifications for the current generation.

        Args:
            input_ids: Prompt token ids of shape [batch, seq_len].
            runtime_kwargs: Per-call parameters for the control.
            **kwargs: Additional generation-time context. In particular, the pipeline forwards
                `attention_mask` (the prompt attention mask matching `input_ids`, or None) here;
                controls that score prompt tokens may consume it to align with the real (non-pad)
                positions instead of re-deriving a mask by token identity.
        """
        pass

    def steer(self,
              model: PreTrainedModel,
              tokenizer: PreTrainedTokenizerBase = None,
              session=None,
              **kwargs) -> None:
        """Optional steering/preparation.

        `session` is a `SteeringSession` on the steering backend, provided by the pipeline.
        """
        pass

    def export_intervention_spec(self, runtime_kwargs: dict | None = None):
        """The control's `InterventionSpec` for intervention-capable backends, or None.

        The spec is the second serialization of the tuple the control's hooks close over,
        emitted from the same transform, gate, and scope objects. Must be called after
        `steer()`. Returns None when the configuration has no wire form (the configuration is
        then hook-only) or when the control does not implement spec export at all.

        Args:
            runtime_kwargs: Per-call parameters, mirroring `get_hooks`; per-item values
                (strengths, positions) serialize into the returned spec.

        Returns:
            The validated `InterventionSpec` with tensor payloads attached, or None.
        """
        return None

    def register_hooks(self, model: PreTrainedModel) -> None:
        """Attach hooks to model.

        If registration fails partway (e.g. an unresolved module path), any handles already
        attached are removed before re-raising, so a partial `__enter__` never leaves hooks on the
        model for subsequent, unrelated generations (`__exit__` is not called when `__enter__` raises).
        """
        try:
            for phase in ("pre", "forward", "backward"):
                for spec in self.hooks[phase]:
                    module = model.get_submodule(spec["module"])
                    if phase == "pre":
                        handle = module.register_forward_pre_hook(spec["hook_func"], with_kwargs=True)
                    elif phase == "forward":
                        handle = module.register_forward_hook(spec["hook_func"], with_kwargs=True)
                    else:
                        handle = module.register_full_backward_hook(spec["hook_func"])
                    self.registered.append(handle)
        except Exception:
            self.remove_hooks()
            raise

    def remove_hooks(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self.registered:
            handle.remove()
        self.registered.clear()

    def set_hooks(self, hooks: dict[str, list[HookSpec]]):
        """Update the hook specifications to be registered."""
        self.hooks = hooks

    def __enter__(self):
        """Context manager entry: register hooks to model.

        Raises:
            RuntimeError: If model reference not set by pipeline
        """
        if self._model_ref is None:
            raise RuntimeError("Model reference not set before entering context.")
        self.register_hooks(self._model_ref)

        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit: clean up all hooks."""
        self.remove_hooks()

    def clone_for_call(self, seed: int | None = None):
        """A per-call clone with independent per-generation mutable state.

        Extends the base shallow clone with fresh hook and handle containers, a deep copy of the
        `_gate` and `_runtime` attributes when present (so the clone's `get_hooks` closures never
        share position or gate state with the original or with sibling clones), and a cleared
        model reference. Steer-time artifacts (steering vectors, transforms, tokenizers) stay
        shared with the original.

        Args:
            seed: Optional seed forwarded to the clone's `reseed()`.

        Returns:
            The clone.
        """
        clone = super().clone_for_call(seed)
        clone.hooks = {"pre": [], "forward": [], "backward": []}
        clone.registered = []
        if not isinstance(getattr(type(self), "_runtime", None), property):
            if getattr(self, "_runtime", None) is not None:
                clone._runtime = copy.deepcopy(self._runtime)
        if not isinstance(getattr(type(self), "_gate", None), property):
            if getattr(self, "_gate", None) is not None:
                clone._gate = copy.deepcopy(self._gate)
        clone._model_ref = None
        return clone

    def reset(self) -> None:
        """Between-generations reset for runtime-backed controls.

        Covers the `_gate`/`_runtime` convention used across `state_control._common`: clear the
        gate, then re-clear the runtime's per-generation counters (preserving its stored prompt
        lengths and mask). No-op for controls that expose neither attribute. Controls with
        additional per-generation state override this and may call `super().reset()`.
        """
        gate = getattr(self, "_gate", None)
        if gate is not None:
            gate.reset()
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.reset_between_generations()


class NoStateControl(StateControl):
    """Identity state control.

    Used as the default when no state control is needed. Returns empty hook dictionaries and skips registration.
    """
    enabled: bool = False
    supports_batching: bool = True

    def get_hooks(self, *_, **__) -> dict[str, list[HookSpec]]:
        """Return empty hooks."""
        return {"pre": [], "forward": [], "backward": []}

    def steer(self,
              model: PreTrainedModel,
              tokenizer=None,
              **kwargs) -> None:
        """Null steering operation."""
        pass

    def register_hooks(self, *_):
        """Null registration operation."""
        pass

    def remove_hooks(self, *_):
        """Null removal operation."""
        pass

    def set_hooks(self, hooks: dict[str, list[HookSpec]]):
        """Null set operation."""
        pass


def _is_concrete_gate(gate) -> bool:
    """True when `gate` is a resolved gate rather than a gate/condition source."""
    from aisteer360.algorithms.state_control._common.gates.base import BaseGate

    return isinstance(gate, BaseGate)


class HookControl(StateControl):
    """A state control that writes its own torch hooks.

    Keeps the abstract `get_hooks(input_ids, runtime_kwargs, **kwargs)` and must fully
    re-derive its per-generation state inside `get_hooks` on every call. Controls whose
    behavior is a tuple of residual-stream interventions subclass `InterventionControl`
    instead; this class is for methods hooking other mechanisms (e.g. attention weights).

    Keeps the conservative in-process generate requirement unless the subclass overrides
    `requirements()`.
    """


class InterventionControl(StateControl):
    """A state control that is a tuple of interventions.

    Subclasses declare an unbound intervention template, usually in `_configure()`; the base
    `steer()` binds it. There is no per-generation protocol on the control: hook construction,
    gate sizing, and position state are owned by `build_hooks`, and lowering to
    `InterventionSpec` is owned by `lower_interventions`.

    Class attributes:
        hook_only_hint: Fix text used in unsupported-generate verdicts when the template has
            no wire form.

    Attributes:
        interventions: The bound interventions, populated by `steer()`.
    """

    supports_batching = True
    hook_only_hint: str | None = None

    tokenizer = None
    interventions: tuple = ()
    _template: tuple = ()

    def steer(self, model=None, tokenizer=None, session=None, **kwargs):
        """Bind the intervention template against the model (or the session's layout).

        Structural facts come from the steering session's layout when a session is given, so a
        fully concrete template (precomputed vectors, manual thresholds) binds with
        `model=None`. Templates carrying sources (fits, searches) resolve them here.

        Args:
            model: The base language model, or None for concrete templates bound against a
                session layout.
            tokenizer: Tokenizer used when fitting sources.
            session: `SteeringSession` on the steering backend, provided by the pipeline.

        Returns:
            The input model, unchanged.
        """
        from aisteer360.algorithms.state_control._common.layout_facts import resolve_layout
        from aisteer360.algorithms.state_control._common.model_layout import resolve_model_layout

        layout = resolve_layout(model, session)
        self._num_layers = layout.num_layers
        self._module_layout = resolve_model_layout(model) if model is not None else None
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.interventions = tuple(
            intervention.bind(model, tokenizer, layout=layout, session=session)
            for intervention in self._template
        )
        return model

    @property
    def _transform(self):
        """The first intervention's transform (None before `steer()`).

        Assignment replaces the transform on the first intervention, so a wrapped or
        instrumented transform takes effect in subsequently built hooks.
        """
        return self.interventions[0].transform if self.interventions else None

    @_transform.setter
    def _transform(self, value) -> None:
        import dataclasses

        if not self.interventions:
            raise AttributeError("No bound interventions; call steer() first.")
        first, *rest = self.interventions
        self.interventions = (dataclasses.replace(first, transform=value), *rest)

    @property
    def _gate(self):
        """The first intervention's gate (None before `steer()`)."""
        return self.interventions[0].gate if self.interventions else None

    def _resolve_module_layout(self, model=None):
        """The module-path layout, resolved from the module tree on first use."""
        layout = getattr(self, "_module_layout", None)
        if layout is None:
            from aisteer360.algorithms.state_control._common.model_layout import resolve_model_layout

            source = model if model is not None else self._model_ref
            if source is None:
                raise RuntimeError(
                    f"{type(self).__name__} was steered without a live model, so hook module "
                    "names are unresolved; provide the model (the pipeline does) or steer with "
                    "a model."
                )
            layout = resolve_model_layout(source)
            self._module_layout = layout
        return layout

    def get_hooks(self, input_ids, runtime_kwargs=None, attention_mask=None, **kwargs):
        """Compile the bound interventions to hooks for the current generation.

        Delegates to `build_hooks`: a fresh hook runtime is created, gates reset to the
        logical batch, and one behavior hook is emitted per (intervention, layer).

        Args:
            input_ids: Prompt token ids of shape `[B, T]` or `[T]`.
            runtime_kwargs: Unused.
            attention_mask: The prompt attention mask matching `input_ids`, forwarded to
                condition scorers on the prefill pass. When None and the tokenizer defines a
                pad token, a mask is inferred from leading and trailing pad runs.
            **kwargs: Generation-time context; `model` is consulted to resolve hook module
                names when steering ran without a live model.

        Returns:
            Hook specifications with `"pre"`, `"forward"`, `"backward"` keys.
        """
        from aisteer360.algorithms.state_control._common.runtime import build_hooks
        from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
        from aisteer360.utils.tokenization import infer_attention_mask_from_ids

        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        layout = self._resolve_module_layout(kwargs.get("model"))
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None) if self.tokenizer is not None else None
        prompt_lens = compute_prompt_lens(ids, pad_token_id)

        if attention_mask is not None:
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.as_tensor(attention_mask)
            prompt_mask = (mask.unsqueeze(0) if mask.ndim == 1 else mask).to(torch.bool)
        elif pad_token_id is not None:
            prompt_mask = infer_attention_mask_from_ids(ids, pad_token_id).to(torch.bool)
        else:
            prompt_mask = None

        return build_hooks(self.interventions, layout, prompt_lens, prompt_mask)

    def export_intervention_spec(self, runtime_kwargs: dict | None = None):
        """The control's `InterventionSpec`, lowered from the bound interventions, or None.

        Must be called after `steer()`. Returns None when the configuration has no wire form.
        """
        from aisteer360.algorithms.state_control._common.specs import lower_interventions

        if not self.interventions or getattr(self, "_num_layers", None) is None:
            return None
        kinds = self.wire_kinds()
        if kinds is None:
            return None
        return lower_interventions(self.interventions, num_layers=self._num_layers)

    def wire_kinds(self):
        """The combined wire kinds of the bound interventions (or the template before
        `steer()`), or None when any intervention is hook-only."""
        from aisteer360.algorithms.state_control._common.specs import combine_kinds

        source = self.interventions or self._template
        return combine_kinds(intervention.wire_kinds() for intervention in source)

    def _steer_requirement(self) -> tuple:
        """The steer-phase alternatives, derived from the template's unbound elements.

        A fully bound template requires nothing at steer, since pure layer selectors resolve
        from structural facts available on any session. Otherwise the strongest declared
        source need wins: any source declaring `steer_needs = "in_process_torch"` (or an
        undeclared source, or a factory-built transform) requires the in-process backend;
        templates whose unbound sources all declare `steer_needs = "hidden_capture"` require
        `HIDDEN_CAPTURE`.
        """
        from aisteer360.algorithms.core.execution.capabilities import Capability
        from aisteer360.algorithms.core.execution.requirements import needs
        from aisteer360.algorithms.state_control._common.transforms.base import (
            BaseTransform,
            unwrap_modifiers,
        )

        needs_torch = False
        needs_capture = False
        hint = None

        def note(source) -> None:
            nonlocal needs_torch, needs_capture, hint
            declared = getattr(source, "steer_needs", None)
            if declared == "none":  # resolution is model-free (e.g. a precomputed vector)
                return
            if declared == "hidden_capture":
                needs_capture = True
            else:
                needs_torch = True
            if hint is None:
                hint = getattr(source, "steer_hint", None)

        for intervention in self._template:
            transform = intervention.transform
            if isinstance(transform, BaseTransform):
                core, wrappers = unwrap_modifiers(transform)
                for element in (core, *wrappers):
                    if not element.is_bound and element.source is not None:
                        note(element.source)
            elif getattr(transform, "steer_needs", None) is not None:
                note(transform)  # a factory declaring its own steer-phase need
            else:  # an undeclared factory slot builds its transform on the live model
                needs_torch = True
                if hint is None:
                    hint = "supply a transform with a concrete artifact, or steer on the huggingface backend"
            gate = intervention.gate
            if not _is_concrete_gate(gate):
                note(gate)

        if needs_torch:
            return needs(Capability.IN_PROCESS_TORCH, hint=hint)
        if needs_capture:
            return needs(Capability.HIDDEN_CAPTURE, hint=hint)
        return ()

    def requirements(self) -> Requirements:
        """Backend requirements derived from the declared interventions, per phase.

        Generate offers the intervention-spec alternative whenever every component of every
        intervention has a wire form; hook-only configurations require the in-process backend.
        Steer requires model-side work exactly when the template carries unbound sources.
        Score is in-process: remote prompt-logprob scoring anchors token scopes at the
        request's prompt end (the end of the prompt-plus-reference concatenation), which would
        silently unanchor prompt-relative interventions.
        """
        from aisteer360.algorithms.core.execution.capabilities import Capability
        from aisteer360.algorithms.core.execution.requirements import Requirements, any_of, needs

        kinds = self.wire_kinds()
        in_process = needs(Capability.IN_PROCESS_TORCH)
        score = needs(
            Capability.IN_PROCESS_TORCH,
            hint=(
                "remote prompt-logprob scoring anchors token scopes at the request's prompt "
                "end, so scoped interventions would not cover the reference; score on the "
                "huggingface backend"
            ),
        )
        steer = self._steer_requirement()
        if kinds is None:
            return Requirements(
                steer=steer,
                generate=needs(Capability.IN_PROCESS_TORCH, hint=self.hook_only_hint),
                score=score,
            )
        return Requirements(
            steer=steer,
            generate=any_of(
                in_process,
                needs(
                    Capability.INTERVENTION_SPECS,
                    kinds=kinds,
                    hint="serve this intervention through the vLLM-Hook plugin",
                ),
            ),
            score=score,
        )

    def clone_for_call(self, seed: int | None = None):
        """A per-call clone whose interventions carry independent gate state.

        Gates are deep-copied with one shared memo across the control's interventions, so a
        gate instance shared by several interventions stays shared inside the clone while
        being isolated from the original and from sibling clones. Transforms, scorers, and
        steer-time artifacts stay shared.
        """
        import dataclasses

        clone = super().clone_for_call(seed)
        if self.interventions:
            memo: dict = {}
            clone.interventions = tuple(
                dataclasses.replace(intervention, gate=copy.deepcopy(intervention.gate, memo))
                for intervention in self.interventions
            )
        return clone
