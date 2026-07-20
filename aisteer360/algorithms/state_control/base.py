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
from abc import ABC
from dataclasses import fields
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.core.base_args import BaseArgs
from aisteer360.core.requirements import Capability, Requirements

if TYPE_CHECKING:
    from aisteer360.algorithms.state_control._common.intervention import InterventionPlan, PromptContext
    from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime

PreHook = Callable[[nn.Module, tuple], tuple | torch.Tensor]
ForwardHook = Callable[[nn.Module, tuple, torch.Tensor], torch.Tensor]
BackwardHook = Callable[[nn.Module, tuple, tuple], tuple]
HookSpec = dict[str, str | PreHook | ForwardHook | BackwardHook]


class StateControl(ABC):
    """Abstract base class for state control steering methods.

    Modifies internal model states during forward passes via hooks.

    A control instance holds per-generation state on `self` (e.g. position offsets, cached masks,
    gate decisions) and therefore supports one in-flight generation at a time; do not share a single
    control instance across concurrently running pipelines.

    Two sanctioned extension points, exactly one of which a concrete subclass overrides (enforced in
    `__init_subclass__`):

    - `plan()` — the declarative path: return an `InterventionPlan` (pure data describing the
        transforms/gating). The base `get_hooks` compiles it to hooks via `compile_plan_to_hooks`,
        and the same plan is portable to server backends where its components export. Preferred for
        residual-stream steering.
    - `get_hooks()` — the hook-level path: return hook specs directly. For controls that cannot be
        expressed declaratively (e.g. PASTA's attention-mask writes); implies
        `Capability.FORWARD_HOOKS`.

    Methods:
        plan(prompt_ctx, runtime_kwargs) -> InterventionPlan | None: Declarative interventions.
        get_hooks(input_ids, runtime_kwargs, **kwargs) -> dict: Hook specs (concrete; routes
            through `plan` unless overridden).
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

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce that a concrete state control overrides exactly one of `plan` / `get_hooks`.

        Raises:
            TypeError: If a concrete subclass overrides neither entry point, or both.
        """
        super().__init_subclass__(**kwargs)
        # skip abstract or internal base subclasses (no-op controls set enabled=False as a marker)
        if getattr(cls, "__abstractmethods__", None):
            return
        overrides_plan = cls.plan is not StateControl.plan
        overrides_hooks = cls.get_hooks is not StateControl.get_hooks
        if not overrides_plan and not overrides_hooks and cls.enabled:
            raise TypeError(
                f"{cls.__name__} must override exactly one of `plan()` or `get_hooks()`; it overrides "
                "neither. Implement `plan()` for declarative controls or `get_hooks()` for hook-level "
                "controls (e.g. attention writes)."
            )
        if overrides_plan and overrides_hooks:
            raise TypeError(
                f"{cls.__name__} overrides both `plan()` and `get_hooks()`; override exactly one. "
                "Declarative controls implement `plan()`; hook-level controls implement `get_hooks()`."
            )

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is None:  # null control
            if args or kwargs:
                raise TypeError(f"{type(self).__name__} accepts no constructor arguments.")
            return

        self.args: BaseArgs = self.Args.validate(*args, **kwargs)

        # move fields to attributes, skipping any name the control exposes as a property
        # (e.g. CAST.condition_point); the raw value stays reachable via self.args.<name>
        for field in fields(self.args):
            if isinstance(getattr(type(self), field.name, None), property):
                continue
            setattr(self, field.name, getattr(self.args, field.name))

        self.hooks: dict[str, list[HookSpec]] = {"pre": [], "forward": [], "backward": []}
        self.registered: list[torch.utils.hooks.RemovableHandle] = []

    def plan(
        self,
        prompt_ctx: "PromptContext",
        runtime_kwargs: dict | None = None,
    ) -> "InterventionPlan | None":
        """Return the declarative interventions for the current generation, or `None`.

        The declarative extension point. Declarative controls override this to describe their
        transforms and gating as data; the base `get_hooks` compiles the result. The default returns
        `None` (the control is hook-level and overrides `get_hooks` instead).

        Args:
            prompt_ctx: Per-generation prompt context (ids, mask, prompt lengths, pad id).
            runtime_kwargs: Per-call parameters for the control.

        Returns:
            An `InterventionPlan`, or `None` for hook-level controls.
        """
        return None

    def _ensure_runtime(self) -> "TransformHookRuntime":
        """Return the control's shared hook runtime (built in `steer()`).

        Raises:
            RuntimeError: If the control has no `_runtime` (steer() not called, or a hook-level
                control that does not use the shared runtime).
        """
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            raise RuntimeError(
                f"{type(self).__name__} has no hook runtime; call steer() before generation "
                "(declarative controls build their runtime in steer())."
            )
        return runtime

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, list[HookSpec]]:
        """Create hook specifications for the current generation.

        The concrete default routes through `plan()`: it builds a `PromptContext`, calls `plan()`,
        and compiles the result via `compile_plan_to_hooks`. Hook-level controls override this
        directly and return hook specs of their own.

        Args:
            input_ids: Prompt token ids of shape [batch, seq_len].
            runtime_kwargs: Per-call parameters for the control.
            attention_mask: The prompt attention mask matching `input_ids` (or None); forwarded to
                condition scorers so they align with real (non-pad) positions.
            **kwargs: Additional generation-time context.

        Raises:
            NotImplementedError: If neither `plan()` nor `get_hooks()` is overridden.
        """
        from aisteer360.algorithms.state_control._common.intervention import PromptContext
        from aisteer360.algorithms.state_control._common.runtime import compile_plan_to_hooks

        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        # prefer a live tokenizer pad id (some controls resolve the pad token after steer), falling
        # back to the id cached at steer time
        pad_token_id = getattr(getattr(self, "tokenizer", None), "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self, "_pad_token_id", None)
        prompt_ctx = PromptContext.from_ids(ids, attention_mask=attention_mask, pad_token_id=pad_token_id)
        intervention_plan = self.plan(prompt_ctx, runtime_kwargs)
        if intervention_plan is None:
            raise NotImplementedError(
                f"{type(self).__name__} overrides neither `plan()` nor `get_hooks()`."
            )
        return compile_plan_to_hooks(
            intervention_plan, runtime=self._ensure_runtime(), prompt_ctx=prompt_ctx
        )

    def steer(self,
              model: PreTrainedModel,
              tokenizer: PreTrainedTokenizerBase = None,
              **kwargs) -> None:
        """Optional steering/preparation."""
        pass

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

    def reset(self):
        """Optional reset call for state control."""
        pass

    def cleanup(self) -> None:
        """Release resources allocated during steer().

        Override this method in subclasses that allocate GPU memory or other resources
        during steering to ensure proper cleanup.
        """
        pass

    def requires(self) -> Requirements:
        """Return the backend capabilities this control needs at generation time.

        Declarative (`plan`-based) controls need `RESIDUAL_WRITE`, plus `SERVER_GATING | HIDDEN_READ`
        when the control carries a condition path. Hook-level controls (overriding `get_hooks`)
        register arbitrary in-process hooks and need `FORWARD_HOOKS`. Disabled controls require
        nothing. Per-component portability (whether a transform exports to the wire) is refined by
        the compiler at validation time.

        Returns:
            The control's `Requirements` (phase `"generate"`).
        """
        if not getattr(self, "enabled", True):
            return Requirements()
        if type(self).plan is not StateControl.plan:
            capabilities = Capability.RESIDUAL_WRITE
            if self._has_condition_path():
                capabilities |= Capability.SERVER_GATING | Capability.HIDDEN_READ
            return Requirements(capabilities=capabilities, phase="generate")
        return Requirements(capabilities=Capability.FORWARD_HOOKS, phase="generate")

    def _has_condition_path(self) -> bool:
        """Whether this control gates behavior on a runtime condition (override to report `True`)."""
        return False

    def portability_hint(self) -> bool:
        """Whether this control's plan is expected to be wire-portable (override as needed).

        Default `True` for declarative controls (refined per-component by the compiler in doc 06)
        and `False` for hook-level controls.
        """
        return type(self).plan is not StateControl.plan


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

    def reset(self):
        """Null reset operation."""
        pass
