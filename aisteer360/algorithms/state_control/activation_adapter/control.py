"""ActivationAdapter: assemble an activation-steering recipe from `_common` components."""
from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.intervention import (
    ConditionSpec,
    HookTarget,
    Intervention,
    InterventionPlan,
    PromptContext,
)
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import ConditionPointSelector
from aisteer360.algorithms.state_control._common.transforms.base import BaseTransform
from aisteer360.algorithms.state_control._common.transforms.context import resolve_transform_slot

from .args import ActivationAdapterArgs

logger = logging.getLogger(__name__)


class ActivationAdapter(StateControl):
    """Composable activation-steering control (single-behavior atom).

    `ActivationAdapter` wires together the `state_control/_common` component families — a transform
    (which carries its own steering artifact), a selector, a gate, and a token scope — so an
    activation-steering recipe can be assembled directly without writing a new control class. It
    edits the residual stream at one or more layers during generation, applying the transform at
    masked positions whenever its gate is open.

    The transform is the sole artifact carrier: it holds a concrete `SteeringVector` / directions
    mapping (bound at construction), or an `ArtifactSource` such as `ContrastiveFit(data=...)` that
    the adapter resolves once at `steer()` time and binds via `transform.bind(ctx)`. The adapter has
    no artifact slots and never sees a `SteeringVector` directly.

    Steering multiple behaviors is done by placing multiple adapters in a pipeline's `controls` list
    (each adapter owns exactly one transform chain / gate / token scope). Joint conditioning is
    achieved by sharing one gate instance across adapters: one **driver** declares the condition
    path (`condition_layer_ids` + `score_fn`) and updates the gate; N **followers** pass the same
    gate instance with `gate_driven_externally=True` and read its decision. Gate reads are
    side-effect-free and `reset()` is idempotent, so the pipeline's per-control reset double-resets
    harmlessly.

    Follower timing: within a forward pass, a follower's behavior hook at layer L reads `is_open()`
    when L forwards, so it observes driver evidence only from condition layers `< L`; evidence from
    layers `>= L` takes effect on the next pass (the same staleness class as a single gated adapter,
    evaluated against the driver's condition layers). When a driver and follower hook the same layer
    index, place the driver before the follower in the pipeline's `controls` list (registration
    order = execution order).

    The adapter operates in two phases:

    1. **Preparation (`steer`, offline)**: resolve the behavior layers (from `layer_ids` or the
       `layer_selector`), build the `TransformContext` (sizes + a resolver closure over the model),
       bind the transform (or invoke the factory), verify the transform covers every behavior layer,
       and construct the shared hook runtime.
    2. **Inference (`get_hooks`, online)**: emit condition hooks (read-only, feeding the gate) and
       behavior hooks (applying the transform) at the resolved layers.

    Batching is native (`supports_batching = True`): gates are row-vectorized, so a gated adapter
    scores and gates each prompt of a batch independently, exactly as per-item calls would. The
    gate rejects scalar scores for multi-row batches, so a mis-specified scorer fails loudly
    rather than silently applying one decision batch-wide.
    """

    Args = ActivationAdapterArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._transform: BaseTransform | None = None
        self._layer_names: list[str] = []
        self._layer_ids: list[int] = []
        self._condition_layer_ids: list[int] = []
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime: TransformHookRuntime | None = None

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Resolve the behavior layers, bind the transform, verify coverage, and build the hook runtime.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data (when the transform carries a source).

        Returns:
            The input model, unchanged.
        """
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names
        num_layers = len(layer_names)

        # behavior-layer resolution
        if self.layer_ids is not None:
            layer_ids = sorted(set(int(lid) for lid in self.layer_ids))
        else:
            if isinstance(self.layer_selector, ConditionPointSelector):
                raise ValueError(
                    "ConditionPointSelector returns a ConditionPoint for gating, not a behavior layer; "
                    "supply layer_ids or a layer selector that returns layer id(s)."
                )
            selected = self.layer_selector.select(num_layers=num_layers)
            layer_ids = sorted(set(selected)) if isinstance(selected, (list, tuple, set)) else [int(selected)]
        self._layer_ids = layer_ids

        for lid in layer_ids:
            if not 0 <= lid < num_layers:
                raise ValueError(f"layer_id {lid} out of range [0, {num_layers}).")

        self._condition_layer_ids = sorted(set(int(lid) for lid in self.condition_layer_ids or []))
        for lid in self._condition_layer_ids:
            if not 0 <= lid < num_layers:
                raise ValueError(f"condition_layer_id {lid} out of range [0, {num_layers}).")

        # transform resolution (no artifact logic; the transform carries its own)
        self._transform = resolve_transform_slot(self.transform, model, tokenizer, layer_ids)

        self._gate = self.gate if self.gate is not None else AlwaysOpenGate()
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None
        self._runtime = TransformHookRuntime(hook_point=self.hook_point)

        return model

    def reset(self):
        """Reset the gate and runtime position/prefill state between generation calls."""
        num_rows = max(self._runtime.num_logical_rows, 1) if self._runtime is not None else 1
        self._gate.reset(num_rows)
        if self._runtime is not None and self._runtime._prompt_lens is not None:
            self._runtime.reset(self._runtime._prompt_lens, self._runtime._prompt_mask)

    def plan(
        self,
        prompt_ctx: PromptContext,
        runtime_kwargs: dict | None = None,
    ) -> InterventionPlan:
        """Return one intervention (behavior transform + optional condition path).

        The condition path is the custom-gate form: an arbitrary `score_fn` feeds the adapter's own
        gate (shared by identity in the driver/follower pattern). A follower carries the shared gate
        with `gate_driven_externally=True` and contributes no condition hooks.

        Args:
            prompt_ctx: Per-generation prompt context (ids, mask, prompt lengths).
            runtime_kwargs: Unused in v1.

        Returns:
            A one-intervention plan.
        """
        condition = None
        if self._condition_layer_ids and not self.gate_driven_externally:
            condition = ConditionSpec(
                targets=[HookTarget(module=self._layer_names[lid], layer_id=lid) for lid in self._condition_layer_ids],
                scorer=self.score_fn,
                comp_mode="mean",
                location=self.hook_point,
            )

        return [
            Intervention(
                targets=[HookTarget(module=self._layer_names[lid], layer_id=lid) for lid in self._layer_ids],
                hook_point=self.hook_point,
                transform=self._transform,
                scope=self.token_scope,
                scope_params={"last_k": self.last_k, "from_position": self.from_position},
                gate=self._gate,
                condition=condition,
                gate_driven_externally=self.gate_driven_externally,
            )
        ]

    def _has_condition_path(self) -> bool:
        """Whether this adapter drives a condition (has condition layers and is not a follower)."""
        return bool(self._condition_layer_ids) and not self.gate_driven_externally

    def cleanup(self) -> None:
        """Drop references to the bound transform and runtime."""
        self._transform = None
        self._runtime = None
