"""ActivationAdapter: assemble an activation-steering recipe from `_common` components."""
from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.layout_facts import resolve_layout
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import ConditionPointSelector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms.base import BaseTransform
from aisteer360.algorithms.state_control._common.transforms.context import resolve_transform_slot

from .args import ActivationAdapterArgs

logger = logging.getLogger(__name__)


class ActivationAdapter(StateControl):
    """Composable activation-steering control (single-behavior atom).

    `ActivationAdapter` wires together the `state_control/_common` component families (a transform
    that carries its own steering artifact, a selector, a gate, and a token scope) so an
    activation-steering recipe can be assembled directly without writing a new control class. It
    edits the residual stream at one or more layers during generation, applying the transform at
    masked positions whenever its gate is open.

    The transform is the sole artifact carrier. It holds a concrete `SteeringVector` / directions
    mapping (bound at construction), or an `ArtifactSource` such as `ContrastiveFit(data=...)` that
    the adapter resolves once at `steer()` time and binds via `transform.bind(ctx)`. The adapter has
    no artifact slots and never sees a `SteeringVector` directly.

    Steering multiple behaviors is done by placing multiple adapters in a pipeline's `controls` list
    (each adapter owns exactly one transform chain / gate / token scope). Joint conditioning is
    achieved by sharing one gate instance across adapters. One driver declares the condition path
    (`condition_layer_ids` + `score_fn`) and updates the gate; N followers pass the same gate
    instance with `gate_driven_externally=True` and read its decision. Gate reads are
    side-effect-free and `reset()` is idempotent, so the pipeline's per-control reset double-resets
    harmlessly.

    Within a forward pass, a follower's behavior hook at layer L reads `is_open()` when L forwards,
    so it observes driver evidence only from condition layers `< L`. Evidence from layers `>= L`
    takes effect on the next pass. When a driver and follower hook the same layer index, place the
    driver before the follower in the pipeline's `controls` list (registration order = execution
    order).

    The adapter operates in two phases:

    1. **Preparation (`steer`, offline)**: resolve the behavior layers (from `layer_ids` or the
       `layer_selector`), build the `TransformContext` (sizes + a resolver closure over the model),
       bind the transform (or invoke the factory), verify the transform covers every behavior layer,
       and construct the shared hook runtime.
    2. **Inference (`get_hooks`, online)**: emit condition hooks (read-only, feeding the gate) and
       behavior hooks (applying the transform) at the resolved layers.

    Batching is native (`supports_batching = True`); gates are row-vectorized, so a gated adapter
    scores and gates each prompt of a batch independently. The gate rejects scalar scores for
    multi-row batches, so a mis-specified scorer fails loudly rather than silently applying one
    decision batch-wide.
    """

    Args = ActivationAdapterArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._transform: BaseTransform | None = None
        self._layer_names: list[str] | None = None
        self._layer_ids: list[int] = []
        self._condition_layer_ids: list[int] = []
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime: TransformHookRuntime | None = None

    def steer(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session=None,
        **__,
    ) -> PreTrainedModel | None:
        """Resolve the behavior layers, bind the transform, verify coverage, and build the hook runtime.

        Structural facts (layer count, sizes, dtype) come from the steering session's layout when a
        session is given; a configuration whose transform carries a concrete artifact therefore
        steers with `model=None`. A transform carrying a fit source requires a live model.

        Args:
            model: The base language model to be steered, or None for concrete-artifact
                configurations steered against a session layout.
            tokenizer: Tokenizer for encoding training data (when the transform carries a source).
            session: `SteeringSession` on the steering backend, provided by the pipeline.

        Returns:
            The input model, unchanged.
        """
        layout = resolve_layout(model, session)
        num_layers = layout.num_layers
        self._layer_names = get_model_layer_list(model)[1] if model is not None else None

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

        # optional scorer attributes (see ConditionScorer): a scorer that records the boundary
        # or model identity it was fitted at is validated against this adapter and model
        scorer_location = getattr(self.score_fn, "location", None)
        if scorer_location is not None and scorer_location != self.hook_point:
            raise ValueError(
                f"Condition scorer expects features at '{scorer_location}' but this adapter "
                f"hooks '{self.hook_point}'. Construct the adapter with "
                f"hook_point='{scorer_location}', or refit the probe with "
                f"location='{self.hook_point}'."
            )
        scorer_fingerprint = getattr(self.score_fn, "model_fingerprint", None)
        if scorer_fingerprint is not None:
            live_fingerprint = layout.model_fingerprint
            if scorer_fingerprint != live_fingerprint:
                raise ValueError(
                    f"Condition scorer was fitted on a different model (fingerprint "
                    f"{scorer_fingerprint!r} vs {live_fingerprint!r}). Refit the probe on this "
                    "model, or disarm the check with allow_model_mismatch=True on "
                    "probe_condition() or Probe.as_condition()."
                )

        # transform resolution (no artifact logic; the transform carries its own)
        self._transform = resolve_transform_slot(self.transform, model, tokenizer, layer_ids, layout=layout)

        self._gate = self.gate if self.gate is not None else AlwaysOpenGate()
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None
        self._runtime = TransformHookRuntime(hook_point=self.hook_point)

        return model

    def _module_names(self, model) -> list[str]:
        """Layer module names, resolved from the module tree on first use."""
        if self._layer_names is None:
            source = model if model is not None else self._model_ref
            if source is None:
                raise RuntimeError(
                    "ActivationAdapter was steered without a live model, so hook module names are "
                    "unresolved; pass `model=` to get_hooks (the pipeline does) or steer with a model."
                )
            _, self._layer_names = get_model_layer_list(source)
        return self._layer_names

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Emit condition (read-only) and behavior hooks for the current generation.

        Condition hooks are registered before behavior hooks so that, at a shared layer, the gate
        update precedes the transform application (registration order = execution order for same-module
        hooks of the same phase).

        Args:
            input_ids: Prompt token ids of shape `[B, T]` or `[T]`.
            runtime_kwargs: Unused.
            attention_mask: The prompt attention mask matching `input_ids` (forwarded by the
                pipeline). Handed to condition scorers on the prefill pass so condition scores
                align with real (non-pad) prompt positions.
            **kwargs: Generation-time context; `model` is consulted to resolve hook module names
                when steering ran without a live model.

        Returns:
            Hook specifications with "pre", "forward", "backward" keys.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        layer_names = self._module_names(kwargs.get("model"))
        prompt_lens = compute_prompt_lens(ids, self._pad_token_id)
        if attention_mask is not None:
            am = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.as_tensor(attention_mask)
            prompt_mask = (am.unsqueeze(0) if am.ndim == 1 else am).to(torch.bool)
        else:
            prompt_mask = None
        self._gate.reset(ids.size(0))
        self._runtime.reset(prompt_lens, prompt_mask)

        # pass opener = lowest hooked layer (over behavior ∪ condition); exactly one hook may
        # open each pass — at a shared opener layer the condition hook registers (and therefore
        # fires) first, so it takes the opener role
        all_layers = self._layer_ids + self._condition_layer_ids
        opener = min(all_layers) if all_layers else None
        behavior_opener = opener if opener not in self._condition_layer_ids else None

        phase = "forward" if self.hook_point == "layer_output" else "pre"
        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        # condition hooks first (so gate.update precedes transform at a shared layer)
        for lid in self._condition_layer_ids:
            hooks[phase].append({
                "module": layer_names[lid],
                "hook_func": self._runtime.build_condition_hook(
                    layer_id=lid,
                    scorer=self.score_fn,
                    gate=self._gate,
                    is_pass_opener=(lid == opener),
                ),
            })

        for lid in self._layer_ids:
            hooks[phase].append({
                "module": layer_names[lid],
                "hook_func": self._runtime.build_behavior_hook(
                    layer_id=lid,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    is_pass_opener=(lid == behavior_opener),
                ),
            })

        return hooks



    def cleanup(self) -> None:
        """Drop references to the bound transform and runtime."""
        self._transform = None
        self._runtime = None
