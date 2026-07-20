"""Inference-Time Intervention (ITI) state control."""
from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.intervention import (
    HookTarget,
    Intervention,
    InterventionPlan,
    PromptContext,
)
from aisteer360.algorithms.state_control._common.model_layout import resolve_model_layout
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import TopKHeadSelector
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.transforms import HeadAdditiveTransform, NormPreservingTransform

from .args import ITIArgs
from .utils import ProbeMassShiftEstimator


class ITI(StateControl):
    """Inference-Time Intervention (ITI).

    Steers model behavior by shifting activations at a sparse set of attention heads
    during inference. The intervention operates at the residual stream level by adding
    direction vectors to head-associated slices of the hidden dimension.

    ITI operates in two phases:

    1. **Offline (during steer())**: For every attention head across all layers,
       extract the head's output activations on labeled true/false statements.
       Train a per-head linear probe; rank heads by probe accuracy. For the
       top-K heads, compute the mass mean shift: direction = mean(activations_true)
       - mean(activations_false).

    2. **Online (during generation)**: At each generated token, for each selected
       (layer, head) pair, add alpha * direction to that head's slice of the
       residual stream. The intervention fires unconditionally on every token
       in the specified token_scope.

    Reference:

    - "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
    Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
    [https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)
    """

    Args = ITIArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] = []
        self._oproj_names: list[str] = []
        self._active_layer_ids: set[int] = set()
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime = TransformHookRuntime(hook_point="layer_input")

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Initialize ITI by training or loading the steering vector.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data.

        Returns:
            The input model, unchanged.
        """
        device = next(model.parameters()).device
        layout = resolve_model_layout(model)
        self._layer_names = layout.layer_names
        self._oproj_names = layout.oproj_names

        # resolve steering vector
        if self.steering_vector is not None:
            sv = self.steering_vector
        else:
            estimator = ProbeMassShiftEstimator()
            sv = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec)

        # move to device
        sv = sv.to(device, dtype=model.dtype)
        self._steering_vector = sv

        # resolve head selection
        if self.selected_heads is not None:
            selected = self.selected_heads
        else:
            if sv.probe_accuracies is None:
                raise ValueError(
                    "steering_vector has no probe_accuracies. "
                    "Either provide selected_heads explicitly or use data to train a new vector."
                )
            selector = TopKHeadSelector(self.num_heads)
            selected = selector.select(steering_vector=sv)

        # group selected heads by layer
        active_heads: dict[int, set[int]] = {}
        for layer_id, head_id in selected:
            active_heads.setdefault(layer_id, set()).add(head_id)

        self._active_layer_ids = set(active_heads.keys())

        # build transform
        head_transform = HeadAdditiveTransform(
            sv,
            active_heads=active_heads,
            strength=self.alpha,
        )

        # fold head-space additions through each active layer's o_proj into residual-space `add`
        # vectors so the transform is wire-portable post-steer (exact for a linear o_proj)
        oproj_weights: dict[int, torch.Tensor] = {}
        for layer_id in self._active_layer_ids:
            oproj = model.get_submodule(self._oproj_names[layer_id])
            weight = getattr(oproj, "weight", None)
            if weight is not None:
                oproj_weights[layer_id] = weight.detach()
        if oproj_weights:
            head_transform.fold_to_residual(oproj_weights)

        transform = NormPreservingTransform(head_transform) if self.use_norm_preservation else head_transform
        self._transform = transform

        # store tokenizer info for hook generation
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None

        return model

    def plan(
        self,
        prompt_ctx: PromptContext,
        runtime_kwargs: dict | None = None,
    ) -> InterventionPlan:
        """Return one head-additive intervention over the active layers' `o_proj` inputs.

        Each target hooks a layer's `o_proj` submodule as a pre-hook (`hook_point="layer_input"`),
        so the transform edits the concatenated per-head attention outputs before the output
        projection — the paper's intervention point (after Att, before Q^h_l).

        Args:
            prompt_ctx: Per-generation prompt context.
            runtime_kwargs: Unused.

        Returns:
            A one-intervention plan, or an empty plan when no heads are active.
        """
        active = sorted(self._active_layer_ids)
        if not active:
            return []
        return [
            Intervention(
                targets=[HookTarget(module=self._oproj_names[lid], layer_id=lid) for lid in active],
                hook_point="layer_input",
                transform=self._transform,
                scope=self.token_scope,
                scope_params={"last_k": self.last_k, "from_position": self.from_position},
            )
        ]

    def reset(self):
        """Reset internal state between generation calls."""
        self._gate.reset()
        if self._runtime._prompt_lens is not None:
            self._runtime.reset(self._runtime._prompt_lens)
