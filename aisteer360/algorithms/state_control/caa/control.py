from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.execution.capabilities import Capability, InterventionKinds
from aisteer360.algorithms.core.execution.interventions import InterventionSpec
from aisteer360.algorithms.core.execution.requirements import Requirements, needs
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.intervention_export import (
    intervention_generate_requirement,
    intervention_spec_from_runtime_config,
)
from aisteer360.algorithms.state_control._common.layout_facts import cast_steering_vector, resolve_layout
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import FixedLayerSelector, FractionalDepthSelector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform, NormPreservingTransform

from aisteer360.algorithms.state_control._common.estimators import ContrastiveDirectionEstimator, MeanDifferenceEstimator
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector

from .args import CAAArgs


class CAA(StateControl):
    """Contrastive Activation Addition (CAA).

    Steers model behavior by adding a learned mean-difference direction
    vector to the residual stream at a single layer during generation.

    CAA operates in two phases:

    1. **Training (offline)**: Given contrastive prompt pairs where each pair
       shares the same question but ends with opposite answer tokens, extract
       residual stream activations at the answer-token position. The steering
       vector is the mean difference between positive and negative activations.

    2. **Inference (online)**: Add `multiplier * v_L` to the residual stream
       at a chosen layer L, at all token positions after the user's prompt.
       A positive multiplier increases the target behavior; negative decreases it.

    Reference:

    - "Steering Llama 2 via Contrastive Activation Addition"
    Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
    [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
    """

    Args = CAAArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] | None = None
        self._layer_id: int = 0
        self._num_layers: int | None = None
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime = TransformHookRuntime(hook_point="layer_output")

    def _intervention_kind_plan(self) -> InterventionKinds | None:
        """Kind names this configuration lowers to; None marks it hook-only."""
        transform = self._transform
        if transform is not None:
            plan = transform.wire_kind_plan()
        else:
            source = self._steering_vector if self._steering_vector is not None else self.steering_vector
            if source is not None and source.is_positional:
                return None
            modifiers = frozenset({"norm_preserving"}) if self.use_norm_preservation else frozenset()
            plan = ("additive", modifiers)
        if plan is None:
            return None
        kind, modifiers = plan
        return InterventionKinds(
            transforms=frozenset({kind}),
            modifiers=modifiers,
            scopes=frozenset({self.token_scope}),
        )

    def requirements(self) -> Requirements:
        """In-process hooks or intervention specs at generate; fitting from data steers in-process."""
        steer = ()
        if self.steering_vector is None:
            steer = needs(
                Capability.HIDDEN_CAPTURE,
                hint=(
                    "supply a fitted `steering_vector`, or run the steer phase on a backend "
                    "with hidden-state capture (huggingface, or offline vLLM with the plugin)"
                ),
            )
        return Requirements(
            steer=steer,
            generate=intervention_generate_requirement(
                self._intervention_kind_plan(),
                hook_only_hint="positional directions have no intervention-spec form; run on the huggingface backend",
            ),
        )

    def export_intervention_spec(self, runtime_kwargs: dict | None = None) -> InterventionSpec | None:
        """The `additive` spec for the steered layer; None for positional configurations."""
        if self._transform is None or self._num_layers is None:
            return None
        return intervention_spec_from_runtime_config(
            transform=self._transform,
            layer_ids=[self._layer_id],
            token_scope=self.token_scope,
            gate=self._gate,
            num_layers=self._num_layers,
            placement="layer_output",
            last_k=self.last_k,
            from_position=self.from_position,
            runtime_kwargs=runtime_kwargs,
        )

    def steer(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session=None,
        **__,
    ) -> PreTrainedModel | None:
        """Initialize CAA by training or loading the steering vector.

        Structural facts (layer count, dtype) come from the steering session's layout when a
        session is given; a vector-supplied configuration therefore steers with `model=None`.
        Fitting from `data` requires a live model.

        Args:
            model: The base language model to be steered, or None for vector-supplied
                configurations steered against a session layout.
            tokenizer: Tokenizer for encoding training data.
            session: `SteeringSession` on the steering backend, provided by the pipeline.

        Returns:
            The input model, unchanged.
        """
        layout = resolve_layout(model, session)
        num_layers = layout.num_layers
        self._num_layers = num_layers
        self._layer_names = get_model_layer_list(model)[1] if model is not None else None

        # resolve steering vector
        if self.steering_vector is not None:
            sv = self.steering_vector
        else:
            if self.train_spec.method == "pca_pairwise":
                estimator = ContrastiveDirectionEstimator()
            else:
                estimator = MeanDifferenceEstimator()
            sv = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec, session=session)

        # clone before the in-place cast/normalize so a caller-supplied vector is never mutated
        sv = cast_steering_vector(sv, layout)

        # optionally normalize the vector
        if self.normalize_vector:
            for layer_id, direction in sv.directions.items():
                norm = direction.norm()
                if norm > 0:
                    sv.directions[layer_id] = direction / norm

        self._steering_vector = sv

        # resolve layer_id via selector
        if self.layer_id is not None:
            selector = FixedLayerSelector(self.layer_id)
        else:
            # heuristic: ~40% depth (paper finds layer 13/32 optimal)
            selector = FractionalDepthSelector(fraction=0.4)
        self._layer_id = selector.select(num_layers=num_layers)

        # validate layer is present in steering vector
        if self._layer_id not in sv.directions:
            raise ValueError(f"Steering vector has no direction for layer {self._layer_id}.")

        # build transform
        transform = AdditiveTransform(
            sv.directions,
            strength=self.multiplier,
        )
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)
        self._transform = transform

        # store tokenizer info for hook generation
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None

        return model

    def _module_names(self, model) -> list[str]:
        """Layer module names, resolved from the module tree on first use."""
        if self._layer_names is None:
            source = model if model is not None else self._model_ref
            if source is None:
                raise RuntimeError(
                    "CAA was steered without a live model, so hook module names are unresolved; "
                    "pass `model=` to get_hooks (the pipeline does) or steer with a model."
                )
            _, self._layer_names = get_model_layer_list(source)
        return self._layer_names

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **kwargs,
    ) -> dict[str, list]:
        """Create forward hook for activation addition at the target layer.

        Registers a forward hook that adds the steering vector to the output of
        the target layer, modifying the residual stream at that point.

        Args:
            input_ids: Input token IDs.
            runtime_kwargs: Runtime parameters (currently unused).
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
        self._runtime.reset(prompt_lens)

        return {
            "pre": [],
            "forward": [{
                "module": layer_names[self._layer_id],
                "hook_func": self._runtime.build_behavior_hook(
                    layer_id=self._layer_id,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    is_pass_opener=True,  # single-layer control: its only hook opens the pass
                ),
            }],
            "backward": [],
        }
