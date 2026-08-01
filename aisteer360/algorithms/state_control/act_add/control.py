"""ActAdd (Activation Addition) control implementation."""
from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.execution.capabilities import Capability, InterventionKinds
from aisteer360.algorithms.core.execution.interventions import InterventionSpec
from aisteer360.algorithms.core.execution.requirements import Requirements, needs
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.estimators import SinglePairEstimator
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.intervention_export import (
    intervention_generate_requirement,
    intervention_spec_from_runtime_config,
)
from aisteer360.algorithms.state_control._common.layout_facts import cast_steering_vector, resolve_layout
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import FixedLayerSelector, FractionalDepthSelector
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform, NormPreservingTransform

from .args import ActAddArgs


class ActAdd(StateControl):
    """Activation Addition (ActAdd).

    Steers model behavior by adding a positional steering vector, computed from a single contrast
    pair of short prompts, to the residual stream at a single layer during the initial forward
    pass.

    Reference:

    - "Steering Language Models With Activation Engineering"
    Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid
    [https://arxiv.org/abs/2308.10248](https://arxiv.org/abs/2308.10248)
    """

    Args = ActAddArgs
    supports_batching = False  # ActAdd uses positional alignment which breaks with left-padding

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] | None = None
        self._layer_id: int = 0
        self._num_layers: int | None = None
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime = TransformHookRuntime(hook_point="layer_input")

    def _intervention_kind_plan(self) -> InterventionKinds | None:
        """Kind names this configuration lowers to; None marks it hook-only.

        Prompt-pair fitting produces positional (`T > 1`) directions, which have no wire form,
        so only broadcast vector-supplied configurations plan kinds. The pre-hook at layer 0
        edits the embedding output, which also has no wire form.
        """
        if self._transform is not None:
            plan = self._transform.wire_kind_plan()
        else:
            source = self._steering_vector if self._steering_vector is not None else self.steering_vector
            if source is None or source.is_positional:
                return None
            plan = ("additive", frozenset({"norm_preserving"}) if self.use_norm_preservation else frozenset())
        if plan is None:
            return None
        if self.layer_id == 0:
            return None
        if self._transform is not None and self._layer_id == 0:
            return None
        kind, modifiers = plan
        return InterventionKinds(
            transforms=frozenset({kind}),
            modifiers=modifiers,
            scopes=frozenset({"all"}),
        )

    def requirements(self) -> Requirements:
        """In-process hooks or intervention specs at generate; fitting from prompts steers in-process."""
        steer = ()
        if self.steering_vector is None:
            steer = needs(
                Capability.IN_PROCESS_TORCH,
                hint="supply a fitted `steering_vector`, or steer on the huggingface backend",
            )
        return Requirements(
            steer=steer,
            generate=intervention_generate_requirement(self._intervention_kind_plan()),
        )

    def export_intervention_spec(self, runtime_kwargs: dict | None = None) -> InterventionSpec | None:
        """The `additive` spec for broadcast directions; None for positional configurations.

        The pre-hook at layer `l` edits the stream entering the layer, which is the wire
        boundary after layer `l - 1`.
        """
        if self._transform is None or self._num_layers is None:
            return None
        return intervention_spec_from_runtime_config(
            transform=self._transform,
            layer_ids=[self._layer_id],
            token_scope="all",
            gate=self._gate,
            num_layers=self._num_layers,
            placement="layer_input",
            runtime_kwargs=runtime_kwargs,
        )

    def steer(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session=None,
        **__,
    ) -> PreTrainedModel | None:
        """Extract or load the steering vector and build the transform.

        Structural facts (layer count, dtype) come from the steering session's layout when a
        session is given; a vector-supplied configuration therefore steers with `model=None`.
        Fitting from a prompt pair requires a live model.

        Args:
            model: The base language model to be steered, or None for vector-supplied
                configurations steered against a session layout.
            tokenizer: Tokenizer for encoding the prompt pair.
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
            if model is None:
                raise ValueError("Fitting ActAdd from a prompt pair requires a live model at steer time.")
            estimator = SinglePairEstimator()
            sv = estimator.fit(
                model,
                tokenizer,
                positive_prompt=self.positive_prompt,
                negative_prompt=self.negative_prompt,
            )

        # clone before any in-place cast/normalize so a caller-supplied vector is never mutated
        sv = cast_steering_vector(sv, layout)

        # resolve layer_id via selector
        if self.layer_id is not None:
            selector = FixedLayerSelector(self.layer_id)
        else:
            # heuristic: ~20% depth (paper uses layer 6/48 for GPT-2-XL)
            selector = FractionalDepthSelector(fraction=0.2, minimum=1)
        self._layer_id = selector.select(num_layers=num_layers)

        if self._layer_id not in sv.directions:
            raise ValueError(f"Steering vector has no direction for layer {self._layer_id}.")

        # optionally normalize per-position vectors
        if self.normalize_vector:
            d = sv.directions[self._layer_id]  # [T, H]
            norms = d.norm(dim=-1, keepdim=True)  # [T, 1]
            sv.directions[self._layer_id] = d / (norms + 1e-8)

        self._steering_vector = sv

        # build transform
        transform = AdditiveTransform(
            sv.directions,
            strength=self.multiplier,
            alignment=self.alignment,
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
                    "ActAdd was steered without a live model, so hook module names are unresolved; "
                    "pass `model=` to get_hooks (the pipeline does) or steer with a model."
                )
            _, self._layer_names = get_model_layer_list(source)
        return self._layer_names

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Register a pre-hook on the target layer.

        The steering vector is added to the residual stream before the target layer processes it
        (h_l input) rather than after (h_l output), and a pre-hook ensures correct layer alignment.
        The token scope is always `"all"`, and spatial control comes from the transform's
        alignment-based positional injection rather than the mask. Injection occurs only during
        prefill, because each decode pass has `seq_len == 1`, so the alignment window never
        intersects it and the runtime's position bookkeeping is unused here.

        Args:
            input_ids: Input token IDs (used only to size prompt lengths).
            runtime_kwargs: Unused.
            **kwargs: Generation-time context; `model` is consulted to resolve hook module names
                when steering ran without a live model.

        Returns:
            Hook specifications.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        layer_names = self._module_names(kwargs.get("model"))
        prompt_lens = compute_prompt_lens(ids, self._pad_token_id)
        self._runtime.reset(prompt_lens)

        return {
            "pre": [{
                "module": layer_names[self._layer_id],
                "hook_func": self._runtime.build_behavior_hook(
                    layer_id=self._layer_id,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope="all",
                    is_pass_opener=True,  # single-layer control: its only hook opens the pass
                ),
            }],
            "forward": [],
            "backward": [],
        }
