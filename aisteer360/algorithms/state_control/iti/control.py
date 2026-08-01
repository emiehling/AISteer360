"""Inference-Time Intervention (ITI) state control."""
from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.execution.capabilities import Capability, InterventionKinds
from aisteer360.algorithms.core.execution.interventions import InterventionSpec
from aisteer360.algorithms.core.execution.requirements import Requirements, needs
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.intervention_export import (
    intervention_generate_requirement,
    intervention_spec_from_runtime_config,
)
from aisteer360.algorithms.state_control._common.layout_facts import cast_steering_vector, resolve_layout
from aisteer360.algorithms.state_control._common.model_layout import resolve_model_layout
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import TopKHeadSelector
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
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
        self._layer_names: list[str] | None = None
        self._oproj_names: list[str] | None = None
        self._active_layer_ids: set[int] = set()
        self._num_layers: int | None = None
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime = TransformHookRuntime(hook_point="layer_input")

    def _intervention_kind_plan(self) -> InterventionKinds | None:
        """Kind names this configuration lowers to; None marks it hook-only.

        The `norm_preserving` wire modifier rescales the per-head stream rather than the full
        residual row, so norm-preserving configurations are hook-only. The wire kind carries
        the `tensor_parallel_size==1` constraint, enforced at submission.
        """
        if self.use_norm_preservation:
            return None
        if self._transform is not None:
            plan = self._transform.wire_kind_plan()
            if plan is None:
                return None
            kind, modifiers = plan
        else:
            kind, modifiers = "head_additive", frozenset()
        return InterventionKinds(
            transforms=frozenset({kind}),
            modifiers=modifiers,
            scopes=frozenset({self.token_scope}),
        )

    def requirements(self) -> Requirements:
        """In-process hooks or intervention specs at generate; fitting always steers in-process.

        Fitting ITI captures pre-`o_proj` per-head activations, a capture kind no backend
        advertises, so `data`-fitted configurations require the in-process backend at steer.
        """
        steer = ()
        if self.steering_vector is None:
            steer = needs(
                Capability.IN_PROCESS_TORCH,
                hint=(
                    "fitting ITI requires head-level capture, which no backend advertises; "
                    "supply `steering_vector` or steer on huggingface"
                ),
            )
        return Requirements(
            steer=steer,
            generate=intervention_generate_requirement(
                self._intervention_kind_plan(),
                hook_only_hint=(
                    "norm preservation over per-head streams has no intervention-spec form; "
                    "run on the huggingface backend"
                ),
            ),
        )

    def export_intervention_spec(self, runtime_kwargs: dict | None = None) -> InterventionSpec | None:
        """The `head_additive` spec over the active layers; None for norm-preserving
        configurations."""
        if self._transform is None or self._num_layers is None:
            return None
        if self._intervention_kind_plan() is None:
            return None
        return intervention_spec_from_runtime_config(
            transform=self._transform,
            layer_ids=sorted(self._active_layer_ids),
            token_scope=self.token_scope,
            gate=self._gate,
            num_layers=self._num_layers,
            placement="o_proj",
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
        """Initialize ITI by training or loading the steering vector.

        Structural facts (dtype) come from the steering session's layout when a session is given;
        a vector-supplied configuration therefore steers with `model=None`. Fitting from `data`
        requires a live model.

        Args:
            model: The base language model to be steered, or None for vector-supplied
                configurations steered against a session layout.
            tokenizer: Tokenizer for encoding training data.
            session: `SteeringSession` on the steering backend, provided by the pipeline.

        Returns:
            The input model, unchanged.
        """
        seam_layout = resolve_layout(model, session)
        self._num_layers = seam_layout.num_layers
        if model is not None:
            module_layout = resolve_model_layout(model)
            self._layer_names = module_layout.layer_names
            self._oproj_names = module_layout.oproj_names
        else:
            self._layer_names = None
            self._oproj_names = None

        # resolve steering vector
        if self.steering_vector is not None:
            sv = self.steering_vector
        else:
            if model is None:
                raise ValueError("Fitting ITI from data requires a live model at steer time.")
            estimator = ProbeMassShiftEstimator()
            sv = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec)

        # clone before the cast so a caller-supplied vector is never mutated
        sv = cast_steering_vector(sv, seam_layout)
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
        transform = HeadAdditiveTransform(
            sv,
            active_heads=active_heads,
            strength=self.alpha,
        )
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)
        self._transform = transform

        # store tokenizer info for hook generation
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None

        return model

    def _module_names(self, model) -> list[str]:
        """Active o_proj module names, resolved from the module tree on first use."""
        if self._oproj_names is None:
            source = model if model is not None else self._model_ref
            if source is None:
                raise RuntimeError(
                    "ITI was steered without a live model, so hook module names are unresolved; "
                    "pass `model=` to get_hooks (the pipeline does) or steer with a model."
                )
            module_layout = resolve_model_layout(source)
            self._layer_names = module_layout.layer_names
            self._oproj_names = module_layout.oproj_names
        return self._oproj_names

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,  # noqa: ARG002
        **kwargs,
    ) -> dict[str, list]:
        """Create pre-hooks on active o_proj modules for pre-projection intervention.

        Registers a pre-hook on each active layer's o_proj. Each pre-hook modifies the input to
        o_proj (the concatenated per-head attention outputs) by adding direction vectors to the
        appropriate head slices, at the positions selected by `token_scope`. The intervention
        point is after Att and before the output projection Q^h_l. The shared runtime tracks
        position, and the lowest active layer opens the pass.

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

        oproj_names = self._module_names(kwargs.get("model"))
        prompt_lens = compute_prompt_lens(ids, self._pad_token_id)
        self._runtime.reset(prompt_lens)

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        active = sorted(self._active_layer_ids)
        if not active:
            return hooks

        opener = active[0]
        for layer_id in active:
            hooks["pre"].append({
                "module": oproj_names[layer_id],
                "hook_func": self._runtime.build_behavior_hook(
                    layer_id=layer_id,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    is_pass_opener=(layer_id == opener),
                ),
            })

        return hooks
