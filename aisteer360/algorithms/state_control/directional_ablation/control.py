"""Directional Ablation control: projects a learned direction out of the residual stream."""
from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.execution.capabilities import Capability, InterventionKinds
from aisteer360.algorithms.core.execution.interventions import InterventionSpec
from aisteer360.algorithms.core.execution.requirements import Requirements, needs
from aisteer360.algorithms.state_control._common.estimators import (
    ContrastiveDirectionEstimator,
    MeanDifferenceEstimator,
)
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.intervention_export import (
    intervention_generate_requirement,
    intervention_spec_from_runtime_config,
)
from aisteer360.algorithms.state_control._common.layout_facts import layout_torch_dtype, resolve_layout
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import FractionalDepthSelector
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms import (
    DirectionalAblationTransform,
    NormPreservingTransform,
)
from aisteer360.algorithms.state_control.base import StateControl

from .args import DirectionalAblationArgs

logger = logging.getLogger(__name__)


class DirectionalAblation(StateControl):
    """Directional Ablation (feature removal via projection).

    Removes a learned feature direction from the residual stream at one or more layers during
    generation, `h' = h - alpha * (d_hat^T h) d_hat` at masked positions. This is the abliteration
    technique of Arditi et al., which learns a direction as the difference in means over
    contrastive data and projects it out.

    The method operates in two phases:

    1. Training (offline). Extract residual activations for contrastive pairs and take the mean
       difference, or the PCA of paired differences, as the feature direction. A precomputed
       direction (or an orthonormal subspace, `K > 1`) may be supplied directly.

    2. Inference (online). At each target layer's output, project the direction out of the
       residual stream at masked positions. `alpha = 1.0` fully removes the component
       (`h'.d_hat == 0`); `alpha < 1.0` gives graded partial suppression.

    Ablation is a projection (idempotent at `alpha=1`, norm-reducing). It can compose with the
    alignment-adaptive gate (`AlignmentAdaptiveTransform`) to ablate only where the feature is
    present.

    Reference:

    - "Refusal in Language Models Is Mediated by a Single Direction"
    Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
    [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)
    """

    Args = DirectionalAblationArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] | None = None
        self._layer_ids: list[int] = []
        self._num_layers: int | None = None
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime = TransformHookRuntime(hook_point="layer_output")

    def _intervention_kind_plan(self) -> InterventionKinds | None:
        """Kind names this configuration lowers to; None marks it hook-only.

        The wire kind removes a single direction's component in full, so graded removal
        (`alpha < 1.0`) and subspace ablation (`K > 1` directions) have no wire form.
        """
        if self._transform is not None:
            plan = self._transform.wire_kind_plan()
        else:
            if self.alpha != 1.0:
                return None
            source = self._steering_vector if self._steering_vector is not None else self.steering_vector
            if source is not None and source.is_positional:
                return None
            plan = (
                "directional_ablation",
                frozenset({"norm_preserving"}) if self.use_norm_preservation else frozenset(),
            )
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
        hook_only_hint = "subspace ablation has no intervention-spec form; run on the huggingface backend"
        if self.alpha != 1.0:
            hook_only_hint = "graded ablation (alpha < 1) has no intervention-spec form; run on the huggingface backend"
        return Requirements(
            steer=steer,
            generate=intervention_generate_requirement(self._intervention_kind_plan(), hook_only_hint=hook_only_hint),
        )

    def export_intervention_spec(self, runtime_kwargs: dict | None = None) -> InterventionSpec | None:
        """The `directional_ablation` spec over the target layers; None for graded or subspace
        configurations."""
        if self._transform is None or self._num_layers is None:
            return None
        return intervention_spec_from_runtime_config(
            transform=self._transform,
            layer_ids=self._layer_ids,
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
        """Fit or load the feature direction and resolve the layers to ablate.

        Structural facts (layer count, dtype) come from the steering session's layout when a
        session is given; a vector-supplied configuration therefore steers with `model=None`.
        Fitting from `data` requires a live model.

        Args:
            model: The base language model to be steered, or None for vector-supplied
                configurations steered against a session layout.
            tokenizer: Tokenizer for encoding training data (when fitting the direction).
            session: `SteeringSession` on the steering backend, provided by the pipeline.

        Returns:
            The input model, unchanged.

        Raises:
            ValueError: If no target layer has a direction in the steering vector.
        """
        layout = resolve_layout(model, session)
        num_layers = layout.num_layers
        self._num_layers = num_layers
        self._layer_names = get_model_layer_list(model)[1] if model is not None else None

        # resolve the direction (identical to CAA)
        if self.steering_vector is not None:
            source = self.steering_vector
        else:
            if self.train_spec.method == "pca_pairwise":
                estimator = ContrastiveDirectionEstimator()
            else:
                estimator = MeanDifferenceEstimator()
            source = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec, session=session)

        # copy directions into a fresh vector (never mutate a caller-supplied steering_vector in
        # place; a precomputed direction may be reused across controls with different filters)
        dtype = layout_torch_dtype(layout)
        start, end = self.layer_range if self.layer_range is not None else (None, None)
        directions = {
            lid: d.clone().to(dtype=dtype)
            for lid, d in source.directions.items()
            if self.layer_range is None or start <= lid < end
        }
        sv = SteeringVector(
            model_type=source.model_type,
            directions=directions,
            explained_variances=source.explained_variances,
        )
        self._steering_vector = sv

        # resolve target layers
        if self.layer_ids is not None:
            target_ids = sorted(set(self.layer_ids))
        else:
            # heuristic: single layer at ~40% depth (matches CAA)
            target_ids = [FractionalDepthSelector(fraction=0.4).select(num_layers=num_layers)]

        self._layer_ids = [lid for lid in target_ids if lid in sv.directions]
        if not self._layer_ids:
            raise ValueError(
                f"No target layer has a direction in the steering vector "
                f"(requested {target_ids}, available {sorted(sv.directions.keys())})."
            )

        # build the transform
        transform = DirectionalAblationTransform(sv.directions, alpha=self.alpha)
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
                    "DirectionalAblation was steered without a live model, so hook module names are "
                    "unresolved; pass `model=` to get_hooks (the pipeline does) or steer with a model."
                )
            _, self._layer_names = get_model_layer_list(source)
        return self._layer_names

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Create a forward hook on each target layer's output to ablate the residual stream.

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

        # the lowest hooked layer opens the pass and advances the shared KV offset once per forward pass
        opener = min(self._layer_ids) if self._layer_ids else None

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        for layer_id in self._layer_ids:
            hooks["forward"].append({
                "module": layer_names[layer_id],
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
