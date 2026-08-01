"""Angular Steering control: rotational activation steering in a learned 2D subspace."""
from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control._common.estimators import SteeringPlaneEstimator
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import get_norm_module_names
from aisteer360.algorithms.state_control._common.layout_facts import layout_torch_dtype, resolve_layout
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms import (
    AlignmentAdaptiveTransform,
    NormPreservingTransform,
    RotationTransform,
)
from aisteer360.algorithms.state_control.base import StateControl

from .args import AngularSteeringArgs

logger = logging.getLogger(__name__)


class AngularSteering(StateControl):
    """Angular Steering.

    Rotates the hidden state within a per-layer 2D plane spanned by a feature axis (row 0 of the
    steering vector) and a companion axis (row 1), leaving the orthogonal complement (the other
    `d_model - 2` directions) untouched. Because a 2D rotation is orthogonal, the intervention is
    norm-preserving by construction and offers continuous control via a single angle.

    The method operates in two phases:

    1. **Offline plane fitting**: A per-layer feature axis is estimated via difference-in-means
       over contrastive data, and a single global companion axis is taken as the first principal
       component across the stacked per-layer feature directions. Gram-Schmidt yields an
       orthonormal `(b1, b2)` per layer. A precomputed `[2, H]`-per-layer plane may be supplied
       directly instead.

    2. **Online rotation**: A `forward_pre_hook` on each layer's normalization sub-modules
       (`input_layernorm` and `post_attention_layernorm`, or `ln_1`/`ln_2` on GPT-2) rotates the
       residual stream entering the norm to the target angle (`mode="target"`) or by the angle
       (`mode="offset"`). Vector addition and directional ablation are special cases of this
       rotation. The adaptive variant rotates only tokens already positively aligned with the
       feature axis, improving coherence on smaller models.

    Each norm module is rotated exactly once, keyed to its own layer's plane. The shared runtime
    tracks position bookkeeping (the KV-cache offset shared across all hooked norms) and opens each
    forward pass on the first-firing norm module (opener convention).

    Reference:

    - "Angular Steering: Behavior Control via Rotation in Activation Space"
    Hieu M. Vu, Tan M. Nguyen
    [https://arxiv.org/abs/2510.26243](https://arxiv.org/abs/2510.26243)
    """

    Args = AngularSteeringArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._gate = AlwaysOpenGate()
        self._norm_modules: list[tuple[int, str]] | None = None
        self._pad_token_id: int | None = None
        self._runtime = TransformHookRuntime(hook_point="layer_input")

    def steer(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        session=None,
        **__,
    ) -> PreTrainedModel | None:
        """Fit or load the steering plane and locate the norm modules to hook.

        Structural facts (dtype) come from the steering session's layout when a session is given;
        a vector-supplied configuration therefore steers with `model=None`. Fitting from `data`
        requires a live model.

        Args:
            model: The base language model to be steered, or None for vector-supplied
                configurations steered against a session layout.
            tokenizer: Tokenizer for encoding training data (when fitting the plane).
            session: `SteeringSession` on the steering backend, provided by the pipeline.

        Returns:
            The input model, unchanged.

        Raises:
            ValueError: If no layers remain after `layer_range` filtering, or if no normalization
                sub-modules can be located for the active layers.
        """
        layout = resolve_layout(model, session)

        # resolve the plane
        if self.steering_vector is not None:
            source = self.steering_vector
        else:
            if model is None:
                raise ValueError("Fitting AngularSteering from data requires a live model at steer time.")
            source = SteeringPlaneEstimator().fit(model, tokenizer, data=self.data, spec=self.train_spec)

        # copy directions into a fresh vector (never mutate a caller-supplied steering_vector in
        # place; a precomputed plane may be reused across controls with different layer_range)
        dtype = layout_torch_dtype(layout)
        start, end = self.layer_range if self.layer_range is not None else (None, None)
        directions = {
            lid: d.clone().to(dtype=dtype)
            for lid, d in source.directions.items()
            if self.layer_range is None or start <= lid < end
        }
        if not directions:
            raise ValueError("No active layers for angular steering after filtering.")

        sv = SteeringVector(
            model_type=source.model_type,
            directions=directions,
            explained_variances=source.explained_variances,
        )
        self._steering_vector = sv

        active_layer_ids = set(sv.directions.keys())

        # build the transform stack
        transform = RotationTransform(sv, angle=self.angle_radians, mode=self.mode)
        if self.adaptive:
            transform = AlignmentAdaptiveTransform(
                transform,
                sv,
                threshold=self.adaptive_threshold,
                use_cosine=self.adaptive_use_cosine,
            )
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)
        self._transform = transform

        # locate the normalization sub-modules to hook (only for active layers)
        self._norm_modules = self._locate_norm_modules(model) if model is not None else None

        # store tokenizer info for hook generation
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None

        return model

    def _locate_norm_modules(self, model) -> list[tuple[int, str]]:
        """The `(layer_id, module_path)` pairs to hook, restricted to active layers.

        Raises:
            ValueError: If no normalization sub-modules can be located for the active layers.
        """
        active_layer_ids = set(self._steering_vector.directions.keys())
        norm_modules = [
            (lid, path) for lid, path in get_norm_module_names(model) if lid in active_layer_ids
        ]
        if not norm_modules:
            raise ValueError("Could not locate any normalization sub-modules to hook.")
        return norm_modules

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Create pre-hooks that rotate the residual stream entering each norm module.

        The shared runtime tracks position bookkeeping. Two norm modules share each `layer_id`, so
        the pass opener is keyed on `module_path` (the first-firing norm module) rather than
        `layer_id`. The pre-attention norm sorts and fires first on both supported families.

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

        if self._norm_modules is None:
            source = kwargs.get("model") if kwargs.get("model") is not None else self._model_ref
            if source is None:
                raise RuntimeError(
                    "AngularSteering was steered without a live model, so hook module names are "
                    "unresolved; pass `model=` to get_hooks (the pipeline does) or steer with a model."
                )
            self._norm_modules = self._locate_norm_modules(source)

        prompt_lens = compute_prompt_lens(ids, self._pad_token_id)
        self._runtime.reset(prompt_lens)

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        opener_path = self._norm_modules[0][1] if self._norm_modules else None
        for layer_id, module_path in self._norm_modules:
            hooks["pre"].append({
                "module": module_path,
                "hook_func": self._runtime.build_behavior_hook(
                    layer_id=layer_id,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    is_pass_opener=(module_path == opener_path),
                ),
            })
        return hooks
