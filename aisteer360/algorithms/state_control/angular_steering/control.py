"""Angular Steering control: rotational activation steering via composable components."""
from __future__ import annotations

import logging
from functools import partial

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.common.estimators import SteeringPlaneEstimator
from aisteer360.algorithms.state_control.common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control.common.hook_utils import (
    extract_hidden_states,
    get_norm_module_names,
    replace_hidden_states,
)
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.token_scope import compute_prompt_lens, make_token_mask
from aisteer360.algorithms.state_control.common.transforms import (
    AlignmentAdaptiveTransform,
    NormPreservingTransform,
    RotationTransform,
)

from .args import AngularSteeringArgs

logger = logging.getLogger(__name__)


class AngularSteering(StateControl):
    """Angular Steering: rotational activation steering in a learned 2D subspace.

    Steers model behavior by rotating hidden-state activations within a 2D
    plane defined by a feature direction and its orthogonal complement. The
    plane is learned from contrastive prompt pairs via
    ``SteeringPlaneEstimator``, which computes a per-layer mean-difference
    direction and a cross-layer PCA component, then orthogonalizes them
    via Gram-Schmidt.

    Angular Steering operates in two phases:

    1. **Training (offline)**: Given contrastive prompt pairs, estimate
       a steering plane (orthonormal basis pair b1, b2) per layer.

    2. **Inference (online)**: At each target layer, project hidden states
       onto the 2D subspace and apply a rotation. In "target" mode,
       activations are rotated TO a fixed angle from b1; in "offset" mode,
       they are rotated BY a fixed angle from their current position.

    The adaptive variant (AAS) additionally filters tokens by cosine
    similarity with the feature direction, only rotating tokens that
    are aligned above a threshold.

    Hooks are registered on normalization submodules (pre-attention and
    pre-MLP norms) rather than layer outputs, following the original
    Angular Steering paper.

    Reference:

    - "Angular Steering: Behavior Control via Rotation in Activation Space"
    Hieu M. Vu, Tan M. Nguyen
    [https://arxiv.org/pdf/2510.26243v1](https://arxiv.org/pdf/2510.26243v1)
    """

    Args = AngularSteeringArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._gate = AlwaysOpenGate()
        self._norm_modules: list[tuple[int, str]] = []
        self._active_layer_ids: set[int] = set()
        self._pad_token_id: int | None = None

        # tracks cumulative position for KV-cached generation
        self._position_offset: int = 0
        self._initial_seq_len: int = 0

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Initialize Angular Steering by estimating or loading the steering plane.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data.

        Returns:
            The input model, unchanged.
        """
        device = next(model.parameters()).device

        # resolve steering vector (K=2 basis pairs)
        if self.steering_vector is not None:
            sv = self.steering_vector
        else:
            estimator = SteeringPlaneEstimator()
            sv = estimator.fit(
                model, tokenizer,
                data=self.data,
                spec=self.train_spec,
                normalize=self.normalize_directions,
            )

        sv = sv.to(device, dtype=model.dtype)

        # filter to layer_range if specified
        if self.layer_range is not None:
            start, end = self.layer_range
            sv.directions = {
                lid: d for lid, d in sv.directions.items() if start <= lid < end
            }

        self._steering_vector = sv
        self._active_layer_ids = set(sv.directions.keys())

        if not self._active_layer_ids:
            logger.warning("No active layers after filtering — Angular Steering will have no effect")

        # build transform stack
        transform = RotationTransform(sv, angle=self.angle, mode=self.mode)

        if self.adaptive:
            transform = AlignmentAdaptiveTransform(
                transform,
                steering_vector=sv,
                threshold=self.adaptive_threshold,
            )

        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)

        self._transform = transform

        # resolve norm module paths for hooking
        all_norm_modules = get_norm_module_names(model)
        self._norm_modules = [
            (lid, path) for lid, path in all_norm_modules if lid in self._active_layer_ids
        ]

        # store tokenizer info for hook generation
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None

        logger.debug(
            "Angular Steering configured: %d active layers, angle=%.4f, mode=%s, adaptive=%s",
            len(self._active_layer_ids), self.angle, self.mode, self.adaptive,
        )

        return model

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **__,
    ) -> dict[str, list]:
        """Create pre-hooks on normalization submodules for rotational steering.

        Registers pre-forward hooks on both pre-attention and pre-MLP norm
        modules at each active layer. The hook rotates the hidden states
        entering the norm module.

        Args:
            input_ids: Input token IDs.
            runtime_kwargs: Runtime parameters (currently unused).

        Returns:
            Hook specifications with "pre", "forward", "backward" keys.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        prompt_lens = compute_prompt_lens(ids, self._pad_token_id)

        # store initial sequence length for position tracking
        self._initial_seq_len = ids.size(1)
        self._position_offset = 0

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        for layer_id, module_path in self._norm_modules:
            hooks["pre"].append({
                "module": module_path,
                "hook_func": partial(
                    self._pre_hook,
                    layer_id=layer_id,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    prompt_lens=prompt_lens,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    control_ref=self,
                ),
            })

        return hooks

    @staticmethod
    def _pre_hook(
        module,
        input_args,
        input_kwargs,
        *,
        layer_id: int,
        transform,
        gate,
        token_scope: str,
        prompt_lens: torch.LongTensor,
        last_k: int | None,
        from_position: int | None,
        control_ref: "AngularSteering",
    ):
        """Apply rotational steering as a pre-forward hook on norm modules.

        Args:
            module: The norm module being hooked.
            input_args: Positional arguments to the forward pass.
            input_kwargs: Keyword arguments to the forward pass.
            layer_id: Index of the current layer.
            transform: The rotation transform (possibly wrapped in decorators).
            gate: The gate (always open for standard Angular Steering).
            token_scope: Which tokens to steer.
            prompt_lens: Per-batch prompt lengths.
            last_k: Number of last tokens when token_scope is "last_k".
            from_position: Starting position when token_scope is "from_position".
            control_ref: Reference to the control for position tracking.

        Returns:
            Tuple of potentially modified (input_args, input_kwargs).
        """
        hidden = extract_hidden_states(input_args, input_kwargs)
        if hidden is None:
            return input_args, input_kwargs

        seq_len = hidden.size(1)

        # determine position offset for KV-cached generation
        if seq_len < control_ref._initial_seq_len:
            position_offset = control_ref._position_offset
            control_ref._position_offset += seq_len
        else:
            position_offset = 0
            control_ref._position_offset = seq_len

        mask = make_token_mask(
            token_scope,
            seq_len=seq_len,
            prompt_lens=prompt_lens.to(hidden.device),
            last_k=last_k,
            from_position=from_position,
            position_offset=position_offset,
        )

        if gate.is_open():
            hidden = transform.apply(
                hidden, layer_id=layer_id, token_mask=mask
            )

        return replace_hidden_states(input_args, input_kwargs, hidden)

    def reset(self):
        """Reset internal state between generation calls."""
        self._gate.reset()
        self._position_offset = 0
