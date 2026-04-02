"""Angular Steering control implementation."""
from __future__ import annotations

from functools import partial

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.common.estimators import AngularDirectionEstimator
from aisteer360.algorithms.state_control.common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control.common.hook_utils import (
    LAYERNORM_INPUT_MODULE,
    LAYERNORM_MID_MODULES,
    get_model_layer_list,
)
from aisteer360.algorithms.state_control.common.selectors import FixedLayerSelector
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.token_scope import compute_prompt_lens, make_token_mask
from aisteer360.algorithms.state_control.common.transforms import AngularTransform

from .args import AngularSteeringArgs



class AngularSteering(StateControl):
    """Angular Steering.

    Steers model behavior by rotating the hidden-state component that lies
    within a learned 2D steering plane, while preserving the out-of-plane
    component exactly.

    The method operates in two phases:

    1. **Training (offline)**: Given contrastive prompt pairs, extract
       activations at the last token position.  Per-sample L2-normalize,
       compute a per-layer candidate direction (normalized mean difference),
       then fit PCA across all layer candidates to obtain a second basis
       vector.  The best candidate (via ``layer_selection``) becomes the
       global first direction shared by all layers.

    2. **Inference (online)**: Forward hooks project the hidden state onto
       the (b1, b2) steering plane and replace the in-plane component with
       a rotation to ``target_degree``:

           ``h' = h − P·h + ‖P·h‖ · v_θ``

       Hooks are registered on each target layer's
       ``post_attention_layernorm`` **and** the next layer's
       ``input_layernorm``, matching the reference implementation.
       When ``layer_id`` is ``None`` (default), all layers are steered.
       When ``layer_id`` is set, only that single layer's layernorms
       are hooked (useful for ablation).

       The rotation formula is inherently norm-preserving: the in-plane
       magnitude ``‖P·h‖`` is held constant and the out-of-plane component
       is untouched.

       With ``adaptive_mode=1``, only positions whose original activation
       has a positive projection onto b1 are steered.

    Common angles:
        - ``0°``  : aligned with the first direction (e.g., harmful)
        - ``180°``: maximally opposite (e.g., maximum refusal)
        - ``90°`` / ``270°``: perpendicular (neutral)

    Reference:

    - "Angular Steering: Behavior Control via Rotation in Activation Space"
      Hieu M. Vu, Tan M. Nguyen
      `<https://arxiv.org/abs/2510.26243>`_
    """

    Args = AngularSteeringArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._steering_vector: SteeringVector | None = None
        self._transform: AngularTransform | None = None
        self._hook_targets: list[tuple[str, int]] = []
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None

        # position tracking for KV-cached generation
        self._position_offset: int = 0
        self._initial_seq_len: int = 0

    # ------------------------------------------------------------------
    # Steering (offline)
    # ------------------------------------------------------------------

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Learn or load the basis pair and build the transform.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data.

        Returns:
            The input model, unchanged.
        """
        device = next(model.parameters()).device
        _, layer_names = get_model_layer_list(model)
        num_layers = len(layer_names)

        # --- resolve steering vector ---
        if self.steering_vector is not None:
            sv = self.steering_vector
        else:
            estimator = AngularDirectionEstimator(strategy=self.layer_selection)
            sv = estimator.fit(
                model,
                tokenizer,
                data=self.data,
                spec=self.train_spec,
            )

        sv = sv.to(device, dtype=model.dtype)
        self._steering_vector = sv

        # --- resolve target layers ---
        if self.layer_id is not None:
            selector = FixedLayerSelector(self.layer_id)
            target_layers = [selector.select(num_layers=num_layers)]
        else:
            target_layers = sorted(
                lid for lid in sv.directions if 0 <= lid < num_layers
            )

        if not target_layers:
            raise ValueError("No valid layer IDs found in steering vector.")

        for lid in target_layers:
            if lid not in sv.directions:
                raise ValueError(
                    f"Steering vector has no direction for layer {lid}."
                )

        # --- discover hook targets ---
        # Hook both the mid-layer layernorm and the next layer's
        # input_layernorm for each target layer, matching the reference.
        module_names = set(dict(model.named_modules()).keys())
        self._hook_targets = []

        for lid in target_layers:
            layer_prefix = layer_names[lid]

            # 1) post_attention_layernorm (or pre_feedforward_layernorm)
            for suffix in LAYERNORM_MID_MODULES:
                path = f"{layer_prefix}.{suffix}"
                if path in module_names:
                    self._hook_targets.append((path, lid))
                    break

            # 2) input_layernorm of the NEXT layer — it operates on the
            #    residual stream after layer `lid`, so uses lid's direction.
            next_lid = lid + 1
            if next_lid < num_layers:
                next_prefix = layer_names[next_lid]
                path = f"{next_prefix}.{LAYERNORM_INPUT_MODULE}"
                if path in module_names:
                    self._hook_targets.append((path, lid))

        if not self._hook_targets:
            raise ValueError(
                "Could not find any hook targets. "
                "Ensure the model has post_attention_layernorm or "
                "pre_feedforward_layernorm modules."
            )

        # --- build transform ---
        self._transform = AngularTransform(
            directions=sv.directions,
            target_degree=self.target_degree,
            adaptive_mode=self.adaptive_mode,
        )

        self._pad_token_id = (
            getattr(tokenizer, "pad_token_id", None) if tokenizer else None
        )

        return model

    # ------------------------------------------------------------------
    # Hook generation (online)
    # ------------------------------------------------------------------

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None = None,
        **__,
    ) -> dict[str, list]:
        """Register forward hooks on layernorm sub-modules.

        Args:
            input_ids: Input token IDs.
            runtime_kwargs: Unused.

        Returns:
            Hook specifications with "pre", "forward", "backward" keys.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        prompt_lens = compute_prompt_lens(ids, self._pad_token_id)

        self._initial_seq_len = ids.size(1)
        self._position_offset = 0

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        last_idx = len(self._hook_targets) - 1
        for i, (module_path, layer_id) in enumerate(self._hook_targets):
            hooks["forward"].append({
                "module": module_path,
                "hook_func": partial(
                    self._forward_hook,
                    layer_id=layer_id,
                    is_last_hook=(i == last_idx),
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

    # ------------------------------------------------------------------
    # Hook implementation
    # ------------------------------------------------------------------

    def _forward_hook(
        self,
        module,
        args,
        kwargs,
        output,
        *,
        layer_id: int,
        is_last_hook: bool,
        transform,
        gate,
        token_scope: str,
        prompt_lens: torch.LongTensor,
        last_k: int | None,
        from_position: int | None,
        control_ref: "AngularSteering",
    ):
        """Apply angular steering to the hooked module's output.

        Args:
            module: The hooked module.
            args: Positional arguments passed to the forward call.
            kwargs: Keyword arguments passed to the forward call.
            output: The module's output (hidden_states, ...) or just hidden_states.
            layer_id: Index of the target layer (used to look up directions).
            is_last_hook: Whether this is the last hook in the forward pass,
                responsible for advancing the position offset.
            transform: The angular transform to apply.
            gate: The gate (always open for angular steering).
            token_scope: Which tokens to steer.
            prompt_lens: Per-batch prompt lengths.
            last_k: Number of last tokens when token_scope is "last_k".
            from_position: Starting position when token_scope is "from_position".
            control_ref: Reference to this control for position tracking.

        Returns:
            Modified output with angular rotation applied to hidden states.
        """
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if hidden is None:
            return output

        seq_len = hidden.size(1)

        # Position offset for KV-cached generation.
        # All hooks in the same forward pass share the offset; only the
        # last hook advances it so multi-layer hooks read the same value.
        if seq_len < control_ref._initial_seq_len:
            position_offset = control_ref._position_offset
            if is_last_hook:
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
                hidden,
                layer_id=layer_id,
                token_mask=mask,
            )

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def reset(self):
        """Reset internal state between generation calls."""
        self._gate.reset()
        self._position_offset = 0
