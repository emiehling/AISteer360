"""Head-level additive transform for activation steering."""
import torch

from .base import BaseTransform


class HeadAdditiveTransform(BaseTransform):
    """Adds scaled direction vectors to specific head slices.

    For each selected (layer, head) pair, it adds a direction vector to the slice
    [h * head_dim : (h+1) * head_dim].

    For ITI, this operates in pre-o_proj space: the input to the output projection
    where each head_dim-sized slice corresponds to an individual attention head's
    output. The directions must be computed in the same space.

    Args:
        head_directions: Nested mapping layer_id -> {head_id -> direction[head_dim]}.
        num_heads: Number of attention heads per layer.
        head_dim: Dimension of each head's output slice.
        strength: Global scaling factor (alpha in ITI).
    """

    def __init__(
        self,
        head_directions: dict[int, dict[int, torch.Tensor]],
        num_heads: int,
        head_dim: int,
        strength: float = 1.0,
    ):
        self.head_directions = head_directions
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.strength = strength

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply head-level additive steering.

        Args:
            hidden_states: Shape [B, T, H] where H = num_heads * head_dim.
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.
            **kwargs: Ignored.

        Returns:
            Modified hidden states, same shape as input.
        """
        heads = self.head_directions.get(layer_id)
        if not heads:
            return hidden_states

        # EM: clone to be safe (if any references the original tensor are made)
        hidden_states = hidden_states.clone()

        for head_id, direction in heads.items():
            start = head_id * self.head_dim
            end = start + self.head_dim
            v = (self.strength * direction).to(dtype=hidden_states.dtype, device=hidden_states.device)
            # scale by token mask so unmasked positions are untouched
            delta = token_mask.unsqueeze(-1).to(hidden_states.dtype) * v.view(1, 1, -1)
            hidden_states[:, :, start:end] = hidden_states[:, :, start:end] + delta

        return hidden_states
