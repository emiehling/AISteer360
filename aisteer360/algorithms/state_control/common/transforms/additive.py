"""Additive activation steering transform."""
import torch

from .base import TransformBase


class AdditiveTransform(TransformBase):
    """Adds a scaled direction vector to hidden states.

    h' = h + mask * strength * direction[layer_id]

    Args:
        directions: Per-layer direction tensors. Typically from a
            SteeringVector.directions dict.
        strength: Global scaling factor.
    """

    def __init__(self, directions: dict[int, torch.Tensor], strength: float = 1.0):
        self.directions = directions
        self.strength = strength

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply additive steering.

        Args:
            hidden_states: Shape [B, T, H].
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.
            **kwargs: Ignored.

        Returns:
            Modified hidden states, same shape as input.
        """
        direction = self.directions.get(layer_id)
        if direction is None:
            return hidden_states

        v = (self.strength * direction).to(dtype=hidden_states.dtype, device=hidden_states.device)
        delta = token_mask.unsqueeze(-1).to(hidden_states.dtype) * v.view(1, 1, -1)
        return hidden_states + delta
