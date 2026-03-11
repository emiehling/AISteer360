"""Rotational activation steering transform."""
import math
from typing import Literal

import torch

from .base import BaseTransform


class RotationTransform(BaseTransform):
    """Rotates activations within a 2D subspace defined by a basis pair.

    For each layer, decomposes hidden states into in-plane and orthogonal
    components, applies a 2D rotation to the in-plane component, and
    reconstructs. Preserves activation norms by construction (rotation is
    an orthogonal transformation).

    Two modes are supported:

    Target mode:
        Rotates each activation TO angle ``theta`` measured from b1.
        The current angle is ``atan2(c2, c1)`` and the rotation applied
        is ``theta - current_angle``.

    Offset mode:
        Rotates each activation BY angle ``theta`` from its current
        position in the plane. The rotation is the same for all tokens.

    Args:
        steering_vector: SteeringVector with K=2 per layer (basis pair).
            Row 0 is b1 (feature direction), row 1 is b2 (orthogonal axis).
        angle: Rotation parameter in radians.
        mode: "target" rotates each activation TO angle theta measured from b1.
            "offset" rotates each activation BY angle theta from its current position.

    Reference:

        - "Angular Steering: Improving LLM Alignment with Simple Activation Rotations"
          Tuan Vu, Thang Nguyen
          [https://arxiv.org/abs/2504.02406](https://arxiv.org/abs/2504.02406)
    """

    def __init__(
        self,
        steering_vector: "SteeringVector",
        angle: float,
        mode: Literal["target", "offset"] = "target",
    ):
        for layer_id, dirs in steering_vector.directions.items():
            if dirs.shape[0] != 2:
                raise ValueError(
                    f"RotationTransform expects K=2 (basis pair), "
                    f"got K={dirs.shape[0]} at layer {layer_id}"
                )
        self.steering_vector = steering_vector
        self.angle = angle
        self.mode = mode

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply rotational steering in the 2D subspace.

        Args:
            hidden_states: Shape [B, T, H].
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.
            **kwargs: Ignored.

        Returns:
            Modified hidden states with rotated activations at masked positions.
        """
        basis = self.steering_vector.directions.get(layer_id)
        if basis is None:
            return hidden_states

        b1 = basis[0].to(hidden_states.device, hidden_states.dtype)  # [H]
        b2 = basis[1].to(hidden_states.device, hidden_states.dtype)  # [H]

        # project onto the 2D subspace: c1 = h · b1, c2 = h · b2
        c1 = hidden_states @ b1  # [B, T]
        c2 = hidden_states @ b2  # [B, T]

        # compute rotation delta per position
        if self.mode == "target":
            current_angle = torch.atan2(c2, c1)  # [B, T]
            delta = self.angle - current_angle
            cos_d = torch.cos(delta)
            sin_d = torch.sin(delta)
        else:
            # offset mode: constant rotation applied uniformly
            cos_d = math.cos(self.angle)
            sin_d = math.sin(self.angle)

        # rotate in-plane coordinates
        c1_new = cos_d * c1 - sin_d * c2
        c2_new = sin_d * c1 + cos_d * c2

        # reconstruct: h' = h_perp + c1_new * b1 + c2_new * b2
        # where h_perp = h - c1 * b1 - c2 * b2
        h_rotated = (
            hidden_states
            - c1.unsqueeze(-1) * b1
            - c2.unsqueeze(-1) * b2
            + c1_new.unsqueeze(-1) * b1
            + c2_new.unsqueeze(-1) * b2
        )

        # apply token mask: only modify masked positions
        mask = token_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states * (1 - mask) + h_rotated * mask
