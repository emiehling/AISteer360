"""Head-level additive transform for activation steering."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import torch

from ..sources import ArtifactSource
from ..steering_vector import SteeringVector
from .base import BaseTransform

if TYPE_CHECKING:
    from .context import TransformContext


class HeadAdditiveTransform(BaseTransform):
    """Adds scaled direction vectors to specific head slices.

    For each selected (layer, head) pair, it adds a direction vector to the slice
    [h * head_dim : (h+1) * head_dim].

    For ITI, this operates in pre-o_proj space: the input to the output projection
    where each head_dim-sized slice corresponds to an individual attention head's
    output. The directions must be computed in the same space.

    Expects a SteeringVector whose directions are shaped [num_heads, head_dim]
    per layer, with num_heads and head_dim metadata set. Only head indices
    present in ``active_heads`` are applied; other heads are left untouched.

    A bare directions mapping cannot carry the required `num_heads`/`head_dim` metadata and is
    rejected; supply a `SteeringVector` (or an `ArtifactSource` resolving to one).

    Args:
        artifact: A `SteeringVector` with per-head directions and `num_heads`/`head_dim` metadata,
            or an `ArtifactSource` (unbound until `bind(ctx)`). Required.
        active_heads: Mapping from layer_id to set of head indices to intervene on.
        strength: Global scaling factor (alpha in ITI).
    """

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor] | ArtifactSource,
        active_heads: dict[int, set[int]],
        strength: float = 1.0,
    ):
        self.active_heads = active_heads
        self.strength = strength
        self._source: ArtifactSource | None = None
        self.steering_vector: SteeringVector | None = None
        self.num_heads: int | None = None
        self.head_dim: int | None = None
        self._folded_residual: dict[int, torch.Tensor] | None = None

        if isinstance(artifact, ArtifactSource):
            self._source = artifact
        elif isinstance(artifact, SteeringVector):
            self.steering_vector = artifact
            self._validate_artifact()
        elif isinstance(artifact, Mapping):
            raise ValueError(
                "HeadAdditiveTransform requires num_heads and head_dim metadata on the SteeringVector; "
                "a bare directions mapping cannot carry it."
            )
        else:
            raise TypeError(
                f"HeadAdditiveTransform artifact must be a SteeringVector or an ArtifactSource; got "
                f"{type(artifact).__name__} (did you mean strength=?)."
            )

    def _validate_artifact(self) -> None:
        """Validate the artifact carries num_heads/head_dim metadata."""
        if self.steering_vector.num_heads is None or self.steering_vector.head_dim is None:
            raise ValueError("HeadAdditiveTransform requires num_heads and head_dim metadata on the SteeringVector.")
        self.num_heads = self.steering_vector.num_heads
        self.head_dim = self.steering_vector.head_dim

    @property
    def is_bound(self) -> bool:
        return self.steering_vector is not None

    def bind(self, ctx: "TransformContext") -> "HeadAdditiveTransform":
        if self.is_bound:
            return self
        return HeadAdditiveTransform(ctx.resolve(self._source), active_heads=self.active_heads, strength=self.strength)

    @property
    def covered_layer_ids(self) -> set[int] | None:
        return set(self.steering_vector.directions.keys()) if self.steering_vector is not None else None

    def fold_to_residual(self, oproj_weights: Mapping[int, torch.Tensor]) -> None:
        """Fold the head-space additions through `o_proj` into residual-space `add` vectors.

        For each active layer, build the pre-`o_proj` delta `x` whose head slices hold
        `strength * direction[h]` for the active heads (zero elsewhere), then compute the
        residual-space vector `W_o @ x`. Because the addition is a per-token delta, the `o_proj`
        bias cancels: `o_proj(h + x) - o_proj(h) = W_o x`. Exact for a linear `o_proj`; a quantized
        projection is only approximate (the wire compiler marks it `degraded`).

        Args:
            oproj_weights: Mapping from active layer id to its `o_proj` weight `[H_out, num_heads*head_dim]`.
        """
        self._require_bound()
        folded: dict[int, torch.Tensor] = {}
        for layer_id, heads in self.active_heads.items():
            weight = oproj_weights.get(layer_id)
            dirs = self.steering_vector.directions.get(layer_id)
            if weight is None or dirs is None or not heads:
                continue
            in_features = self.num_heads * self.head_dim
            x = dirs.new_zeros(in_features)
            for head_id in heads:
                start = head_id * self.head_dim
                x[start:start + self.head_dim] = self.strength * dirs[head_id]
            # W_o @ x  ->  [H_out]; weight is [H_out, in_features]
            folded[layer_id] = (weight.to(dtype=x.dtype, device=x.device) @ x).detach()
        self._folded_residual = folded

    def export_payload(self) -> dict | None:
        """Export the folded residual `add` vectors, or `None` if folding has not run.

        `fold_to_residual` must be called at `steer()` time; without it the head-space transform has
        no residual-space equivalent and is treated as non-portable.
        """
        if self._folded_residual is None:
            return None
        from ..intervention import ArtifactHandle

        return {
            "kind": "add",
            "scale": 1.0,
            "vectors": {int(lid): ArtifactHandle(vector, role="direction") for lid, vector in self._folded_residual.items()},
        }

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
        self._require_bound()
        heads = self.active_heads.get(layer_id)
        if not heads:
            return hidden_states

        dirs = self.steering_vector.directions.get(layer_id)
        if dirs is None:
            return hidden_states

        hidden_states = hidden_states.clone()

        for head_id in heads:
            start = head_id * self.head_dim
            end = start + self.head_dim
            direction = dirs[head_id]  # [head_dim]
            v = (self.strength * direction).to(dtype=hidden_states.dtype, device=hidden_states.device)
            # scale by token mask so unmasked positions are untouched
            delta = token_mask.unsqueeze(-1).to(hidden_states.dtype) * v.view(1, 1, -1)
            hidden_states[:, :, start:end] = hidden_states[:, :, start:end] + delta

        return hidden_states
