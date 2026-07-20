"""Target-projection transform: set the component along a direction to a target scalar."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import torch

from ..sources import ArtifactSource
from ..steering_vector import SteeringVector
from .base import BaseTransform

if TYPE_CHECKING:
    from .context import TransformContext


class TargetProjectionTransform(BaseTransform):
    """Sets each masked hidden state's component along a per-layer direction to a target scalar.

    For a unit direction `v̂` at a layer and a target scalar `t`:

        h' = h + (t − h·v̂) · v̂          (applied at masked positions)

    so the post-edit projection onto `v̂` equals `t` exactly, leaving the orthogonal complement
    unchanged. This gives an HF-side counterpart to vLLM-Hook's language-steering use case (the
    former "projection adjustment"): drive the activation's coordinate along a learned axis to a
    fixed value rather than adding a fixed increment.

    Directions are normalized to unit length internally (cached per `(layer_id, device, dtype)`).
    Targets default to `0.0` for any layer absent from `targets` (i.e. ablate the component).

    Args:
        artifact: The direction artifact — a `SteeringVector`, a per-layer directions mapping
            (`Mapping[int, Tensor]`, each `[H]` or `[1, H]`), or an `ArtifactSource` (unbound until
            `bind(ctx)`). Required.
        targets: Per-layer target scalar for the projection onto that layer's direction. Layers
            absent from the mapping use `0.0`.
    """

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor] | ArtifactSource,
        targets: Mapping[int, float] | None = None,
    ):
        self.targets = {int(k): float(v) for k, v in (targets or {}).items()}
        self._source: ArtifactSource | None = None
        self.directions: dict[int, torch.Tensor] | None = None
        self._unit_cache: dict[tuple, torch.Tensor] = {}  # (layer_id, device, dtype) -> [H] unit vector

        if isinstance(artifact, ArtifactSource):
            self._source = artifact
        elif isinstance(artifact, SteeringVector):
            self.directions = artifact.directions
        elif isinstance(artifact, Mapping):
            self.directions = dict(artifact)
        else:
            raise TypeError(
                f"TargetProjectionTransform artifact must be a SteeringVector, a Mapping[int, Tensor], "
                f"or an ArtifactSource; got {type(artifact).__name__}."
            )

    @property
    def is_bound(self) -> bool:
        return self.directions is not None

    def bind(self, ctx: "TransformContext") -> "TargetProjectionTransform":
        if self.is_bound:
            return self
        return TargetProjectionTransform(ctx.resolve(self._source), targets=self.targets)

    @property
    def covered_layer_ids(self) -> set[int] | None:
        return set(self.directions.keys()) if self.directions is not None else None

    def export_payload(self) -> dict | None:
        """Export as the wire `target_projection` kind: per-layer directions plus target scalars."""
        if self.directions is None:
            return None
        from ..intervention import ArtifactHandle

        return {
            "kind": "target_projection",
            "targets": {int(lid): float(self.targets.get(lid, 0.0)) for lid in self.directions},
            "vectors": {int(lid): ArtifactHandle(tensor, role="direction") for lid, tensor in self.directions.items()},
        }

    def _unit(self, layer_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        """Return the cached unit direction `[H]` for a layer, or None when absent/degenerate."""
        key = (layer_id, device, dtype)
        cached = self._unit_cache.get(key)
        if cached is None:
            raw = self.directions.get(layer_id)
            if raw is None:
                return None
            vector = raw.to(device=device, dtype=dtype)
            if vector.ndim == 2:
                vector = vector[0]  # [K, H] -> feature row
            norm = vector.norm()
            if norm <= 1e-8:
                return None
            cached = vector / norm
            self._unit_cache[key] = cached
        return cached

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Set the projection onto the layer direction to the target scalar at masked positions.

        Args:
            hidden_states: Shape `[B, T, H]`.
            layer_id: Which layer this is being applied at.
            token_mask: Shape `[B, T]`. True at positions to modify.
            **kwargs: Ignored.

        Returns:
            Modified hidden states, same shape as input.
        """
        self._require_bound()
        unit = self._unit(layer_id, hidden_states.device, hidden_states.dtype)
        if unit is None:
            return hidden_states

        target = self.targets.get(layer_id, 0.0)
        projection = torch.einsum("bth,h->bt", hidden_states, unit)  # [B, T]
        delta_coeff = (target - projection).unsqueeze(-1)            # [B, T, 1]
        delta = delta_coeff * unit.view(1, 1, -1)                    # [B, T, H]
        delta = delta * token_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states + delta
