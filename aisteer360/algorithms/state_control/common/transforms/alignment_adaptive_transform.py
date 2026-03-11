"""Alignment-based adaptive filtering transform decorator."""
import torch
import torch.nn.functional as F

from .base import BaseTransform


class AlignmentAdaptiveTransform(BaseTransform):
    """Wraps any transform to only apply at tokens aligned with a feature direction.

    Computes per-token cosine similarity between hidden states and a direction
    from the steering vector. Positions below the threshold are masked out
    before the inner transform is applied, leaving those tokens unchanged.

    This is the decorator that implements Adaptive Angular Steering (AAS) when
    wrapping a ``RotationTransform``, but it composes with any ``BaseTransform``
    to produce adaptive variants of other methods (e.g., adaptive-CAA,
    adaptive-ITI).

    Args:
        inner: The transform to wrap.
        steering_vector: SteeringVector containing the alignment direction.
        threshold: Minimum cosine similarity for a token to be modified.
        direction_idx: Which row of the [K, D] direction tensor to use as
            the alignment direction (default: 0, i.e., the feature direction).

    Reference:

        - "Angular Steering: Improving LLM Alignment with Simple Activation Rotations"
          Tuan Vu, Thang Nguyen
          [https://arxiv.org/abs/2504.02406](https://arxiv.org/abs/2504.02406)
    """

    def __init__(
        self,
        inner: BaseTransform,
        steering_vector: "SteeringVector",
        threshold: float = 0.0,
        direction_idx: int = 0,
    ):
        self.inner = inner
        self.steering_vector = steering_vector
        self.threshold = threshold
        self.direction_idx = direction_idx

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply inner transform only at tokens aligned with the feature direction.

        Args:
            hidden_states: Shape [B, T, H].
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.
            **kwargs: Passed through to inner transform.

        Returns:
            Modified hidden states, same shape as input.
        """
        dirs = self.steering_vector.directions.get(layer_id)
        if dirs is not None:
            direction = dirs[self.direction_idx].to(hidden_states.device, hidden_states.dtype)  # [D]

            # cosine similarity between each token and the alignment direction
            sim = F.cosine_similarity(
                hidden_states,
                direction.unsqueeze(0).unsqueeze(0),  # [1, 1, D]
                dim=-1,
            )  # [B, T]

            # narrow the mask: only keep tokens above the threshold
            token_mask = token_mask & (sim > self.threshold)

        return self.inner.apply(
            hidden_states, layer_id=layer_id, token_mask=token_mask, **kwargs
        )
