"""Angular Steering argument validation."""
from dataclasses import dataclass, field
from typing import Literal

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.state_control.common.specs import (
    ContrastivePairs,
    VectorTrainSpec,
    as_contrastive_pairs,
)
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.token_scope import TokenScope


@dataclass
class AngularSteeringArgs(BaseArgs):
    """Arguments for Angular Steering.

    Users provide EITHER a pre-computed steering vector (with K=2 basis
    pairs per layer) OR contrastive training data.

    Attributes:
        steering_vector: Pre-computed steering vector with K=2 per layer.
            If provided, skip estimation.
        data: Contrastive pairs for estimating the steering plane.
            Required if steering_vector is None.
        train_spec: Controls extraction method and accumulation mode for
            the underlying mean-difference estimation.
        angle: Rotation parameter in radians.
        mode: "target" rotates each activation TO angle theta measured from
            b1. "offset" rotates each activation BY angle theta from its
            current position.
        adaptive: If True, wrap the rotation in an alignment-adaptive filter
            that only modifies tokens whose activations are aligned with the
            feature direction above ``adaptive_threshold``.
        adaptive_threshold: Minimum cosine similarity for a token to be
            modified. Only used when adaptive is True.
        layer_range: Tuple of (start, end) layer indices (inclusive start,
            exclusive end) to restrict steering to. None means all layers
            present in the steering vector.
        token_scope: Which tokens to steer.
        last_k: Required when token_scope == "last_k".
        from_position: Required when token_scope == "from_position".
        normalize_directions: If True, L2-normalize the feature directions
            before constructing the steering plane.
        use_norm_preservation: If True, wrap transform in NormPreservingTransform.
    """

    # steering vector source (provide exactly one)
    steering_vector: SteeringVector | None = None
    data: ContrastivePairs | dict | None = None

    # training configuration
    train_spec: VectorTrainSpec | dict = field(
        default_factory=lambda: VectorTrainSpec(method="mean_diff", accumulate="last_token")
    )

    # rotation configuration
    angle: float = 0.0
    mode: Literal["target", "offset"] = "target"

    # adaptive filtering
    adaptive: bool = False
    adaptive_threshold: float = 0.0

    # layer selection
    layer_range: tuple[int, int] | None = None

    # inference configuration
    token_scope: TokenScope = "all"
    last_k: int | None = None
    from_position: int | None = None
    normalize_directions: bool = True
    use_norm_preservation: bool = False

    def __post_init__(self):
        # exactly one of steering_vector or data must be provided
        if self.steering_vector is None and self.data is None:
            raise ValueError("Provide either steering_vector or data.")
        if self.steering_vector is not None and self.data is not None:
            raise ValueError("Provide steering_vector or data, not both.")

        # validate steering_vector if provided
        if self.steering_vector is not None:
            self.steering_vector.validate()
            for layer_id, dirs in self.steering_vector.directions.items():
                if dirs.shape[0] != 2:
                    raise ValueError(
                        f"Angular Steering requires K=2 (basis pair) per layer, "
                        f"got K={dirs.shape[0]} at layer {layer_id}."
                    )

        # normalize dict inputs
        if self.data is not None and not isinstance(self.data, ContrastivePairs):
            object.__setattr__(self, "data", as_contrastive_pairs(self.data))

        if isinstance(self.train_spec, dict):
            object.__setattr__(self, "train_spec", VectorTrainSpec(**self.train_spec))

        # validate layer_range
        if self.layer_range is not None:
            start, end = self.layer_range
            if start < 0 or end <= start:
                raise ValueError(f"layer_range must satisfy 0 <= start < end, got ({start}, {end}).")

        # token scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")
