"""Angular steering argument validation."""
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

    Users provide EITHER a pre-computed steering vector OR training data.
    If data is provided, the basis pair is fitted during ``steer()``.

    Attributes:
        steering_vector: Pre-computed steering vector with ``K=2`` per layer
            (row 0 = first direction, row 1 = second direction).
            If provided, skip training.
        data: Contrastive pairs for training.  Positives define the first
            direction (e.g., harmful prompts); negatives the opposite.
            Required if ``steering_vector`` is ``None``.
        train_spec: Controls batch size for hidden-state extraction.
        layer_id: Layer to apply steering at.  If ``None`` (default), all
            layers with directions in the steering vector are steered,
            matching the reference implementation.  Set explicitly for
            single-layer ablation.
        layer_selection: Strategy for auto-selecting the best layer during
            direction extraction (used by ``AngularDirectionEstimator``).
            ``"max_sim"`` picks the layer whose candidate direction has the
            highest mean pairwise cosine similarity with all other layers.
            ``"max_norm"`` picks the layer with the largest separation
            between positive and negative representations.
        target_degree: Rotation angle in degrees.  ``0`` = aligned with first
            direction; ``180`` = maximally opposite (e.g., maximum refusal);
            ``90`` / ``270`` = perpendicular.
        adaptive_mode: ``0`` = always steer all masked positions.
            ``1`` = only steer positions with positive alignment to the first
            direction (recommended).
        token_scope: Which tokens to steer.
        last_k: Required when ``token_scope == "last_k"``.
        from_position: Required when ``token_scope == "from_position"``.
    """

    # steering vector source (provide exactly one)
    steering_vector: SteeringVector | None = None
    data: ContrastivePairs | dict | None = None

    # training configuration
    train_spec: VectorTrainSpec | dict = field(
        default_factory=lambda: VectorTrainSpec(
            method="mean_diff", accumulate="last_token"
        )
    )

    # inference configuration
    layer_id: int | None = None
    layer_selection: Literal["max_sim", "max_norm"] = "max_sim"
    target_degree: float = 180.0
    adaptive_mode: int = 1
    token_scope: TokenScope = "all"
    last_k: int | None = None
    from_position: int | None = None

    def __post_init__(self):
        # exactly one of steering_vector or data must be provided
        if self.steering_vector is None and self.data is None:
            raise ValueError("Provide either steering_vector or data.")
        if self.steering_vector is not None and self.data is not None:
            raise ValueError("Provide steering_vector or data, not both.")

        # validate steering_vector if provided
        if self.steering_vector is not None:
            self.steering_vector.validate()

        # normalize dict inputs
        if self.data is not None and not isinstance(self.data, ContrastivePairs):
            object.__setattr__(self, "data", as_contrastive_pairs(self.data))

        if isinstance(self.train_spec, dict):
            object.__setattr__(self, "train_spec", VectorTrainSpec(**self.train_spec))

        # validate layer_id if provided
        if self.layer_id is not None and self.layer_id < 0:
            raise ValueError("layer_id must be >= 0.")

        if self.adaptive_mode not in (0, 1):
            raise ValueError("adaptive_mode must be 0 or 1.")

        # token scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError(
                "last_k must be >= 1 when token_scope is 'last_k'."
            )
        if self.token_scope == "from_position" and (
            self.from_position is None or self.from_position < 0
        ):
            raise ValueError(
                "from_position must be >= 0 when token_scope is 'from_position'."
            )
