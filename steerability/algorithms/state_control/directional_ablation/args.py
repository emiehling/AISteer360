"""Directional Ablation argument validation."""
from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.internals.data import ContrastivePairs, as_contrastive_pairs
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.token_scope import ScopeKind


@dataclass
class DirectionalAblationArgs(BaseArgs):
    """Arguments for Directional Ablation.

    Exactly one of `steering_vector` (a precomputed vector) or `data` (contrastive pairs) must be
    given. With `data`, the feature direction is fitted during `steer()` with the method given by
    `train_spec.method`. The default `train_spec` is `VectorTrainSpec(method="mean_diff",
    accumulate="last_token")`, which takes the difference in means over the last-token hidden
    states of the contrastive data. A `train_spec` given as a dict or `VectorTrainSpec` uses the
    `VectorTrainSpec` defaults for the fields it omits (e.g., `method="pca_pairwise"`). A
    precomputed vector may contain any `K >= 1` directions per layer. `K=1` ablates a single
    direction and `K>1` ablates the whole subspace.

    Attributes:
        steering_vector: Precomputed direction(s), `[K, H]` per layer. When given, no fit runs.
        data: Contrastive pairs for fitting the direction. Required if `steering_vector` is None. A
            dict is converted to `ContrastivePairs`.
        train_spec: Fit configuration (extraction method and accumulation mode). A dict is converted
            to `VectorTrainSpec`.
        alpha: Ablation strength in `[0, 1]`. `1.0` fully removes the component (`h'.d == 0`), and
            values below `1.0` give graded partial suppression. Values outside `[0, 1]` raise
            `ValueError` (use rotation or additive steering to induce the opposite behavior).
        layer_ids: Layers to ablate at. If None, a single layer at about 40% depth is used.
        layer_range: Optional half-open `[start, end)` range that restricts the layers of the
            fitted or precomputed directions.
        token_scope: Which tokens to ablate (`"all"`, `"after_prompt"`, `"last_k"`, or
            `"from_position"`).
        last_k: Number of final positions of each forward pass to ablate. Required (at least 1)
            when `token_scope` is `"last_k"`.
        from_position: The absolute position from which ablation starts. Required (at least 0)
            when `token_scope` is `"from_position"`.
        use_norm_preservation: If True, wrap the transform in `NormPreservingTransform`, which
            rescales any position whose norm increased back to its original norm. Defaults to
            False. Ablation does not increase the norm, and the wrapper therefore has no effect up
            to rounding.
    """

    # direction source (provide exactly one)
    steering_vector: SteeringVector | None = None
    data: ContrastivePairs | dict | None = None

    # training configuration
    train_spec: VectorTrainSpec | dict = field(
        default_factory=lambda: VectorTrainSpec(method="mean_diff", accumulate="last_token")
    )

    # ablation configuration
    alpha: float = 1.0  # ablation strength in [0, 1]

    # layer configuration
    layer_ids: list[int] | None = None  # explicit layers; None -> heuristic (~40% depth)
    layer_range: tuple[int, int] | None = None  # optional half-open [start, end) filter

    # inference configuration
    token_scope: ScopeKind = "all"
    last_k: int | None = None
    from_position: int | None = None
    use_norm_preservation: bool = False

    def __post_init__(self):
        # exactly one of steering_vector or data must be provided
        if self.steering_vector is None and self.data is None:
            raise ValueError("Provide either steering_vector or data.")
        if self.steering_vector is not None and self.data is not None:
            raise ValueError("Provide steering_vector or data, not both.")

        # validate a precomputed vector (any K >= 1 is accepted)
        if self.steering_vector is not None:
            self.steering_vector.validate()

        # normalize dict inputs
        if self.data is not None and not isinstance(self.data, ContrastivePairs):
            object.__setattr__(self, "data", as_contrastive_pairs(self.data))

        if isinstance(self.train_spec, dict):
            object.__setattr__(self, "train_spec", VectorTrainSpec(**self.train_spec))

        # validate alpha
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}.")

        # validate layer_ids if provided
        if self.layer_ids is not None:
            if len(self.layer_ids) == 0:
                raise ValueError("layer_ids must be non-empty when provided.")
            if any(lid < 0 for lid in self.layer_ids):
                raise ValueError("layer_ids must all be >= 0.")

        # validate layer_range if provided
        if self.layer_range is not None:
            start, end = self.layer_range
            if start < 0 or end <= start:
                raise ValueError(
                    f"layer_range must be a half-open [start, end) with 0 <= start < end; got {self.layer_range}."
                )

        # token scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")
