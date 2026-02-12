"""CAST argument validation."""
from dataclasses import dataclass, field
from typing import Sequence

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.state_control.common.specs import (
    Comparator,
    CompMode,
    ConditionSearchSpec,
    ContrastivePairs,
    VectorTrainSpec,
    as_contrastive_pairs,
)
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.token_scope import TokenScope


@dataclass
class CASTArgs(BaseArgs):
    """Arguments for CAST (Conditional Activation Steering).

    Users provide EITHER pre-computed vectors OR training data. If data is
    provided, vectors are fitted during steer(). If vectors are provided,
    data is ignored.

    All layer validation happens in steer() once the model is known.
    """

    # behavior
    behavior_vector: SteeringVector | None = None
    behavior_data: ContrastivePairs | dict | None = None
    behavior_fit: VectorTrainSpec = field(default_factory=VectorTrainSpec)
    behavior_layer_ids: Sequence[int] | None = None
    behavior_vector_strength: float = 1.0

    # condition
    condition_vector: SteeringVector | None = None
    condition_data: ContrastivePairs | dict | None = None
    condition_fit: VectorTrainSpec = field(
        default_factory=lambda: VectorTrainSpec(accumulate="all")
    )
    search: ConditionSearchSpec = field(default_factory=ConditionSearchSpec)
    condition_layer_ids: Sequence[int] | None = None
    condition_vector_threshold: float | None = None
    condition_comparator_threshold_is: Comparator = "larger"
    condition_threshold_comparison_mode: CompMode = "mean"

    # hook behavior
    apply_behavior_on_first_call: bool = True
    use_ooi_preventive_normalization: bool = False
    use_explained_variance: bool = False
    token_scope: TokenScope = "all"
    last_k: int | None = None

    def __post_init__(self):
        # rule 1: model-agnostic validation only

        if self.behavior_vector_strength < 0:
            raise ValueError("behavior_vector_strength must be >= 0.")

        if self.behavior_vector is not None:
            self.behavior_vector.validate()
        if self.condition_vector is not None:
            self.condition_vector.validate()

        # normalize dict inputs to ContrastivePairs
        if self.behavior_data is not None and not isinstance(self.behavior_data, ContrastivePairs):
            object.__setattr__(self, "behavior_data", as_contrastive_pairs(self.behavior_data))
        if self.condition_data is not None and not isinstance(self.condition_data, ContrastivePairs):
            object.__setattr__(self, "condition_data", as_contrastive_pairs(self.condition_data))

        # must have at least one source for behavior
        if self.behavior_vector is None and self.behavior_data is None:
            raise ValueError("Provide either behavior_vector or behavior_data.")

        # if condition vector is given, condition layers should also be given
        # (or search.auto_find should be True)
        if self.condition_vector is not None and self.condition_layer_ids is None and not self.search.auto_find:
            raise ValueError(
                "When condition_vector is provided without condition_layer_ids, "
                "search.auto_find must be True."
            )

        # token scope cross-check
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
