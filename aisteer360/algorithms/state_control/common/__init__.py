"""State control component library.

Re-exports the main public types so users can do:
    from aisteer360.algorithms.state_control.common import SteeringVector, ContrastivePairs, ...
"""
from .specs import (
    Comparator,
    CompMode,
    ConditionSearchSpec,
    ContrastivePairs,
    VectorTrainSpec,
    as_contrastive_pairs,
)
from .steering_vector import SteeringVector
