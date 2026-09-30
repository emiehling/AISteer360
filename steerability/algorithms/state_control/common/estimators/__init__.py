"""Estimator components for state control."""
from .base import BaseEstimator
from .contrastive_direction import ContrastiveDirectionEstimator
from .mean_difference import MeanDifferenceEstimator
from .single_pair import SinglePairEstimator
from .steering_plane import SteeringPlaneEstimator


def estimator_for(method: str) -> MeanDifferenceEstimator | ContrastiveDirectionEstimator:
    """Return the built-in direction estimator for a `VectorTrainSpec.method` value.

    `"mean_diff"` maps to `MeanDifferenceEstimator`. Every other value maps to
    `ContrastiveDirectionEstimator`, which fits `"pca_pairwise"` and `"pca_center"`. For any other
    method, its `fit()` raises `ValueError` after the hidden states are extracted.

    Args:
        method: The direction-extraction method, e.g., `"mean_diff"`, `"pca_pairwise"`, or
            `"pca_center"`.

    Returns:
        A new estimator instance.
    """
    if method == "mean_diff":
        return MeanDifferenceEstimator()
    return ContrastiveDirectionEstimator()
