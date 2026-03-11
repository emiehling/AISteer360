"""Steering plane estimator for Angular Steering."""
import logging

import torch
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..specs import ContrastivePairs, VectorTrainSpec
from ..steering_vector import SteeringVector
from .base import BaseEstimator
from .mean_difference_estimator import MeanDifferenceEstimator

logger = logging.getLogger(__name__)


class SteeringPlaneEstimator(BaseEstimator[SteeringVector]):
    """Learns a 2D steering plane per layer for rotational activation steering.

    For each layer, produces an orthonormal basis pair (b1, b2) that defines
    a plane in activation space:

    1. Compute the mean-difference feature direction per layer via
       ``MeanDifferenceEstimator``.
    2. Stack all per-layer feature directions and run PCA across layers to
       extract the principal component (PC0), capturing the direction of
       maximum cross-layer variance.
    3. Per layer, construct the orthonormal basis via Gram-Schmidt:
       ``b1 = normalize(d_feat[layer])``
       ``b2 = normalize(PC0 - (PC0 · b1) * b1)``

    The resulting ``SteeringVector`` has ``directions[layer] = [2, H]``
    where row 0 is b1 (feature direction) and row 1 is b2 (orthogonal axis).

    Reference:

        - "Angular Steering: Improving LLM Alignment with Simple Activation Rotations"
          Tuan Vu, Thang Nguyen
          [https://arxiv.org/abs/2504.02406](https://arxiv.org/abs/2504.02406)
    """

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: ContrastivePairs,
        spec: VectorTrainSpec,
        normalize: bool = True,
    ) -> SteeringVector:
        """Extract steering plane basis pairs.

        Args:
            model: Model to extract hidden states from.
            tokenizer: Tokenizer for encoding the contrastive pairs.
            data: The positive/negative text pairs.
            spec: Training configuration (method, accumulate, batch_size).
            normalize: If True, L2-normalize b1 before Gram-Schmidt.

        Returns:
            SteeringVector with K=2 per layer (orthonormal basis pair).
        """
        # step 1: get per-layer mean-difference directions
        mean_diff_estimator = MeanDifferenceEstimator()
        mean_diff_sv = mean_diff_estimator.fit(model, tokenizer, data=data, spec=spec)

        model_type = mean_diff_sv.model_type
        layer_ids = sorted(mean_diff_sv.directions.keys())
        num_layers = len(layer_ids)
        logger.debug("Computing steering planes for %d layers", num_layers)

        # step 2: stack all feature directions and run cross-layer PCA
        # each direction is [1, H], squeeze to [H]
        feat_dirs = []
        for lid in layer_ids:
            d = mean_diff_sv.directions[lid].squeeze(0)  # [H]
            feat_dirs.append(d)

        feat_matrix = torch.stack(feat_dirs, dim=0).float()  # [num_layers, H]

        pca = PCA(n_components=1)
        pca.fit(feat_matrix.numpy())
        pc0 = torch.tensor(pca.components_[0], dtype=torch.float32)  # [H]
        logger.debug("Cross-layer PCA explained variance: %.4f", pca.explained_variance_ratio_[0])

        # step 3: per layer, Gram-Schmidt to build orthonormal basis
        directions: dict[int, torch.Tensor] = {}
        for i, lid in enumerate(layer_ids):
            d_feat = feat_dirs[i].float()  # [H]

            # b1 = normalize(d_feat)
            b1_norm = d_feat.norm()
            if b1_norm < 1e-10:
                logger.warning("Near-zero feature direction at layer %d, skipping", lid)
                continue
            b1 = d_feat / b1_norm if normalize else d_feat

            # b2 = normalize(pc0 - (pc0 · b1) * b1)
            proj = (pc0 @ b1) * b1
            b2_raw = pc0 - proj
            b2_norm = b2_raw.norm()
            if b2_norm < 1e-10:
                logger.warning(
                    "PC0 nearly collinear with feature direction at layer %d, skipping", lid
                )
                continue
            b2 = b2_raw / b2_norm

            directions[lid] = torch.stack([b1, b2], dim=0)  # [2, H]

        logger.debug("Finished fitting steering planes for %d layers", len(directions))
        return SteeringVector(
            model_type=model_type,
            directions=directions,
        )
