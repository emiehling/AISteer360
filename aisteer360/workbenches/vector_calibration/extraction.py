"""Steering vector extraction from contrastive pairs."""
import logging
from typing import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.common.estimators import (
    ContrastiveDirectionEstimator,
    MeanDifferenceEstimator,
)
from aisteer360.algorithms.state_control.common.specs import (
    ContrastivePairs,
    VectorTrainSpec,
)
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector

from .configs import ExtractionConfig

logger = logging.getLogger(__name__)

ESTIMATOR_REGISTRY = {
    "mean_diff": MeanDifferenceEstimator,
    "pca_pairwise": ContrastiveDirectionEstimator,
}


class SteeringVectorExtractor:
    """Extracts a `SteeringVector` from contrastive pairs.

    Wraps the existing `BaseEstimator` subclasses with the additional post-processing options exposed by the
    calibration dashboard (normalize, per-layer rescale, layer filtering).

    Note:
        The `center` flag is accepted for API completeness but the underlying estimators already subtract the
        positive/negative means as part of their computation, so this flag is currently a no-op.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config

    def extract(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        pairs: ContrastivePairs,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> SteeringVector:
        """Run extraction and return a `SteeringVector`.

        Args:
            model: The steered model (hidden states are extracted from this).
            tokenizer: Corresponding tokenizer.
            pairs: Contrastive pairs from the generation stage.
            on_progress: Optional `(completed, total)` callback fired at coarse stage boundaries
                (fit start, fit complete, post-processing complete).

        Returns:
            `SteeringVector` with one direction per extracted layer.
        """
        cfg = self.config

        estimator_cls = ESTIMATOR_REGISTRY.get(cfg.method)
        if estimator_cls is None:
            raise ValueError(
                f"Unknown estimator '{cfg.method}'. Available: {list(ESTIMATOR_REGISTRY)}"
            )

        if on_progress:
            on_progress(0, 2)

        sv = self._fit_with_oom_retry(estimator_cls, model, tokenizer, pairs, cfg)

        if on_progress:
            on_progress(1, 2)

        # layer filtering
        if cfg.layers != "all":
            allowed = set(cfg.layers)
            sv.directions = {k: v for k, v in sv.directions.items() if k in allowed}
            if sv.explained_variances is not None:
                sv.explained_variances = {
                    k: v for k, v in sv.explained_variances.items() if k in allowed
                }

        # L2 normalization
        if cfg.normalize:
            for layer_id, direction in sv.directions.items():
                norm = direction.norm()
                if norm > 0:
                    sv.directions[layer_id] = direction / norm

        # per-layer rescale by explained variance (PCA only)
        if cfg.per_layer_rescale and sv.explained_variances:
            for layer_id, direction in sv.directions.items():
                var = sv.explained_variances.get(layer_id, 1.0)
                sv.directions[layer_id] = direction * var
        elif cfg.per_layer_rescale:
            logger.warning(
                "per_layer_rescale requested but no explained variances available (method=%s); skipping.",
                cfg.method,
            )

        if on_progress:
            on_progress(2, 2)

        return sv

    @staticmethod
    def _fit_with_oom_retry(
        estimator_cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        pairs: ContrastivePairs,
        cfg: ExtractionConfig,
    ) -> SteeringVector:
        """Run ``estimator.fit()``, halving ``batch_size`` on CUDA OOM until it succeeds.

        Extraction calls the model with ``output_hidden_states=True``, which forces every
        layer's hidden state to be materialised simultaneously.  For hybrid architectures
        (e.g. Mamba-2/Transformer) this can require vastly more memory than normal
        inference, so the configured ``batch_size`` may be too large even when generation
        at the same batch size succeeds.
        """
        batch_size = cfg.batch_size

        while batch_size >= 1:
            spec = VectorTrainSpec(
                method=cfg.method,
                accumulate=cfg.accumulate,
                batch_size=batch_size,
            )
            try:
                estimator = estimator_cls()
                return estimator.fit(model, tokenizer, data=pairs, spec=spec)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise
                new_bs = max(1, batch_size // 2)
                logger.warning(
                    "Extraction OOM at batch_size=%d; retrying with batch_size=%d.",
                    batch_size,
                    new_bs,
                )
                batch_size = new_bs

        # unreachable, but keeps the type checker happy
        raise RuntimeError("Extraction failed.")