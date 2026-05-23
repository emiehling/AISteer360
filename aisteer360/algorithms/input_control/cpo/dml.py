"""Causal reward model trained via Double Machine Learning (DML).

Implements a partial linear model with K-fold cross-fitted nuisance estimation, expressed in plain sklearn primitives.
The effect estimator predicts the residualized outcome from the concatenation of the (residualized) treatment embedding
and the query embedding, giving a per-(query, prompt) effect estimate at inference.

This is a deliberate simplification of `econml.dml.CausalForestDML`; users wanting the full causal-forest story can pass
their own `effect_estimator_factory`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)


@dataclass
class CausalRewardModel:
    """Trained causal reward model τ̂(x, t).

    Attributes:
        effect_estimator: sklearn estimator fit on `[x, z̃]` features to predict ỹ.
        nuisance_outcome: m̂(x); predicts y from the query embedding.
        nuisance_treatment: ê(x); predicts z (multi-output) from the query embedding.
        pca: Optional `sklearn.decomposition.PCA` fitted on prompt embeddings, or None.
        nuisance_estimator_class: Class name of the nuisance estimators (for serialization metadata).
        effect_estimator_class: Class name of the effect estimator (for serialization metadata).
    """

    effect_estimator: Any
    nuisance_outcome: Any
    nuisance_treatment: Any
    pca: Any | None = None
    nuisance_estimator_class: str = ""
    effect_estimator_class: str = ""

    metadata: dict = field(default_factory=dict)

    def predict(self, query_emb: np.ndarray, prompt_emb: np.ndarray) -> np.ndarray:
        """τ̂(x, t) for batched (query, prompt) pairs.

        Args:
            query_emb: `[N, d_x]` query embeddings.
            prompt_emb: `[N, d_z]` prompt embeddings (post-PCA if applicable).

        Returns:
            `[N]` predicted causal effect of each (query, prompt) pair.
        """
        query_emb = np.asarray(query_emb, dtype=np.float32)
        prompt_emb = np.asarray(prompt_emb, dtype=np.float32)
        if query_emb.ndim != 2 or prompt_emb.ndim != 2:
            raise ValueError("query_emb and prompt_emb must both be 2D")
        if query_emb.shape[0] != prompt_emb.shape[0]:
            raise ValueError(
                f"query_emb and prompt_emb must share the leading dim "
                f"(got {query_emb.shape[0]} vs {prompt_emb.shape[0]})"
            )

        z_hat = self.nuisance_treatment.predict(query_emb)
        if z_hat.ndim == 1:
            z_hat = z_hat.reshape(-1, 1)
        z_residual = prompt_emb - z_hat
        features = np.concatenate([query_emb, z_residual], axis=1)
        return self.effect_estimator.predict(features).astype(np.float32)


def _default_nuisance_outcome(rng_seed: int) -> Callable[[], Any]:
    def factory() -> Any:
        return GradientBoostingRegressor(random_state=rng_seed)
    return factory


def _default_nuisance_treatment(rng_seed: int) -> Callable[[], Any]:
    def factory() -> Any:
        return MultiOutputRegressor(GradientBoostingRegressor(random_state=rng_seed))
    return factory


def _default_effect_estimator(rng_seed: int) -> Callable[[], Any]:
    def factory() -> Any:
        return RandomForestRegressor(random_state=rng_seed)
    return factory


def train_causal_reward_model(
    query_embeddings: np.ndarray,
    prompt_embeddings: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_folds: int = 5,
    embedding_dim_reduction: int | None = None,
    nuisance_outcome_factory: Callable[[], Any] | None = None,
    nuisance_treatment_factory: Callable[[], Any] | None = None,
    effect_estimator_factory: Callable[[], Any] | None = None,
    rng_seed: int = 0,
) -> CausalRewardModel:
    """Train a causal reward model via Double Machine Learning.

    Algorithm (DML2 with simplified effect step):

      1. Optionally PCA-reduce `prompt_embeddings` to `embedding_dim_reduction`.
      2. K-fold partition. For each fold, fit nuisance models on out-of-fold data, predict residuals on the held-out
         fold.
      3. Pool residuals across folds and fit `effect_estimator` on `[x, z̃]` features to predict ỹ.
      4. Refit final nuisance models on the full data for inference.

    Args:
        query_embeddings: `[N, d_x]` query/confounder features.
        prompt_embeddings: `[N, d_z_raw]` treatment features.
        outcomes: `[N]` observed scores.
        n_folds: K for cross-fitting.
        embedding_dim_reduction: If set, PCA components for prompt embeddings.
        nuisance_outcome_factory: Factory for m̂(x). Defaults to GradientBoostingRegressor.
        nuisance_treatment_factory: Factory for ê(x). Defaults to MultiOutputRegressor(GradientBoostingRegressor).
        effect_estimator_factory: Factory for τ̂. Defaults to RandomForestRegressor.
        rng_seed: RNG seed for fold splitting and default estimators.

    Returns:
        A `CausalRewardModel` ready for inference via `predict`.
    """
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    prompt_embeddings = np.asarray(prompt_embeddings, dtype=np.float32)
    outcomes = np.asarray(outcomes, dtype=np.float32).reshape(-1)

    n = query_embeddings.shape[0]
    if n != prompt_embeddings.shape[0] or n != outcomes.shape[0]:
        raise ValueError("query_embeddings, prompt_embeddings, outcomes must share the leading dim")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n < n_folds:
        raise ValueError(f"Need at least n_folds={n_folds} rows; got {n}")

    nuisance_outcome_factory = nuisance_outcome_factory or _default_nuisance_outcome(rng_seed)
    nuisance_treatment_factory = nuisance_treatment_factory or _default_nuisance_treatment(rng_seed)
    effect_estimator_factory = effect_estimator_factory or _default_effect_estimator(rng_seed)

    pca = None
    if embedding_dim_reduction is not None:
        if embedding_dim_reduction < 1:
            raise ValueError("embedding_dim_reduction must be >= 1")
        n_components = min(embedding_dim_reduction, prompt_embeddings.shape[1], n)
        pca = PCA(n_components=n_components, random_state=rng_seed)
        prompt_embeddings = pca.fit_transform(prompt_embeddings).astype(np.float32)

    if prompt_embeddings.ndim == 1:
        prompt_embeddings = prompt_embeddings.reshape(-1, 1)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng_seed)
    y_residuals = np.zeros_like(outcomes)
    z_residuals = np.zeros_like(prompt_embeddings)

    for train_idx, holdout_idx in kf.split(query_embeddings):
        m_hat = nuisance_outcome_factory()
        e_hat = nuisance_treatment_factory()

        m_hat.fit(query_embeddings[train_idx], outcomes[train_idx])
        e_hat.fit(query_embeddings[train_idx], prompt_embeddings[train_idx])

        y_pred = m_hat.predict(query_embeddings[holdout_idx])
        z_pred = e_hat.predict(query_embeddings[holdout_idx])
        if z_pred.ndim == 1:
            z_pred = z_pred.reshape(-1, 1)

        y_residuals[holdout_idx] = outcomes[holdout_idx] - y_pred
        z_residuals[holdout_idx] = prompt_embeddings[holdout_idx] - z_pred

    effect_features = np.concatenate([query_embeddings, z_residuals], axis=1)
    effect_estimator = effect_estimator_factory()
    effect_estimator.fit(effect_features, y_residuals)

    final_outcome = nuisance_outcome_factory()
    final_treatment = nuisance_treatment_factory()
    final_outcome.fit(query_embeddings, outcomes)
    final_treatment.fit(query_embeddings, prompt_embeddings)

    return CausalRewardModel(
        effect_estimator=effect_estimator,
        nuisance_outcome=final_outcome,
        nuisance_treatment=final_treatment,
        pca=pca,
        nuisance_estimator_class=type(final_outcome).__name__,
        effect_estimator_class=type(effect_estimator).__name__,
        metadata={
            "n_folds": int(n_folds),
            "n_samples": int(n),
            "d_x": int(query_embeddings.shape[1]),
            "d_z": int(prompt_embeddings.shape[1]),
            "embedding_dim_reduction": embedding_dim_reduction,
            "rng_seed": int(rng_seed),
        },
    )
