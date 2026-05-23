"""Arguments for the CPO input control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.input_control.cpo.embedder import Embedder


@dataclass
class CPOArgs(BaseArgs):
    """Arguments for CPO (Causal Prompt Optimization).

    Required:
        training_data: Observational triples. Each dict has:

            - `"query"`: str (the user input)
            - `"prompt"`: str (the prompt template that was used)
            - `"outcome"`: float (the score / reward)

        prompt_pool: Candidate prompt templates. Required; not derived from training data. Pool entries are typically a
            superset of the unique prompts seen in `training_data`, possibly with LLM-refined variants.

    Optional:
        query_embedder: HF model identifier or `Embedder`-conforming instance.
        prompt_embedder: If None, reuse `query_embedder`. Otherwise an HF id or `Embedder`.
        embedder_kwargs: Kwargs forwarded to `HFMeanPoolEmbedder` when string identifiers are passed.
        embedding_dim_reduction: PCA target dim for prompt embeddings; None disables PCA.
        n_folds: K for cross-fitted nuisance estimation. Must be >= 2.
        nuisance_outcome_factory: Factory for the m̂(x) regressor. None uses the GBM default.
        nuisance_treatment_factory: Factory for the ê(x) multi-output regressor. None uses the GBM default.
        effect_estimator_factory: Factory for τ̂. None uses the random forest default.
        selection_temperature: 0.0 = argmax over pool. >0 = softmax sample.
        seed: RNG seed.
    """

    training_data: list[dict] = field(default_factory=list)
    prompt_pool: list[str] = field(default_factory=list)

    query_embedder: str | Embedder = "sentence-transformers/all-MiniLM-L6-v2"
    prompt_embedder: str | Embedder | None = None

    embedder_kwargs: dict | None = None

    embedding_dim_reduction: int | None = None

    n_folds: int = 5
    nuisance_outcome_factory: Callable[[], Any] | None = None
    nuisance_treatment_factory: Callable[[], Any] | None = None
    effect_estimator_factory: Callable[[], Any] | None = None

    selection_temperature: float = 0.0

    seed: int = 0

    def __post_init__(self) -> None:
        if not self.training_data:
            raise ValueError("training_data must be non-empty")
        if not self.prompt_pool:
            raise ValueError("prompt_pool must be non-empty")
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if self.embedding_dim_reduction is not None and self.embedding_dim_reduction < 1:
            raise ValueError("embedding_dim_reduction must be >= 1 if set")
        if self.selection_temperature < 0.0:
            raise ValueError("selection_temperature must be >= 0")

        required = {"query", "prompt", "outcome"}
        for i, row in enumerate(self.training_data):
            missing = required - set(row.keys())
            if missing:
                raise ValueError(
                    f"training_data[{i}] missing required keys: {sorted(missing)}"
                )
