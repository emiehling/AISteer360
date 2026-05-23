"""CausalPoolMemory: trained causal model + frozen prompt pool."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np

from aisteer360.algorithms.input_control.cpo.dml import CausalRewardModel


@dataclass
class CausalPoolMemory:
    """Memory for CPO: trained `CausalRewardModel` + frozen prompt pool with embeddings.

    Attributes:
        pool: Candidate template strings.
        pool_embeddings: `[K, d_z]` embeddings of the pool (post-PCA if used).
        causal_model: A `CausalRewardModel` ready for τ̂ predictions.
        query_embedder_name_or_path: Identifier for re-loading the query embedder during deserialization, or None if
            the user supplied a `Callable`.
        prompt_embedder_name_or_path: Same for the prompt embedder.

    Used only by CPO at Phase 6. May be promoted to `common/memory/` if a second method adopts a similar shape.
    """

    pool: list[str]
    pool_embeddings: np.ndarray
    causal_model: CausalRewardModel
    query_embedder_name_or_path: str | None = None
    prompt_embedder_name_or_path: str | None = None

    model_type: str = field(default="causal_pool", init=False)

    _EXTENSION = ".cpm"

    def save(self, path: str) -> None:
        """Save to a directory `<path>.cpm/` containing `meta.json`, `pool.json`, `pool_embeddings.npy`,
        `causal_model.joblib`.

        Directory format chosen for inspectability — users can examine the pool and metadata without unpickling.

        Args:
            path: Output path (directory). `.cpm` extension appended if not present.
        """
        if not path.endswith(self._EXTENSION):
            path += self._EXTENSION
        os.makedirs(path, exist_ok=True)

        meta = {
            "model_type": self.model_type,
            "query_embedder_name_or_path": self.query_embedder_name_or_path,
            "prompt_embedder_name_or_path": self.prompt_embedder_name_or_path,
            "pool_size": len(self.pool),
            "pool_embedding_shape": list(self.pool_embeddings.shape),
            "nuisance_estimator_class": self.causal_model.nuisance_estimator_class,
            "effect_estimator_class": self.causal_model.effect_estimator_class,
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        with open(os.path.join(path, "pool.json"), "w", encoding="utf-8") as f:
            json.dump(self.pool, f, ensure_ascii=False, indent=2)

        np.save(os.path.join(path, "pool_embeddings.npy"), self.pool_embeddings)

        joblib.dump(self.causal_model, os.path.join(path, "causal_model.joblib"))

    @classmethod
    def load(cls, path: str) -> "CausalPoolMemory":
        """Load a `CausalPoolMemory` from a directory.

        Note: `causal_model.joblib` is unpickled via `joblib.load`, which executes arbitrary Python on a malicious
        artifact. Only load files from trusted sources.

        Args:
            path: Directory path. `.cpm` extension appended if not present.

        Returns:
            Loaded `CausalPoolMemory` instance.

        Raises:
            ValueError: If the meta `model_type` does not match this class.
        """
        if not path.endswith(cls._EXTENSION):
            path += cls._EXTENSION

        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model_type") != "causal_pool":
            raise ValueError(
                f"Cannot load CausalPoolMemory: meta model_type is "
                f"{meta.get('model_type')!r}, expected 'causal_pool'."
            )

        with open(os.path.join(path, "pool.json"), encoding="utf-8") as f:
            pool = json.load(f)

        pool_embeddings = np.load(os.path.join(path, "pool_embeddings.npy"))
        causal_model = joblib.load(os.path.join(path, "causal_model.joblib"))

        return cls(
            pool=pool,
            pool_embeddings=pool_embeddings,
            causal_model=causal_model,
            query_embedder_name_or_path=meta.get("query_embedder_name_or_path"),
            prompt_embedder_name_or_path=meta.get("prompt_embedder_name_or_path"),
        )
