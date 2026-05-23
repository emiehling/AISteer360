"""CPO: Causal Prompt Optimization input control."""
from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.cpo.args import CPOArgs
from aisteer360.algorithms.input_control.cpo.dml import train_causal_reward_model
from aisteer360.algorithms.input_control.cpo.embedder import Embedder, HFMeanPoolEmbedder
from aisteer360.algorithms.input_control.cpo.memory import CausalPoolMemory

logger = logging.getLogger(__name__)


class CPO(InputControl):
    """Causal Prompt Optimization: per-query template selection via a DML-trained causal reward model.

    Two-stage design:

      1. `steer()` trains a causal reward model τ̂(x, t) from observational `(query, prompt, outcome)` triples and
         freezes the prompt pool.
      2. `adapt()` embeds the incoming query, evaluates τ̂ for every pool member, picks the argmax (or softmax-samples
         when `selection_temperature > 0`), and assembles the chosen template with the user input via the model's chat
         template.

    Unlike GEPA / FewShot / SCOPE, the adapter is *instance-adaptive*: the same control can produce different steered
    inputs for different queries. The selection model runs in serving (one model forward pass per query, not an LLM
    call).

    `is_stateful = False`. CPO has no `observe()` and no per-call state across invocations.

    Reference:

      - "Optimizing Prompts for Large Language Models: A Causal Approach"
        Chen et al.
        [https://arxiv.org/abs/2602.01711](https://arxiv.org/abs/2602.01711)
    """

    Args = CPOArgs
    is_stateful: bool = False
    supports_batching: bool = False

    tokenizer: PreTrainedTokenizer | None = None
    memory: CausalPoolMemory | None = None
    _query_embedder: Embedder | None = None
    _prompt_embedder: Embedder | None = None
    _rng: random.Random | None = None

    def steer(
        self,
        model: Any = None,
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs,
    ) -> None:
        """Train the causal reward model.

        `model` is ignored — CPO learns from observational data, not from task-LM rollouts.
        """
        self.tokenizer = tokenizer
        self._rng = random.Random(self.seed)

        self._query_embedder = self._resolve_embedder(self.query_embedder)
        self._prompt_embedder = (
            self._resolve_embedder(self.prompt_embedder)
            if self.prompt_embedder is not None
            else self._query_embedder
        )

        train_queries = [row["query"] for row in self.training_data]
        train_prompts = [row["prompt"] for row in self.training_data]
        outcomes = np.array([row["outcome"] for row in self.training_data], dtype=np.float32)

        query_embs = self._query_embedder.embed(train_queries)
        prompt_embs = self._prompt_embedder.embed(train_prompts)

        crm = train_causal_reward_model(
            query_embeddings=query_embs,
            prompt_embeddings=prompt_embs,
            outcomes=outcomes,
            n_folds=self.n_folds,
            embedding_dim_reduction=self.embedding_dim_reduction,
            nuisance_outcome_factory=self.nuisance_outcome_factory,
            nuisance_treatment_factory=self.nuisance_treatment_factory,
            effect_estimator_factory=self.effect_estimator_factory,
            rng_seed=self.seed,
        )

        pool_raw_embs = self._prompt_embedder.embed(self.prompt_pool)
        if crm.pca is not None:
            pool_embs = crm.pca.transform(pool_raw_embs).astype(np.float32)
        else:
            pool_embs = pool_raw_embs.astype(np.float32)

        self.memory = CausalPoolMemory(
            pool=list(self.prompt_pool),
            pool_embeddings=pool_embs,
            causal_model=crm,
            query_embedder_name_or_path=(
                self.query_embedder if isinstance(self.query_embedder, str) else None
            ),
            prompt_embedder_name_or_path=(
                self.prompt_embedder if isinstance(self.prompt_embedder, str) else None
            ),
        )

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        prior: Output | None = None,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Per-query template selection: embed query → score pool → pick argmax → assemble."""
        if self.tokenizer is None or self.memory is None:
            raise RuntimeError("CPO needs to be steered first.")
        if self._query_embedder is None:
            raise RuntimeError(
                "CPO query embedder is missing. If you reloaded a saved memory, you must re-construct the embedder "
                "before adapting."
            )

        is_tensor = isinstance(input_ids, torch.Tensor)
        if is_tensor and input_ids.ndim > 1 and input_ids.shape[0] > 1:
            raise NotImplementedError(
                "CPO.adapt currently handles single-sequence input only (batch dim must be 1 if present)."
            )

        if is_tensor:
            ids_list = input_ids.reshape(-1).tolist()
        elif input_ids and isinstance(input_ids[0], list):
            if len(input_ids) > 1:
                raise NotImplementedError(
                    "CPO.adapt currently handles single-sequence input only (batch dim must be 1 if present)."
                )
            ids_list = list(input_ids[0])
        else:
            ids_list = list(input_ids)

        query_text = self.tokenizer.decode(ids_list, skip_special_tokens=True)

        q_emb = self._query_embedder.embed([query_text])
        K = len(self.memory.pool)
        q_emb_tiled = np.repeat(q_emb, K, axis=0)
        scores = self.memory.causal_model.predict(
            query_emb=q_emb_tiled,
            prompt_emb=self.memory.pool_embeddings,
        )

        if self.selection_temperature == 0.0:
            chosen_idx = int(np.argmax(scores))
        else:
            chosen_idx = self._softmax_sample(scores, self.selection_temperature)

        chosen_template = self.memory.pool[chosen_idx]
        return self._apply_template(input_ids, chosen_template)

    def cleanup(self) -> None:
        """Release embedder models. Idempotent. Avoids double-cleanup when query and prompt embedders share an
        instance."""
        if self._query_embedder is not None and hasattr(self._query_embedder, "cleanup"):
            self._query_embedder.cleanup()
        if (
            self._prompt_embedder is not None
            and self._prompt_embedder is not self._query_embedder
            and hasattr(self._prompt_embedder, "cleanup")
        ):
            self._prompt_embedder.cleanup()
        self._query_embedder = None
        self._prompt_embedder = None

    def _resolve_embedder(self, spec: str | Embedder) -> Embedder:
        if isinstance(spec, str):
            return HFMeanPoolEmbedder(spec, **(self.embedder_kwargs or {}))
        return spec

    def _softmax_sample(self, scores: np.ndarray, temperature: float) -> int:
        scaled = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-12)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        rng = self._rng or random.Random()
        u = rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += float(p)
            if u <= cumulative:
                return i
        return len(probs) - 1

    def _apply_template(
        self,
        input_ids: list[int] | torch.Tensor,
        template: str,
    ) -> list[int] | torch.Tensor:
        is_tensor = isinstance(input_ids, torch.Tensor)
        original_device = input_ids.device if is_tensor else None
        original_dtype = input_ids.dtype if is_tensor else None

        if is_tensor:
            if input_ids.ndim == 1:
                ids_list = input_ids.tolist()
                single_sequence = True
            else:
                ids_list = input_ids.reshape(-1).tolist()
                single_sequence = input_ids.shape[0] == 1
        else:
            if input_ids and isinstance(input_ids[0], list):
                ids_list = list(input_ids[0])
                single_sequence = True
            else:
                ids_list = list(input_ids)
                single_sequence = True

        original_text = self.tokenizer.decode(ids_list, skip_special_tokens=True)
        has_chat_template = (
            hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template
        )

        if has_chat_template:
            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": original_text},
            ]
            adapted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            adapted_text = f"{template}\n\n{original_text}"

        adapted_tokens = self.tokenizer.encode(adapted_text, add_special_tokens=False)

        if is_tensor:
            tensor_out = torch.tensor(adapted_tokens, dtype=original_dtype, device=original_device)
            if single_sequence and (input_ids.ndim == 1):
                return tensor_out
            return tensor_out.unsqueeze(0)
        return adapted_tokens
