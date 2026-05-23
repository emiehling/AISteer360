"""EPR: Efficient Prompt Retrieval input control."""
from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.epr.args import EPRArgs
from aisteer360.algorithms.input_control.epr.contrastive import (
    ContrastiveConfig,
    train_contrastive_retriever,
)
from aisteer360.algorithms.input_control.epr.encoders import HFEncoder
from aisteer360.algorithms.input_control.epr.memory import RetrievalMemory
from aisteer360.algorithms.input_control.epr.retrieval import (
    bm25_search,
    build_bm25_index,
    dense_top_k,
)
from aisteer360.algorithms.input_control.epr.scoring import generate_contrastive_data

logger = logging.getLogger(__name__)


class EPR(InputControl):
    """Efficient Prompt Retrieval: per-query demonstration selection via a trained dense retriever.

    At serve time: embed the query, find the K most similar training examples via inner-product over the pre-computed
    corpus embeddings, sort by ascending similarity (closest example last, nearest to the query token), assemble into
    a single prompt via the configured templates.

    Three modes:

      - `"bm25"`: unsupervised sparse retrieval (TF-IDF cosine).
      - `"dense"`: unsupervised dense retrieval (off-the-shelf encoder).
      - `"epr"` (default): trained dense retriever per the paper.

    `is_stateful = False`; the adapter is instance-adaptive but deterministic given the trained retriever.

    Reference:

      - "Learning to Retrieve Prompts for In-Context Learning"
        Ohad Rubin, Jonathan Herzig, Jonathan Berant
        [https://arxiv.org/abs/2112.08633](https://arxiv.org/abs/2112.08633)
    """

    Args = EPRArgs
    is_stateful: bool = False
    supports_batching: bool = False

    tokenizer: PreTrainedTokenizer | None = None
    memory: RetrievalMemory | None = None
    _input_encoder: HFEncoder | None = None
    _prompt_encoder: HFEncoder | None = None
    _scoring_lm: Any = None
    _owned_scoring_lm: bool = False

    def steer(
        self,
        model: Any = None,
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs,
    ) -> None:
        """Train (or build) the retriever and freeze the corpus embeddings."""
        self.tokenizer = tokenizer

        if self.mode == "bm25":
            self._steer_bm25()
        elif self.mode == "dense":
            self._steer_dense_unsupervised()
        else:  # "epr"
            if model is None and self.scoring_lm is None:
                raise ValueError(
                    "EPR mode='epr' requires either a task `model` passed via steer() (used as scoring LM) or an "
                    "explicit `scoring_lm` in EPRArgs."
                )
            self._steer_epr(model, tokenizer)

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        prior: Output | None = None,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Decode query → retrieve top-K demos → format prompt → re-encode."""
        if self.tokenizer is None or self.memory is None:
            raise RuntimeError("EPR needs to be steered first.")

        is_tensor = isinstance(input_ids, torch.Tensor)
        if is_tensor and input_ids.ndim > 1 and input_ids.shape[0] > 1:
            raise NotImplementedError(
                "EPR.adapt currently handles single-sequence input only (batch dim must be 1 if present)."
            )
        if not is_tensor and input_ids and isinstance(input_ids[0], list) and len(input_ids) > 1:
            raise NotImplementedError(
                "EPR.adapt currently handles single-sequence input only (batch dim must be 1 if present)."
            )

        query_text = self._decode_single(input_ids)
        demo_indices = self._retrieve(query_text, k=self.n_demonstrations)
        demos = [self.memory.corpus[i] for i in demo_indices]

        # paper convention: ascending similarity; closest demo appears LAST
        demos = list(reversed(demos))

        prompt = self._assemble_prompt(demos, query_text)
        return self._encode_to_ids(prompt, original_input_ids=input_ids)

    def cleanup(self) -> None:
        """Release encoders and any owned scoring LM. Idempotent."""
        if self._input_encoder is not None and hasattr(self._input_encoder, "cleanup"):
            self._input_encoder.cleanup()
        if (
            self._prompt_encoder is not None
            and self._prompt_encoder is not self._input_encoder
            and hasattr(self._prompt_encoder, "cleanup")
        ):
            self._prompt_encoder.cleanup()
        if self._scoring_lm is not None and self._owned_scoring_lm:
            del self._scoring_lm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._input_encoder = None
        self._prompt_encoder = None
        self._scoring_lm = None
        self._owned_scoring_lm = False

    # mode-specific steer methods

    def _steer_bm25(self) -> None:
        index = build_bm25_index([row["input"] for row in self.corpus])
        self.memory = RetrievalMemory(
            corpus=list(self.corpus),
            mode="bm25",
            bm25_state=index,
            demo_template=self.demo_template,
            demo_separator=self.demo_separator,
        )

    def _steer_dense_unsupervised(self) -> None:
        encoder = HFEncoder(
            self.base_encoder_name_or_path,
            pooling=self.encoder_pooling,
            batch_size=self.encoder_batch_size,
            device=self._resolve_device(),
            max_length=self.encoder_max_length,
            trainable=False,
        )
        demo_texts = [self.demo_template.format(**row) for row in self.corpus]
        embeddings = encoder.embed(demo_texts)
        self.memory = RetrievalMemory(
            corpus=list(self.corpus),
            mode="dense",
            dense_embeddings=embeddings,
            input_encoder_name_or_path=self.base_encoder_name_or_path,
            prompt_encoder_name_or_path=self.base_encoder_name_or_path,
            encoder_pooling=self.encoder_pooling,
            demo_template=self.demo_template,
            demo_separator=self.demo_separator,
        )
        self._input_encoder = encoder
        self._prompt_encoder = encoder

    def _steer_epr(self, model: Any, tokenizer: Any) -> None:
        scoring_lm, scoring_tokenizer = self._resolve_scoring_lm(model, tokenizer)
        self._scoring_lm = scoring_lm

        # Stage 1a: BM25 candidate index over OUTPUTS (paper convention)
        bm25 = build_bm25_index([row["output"] for row in self.corpus])

        # Stage 1b: score candidates via the scoring LM and label positives/negatives
        labeled_data = generate_contrastive_data(
            corpus=list(self.corpus),
            bm25_index=bm25,
            scoring_lm=scoring_lm,
            scoring_tokenizer=scoring_tokenizer,
            candidate_set_size=self.candidate_set_size,
            n_positives=self.n_positives,
            n_negatives=self.n_negatives,
            demo_template=self.demo_template,
            batch_size=self.scoring_batch_size,
        )

        # release scoring LM if we own it
        if self._owned_scoring_lm:
            del self._scoring_lm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._scoring_lm = None
        self._owned_scoring_lm = False

        if not labeled_data:
            raise RuntimeError(
                "EPR Stage 1 produced no labeled examples. Check candidate_set_size / corpus size."
            )

        # Stage 2: train dense retriever
        config = self.contrastive_config or ContrastiveConfig(seed=self.seed)
        input_encoder, prompt_encoder = train_contrastive_retriever(
            labeled_data=labeled_data,
            base_encoder_name_or_path=self.base_encoder_name_or_path,
            pooling=self.encoder_pooling,
            config=config,
            device=self._resolve_device(),
            demo_template=self.demo_template,
        )

        # Stage 3: embed corpus with the trained prompt encoder
        demo_texts = [self.demo_template.format(**row) for row in self.corpus]
        embeddings = prompt_encoder.embed(demo_texts)

        self.memory = RetrievalMemory(
            corpus=list(self.corpus),
            mode="epr",
            dense_embeddings=embeddings,
            input_encoder_name_or_path=None,
            prompt_encoder_name_or_path=None,
            encoder_pooling=self.encoder_pooling,
            demo_template=self.demo_template,
            demo_separator=self.demo_separator,
        )
        self._input_encoder = input_encoder
        self._prompt_encoder = prompt_encoder

    # retrieval

    def _retrieve(self, query_text: str, k: int) -> list[int]:
        if self.memory is None:
            raise RuntimeError("EPR memory is not populated.")
        if self.memory.mode == "bm25":
            return bm25_search(self.memory.bm25_state, query_text, k=k)

        if self._input_encoder is None:
            raise RuntimeError(
                "EPR input encoder is not loaded. If you reloaded a saved memory, re-attach the encoder before "
                "calling adapt()."
            )
        q_emb = self._input_encoder.embed([query_text])[0]
        indices, _ = dense_top_k(
            q_emb,
            self.memory.dense_embeddings,
            k=k,
            use_faiss=self.use_faiss,
        )
        return list(int(i) for i in indices)

    # assembly

    def _assemble_prompt(self, demos: list[dict], query_text: str) -> str:
        demo_str = self.memory.demo_separator.join(
            self.memory.demo_template.format(**d) for d in demos
        )
        return self.final_template.format(
            demonstrations=demo_str,
            query=query_text,
        )

    # helpers

    def _resolve_scoring_lm(self, model: Any, tokenizer: Any) -> tuple[Any, Any]:
        """Three cases:

          1. `scoring_lm=None` → use task `model`/`tokenizer` from steer() (LM-as-a-service).
          2. `scoring_lm=str` → load HF causal LM (LM-as-a-proxy); we own it and release after Stage 1.
          3. `scoring_lm` is a `PreTrainedModel` or callable → use directly; we don't own it.
        """
        self._owned_scoring_lm = False

        if self.scoring_lm is None:
            if model is None:
                raise ValueError("EPR scoring LM resolution: no scoring_lm and no task model provided.")
            return model, self.scoring_tokenizer if self.scoring_tokenizer is not None else tokenizer

        if isinstance(self.scoring_lm, str):
            logger.info("EPR loading scoring LM %r", self.scoring_lm)
            lm = AutoModelForCausalLM.from_pretrained(self.scoring_lm, trust_remote_code=True)
            lm.to(self._resolve_device())
            tok = self.scoring_tokenizer
            if tok is None:
                tok = AutoTokenizer.from_pretrained(self.scoring_lm, trust_remote_code=True)
            elif isinstance(tok, str):
                tok = AutoTokenizer.from_pretrained(tok, trust_remote_code=True)
            self._owned_scoring_lm = True
            return lm, tok

        return self.scoring_lm, self.scoring_tokenizer

    def _resolve_device(self) -> str:
        if self.encoder_device is not None:
            return self.encoder_device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _decode_single(self, input_ids: list[int] | torch.Tensor) -> str:
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.reshape(-1).tolist()
        elif input_ids and isinstance(input_ids[0], list):
            ids_list = list(input_ids[0])
        else:
            ids_list = list(input_ids)
        return self.tokenizer.decode(ids_list, skip_special_tokens=True)

    def _encode_to_ids(
        self,
        prompt: str,
        original_input_ids: list[int] | torch.Tensor,
    ) -> list[int] | torch.Tensor:
        is_tensor = isinstance(original_input_ids, torch.Tensor)
        if self.max_prompt_tokens is not None:
            tokens = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=int(self.max_prompt_tokens),
            )
        else:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        if is_tensor:
            device = original_input_ids.device
            dtype = original_input_ids.dtype
            tensor_out = torch.tensor(tokens, dtype=dtype, device=device)
            if original_input_ids.ndim == 1:
                return tensor_out
            return tensor_out.unsqueeze(0)

        if original_input_ids and isinstance(original_input_ids[0], list):
            return [tokens]
        return tokens

    # for testing / loading-from-memory workflows

    def attach_encoders(
        self,
        input_encoder: HFEncoder,
        prompt_encoder: HFEncoder | None = None,
    ) -> None:
        """Attach pre-built encoders (for use after loading a saved `RetrievalMemory`).

        If `prompt_encoder` is None, reuses `input_encoder` for both sides.
        """
        self._input_encoder = input_encoder
        self._prompt_encoder = prompt_encoder if prompt_encoder is not None else input_encoder
