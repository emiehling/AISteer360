"""Arguments for the EPR input control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.input_control.epr.contrastive import ContrastiveConfig


@dataclass
class EPRArgs(BaseArgs):
    """Arguments for EPR (Efficient Prompt Retrieval).

    Required:
        corpus: Training (and serving) corpus. Each dict has keys `"input"` and `"output"`.

    Mode:
        mode: One of:

            - `"bm25"`: TF-IDF/BM25 sparse retrieval. No training; no encoder.
            - `"dense"`: Off-the-shelf dense encoder. No training.
            - `"epr"` (default): Train a dense retriever via LM-supervised contrastive learning per the paper.

    Retrieval at serve time:
        n_demonstrations: Number of demonstrations to retrieve and prepend per query.
        max_prompt_tokens: If set, truncate retrieved demos to fit. None = no truncation.
        use_faiss: Use FAISS for similarity search (must be installed). Default numpy.

    Encoder (dense / epr modes):
        base_encoder_name_or_path: HF model id or local path. For `dense` mode used directly; for `epr` mode used as
            starting weights for both encoders.
        encoder_pooling: `"cls"` or `"mean"`.
        encoder_max_length: Token truncation.
        encoder_batch_size: Batch size for encoding.
        encoder_device: Override torch device. None = auto (cuda if available, else cpu).

    EPR-specific:
        scoring_lm: Scoring LM for Stage 1. None = use the task model from `steer()`'s `model` argument
            (LM-as-a-service). String = HF identifier (LM-as-a-proxy; loaded by EPR). Otherwise a preloaded
            `PreTrainedModel` or a callable for test stubs.
        scoring_tokenizer: Tokenizer matching `scoring_lm`. None and `scoring_lm` is None ⇒ use the task tokenizer.
            None and `scoring_lm` is a string ⇒ load tokenizer from the same identifier.
        candidate_set_size: L in the paper. Top-L candidates per training example from BM25.
        n_positives: k_top in the paper.
        n_negatives: k_bottom in the paper.
        contrastive_config: Hyperparameters for the Stage 2 trainer. None ⇒ defaults.
        scoring_batch_size: Batch size for the Stage 1 scoring LM forward passes.

    Assembly templates:
        demo_template: Format string for each demonstration.
        demo_separator: Joins demonstrations.
        final_template: Format string with placeholders `{demonstrations}` and `{query}`.

    Other:
        seed: RNG seed.
    """

    corpus: list[dict] = field(default_factory=list)

    mode: Literal["bm25", "dense", "epr"] = "epr"

    n_demonstrations: int = 5
    max_prompt_tokens: int | None = None
    use_faiss: bool = False

    base_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_pooling: Literal["cls", "mean"] = "mean"
    encoder_max_length: int = 256
    encoder_batch_size: int = 32
    encoder_device: str | None = None

    scoring_lm: Any = None
    scoring_tokenizer: Any = None
    candidate_set_size: int = 50
    n_positives: int = 5
    n_negatives: int = 5
    contrastive_config: ContrastiveConfig | None = None
    scoring_batch_size: int = 8

    demo_template: str = "Input: {input}\nOutput: {output}"
    demo_separator: str = "\n\n"
    final_template: str = "{demonstrations}\n\nInput: {query}\nOutput:"

    seed: int = 0

    def __post_init__(self) -> None:
        if not self.corpus:
            raise ValueError("corpus must be non-empty")
        if self.mode not in ("bm25", "dense", "epr"):
            raise ValueError(f"mode must be 'bm25', 'dense', or 'epr'; got {self.mode!r}")
        if self.n_demonstrations < 1:
            raise ValueError("n_demonstrations must be >= 1")
        if self.candidate_set_size < self.n_positives + self.n_negatives:
            raise ValueError(
                "candidate_set_size must be >= n_positives + n_negatives "
                f"(got {self.candidate_set_size} < {self.n_positives} + {self.n_negatives})"
            )
        if self.n_positives < 1 or self.n_negatives < 1:
            raise ValueError("n_positives and n_negatives must be >= 1")
        for i, row in enumerate(self.corpus):
            if not isinstance(row, dict) or "input" not in row or "output" not in row:
                raise ValueError(
                    f"corpus[{i}] must have 'input' and 'output' keys; got {row!r}"
                )
