"""Stage 1 of EPR: LM-conditional candidate scoring + contrastive data labeling.

For each anchor `(x, y)` in the corpus, retrieve a candidate set via the unsupervised BM25 index, score each
candidate `c` by `P(y | demo(c), x)` under a scoring LM, and label the top-k candidates as positives and the
bottom-k as negatives for downstream contrastive training.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np
import torch

from aisteer360.algorithms.input_control.epr.retrieval import bm25_search

logger = logging.getLogger(__name__)


def score_candidates_with_lm(
    anchor_input: str,
    anchor_output: str,
    candidates: list[dict],
    scoring_lm,
    scoring_tokenizer,
    demo_template: str = "Input: {input}\nOutput: {output}",
    batch_size: int = 8,
) -> np.ndarray:
    """Compute `log P(anchor_output | demo(c), anchor_input)` for each candidate `c`.

    Implementation: teacher-forced forward pass — concatenate `[demo(c), anchor_input, anchor_output]`, extract logits
    at the positions corresponding to the `anchor_output` tokens, and sum log-probs.

    Returns a numpy array of length `len(candidates)`. Higher = better candidate.

    `scoring_lm` may be either a HuggingFace `PreTrainedModel` (in which case `scoring_tokenizer` is used) or any
    callable matching the signature `(anchor_input, anchor_output, candidates, demo_template) -> np.ndarray` (used for
    test stubs).

    Args:
        anchor_input: The training example's input string.
        anchor_output: The training example's output string.
        candidates: Each dict has `"input"` and `"output"`.
        scoring_lm: HF causal LM or scoring callable.
        scoring_tokenizer: HF tokenizer (ignored if `scoring_lm` is a callable).
        demo_template: Template applied to each candidate.
        batch_size: Forward-pass batch size when `scoring_lm` is an HF model.

    Returns:
        `[len(candidates)]` float array of summed log-probabilities.
    """
    if not candidates:
        return np.zeros((0,), dtype=np.float32)

    if callable(scoring_lm) and not isinstance(scoring_lm, torch.nn.Module):
        scores = scoring_lm(
            anchor_input=anchor_input,
            anchor_output=anchor_output,
            candidates=candidates,
            demo_template=demo_template,
        )
        return np.asarray(scores, dtype=np.float32)

    return _score_candidates_hf(
        anchor_input=anchor_input,
        anchor_output=anchor_output,
        candidates=candidates,
        scoring_lm=scoring_lm,
        scoring_tokenizer=scoring_tokenizer,
        demo_template=demo_template,
        batch_size=batch_size,
    )


def _score_candidates_hf(
    anchor_input: str,
    anchor_output: str,
    candidates: list[dict],
    scoring_lm,
    scoring_tokenizer,
    demo_template: str,
    batch_size: int,
) -> np.ndarray:
    if scoring_tokenizer is None:
        raise ValueError("scoring_tokenizer must be provided when scoring_lm is an HF model.")

    if scoring_tokenizer.pad_token_id is None:
        if scoring_tokenizer.eos_token_id is not None:
            scoring_tokenizer.pad_token = scoring_tokenizer.eos_token
        else:
            scoring_tokenizer.add_special_tokens({"pad_token": "<pad>"})

    device = next(scoring_lm.parameters()).device
    pad_id = scoring_tokenizer.pad_token_id

    output_ids_list = scoring_tokenizer(
        anchor_output, add_special_tokens=False, return_tensors=None
    )["input_ids"]
    if not output_ids_list:
        # nothing to score against; return uninformative zeros
        return np.zeros((len(candidates),), dtype=np.float32)
    output_len = len(output_ids_list)

    scores = np.zeros((len(candidates),), dtype=np.float32)

    was_training = scoring_lm.training
    scoring_lm.eval()

    try:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            prefix_ids: list[list[int]] = []
            full_ids: list[list[int]] = []
            for cand in batch:
                demo_text = demo_template.format(**cand)
                prefix_text = f"{demo_text}\n\n{anchor_input}\n"
                prefix = scoring_tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                prefix_ids.append(prefix)
                full_ids.append(prefix + output_ids_list)

            max_len = max(len(seq) for seq in full_ids)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            for i, seq in enumerate(full_ids):
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                attention_mask[i, :len(seq)] = 1

            with torch.no_grad():
                model_out = scoring_lm(input_ids=input_ids, attention_mask=attention_mask)
            logits = model_out.logits

            log_probs = torch.log_softmax(logits, dim=-1)
            for i in range(len(batch)):
                p_len = len(prefix_ids[i])
                # logits at positions [p_len-1 .. p_len+output_len-2] predict tokens [p_len .. p_len+output_len-1]
                start_pos = p_len - 1
                end_pos = p_len - 1 + output_len
                token_targets = full_ids[i][p_len:p_len + output_len]
                if start_pos < 0 or end_pos > log_probs.shape[1]:
                    scores[start + i] = float("-inf")
                    continue
                gathered = log_probs[i, start_pos:end_pos, :]
                target = torch.tensor(token_targets, dtype=torch.long, device=device)
                token_lp = gathered.gather(1, target.unsqueeze(1)).squeeze(1)
                scores[start + i] = float(token_lp.sum().item())
    finally:
        if was_training:
            scoring_lm.train()

    return scores


def generate_contrastive_data(
    corpus: list[dict],
    bm25_index: dict,
    scoring_lm,
    scoring_tokenizer,
    candidate_set_size: int = 50,
    n_positives: int = 5,
    n_negatives: int = 5,
    demo_template: str = "Input: {input}\nOutput: {output}",
    batch_size: int = 8,
    progress: Callable[[Iterable], Iterable] | None = None,
) -> list[dict]:
    """For each example in the corpus, build a labeled training instance.

    Steps per anchor:

      1. BM25 retrieve top-`candidate_set_size + 1` candidates (skip the anchor itself if it appears).
      2. Score each candidate via the scoring LM.
      3. Take the top-`n_positives` (highest score) as positives, bottom-`n_negatives` (lowest score) as negatives.

    Args:
        corpus: List of `{"input": str, "output": str}` dicts.
        bm25_index: Output of `build_bm25_index` (over corpus outputs, per paper convention).
        scoring_lm: HF causal LM or scoring callable.
        scoring_tokenizer: HF tokenizer (or None when `scoring_lm` is a callable).
        candidate_set_size: L in the paper.
        n_positives: k_top in the paper.
        n_negatives: k_bottom in the paper.
        demo_template: Template applied to each demo when constructing the scoring prompt.
        batch_size: Forward-pass batch size.
        progress: Optional progress wrapper (e.g. `tqdm`).

    Returns:
        List of training instances of the form
        `{"anchor": {...}, "positives": [...], "negatives": [...]}`.
    """
    if candidate_set_size < n_positives + n_negatives:
        raise ValueError(
            "candidate_set_size must be >= n_positives + n_negatives "
            f"(got {candidate_set_size} < {n_positives} + {n_negatives})"
        )

    iterator: Iterable = range(len(corpus))
    if progress is not None:
        iterator = progress(iterator)

    labeled: list[dict] = []
    for i in iterator:
        anchor = corpus[i]
        # retrieve one extra in case the anchor is in its own top-K
        cand_indices = bm25_search(bm25_index, anchor["output"], k=candidate_set_size + 1)
        cand_indices = [j for j in cand_indices if j != i][:candidate_set_size]
        if len(cand_indices) < n_positives + n_negatives:
            logger.debug(
                "Anchor %d has fewer candidates (%d) than n_positives + n_negatives (%d); skipping.",
                i, len(cand_indices), n_positives + n_negatives,
            )
            continue
        candidates = [corpus[j] for j in cand_indices]

        scores = score_candidates_with_lm(
            anchor_input=anchor["input"],
            anchor_output=anchor["output"],
            candidates=candidates,
            scoring_lm=scoring_lm,
            scoring_tokenizer=scoring_tokenizer,
            demo_template=demo_template,
            batch_size=batch_size,
        )

        order = np.argsort(-scores)
        pos = [candidates[idx] for idx in order[:n_positives]]
        neg = [candidates[idx] for idx in order[-n_negatives:][::-1]]

        labeled.append({"anchor": anchor, "positives": pos, "negatives": neg})

    return labeled
