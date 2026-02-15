"""Mean difference estimator for CAA steering vectors."""
import logging
from typing import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.common.estimators.base import BaseEstimator
from aisteer360.algorithms.state_control.common.estimators.contrastive_direction_estimator import (
    _layerwise_tokenwise_hidden,
)
from aisteer360.algorithms.state_control.common.specs import ContrastivePairs, VectorTrainSpec
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector

logger = logging.getLogger(__name__)


def _tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pos_texts: Sequence[str],
    neg_texts: Sequence[str],
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Tokenize positive/negative pairs together to ensure consistent padding.

    Interleaves pairs before tokenization so each (pos, neg) pair shares the same
    padding length. This ensures token alignment for shared prefixes, which is
    important because different padding can subtly change attention patterns.

    Args:
        tokenizer: Tokenizer to use.
        pos_texts: List of positive text strings.
        neg_texts: List of negative text strings (same length as pos_texts).
        device: Target device.

    Returns:
        Tuple of (enc_pos, enc_neg) dictionaries with input_ids and attention_mask.
    """
    # interleave: [pos0, neg0, pos1, neg1, ...]
    interleaved = []
    for pos, neg in zip(pos_texts, neg_texts):
        interleaved.append(pos)
        interleaved.append(neg)

    enc = tokenizer(
        interleaved,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # de-interleave: even indices are positive, odd indices are negative
    enc_pos = {k: v[0::2] for k, v in enc.items()}
    enc_neg = {k: v[1::2] for k, v in enc.items()}

    return enc_pos, enc_neg


def _get_last_token_positions(
    attention_mask: torch.Tensor | None,
    seq_len: int,
    num_samples: int,
) -> torch.LongTensor:
    """Find the last non-pad token position for each sample.

    Args:
        attention_mask: Shape [N, T] or None.
        seq_len: Sequence length T.
        num_samples: Number of samples N.

    Returns:
        Tensor of shape [N] with last token positions.
    """
    if attention_mask is None:
        # no padding, last token is at seq_len - 1
        return torch.full((num_samples,), seq_len - 1, dtype=torch.long)

    # for each sample, find the last position where attention_mask == 1
    # this handles both left-padded and right-padded sequences
    positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand(num_samples, -1)
    # mask out padded positions with -1
    masked_positions = torch.where(attention_mask == 1, positions, torch.tensor(-1, device=attention_mask.device))
    return masked_positions.max(dim=1).values


def _select_at_positions(
    hidden: torch.Tensor,
    positions: torch.LongTensor,
) -> torch.Tensor:
    """Select hidden states at specified positions for each sample.

    Args:
        hidden: Shape [N, T, H].
        positions: Shape [N] with position indices.

    Returns:
        Tensor of shape [N, H].
    """
    N, _, H = hidden.shape
    # gather at the specified positions
    idx = positions.view(N, 1, 1).expand(N, 1, H)
    return hidden.gather(dim=1, index=idx).squeeze(1)


class MeanDifferenceEstimator(BaseEstimator[SteeringVector]):
    """Learns per-layer steering vectors using the Mean Difference method.

    For each layer, computes:
        v_L = mean(a_L(positive) - a_L(negative))

    where activations are extracted at the last non-pad token position
    of each example (the answer letter in the CAA prompt format).

    This differs from ContrastiveDirectionEstimator which uses PCA on the
    pairwise differences. Mean Difference takes the centroid directly, while
    PCA finds the direction of maximum variance. They converge when difference
    vectors are nearly collinear.
    """

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: ContrastivePairs,
        spec: VectorTrainSpec,
    ) -> SteeringVector:
        """Extract steering vectors using mean difference.

        Args:
            model: Model to extract hidden states from.
            tokenizer: Tokenizer for encoding the contrastive pairs.
            data: The positive/negative text pairs.
            spec: Training configuration (method, accumulate, batch_size).

        Returns:
            SteeringVector with one direction per layer.
        """
        device = next(model.parameters()).device
        model_type = getattr(model.config, "model_type", "unknown")

        # build full texts
        if data.prompts is not None:
            pos_texts = [p + c for p, c in zip(data.prompts, data.positives)]
            neg_texts = [p + c for p, c in zip(data.prompts, data.negatives)]
        else:
            pos_texts = list(data.positives)
            neg_texts = list(data.negatives)

        logger.debug("Tokenizing %d positive and %d negative examples", len(pos_texts), len(neg_texts))

        # tokenize pairs together to ensure consistent padding and token alignment
        enc_pos, enc_neg = _tokenize_pairs(tokenizer, pos_texts, neg_texts, device)

        # extract hidden states
        logger.debug("Extracting hidden states with batch_size=%d", spec.batch_size)
        hs_pos = _layerwise_tokenwise_hidden(model, enc_pos, batch_size=spec.batch_size)
        hs_neg = _layerwise_tokenwise_hidden(model, enc_neg, batch_size=spec.batch_size)

        num_samples = len(pos_texts)
        num_layers = len(hs_pos)
        logger.debug("Computing mean difference directions for %d layers", num_layers)

        # determine how to aggregate hidden states based on accumulate mode
        directions: dict[int, torch.Tensor] = {}

        # get attention masks for position selection
        attn_pos = enc_pos.get("attention_mask")
        attn_neg = enc_neg.get("attention_mask")
        if attn_pos is not None:
            attn_pos = attn_pos.cpu()
        if attn_neg is not None:
            attn_neg = attn_neg.cpu()

        for layer_id in range(num_layers):
            hp = hs_pos[layer_id]  # [N, T, H]
            hn = hs_neg[layer_id]  # [N, T, H]

            if spec.accumulate == "last_token":
                # select activation at last non-pad token
                pos_positions = _get_last_token_positions(attn_pos, hp.size(1), num_samples)
                neg_positions = _get_last_token_positions(attn_neg, hn.size(1), num_samples)
                hp_agg = _select_at_positions(hp, pos_positions)  # [N, H]
                hn_agg = _select_at_positions(hn, neg_positions)  # [N, H]
            elif spec.accumulate == "all":
                # mean pool over all tokens
                hp_agg = hp.mean(dim=1)  # [N, H]
                hn_agg = hn.mean(dim=1)  # [N, H]
            else:
                raise ValueError(f"MeanDifferenceEstimator does not support accumulate='{spec.accumulate}'")

            # compute mean difference: v = mean(h_pos - h_neg)
            diffs = hp_agg - hn_agg  # [N, H]
            direction = diffs.mean(dim=0)  # [H]

            directions[layer_id] = direction.unsqueeze(0).to(dtype=torch.float32)  # [1, H]

        logger.debug("Finished fitting mean difference directions")
        return SteeringVector(
            model_type=model_type,
            directions=directions,
        )
