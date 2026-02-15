"""Token scope utilities for computing position masks."""
from typing import Literal

import torch

TokenScope = Literal["all", "after_prompt", "last_k", "from_position"]


def compute_prompt_lens(
    input_ids: torch.LongTensor,
    pad_token_id: int | None,
) -> torch.LongTensor:
    """Compute per-batch-item prompt lengths from input_ids.

    For left-padded sequences, the prompt length is the number of non-pad
    tokens. For sequences without padding, it equals seq_len.

    Args:
        input_ids: Shape [B, T] or [T].
        pad_token_id: The pad token id, or None if no padding.

    Returns:
        Tensor of shape [B] with prompt lengths.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if pad_token_id is None:
        return torch.full((input_ids.size(0),), input_ids.size(1), dtype=torch.long, device=input_ids.device)
    return (input_ids != pad_token_id).sum(dim=1)


def make_token_mask(
    scope: TokenScope,
    *,
    seq_len: int,
    prompt_lens: torch.LongTensor,
    last_k: int | None = None,
    from_position: int | None = None,
    position_offset: int = 0,
) -> torch.BoolTensor:
    """Build a [B, T] boolean mask selecting which tokens to transform.

    Args:
        scope: Which tokens to include.
            "all" - every position is True.
            "after_prompt" - only positions beyond the prompt are True.
                During autoregressive generation with KV cache, the hook may
                see only newly generated tokens (seq_len=1). Use position_offset
                to indicate the cumulative position in the full sequence.
            "last_k" - only the last k positions are True.
            "from_position" - positions >= from_position are True. Useful for
                single forward pass evaluation (e.g., logit scoring) where you
                want to steer from a specific point within the prompt.
        seq_len: Current sequence length T (may be 1 during KV-cached generation).
        prompt_lens: Shape [B], per-item prompt lengths.
        last_k: Required when scope == "last_k".
        from_position: Required when scope == "from_position". The absolute
            position from which to start steering (inclusive).
        position_offset: Cumulative position offset for KV-cached generation.
            When the model processes token N in the sequence but only passes
            a single token to the hook (seq_len=1), set position_offset=N so
            that "after_prompt" correctly identifies generated tokens.

    Returns:
        Boolean tensor of shape [B, T].
    """
    B = prompt_lens.size(0)
    device = prompt_lens.device

    # compute absolute positions in the full sequence
    local_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
    absolute_positions = local_positions + position_offset

    if scope == "all":
        return torch.ones(B, seq_len, dtype=torch.bool, device=device)
    elif scope == "after_prompt":
        return absolute_positions >= prompt_lens.unsqueeze(1)
    elif scope == "last_k":
        if last_k is None or last_k < 1:
            raise ValueError("last_k must be >= 1 when scope is 'last_k'.")
        # for last_k, we want the last k positions relative to current seq_len
        return local_positions >= (seq_len - last_k)
    elif scope == "from_position":
        if from_position is None or from_position < 0:
            raise ValueError("from_position must be >= 0 when scope is 'from_position'.")
        return absolute_positions >= from_position
    else:
        raise ValueError(f"Unknown token scope: {scope!r}")
