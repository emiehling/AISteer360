"""Token scope utilities for computing position masks."""
from typing import Literal

import torch

TokenScope = Literal["all", "after_prompt", "last_k"]


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
) -> torch.BoolTensor:
    """Build a [B, T] boolean mask selecting which tokens to transform.

    Args:
        scope: Which tokens to include.
            "all" - every position is True.
            "after_prompt" - only positions beyond the prompt are True
                (for autoregressive generation where T > prompt_len).
            "last_k" - only the last k positions are True.
        seq_len: Current sequence length T.
        prompt_lens: Shape [B], per-item prompt lengths.
        last_k: Required when scope == "last_k".

    Returns:
        Boolean tensor of shape [B, T].
    """
    B = prompt_lens.size(0)
    device = prompt_lens.device
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

    if scope == "all":
        return torch.ones(B, seq_len, dtype=torch.bool, device=device)
    elif scope == "after_prompt":
        return positions >= prompt_lens.unsqueeze(1)
    elif scope == "last_k":
        if last_k is None or last_k < 1:
            raise ValueError("last_k must be >= 1 when scope is 'last_k'.")
        return positions >= (seq_len - last_k)
    else:
        raise ValueError(f"Unknown token scope: {scope!r}")
