"""Hooked/forward capture of hidden states at module boundaries."""
from typing import Callable, Literal

import torch
from transformers import PreTrainedModel

HiddenStateLocation = Literal["layer_output", "layer_input"]


def capture_hidden(
    enc: dict[str, torch.Tensor],
    *,
    model: PreTrainedModel | None = None,
    session=None,
    batch_size: int = 8,
    on_batch: Callable[[], None] | None = None,
    location: HiddenStateLocation = "layer_output",
) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
    """Per-layer hidden states for `enc` through the in-process funnel or a capture session.

    With a live model (given directly, or reachable through an in-process session), the
    extraction runs `layerwise_tokenwise_hidden` with the caller's batch size and progress
    callback, preserving the in-process layout: the returned mask is `enc`'s own. With a
    remote capture-capable session, each row of `enc` becomes a token-id prompt (padding
    positions dropped) served by `session.capture` over every decoder layer; the returned
    tensors are right-padded to the batch's longest prompt and the returned mask describes
    that layout, so pooling must use the returned mask rather than `enc`'s.

    Args:
        enc: Tokenized input with `input_ids` and optionally `attention_mask`.
        model: A live model, used when no session provides one.
        session: A `SteeringSession`; in-process sessions expose their live model, remote
            sessions serve `capture`.
        batch_size: Forward batch size on the in-process path.
        on_batch: Per-batch progress callback on the in-process path.
        location: Which residual-stream boundary each layer key maps to.

    Returns:
        A `(hidden, attention_mask)` pair with `hidden[l]` of shape `[N, T, H]` on CPU.

    Raises:
        ValueError: If neither a live model nor a session is available.
    """
    live_model = model
    if live_model is None and session is not None:
        try:
            live_model = session.model
        except (AttributeError, RuntimeError):
            live_model = None
    if live_model is not None:
        hidden = layerwise_tokenwise_hidden(
            live_model, enc, batch_size=batch_size, on_batch=on_batch, location=location
        )
        attention_mask = enc.get("attention_mask")
        return hidden, attention_mask.cpu() if attention_mask is not None else None
    if session is None or not callable(getattr(session, "capture", None)):
        raise ValueError("Hidden-state extraction requires a live model or a capture-capable session.")

    from aisteer360.algorithms.core.execution.payloads import PreparedPrompt

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    prompts = [
        PreparedPrompt.from_token_ids(
            input_ids[index:index + 1],
            attention_mask[index:index + 1] if attention_mask is not None else None,
        )
        for index in range(input_ids.size(0))
    ]
    result = session.capture(
        prompts, layers=list(range(session.layout.num_layers)), mode="all_tokens", location=location
    )
    if on_batch is not None:
        on_batch()
    return dict(result.hidden), result.attention_mask


@torch.no_grad()
def layerwise_tokenwise_hidden(
    model: PreTrainedModel,
    enc: dict[str, torch.Tensor],
    batch_size: int = 8,
    on_batch: Callable[[], None] | None = None,
    *,
    location: HiddenStateLocation = "layer_output",
) -> dict[int, torch.Tensor]:
    """Extract per-layer hidden states for all tokens.

    `outputs.hidden_states` is a tuple of `num_layers + 1` tensors: index 0 is the embedding output
    (the input to layer 0) and index `i` is the output of layer `i - 1`.

    - `location="layer_output"`: key `l` maps to the output of layer `l` (`hidden_states[l + 1]`).
    - `location="layer_input"`: key `l` maps to the input of layer `l`, i.e. the output of layer
        `l - 1` (`hidden_states[l]`), the boundary a layer pre-hook observes.

    Args:
        model: The model to extract from.
        enc: Tokenized input with input_ids and attention_mask.
        batch_size: Batch size for forward passes.
        on_batch: Optional callable invoked after each batch finishes. Used by callers to surface
            progress to the UI.
        location: Which residual-stream boundary each layer key maps to.

    Returns:
        Dict mapping layer_id (`0 .. num_layers - 1`) to tensor of shape [N, T, H].

    Raises:
        ValueError: If `location` is unsupported or the number of mapped states does not equal the
            model's layer count.
    """
    if location not in ("layer_output", "layer_input"):
        raise ValueError(f"Unsupported hidden-state location: {location!r}.")

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    N = input_ids.size(0)

    # collect states per layer
    all_hidden: dict[int, list[torch.Tensor]] = {}
    num_layers: int | None = None

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_ids = input_ids[start:end]
        batch_mask = attention_mask[start:end] if attention_mask is not None else None

        outputs = model(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        num_layers = len(outputs.hidden_states) - 1
        layer_states = outputs.hidden_states[1:] if location == "layer_output" else outputs.hidden_states[:-1]
        for layer_idx, hs in enumerate(layer_states):
            all_hidden.setdefault(layer_idx, []).append(hs.cpu())

        if on_batch is not None:
            on_batch()

    result = {layer_idx: torch.cat(tensors, dim=0) for layer_idx, tensors in all_hidden.items()}

    if num_layers is not None and len(result) != num_layers:
        raise ValueError(
            f"Expected {num_layers} mapped hidden states for location={location!r}, got {len(result)}."
        )

    return result
