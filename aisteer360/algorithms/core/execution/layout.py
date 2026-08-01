"""Structural model facts available on every backend as session contract."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelLayout:
    """Structural facts about the pipeline model, available without a live module tree.

    Layer indices are the canonical coordinates for steer-phase layer selection; module names
    are an in-process serialization detail resolved only at hook construction time. Client-side
    tensor preparation uses the layout's dtype, and device placement is handled in process or by
    the worker rather than at steer time.

    This type is distinct from `aisteer360.algorithms.state_control._common.model_layout
    .ModelLayout`, which names architecture-specific module paths for hook construction.

    Attributes:
        num_layers: Number of decoder layers.
        hidden_size: Residual-stream width.
        num_attention_heads: Number of attention heads, or None when the model config does not
            state one.
        head_dim: Per-head dimension (the config's value, else `hidden_size` divided by
            `num_attention_heads`), or None when neither is derivable.
        dtype: Canonical dtype string, e.g. `"bfloat16"`.
        model_fingerprint: A 16-character hex digest identifying the model weights and config.
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int | None
    head_dim: int | None
    dtype: str
    model_fingerprint: str
