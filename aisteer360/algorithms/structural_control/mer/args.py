from dataclasses import dataclass, field
from typing import Any, Literal

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class MERArgs(BaseArgs):

    train_dataset: Any = field(
        default=None,
        metadata={
            "help": (
                "Dataset of new (sequential) training examples. Should be an iterable of dicts consumable by a "
                "HuggingFace data collator (e.g., pre-tokenized with 'input_ids' / 'attention_mask')."
            )
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of passes over `train_dataset`."},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "Optional hard limit on total micro-steps (batches) processed. -1 means no explicit limit."},
    )
    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": "Effective batch size per optimization step (new + replay samples)."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of forward/backward passes to accumulate before each optimizer step."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Base learning rate for AdamW optimizer."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW optimizer."},
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for linear LR scheduler."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping norm (0 disables clipping)."},
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Interval (in micro-steps) for logging progress."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed used for replay sampling etc."},
    )

    # replay configuration
    enable_replay: bool = field(
        default=True,
        metadata={"help": "Enable experience replay. If False, MER reduces to Reptile / KL-only training."},
    )
    replay_buffer_size: int = field(
        default=10_000,
        metadata={"help": "Max number of examples stored in replay buffer."},
    )
    replay_ratio: float = field(
        default=0.25,
        metadata={"help": "Fraction of each effective batch that should come from replay buffer."},
    )
    replay_start_step: int = field(
        default=0,
        metadata={"help": "Micro-steps to run before enabling replay (warmup with pure new data)."},
    )
    reservoir_sampling: bool = field(
        default=True,
        metadata={"help": "If True, maintain replay buffer via reservoir sampling; otherwise use FIFO replacement."},
    )

    # KL regularization configuration
    enable_kl: bool = field(
        default=False,
        metadata={"help": "Enable KL regularization against a frozen teacher."},
    )
    kl_teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional HF id / path for a frozen teacher model. If None and KL is enabled, a frozen copy of the "
                "initial model is used as teacher."
            )
        },
    )
    kl_weight: float = field(
        default=1.0,
        metadata={"help": "Weight λ on the KL regularization term."},
    )
    kl_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature T for soft targets in KL."},
    )
    kl_apply_on: Literal["all", "replay", "new"] = field(
        default="all",
        metadata={"help": "Where to apply KL: 'all' tokens, only 'replay' examples, or only 'new' examples."},
    )

    # Reptile configuration
    enable_reptile: bool = field(
        default=True,
        metadata={"help": "Enable Reptile-style meta-updates on top of standard gradient updates."},
    )
    reptile_meta_lr: float = field(
        default=0.1,
        metadata={"help": "Interpolation coefficient ε in Reptile meta-update."},
    )
    reptile_update_interval: int = field(
        default=100,
        metadata={"help": "Number of optimizer steps between Reptile updates."},
    )

    # validation
    def __post_init__(self) -> None:

        if self.train_dataset is None:
            raise ValueError("MERArgs.train_dataset must be provided.")

        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive.")

        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be in [0, 1].")

        if self.replay_buffer_size < 0:
            raise ValueError("replay_buffer_size must be non-negative.")

        if self.enable_reptile and self.reptile_meta_lr <= 0.0:
            raise ValueError("reptile_meta_lr must be strictly positive when Reptile is enabled.")

        if self.enable_reptile and self.reptile_update_interval <= 0:
            raise ValueError("reptile_update_interval must be > 0 when Reptile is enabled.")

        if self.enable_kl and self.kl_weight <= 0.0:
            raise ValueError("kl_weight must be strictly positive when KL regularization is enabled.")

        if self.kl_temperature <= 0.0:
            raise ValueError("kl_temperature must be strictly positive.")
