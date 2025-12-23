from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class MERArgs(BaseArgs):

    # data
    train_dataset: Any = field(
        default=None,
        metadata={"help": "Supervised fine-tuning dataset (HF Dataset or compatible)."}
    )
    replay_dataset: Any | None = field(
        default=None,
        metadata={"help": "Dataset for replay."}
    )

    # replay
    replay_enabled: bool = field(
        default=False,
        metadata={"help": "Enable approximate replay by mixing replay_dataset into training."}
    )
    replay_rate: float = field(
        default=0.0,
        metadata={"help": "Replay rate rho: replay examples per task example."},
    )
    replay_seed: int = field(
        default=42,
        metadata={"help": "Random seed for sampling/shuffling replay examples."}
    )

    # KL reg
    kl_enabled: bool = field(
        default=False,
        metadata={"help": "Enable KL divergence regularization against base model."}
    )
    kl_beta: float = field(
        default=0.01,
        metadata={"help": "KL coefficient beta."}
    )

    # reptile
    reptile_enabled: bool = field(
        default=False,
        metadata={"help": "Enable Reptile-style meta-updates for gradient alignment."}
    )
    reptile_steps: int = field(
        default=500,
        metadata={"help": "Optimizer steps between Reptile meta-updates."}
    )
    reptile_lr: float = field(
        default=0.1,
        metadata={"help": "Reptile interpolation coefficient eta in (0, 1]."}
    )

    # training hyperparameters
    output_dir: str = field(
        default="mer_output",
        metadata={"help": "Output directory for checkpoints and logs."}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device."}
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Number of training epochs."}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate (paper uses constant LR)."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps."}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log every N steps."}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Pack multiple short sequences into one."}
    )
    save_strategy: str = field(
        default="no",
        metadata={"help": "Checkpoint save strategy ('no', 'epoch', 'steps')."}
    )
    report_to: str = field(
        default="none",
        metadata={"help": "Reporting destination ('none', 'wandb', 'tensorboard')."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )

    # LoRA
    use_peft: bool = field(
        default=True,
        metadata={"help": "Use LoRA adapter."}
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha (scaling = alpha/r)."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate."}
    )
    lora_target_modules: Sequence[str] | None = field(
        default=None,
        metadata={"help": "Modules to apply LoRA to (None = auto-detect)."}
    )
    merge_lora_after_train: bool = field(
        default=False,
        metadata={"help": "Merge LoRA weights into base model after training."}
    )

    # validation
    def __post_init__(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided.")

        if self.replay_enabled:
            if self.replay_rate <= 0.0:
                raise ValueError("replay_rate must be > 0 when replay_enabled=True.")
            if self.replay_dataset is None:
                raise ValueError("replay_dataset must be provided when replay_enabled=True.")

        if self.kl_enabled and self.kl_beta <= 0:
            raise ValueError("kl_beta must be > 0 when kl_enabled=True.")

        if self.reptile_enabled:
            if self.reptile_steps <= 0:
                raise ValueError("reptile_steps must be > 0 when reptile_enabled=True.")
            if not (0.0 < self.reptile_lr <= 1.0):
                raise ValueError("reptile_lr must be in (0, 1] when reptile_enabled=True.")
