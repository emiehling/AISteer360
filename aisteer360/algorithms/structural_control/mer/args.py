from dataclasses import dataclass, field
from typing import Any, Literal

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class MERArgs(BaseArgs):

    train_dataset: Any = field(
        default=None,
        metadata={
            "help": (
                "Dataset of new (sequential) training examples. Must be an iterable of dicts with 'input_ids' and "
                "optionally 'attention_mask' keys (pre-tokenized). Use HuggingFace datasets with tokenizer.map() to "
                "prepare."
            )
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of passes over `train_dataset`."},
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": (
                "Hard limit on total micro-steps (forward passes) processed. -1 means no limit. "
                "NOTE: This differs from HuggingFace convention where max_steps refers to optimizer steps. "
                "To limit optimizer steps, set max_steps = desired_optimizer_steps * gradient_accumulation_steps."
            )
        },
    )
    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": "Effective batch size per optimization step (new + replay samples combined)."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of forward/backward passes to accumulate before each optimizer step."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Peak learning rate for AdamW optimizer."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay (L2 penalty) for AdamW optimizer."},
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for linear LR scheduler (in optimizer steps)."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping. Set to 0 to disable."},
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log training metrics every N micro-steps. Set to 0 to disable."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for replay sampling and shuffling."},
    )

    # replay configuration
    enable_replay: bool = field(
        default=True,
        metadata={"help": "Enable experience replay. If False, MER reduces to Reptile/KL-only training."},
    )
    replay_buffer_size: int = field(
        default=10_000,
        metadata={
            "help": (
                "Maximum number of examples stored in the replay buffer. "
                "Memory usage scales with buffer_size * avg_sequence_length * dtype_size."
            )
        },
    )
    replay_ratio: float = field(
        default=0.25,
        metadata={
            "help": (
                "Fraction alpha of each batch drawn from replay buffer. "
                "New examples per batch = batch_size * (1 - alpha), replay = batch_size * alpha."
            )
        },
    )
    replay_start_step: int = field(
        default=0,
        metadata={"help": "Micro-steps of warmup with pure new data before enabling replay mixing."},
    )
    reservoir_sampling: bool = field(
        default=True,
        metadata={
            "help": (
                "Use reservoir sampling for buffer updates (uniform over all seen examples). "
                "If False, uses FIFO replacement (recent examples only)."
            )
        },
    )

    # KL regularization configuration
    enable_kl: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable KL regularization against a frozen teacher model. "
                "WARNING: Doubles memory usage if teacher is cloned from student."
            )
        },
    )
    kl_teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "HuggingFace model ID or path for frozen teacher. If None and KL is enabled, "
                "a frozen copy of the initial model is used (doubles GPU memory)."
            )
        },
    )
    kl_weight: float = field(
        default=1.0,
        metadata={"help": "Coefficient lambda on the KL regularization term: loss = CE + lambda * KL."},
    )
    kl_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature T for softening distributions in KL computation."},
    )
    kl_apply_on: Literal["all", "replay", "new"] = field(
        default="all",
        metadata={
            "help": (
                "Which batch rows to compute KL over: 'all' tokens, only 'replay' examples, "
                "or only 'new' examples. KL is normalized per selected token for comparability."
            )
        },
    )

    # reptile configuration
    enable_reptile: bool = field(
        default=True,
        metadata={"help": "Enable Reptile-style meta-updates for improved generalization."},
    )
    reptile_meta_lr: float = field(
        default=0.1,
        metadata={
            "help": (
                "Interpolation coefficient epsilon for Reptile: "
                "theta <- theta_anchor + epsilon * (theta_current - theta_anchor). "
                "Higher values = faster adaptation, lower = more stable."
            )
        },
    )
    reptile_update_interval: int = field(
        default=100,
        metadata={"help": "Number of optimizer steps between Reptile meta-updates."},
    )
    reptile_reset_per_epoch: bool = field(
        default=False,
        metadata={"help": "Reset Reptile anchor at the start of each epoch."},
    )
    reptile_offload_anchor: bool = field(
        default=False,
        metadata={
            "help": (
                "Store Reptile anchor parameters on CPU to save GPU memory. "
                "Recommended for models >2B parameters."
            )
        },
    )

    # buffer implementation selection
    buffer_type: Literal["memory", "disk"] = field(
        default="memory",
        metadata={
            "help": (
                "Replay buffer implementation: 'memory' for in-memory (fast, limited by RAM), "
                "'disk' for disk-backed with async prefetch (slower, larger capacity). "
                "For Megatron-scale training, see https://github.com/chandar-lab/continual-pretraining"
            )
        },
    )
    buffer_storage_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Directory for disk-backed buffer storage. If None, uses a temp directory. "
                "Set this for training resumption support."
            )
        },
    )
    buffer_cache_size: int = field(
        default=1000,
        metadata={"help": "Prefetch cache size for disk-backed buffer."},
    )

    # validation
    def __post_init__(self) -> None:
        if self.train_dataset is None:
            raise ValueError("MERArgs.train_dataset must be provided.")

        # dataset structure
        if hasattr(self.train_dataset, "__getitem__"):
            try:
                sample = self.train_dataset[0]
                if not isinstance(sample, dict):
                    raise ValueError(
                        f"train_dataset items must be dicts, got {type(sample).__name__}."
                    )
                if "input_ids" not in sample:
                    raise ValueError(
                        "train_dataset items must contain 'input_ids' key. "
                        "Pre-tokenize your dataset before passing to MER."
                    )
            except (IndexError, TypeError):
                pass  # empty or non-indexable dataset; will fail at runtime

        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive.")

        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be in [0.0, 1.0].")

        if self.replay_buffer_size < 0:
            raise ValueError("replay_buffer_size must be non-negative.")

        if self.enable_reptile:
            if self.reptile_meta_lr <= 0.0:
                raise ValueError("reptile_meta_lr must be positive when Reptile is enabled.")
            if self.reptile_update_interval <= 0:
                raise ValueError("reptile_update_interval must be positive when Reptile is enabled.")

        if self.enable_kl:
            if self.kl_weight <= 0.0:
                raise ValueError("kl_weight must be positive when KL is enabled.")
            if self.kl_temperature <= 0.0:
                raise ValueError("kl_temperature must be positive when KL is enabled.")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")

        if self.buffer_cache_size <= 0:
            raise ValueError("buffer_cache_size must be positive.")
