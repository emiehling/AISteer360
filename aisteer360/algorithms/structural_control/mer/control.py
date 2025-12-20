import math
import random
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.algorithms.structural_control.mer.args import MERArgs


class _ReplayBuffer:
    """Simple in‑memory replay buffer with reservoir or FIFO replacement."""

    def __init__(self, capacity: int, reservoir: bool = True) -> None:
        self.capacity = int(capacity)
        self.reservoir = reservoir
        self._examples: list[dict[str, Any]] = []
        self._total_seen: int = 0

    def __len__(self) -> int:
        return len(self._examples)

    def add_many(self, examples: Sequence[dict[str, Any]]) -> None:
        for example in examples:
            self.add(example)

    def add(self, example: dict[str, Any]) -> None:
        if self.capacity == 0:
            return

        self._total_seen += 1
        if len(self._examples) < self.capacity:
            # still filling
            self._examples.append(example)
            return

        if self.reservoir:
            # reservoir sampling
            replacement_idx = random.randint(0, self._total_seen - 1)
            if replacement_idx < self.capacity:
                self._examples[replacement_idx] = example
        else:
            # FIFO
            self._examples.pop(0)
            self._examples.append(example)

    def sample(self, num_samples: int) -> list[dict[str, Any]]:
        if not self._examples or num_samples <= 0:
            return []
        num_samples = min(int(num_samples), len(self._examples))
        return random.sample(self._examples, num_samples)


class MER(StructuralControl):
    """Meta‑Experience Replay (MER).

    todo (@Matt): refine/add details here and reference(s) (see other docstrings in the toolkit for inspiration)

    - Experience replay: maintain a buffer of past examples and mix them into each batch at a specified ratio alpha
    - KL regularization: optional KL(p_teacher || p_student) term to stabilize updates against a frozen teacher
    - Reptile meta-updates: every k optimizer steps, interpolate parameters between a snapshot theta_0 and current theta

    """

    Args = MERArgs

    # placeholders attached in steer()
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    device: torch.device | str | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._teacher: PreTrainedModel | None = None  # for KL

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Run MER training and return the updated model instance."""

        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(
                "MER requires a tokenizer (either passed explicitly or attached to the model as `model.tokenizer`)."
            )
        self.device = next(model.parameters()).device

        # # seeding for replay sampling etc.
        # random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(self.seed)

        # optional teacher for KL regularization
        if self.enable_kl:
            self._init_teacher()

        # data loader over new data (replay is mixed in inside the loop)
        new_batch_size = self._compute_new_batch_size()
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=new_batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,  # list[dict]; let LM collator do tensorization
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # optimizer & LR scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps_estimate = self._estimate_total_steps(len(data_loader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps_estimate,
        )

        # replay buffer (todo: disk-backed version later?)
        replay_buffer = _ReplayBuffer(
            capacity=self.replay_buffer_size,
            reservoir=self.reservoir_sampling,
        )

        # Reptile meta-state
        anchor_params: list[torch.Tensor] | None = None
        steps_since_anchor = 0
        global_step = 0  # micro-steps (batches)

        self.model.train()

        for epoch in range(self.num_train_epochs):
            for batch_examples in data_loader:
                # batch_examples is list[dict] from collate_fn=lambda batch: batch
                if isinstance(batch_examples, dict):
                    new_examples = [batch_examples]
                else:
                    new_examples = list(batch_examples)

                # update replay buffer with new data
                if self.enable_replay:
                    replay_buffer.add_many(new_examples)

                # build mixed batch (new + replay)
                mixed_examples = self._build_mixed_batch(
                    new_examples=new_examples,
                    replay_buffer=replay_buffer,
                    current_step=global_step,
                )
                if not mixed_examples:
                    continue

                # collate and move to device
                batch = collator(mixed_examples)
                batch = {key: tensor.to(self.device) for key, tensor in batch.items()}

                # forward pass (LM loss)
                outputs = self.model(**batch)
                if outputs.loss is None:
                    raise RuntimeError("Model did not return a loss; ensure the collator provides `labels`.")
                loss = outputs.loss

                # optional KL regularization
                if self.enable_kl:
                    kl_loss = self._compute_kl_loss(
                        batch=batch,
                        student_logits=outputs.logits,
                        new_examples=new_examples,
                    )
                    loss = loss + self.kl_weight * kl_loss

                # gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    # gradient clipping
                    if self.max_grad_norm and self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Reptile meta-update
                    if self.enable_reptile:
                        if anchor_params is None:
                            anchor_params = self._snapshot_model()
                            steps_since_anchor = 0

                        steps_since_anchor += 1
                        if steps_since_anchor >= self.reptile_update_interval:
                            self._apply_reptile_update(anchor_params)
                            anchor_params = self._snapshot_model()
                            steps_since_anchor = 0

                global_step += 1

                # logging
                if self.logging_steps and global_step % self.logging_steps == 0:
                    suffix = " + KL" if self.enable_kl else ""
                    print(
                        f"[MER] step={global_step} epoch={epoch} loss={loss.item():.4f}{suffix}",
                        flush=True,
                    )

                # max_steps micro-step limit
                if 0 < self.max_steps <= global_step:
                    break

            if 0 < self.max_steps <= global_step:
                break

        self.model.eval()
        return self.model

    # data / batching helpers
    def _compute_new_batch_size(self) -> int:
        """Number of new examples per micro-step.

        Total effective batch size is `per_device_train_batch_size`; alpha fraction `replay_ratio` of that is allocated to replay (if enabled).
        """
        if not self.enable_replay or self.replay_ratio <= 0.0:
            return self.per_device_train_batch_size

        new_batch_size = int(round(self.per_device_train_batch_size * (1.0 - self.replay_ratio)))
        return max(new_batch_size, 1)

    def _estimate_total_steps(self, batches_per_epoch: int) -> int:
        """Rough step count for scheduler construction."""
        if batches_per_epoch == 0:
            return 1
        total_steps = batches_per_epoch * self.num_train_epochs
        if self.max_steps > 0:
            total_steps = min(total_steps, self.max_steps)

        # convert micro-steps to optimizer steps
        total_steps = math.ceil(total_steps / max(1, self.gradient_accumulation_steps))
        return max(total_steps, 1)

    def _build_mixed_batch(
        self,
        new_examples: Sequence[dict[str, Any]],
        replay_buffer: _ReplayBuffer,
        current_step: int,
    ) -> list[dict[str, Any]]:
        """Return a list of examples mixing new data and replay samples."""
        batch: list[dict[str, Any]] = list(new_examples)

        if not self.enable_replay:
            return batch

        # warmup period (new data)
        if current_step < self.replay_start_step:
            return batch

        if self.replay_ratio <= 0.0 or len(replay_buffer) == 0:
            return batch

        target_batch_size = self.per_device_train_batch_size
        num_new = len(new_examples)
        num_replay = max(target_batch_size - num_new, 0)
        if num_replay == 0:
            return batch

        replay_samples = replay_buffer.sample(num_replay)
        if replay_samples:
            batch.extend(replay_samples)

        return batch

    # teacher / KL helpers
    def _init_teacher(self) -> None:
        """Instantiate frozen teacher model for KL regularization.

        If `kl_teacher_model_name_or_path` is provided, load a separate teacher; otherwise clone initial model weights.
        """
        if self.kl_teacher_model_name_or_path:
            teacher = AutoModelForCausalLM.from_pretrained(
                self.kl_teacher_model_name_or_path,
                trust_remote_code=getattr(self.model.config, "trust_remote_code", False),
            )
        else:
            # todo: for large models pass a separate teacher path instead of cloning
            teacher = AutoModelForCausalLM.from_config(self.model.config)
            teacher.load_state_dict(self.model.state_dict())

        teacher.to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self._teacher = teacher

    def _compute_kl_loss(
        self,
        batch: dict[str, torch.Tensor],
        student_logits: torch.Tensor,
        new_examples: Sequence[dict[str, Any]],
    ) -> torch.Tensor:
        """KL(p_teacher || p_student) over token distributions.

        `kl_apply_on` controls whether we apply KL over all tokens, only those from replay examples, or only those from
        new examples.

        # todo: this is approximated by assuming new examples appear first in the batch (as constructed in `_build_mixed_batch`).
        """
        assert self._teacher is not None, "Teacher must be initialized when KL is enabled."

        with torch.no_grad():
            teacher_out = self._teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            teacher_logits = teacher_out.logits

        temperature = float(self.kl_temperature)

        # [B, L, V]
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        if self.kl_apply_on != "all":
            num_new = len(new_examples)
            batch_size = student_logits.size(0)
            row_mask = torch.zeros(batch_size, dtype=torch.bool, device=student_logits.device)

            if self.kl_apply_on == "new":
                row_mask[:num_new] = True
            else:  # replay
                if num_new < batch_size:
                    row_mask[num_new:] = True

            # broadcast to [B, L, V]
            row_mask = row_mask.view(batch_size, 1, 1)
            student_log_probs = student_log_probs * row_mask
            teacher_probs = teacher_probs * row_mask

        # forward KL (sum p_teacher * (log p_teacher - log p_student))
        kl_divergence = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
            log_target=False,
        ) * (temperature * temperature)

        return kl_divergence

    # Reptile helpers
    def _snapshot_model(self) -> list[torch.Tensor]:
        """Clone current trainable parameters."""
        return [param.detach().clone() for param in self.model.parameters() if param.requires_grad]

    def _apply_reptile_update(self, anchor_params: list[torch.Tensor]) -> None:
        """Reptile meta-update"""
        with torch.no_grad():
            param_idx = 0
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                anchor_param = anchor_params[param_idx]
                param.data = anchor_param + self.reptile_meta_lr * (param.data - anchor_param)
                param_idx += 1
