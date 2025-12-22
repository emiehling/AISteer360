import math
import random
import warnings
from collections.abc import Sequence
from pathlib import Path
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
from aisteer360.algorithms.structural_control.mer.utils.buffers import (
    DiskReplayBuffer,
    InMemoryReplayBuffer,
    ReplayBuffer,
)


class MER(StructuralControl):
    """Meta-Experience Replay (MER) structural control method.

    # todo: refine

    MER combines three techniques for continual learning:
    - Experience replay: Mix past examples into training batches
    - KL regularization: Preserve base model behavior via distillation
    - Reptile meta-updates: Interpolate toward task-agnostic parameters

    Two buffer implementations are available:
    - In-memory (default): for small experiments (<100K examples)
    - Disk-backed: Larger capacity with async prefetching (for bigger buffers)

    Args:
        train_dataset: Pre-tokenized dataset with 'input_ids' keys.
        enable_replay: Whether to use experience replay.
        enable_kl: Whether to add KL regularization.
        enable_reptile: Whether to apply Reptile meta-updates.
        See MERArgs for full parameter documentation.

    Reference:

    - "Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models"
    Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Benjamin Therien, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, Irina Rish
    https://arxiv.org/abs/2508.01908
    """

    Args = MERArgs

    # placeholders attached in steer()
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    device: torch.device | str | None = None
    _teacher: PreTrainedModel | None = None  # for KL reg

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase | None = None,
            **__,
    ) -> PreTrainedModel:
        """Run MER training and return the updated model.

        Args:
            model: The model to train.
            tokenizer: Tokenizer for the model. Required for data collation.

        Returns:
            The trained model (same instance, modified in-place).

        Raises:
            ValueError: If tokenizer is not provided or discoverable.
            RuntimeError: If model does not return loss during training.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(
                "MER requires a tokenizer. Pass it explicitly or attach as model.tokenizer."
            )
        self.device = next(model.parameters()).device

        # ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # # seed
        # random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(self.seed)

        # initialize teacher for KL regularization
        if self.enable_kl:
            self._init_teacher()

        # data loader (replay mixed in during iteration)
        new_batch_size = self._compute_new_batch_size()
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=new_batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,  # keep as list[dict] for replay mixing
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = self._estimate_total_steps(len(data_loader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        # initialize replay buffer based on configuration
        replay_buffer: ReplayBuffer | None = None
        if self.enable_replay:
            replay_buffer = self._create_replay_buffer()
            if isinstance(replay_buffer, DiskReplayBuffer):
                replay_buffer.start_prefetch()

        # reptile state
        anchor_params: list[torch.Tensor] | None = None
        steps_since_anchor = 0
        optimizer_step_count = 0
        global_step = 0  # Micro-steps (forward passes)

        self.model.train()

        for epoch in range(self.num_train_epochs):
            # Optional: reset anchor at epoch boundary
            if self.enable_reptile and self.reptile_reset_per_epoch and epoch > 0:
                anchor_params = self._snapshot_model()
                steps_since_anchor = 0

            for batch_examples in data_loader:
                # normalize to list
                if isinstance(batch_examples, dict):
                    new_examples = [batch_examples]
                else:
                    new_examples = list(batch_examples)

                # add new examples to replay buffer
                if replay_buffer is not None:
                    replay_buffer.add_many(new_examples)

                # build mixed batch
                mixed_examples = self._build_mixed_batch(
                    new_examples=new_examples,
                    replay_buffer=replay_buffer,
                    current_step=global_step,
                )
                if not mixed_examples:
                    global_step += 1
                    continue

                # collate and move to device
                batch = collator(mixed_examples)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # forward pass
                outputs = self.model(**batch)
                if outputs.loss is None:
                    raise RuntimeError(
                        "Model did not return loss. Ensure DataCollator provides 'labels'."
                    )
                loss = outputs.loss
                kl_loss: torch.Tensor | None = None

                # KL reg
                if self.enable_kl:
                    kl_loss = self._compute_kl_loss(
                        batch=batch,
                        student_logits=outputs.logits,
                        num_new=len(new_examples),
                    )
                    loss = loss + self.kl_weight * kl_loss

                # gradient accumulation
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step_count += 1

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

                # # logging
                # if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                #     lr = scheduler.get_last_lr()[0]
                #     msg = (
                #         f"[MER] step={global_step} epoch={epoch} "
                #         f"loss={loss.item():.4f} lr={lr:.2e}"
                #     )
                #     if kl_loss is not None:
                #         msg += f" kl={kl_loss.item():.4f}"
                #     print(msg, flush=True)

                # check max_steps limit
                if 0 < self.max_steps <= global_step:
                    break

            if 0 < self.max_steps <= global_step:
                break

        # cleanup
        self.model.eval()

        if self._teacher is not None:
            del self._teacher
            self._teacher = None

        # save and clean up disk buffer if used
        if replay_buffer is not None and isinstance(replay_buffer, DiskReplayBuffer):
            if self.buffer_storage_dir is not None:
                replay_buffer.save_state(Path(self.buffer_storage_dir) / "buffer_state.json")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.model

    # buffer creation
    def _create_replay_buffer(self) -> ReplayBuffer:
        """Create replay buffer based on configuration."""
        if self.buffer_type == "disk":
            return DiskReplayBuffer(
                capacity=self.replay_buffer_size,
                storage_dir=self.buffer_storage_dir,
                cache_size=self.buffer_cache_size,
                reservoir=self.reservoir_sampling,
            )
        else:
            return InMemoryReplayBuffer(
                capacity=self.replay_buffer_size,
                reservoir=self.reservoir_sampling,
            )

    # batch construction helpers
    def _compute_new_batch_size(self) -> int:
        """Compute number of new examples per micro-step.

        Total batch = per_device_train_batch_size.
        New examples = (1 - replay_ratio) * batch_size.
        """
        if not self.enable_replay or self.replay_ratio <= 0.0:
            return self.per_device_train_batch_size

        new_size = int(round(self.per_device_train_batch_size * (1.0 - self.replay_ratio)))
        return max(new_size, 1)

    def _estimate_total_steps(self, batches_per_epoch: int) -> int:
        """Estimate total optimizer steps for scheduler."""
        if batches_per_epoch == 0:
            return 1

        total_micro = batches_per_epoch * self.num_train_epochs
        if self.max_steps > 0:
            total_micro = min(total_micro, self.max_steps)

        # convert micro-steps to optimizer steps
        total_opt = math.ceil(total_micro / max(1, self.gradient_accumulation_steps))
        return max(total_opt, 1)

    def _build_mixed_batch(
            self,
            new_examples: Sequence[dict[str, Any]],
            replay_buffer: ReplayBuffer | None,
            current_step: int,
    ) -> list[dict[str, Any]]:
        """Mix new examples with replay samples.

        Returns list with new examples first, then replay samples.
        This ordering is assumed by _compute_kl_loss for masking.
        """
        batch: list[dict[str, Any]] = list(new_examples)

        if not self.enable_replay or replay_buffer is None:
            return batch

        # warmup period: new data
        if current_step < self.replay_start_step:
            return batch

        if self.replay_ratio <= 0.0 or len(replay_buffer) == 0:
            return batch

        # fill remaining batch slots with replay
        num_replay = max(self.per_device_train_batch_size - len(new_examples), 0)
        if num_replay > 0:
            # Use cache-aware sampling for disk buffers
            if isinstance(replay_buffer, DiskReplayBuffer):
                replay_samples = replay_buffer.sample_from_cache(num_replay)
            else:
                replay_samples = replay_buffer.sample(num_replay)
            batch.extend(replay_samples)

        return batch

    # KL regularization helpers
    def _init_teacher(self) -> None:
        """Initialize frozen teacher model for KL regularization."""
        if self.kl_teacher_model_name_or_path:
            teacher = AutoModelForCausalLM.from_pretrained(
                self.kl_teacher_model_name_or_path,
                torch_dtype=self.model.dtype if hasattr(self.model, "dtype") else None,
            )
        else:
            # clone current model as teacher
            warnings.warn(
                "Cloning model as KL teacher doubles GPU memory. "
                "For models >2B parameters, provide kl_teacher_model_name_or_path or disable KL regularization.",
                UserWarning,
            )
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
            num_new: int,
    ) -> torch.Tensor:
        """Compute KL(teacher || student).

        KL is computed only over tokens selected by kl_apply_on, and normalized by the number of selected tokens for
        consistent scaling across modes.
        """
        assert self._teacher is not None

        with torch.no_grad():
            teacher_out = self._teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            teacher_logits = teacher_out.logits

        temperature = float(self.kl_temperature)
        batch_size = student_logits.size(0)

        # compute log probs / probs
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # per-token KL: [B, L, V]
        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
            log_target=False,
        )
        # sum over vocab dimension: [B, L]
        kl_per_position = kl_per_token.sum(dim=-1)

        # build mask based on kl_apply_on
        if self.kl_apply_on == "all":
            row_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        elif self.kl_apply_on == "new":
            row_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            row_mask[:num_new] = True
        else:  # "replay"
            row_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            if num_new < batch_size:
                row_mask[num_new:] = True

        # apply mask and normalize; expand row_mask to [B, L]
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            token_mask = row_mask.unsqueeze(1) & (attention_mask == 1)
        else:
            token_mask = row_mask.unsqueeze(1).expand_as(kl_per_position)

        num_selected = token_mask.sum()
        if num_selected == 0:
            return torch.tensor(0.0, device=self.device)

        kl_masked = (kl_per_position * token_mask).sum()
        kl_normalized = kl_masked / num_selected

        # temperature scaling
        return kl_normalized * (temperature ** 2)

    # reptile helpers
    def _snapshot_model(self) -> list[torch.Tensor]:
        """Clone trainable parameters, optionally to CPU."""
        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                cloned = param.detach().clone()
                if self.reptile_offload_anchor:
                    cloned = cloned.to("cpu")
                params.append(cloned)
        return params

    def _apply_reptile_update(self, anchor_params: list[torch.Tensor]) -> None:
        """Apply Reptile interpolation: θ ← θ_anchor + ε * (θ - θ_anchor)."""
        with torch.no_grad():
            idx = 0
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue

                anchor = anchor_params[idx]
                if self.reptile_offload_anchor:
                    anchor = anchor.to(param.device)

                # theta_new = theta_anchor + eps * (theta_current - theta_anchor) = (1 - eps) * theta_anchor + eps * theta_current
                param.data.mul_(self.reptile_meta_lr).add_(
                    anchor, alpha=(1.0 - self.reptile_meta_lr)
                )
                idx += 1
