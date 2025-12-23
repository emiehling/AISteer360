import torch
from trl import SFTTrainer

from aisteer360.algorithms.structural_control.mer.args import MERArgs


class MERTrainer(SFTTrainer):
    """SFTTrainer with KL divergence regularization and Reptile meta-updates.

    todo: augment description
    """

    def __init__(self, *args, mer_config: MERArgs, **kwargs):
        super().__init__(*args, **kwargs)
        self.mer_config = mer_config

        # reptile state
        self._reptile_anchor: dict[str, torch.Tensor] | None = None
        self._steps_since_anchor: int = 0

        # loss tracking
        self._sft_loss_sum: float = 0.0
        self._kl_loss_sum: float = 0.0
        self._step_count: int = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs,
    ):
        """Compute the combined SFT + KL loss.

        Extends the base SFTTrainer loss computation to add KL divergence regularization when enabled. The KL term
        penalizes deviation from the base model's output distribution.

        Loss = L_SFT + beta * D_KL(pi_{theta + phi} || pi_theta)

        Args:
            model: The model being trained.
            inputs: Batch of input data.
            return_outputs: Whether to return model outputs alongside loss.
            num_items_in_batch: Number of items in the batch (for gradient accumulation).
            **kwargs: Additional arguments.

        Returns:
            If return_outputs is True: tuple of (loss, outputs)
            Otherwise: loss tensor
        """
        need_outputs_for_kl = self.mer_config.kl_enabled
        internal_return_outputs = return_outputs or need_outputs_for_kl

        # compute base SFT loss
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=internal_return_outputs,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )

        if internal_return_outputs:
            loss, outputs = result
        else:
            loss = result
            outputs = None

        # track SFT loss for logging
        self._sft_loss_sum += loss.detach().item()
        self._step_count += 1

        # add KL regularization if enabled
        if self.mer_config.kl_enabled:
            kl_loss = self._compute_kl_loss(model, inputs, outputs.logits)
            loss = loss + self.mer_config.kl_beta * kl_loss
            self._kl_loss_sum += kl_loss.detach().item()

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _compute_kl_loss(
        model,
        inputs: dict,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between fine-tuned and base model.

        Computes D_KL(pi_{theta + phi} || pi_theta).

        Args:
            model: Model with LoRA adapters.
            inputs: Input batch dictionary.
            student_logits: Logits from the fine-tuned model (with adapters).

        Returns:
            Scalar KL divergence loss, averaged over non-padding tokens.
        """
        labels = inputs.get("labels")

        # shift logits
        shift_student = student_logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]

        # get base model logits (by disabling adapters)
        with torch.no_grad():
            model.disable_adapter_layers()
            base_outputs = model(**inputs)
            model.enable_adapter_layers()
            shift_base = base_outputs.logits[..., :-1, :].contiguous()

        # compute KL divergence
        student_logp = torch.log_softmax(shift_student, dim=-1)
        base_logp = torch.log_softmax(shift_base, dim=-1)
        student_p = torch.exp(student_logp)

        kl_per_token = (student_p * (student_logp - base_logp)).sum(dim=-1)

        if labels is not None:
            mask = (labels[..., 1:] != -100).float()
            kl_loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
        else:
            kl_loss = kl_per_token.mean()

        return kl_loss

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
    ):
        """Perform a single training step with optional Reptile meta-update.

        Reptile update: theta = theta_anchor + eta * (theta - theta_anchor)

        Args:
            model: The model being trained.
            inputs: Batch of input data.
            num_items_in_batch: Number of items in the batch.

        Returns:
            Loss tensor from the training step.
        """
        # reptile (save anchor)
        if self.mer_config.reptile_enabled and self._steps_since_anchor == 0:
            self._reptile_anchor = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

        loss = super().training_step(model, inputs, num_items_in_batch)

        # Reptile (meta-update)
        if self.mer_config.reptile_enabled:
            self._steps_since_anchor += 1
            if self._steps_since_anchor >= self.mer_config.reptile_steps:
                self._apply_reptile_update(model)
                self._steps_since_anchor = 0

        return loss

    def _apply_reptile_update(self, model) -> None:
        """Apply Reptile meta-update to interpolate toward anchor weights.

        Args:
            model: Model to update.
        """
        if self._reptile_anchor is None:
            return

        eta = self.mer_config.reptile_lr

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self._reptile_anchor:
                    anchor = self._reptile_anchor[name]
                    param.data = anchor + eta * (param.data - anchor)

    def log(self, logs: dict[str, float], **kwargs) -> None:
        """Log metrics including MER-specific losses."""
        if self._step_count > 0:
            logs["mer/sft_loss"] = self._sft_loss_sum / self._step_count

            if self.mer_config.kl_enabled:
                logs["mer/kl_loss"] = self._kl_loss_sum / self._step_count

            # reset
            self._sft_loss_sum = 0.0
            self._kl_loss_sum = 0.0
            self._step_count = 0

        super().log(logs, **kwargs)
