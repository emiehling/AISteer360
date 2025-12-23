import math
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig

from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.algorithms.structural_control.mer.args import MERArgs
from aisteer360.algorithms.structural_control.mer.utils.mer_trainer import MERTrainer


class MER(StructuralControl):
    """
    Implementation of MER (Meta-Experience Replay) from todo reference.

    todo: description

    The method addresses the stability-plasticity tradeoff through three mechanisms:

    1. Approximate Experience Replay: todo

    2. KL Divergence Regularization: todo

    3. Reptile Meta-Updates: todo

    Args:
        todo

    Reference:
    - todo: "title", authors, paper link
    """

    Args = MERArgs

    # placeholders
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    device: torch.device | str | None = None

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase | None = None,
            **kwargs: Any,
    ) -> PreTrainedModel:
        """Apply MER fine-tuning to the model.

        Args:
            model: Base language model to fine-tune.
            tokenizer: Tokenizer for the model. If None, attempts to retrieve from model attributes.
            **kwargs: Additional arguments (unused).

        Returns:
            PreTrainedModel: Fine-tuned model. If `merge_lora_after_train=True`, returns the model with LoRA weights
            merged; otherwise, returns the model with LoRA adapter.

        Raises:
            ValueError: If dataset format is incompatible.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = next(model.parameters()).device

        # ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # build mixed dataset (task + replay)
        train_dataset = self._build_train_dataset()

        # configure LoRA
        peft_config = None
        if self.use_peft:
            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=list(self.lora_target_modules) if self.lora_target_modules else None,
                bias="none",
                task_type="CAUSAL_LM",
            )

        # configure SFT training
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            report_to=[] if self.report_to == "none" else [self.report_to],
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            seed=self.seed,
            lr_scheduler_type="constant",
            max_seq_length=self.max_length,
            dataset_text_field="text",
            packing=self.packing,
        )

        # create/run trainer
        trainer = MERTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
            mer_config=self.args,
        )

        trainer.train()
        trained_model = trainer.model

        if self.use_peft and self.merge_lora_after_train:
            trained_model = trained_model.merge_and_unload()

        return trained_model

    def _build_train_dataset(self) -> Dataset:
        """Build the combined training dataset with task and replay examples.

        Returns:
            Combined and shuffled dataset with 'text' column.

        Raises:
            ValueError: If dataset cannot be converted to required format.
        """
        task_ds = self._ensure_text_column(self.train_dataset)
        n_task = len(task_ds)

        # return task-only if replay is disabled
        if not self.replay_enabled or self.replay_rate <= 0:
            return task_ds

        # replay dataset
        replay_ds = self._ensure_text_column(self.replay_dataset)
        n_replay = int(math.ceil(self.replay_rate * n_task))
        replay_shuffled = replay_ds.shuffle(seed=self.replay_seed)
        if len(replay_shuffled) >= n_replay:
            replay_sampled = replay_shuffled.select(range(n_replay))
        else:
            repeats = math.ceil(n_replay / len(replay_shuffled))
            replay_repeated = concatenate_datasets([replay_shuffled] * repeats)
            replay_sampled = replay_repeated.select(range(n_replay))

        # combine and shuffle
        combined = concatenate_datasets([task_ds, replay_sampled])
        combined = combined.shuffle(seed=self.replay_seed)

        return combined

    @staticmethod
    def _ensure_text_column(dataset: Dataset) -> Dataset:
        """Ensure dataset has a 'text' column for SFTTrainer.

        Args:
            dataset: Input dataset to process.

        Returns:
            Dataset with 'text' column.

        Raises:
            ValueError: If dataset format is not recognized.
        """
        if "text" in dataset.column_names:
            return dataset.select_columns(["text"])

        if "prompt" in dataset.column_names and "response" in dataset.column_names:
            return dataset.map(
                lambda x: {"text": f"{x['prompt']}{x['response']}"},
                remove_columns=dataset.column_names,
            )

        raise ValueError(
            f"Cannot find or create 'text' column from dataset columns: {dataset.column_names}. "
            "Expected either 'text' column or both 'prompt' and 'response' columns."
        )
