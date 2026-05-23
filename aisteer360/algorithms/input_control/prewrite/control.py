"""PRewrite: input rewriting via reinforcement learning."""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.prewrite.args import PRewriteArgs
from aisteer360.algorithms.input_control.prewrite.memory import ModelMemory
from aisteer360.algorithms.input_control.prewrite.templates import (
    DEFAULT_PER_QUERY_META_PROMPT,
    DEFAULT_STATIC_META_PROMPT,
)
from aisteer360.algorithms.input_control.prewrite.trainer import PRewriteTrainer

logger = logging.getLogger(__name__)


class PRewrite(InputControl):
    """Input rewriting via reinforcement learning.

    Trains a small rewriter LLM to produce improved instructions for a task LLM, using a PPO-style policy gradient
    with task-metric reward and a KL penalty against the frozen reference rewriter.

    Two modes:

      - `mode="per_query"` (default): the rewriter conditions on each user query at serve time, producing
        query-specific rewrites. Memory = `ModelMemory` holding the trained rewriter. First method in AISteer360 with
        this shape.
      - `mode="static"` (PRewrite-I): the rewriter conditions only on the initial prompt. At end of `steer()`, one
        rewrite is cached and the rewriter is discarded. Memory = `TextMemory`.

    `is_stateful = False`.

    Reference:

      - "PRewrite: Prompt Rewriting with Reinforcement Learning"
        Weize Kong, Spurthi Amba Hombaiah, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky
        [https://arxiv.org/abs/2401.08189](https://arxiv.org/abs/2401.08189)
    """

    Args = PRewriteArgs
    is_stateful: bool = False
    supports_batching: bool = False

    tokenizer: PreTrainedTokenizer | None = None
    memory: ModelMemory | TextMemory | None = None
    _rewriter_gen_kwargs_at_serve: dict | None = None
    _trainer_factory: Callable[..., PRewriteTrainer] | None = None

    def steer(
        self,
        model: Any = None,
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs,
    ) -> None:
        """Run training. `model` is the task LM (frozen during training)."""
        if model is None:
            raise ValueError("PRewrite requires a task model passed via `model`.")
        self.tokenizer = tokenizer

        rewriter_model, rewriter_tokenizer = self._load_rewriter()

        if self.use_peft:
            rewriter_model = self._wrap_peft(rewriter_model)

        trainer = self._build_trainer(
            rewriter_model=rewriter_model,
            rewriter_tokenizer=rewriter_tokenizer,
            task_model=model,
            task_tokenizer=tokenizer,
        )

        trained_rewriter = trainer.train(self.training_data)

        if self.mode == "per_query":
            trained_rewriter.eval()
            for parameter in trained_rewriter.parameters():
                parameter.requires_grad_(False)

            self.memory = ModelMemory(
                model_name_or_path=self.rewriter_model_name_or_path,
                model=trained_rewriter,
                tokenizer=rewriter_tokenizer,
                extras={
                    "mode": "per_query",
                    "use_peft": bool(self.use_peft),
                    "training_config": self._serializable_training_config(),
                    "meta_prompt": self._resolved_meta_prompt(),
                    "initial_prompt": self.initial_prompt,
                },
            )
            gen_kwargs = self._resolved_rewriter_gen_kwargs()
            self._rewriter_gen_kwargs_at_serve = {
                **gen_kwargs,
                "do_sample": False,
                "temperature": 0.0,
            }
        else:  # static
            rewritten = trainer._generate_one_rewrite()
            self.memory = TextMemory(
                instruction=rewritten,
                extras={
                    "mode": "static",
                    "initial_prompt": self.initial_prompt,
                    "meta_prompt": self._resolved_meta_prompt(),
                },
            )
            del trained_rewriter
            del rewriter_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        prior: Output | None = None,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Mode-dependent transformation of `input_ids`.

        per_query: decode → run rewriter → assemble (rewritten, decoded_input) via chat template → re-encode.
        static: prepend `memory.instruction` as system message (same shape as GEPA's adapt).
        """
        if self.tokenizer is None or self.memory is None:
            raise RuntimeError("PRewrite needs to be steered first.")

        if isinstance(self.memory, TextMemory):
            return self._adapt_static(input_ids)
        if isinstance(self.memory, ModelMemory):
            return self._adapt_per_query(input_ids)
        raise RuntimeError(f"Unexpected memory type: {type(self.memory).__name__}")

    def cleanup(self) -> None:
        """Release the rewriter (when held). Idempotent and safe with `TextMemory`."""
        if isinstance(self.memory, ModelMemory):
            self.memory.cleanup()
        self.memory = None

    def _load_rewriter(self):
        load_kwargs = self.rewriter_load_kwargs or {}
        rewriter_model = AutoModelForCausalLM.from_pretrained(
            self.rewriter_model_name_or_path, trust_remote_code=True, **load_kwargs
        )
        rewriter_tokenizer = AutoTokenizer.from_pretrained(
            self.rewriter_model_name_or_path, trust_remote_code=True
        )
        if rewriter_tokenizer.pad_token_id is None:
            if rewriter_tokenizer.eos_token_id is not None:
                rewriter_tokenizer.pad_token = rewriter_tokenizer.eos_token
        return rewriter_model, rewriter_tokenizer

    def _wrap_peft(self, model):
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(**(self.lora_kwargs or {}))
        return get_peft_model(model, config)

    def _build_trainer(
        self,
        *,
        rewriter_model,
        rewriter_tokenizer,
        task_model,
        task_tokenizer,
    ) -> PRewriteTrainer:
        if self._trainer_factory is not None:
            return self._trainer_factory(
                rewriter_model=rewriter_model,
                rewriter_tokenizer=rewriter_tokenizer,
                task_model=task_model,
                task_tokenizer=task_tokenizer,
                feedback_metric=self.feedback_metric,
                meta_prompt=self._resolved_meta_prompt(),
                initial_prompt=self.initial_prompt,
                mode=self.mode,
                config=self._build_ppo_config(),
                rewriter_gen_kwargs=self._resolved_rewriter_gen_kwargs(),
                task_gen_kwargs=self.task_gen_kwargs or {},
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                kl_coef=self.kl_coef,
                learning_rate=self.learning_rate,
                seed=self.seed,
            )
        return PRewriteTrainer(
            rewriter_model=rewriter_model,
            rewriter_tokenizer=rewriter_tokenizer,
            task_model=task_model,
            task_tokenizer=task_tokenizer,
            feedback_metric=self.feedback_metric,
            meta_prompt=self._resolved_meta_prompt(),
            initial_prompt=self.initial_prompt,
            mode=self.mode,
            config=self._build_ppo_config(),
            rewriter_gen_kwargs=self._resolved_rewriter_gen_kwargs(),
            task_gen_kwargs=self.task_gen_kwargs or {},
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            kl_coef=self.kl_coef,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )

    def _build_ppo_config(self):
        """Construct a `trl.PPOConfig` from `PRewriteArgs` fields, filtering unrecognized kwargs.

        The config object is held for hyperparameter consistency and future TRL migration; the trainer's policy
        update uses these values directly.
        """
        try:
            from trl import PPOConfig
        except ImportError:
            return None

        from dataclasses import fields, is_dataclass
        import inspect

        candidate_kwargs = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "mini_batch_size": self.mini_batch_size,
            "num_ppo_epochs": self.ppo_epochs,
            "ppo_epochs": self.ppo_epochs,
            "kl_coef": self.kl_coef,
            "seed": self.seed,
            "output_dir": self.output_dir,
        }
        if is_dataclass(PPOConfig):
            allowed = {f.name for f in fields(PPOConfig)}
        else:
            try:
                allowed = set(inspect.signature(PPOConfig).parameters.keys())
            except (TypeError, ValueError):
                allowed = set(candidate_kwargs.keys())
        filtered = {k: v for k, v in candidate_kwargs.items() if k in allowed and v is not None}
        try:
            return PPOConfig(**filtered)
        except Exception as exc:
            logger.debug("Could not build PPOConfig with filtered kwargs %s: %s", filtered, exc)
            return None

    def _resolved_meta_prompt(self) -> str:
        if self.meta_prompt is not None:
            return self.meta_prompt
        if self.mode == "per_query":
            return DEFAULT_PER_QUERY_META_PROMPT
        return DEFAULT_STATIC_META_PROMPT

    def _resolved_rewriter_gen_kwargs(self) -> dict:
        defaults = {"max_new_tokens": 128}
        return {**defaults, **(self.rewriter_gen_kwargs or {})}

    def _serializable_training_config(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "mini_batch_size": self.mini_batch_size,
            "ppo_epochs": self.ppo_epochs,
            "learning_rate": self.learning_rate,
            "kl_coef": self.kl_coef,
            "seed": self.seed,
        }

    def _adapt_per_query(self, input_ids):
        assert isinstance(self.memory, ModelMemory)
        if self.memory.model is None or self.memory.tokenizer is None:
            raise RuntimeError("ModelMemory model/tokenizer not loaded.")

        is_tensor = isinstance(input_ids, torch.Tensor)
        original_device = input_ids.device if is_tensor else None
        original_dtype = input_ids.dtype if is_tensor else None

        if is_tensor:
            if input_ids.ndim > 1 and input_ids.shape[0] > 1:
                raise NotImplementedError(
                    "PRewrite per-query adapt currently handles single-sequence input only."
                )
            ids_list = input_ids.reshape(-1).tolist()
            single_sequence = (input_ids.ndim == 1) or (input_ids.shape[0] == 1)
        else:
            if input_ids and isinstance(input_ids[0], list):
                if len(input_ids) > 1:
                    raise NotImplementedError(
                        "PRewrite per-query adapt currently handles single-sequence input only."
                    )
                ids_list = list(input_ids[0])
                single_sequence = True
            else:
                ids_list = list(input_ids)
                single_sequence = True

        query_text = self.tokenizer.decode(ids_list, skip_special_tokens=True)
        rewriter_tokenizer = self.memory.tokenizer
        rewriter_model = self.memory.model

        rewriter_prompt = self._resolved_meta_prompt().format(
            initial_prompt=self.initial_prompt, query=query_text
        )
        device = next(rewriter_model.parameters()).device
        encoded = rewriter_tokenizer(rewriter_prompt, return_tensors="pt").to(device)
        gen_kwargs = self._rewriter_gen_kwargs_at_serve or {
            "max_new_tokens": 128, "do_sample": False, "temperature": 0.0,
        }
        with torch.no_grad():
            out = rewriter_model.generate(**encoded, **gen_kwargs)
        new_tokens = out[0, encoded["input_ids"].size(1):]
        rewritten_instruction = rewriter_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        adapted_tokens = self._assemble_with_instruction(query_text, rewritten_instruction)

        if is_tensor:
            tensor_out = torch.tensor(adapted_tokens, dtype=original_dtype, device=original_device)
            if single_sequence and (input_ids.ndim == 1):
                return tensor_out
            return tensor_out.unsqueeze(0)
        return adapted_tokens

    def _adapt_static(self, input_ids):
        assert isinstance(self.memory, TextMemory)
        instruction = self.memory.instruction
        if not instruction:
            return input_ids

        is_tensor = isinstance(input_ids, torch.Tensor)
        original_device = input_ids.device if is_tensor else None
        original_dtype = input_ids.dtype if is_tensor else None

        if is_tensor:
            if input_ids.ndim == 1:
                batch_input_ids = [input_ids.tolist()]
                single_sequence = True
            else:
                batch_input_ids = input_ids.tolist()
                single_sequence = False
        else:
            if input_ids and isinstance(input_ids[0], int):
                batch_input_ids = [list(input_ids)]
                single_sequence = True
            else:
                batch_input_ids = [list(seq) for seq in input_ids]
                single_sequence = False

        adapted_batch = []
        for ids_single in batch_input_ids:
            query_text = self.tokenizer.decode(ids_single, skip_special_tokens=True)
            adapted_batch.append(self._assemble_with_instruction(query_text, instruction))

        max_len = max(len(seq) for seq in adapted_batch)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        padded = [seq + [pad_id] * (max_len - len(seq)) for seq in adapted_batch]

        if is_tensor:
            result = torch.tensor(padded, dtype=original_dtype, device=original_device)
            if single_sequence:
                result = result.squeeze(0)
            return result
        if single_sequence:
            return padded[0]
        return padded

    def _assemble_with_instruction(self, query_text: str, instruction: str) -> list[int]:
        has_chat_template = (
            hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template
        )
        if has_chat_template:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": query_text},
            ]
            adapted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            adapted_text = f"{instruction}\n\n{query_text}"
        return self.tokenizer.encode(adapted_text, add_special_tokens=False)
