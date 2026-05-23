"""SCOPE: Self-evolving Context Optimization via Prompt Evolution."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizer

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.scope.args import SCOPEArgs
from aisteer360.algorithms.input_control.scope.memory import Rule, RuleStreamMemory
from aisteer360.algorithms.input_control.scope.meta_agents import (
    GuidelineClassifier,
    GuidelineGenerator,
    GuidelineSelector,
    MemoryOptimizer,
    ReflectionLM,
)

logger = logging.getLogger(__name__)


class _HFReflectionLM:
    """Wraps an HF causal LM as a callable `(prompt) -> response` with cleanup support.

    Mirrors the pattern in `gepa.control._HFReflectionLM`. The duplication is intentional in Phase 5; a future cleanup
    phase may extract a shared utility under `common/`.
    """

    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gen_kwargs = kwargs.pop("gen_kwargs", {}) if "gen_kwargs" in kwargs else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._gen_kwargs = gen_kwargs

    def __call__(self, prompt: str) -> str:
        device = self.model.device
        encoded = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(**encoded, **self._gen_kwargs)
        new_tokens = out[:, encoded["input_ids"].size(1):]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    def cleanup(self) -> None:
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SCOPE(InputControl):
    """Self-evolving Context Optimization via Prompt Evolution.

    SCOPE updates its memory in response to each model output. At `adapt()` time, the current memory is assembled
    into a system prompt that prepends the user's input. At `observe()` time, the four meta-agents synthesize, select,
    classify, and (sometimes) consolidate a new rule into one of two streams.

    The strategic stream persists across sessions; the tactical stream is reset by `reset_session()`.

    Reference:

      - "SCOPE: Prompt Evolution for Enhancing Agent Effectiveness"
        Pei et al.
        [https://arxiv.org/abs/2512.15374](https://arxiv.org/abs/2512.15374)
    """

    Args = SCOPEArgs
    is_stateful: bool = True
    supports_batching: bool = False

    tokenizer: PreTrainedTokenizer | None = None
    memory: RuleStreamMemory | None = None
    _reflection_lm: ReflectionLM | None = None
    _generator: GuidelineGenerator | None = None
    _selector: GuidelineSelector | None = None
    _classifier: GuidelineClassifier | None = None
    _optimizer: MemoryOptimizer | None = None

    def steer(
        self,
        model=None,
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer

        self.memory = RuleStreamMemory()
        if self.seed_rules:
            for rule in self.seed_rules:
                if rule.stream == "strategic":
                    self.memory.strategic.append(rule)
                else:
                    self.memory.tactical.append(rule)

        self._reflection_lm = self._resolve_reflection_lm()

        optimizer_templates = self.optimizer_templates or {}
        self._generator = GuidelineGenerator(
            lm=self._reflection_lm,
            n_candidates=self.n_candidates,
            template=self.generator_template,
        )
        self._selector = GuidelineSelector(
            lm=self._reflection_lm,
            template=self.selector_template,
        )
        self._classifier = GuidelineClassifier(
            lm=self._reflection_lm,
            template=self.classifier_template,
        )
        self._optimizer = MemoryOptimizer(
            lm=self._reflection_lm,
            conflict_template=optimizer_templates.get("conflict_template"),
            subsumption_template=optimizer_templates.get("subsumption_template"),
            consolidation_template=optimizer_templates.get("consolidation_template"),
        )

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        prior: Output | None = None,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Assemble base_prompt + strategic + tactical rules as a system message and prepend to input.

        Note: `prior` is ignored. SCOPE consumes the previous output in `observe()`, not in `adapt()`. The pipeline
        passes `prior` because `is_stateful=True`; SCOPE simply doesn't need it.
        """
        if self.tokenizer is None:
            raise RuntimeError("SCOPE needs a tokenizer; call .steer() first.")

        system_prompt = self._build_system_prompt()
        if not system_prompt:
            return input_ids
        return self._apply_system_prompt(input_ids, system_prompt)

    def observe(
        self,
        input_ids: torch.Tensor,
        output: Output,
        runtime_kwargs: dict | None = None,
    ) -> None:
        """Run the four meta-agent pipeline and update memory in place."""
        if self.memory is None or self._generator is None:
            return

        input_text = self._decode(input_ids)
        response_text = self._decode(output.output_ids)

        if self.trigger_predicate is not None:
            if not self.trigger_predicate(input_text, response_text):
                return

        candidates = self._generator.synthesize(
            input_text=input_text,
            response_text=response_text,
            current_memory=self.memory,
        )
        if not candidates:
            return

        chosen_text = self._selector.select(
            candidates=candidates,
            current_memory=self.memory,
            input_text=input_text,
            response_text=response_text,
        )

        stream, confidence = self._classifier.classify(
            guideline=chosen_text,
            current_memory=self.memory,
        )

        target_stream = (
            "strategic"
            if stream == "strategic" and confidence >= self.confidence_threshold
            else "tactical"
        )

        new_rule = Rule(
            text=chosen_text,
            confidence=confidence,
            stream=target_stream,
            created_at=time.time(),
            metadata={
                "source_input_text": input_text,
                "source_response_text": response_text,
                "synthesis_mode": "unified",
            },
        )

        if new_rule.stream == "strategic":
            self.memory.strategic.append(new_rule)
            if len(self.memory.strategic) > self.strategic_max_size:
                self.memory.strategic = self._optimizer.consolidate(self.memory.strategic)
        else:
            self.memory.tactical.append(new_rule)
            if (
                self.tactical_max_size is not None
                and len(self.memory.tactical) > self.tactical_max_size
            ):
                self.memory.tactical = self.memory.tactical[-self.tactical_max_size:]

    def reset_session(self) -> None:
        """Clear tactical memory for a new task/session. Strategic memory persists."""
        if self.memory is not None:
            self.memory.reset_tactical()

    def cleanup(self) -> None:
        """Release the reflection LM (idempotent)."""
        if self._reflection_lm is not None and hasattr(self._reflection_lm, "cleanup"):
            self._reflection_lm.cleanup()
        self._reflection_lm = None
        self._generator = None
        self._selector = None
        self._classifier = None
        self._optimizer = None

    def _resolve_reflection_lm(self) -> ReflectionLM:
        if callable(self.reflection_lm):
            return self.reflection_lm
        load_kwargs = self.reflection_lm_kwargs or {}
        return _HFReflectionLM(self.reflection_lm, **load_kwargs)

    def _build_system_prompt(self) -> str:
        parts: list[str] = []
        if self.base_prompt:
            parts.append(self.base_prompt)
        if self.memory:
            if self.memory.strategic:
                parts.append("\nStrategic guidelines (persistent):")
                for rule in self.memory.strategic:
                    parts.append(f"- {rule.text}")
            if self.memory.tactical:
                parts.append("\nTactical guidelines (this session):")
                for rule in self.memory.tactical:
                    parts.append(f"- {rule.text}")
        return "\n".join(parts).strip()

    def _apply_system_prompt(
        self,
        input_ids: list[int] | torch.Tensor,
        system_prompt: str,
    ) -> list[int] | torch.Tensor:
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

        has_chat_template = (
            hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template
        )

        adapted_batch: list[list[int]] = []
        for ids_single in batch_input_ids:
            original_text = self.tokenizer.decode(ids_single, skip_special_tokens=True)
            if has_chat_template:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original_text},
                ]
                adapted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                adapted_text = f"{system_prompt}\n\n{original_text}"
            adapted_tokens = self.tokenizer.encode(adapted_text, add_special_tokens=False)
            adapted_batch.append(adapted_tokens)

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

    def _decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
            if ids.ndim == 2:
                ids = ids[0]
            ids_list = ids.tolist()
        else:
            ids_list = list(ids)
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(ids_list, skip_special_tokens=True)
