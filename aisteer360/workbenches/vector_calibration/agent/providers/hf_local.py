"""Local HuggingFace providers.

The generation provider owns an `AutoModelForCausalLM` and reproduces the existing
`ContrastivePairGenerator._generate_batch` logic. The judge provider wraps `LLMJudgeMetric`
unchanged so the HF judge code path stays byte-identical to today.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric

from .base import GenerationProvider, JudgeProvider

logger = logging.getLogger(__name__)


class _CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_check: Callable[[], bool]):
        self._cancel_check = cancel_check
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self._cancel_check():
            self.cancelled = True
            return True
        return False


class HFGenerationProvider(GenerationProvider):
    """Generate using a locally-loaded HF causal-LM."""

    def __init__(
        self,
        model_id: str,
        *,
        hf_token: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.model_id = model_id
        self._token = hf_token
        logger.info("Loading HF generator: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.tokenizer = ensure_pad_token(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_token,
        )

    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[str]:
        if not user_prompts:
            return []

        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ]
            for p in user_prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in messages_batch
        ]
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        stopping_criteria = None
        cancel_criterion: _CancelStoppingCriteria | None = None
        if cancel_check is not None:
            cancel_criterion = _CancelStoppingCriteria(cancel_check)
            stopping_criteria = StoppingCriteriaList([cancel_criterion])

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )

        if cancel_criterion is not None and cancel_criterion.cancelled:
            return []

        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )

    def close(self) -> None:
        self.model = None  # type: ignore[assignment]
        self.tokenizer = None  # type: ignore[assignment]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFJudgeProvider(JudgeProvider):
    """Wrap `LLMJudgeMetric` with the JudgeProvider surface.

    The template is passed through at call time (not at construction) so that a single provider
    instance can serve different calibration configurations. `LLMJudgeMetric` expects the template
    to contain `{response}` (and optionally `{prompt}`, `{lower_bound}`, `{upper_bound}`); we
    pass those placeholders through verbatim.
    """

    def __init__(self, config: dict, *, hf_token: str | None = None):
        self.config = dict(config)
        self._token = hf_token
        self._metric: LLMJudgeMetric | None = None

    def _build_metric(self, template: str, scale: tuple[float, float]) -> LLMJudgeMetric:
        if self._metric is not None:
            return self._metric
        logger.info("Loading HF judge: %s", self.config["model"])
        batch_size = int(self.config.get("batch_size", 8))
        self._metric = LLMJudgeMetric(
            model_or_id=self.config["model"],
            prompt_template=template,
            scale=scale,
            batch_size=batch_size,
        )
        return self._metric

    def score(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        template: str,
        scale: tuple[float, float],
    ) -> dict[str, Any]:
        metric = self._build_metric(template, scale)
        # re-render the template-bound internals if the template changed between calls
        if metric.base_prompt_template != template.strip():
            metric.base_prompt_template = template.strip()
        return metric.compute(responses=responses, prompts=prompts)

    def close(self) -> None:
        self._metric = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
