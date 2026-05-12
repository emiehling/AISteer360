"""Contrastive pair generation using an LLM."""
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.algorithms.state_control.common.specs import ContrastivePairs

from .configs import GenerationConfig
from .results import GenerationResult

logger = logging.getLogger(__name__)


class ContrastivePairGenerator:
    """Generates contrastive response pairs for steering vector training.

    For each seed prompt, the generator model is invoked twice: once with the positive system prompt (producing
    behavior-exhibiting responses) and once with the negative system prompt (producing behavior-absent responses).
    The result is a `ContrastivePairs` object that plugs directly into the existing estimator infrastructure.

    The generator model is loaded and unloaded inside `generate()` to free VRAM for the steered model in
    subsequent stages. If the caller provides a pre-loaded model and tokenizer (e.g., because the generator is
    the steered model), those are used instead.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate(
        self,
        model=None,
        tokenizer=None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> GenerationResult:
        """Produce contrastive pairs.

        Args:
            model: Optional pre-loaded generator model.
            tokenizer: Optional pre-loaded tokenizer.
            on_progress: Callback receiving `(completed_pairs, total_pairs)`.

        Returns:
            `GenerationResult` containing the `ContrastivePairs`.
        """
        cfg = self.config
        owns_model = model is None

        if owns_model:
            logger.info("Loading generator model: %s", cfg.generator_model)
            tokenizer = AutoTokenizer.from_pretrained(cfg.generator_model)
            tokenizer = ensure_pad_token(tokenizer)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.generator_model, device_map="auto", torch_dtype="auto"
            )

        seeds = self._resolve_seeds(cfg)

        positives: list[str] = []
        negatives: list[str] = []
        prompts_used: list[str] = []
        total = cfg.n_pairs

        for batch_start in range(0, total, cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, total)
            batch_seeds = seeds[batch_start:batch_end]

            pos_batch = self._generate_batch(
                model, tokenizer, batch_seeds,
                system_prompt=cfg.positive_prompt, cfg=cfg,
            )
            neg_batch = self._generate_batch(
                model, tokenizer, batch_seeds,
                system_prompt=cfg.negative_prompt, cfg=cfg,
            )

            positives.extend(pos_batch)
            negatives.extend(neg_batch)
            prompts_used.extend(batch_seeds)

            if on_progress:
                on_progress(len(positives), total)

        if owns_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pairs = ContrastivePairs(
            positives=positives,
            negatives=negatives,
            prompts=prompts_used,
        )
        return GenerationResult(
            pairs=pairs,
            seed_prompts_used=prompts_used,
            config=asdict(cfg),
        )

    @staticmethod
    def _resolve_seeds(cfg: GenerationConfig) -> list[str]:
        """Load, sample, or cycle seed prompts to reach `n_pairs`."""
        if isinstance(cfg.seed_prompts, str):
            raw_path = Path(cfg.seed_prompts)
            text = raw_path.read_text()
            if raw_path.suffix == ".jsonl":
                seeds = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    seeds.append(obj if isinstance(obj, str) else obj["prompt"])
            else:
                raw = json.loads(text)
                seeds = [s if isinstance(s, str) else s["prompt"] for s in raw]
        elif cfg.seed_prompts is not None:
            seeds = list(cfg.seed_prompts)
        else:
            raise ValueError("seed_prompts must be provided.")

        if not seeds:
            raise ValueError("seed_prompts is empty after resolution.")

        rng = random.Random(cfg.seed)
        if len(seeds) >= cfg.n_pairs:
            return rng.sample(seeds, cfg.n_pairs)
        full = seeds * (cfg.n_pairs // len(seeds) + 1)
        rng.shuffle(full)
        return full[: cfg.n_pairs]

    @staticmethod
    def _generate_batch(
        model,
        tokenizer,
        seed_prompts: list[str],
        system_prompt: str,
        cfg: GenerationConfig,
    ) -> list[str]:
        """Generate a batch of responses with the given system prompt."""
        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": seed},
            ]
            for seed in seed_prompts
        ]

        texts = [
            tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in messages_batch
        ]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )
