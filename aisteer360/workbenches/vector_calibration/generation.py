"""Contrastive pair generation using an LLM."""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from aisteer360.utils.tokenization import ensure_pad_token
from aisteer360.algorithms.state_control._common.specs import ContrastivePairs

from .configs import GenerationConfig
from .results import GenerationResult

if TYPE_CHECKING:
    from aisteer360.workbenches.common.agent.providers.base import GenerationProvider

logger = logging.getLogger(__name__)


class _CancelStoppingCriteria(StoppingCriteria):
    """Stops `model.generate` as soon as `cancel_check()` returns True."""

    def __init__(self, cancel_check: Callable[[], bool]):
        self._cancel_check = cancel_check
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self._cancel_check():
            self.cancelled = True
            return True
        return False


class ContrastivePairGenerator:
    """Generates contrastive response pairs for steering vector training.

    For each seed prompt, the generator model is invoked twice: once with the positive system prompt (producing
    behavior-exhibiting responses) and once with the negative system prompt (producing behavior-absent responses).
    The result is a `ContrastivePairs` object that plugs directly into the existing estimator infrastructure.

    The generator model is loaded and unloaded inside `generate()` to free VRAM for the steered model in
    subsequent stages. If the caller provides a pre-loaded model and tokenizer (e.g., because the generator is
    the steered model), those are used instead.
    """

    def __init__(self, config: GenerationConfig, provider: "GenerationProvider | None" = None):
        self.config = config
        self.provider = provider

    def generate(
        self,
        model=None,
        tokenizer=None,
        on_progress: Callable[[int, int], None] | None = None,
        output_path: Path | None = None,
        behavior: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        """Produce contrastive pairs.

        Args:
            model: Optional pre-loaded generator model.
            tokenizer: Optional pre-loaded tokenizer.
            on_progress: Callback receiving `(completed_pairs, total_pairs)`.
            output_path: Optional path to a JSONL file. When set, each completed batch is appended immediately so
                that an interrupted run can be resumed. If the file already exists, its valid lines are loaded and
                generation resumes from the next pair.
            behavior: Behavior label written into each JSONL record. Required when `output_path` is set.
            cancel_check: Optional callable polled after each batch; when it returns True, generation stops and
                returns a `GenerationResult` containing whatever pairs have been completed so far.

        Returns:
            `GenerationResult` containing the `ContrastivePairs`.
        """
        cfg = self.config

        if output_path is not None and behavior is None:
            raise ValueError("behavior must be provided when output_path is set.")

        seeds = self._resolve_seeds(cfg)
        total = len(seeds)

        positives: list[str] = []
        negatives: list[str] = []
        prompts_used: list[str] = []

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and output_path.stat().st_size > 0:
                existing = self._read_existing_records(output_path)
                n_done = len(existing)
                logger.info("Resuming generation: %d/%d pairs already on disk.", n_done, total)
                for rec in existing:
                    prompts_used.append(rec["prompt"])
                    positives.append(rec["positive"])
                    negatives.append(rec["negative"])
            else:
                n_done = 0
        else:
            n_done = 0

        if n_done >= total:
            pairs = ContrastivePairs(
                positives=positives[:total],
                negatives=negatives[:total],
                prompts=prompts_used[:total],
            )
            return GenerationResult(
                pairs=pairs,
                seed_prompts_used=prompts_used[:total],
                config=asdict(cfg),
            )

        owns_model = False
        if self.provider is None:
            owns_model = model is None
            if owns_model:
                logger.info("Loading generator model: %s", cfg.generator_model)
                tokenizer = AutoTokenizer.from_pretrained(cfg.generator_model)
                tokenizer = ensure_pad_token(tokenizer)
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.generator_model, device_map="auto", torch_dtype="auto"
                )

        for batch_start in range(n_done, total, cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, total)
            batch_seeds = seeds[batch_start:batch_end]

            if cancel_check is not None and cancel_check():
                logger.info("Generation cancelled before batch at %d/%d pairs.", len(positives), total)
                break

            if self.provider is not None:
                pos_batch = self.provider.generate_batch(
                    cfg.positive_prompt, batch_seeds,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    cancel_check=cancel_check,
                )
                if not pos_batch:
                    if cancel_check is not None and cancel_check():
                        logger.info(
                            "Generation cancelled mid-batch at %d/%d pairs.", len(positives), total
                        )
                        break
                    raise RuntimeError(
                        f"Generation provider returned no results for the positive batch "
                        f"(batch starting at {batch_start}). Check provider logs for details."
                    )
                neg_batch = self.provider.generate_batch(
                    cfg.negative_prompt, batch_seeds,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    cancel_check=cancel_check,
                )
                if not neg_batch:
                    if cancel_check is not None and cancel_check():
                        logger.info(
                            "Generation cancelled mid-batch at %d/%d pairs.", len(positives), total
                        )
                        break
                    raise RuntimeError(
                        f"Generation provider returned no results for the negative batch "
                        f"(batch starting at {batch_start}). Check provider logs for details."
                    )
            else:
                pos_batch, pos_cancelled = self._generate_batch(
                    model, tokenizer, batch_seeds,
                    system_prompt=cfg.positive_prompt, cfg=cfg,
                    cancel_check=cancel_check,
                )
                if pos_cancelled:
                    logger.info(
                        "Generation cancelled mid-batch at %d/%d pairs.", len(positives), total
                    )
                    break

                neg_batch, neg_cancelled = self._generate_batch(
                    model, tokenizer, batch_seeds,
                    system_prompt=cfg.negative_prompt, cfg=cfg,
                    cancel_check=cancel_check,
                )
                if neg_cancelled:
                    logger.info(
                        "Generation cancelled mid-batch at %d/%d pairs.", len(positives), total
                    )
                    break

            positives.extend(pos_batch)
            negatives.extend(neg_batch)
            prompts_used.extend(batch_seeds)

            if output_path is not None:
                with open(output_path, "a") as f:
                    for seed, pos, neg in zip(batch_seeds, pos_batch, neg_batch):
                        record = {
                            "prompt": seed,
                            "positive": pos,
                            "negative": neg,
                            "behavior": behavior,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if on_progress:
                on_progress(len(positives), total)

            if cancel_check is not None and cancel_check():
                logger.info("Generation cancelled after %d/%d pairs.", len(positives), total)
                break

        if owns_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not positives:
            # cancelled before any batch completed; placeholder entries satisfy
            # ContrastivePairs validation, and empty seed_prompts_used signals zero real pairs.
            return GenerationResult(
                pairs=ContrastivePairs(positives=[""], negatives=[""]),
                seed_prompts_used=[],
                config=asdict(cfg),
            )

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
    def _read_existing_records(path: Path) -> list[dict]:
        """Load valid JSONL records from an existing pairs file.

        A malformed trailing line (from a crash mid-write) is logged and discarded so that generation resumes
        from the last fully-flushed batch boundary.
        """
        records: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Discarding malformed trailing line in %s", path)
                    break
        return records

    @staticmethod
    def _resolve_seeds(cfg: GenerationConfig) -> list[str]:
        """Load seed prompts and return them in shuffled order.

        The number of pairs produced equals the length of the resolved seed list.
        """
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
        rng.shuffle(seeds)
        return seeds

    @staticmethod
    def _generate_batch(
        model,
        tokenizer,
        seed_prompts: list[str],
        system_prompt: str,
        cfg: GenerationConfig,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[str], bool]:
        """Generate a batch of responses with the given system prompt.

        Returns a tuple of (decoded outputs, cancelled). When `cancel_check` fires during generation, the stopping
        criterion aborts `model.generate` and the returned `cancelled` flag is True; callers should discard the
        partially-generated outputs.
        """
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

        stopping_criteria = None
        cancel_criterion: _CancelStoppingCriteria | None = None
        if cancel_check is not None:
            cancel_criterion = _CancelStoppingCriteria(cancel_check)
            stopping_criteria = StoppingCriteriaList([cancel_criterion])

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )

        cancelled = bool(cancel_criterion and cancel_criterion.cancelled)
        if cancelled:
            return [], True

        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        ), False
