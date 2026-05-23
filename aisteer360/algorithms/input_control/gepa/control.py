"""GEPA: reflective prompt evolution input control."""
from __future__ import annotations

import random
import warnings
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizer

from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.common.candidate import Candidate
from aisteer360.algorithms.input_control.common.memory.text import TextMemory
from aisteer360.algorithms.input_control.gepa.archive import PerInstanceParetoArchive
from aisteer360.algorithms.input_control.gepa.args import GEPAArgs
from aisteer360.algorithms.input_control.gepa.proposers import (
    GEPAReflectionProposer,
    MergeProposer,
)
from aisteer360.algorithms.input_control.gepa.scorer import FeedbackScorer


class _HFReflectionLM:
    """Wraps an HF causal LM as a callable `(prompt) -> response`, with cleanup support."""

    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._gen_kwargs = kwargs.pop("gen_kwargs", {}) if "gen_kwargs" in kwargs else {}

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


class GEPA(InputControl):
    """Reflective prompt evolution for textual instruction memory.

    GEPA optimizes the `instruction` field of a `TextMemory` by:

    1. Maintaining a per-instance Pareto frontier of candidates over the validation data.
    2. Each iteration: sample a candidate from the frontier; reflect on a fresh minibatch of training rollouts (or
       merge two candidates); evaluate the new candidate on the validation set; ingest into the archive.
    3. Repeat until the metric-call budget is exhausted.

    At serve time, GEPA's `adapt()` is a static template fill: the optimized instruction is prepended to the user's
    prompt via the model's chat template (or directly if no chat template is available). No model in the serving path;
    no state across invocations.

    Reference:

      - "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
        Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Jha, Krista Opsahl-Ong, Michael J Ryan,
        Anna L Goldie, Christopher Potts, Matei Zaharia, Omar Khattab
        [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)
    """

    Args = GEPAArgs
    supports_batching: bool = True
    is_stateful: bool = False

    tokenizer: PreTrainedTokenizer | None = None
    memory: TextMemory | None = None
    _reflection_lm: Callable[[str], str] | None = None

    def steer(
        self,
        model=None,
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self._reflection_lm = self._resolve_reflection_lm()

        adapter = self._make_adapter(tokenizer)

        scorer = FeedbackScorer(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            feedback_metric=self.feedback_metric,
            gen_kwargs=self.gen_kwargs,
        )

        reflection_proposer = GEPAReflectionProposer(
            reflection_lm=self._reflection_lm,
            n_candidates=1,
        )
        merge_proposer = (
            MergeProposer(reflection_lm=self._reflection_lm, rng_seed=self.seed)
            if self.use_merge else None
        )

        archive = PerInstanceParetoArchive(rng_seed=self.seed)

        val_data = self.val_data
        if val_data is None:
            warnings.warn(
                "GEPA: no val_data provided; using train_data for both reflection minibatches and Pareto tracking. "
                "Consider providing a separate val_data to avoid overfitting.",
                UserWarning,
            )
            val_data = self.train_data

        initial_memory = TextMemory(instruction=self.seed_instruction)
        initial_candidate = Candidate(memory=initial_memory)

        best = self._run_gepa_loop(
            initial=initial_candidate,
            scorer=scorer,
            reflection_proposer=reflection_proposer,
            merge_proposer=merge_proposer,
            archive=archive,
            train_data=self.train_data,
            val_data=val_data,
        )

        self.memory = best.memory
        self.cleanup()

    def cleanup(self) -> None:
        """Release the reflection LM if one was loaded."""
        if self._reflection_lm is not None and hasattr(self._reflection_lm, "cleanup"):
            self._reflection_lm.cleanup()
        self._reflection_lm = None

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        prior: Output | None = None,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Prepend the optimized instruction to `input_ids` via the model's chat template.

        Static adapter -- no model, no state across calls.
        """
        if self.tokenizer is None:
            raise RuntimeError("GEPA needs a tokenizer; call .steer() first.")
        if self.memory is None or not self.memory.instruction:
            return input_ids
        return self._apply_instruction(input_ids, self.memory.instruction)

    def _apply_instruction(
        self,
        input_ids: list[int] | torch.Tensor,
        instruction: str,
    ) -> list[int] | torch.Tensor:
        """Apply the given instruction to input_ids; shared by `adapt()` and the train-time adapter."""
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

        adapted_batch = []
        for ids_single in batch_input_ids:
            original_text = self.tokenizer.decode(ids_single, skip_special_tokens=True)
            if has_chat_template:
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": original_text},
                ]
                adapted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                adapted_text = f"{instruction}\n\n{original_text}"
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

    def _make_adapter(self, tokenizer: PreTrainedTokenizer) -> Callable[[Any, Any], Any]:
        """Return `Callable[(input_ids, TextMemory), steered_input_ids]` for FeedbackScorer."""

        def adapter(input_ids, memory):
            instruction = memory.instruction if isinstance(memory, TextMemory) and memory.instruction else None
            if not instruction:
                return input_ids
            return self._apply_instruction(input_ids, instruction)

        return adapter

    def _resolve_reflection_lm(self) -> Callable[[str], str]:
        if callable(self.reflection_lm):
            return self.reflection_lm
        load_kwargs = self.reflection_lm_kwargs or {}
        return _HFReflectionLM(self.reflection_lm, **load_kwargs)

    def _run_gepa_loop(
        self,
        initial: Candidate,
        scorer: FeedbackScorer,
        reflection_proposer: GEPAReflectionProposer,
        merge_proposer: MergeProposer | None,
        archive: PerInstanceParetoArchive,
        train_data: list[dict],
        val_data: list[dict],
    ) -> Candidate:
        rng = random.Random(self.seed)

        initial_traces = scorer.score([initial], val_data)
        archive.ingest([initial], initial_traces)
        budget_used = len(val_data)
        merge_invocations_used = 0
        step = 0

        while budget_used < self.max_metric_calls:
            step += 1
            parent = archive.select_for_mutation()

            do_merge = (
                merge_proposer is not None
                and merge_invocations_used < self.max_merge_invocations
                and step % self.merge_interval == 0
                and len(list(archive.members())) >= 2
            )

            if do_merge:
                children = merge_proposer.propose(parent, archive)
                merge_invocations_used += 1
            else:
                k = min(self.reflection_minibatch_size, len(train_data))
                minibatch = rng.sample(train_data, k)
                minibatch_traces = scorer.score([parent], minibatch)[0]
                budget_used += len(minibatch)

                if self.skip_perfect_score and minibatch_traces and all(
                    isinstance(t.score, (int, float)) and t.score >= self.perfect_score
                    for t in minibatch_traces
                ):
                    continue

                children = reflection_proposer.propose(
                    parent, archive, traces=minibatch_traces
                )

            if not children:
                continue

            children_traces = scorer.score(children, val_data)
            budget_used += len(children) * len(val_data)
            archive.ingest(children, children_traces)

        return archive.best()
