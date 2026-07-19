"""Calibration sweep: evaluate steering vectors across a layer x multiplier grid."""
from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.token_scope import (
    compute_prompt_lens,
    make_token_mask,
)
from aisteer360.algorithms.state_control._common.transforms import (
    AdditiveTransform,
    NormPreservingTransform,
)

from .configs import CalibrationConfig, JudgeConfig, QualityGate
from .results import CalibrationResult, CellResult

if TYPE_CHECKING:
    from aisteer360.workbenches.common.agent.providers.base import JudgeProvider

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "calibration_checkpoint.json"
_PPL_BATCH_CAP = 2  # Max batch size for perplexity forward passes (logits are [B, seq, vocab]).
_PROBE_PROMPTS = 4  # Number of eval prompts generated in the pre-screen probe phase.


@dataclass
class _EvalBatch:
    """A pre-tokenized batch of eval prompts, reused read-only across every cell.

    Eval prompts are invariant over the sweep, so the chat template and tokenization are applied once
    up front. `model.generate` does not mutate its `input_ids`/`attention_mask`, so the tensors here are
    safe to share across cells.

    Attributes:
        prompts: The raw prompt strings in this batch.
        inputs: Tokenized batch (`input_ids`, `attention_mask`) already moved to the model device.
        prompt_lens: Per-item prompt lengths, used to build the steering token mask.
        initial_seq_len: Input sequence length, used to seed the per-cell hook state.
    """

    prompts: list[str]
    inputs: Any
    prompt_lens: "torch.Tensor"
    initial_seq_len: int


@dataclass
class _Baseline:
    """Unsteered baseline, evaluated once and reused for `multiplier == 0` cells.

    Attributes:
        score: Mean judge score over the baseline generations.
        perplexity: Mean baseline perplexity (NaN when perplexity is disabled).
        texts: Per-prompt baseline generations.
        scores: Per-prompt judge scores.
        reasons: Per-prompt judge reasons (may be all `None`).
    """

    score: float
    perplexity: float
    texts: list[str]
    scores: list[float] = field(default_factory=list)
    reasons: list[str | None] = field(default_factory=list)


class CalibrationSweep:
    """Grid-sweep evaluation of a steering vector.

    For each `(layer, multiplier)` cell in the sweep grid:

      1. Register a forward hook that adds `multiplier * direction[layer]` to the residual stream at that layer.
      2. Generate responses to eval prompts under the hook.
      3. Score the responses with the judge model.
      4. Compute perplexity and coherence metrics.
      5. Apply the quality gate.

    The steered model is loaded once by the caller and hooks are swapped per cell. The judge model is loaded
    internally at the start of `run()` and released when the sweep completes.

    Supports checkpoint and resume: completed cells are written to disk after each cell, and skipped on restart.
    """

    def __init__(self, config: CalibrationConfig):
        self.config = config

    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        steering_vector: SteeringVector,
        eval_prompts: list[str],
        save_dir: str | Path | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        judge_provider: "JudgeProvider | None" = None,
    ) -> CalibrationResult:
        """Execute the full calibration sweep.

        Args:
            model: The steered model (already loaded).
            tokenizer: Corresponding tokenizer.
            steering_vector: Pre-computed steering vector.
            eval_prompts: Prompts to generate on at each cell.
            save_dir: Directory for checkpoints (created if missing).
            on_progress: Callback receiving a dict with keys `"completed"`, `"total"`, `"current_cell"`,
                `"elapsed_s"`.

        Returns:
            `CalibrationResult` with all cell evaluations.
        """
        cfg = self.config
        save_dir_path = Path(save_dir) if save_dir else None
        if save_dir_path:
            save_dir_path.mkdir(parents=True, exist_ok=True)

        layers, multipliers = self._build_grid(steering_vector)
        total_cells = len(layers) * len(multipliers)

        completed: dict[tuple[int, float], CellResult] = {}
        if save_dir_path:
            completed = self._load_checkpoint(save_dir_path)

        judge = self._build_judge(cfg.judge, provider=judge_provider)

        # tokenize the invariant eval prompts once; reused read-only for the baseline and every cell.
        eval_batches = self._prepare_eval_batches(tokenizer, eval_prompts, cfg.batch_size, model.device)
        probe_prompts = eval_prompts[:_PROBE_PROMPTS]
        probe_batches = self._prepare_eval_batches(tokenizer, probe_prompts, cfg.batch_size, model.device)

        baseline = self._evaluate_baseline(model, tokenizer, eval_batches, eval_prompts, judge, cfg)

        _, layer_names = get_model_layer_list(model)

        cells: list[CellResult] = list(completed.values())
        start_time = time.monotonic()
        done = len(cells)

        def _commit(cell: CellResult) -> None:
            nonlocal done
            cells.append(cell)
            completed[(cell.layer, cell.multiplier)] = cell
            done += 1
            if save_dir_path:
                self._save_checkpoint(save_dir_path, cells)
            if on_progress:
                on_progress(
                    {
                        "completed": done,
                        "total": total_cells,
                        "current_cell": {"layer": cell.layer, "multiplier": cell.multiplier},
                        "elapsed_s": time.monotonic() - start_time,
                    }
                )
            # Release cached GPU memory between cells to prevent fragmentation
            # from accumulating across hundreds of generate→perplexity cycles.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for layer in layers:
            if layer not in steering_vector.directions:
                continue

            for mult in multipliers:
                if (layer, mult) in completed:
                    continue

                # mult == 0 adds exactly zero under greedy decoding, so the steered output is identical
                # to the unsteered baseline; reuse it instead of re-generating and re-judging.
                if abs(mult) < 1e-9:
                    _commit(self._baseline_cell(layer, mult, baseline, model, tokenizer, eval_prompts))
                    continue

                transform = self._build_transform(
                    steering_vector, layer, mult, cfg.transform
                )

                # probe phase: generate on a small subset and skip the full pass for cells that are
                # unambiguously degenerate (every probe response empty or single-token looping). Such
                # cells fail the quality gate under full evaluation too, so they can never be the peak.
                probe_generations = self._generate_with_hook(
                    model=model,
                    tokenizer=tokenizer,
                    eval_batches=probe_batches,
                    layer_name=layer_names[layer],
                    layer_id=layer,
                    transform=transform,
                    token_scope=cfg.token_scope,
                    max_new_tokens=cfg.max_new_tokens,
                )
                probe_texts = [g["steered_text"] for g in probe_generations]
                probe_coherence = self._compute_coherence(
                    model, tokenizer, probe_prompts, probe_texts, baseline.texts
                )
                if probe_texts and probe_coherence == 0.0:
                    for i, gen in enumerate(probe_generations):
                        gen["baseline_text"] = baseline.texts[i] if i < len(baseline.texts) else None
                        gen["judge_score"] = None
                        gen["judge_reason"] = None
                    _commit(
                        CellResult(
                            layer=layer,
                            multiplier=mult,
                            score_mean=float("nan"),
                            score_delta=float("nan"),
                            coherence=probe_coherence,
                            perplexity=float("nan"),
                            perplexity_delta=float("nan"),
                            coherent=False,
                            generations=probe_generations,
                        )
                    )
                    continue

                generations = self._generate_with_hook(
                    model=model,
                    tokenizer=tokenizer,
                    eval_batches=eval_batches,
                    layer_name=layer_names[layer],
                    layer_id=layer,
                    transform=transform,
                    token_scope=cfg.token_scope,
                    max_new_tokens=cfg.max_new_tokens,
                )

                steered_texts = [g["steered_text"] for g in generations]
                scores, reasons = judge.score_batch(
                    prompts=eval_prompts, responses=steered_texts
                )

                score_mean = _nan_mean(scores)

                if cfg.compute_perplexity:
                    # Free generation KV-cache / Mamba state before the perplexity
                    # forward pass, which has a very different memory profile.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    perplexity = self._compute_mean_perplexity(
                        model, tokenizer, eval_prompts, steered_texts, batch_size=cfg.batch_size,
                    )
                else:
                    perplexity = float("nan")
                coherence = self._compute_coherence(
                    model, tokenizer, eval_prompts, steered_texts, baseline.texts
                )

                for i, gen in enumerate(generations):
                    gen["baseline_text"] = baseline.texts[i] if i < len(baseline.texts) else None
                    gen["judge_score"] = scores[i] if i < len(scores) else None
                    gen["judge_reason"] = reasons[i] if reasons and i < len(reasons) else None

                _commit(
                    CellResult(
                        layer=layer,
                        multiplier=mult,
                        score_mean=score_mean,
                        score_delta=score_mean - baseline.score,
                        coherence=coherence,
                        perplexity=perplexity,
                        perplexity_delta=perplexity - baseline.perplexity,
                        coherent=self._passes_gate(
                            coherence, perplexity, baseline.perplexity, cfg.quality_gate
                        ),
                        generations=generations,
                    )
                )

        del judge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        coherent_cells = [c for c in cells if c.coherent]
        peak = max(coherent_cells, key=lambda c: c.score_delta) if coherent_cells else None

        return CalibrationResult(
            cells=cells,
            baseline_score=baseline.score,
            baseline_perplexity=baseline.perplexity,
            peak_cell=peak,
            grid_shape=(len(layers), len(multipliers)),
            layers=layers,
            multipliers=multipliers,
            config=asdict(cfg),
        )

    # ── grid / transform helpers ─────────────────────────────────────

    def _build_grid(
        self, steering_vector: SteeringVector
    ) -> tuple[list[int], list[float]]:
        """Resolve the grid axes from config and the steering vector's available layers."""
        cfg = self.config.sweep

        available_layers = sorted(steering_vector.directions.keys())
        if not available_layers:
            raise ValueError("SteeringVector has no directions.")

        if cfg.layer_range is not None:
            start, end = cfg.layer_range
        else:
            start, end = available_layers[0], available_layers[-1]
        layers = [layer for layer in range(start, end + 1, cfg.layer_step) if layer in steering_vector.directions]

        lo, hi = cfg.multiplier_range
        if cfg.multiplier_step <= 0:
            raise ValueError("sweep.multiplier_step must be > 0.")
        n_steps = int(round((hi - lo) / cfg.multiplier_step))
        multipliers = [round(lo + i * cfg.multiplier_step, 4) for i in range(n_steps + 1)]
        return layers, multipliers

    @staticmethod
    def _build_transform(
        sv: SteeringVector,
        layer: int,
        multiplier: float,
        method: str,
    ):
        """Build a transform for a single (layer, multiplier) cell."""
        single_layer_dirs = {layer: sv.directions[layer]}
        transform = AdditiveTransform(single_layer_dirs, strength=multiplier)
        if method == "norm_preserving":
            transform = NormPreservingTransform(transform)
        elif method != "additive":
            raise ValueError(f"Unknown transform '{method}'.")
        return transform

    # ── generation with hook ─────────────────────────────────────────

    @staticmethod
    def _prepare_eval_batches(
        tokenizer: PreTrainedTokenizerBase,
        prompts: list[str],
        batch_size: int,
        device,
    ) -> list[_EvalBatch]:
        """Tokenize the eval prompts once into per-batch inputs moved onto `device`.

        The returned batches are invariant across the sweep and reused read-only; `model.generate` does
        not mutate its inputs, so the same tensors are safe to feed to the baseline pass and every cell.
        """
        batches: list[_EvalBatch] = []
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start: batch_start + batch_size]
            inputs = _tokenize_chat(tokenizer, batch_prompts).to(device)
            batches.append(
                _EvalBatch(
                    prompts=batch_prompts,
                    inputs=inputs,
                    prompt_lens=compute_prompt_lens(inputs["input_ids"]),
                    initial_seq_len=inputs["input_ids"].size(1),
                )
            )
        return batches

    def _generate_with_hook(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_batches: list[_EvalBatch],
        layer_name: str,
        layer_id: int,
        transform,
        token_scope: str,
        max_new_tokens: int,
    ) -> list[dict[str, Any]]:
        """Generate responses with a steering hook registered.

        Consumes pre-tokenized `eval_batches` (see `_prepare_eval_batches`); a fresh `hook_state` is
        created per batch since `position_offset` is written during generation.

        Returns:
            A list of dicts with keys `prompt`, `steered_text`.
        """
        generations: list[dict[str, Any]] = []

        for batch in eval_batches:
            batch_prompts = batch.prompts
            inputs = batch.inputs
            prompt_lens = batch.prompt_lens

            hook_state = {"position_offset": 0, "initial_seq_len": batch.initial_seq_len}

            def _hook(
                module,
                args,
                kwargs,
                output,
                *,
                _layer_id=layer_id,
                _transform=transform,
                _prompt_lens=prompt_lens,
                _scope=token_scope,
                _state=hook_state,
            ):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden is None:
                    return output

                seq_len = hidden.size(1)
                if seq_len < _state["initial_seq_len"]:
                    offset = _state["position_offset"]
                    _state["position_offset"] += seq_len
                else:
                    offset = 0
                    _state["position_offset"] = seq_len

                mask = make_token_mask(
                    _scope,
                    seq_len=seq_len,
                    prompt_lens=_prompt_lens.to(hidden.device),
                    position_offset=offset,
                )
                hidden = _transform.apply(hidden, layer_id=_layer_id, token_mask=mask)
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

            layer_module = model.get_submodule(layer_name)
            handle = layer_module.register_forward_hook(_hook, with_kwargs=True)
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            finally:
                handle.remove()

            prompt_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                output_ids[:, prompt_len:], skip_special_tokens=True
            )

            for prompt, text in zip(batch_prompts, decoded):
                generations.append({"prompt": prompt, "steered_text": text})

        return generations

    # ── quality gate ─────────────────────────────────────────────────

    @staticmethod
    def _passes_gate(
        coherence: float,
        perplexity: float,
        baseline_ppl: float,
        gate: QualityGate,
    ) -> bool:
        coh_ok = coherence >= gate.coherence_threshold
        if not math.isfinite(perplexity):
            return coh_ok
        ppl_ok = (
            perplexity <= baseline_ppl * gate.perplexity_max_ratio
            if baseline_ppl > 0 and math.isfinite(baseline_ppl)
            else True
        )
        return coh_ok and ppl_ok

    # ── baseline ─────────────────────────────────────────────────────

    def _evaluate_baseline(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_batches: list[_EvalBatch],
        prompts: list[str],
        judge: "_Judge",
        cfg: CalibrationConfig,
    ) -> _Baseline:
        """Generate with no steering and score it; returns a `_Baseline` reused for `mult == 0` cells."""
        baseline_texts: list[str] = []
        for batch in eval_batches:
            inputs = batch.inputs
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                output_ids[:, prompt_len:], skip_special_tokens=True
            )
            baseline_texts.extend(decoded)

        scores, reasons = judge.score_batch(prompts=prompts, responses=baseline_texts)
        baseline_score = _nan_mean(scores)

        if cfg.compute_perplexity:
            # Free generation KV-cache / Mamba state before perplexity forward passes.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            baseline_perplexity = self._compute_mean_perplexity(
                model, tokenizer, prompts, baseline_texts, batch_size=cfg.batch_size,
            )
        else:
            baseline_perplexity = float("nan")
        return _Baseline(
            score=baseline_score,
            perplexity=baseline_perplexity,
            texts=baseline_texts,
            scores=list(scores),
            reasons=list(reasons),
        )

    def _baseline_cell(
        self,
        layer: int,
        multiplier: float,
        baseline: _Baseline,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompts: list[str],
    ) -> CellResult:
        """Build a `CellResult` for a `multiplier == 0` cell directly from the cached baseline.

        Output is mathematically identical to a full evaluation at `mult == 0` (greedy decoding adds zero),
        so generation and judging are skipped and the baseline texts/scores are reused verbatim.
        """
        generations: list[dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            generations.append(
                {
                    "prompt": prompt,
                    "steered_text": baseline.texts[i] if i < len(baseline.texts) else "",
                    "baseline_text": baseline.texts[i] if i < len(baseline.texts) else None,
                    "judge_score": baseline.scores[i] if i < len(baseline.scores) else None,
                    "judge_reason": baseline.reasons[i] if i < len(baseline.reasons) else None,
                }
            )
        coherence = self._compute_coherence(
            model, tokenizer, prompts, baseline.texts, baseline.texts
        )
        return CellResult(
            layer=layer,
            multiplier=multiplier,
            score_mean=baseline.score,
            score_delta=0.0,
            coherence=coherence,
            perplexity=baseline.perplexity,
            perplexity_delta=0.0,
            coherent=self._passes_gate(
                coherence, baseline.perplexity, baseline.perplexity, self.config.quality_gate
            ),
            generations=generations,
        )

    # ── perplexity / coherence ───────────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def _compute_mean_perplexity(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompts: list[str],
        responses: list[str],
        batch_size: int = 8,
    ) -> float:
        """Mean perplexity of `responses` conditioned on their prompts.

        Computes per-response NLL on the response tokens only (prompt tokens are masked out of the loss).
        Processes prompt-response pairs in batches for efficiency.

        Note: The effective batch size is capped at ``_PPL_BATCH_CAP`` (default 2) regardless of
        the caller-supplied ``batch_size``, because perplexity computation materialises a
        ``[batch, seq_len, vocab_size]`` logits tensor whose memory footprint is orders of
        magnitude larger than the KV-cache used during generation.  Passing ``labels`` to the
        model lets it compute the loss internally (avoiding an extra ``.contiguous()`` copy of
        the shifted logits), but the logits themselves are still allocated during the forward
        pass.
        """
        if not responses:
            return float("nan")

        device = next(model.parameters()).device
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        full_ids_list: list[torch.Tensor] = []
        prompt_lens: list[int] = []
        for prompt, response in zip(prompts, responses):
            prompt_ids = _tokenize_chat(tokenizer, [prompt])["input_ids"][0]
            full_text = tokenizer.decode(prompt_ids, skip_special_tokens=False) + response
            full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
            if full_ids.size(0) <= prompt_ids.size(0):
                continue
            full_ids_list.append(full_ids)
            prompt_lens.append(int(prompt_ids.size(0)))

        if not full_ids_list:
            return float("nan")

        # Cap batch size: perplexity needs [B, seq, vocab] logits which is far larger than
        # the KV-cache used during generation.  batch_size=2 keeps peak alloc manageable even
        # for 128k-vocab models with long sequences.
        ppl_bs = min(batch_size, _PPL_BATCH_CAP)

        total_nll = 0.0
        total_tokens = 0

        for batch_start in range(0, len(full_ids_list), ppl_bs):
            batch_ids = full_ids_list[batch_start: batch_start + ppl_bs]
            batch_prompt_lens = prompt_lens[batch_start: batch_start + ppl_bs]
            max_len = max(t.size(0) for t in batch_ids)

            input_ids = torch.full(
                (len(batch_ids), max_len), pad_id, dtype=torch.long
            )
            attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
            labels = torch.full((len(batch_ids), max_len), -100, dtype=torch.long)
            for i, (ids, p_len) in enumerate(zip(batch_ids, batch_prompt_lens)):
                seq_len = ids.size(0)
                input_ids[i, :seq_len] = ids
                attention_mask[i, :seq_len] = 1
                labels[i, p_len:seq_len] = ids[p_len:]

            # The model shifts labels internally: shift_labels = labels[..., 1:].
            # Position 0 of every sequence is within the prompt (always -100), so the
            # shift doesn't reduce the count of scoreable tokens.  The model's loss is
            # the mean over count(shift_labels != -100) == count(labels != -100).
            n_tokens = int((labels != -100).sum().item())
            if n_tokens <= 0:
                continue

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Pass labels so the model computes cross-entropy internally.  This avoids
            # pulling out.logits into Python and calling .contiguous() on the shifted slice,
            # which would allocate a second [B, seq-1, vocab] tensor and double peak memory.
            # The model returns the mean loss over all non-(-100) label tokens.
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            batch_nll = float(out.loss) * n_tokens
            # Free the output (and its logits tensor) immediately.
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_nll += batch_nll
            total_tokens += n_tokens

        if total_tokens == 0:
            return float("nan")
        return float(math.exp(total_nll / total_tokens))

    @staticmethod
    def _compute_coherence(
        model: PreTrainedModel,  # noqa: ARG004
        tokenizer: PreTrainedTokenizerBase,  # noqa: ARG004
        prompts: list[str],  # noqa: ARG004
        responses: list[str],
        baseline_texts: list[str],  # noqa: ARG004
    ) -> float:
        """Cheap proxy for coherence: fraction of responses that are non-empty and non-degenerate.

        Degenerate here means: empty after stripping, or a single token repeated more than 80% of the time.
        Returns a value in `[0, 1]`.
        """
        if not responses:
            return 0.0
        good = 0
        for text in responses:
            stripped = text.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            if not tokens:
                continue
            most_common = Counter(tokens).most_common(1)[0][1]
            if most_common / len(tokens) > 0.8 and len(tokens) > 4:
                continue
            good += 1
        return good / len(responses)

    # ── judge ────────────────────────────────────────────────────────

    @staticmethod
    def _build_judge(
        judge_config: JudgeConfig,
        provider: "JudgeProvider | None" = None,
    ) -> "_Judge":
        """Build the judge adapter. Uses `provider` if given, else falls back to HF-local."""
        if provider is None:
            from aisteer360.workbenches.common.agent.providers.hf_local import HFJudgeProvider
            provider = HFJudgeProvider(config=asdict(judge_config))
        return _Judge(judge_config, provider=provider)

    # ── checkpointing ────────────────────────────────────────────────

    @staticmethod
    def _save_checkpoint(save_dir: Path, cells: list[CellResult]) -> None:
        path = save_dir / _CHECKPOINT_FILENAME
        # exclude the bulky per-prompt `generations` (mirroring CalibrationResult.save): the checkpoint
        # is rewritten in full after every cell, so carrying generation text makes it O(N^2) bytes.
        data = [
            {k: v for k, v in asdict(c).items() if k != "generations"}
            for c in cells
        ]
        path.write_text(json.dumps(data))

    @staticmethod
    def _load_checkpoint(save_dir: Path) -> dict[tuple[int, float], CellResult]:
        path = save_dir / _CHECKPOINT_FILENAME
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Could not parse checkpoint at %s; starting fresh.", path)
            return {}
        cells = [CellResult(**d) for d in data]
        return {(c.layer, c.multiplier): c for c in cells}


# ── helpers ──────────────────────────────────────────────────────────

def _tokenize_chat(tokenizer: PreTrainedTokenizerBase, prompts: list[str]):
    """Apply the tokenizer's chat template and return a padded batch on CPU."""
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    ]
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)


def _nan_mean(values: list[float]) -> float:
    """Mean over non-NaN entries; returns NaN when the list is empty or all NaN."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _render_judge_prompt(config: JudgeConfig) -> str:
    """Build the judge prompt template from `rating_scale` rows, or fall back to `criteria`."""
    if not config.rating_scale:
        return config.criteria

    lines = [f"{int(label)}: {description.strip()}" for label, description in config.rating_scale]
    lo, hi = config.scale

    parts: list[str] = []

    system_prompt = getattr(config, "system_prompt", "") or ""
    if system_prompt.strip():
        parts.append(system_prompt.strip())
        parts.append("")

    parts.append(
        "If the response is incoherent, nonsensical, unreadable, or consists of "
        "repeated/looping tokens, words, or phrases (even if individually well-formed), "
        "respond with:\n"
        '{{"score": -1, "reason": "incoherent"}}\n'
        "Otherwise, rate on this scale:"
    )
    parts.append("")
    parts.extend(lines)
    parts.append("")
    parts.append(
        f'Respond with JSON only: {{{{"score": <{lo}-{hi}>, "reason": "<one sentence>"}}}}'
    )
    parts.append("")
    parts.append("Response:\n{response}")

    return "\n".join(parts)


class _Judge:
    """Small adapter that routes scoring calls through a `JudgeProvider`.

    Reasons are left as `None` because the default judge template only captures a numeric score;
    custom rubrics that emit a `reason` field can be surfaced later.
    """

    def __init__(self, config: JudgeConfig, provider: "JudgeProvider"):
        template = _render_judge_prompt(config)
        if "{response}" not in template:
            raise ValueError("Judge prompt must include a '{response}' placeholder.")
        self.template = template
        self.scale = tuple(config.scale)
        self.provider = provider

    def score_batch(
        self, prompts: list[str], responses: list[str]
    ) -> tuple[list[float], list[str | None]]:
        result = self.provider.score(
            prompts=prompts,
            responses=responses,
            template=self.template,
            scale=self.scale,
        )
        scores = list(result["scores"])
        reasons: list[str | None] = [None] * len(scores)
        return scores, reasons