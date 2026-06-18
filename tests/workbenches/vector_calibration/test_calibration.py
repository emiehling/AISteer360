"""Tests for CalibrationSweep performance optimizations (O1-O5).

Covers:

  - O1: `multiplier == 0` cells reuse the cached baseline (no steered generation).
  - O2: linear `_compute_coherence` matches the original quadratic implementation.
  - O3: eval prompts are tokenized once for the whole sweep.
  - O4: the per-cell checkpoint drops `generations` and round-trips scalar fields.
  - O5: unambiguously degenerate cells are pre-screened out of the full evaluation.

O1/O5 use a monkeypatched orchestration harness (no model). O3 uses a tiny CPU model.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.workbenches.common.agent.providers.base import JudgeProvider
from aisteer360.workbenches.vector_calibration import calibration as calib_mod
from aisteer360.workbenches.vector_calibration.calibration import CalibrationSweep, _Baseline
from aisteer360.workbenches.vector_calibration.configs import (
    CalibrationConfig,
    JudgeConfig,
    QualityGate,
    SweepGrid,
)
from aisteer360.workbenches.vector_calibration.results import CellResult


# ── O2: linear coherence ────────────────────────────────────────────────


def _coherence_reference(responses: list[str]) -> float:
    """The original O(unique x total) implementation, kept here as an oracle."""
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
        most_common = max(tokens.count(t) for t in set(tokens))
        if most_common / len(tokens) > 0.8 and len(tokens) > 4:
            continue
        good += 1
    return good / len(responses)


@pytest.mark.parametrize(
    "responses",
    [
        [],
        [""],
        ["   ", "\n"],
        ["hello world this is a fine response"],
        ["a a a a a a a a"],  # looping; degenerate (ratio 1.0, len > 4)
        ["one two three four"],  # repeated-free, len == 4 (not > 4) → coherent
        ["good response here ok thanks", "x x x x x x x"],  # one good, one looping
        ["the the the the", "varied tokens make a sentence"],  # len 4 not > 4 → both good
    ],
)
def test_compute_coherence_matches_reference(responses: list[str]) -> None:
    new = CalibrationSweep._compute_coherence(None, None, None, responses, None)
    old = _coherence_reference(responses)
    assert new == old


def test_compute_coherence_degeneracy_rule() -> None:
    # >0.8 repetition AND len > 4 → degenerate (0.0); a 4-token repeat is too short to flag.
    assert CalibrationSweep._compute_coherence(None, None, None, ["a a a a a a"], None) == 0.0
    assert CalibrationSweep._compute_coherence(None, None, None, ["a a a a"], None) == 1.0


# ── O4: slim checkpoint ─────────────────────────────────────────────────


def _make_cell(layer: int, mult: float, n_generations: int = 0) -> CellResult:
    return CellResult(
        layer=layer,
        multiplier=mult,
        score_mean=1.5,
        score_delta=0.5,
        coherence=1.0,
        perplexity=float("nan"),
        perplexity_delta=float("nan"),
        coherent=True,
        generations=[{"prompt": f"p{i}", "steered_text": "word " * 50} for i in range(n_generations)],
    )


def test_checkpoint_excludes_generations(tmp_path: Path) -> None:
    cells = [_make_cell(0, -0.25, n_generations=30), _make_cell(1, 0.5, n_generations=30)]
    CalibrationSweep._save_checkpoint(tmp_path, cells)

    raw = (tmp_path / "calibration_checkpoint.json").read_text()
    assert "generations" not in raw
    assert "steered_text" not in raw

    loaded = CalibrationSweep._load_checkpoint(tmp_path)
    assert set(loaded.keys()) == {(0, -0.25), (1, 0.5)}
    for key, cell in loaded.items():
        assert cell.generations == []
        original = next(c for c in cells if (c.layer, c.multiplier) == key)
        assert cell.score_mean == original.score_mean
        assert cell.score_delta == original.score_delta
        assert cell.coherent == original.coherent


def test_checkpoint_size_independent_of_generation_volume(tmp_path: Path) -> None:
    light = tmp_path / "light"
    heavy = tmp_path / "heavy"
    light.mkdir()
    heavy.mkdir()

    CalibrationSweep._save_checkpoint(light, [_make_cell(0, 0.25, n_generations=0)])
    CalibrationSweep._save_checkpoint(heavy, [_make_cell(0, 0.25, n_generations=200)])

    assert (light / "calibration_checkpoint.json").stat().st_size == (
        heavy / "calibration_checkpoint.json"
    ).stat().st_size


# ── O1 / O5: orchestration harness (no model) ───────────────────────────


class _StubJudge(JudgeProvider):
    """Scores every response with a constant value so deltas are deterministic."""

    def __init__(self, score: float = 4.0):
        self._value = score

    def score(self, prompts, responses, *, template, scale):
        scores = [self._value] * len(responses)
        return {"scores": scores, "mean_score": self._value, "raw_scores": [scores]}


def _make_config(multipliers: tuple[float, float], step: float, n_eval: int) -> CalibrationConfig:
    return CalibrationConfig(
        judge=JudgeConfig(model="test/model", criteria="Rate. Response: {response}"),
        sweep=SweepGrid(
            multiplier_range=multipliers,
            multiplier_step=step,
            layer_range=(0, 1),
            layer_step=1,
        ),
        quality_gate=QualityGate(coherence_threshold=0.95, perplexity_max_ratio=2.0),
        compute_perplexity=False,
        max_new_tokens=8,
        n_eval_prompts=n_eval,
    )


@pytest.fixture
def harness(monkeypatch):
    """A CalibrationSweep with model/tokenizer touchpoints stubbed out.

    Returns a builder that runs the sweep with a caller-supplied `gen_fn(layer_id, strength, prompts)`
    controlling steered text, and records every `_generate_with_hook` call as `(layer_id, strength, n)`.
    """
    eval_prompts = [f"prompt {i}" for i in range(6)]
    directions = {0: torch.ones(1, 4), 1: torch.ones(1, 4)}
    steering_vector = SteeringVector(model_type="llama", directions=directions)
    baseline = _Baseline(
        score=2.0,
        perplexity=float("nan"),
        texts=[f"baseline {i}" for i in range(len(eval_prompts))],
        scores=[2.0] * len(eval_prompts),
        reasons=[None] * len(eval_prompts),
    )

    model = SimpleNamespace(device=torch.device("cpu"))

    monkeypatch.setattr(
        calib_mod,
        "get_model_layer_list",
        lambda m: ([None, None], ["model.layers.0", "model.layers.1"]),
    )

    def _run(gen_fn):
        cfg = _make_config((-0.5, 0.5), 0.5, len(eval_prompts))
        sweep = CalibrationSweep(cfg)
        calls: list[tuple[int, float, int]] = []

        # eval prompts pass through untouched so the stubbed _generate_with_hook can read them.
        monkeypatch.setattr(sweep, "_prepare_eval_batches", lambda tok, prompts, bs, dev: list(prompts))
        monkeypatch.setattr(
            sweep,
            "_evaluate_baseline",
            lambda *a, **k: baseline,
        )

        def _fake_generate(
            *, model, tokenizer, eval_batches, layer_name, layer_id, transform, token_scope, max_new_tokens
        ):
            prompts = list(eval_batches)
            calls.append((layer_id, transform.strength, len(prompts)))
            texts = gen_fn(layer_id, transform.strength, prompts)
            return [{"prompt": p, "steered_text": t} for p, t in zip(prompts, texts)]

        monkeypatch.setattr(sweep, "_generate_with_hook", _fake_generate)

        result = sweep.run(
            model=model,
            tokenizer=None,
            steering_vector=steering_vector,
            eval_prompts=eval_prompts,
            judge_provider=_StubJudge(score=4.0),
        )
        return result, calls

    _run.eval_prompts = eval_prompts
    _run.baseline = baseline
    return _run


def test_o1_zero_multiplier_reuses_baseline(harness) -> None:
    # coherent text everywhere so nothing is pre-screened.
    def gen_fn(layer, strength, prompts):
        return [f"steered {i} sentence here" for i in range(len(prompts))]

    result, calls = harness(gen_fn)

    # _generate_with_hook is never invoked with strength == 0.
    assert all(abs(strength) > 1e-9 for _, strength, _ in calls)

    zero_cells = [c for c in result.cells if c.multiplier == 0.0]
    assert len(zero_cells) == 2  # one per layer
    for cell in zero_cells:
        assert cell.score_delta == 0.0
        assert cell.score_mean == harness.baseline.score
        assert cell.perplexity_delta == 0.0
        steered = [g["steered_text"] for g in cell.generations]
        assert steered == harness.baseline.texts


def test_o5_prescreens_degenerate_cells(harness) -> None:
    # strength 0.5 saturates → looping text; strength -0.5 stays coherent.
    def gen_fn(layer, strength, prompts):
        if strength > 0:
            return ["x x x x x x x x" for _ in prompts]
        return [f"a coherent sentence number {i}" for i in range(len(prompts))]

    result, calls = harness(gen_fn)

    by_strength: dict[float, list[int]] = {}
    for _, strength, n in calls:
        by_strength.setdefault(round(strength, 4), []).append(n)

    probe_n = min(4, len(harness.eval_prompts))
    full_n = len(harness.eval_prompts)

    # degenerate cells: only the probe runs (no full pass).
    assert by_strength.get(0.5) == [probe_n, probe_n]
    # coherent cells: probe + full pass.
    assert sorted(by_strength.get(-0.5)) == sorted([probe_n, full_n, probe_n, full_n])

    degenerate = [c for c in result.cells if c.multiplier == 0.5]
    assert degenerate, "expected saturated cells in the grid"
    for cell in degenerate:
        assert cell.coherent is False
        assert cell.score_mean != cell.score_mean  # NaN
        assert cell.coherence == 0.0

    # peak is drawn only from coherent cells and is never a pre-screened one.
    assert result.peak_cell is not None
    assert result.peak_cell.coherent is True
    assert result.peak_cell.multiplier != 0.5


# ── O3: tokenization caching (tiny CPU model) ───────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - network/cache dependent
        pytest.skip(f"Could not load {model_id}: {exc}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def test_o3_tokenizes_eval_prompts_once(tiny_model, monkeypatch) -> None:
    model, tokenizer = tiny_model
    hidden = model.config.hidden_size
    directions = {i: torch.ones(1, hidden) for i in range(model.config.num_hidden_layers)}
    steering_vector = SteeringVector(model_type="llama", directions=directions)

    eval_prompts = ["hello there", "how are you", "tell me a story"]

    cfg = _make_config((-0.5, 0.5), 0.5, len(eval_prompts))
    cfg.sweep.layer_range = (0, model.config.num_hidden_layers - 1)
    cfg.max_new_tokens = 4
    sweep = CalibrationSweep(cfg)

    calls = {"n": 0}
    original = calib_mod._tokenize_chat

    def _counting_tokenize(tok, prompts):
        calls["n"] += 1
        return original(tok, prompts)

    monkeypatch.setattr(calib_mod, "_tokenize_chat", _counting_tokenize)

    sweep.run(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector,
        eval_prompts=eval_prompts,
        judge_provider=_StubJudge(score=3.0),
    )

    # one tokenization for the full eval batch + one for the probe batch, regardless of grid size.
    assert calls["n"] == 2
