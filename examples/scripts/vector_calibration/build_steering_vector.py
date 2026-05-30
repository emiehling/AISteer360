"""Build a calibrated steering vector end to end as a plain sequence of function calls:

    generation  -> contrastive response pairs              (ContrastivePairGenerator)
    extraction  -> per-layer steering directions           (SteeringVectorExtractor)
    calibration -> best (layer, multiplier) operating point (CalibrationSweep)

The model is loaded once and shared across all three stages. By default everything runs locally via
Hugging Face (no API keys); swap in `anthropic`/`openai` providers on `GenerationConfig`/`JudgeConfig`
if you prefer hosted models for the generator and judge.

The sweep grid here is intentionally small so the example finishes quickly; widen `SweepGrid` for a
real calibration.

Run:
    python examples/scripts/vector_calibration/build_steering_vector.py

This is the code version of the interactive vector-calibration workbench.
"""
from __future__ import annotations

import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.algorithms.state_control.common.specs import ContrastivePairs
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.workbenches.vector_calibration.calibration import CalibrationSweep
from aisteer360.workbenches.vector_calibration.configs import (
    CalibrationConfig,
    ExtractionConfig,
    GenerationConfig,
    JudgeConfig,
    QualityGate,
    SweepGrid,
)
from aisteer360.workbenches.vector_calibration.extraction import SteeringVectorExtractor
from aisteer360.workbenches.vector_calibration.generation import ContrastivePairGenerator
from aisteer360.workbenches.vector_calibration.results import CalibrationResult

logger = logging.getLogger(__name__)

# a small, open, chat-tuned model so the example needs no auth and fits on modest hardware.
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BEHAVIOR = "warmth"

# seed prompts steer the generator into producing a contrastive pair per prompt (warm vs cold).
SEED_PROMPTS = [
    "My flight got cancelled and I'm stuck at the airport overnight.",
    "I just submitted my first research paper.",
    "I think I bombed my job interview today.",
    "My dog has been sick all week.",
    "I finally paid off my student loans.",
    "I'm nervous about moving to a new city next month.",
    "I burned dinner right before the guests arrived.",
    "I got promoted at work.",
]

# held-out prompts the calibration sweep generates on and scores at each (layer, multiplier) cell.
EVAL_PROMPTS = [
    "I had a really rough day.",
    "I'm feeling overwhelmed with everything going on.",
    "Can you help me figure out what to do next?",
    "Nothing seems to be going right lately.",
]


def load_model(model_id: str = MODEL, device_map: str = "auto"):
    """Load the steered model and tokenizer once for reuse across all three stages."""
    logger.info("Loading model: %s", model_id)
    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(model_id))
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype="auto")
    model.eval()
    return model, tokenizer


def generate_pairs(model, tokenizer) -> ContrastivePairs:
    """Stage 1: produce contrastive (positive, negative) response pairs from the seed prompts."""
    config = GenerationConfig(
        generator_model=MODEL,
        behavior=BEHAVIOR,
        positive_prompt="You are deeply warm, empathetic, and emotionally supportive.",
        negative_prompt="You are cold, clinical, and emotionally detached.",
        seed_prompts=SEED_PROMPTS,
        max_new_tokens=64,
    )
    generator = ContrastivePairGenerator(config)
    # passing the preloaded model reuses it instead of loading a separate generator copy.
    result = generator.generate(model=model, tokenizer=tokenizer)
    return result.pairs


def extract_vector(model, tokenizer, pairs: ContrastivePairs) -> SteeringVector:
    """Stage 2: fit a per-layer steering direction from the contrastive pairs' hidden states."""
    config = ExtractionConfig(
        method="mean_diff",  # mean of (positive - negative) activations per layer
        accumulate="last_token",
        normalize=True,
    )
    extractor = SteeringVectorExtractor(config)
    return extractor.extract(model, tokenizer, pairs)


def calibrate(model, tokenizer, steering_vector: SteeringVector) -> CalibrationResult:
    """Stage 3: sweep (layer, multiplier), generate steered text, and judge it to find the peak."""
    config = CalibrationConfig(
        judge=JudgeConfig(
            model=MODEL,
            rating_scale=[
                (1, "cold, curt, or impersonal"),
                (3, "neutral, matter-of-fact"),
                (5, "warm, empathetic, and emotionally supportive"),
            ],
        ),
        sweep=SweepGrid(
            multiplier_range=(-2.0, 2.0),
            multiplier_step=0.5,
            layer_range=(6, 14),  # a middle band of layers keeps the demo grid small
            layer_step=2,
        ),
        quality_gate=QualityGate(coherence_threshold=0.8),
        max_new_tokens=64,
    )
    sweep = CalibrationSweep(config)
    # align the vector to the model's device/dtype before applying it in the forward hook.
    sv = steering_vector.to(model.device, dtype=model.dtype)
    # no judge_provider passed -> CalibrationSweep builds a local HF judge from the config above.
    return sweep.run(
        model=model,
        tokenizer=tokenizer,
        steering_vector=sv,
        eval_prompts=EVAL_PROMPTS,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    model, tokenizer = load_model()

    pairs = generate_pairs(model, tokenizer)
    logger.info("Stage 1 complete: %d contrastive pairs", len(pairs.positives))

    steering_vector = extract_vector(model, tokenizer, pairs)
    logger.info("Stage 2 complete: directions for %d layers", len(steering_vector.directions))

    result = calibrate(model, tokenizer, steering_vector)
    logger.info("Stage 3 complete: %d cells evaluated", len(result.cells))

    peak = result.peak_cell
    if peak is None:
        print("\nNo cell passed the quality gate; try a wider multiplier range or a lower threshold.")
    else:
        print(
            f"\nPeak operating point:"
            f"\n  layer       = {peak.layer}"
            f"\n  multiplier  = {peak.multiplier}"
            f"\n  score_delta = {peak.score_delta:+.3f} (baseline score {result.baseline_score:.3f})"
            f"\n  coherence   = {peak.coherence:.3f}"
        )


if __name__ == "__main__":
    main()
