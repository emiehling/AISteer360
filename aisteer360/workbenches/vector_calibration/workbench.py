"""Top-level orchestrator for the vector calibration workbench."""
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.algorithms.state_control.common.specs import ContrastivePairs
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector

from .calibration import CalibrationSweep
from .configs import CalibrationBuilderConfig
from .extraction import SteeringVectorExtractor
from .generation import ContrastivePairGenerator
from .results import CalibrationResult, GenerationResult

logger = logging.getLogger(__name__)


class VectorCalibrationWorkbench:
    """End-to-end vector calibration workbench.

    Orchestrates three stages:

      1. **Generation** produce contrastive pairs using a generator LLM.
      2. **Extraction** fit a steering vector from those pairs using the steered model's hidden states.
      3. **Calibration** sweep over (layer, multiplier), generate steered text, and score with an LLM judge to
         find the optimal operating point.

    Each stage can be run independently or as part of the full pipeline. Intermediate artifacts are saved to
    `save_dir` for inspection and resume.

    Typical usage::

        workbench = VectorCalibrationWorkbench(config)

        # full pipeline
        result = workbench.run()

        # or stage-by-stage
        gen_result = workbench.run_generation()
        sv = workbench.run_extraction(pairs=gen_result.pairs)
        cal_result = workbench.run_calibration(steering_vector=sv)
    """

    def __init__(self, config: CalibrationBuilderConfig):
        self.config = config
        self._save_dir = Path(config.save_dir) if config.save_dir else None
        self._model = None
        self._tokenizer = None

    def _ensure_steered_model(self) -> None:
        """Load the steered model once."""
        if self._model is not None:
            return
        logger.info("Loading steered model: %s", self.config.steered_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.steered_model)
        self._tokenizer = ensure_pad_token(self._tokenizer)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.steered_model,
            device_map=self.config.device_map,
            **self.config.hf_model_kwargs,
        )

    # ── stage 1 ──────────────────────────────────────────────────────

    def run_generation(
        self,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> GenerationResult:
        """Run the contrastive pair generation stage.

        Loads the generator model (if different from the steered model), produces pairs, and saves them to
        `save_dir/pairs.json`.
        """
        gen = ContrastivePairGenerator(self.config.generation)

        gen_model_name = self.config.generation.generator_model
        if gen_model_name == self.config.steered_model:
            self._ensure_steered_model()
            result = gen.generate(
                model=self._model, tokenizer=self._tokenizer, on_progress=on_progress
            )
        else:
            result = gen.generate(on_progress=on_progress)

        if self._save_dir:
            result.save(self._save_dir / "pairs.json")

        return result

    # ── stage 2 ──────────────────────────────────────────────────────

    def run_extraction(
        self,
        pairs: ContrastivePairs | None = None,
    ) -> SteeringVector:
        """Run the steering vector extraction stage.

        Args:
            pairs: Contrastive pairs. If None, loads from `save_dir/pairs.json`.

        Returns:
            Fitted `SteeringVector`.
        """
        if pairs is None:
            if self._save_dir is None:
                raise ValueError(
                    "pairs is None and no save_dir is configured; nothing to load."
                )
            pairs = GenerationResult.load(self._save_dir / "pairs.json").pairs

        self._ensure_steered_model()
        extractor = SteeringVectorExtractor(self.config.extraction)
        sv = extractor.extract(self._model, self._tokenizer, pairs)

        if self._save_dir:
            sv.save(str(self._save_dir / f"{self.config.generation.behavior}.svec"))

        return sv

    # ── stage 3 ──────────────────────────────────────────────────────

    def run_calibration(
        self,
        steering_vector: SteeringVector | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> CalibrationResult:
        """Run the calibration sweep.

        Args:
            steering_vector: Pre-computed vector. If None, loads from `save_dir`.
            on_progress: Progress callback.

        Returns:
            `CalibrationResult` with the full grid and peak cell.
        """
        if steering_vector is None:
            if self._save_dir is None:
                raise ValueError(
                    "steering_vector is None and no save_dir is configured; nothing to load."
                )
            path = self._save_dir / f"{self.config.generation.behavior}.svec"
            steering_vector = SteeringVector.load(str(path))

        self._ensure_steered_model()
        sv = steering_vector.to(self._model.device, dtype=self._model.dtype)

        eval_prompts = self._resolve_eval_prompts()

        sweep = CalibrationSweep(self.config.calibration)
        result = sweep.run(
            model=self._model,
            tokenizer=self._tokenizer,
            steering_vector=sv,
            eval_prompts=eval_prompts,
            save_dir=self._save_dir,
            on_progress=on_progress,
        )

        if self._save_dir:
            result.save(self._save_dir / "calibration_result.json")

        return result

    # ── full pipeline ────────────────────────────────────────────────

    def run(
        self,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> CalibrationResult:
        """Run all three stages end-to-end.

        Args:
            on_progress: Callback receiving `(stage_name, stage_progress_data)`.
        """

        def _gen_progress(done, total):
            if on_progress:
                on_progress("generation", {"completed": done, "total": total})

        def _cal_progress(data):
            if on_progress:
                on_progress("calibration", data)

        gen_result = self.run_generation(on_progress=_gen_progress)
        sv = self.run_extraction(pairs=gen_result.pairs)
        result = self.run_calibration(steering_vector=sv, on_progress=_cal_progress)
        return result

    def cleanup(self) -> None:
        """Release the steered model's GPU resources."""
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── helpers ──────────────────────────────────────────────────────

    def _resolve_eval_prompts(self) -> list[str]:
        """Get evaluation prompts from config or saved pairs."""
        cfg = self.config.calibration
        if cfg.eval_prompts is not None:
            if isinstance(cfg.eval_prompts, str):
                raw_path = Path(cfg.eval_prompts)
                text = raw_path.read_text()
                if raw_path.suffix == ".jsonl":
                    prompts = []
                    for line in text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        prompts.append(obj if isinstance(obj, str) else obj["prompt"])
                    return prompts
                raw = json.loads(text)
                return [p if isinstance(p, str) else p["prompt"] for p in raw]
            return list(cfg.eval_prompts)

        if self._save_dir and (self._save_dir / "pairs.json").exists():
            gen = GenerationResult.load(self._save_dir / "pairs.json")
            rng = random.Random(42)
            n = min(cfg.n_eval_prompts, len(gen.seed_prompts_used))
            return rng.sample(gen.seed_prompts_used, n)

        raise ValueError(
            "No eval_prompts provided and no saved pairs to sample from."
        )
