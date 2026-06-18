"""Top-level orchestrator for the vector calibration workbench."""
from __future__ import annotations

import datetime
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.algorithms.state_control._common.specs import ContrastivePairs
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector

from .calibration import CalibrationSweep
from .configs import CalibrationBuilderConfig
from .extraction import SteeringVectorExtractor
from .generation import ContrastivePairGenerator
from .results import CalibrationResult, GenerationResult

if TYPE_CHECKING:
    from .agent.providers.base import GenerationProvider, JudgeProvider

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
        self._run_dir: Path | None = None
        self._model = None
        self._tokenizer = None

    def _resolve_run_dir(
        self,
        create: bool = False,
        run_dir: str | Path | None = None,
    ) -> Path:
        """Return the active run directory, optionally creating a new one.

        Args:
            create: If True, create a new timestamped run directory (used by `run_generation`).
            run_dir: Explicit override. When set, use this path verbatim (creating it if missing).

        Returns:
            The active run directory path.

        Raises:
            ValueError: When `create=False`, no explicit `run_dir` is given, no active run dir exists, and no
                existing directory matches the configured behavior.
        """
        if run_dir is not None:
            self._run_dir = Path(run_dir)
            self._run_dir.mkdir(parents=True, exist_ok=True)
            return self._run_dir

        if self._run_dir is not None:
            return self._run_dir

        if self._save_dir is None:
            raise ValueError("No save_dir is configured; cannot resolve a run directory.")

        behavior = self.config.generation.behavior
        if create:
            ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
            self._run_dir = self._save_dir / f"{behavior}_{ts}"
            self._run_dir.mkdir(parents=True, exist_ok=True)
            return self._run_dir

        candidates = sorted(
            self._save_dir.glob(f"{behavior}_*"),
            key=lambda p: p.name,
            reverse=True,
        )
        candidates = [c for c in candidates if c.is_dir()]
        if not candidates:
            raise ValueError(
                f"No existing run directory found for behavior '{behavior}' in {self._save_dir}"
            )
        self._run_dir = candidates[0]
        return self._run_dir

    def _ensure_steered_model(self) -> None:
        """Load the steered model once."""
        if self._model is not None:
            return
        logger.info("Loading steered model: %s", self.config.steered_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.steered_model)
        self._tokenizer = ensure_pad_token(self._tokenizer)

        load_kwargs = dict(self.config.hf_model_kwargs)
        load_kwargs.setdefault("torch_dtype", "auto")

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.steered_model,
            device_map=self.config.device_map,
            **load_kwargs,
        )

    # ── stage 1 ──────────────────────────────────────────────────────

    def run_generation(
        self,
        on_progress: Callable[[int, int], None] | None = None,
        run_dir: str | Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
        generation_provider: "GenerationProvider | None" = None,
    ) -> GenerationResult:
        """Run the contrastive pair generation stage.

        Creates (or resumes, when `run_dir` is provided) a timestamped run directory under `save_dir` and writes
        each completed batch to `{run_dir}/pairs.jsonl`.
        """
        active_run_dir = self._resolve_run_dir(create=True, run_dir=run_dir) if self._save_dir else None
        output_path = active_run_dir / "pairs.jsonl" if active_run_dir else None

        if active_run_dir is not None:
            meta = {
                "behavior": self.config.generation.behavior,
                "generator_model": self.config.generation.generator_model,
                "positive_prompt": self.config.generation.positive_prompt,
                "negative_prompt": self.config.generation.negative_prompt,
                "created": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            meta_path = active_run_dir / "run_meta.json"
            if not meta_path.exists():
                meta_path.write_text(json.dumps(meta, indent=2))

        gen = ContrastivePairGenerator(self.config.generation, provider=generation_provider)
        behavior = self.config.generation.behavior

        gen_model_name = self.config.generation.generator_model
        if generation_provider is not None:
            result = gen.generate(
                on_progress=on_progress,
                output_path=output_path,
                behavior=behavior if output_path is not None else None,
                cancel_check=cancel_check,
            )
            return result
        if gen_model_name == self.config.steered_model:
            self._ensure_steered_model()
            result = gen.generate(
                model=self._model,
                tokenizer=self._tokenizer,
                on_progress=on_progress,
                output_path=output_path,
                behavior=behavior if output_path is not None else None,
                cancel_check=cancel_check,
            )
        else:
            result = gen.generate(
                on_progress=on_progress,
                output_path=output_path,
                behavior=behavior if output_path is not None else None,
                cancel_check=cancel_check,
            )

        return result

    # ── stage 2 ──────────────────────────────────────────────────────

    def run_extraction(
        self,
        pairs: ContrastivePairs | None = None,
        run_dir: str | Path | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> SteeringVector:
        """Run the steering vector extraction stage.

        Args:
            pairs: Contrastive pairs. If None, loads from `{run_dir}/pairs.jsonl`.
            run_dir: Explicit run directory override. If None, uses the active run dir or the most recent
                existing run dir for this behavior.
            on_progress: Optional `(completed, total)` callback fired as each forward-pass batch
                completes inside the estimator.

        Returns:
            Fitted `SteeringVector`.
        """
        active_run_dir = self._resolve_run_dir(create=False, run_dir=run_dir) if self._save_dir else None

        if pairs is None:
            if active_run_dir is None:
                raise ValueError(
                    "pairs is None and no save_dir is configured; nothing to load."
                )
            pairs = GenerationResult.load(active_run_dir / "pairs.jsonl").pairs

        self._ensure_steered_model()
        extractor = SteeringVectorExtractor(self.config.extraction)
        sv = extractor.extract(self._model, self._tokenizer, pairs, on_progress=on_progress)

        if active_run_dir is not None:
            sv.save(str(active_run_dir / f"{self.config.generation.behavior}.svec"))

        return sv

    # ── stage 3 ──────────────────────────────────────────────────────

    def run_calibration(
        self,
        steering_vector: SteeringVector | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        run_dir: str | Path | None = None,
        judge_provider: "JudgeProvider | None" = None,
    ) -> CalibrationResult:
        """Run the calibration sweep.

        Args:
            steering_vector: Pre-computed vector. If None, loads from the active run directory.
            on_progress: Progress callback.
            run_dir: Explicit run directory override. If None, uses the active run dir or the most recent
                existing run dir for this behavior.

        Returns:
            `CalibrationResult` with the full grid and peak cell.
        """
        active_run_dir = self._resolve_run_dir(create=False, run_dir=run_dir) if self._save_dir else None

        if steering_vector is None:
            if active_run_dir is None:
                raise ValueError(
                    "steering_vector is None and no save_dir is configured; nothing to load."
                )
            path = active_run_dir / f"{self.config.generation.behavior}.svec"
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
            save_dir=active_run_dir,
            on_progress=on_progress,
            judge_provider=judge_provider,
        )

        if active_run_dir is not None:
            result.save(active_run_dir / "calibration_result.json")

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

        if self._run_dir and (self._run_dir / "pairs.jsonl").exists():
            gen = GenerationResult.load(self._run_dir / "pairs.jsonl")
            rng = random.Random(42)
            n = min(cfg.n_eval_prompts, len(gen.seed_prompts_used))
            return rng.sample(gen.seed_prompts_used, n)

        raise ValueError(
            "No eval_prompts provided and no saved pairs to sample from."
        )
