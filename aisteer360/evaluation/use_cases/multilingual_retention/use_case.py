import json
from pathlib import Path
from typing import Any

from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.generation_utils import batch_retry_generate

_EVAL_REQ_KEYS = ["language"]  # plus one of "prompt" or "text"


class MultilingualRetention(UseCase):
    """Multilingual retention evaluation use case.

    This use case evaluates a (potentially continually pre-trained) model on a union of prompts drawn from multiple
    languages.

    Typical workflow for catastrophic forgetting:
        - evaluation_data contains prompts in languages the base model already knows (e.g., English) plus new ones
          (e.g., French, German).
        - For each steering pipeline (baseline, LoRA, MER, ...), `generate()` asks the model to respond to these prompts.
        - `evaluate()` passes prompts, responses, languages, and ids to metrics

    Required evaluation data fields:
        - "id": unique identifier for each instance
        - "language": language/task label (e.g. "en", "fr", "de")
        - one of:
            * "prompt": fully-formed prompt to send to the model, OR
            * "text": raw text, which will be used as the prompt

    Optional fields:
        - "reference": reference text (for tasks like translation, QA, etc.)
    """

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]):
        """Validate that each instance has the required fields."""
        if "id" not in evaluation_data:
            raise ValueError("The evaluation data must include an 'id' key.")

        missing = [k for k in _EVAL_REQ_KEYS if k not in evaluation_data]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        if "prompt" not in evaluation_data and "text" not in evaluation_data:
            raise ValueError(
                "Each evaluation instance must have either a 'prompt' or 'text' field."
            )

        lang_val = evaluation_data["language"]
        if lang_val is None:
            raise ValueError(
                f"Field 'language' is null for example id={evaluation_data.get('id')}."
            )

    def _get_prompt(self, instance: dict[str, Any]) -> str:
        """Resolve prompt text from an evaluation instance."""
        if "prompt" in instance and instance["prompt"] is not None:
            return str(instance["prompt"])
        return str(instance["text"])

    def generate(
        self,
        model_or_pipeline,
        tokenizer,
        gen_kwargs: dict | None = None,
        runtime_overrides: dict[tuple[str, str], str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate model responses for multilingual prompts.

        Args:
            model_or_pipeline: Either a HuggingFace model or SteeringPipeline.
            tokenizer: Tokenizer for encoding/decoding text.
            gen_kwargs: Optional generation params
            runtime_overrides: Optional mapping for steering-time kwargs

        Returns:
            List of generation dicts, each containing:
                - "id": original evaluation id
                - "language": language/tag from evaluation_data
                - "prompt": text actually sent to the model
                - "response": raw decoded model output
                - "reference": optional reference text (if present in eval data)
        """
        if not self.evaluation_data:
            print("No evaluation data provided.")
            return []

        gen_kwargs = dict(gen_kwargs or {})

        # build prompt data
        prompt_data: list[dict[str, Any]] = []
        for instance in self.evaluation_data:
            prompt = self._get_prompt(instance)
            prompt_data.append(
                {
                    "id": instance["id"],
                    "prompt": prompt,
                    "language": instance["language"],
                    "reference": instance.get("reference"),
                }
            )

        responses = batch_retry_generate(
            prompt_data=prompt_data,
            model_or_pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            parse_fn=None,  # use string directly
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            evaluation_data=self.evaluation_data,
        )

        generations: list[dict[str, Any]] = []
        for inst, response in zip(prompt_data, responses):
            generations.append(
                {
                    "id": inst["id"],
                    "language": inst["language"],
                    "prompt": inst["prompt"],
                    "response": response,
                    "reference": inst.get("reference"),
                }
            )

        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluate model generations using configured metrics.

        Builds a flat eval_data dict and passes it to each Metric in `self.evaluation_metrics`.

        Args:
            generations: Output of `generate()`, one dict per evaluation instance.

        Returns:
            Dictionary mapping metric_name -> metric_result (arbitrary dicts).
        """
        if not generations:
            return {}

        eval_data = {
            "responses": [g["response"] for g in generations],
            "prompts": [g["prompt"] for g in generations],
            "languages": [g["language"] for g in generations],
            "ids": [g["id"] for g in generations],
            "references": [g.get("reference") for g in generations],
        }

        scores: dict[str, dict[str, Any]] = {}
        for metric in self.evaluation_metrics:
            scores[metric.name] = metric(**eval_data)

        return scores

    def export(self, profiles: dict[str, Any], save_dir) -> None:
        """Export evaluation profiles to (tabbed) JSON format."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=4, ensure_ascii=False)
