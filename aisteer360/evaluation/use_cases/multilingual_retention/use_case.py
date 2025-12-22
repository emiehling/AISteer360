import json
from pathlib import Path
from typing import Any

from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.generation_utils import batch_retry_generate

_EVALUATION_REQ_KEYS = ["language"]


class MultilingualRetention(UseCase):
    """Multilingual retention evaluation use case using the OSCAR dataset.

    Evaluates a (potentially continually pre-trained) model on a union of prompts drawn from multiple languages. Typical
    workflow involves generating responses across base and additional languages, then passing prompts, responses,
    languages, and ids to evaluation metrics.

    The evaluation data should contain prompts with language labels where models respond to multilingual inputs. Each
    instance must include either a 'prompt' field (fully-formed prompt) or a 'text' field (raw text used as prompt).

    Attributes:
        evaluation_data: List of instances containing prompts and language metadata.
    """

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]):
        """Validates that evaluation data contains required fields for multilingual evaluation.

        Ensures each data instance has the necessary keys and non-null values for the evaluation.

        Args:
            evaluation_data: Dictionary containing a single evaluation instance with prompt/text and language information.

        Raises:
            ValueError: If required keys ('id', 'language') are missing, if neither 'prompt' nor 'text' is present,
                or if any required fields contain null values.
        """
        if "id" not in evaluation_data:
            raise ValueError("The evaluation data must include an 'id' key.")

        missing_keys = [k for k in _EVALUATION_REQ_KEYS if k not in evaluation_data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        if "prompt" not in evaluation_data and "text" not in evaluation_data:
            raise ValueError("Each evaluation instance must have either a 'prompt' or 'text' field.")

        lang_val = evaluation_data["language"]
        if lang_val is None:
            raise ValueError(f"Field 'language' is null for example id={evaluation_data.get('id')}.")

    def _get_prompt(self, instance: dict[str, Any]) -> str:
        """Resolves prompt text from an evaluation instance.

        Args:
            instance: Dictionary containing either a 'prompt' or 'text' field.

        Returns:
            The prompt string to send to the model.
        """
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
        """Generates model responses for multilingual prompts.

        Processes evaluation data to create prompts and generates model responses for each language instance.

        Args:
            model_or_pipeline: Either a HuggingFace model or SteeringPipeline instance to use for generation.
            tokenizer: Tokenizer for encoding/decoding text.
            gen_kwargs: Optional generation parameters passed to the model's generate method.
            runtime_overrides: Optional runtime parameter overrides for steering controls, structured as {(pipeline_name, param_name): value}.

        Returns:
            List of generation dictionaries, each containing:

                - "id": Original evaluation id
                - "language": Language/tag from evaluation_data
                - "prompt": Model input text
                - "response": Raw decoded model output
                - "reference": Optional reference text (if present in eval data)
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
            parse_fn=None,
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
        """Evaluates generated responses using configured metrics.

        Extracts responses, prompts, languages, and references from generations and computes scores using all
        evaluation metrics specified during initialization.

        Args:
            generations: List of generation dictionaries returned by the `generate()` method, each containing
                response, prompt, language, id, and optional reference fields.

        Returns:
            Dictionary of scores keyed by `metric_name`.
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

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        """Exports evaluation profiles to (tabbed) JSON format."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=4, ensure_ascii=False)
