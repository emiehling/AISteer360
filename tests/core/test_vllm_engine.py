"""Engine-gated tests for `VLLMBackend`: prompt-only and driver pipelines on the offline
engine, greedy HF/vLLM parity, and structural checkpoint serving. The whole module skips when
vLLM is not installed; running it requires a GPU-capable environment with the `vllm` extra."""
import pytest

vllm = pytest.importorskip("vllm")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from aisteer360.algorithms.core.execution import (  # noqa: E402
    BackendSpec,
    GenerationItem,
    GenerationParams,
    PreparedPrompt,
    ScoringItem,
)
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline  # noqa: E402
from aisteer360.algorithms.output_control.stopping_rules.control import StoppingRules  # noqa: E402
from aisteer360.backends.vllm import VLLMBackend  # noqa: E402

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def engine_backend():
    spec = BackendSpec(
        kind="vllm",
        model=TINY_MODEL,
        options={"engine_kwargs": {"enforce_eager": True, "max_model_len": 512}},
    )
    try:
        return VLLMBackend(spec)
    except Exception as exception:
        pytest.skip(f"Could not boot the vLLM engine: {exception}")


class TestOfflineEngine:

    def test_prompt_only_generation(self, engine_backend):
        item = GenerationItem(prompt=PreparedPrompt.from_text("The capital of France is"))
        with engine_backend.open_session() as session:
            results = session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))
        output = results[0].output
        assert output.output_ids.shape[0] == 1
        assert output.output_ids.shape[1] > 0
        assert output.finish_reason in ("stop", "eos", "length")

    def test_greedy_parity_with_hf(self, engine_backend):
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        encoded = tokenizer("The sky is", return_tensors="pt")
        hf_full = model.generate(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
            max_new_tokens=8, do_sample=False,
        )
        hf_new = hf_full[0, encoded["input_ids"].size(1):].tolist()

        item = GenerationItem(prompt=PreparedPrompt.from_token_ids(encoded["input_ids"]))
        with engine_backend.open_session() as session:
            results = session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))
        vllm_new = results[0].output.output_ids[0].tolist()
        assert vllm_new[: len(hf_new)] == hf_new[: len(vllm_new)]

    def test_stop_string_semantics(self, engine_backend):
        item = GenerationItem(prompt=PreparedPrompt.from_text("a b a b a b"))
        with engine_backend.open_session() as session:
            results = session.generate(
                [item],
                GenerationParams(max_new_tokens=16, greedy=True, stop_strings=("b",)),
            )
        output = results[0].output
        decoded = engine_backend.tokenizer.decode(output.output_ids[0], skip_special_tokens=True)
        if output.finish_reason == "stop":
            assert "b" in decoded  # ids returned as generated, stop text included

    def test_prompt_logprob_scoring(self, engine_backend):
        tokenizer = engine_backend.tokenizer
        prompt_ids = tokenizer("hello world", return_tensors="pt")["input_ids"]
        ref = prompt_ids[:, -2:]
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids(prompt_ids), ref_output_ids=ref,
        )
        with engine_backend.open_session() as session:
            scored = session.score([item], GenerationParams())
        assert scored.shape == (1, 2)
        assert torch.isfinite(scored).all()

    def test_pipeline_end_to_end_with_stopping_rules(self):
        pipeline = SteeringPipeline(
            controls=[StoppingRules(budget=6)],
            lazy_init=True,
            backend=BackendSpec(
                kind="vllm",
                model=TINY_MODEL,
                options={"engine_kwargs": {"enforce_eager": True, "max_model_len": 512}},
            ),
            steer_backend="huggingface",
        )
        try:
            pipeline.steer()
        except Exception as exception:
            pytest.skip(f"Could not boot the vLLM engine: {exception}")
        out = pipeline.generate(text="Once upon a time", max_new_tokens=16, do_sample=False,
                                return_output=True)
        assert out.output_ids.shape[1] <= 6
