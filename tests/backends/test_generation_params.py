"""Golden tests for `GenerationParams` rendering (doc 02 §4, §7)."""
import pytest

from aisteer360.backends.errors import UnsupportedGenerationParam
from aisteer360.backends.generation_params import GenerationParams

# a table of gen_kwargs sets; to_hf_kwargs must return each byte-identically
_HF_PASSTHROUGH_CASES = [
    {},
    {"max_new_tokens": 32},
    {"max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9},
    {"do_sample": False},
    {"do_sample": True, "temperature": 1.0},
    {"num_beams": 4},
    {"top_k": 50, "min_p": 0.05, "repetition_penalty": 1.1},
    {"num_return_sequences": 3},
    {"stop": ["</s>"]},
    {"seed": 123},
    {"return_full_sequence": True},
    {"output_hidden_states": True},
    {"bad_words_ids": [[1, 2]]},
    {"max_new_tokens": 16, "eta_cutoff": 0.001},  # unknown key still passes through on HF
    {"n": 2, "stop": "END", "seed": 0, "temperature": 0.0},
]


@pytest.mark.parametrize("gen_kwargs", _HF_PASSTHROUGH_CASES)
def test_to_hf_kwargs_is_exact_passthrough(gen_kwargs):
    params = GenerationParams.from_gen_kwargs(gen_kwargs)
    assert params.to_hf_kwargs() == gen_kwargs


def test_max_length_rejected():
    with pytest.raises(UnsupportedGenerationParam, match="max_length"):
        GenerationParams.from_gen_kwargs({"max_length": 128})


class TestOpenAIRendering:
    def test_max_new_tokens_maps_to_max_tokens(self):
        kwargs, extra = GenerationParams.from_gen_kwargs({"max_new_tokens": 64}).to_openai_kwargs()
        assert kwargs == {"max_tokens": 64}
        assert extra == {}

    def test_same_named_passthrough(self):
        kwargs, extra = GenerationParams.from_gen_kwargs(
            {"temperature": 0.5, "top_p": 0.8, "n": 2, "seed": 7, "stop": ["x"]}
        ).to_openai_kwargs()
        assert kwargs == {"temperature": 0.5, "top_p": 0.8, "n": 2, "seed": 7, "stop": ["x"]}
        assert extra == {}

    def test_greedy_maps_to_temperature_zero(self):
        kwargs, _ = GenerationParams.from_gen_kwargs({"do_sample": False}).to_openai_kwargs()
        assert kwargs["temperature"] == 0.0

    def test_num_return_sequences_maps_to_n(self):
        kwargs, _ = GenerationParams.from_gen_kwargs({"num_return_sequences": 4}).to_openai_kwargs()
        assert kwargs["n"] == 4

    def test_vllm_extensions_route_to_extra_body(self):
        kwargs, extra = GenerationParams.from_gen_kwargs(
            {"top_k": 40, "min_p": 0.1, "repetition_penalty": 1.2, "prompt_logprobs": 0}
        ).to_openai_kwargs()
        assert kwargs == {}
        assert extra == {"top_k": 40, "min_p": 0.1, "repetition_penalty": 1.2, "prompt_logprobs": 0}

    def test_hf_control_flags_dropped_silently(self):
        kwargs, extra = GenerationParams.from_gen_kwargs(
            {"max_new_tokens": 8, "return_full_sequence": True, "output_hidden_states": True}
        ).to_openai_kwargs()
        assert kwargs == {"max_tokens": 8}
        assert extra == {}

    @pytest.mark.parametrize("key, value", [("num_beams", 4), ("bad_words_ids", [[1]]), ("constraints", [1])])
    def test_unsupported_raises_when_strict(self, key, value):
        with pytest.raises(UnsupportedGenerationParam, match=key):
            GenerationParams.from_gen_kwargs({key: value}).to_openai_kwargs(strict=True)

    def test_unknown_key_raises_when_strict(self):
        with pytest.raises(UnsupportedGenerationParam, match="eta_cutoff"):
            GenerationParams.from_gen_kwargs({"eta_cutoff": 0.1}).to_openai_kwargs(strict=True)

    def test_non_strict_drops_with_warning(self):
        with pytest.warns(UserWarning, match="unsupported"):
            kwargs, extra = GenerationParams.from_gen_kwargs(
                {"max_new_tokens": 8, "num_beams": 4, "eta_cutoff": 0.1}
            ).to_openai_kwargs(strict=False)
        assert kwargs == {"max_tokens": 8}
        assert extra == {}


def test_replace_updates_canonical_and_raw():
    params = GenerationParams.from_gen_kwargs({"max_new_tokens": 10, "temperature": 0.5})
    updated = params.replace(max_new_tokens=5)
    assert updated.max_new_tokens == 5
    assert updated.to_hf_kwargs()["max_new_tokens"] == 5
    # original untouched
    assert params.max_new_tokens == 10
