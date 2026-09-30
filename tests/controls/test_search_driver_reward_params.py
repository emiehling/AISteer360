"""Row-scoped `reward_params` on `SearchDriver`: normalization of the two delivery forms and
the value reaching the scorer's params, including through `SampleSequenceScorer`.

Hub-free, using the tiny-model fixtures and the lazy-init `_pipeline` pattern.
"""
import pytest
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.best_of_n.control import BestOfN
from steerability.algorithms.output_control.common.drivers.search import _resolve_reward_params
from steerability.algorithms.output_control.common.scorers.sample import SampleSequenceScorer
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _pipeline(controls, model=None, tokenizer=None):
    if model is None:
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    if tokenizer is None:
        tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, model=model, tokenizer=tokenizer)
    pipeline.steer()
    return pipeline, model, tokenizer


class TestResolveRewardParams:
    def test_mapping_is_one_rows_value(self):
        assert _resolve_reward_params({"reward_params": {"target": "Yes"}}) == [{"target": "Yes"}]

    def test_mapping_applies_to_every_row(self):
        resolved = _resolve_reward_params({"reward_params": {"target": "Yes"}}, batch_size=2)
        assert resolved == [{"target": "Yes"}, {"target": "Yes"}]
        assert resolved[0] is not resolved[1]

    def test_mapping_is_copied(self):
        source = {"target": "Yes"}
        resolved = _resolve_reward_params({"reward_params": source})
        resolved[0]["target"] = "No"
        assert source == {"target": "Yes"}

    def test_one_element_sequence_is_the_rows_value(self):
        assert _resolve_reward_params({"reward_params": [{"reference": "Paris"}]}) == [{"reference": "Paris"}]

    def test_sequence_holds_one_mapping_per_row(self):
        resolved = _resolve_reward_params({"reward_params": [{"a": 1}, None]}, batch_size=2)
        assert resolved == [{"a": 1}, {}]

    def test_none_value_gives_empty(self):
        assert _resolve_reward_params({"reward_params": None}, batch_size=2) == [{}, {}]

    def test_singleton_none_element_gives_empty(self):
        assert _resolve_reward_params({"reward_params": [None]}) == [{}]

    def test_missing_key_gives_empty(self):
        assert _resolve_reward_params({}) == [{}]

    def test_sequence_of_another_length_raises_value_error(self):
        with pytest.raises(ValueError, match="one mapping per prompt row"):
            _resolve_reward_params({"reward_params": [{"a": 1}, {"b": 2}]}, batch_size=1)

    def test_empty_sequence_raises_value_error(self):
        with pytest.raises(ValueError, match="length 0"):
            _resolve_reward_params({"reward_params": []})

    def test_non_mapping_element_raises_type_error(self):
        with pytest.raises(TypeError, match="rows must be mappings"):
            _resolve_reward_params({"reward_params": ["not a mapping"]})

    def test_scalar_value_raises_type_error(self):
        with pytest.raises(TypeError, match="mapping or a sequence of mappings"):
            _resolve_reward_params({"reward_params": 7})

    def test_string_value_raises_type_error(self):
        with pytest.raises(TypeError, match="mapping or a sequence of mappings"):
            _resolve_reward_params({"reward_params": "reference"})


class TestRewardParamsReachScorer:
    def _recording_scorer(self, seen):
        def scorer(prompt, continuations, params):
            seen.append(params)
            return [0.0] * len(continuations)
        return scorer

    def test_mapping_form_reaches_params(self):
        seen: list[dict] = []
        bon = BestOfN(n=2, scorer=self._recording_scorer(seen))
        pipeline, _, tokenizer = _pipeline([bon])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        pipeline.generate(
            input_ids=prompt,
            runtime_kwargs={"reward_params": {"target": "Yes"}},
            max_new_tokens=3,
            do_sample=True,
            eos_token_id=None,
        )
        assert seen[0]["target"] == "Yes"
        # the driver's own search keys are present
        assert seen[0]["num_candidates"] == 2
        assert seen[0]["keep_k"] == 1
        assert seen[0]["max_iterations"] == 1
        assert "segment_len" in seen[0]

    def test_one_element_sequence_form_reaches_params(self):
        seen: list[dict] = []
        bon = BestOfN(n=2, scorer=self._recording_scorer(seen))
        pipeline, _, tokenizer = _pipeline([bon])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        pipeline.generate(
            input_ids=prompt,
            runtime_kwargs={"reward_params": [{"target": "Yes"}]},
            max_new_tokens=3,
            do_sample=True,
            eos_token_id=None,
        )
        assert seen[0]["target"] == "Yes"
        assert seen[0]["num_candidates"] == 2
        assert "segment_len" in seen[0]

    def test_sequence_form_reaches_each_row_of_a_batch(self):
        seen: list[tuple[str, str]] = []

        def scorer(prompt, continuations, params):
            seen.append((prompt, params["target"]))
            return [0.0] * len(continuations)

        pipeline, _, tokenizer = _pipeline([BestOfN(n=2, scorer=scorer)])
        batch = tokenizer(["the cat sat", "dog"], return_tensors="pt", padding=True)
        pipeline.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            runtime_kwargs={"reward_params": [{"target": "first"}, {"target": "second"}]},
            max_new_tokens=3,
            do_sample=True,
            eos_token_id=None,
        )
        assert seen == [("the cat sat", "first"), ("dog", "second")]

    def test_absent_reward_params_leaves_search_keys_only(self):
        seen: list[dict] = []
        bon = BestOfN(n=2, scorer=self._recording_scorer(seen))
        pipeline, _, tokenizer = _pipeline([bon])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        pipeline.generate(input_ids=prompt, max_new_tokens=3, do_sample=True, eos_token_id=None)
        assert set(seen[0]) == {"segment_len", "num_candidates", "keep_k", "max_iterations"}


class TestSampleSequenceScorerRow:
    def test_per_row_reference_reaches_the_row(self):
        seen_rows: list[dict] = []

        def row_scorer(response, row):
            seen_rows.append(dict(row))
            return 0.0

        bon = BestOfN(n=2, scorer=SampleSequenceScorer(row_scorer))
        pipeline, _, tokenizer = _pipeline([bon])
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        pipeline.generate(
            input_ids=prompt,
            runtime_kwargs={"reward_params": [{"reference": "Paris"}]},
            max_new_tokens=3,
            do_sample=True,
            eos_token_id=None,
        )
        assert seen_rows
        for row in seen_rows:
            assert row["reference"] == "Paris"
            assert "input" in row
