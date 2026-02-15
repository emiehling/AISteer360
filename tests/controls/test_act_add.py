"""Tests for ActAdd (Activation Addition) control."""
import pytest
import torch

from aisteer360.algorithms.state_control.act_add import ActAdd, ActAddArgs
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.estimators import SinglePairEstimator


class TestActAddArgs:
    """Tests for ActAddArgs validation."""

    def test_valid_with_prompts(self):
        """Test creation with positive/negative prompts."""
        args = ActAddArgs(
            positive_prompt="Love",
            negative_prompt="Hate",
        )
        assert args.positive_prompt == "Love"
        assert args.negative_prompt == "Hate"
        assert args.steering_vector is None

    def test_valid_with_steering_vector(self):
        """Test creation with pre-computed steering vector."""
        sv = SteeringVector(
            model_type="llama",
            directions={0: torch.randn(5, 64)},  # positional [T=5, H=64]
            explained_variances={0: 1.0},
        )
        args = ActAddArgs(steering_vector=sv)
        assert args.steering_vector is sv
        assert args.positive_prompt is None
        assert args.negative_prompt is None

    def test_neither_source_raises(self):
        """Test that providing neither prompts nor vector raises."""
        with pytest.raises(ValueError, match="Provide either"):
            ActAddArgs()

    def test_both_sources_raises(self):
        """Test that providing both prompts and vector raises."""
        sv = SteeringVector(
            model_type="llama",
            directions={0: torch.randn(5, 64)},
            explained_variances={0: 1.0},
        )
        with pytest.raises(ValueError, match="Provide either"):
            ActAddArgs(
                steering_vector=sv,
                positive_prompt="Love",
                negative_prompt="Hate",
            )

    def test_partial_prompts_raises(self):
        """Test that providing only one prompt raises."""
        with pytest.raises(ValueError, match="Provide either"):
            ActAddArgs(positive_prompt="Love")
        with pytest.raises(ValueError, match="Provide either"):
            ActAddArgs(negative_prompt="Hate")

    def test_negative_layer_id_raises(self):
        """Test that negative layer_id raises."""
        with pytest.raises(ValueError, match="layer_id must be >= 0"):
            ActAddArgs(
                positive_prompt="Love",
                negative_prompt="Hate",
                layer_id=-1,
            )

    def test_negative_alignment_raises(self):
        """Test that negative alignment raises."""
        with pytest.raises(ValueError, match="alignment must be >= 0"):
            ActAddArgs(
                positive_prompt="Love",
                negative_prompt="Hate",
                alignment=-1,
            )

    def test_default_values(self):
        """Test default parameter values."""
        args = ActAddArgs(positive_prompt="A", negative_prompt="B")
        assert args.layer_id is None
        assert args.multiplier == 1.0
        assert args.alignment == 1
        assert args.normalize_vector is False
        assert args.use_norm_preservation is False


class TestSinglePairEstimator:
    """Tests for SinglePairEstimator."""

    def test_produces_positional_steering_vector(self, model_and_tokenizer):
        """Test that estimator produces [T, H] steering vectors."""
        model, tokenizer = model_and_tokenizer

        estimator = SinglePairEstimator()
        sv = estimator.fit(
            model,
            tokenizer,
            positive_prompt="Love",
            negative_prompt="Hate",
        )

        assert isinstance(sv, SteeringVector)
        assert sv.model_type == model.config.model_type

        # check directions are 2D [T, H]
        for layer_id, direction in sv.directions.items():
            assert direction.ndim == 2
            assert direction.size(1) == model.config.hidden_size

        # both prompts are short, so T should be small but > 0
        assert sv.num_tokens > 0
        # directions should be positional (T > 1 for multi-token prompts)
        # for very short prompts this might be T=1, so just check structure
        assert sv.is_positional or sv.num_tokens == 1

    def test_layer_ids_filter(self, model_and_tokenizer):
        """Test that layer_ids parameter filters directions."""
        model, tokenizer = model_and_tokenizer

        estimator = SinglePairEstimator()
        sv = estimator.fit(
            model,
            tokenizer,
            positive_prompt="Love",
            negative_prompt="Hate",
            layer_ids=[0, 1],
        )

        assert set(sv.directions.keys()) == {0, 1}


class TestActAdd:
    """Tests for ActAdd control."""

    def test_steer_with_prompts(self, model_and_tokenizer):
        """Test steering with prompt pair extracts vector and builds transform."""
        model, tokenizer = model_and_tokenizer

        control = ActAdd(
            positive_prompt="Love",
            negative_prompt="Hate",
        )
        control.steer(model, tokenizer)

        assert control._steering_vector is not None
        assert control._transform is not None
        assert len(control._layer_names) > 0

    def test_steer_with_precomputed_vector(self, model_and_tokenizer):
        """Test steering with pre-computed steering vector."""
        model, tokenizer = model_and_tokenizer

        # create a positional steering vector
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        sv = SteeringVector(
            model_type=model.config.model_type,
            directions={i: torch.randn(3, hidden_size) for i in range(num_layers)},
            explained_variances={i: 1.0 for i in range(num_layers)},
        )

        control = ActAdd(steering_vector=sv)
        control.steer(model, tokenizer)

        assert control._steering_vector is sv

    def test_get_hooks_returns_forward_hook(self, model_and_tokenizer):
        """Test that get_hooks returns a forward hook specification."""
        model, tokenizer = model_and_tokenizer

        control = ActAdd(
            positive_prompt="Love",
            negative_prompt="Hate",
        )
        control.steer(model, tokenizer)

        input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
        hooks = control.get_hooks(input_ids)

        assert "forward" in hooks
        assert len(hooks["forward"]) == 1
        assert "module" in hooks["forward"][0]
        assert "hook_func" in hooks["forward"][0]

    def test_layer_id_heuristic(self, model_and_tokenizer):
        """Test that layer_id heuristic uses ~20% depth."""
        model, tokenizer = model_and_tokenizer

        control = ActAdd(
            positive_prompt="Love",
            negative_prompt="Hate",
        )
        control.steer(model, tokenizer)

        num_layers = len(control._layer_names)
        expected_layer = max(1, num_layers // 5)
        assert control._layer_id == expected_layer

    def test_explicit_layer_id(self, model_and_tokenizer):
        """Test that explicit layer_id overrides heuristic."""
        model, tokenizer = model_and_tokenizer

        control = ActAdd(
            positive_prompt="Love",
            negative_prompt="Hate",
            layer_id=1,  # use layer 1 (test models have 2 layers)
        )
        control.steer(model, tokenizer)

        assert control._layer_id == 1

    def test_normalize_vector(self, model_and_tokenizer):
        """Test that normalize_vector normalizes per-token directions."""
        model, tokenizer = model_and_tokenizer

        control = ActAdd(
            positive_prompt="Love",
            negative_prompt="Hate",
            normalize_vector=True,
        )
        control.steer(model, tokenizer)

        # check that directions at the target layer are normalized per-token
        direction = control._steering_vector.directions[control._layer_id]
        norms = direction.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_registry_entry(self):
        """Test that registry entry is correctly configured."""
        from aisteer360.algorithms.state_control.act_add import STEERING_METHOD

        assert STEERING_METHOD["category"] == "state_control"
        assert STEERING_METHOD["name"] == "act_add"
        assert STEERING_METHOD["control"] is ActAdd
        assert STEERING_METHOD["args"] is ActAddArgs
