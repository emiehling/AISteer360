import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.angular_steering import AngularSteering
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector


def create_dummy_steering_vector(model_type, hidden_size, num_layers):
    """Create a dummy K=2 steering vector for angular steering tests."""
    directions = {
        k: torch.randn(2, hidden_size)
        for k in range(num_layers)
    }
    explained_variances = {k: float(k) / num_layers for k in range(num_layers)}
    return SteeringVector(
        model_type=model_type,
        directions=directions,
        explained_variances=explained_variances,
    )


PROMPT_TEXT = "Tell me something interesting about the ocean."


@pytest.mark.parametrize("target_degree", [0.0, 90.0, 180.0, 270.0])
def test_angular_steering_generates(model_and_tokenizer, device: torch.device, target_degree: float):
    """Verify that AngularSteering steers and generates on each angle."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    model_type = model.config.model_type
    hidden_size = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd")
    )
    num_layers = (
        getattr(model.config, "num_hidden_layers", None)
        or getattr(model.config, "n_layer")
    )

    sv = create_dummy_steering_vector(model_type, hidden_size, num_layers)

    control = AngularSteering(
        steering_vector=sv,
        layer_id=0,
        target_degree=target_degree,
        adaptive_mode=1,
    )

    pipeline = SteeringPipeline(
        controls=[control],
        lazy_init=True,
        device_map=device,
    )
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    out_ids = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)

    assert isinstance(out_ids, torch.Tensor), "Output is not a torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(1) >= 1, "No new tokens generated"


@pytest.mark.parametrize("adaptive_mode", [0, 1])
def test_angular_steering_adaptive_modes(model_and_tokenizer, device: torch.device, adaptive_mode: int):
    """Verify both adaptive_mode=0 and adaptive_mode=1 run without error."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    model_type = model.config.model_type
    hidden_size = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd")
    )
    num_layers = (
        getattr(model.config, "num_hidden_layers", None)
        or getattr(model.config, "n_layer")
    )

    sv = create_dummy_steering_vector(model_type, hidden_size, num_layers)

    control = AngularSteering(
        steering_vector=sv,
        layer_id=0,
        target_degree=180.0,
        adaptive_mode=adaptive_mode,
    )

    pipeline = SteeringPipeline(
        controls=[control],
        lazy_init=True,
        device_map=device,
    )
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    out_ids = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)

    assert out_ids.ndim == 2


def test_angular_steering_auto_layer_selection(model_and_tokenizer, device: torch.device):
    """Verify that auto layer selection via explained_variances works."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    model_type = model.config.model_type
    hidden_size = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd")
    )
    num_layers = (
        getattr(model.config, "num_hidden_layers", None)
        or getattr(model.config, "n_layer")
    )

    sv = create_dummy_steering_vector(model_type, hidden_size, num_layers)

    # no layer_id — should auto-select via explained_variances
    control = AngularSteering(
        steering_vector=sv,
        target_degree=180.0,
        adaptive_mode=1,
    )

    pipeline = SteeringPipeline(
        controls=[control],
        lazy_init=True,
        device_map=device,
    )
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    hooked_layers = {lid for _, lid in control._hook_targets}
    assert all(lid in sv.directions for lid in hooked_layers), (
        f"Hooked layers {hooked_layers} not all in steering vector"
    )

    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    out_ids = pipeline.generate(input_ids=prompt_ids, max_new_tokens=8)

    assert out_ids.ndim == 2


def test_angular_steering_missing_layer_raises(model_and_tokenizer, device: torch.device):
    """Verify that requesting a layer not in the steering vector raises ValueError."""
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    model_type = model.config.model_type
    hidden_size = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd")
    )

    # vector only has layer 0
    sv = SteeringVector(
        model_type=model_type,
        directions={0: torch.randn(2, hidden_size)},
        explained_variances={0: 1.0},
    )

    control = AngularSteering(
        steering_vector=sv,
        layer_id=999,  # does not exist
        target_degree=180.0,
        adaptive_mode=1,
    )

    pipeline = SteeringPipeline(
        controls=[control],
        lazy_init=True,
        device_map=device,
    )
    pipeline.model = base_model.to(device)
    pipeline.tokenizer = tokenizer

    with pytest.raises(ValueError):
        pipeline.steer()
