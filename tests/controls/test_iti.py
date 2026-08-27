"""ITI head-geometry handling.

Pins that `head_geometry` reads the per-layer attention geometry off the module tree (matching the
config on uniform-head models, including a LoRA wrapper), and that the ITI estimator fails loudly on
a model with heterogeneous head geometry before running any forward pass.
"""
import pytest

from steerability.algorithms.core.internals.data import LabeledExamples
from steerability.algorithms.core.internals.model_layout import head_geometry, resolve_model_layout
from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec
from steerability.algorithms.state_control.iti.utils.estimator import ProbeMassShiftEstimator
from tests.utils.tiny_models import heterogeneous_head_stub, tiny_gpt2, tiny_llama, tiny_lora

LAYERS = 3
HIDDEN = 32
HEADS = 4


@pytest.mark.parametrize("factory", [tiny_llama, tiny_gpt2, tiny_lora])
def test_head_geometry_matches_config_on_uniform_models(factory):
    model = factory() if factory is tiny_lora else factory(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
    layout = resolve_model_layout(model)
    for layer_id in range(layout.num_layers):
        geometry = head_geometry(model, layout, layer_id)
        assert geometry.num_heads * geometry.head_dim == HIDDEN


def test_iti_raises_on_heterogeneous_geometry_before_forward():
    """The estimator raises the heterogeneous-geometry error naming layers, before any forward."""
    stub = heterogeneous_head_stub(num_layers=4, hidden=HIDDEN)
    data = LabeledExamples(positives=["a", "b"], negatives=["c", "d"])
    spec = VectorTrainSpec(method="mean_diff", accumulate="last_token")
    with pytest.raises(ValueError, match="uniform attention head geometry") as excinfo:
        ProbeMassShiftEstimator().fit(stub, tokenizer=None, data=data, spec=spec)
    message = str(excinfo.value)
    assert "num_heads" in message and "head_dim" in message
