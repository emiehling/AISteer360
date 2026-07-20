"""Plan purity and export tests for migrated declarative controls (doc 03 §7).

`plan()` must be pure (same inputs → equal plans across calls) and, where the control claims wire
portability, its transform/scorer components must export via `export_payload`.
"""
import torch

from aisteer360.algorithms.state_control._common.intervention import PromptContext
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.caa.control import CAA
from aisteer360.algorithms.state_control.directional_ablation.control import DirectionalAblation
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


def _ctx():
    ids = torch.tensor([[3, 4, 5, 6]])
    return PromptContext.from_ids(ids, attention_mask=torch.ones_like(ids), pad_token_id=2)


def _steered_caa():
    model = tiny_llama(num_layers=4, hidden=32, heads=4, vocab=100)
    sv = SteeringVector(model_type="llama", directions={1: torch.randn(1, 32)})
    caa = CAA(steering_vector=sv, layer_id=1, multiplier=2.0)
    caa.steer(model, wordlevel_tokenizer())
    return caa


def _steered_ablation():
    model = tiny_llama(num_layers=4, hidden=32, heads=4, vocab=100)
    sv = SteeringVector(model_type="llama", directions={1: torch.randn(1, 32), 2: torch.randn(1, 32)})
    control = DirectionalAblation(steering_vector=sv, layer_ids=[1, 2], alpha=0.5)
    control.steer(model, wordlevel_tokenizer())
    return control


def test_caa_plan_is_pure():
    caa = _steered_caa()
    ctx = _ctx()
    plan_a = caa.plan(ctx, None)
    plan_b = caa.plan(ctx, None)
    assert len(plan_a) == len(plan_b) == 1
    a, b = plan_a[0], plan_b[0]
    assert a.hook_point == b.hook_point == "layer_output"
    assert a.layer_ids == b.layer_ids == [1]
    assert a.scope == b.scope
    # same bound transform instance (pure: no re-fit)
    assert a.transform is b.transform


def test_caa_transform_exports_add():
    caa = _steered_caa()
    payload = caa.plan(_ctx(), None)[0].transform.export_payload()
    assert payload["kind"] == "add"
    assert payload["scale"] == 2.0
    assert set(payload["vectors"].keys()) == {1}


def test_ablation_plan_exports_ablate_with_alpha():
    control = _steered_ablation()
    intervention = control.plan(_ctx(), None)[0]
    assert intervention.hook_point == "layer_output"
    assert sorted(intervention.layer_ids) == [1, 2]
    payload = intervention.transform.export_payload()
    assert payload["kind"] == "ablate"
    assert payload["alpha"] == 0.5
    assert set(payload["vectors"].keys()) == {1, 2}
