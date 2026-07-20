"""ITI o_proj folding equivalence (doc 03 §5, §7).

The folded residual `add` (per-layer `W_o · Σ_h pad(d_h)`) must equal the head-space addition pushed
through `o_proj`, so the exported wire form reproduces the in-process intervention exactly.
"""
import torch

from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.transforms.head_additive import HeadAdditiveTransform


def test_folded_residual_equals_head_addition_through_oproj():
    torch.manual_seed(0)
    num_heads, head_dim = 4, 8
    hidden_size = num_heads * head_dim
    strength = 3.0

    # per-head directions for one layer; active heads {0, 2}
    directions = {5: torch.randn(num_heads, head_dim, dtype=torch.float32)}
    sv = SteeringVector(model_type="test", directions=directions, num_heads=num_heads, head_dim=head_dim)
    active_heads = {5: {0, 2}}
    transform = HeadAdditiveTransform(sv, active_heads=active_heads, strength=strength)

    # a linear o_proj: [H_out, num_heads*head_dim]
    o_proj_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)

    # head-space delta x (pre-o_proj): active head slices = strength * direction, else 0
    x = torch.zeros(hidden_size, dtype=torch.float32)
    for head_id in active_heads[5]:
        x[head_id * head_dim:(head_id + 1) * head_dim] = strength * directions[5][head_id]
    expected_residual = o_proj_weight @ x  # W_o @ x

    transform.fold_to_residual({5: o_proj_weight})
    payload = transform.export_payload()
    assert payload["kind"] == "add"
    assert payload["scale"] == 1.0
    folded = payload["vectors"][5].tensor
    assert torch.allclose(folded, expected_residual, atol=1e-5)


def test_export_none_before_fold():
    directions = {0: torch.randn(4, 8)}
    sv = SteeringVector(model_type="test", directions=directions, num_heads=4, head_dim=8)
    transform = HeadAdditiveTransform(sv, active_heads={0: {1}}, strength=1.0)
    assert transform.export_payload() is None  # not portable until folded
