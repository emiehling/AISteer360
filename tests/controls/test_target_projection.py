"""Tests for `TargetProjectionTransform` (doc 03 §5, §7)."""
import pytest
import torch

from aisteer360.algorithms.state_control._common.transforms import TargetProjectionTransform


def _unit(vec):
    return vec / vec.norm()


def test_sets_projection_to_target_closed_form():
    torch.manual_seed(0)
    direction = torch.randn(8)
    transform = TargetProjectionTransform({0: direction}, targets={0: 1.5})
    hidden = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)

    out = transform.apply(hidden, layer_id=0, token_mask=mask)

    # closed form: h' = h + (t - h·v̂) v̂  =>  h'·v̂ == t at every masked position
    v_hat = _unit(direction)
    projections = torch.einsum("bth,h->bt", out, v_hat)
    assert torch.allclose(projections, torch.full((2, 3), 1.5), atol=1e-5)

    # orthogonal complement unchanged
    expected = hidden + (1.5 - torch.einsum("bth,h->bt", hidden, v_hat)).unsqueeze(-1) * v_hat
    assert torch.allclose(out, expected, atol=1e-5)


def test_default_target_zero_ablates_component():
    direction = torch.randn(8)
    transform = TargetProjectionTransform({0: direction})  # no targets -> 0.0
    hidden = torch.randn(1, 4, 8)
    out = transform.apply(hidden, layer_id=0, token_mask=torch.ones(1, 4, dtype=torch.bool))
    v_hat = _unit(direction)
    projections = torch.einsum("bth,h->bt", out, v_hat)
    assert torch.allclose(projections, torch.zeros(1, 4), atol=1e-5)


def test_respects_token_mask():
    direction = torch.randn(8)
    transform = TargetProjectionTransform({0: direction}, targets={0: 2.0})
    hidden = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, False, True]])
    out = transform.apply(hidden, layer_id=0, token_mask=mask)
    # unmasked position 1 unchanged
    assert torch.allclose(out[:, 1], hidden[:, 1])
    # masked positions changed
    assert not torch.allclose(out[:, 0], hidden[:, 0])


def test_absent_layer_is_noop():
    transform = TargetProjectionTransform({0: torch.randn(8)}, targets={0: 1.0})
    hidden = torch.randn(1, 2, 8)
    out = transform.apply(hidden, layer_id=5, token_mask=torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(out, hidden)


def test_dtype_and_device_preserved():
    direction = torch.randn(8, dtype=torch.float32)
    transform = TargetProjectionTransform({0: direction}, targets={0: 1.0})
    hidden = torch.randn(1, 2, 8, dtype=torch.float32)
    out = transform.apply(hidden, layer_id=0, token_mask=torch.ones(1, 2, dtype=torch.bool))
    assert out.dtype == hidden.dtype
    assert out.device == hidden.device


def test_export_payload_target_projection_kind():
    transform = TargetProjectionTransform({0: torch.randn(8), 1: torch.randn(8)}, targets={0: 1.0})
    payload = transform.export_payload()
    assert payload["kind"] == "target_projection"
    assert set(payload["vectors"].keys()) == {0, 1}
    assert payload["targets"][0] == 1.0
    assert payload["targets"][1] == 0.0  # default


def test_unbound_source_raises_on_apply():
    from aisteer360.algorithms.state_control._common.sources import ContrastiveFit

    src = ContrastiveFit(data={"positives": ["a"], "negatives": ["b"]})
    transform = TargetProjectionTransform(src)
    assert transform.is_bound is False
    with pytest.raises(RuntimeError, match="unbound"):
        transform.apply(torch.randn(1, 2, 8), layer_id=0, token_mask=torch.ones(1, 2, dtype=torch.bool))
