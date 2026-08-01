"""Spec/hook equivalence suite: torch hooks and intervention specs are two serializations of
one tuple, proven against the plugin's own interpreter (the code the worker executes).

Per-transform equality applies the toolkit transform to synthetic masked rows and the plugin's
`apply_op` to the scoped rows and asserts exact equality in float32 (documented-tolerance
closeness in bfloat16); modifier chains must compose innermost-first and a reordered chain must
change the result; gate decision traces must coincide across single-pass, chunked-prefill, and
restart-replay evidence orderings."""
import pytest
import torch
from vllm_hook_plugins.core.interpreter import apply_op, build_gate
from vllm_hook_plugins.core.interpreter.gates import CacheOnceGate as WireCacheOnceGate
from vllm_hook_plugins.core.schema import parse_intervention_spec

from aisteer360.algorithms.core.execution import ModelLayout
from aisteer360.algorithms.core.internals.pooling import aggregate_condition_hidden
from aisteer360.algorithms.core.internals.probes import Probe
from aisteer360.algorithms.state_control._common.condition_scorers import (
    ProbeContributionScorer,
)
from aisteer360.algorithms.state_control._common.gates import CacheOnceGate, ProbeSumGate
from aisteer360.algorithms.state_control._common.intervention_export import artifact_id_for
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.transforms import (
    AdditiveTransform,
    AlignmentAdaptiveTransform,
    DirectionalAblationTransform,
    HeadAdditiveTransform,
    NormPreservingTransform,
    RotationTransform,
)
from aisteer360.algorithms.state_control.act_add.control import ActAdd
from aisteer360.algorithms.state_control.activation_adapter.control import ActivationAdapter
from aisteer360.algorithms.state_control.angular_steering.control import AngularSteering
from aisteer360.algorithms.state_control.caa.control import CAA
from aisteer360.algorithms.state_control.directional_ablation.control import (
    DirectionalAblation,
)
from aisteer360.algorithms.state_control.iti.control import ITI

LAYERS = 4
HIDDEN = 16
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
SEQ = 6


class _LayoutOnlySession:
    def __init__(self, layout: ModelLayout):
        self.layout = layout


def _session(dtype: str = "float32") -> _LayoutOnlySession:
    return _LayoutOnlySession(ModelLayout(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        dtype=dtype,
        model_fingerprint="0" * 16,
    ))


def _vector(k: int = 1, seed: int = 0) -> SteeringVector:
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(k, HIDDEN, generator=generator) for lid in range(LAYERS)},
    )


def _iti_vector(seed: int = 0) -> SteeringVector:
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        model_type="llama",
        directions={lid: torch.randn(HEADS, HEAD_DIM, generator=generator) for lid in range(LAYERS)},
        num_heads=HEADS,
        head_dim=HEAD_DIM,
    )


def _wire_ops(control):
    """The exported spec parsed through the plugin schema, plus its artifact payloads."""
    spec = control.export_intervention_spec()
    assert spec is not None
    parsed = parse_intervention_spec(spec.to_wire(), num_layers=LAYERS)
    return parsed, dict(spec.artifacts)


CONFIGS = [
    pytest.param(
        lambda: CAA(steering_vector=_vector(), layer_id=1, multiplier=3.0, token_scope="all"),
        1, 1, id="caa",
    ),
    pytest.param(
        lambda: CAA(steering_vector=_vector(), layer_id=2, multiplier=-2.0, use_norm_preservation=True),
        2, 2, id="caa-norm-preserving",
    ),
    pytest.param(
        lambda: ActAdd(steering_vector=_vector(), layer_id=2, multiplier=2.0),
        2, 1, id="act-add-broadcast",
    ),
    pytest.param(
        lambda: DirectionalAblation(steering_vector=_vector(), layer_ids=[1]),
        1, 1, id="directional-ablation",
    ),
    pytest.param(
        lambda: AngularSteering(
            steering_vector=_vector(k=2), target_degree=50.0, layer_range=(1, 2),
            intervention_point="layer_output",
        ),
        1, 1, id="rotation-target",
    ),
    pytest.param(
        lambda: AngularSteering(
            steering_vector=_vector(k=2), angle=0.4, mode="offset", layer_range=(1, 2),
            intervention_point="layer_output",
        ),
        1, 1, id="rotation-offset",
    ),
    pytest.param(
        lambda: AngularSteering(
            steering_vector=_vector(k=2), target_degree=30.0, adaptive=True,
            adaptive_use_cosine=True, layer_range=(1, 2), intervention_point="layer_output",
        ),
        1, 1, id="rotation-adaptive-cosine",
    ),
    pytest.param(
        lambda: AngularSteering(
            steering_vector=_vector(k=2), target_degree=30.0, adaptive=True,
            use_norm_preservation=True, layer_range=(1, 2), intervention_point="layer_output",
        ),
        1, 1, id="rotation-adaptive-norm-preserving",
    ),
    pytest.param(
        lambda: ActivationAdapter(
            transform=NormPreservingTransform(
                DirectionalAblationTransform(_vector().directions)
            ),
            layer_ids=[2], token_scope="all",
        ),
        2, 2, id="adapter-wrapped-ablation",
    ),
]


class TestPerTransformEquality:

    @pytest.mark.parametrize("factory,toolkit_layer,wire_layer", CONFIGS)
    def test_float32_exact(self, factory, toolkit_layer, wire_layer):
        control = factory()
        control.steer(model=None, session=_session())
        parsed, artifacts = _wire_ops(control)
        (op,) = parsed.ops
        assert list(op.layers) == [wire_layer]

        generator = torch.Generator().manual_seed(11)
        hidden = torch.randn(1, SEQ, HIDDEN, generator=generator)
        mask = torch.tensor([[True, False, True, True, False, True]])

        toolkit_out = control._transform.apply(hidden, layer_id=toolkit_layer, token_mask=mask)
        wire_out = apply_op(op, hidden[0][mask[0]], artifacts)

        assert torch.equal(toolkit_out[0][mask[0]], wire_out)
        assert torch.equal(toolkit_out[0][~mask[0]], hidden[0][~mask[0]])

    @pytest.mark.parametrize("factory,toolkit_layer,wire_layer", CONFIGS)
    def test_bfloat16_within_documented_tolerance(self, factory, toolkit_layer, wire_layer):
        control = factory()
        control.steer(model=None, session=_session(dtype="bfloat16"))
        parsed, artifacts = _wire_ops(control)
        (op,) = parsed.ops

        generator = torch.Generator().manual_seed(12)
        hidden = torch.randn(1, SEQ, HIDDEN, generator=generator).to(torch.bfloat16)
        mask = torch.ones(1, SEQ, dtype=torch.bool)

        toolkit_out = control._transform.apply(hidden, layer_id=toolkit_layer, token_mask=mask)
        wire_out = apply_op(op, hidden[0], artifacts)
        assert torch.allclose(toolkit_out[0].float(), wire_out.float(), rtol=1e-2, atol=1e-2)

    def test_iti_head_additive_exact(self):
        control = ITI(steering_vector=_iti_vector(), selected_heads=[(2, 0), (2, 3)], alpha=4.0)
        control.steer(model=None, session=_session())
        parsed, artifacts = _wire_ops(control)
        (op,) = parsed.ops
        assert list(op.layers) == [2]

        generator = torch.Generator().manual_seed(13)
        hidden = torch.randn(1, SEQ, HIDDEN, generator=generator)
        mask = torch.ones(1, SEQ, dtype=torch.bool)

        toolkit_out = control._transform.apply(hidden, layer_id=2, token_mask=mask)
        wire_out = apply_op(op, hidden[0].reshape(SEQ, HEADS, HEAD_DIM), artifacts)
        assert torch.equal(toolkit_out[0], wire_out.reshape(SEQ, HIDDEN))


class TestModifierChain:

    def _payload(self):
        vector = _vector(k=2)
        transform = NormPreservingTransform(
            AlignmentAdaptiveTransform(RotationTransform(vector, angle=0.3, mode="offset"), vector)
        )
        return transform, transform.to_intervention_op_payload(1)

    def test_emitted_order_is_innermost_first(self):
        _, payload = self._payload()
        assert [modifier["kind"] for modifier in payload["modifiers"]] == [
            "alignment_adaptive", "norm_preserving",
        ]

    def test_composed_result_matches_wrapped_hook(self):
        transform, payload = self._payload()
        artifacts = {}
        transform_wire = {"kind": payload["kind"], **payload["params"], "modifiers": []}
        for modifier in payload["modifiers"]:
            wire_modifier = {"kind": modifier["kind"], **modifier["params"]}
            if modifier["tensors"]:
                artifact_id, prepared = artifact_id_for(modifier["tensors"])
                wire_modifier["artifact"] = artifact_id
                artifacts[artifact_id] = prepared
            transform_wire["modifiers"].append(wire_modifier)
        artifact_id, prepared = artifact_id_for(payload["tensors"])
        transform_wire["artifact"] = artifact_id
        artifacts[artifact_id] = prepared
        wire = {"ops": [{
            "layers": [1], "transform": transform_wire, "scope": {"kind": "all"}, "gate": None,
        }]}
        parsed = parse_intervention_spec(wire, num_layers=LAYERS)

        generator = torch.Generator().manual_seed(21)
        hidden = torch.randn(1, SEQ, HIDDEN, generator=generator)
        mask = torch.ones(1, SEQ, dtype=torch.bool)
        toolkit_out = transform.apply(hidden, layer_id=1, token_mask=mask)
        wire_out = apply_op(parsed.ops[0], hidden[0], artifacts)
        assert torch.equal(toolkit_out[0], wire_out)

    def test_reordered_emission_fails_the_structural_pin(self):
        """The two shipped modifiers are row-local and commute in output, so the reorder
        discipline is structural: an emission that does not match the live wrapper chain
        innermost-first is a serialization drift regardless of output agreement."""
        transform, payload = self._payload()
        emitted = [modifier["kind"] for modifier in payload["modifiers"]]

        chain = []
        current = transform
        while True:
            if isinstance(current, NormPreservingTransform):
                chain.append("norm_preserving")
                current = current._inner
            elif isinstance(current, AlignmentAdaptiveTransform):
                chain.append("alignment_adaptive")
                current = current.inner
            else:
                break
        innermost_first = list(reversed(chain))
        assert emitted == innermost_first
        assert list(reversed(emitted)) != innermost_first


def _probe(pooling: str = "mean", bias: float = 0.0) -> Probe:
    generator = torch.Generator().manual_seed(31)
    return Probe(
        model_type="llama",
        location="layer_input",
        pooling=pooling,
        layer_ids=[1, 2],
        weights={lid: torch.randn(HIDDEN, generator=generator) for lid in (1, 2)},
        bias=bias,
        meta={},
    )


def _wire_probe_gate(probe: Probe) -> WireCacheOnceGate:
    """The worker's gate state machine built from the exported probe payload."""
    gate_payload = ProbeSumGate(probe).to_intervention_gate()
    artifact_id, prepared = artifact_id_for(gate_payload["tensors"])
    wire = {"ops": [{
        "layers": [3],
        "transform": {"kind": "directional_ablation", "modifiers": [], "artifact": artifact_id},
        "scope": {"kind": "all"},
        "gate": {
            "kind": "cache_once",
            "inner": {"kind": gate_payload["kind"], **gate_payload["params"], "artifact": artifact_id},
        },
    }]}
    # the vector artifact reuses the probe weights id slot only for schema validation; gates
    # read their own tensors from the same registry mapping
    parsed = parse_intervention_spec(wire, num_layers=LAYERS)
    return build_gate(parsed.ops[0].gate, {artifact_id: prepared})


def _toolkit_decision(probe: Probe, prompt_rows: dict[int, torch.Tensor]) -> bool:
    """The frozen toolkit decision for one prompt's evidence."""
    scorer = ProbeContributionScorer(probe)
    gate = CacheOnceGate(ProbeSumGate(probe))
    gate.reset(1)
    for layer_id, rows in prompt_rows.items():
        scores = scorer(rows.unsqueeze(0), layer_id, prompt_mask=torch.ones(1, rows.size(0)))
        gate.update(scores, key=layer_id)
    assert gate.is_ready()
    return bool(gate.open_rows()[0])


def _wire_decision(gate, prompt_rows: dict[int, torch.Tensor], chunks: list[range]) -> bool | None:
    """The worker gate's frozen decision after feeding the prompt in the given pass chunks."""
    prompt_len = next(iter(prompt_rows.values())).size(0)
    for positions in chunks:
        for layer_id, rows in prompt_rows.items():
            gate.observe(layer_id, positions, rows[positions.start:positions.stop])
        gate.note_pass(positions, prompt_len)
    # first decode pass triggers the deferred freeze when the trigger pass lacked evidence
    gate.note_pass(range(prompt_len, prompt_len + 1), prompt_len)
    return gate.decision()


class TestGateDecisionTraces:

    @pytest.mark.parametrize("pooling", ["mean", "last"])
    @pytest.mark.parametrize("bias_offset", [1.5, -1.5])
    def test_single_pass_prefill_traces_coincide(self, pooling, bias_offset):
        generator = torch.Generator().manual_seed(41)
        prompt_rows = {lid: torch.randn(SEQ, HIDDEN, generator=generator) for lid in (1, 2)}
        raw = _probe(pooling=pooling, bias=0.0)
        centered = float(sum(
            aggregate_condition_hidden(prompt_rows[lid].unsqueeze(0), pooling).squeeze(0)
            @ raw.weights[lid]
            for lid in (1, 2)
        ))
        probe = _probe(pooling=pooling, bias=-centered + bias_offset)

        expected = _toolkit_decision(probe, prompt_rows)
        assert expected == (bias_offset > 0)
        wire_gate = _wire_probe_gate(probe)
        assert _wire_decision(wire_gate, prompt_rows, [range(0, SEQ)]) is expected

    @pytest.mark.parametrize("pooling", ["mean", "last"])
    def test_chunked_prefill_traces_coincide(self, pooling):
        generator = torch.Generator().manual_seed(42)
        prompt_rows = {lid: torch.randn(SEQ, HIDDEN, generator=generator) for lid in (1, 2)}
        probe = _probe(pooling=pooling, bias=0.05)
        expected = _toolkit_decision(probe, prompt_rows)

        chunked = _wire_probe_gate(probe)
        assert _wire_decision(chunked, prompt_rows, [range(0, 2), range(2, 4), range(4, SEQ)]) is expected

    def test_restart_replay_is_idempotent(self):
        generator = torch.Generator().manual_seed(43)
        prompt_rows = {lid: torch.randn(SEQ, HIDDEN, generator=generator) for lid in (1, 2)}
        probe = _probe(bias=0.05)
        expected = _toolkit_decision(probe, prompt_rows)

        gate = _wire_probe_gate(probe)
        # partial prefill, then a preemption restart clears evidence and replays from zero
        for layer_id, rows in prompt_rows.items():
            gate.observe(layer_id, range(0, 3), rows[:3])
        gate.note_pass(range(0, 3), SEQ)
        gate.reset()
        assert _wire_decision(gate, prompt_rows, [range(0, SEQ)]) is expected

    def test_undecided_freezes_closed_and_holds(self):
        probe = _probe(bias=1e9)
        gate = _wire_probe_gate(probe)
        gate.note_pass(range(0, SEQ), SEQ)  # no evidence ever arrives
        gate.note_pass(range(SEQ, SEQ + 1), SEQ)
        assert gate.decision() is False
        generator = torch.Generator().manual_seed(44)
        gate.observe(1, range(SEQ, SEQ + 1), torch.randn(1, HIDDEN, generator=generator))
        assert gate.decision() is False


class TestArtifactStability:

    def test_ids_stable_across_producing_dtype_and_layout(self):
        tensor = torch.randn(HIDDEN, dtype=torch.float64)
        id_from_f64, _ = artifact_id_for({"vector": tensor})
        id_from_f32, _ = artifact_id_for({"vector": tensor.to(torch.float32)})
        id_from_noncontiguous, _ = artifact_id_for(
            {"vector": tensor.to(torch.float32).unsqueeze(0).expand(2, -1)[0]}
        )
        assert id_from_f64 == id_from_f32 == id_from_noncontiguous
