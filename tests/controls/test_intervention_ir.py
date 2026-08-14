"""Component tests for the intervention IR and self-describing wire forms.

Pins the properties the two compilers rely on: the toolkit's wire-kind vocabulary matches the
plugin's tables, every exportable component's `apply` matches the plugin interpreter's math on
the same tensors, gate reset is idempotent (the shared-gate double-reset), IR dataclasses carry
no object-valued instance defaults, and binding never changes declared kinds.
"""
import dataclasses

import pytest
import torch

from aisteer360.algorithms.core.execution.contracts import InterventionKinds
from aisteer360.algorithms.core.execution.payloads import ModelFacts
from aisteer360.algorithms.core.internals.probes.probe import Probe
from aisteer360.algorithms.state_control._common.condition_scorers import ProbeContributionScorer, ProjectedCosineScorer
from aisteer360.algorithms.state_control._common.gates import (
    AlwaysOpenGate,
    CacheOnceGate,
    MultiKeyThresholdGate,
    ProbeSumGate,
)
from aisteer360.algorithms.state_control._common.selectors import FractionalDepthSelector
from aisteer360.algorithms.state_control._common.specs import (
    Condition,
    Intervention,
    TokenScope,
    WireForm,
    combine_kinds,
    lower_interventions,
)
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.transforms import (
    AdditiveTransform,
    AlignmentAdaptiveTransform,
    DirectionalAblationTransform,
    HeadAdditiveTransform,
    NormPreservingTransform,
    RotationTransform,
)
from aisteer360.algorithms.state_control._common.transforms.base import unwrap_modifiers

plugin_kinds = pytest.importorskip("vllm_hook_plugins.core.kinds")
from vllm_hook_plugins.core.interpreter import MODIFIERS, TRANSFORMS  # noqa: E402

H = 16


def _probe(layer_ids=(2,), hidden=H) -> Probe:
    generator = torch.Generator().manual_seed(7)
    weights = {
        lid: torch.randn(hidden, generator=generator, dtype=torch.float32) for lid in layer_ids
    }
    return Probe(
        model_type="test", location="layer_output", pooling="mean",
        layer_ids=list(layer_ids), weights=weights, bias=-0.25,
    )


def _layout(num_layers=8) -> ModelFacts:
    return ModelFacts(
        num_layers=num_layers, hidden_size=H, num_attention_heads=4, head_dim=H // 4,
        dtype="float32", model_fingerprint="test-fingerprint",
    )


class TestWireKindTables:
    """The toolkit's kind vocabulary is pinned to the plugin's tables."""

    def test_component_wire_kinds_match_plugin_tables(self):
        assert AdditiveTransform.wire_kind in plugin_kinds.TRANSFORM_KINDS
        assert DirectionalAblationTransform.wire_kind in plugin_kinds.TRANSFORM_KINDS
        assert RotationTransform.wire_kind in plugin_kinds.TRANSFORM_KINDS
        assert HeadAdditiveTransform.wire_kind in plugin_kinds.TRANSFORM_KINDS
        assert NormPreservingTransform.wire_kind in plugin_kinds.MODIFIER_KINDS
        assert AlignmentAdaptiveTransform.wire_kind in plugin_kinds.MODIFIER_KINDS
        assert AlwaysOpenGate.wire_kind in plugin_kinds.GATE_KINDS
        assert CacheOnceGate.wire_kind in plugin_kinds.GATE_KINDS
        assert MultiKeyThresholdGate.wire_kind in plugin_kinds.GATE_KINDS
        assert ProbeSumGate.wire_kind in plugin_kinds.GATE_KINDS

    def test_backend_seed_advertisement_matches_plugin_tables(self):
        from aisteer360.backends.vllm import _PLUGIN_INTERVENTION_KINDS as seed

        assert seed.transforms == plugin_kinds.TRANSFORM_KINDS
        assert seed.modifiers == plugin_kinds.MODIFIER_KINDS
        assert seed.scopes == plugin_kinds.SCOPE_KINDS
        assert seed.gates == plugin_kinds.GATE_KINDS
        assert dict(seed.constraints) == dict(plugin_kinds.CONSTRAINTS)

    def test_scope_kinds_are_total_on_the_wire(self):
        for kind, extra in (
            ("all", {}), ("after_prompt", {}), ("last_k", {"last_k": 3}),
            ("from_position", {"from_position": 5}),
        ):
            form = TokenScope(kind, **extra).export()
            assert form.kind in plugin_kinds.SCOPE_KINDS


class TestNumericalParity:
    """Toolkit `apply` and the plugin interpreter compute the same edit on the same tensors."""

    def setup_method(self):
        generator = torch.Generator().manual_seed(11)
        self.stream = torch.randn(5, H, generator=generator, dtype=torch.float32)
        self.hidden = self.stream.unsqueeze(0)  # [1, 5, H]
        self.mask = torch.ones(1, 5, dtype=torch.bool)
        self.vector = torch.randn(H, generator=generator, dtype=torch.float32)

    def test_additive(self):
        transform = AdditiveTransform({0: self.vector.unsqueeze(0)}, strength=3.5)
        ours = transform.apply(self.hidden, layer_id=0, token_mask=self.mask)[0]
        theirs = TRANSFORMS["additive"](self.stream, vector=self.vector, strength=3.5)
        torch.testing.assert_close(ours, theirs)

    def test_directional_ablation(self):
        transform = DirectionalAblationTransform({0: self.vector.unsqueeze(0)})
        ours = transform.apply(self.hidden, layer_id=0, token_mask=self.mask)[0]
        theirs = TRANSFORMS["directional_ablation"](self.stream, vector=self.vector)
        torch.testing.assert_close(ours, theirs)

    @pytest.mark.parametrize("mode", ["target", "offset"])
    def test_rotation(self, mode):
        generator = torch.Generator().manual_seed(13)
        basis = torch.randn(2, H, generator=generator, dtype=torch.float32)
        transform = RotationTransform({0: basis}, angle=0.7, mode=mode)
        ours = transform.apply(self.hidden, layer_id=0, token_mask=self.mask)[0]
        theirs = TRANSFORMS["rotation"](self.stream, basis=basis, angle=0.7, mode=mode)
        torch.testing.assert_close(ours, theirs)

    def test_head_additive(self):
        num_heads, head_dim = 4, H // 4
        generator = torch.Generator().manual_seed(17)
        directions = torch.randn(num_heads, head_dim, generator=generator, dtype=torch.float32)
        steering_vector = SteeringVector(
            model_type="test", directions={0: directions}, num_heads=num_heads, head_dim=head_dim,
        )
        transform = HeadAdditiveTransform(steering_vector, active_heads={0: {1, 3}}, strength=2.0)
        ours = transform.apply(self.hidden, layer_id=0, token_mask=self.mask)[0]

        dense = transform.export(0).tensors["vector"]
        heads = self.stream.view(5, num_heads, head_dim)
        theirs = TRANSFORMS["head_additive"](heads, vector=dense, strength=2.0).reshape(5, H)
        torch.testing.assert_close(ours, theirs)

    def test_norm_preserving_modifier(self):
        transform = NormPreservingTransform(AdditiveTransform({0: self.vector.unsqueeze(0)}, strength=8.0))
        ours = transform.apply(self.hidden, layer_id=0, token_mask=self.mask)[0]
        inner = lambda stream: TRANSFORMS["additive"](stream, vector=self.vector, strength=8.0)
        theirs = MODIFIERS["norm_preserving"](inner)(self.stream)
        torch.testing.assert_close(ours, theirs)

    @pytest.mark.parametrize("use_cosine", [False, True])
    def test_alignment_adaptive_modifier(self, use_cosine):
        transform = AlignmentAdaptiveTransform(
            AdditiveTransform({0: self.vector.unsqueeze(0)}, strength=2.0),
            {0: self.vector.unsqueeze(0)},
            threshold=0.0,
            use_cosine=use_cosine,
        )
        ours = transform.apply(self.hidden, layer_id=0, token_mask=self.mask)[0]
        inner = lambda stream: TRANSFORMS["additive"](stream, vector=self.vector, strength=2.0)
        theirs = MODIFIERS["alignment_adaptive"](
            inner, vector=self.vector, threshold=0.0, use_cosine=use_cosine,
        )(self.stream)
        torch.testing.assert_close(ours, theirs)


class TestGateResetIdempotence:
    """Shared-gate composition double-resets one instance; the second reset must be a no-op."""

    @pytest.mark.parametrize("make_gate", [
        lambda: AlwaysOpenGate(),
        lambda: MultiKeyThresholdGate(threshold=0.1, comparator="score_above", expected_keys={2}),
        lambda: ProbeSumGate(_probe()),
        lambda: CacheOnceGate(MultiKeyThresholdGate(threshold=0.1, comparator="score_above", expected_keys={2})),
    ])
    def test_double_reset_equals_single_reset(self, make_gate):
        single = make_gate()
        single.reset(3)
        double = make_gate()
        double.reset(3)
        double.reset(3)
        scores = torch.tensor([0.3, 0.05, 0.2])
        single.update(scores, key=2)
        double.update(scores, key=2)
        assert torch.equal(single.open_rows(), double.open_rows())
        assert single.is_ready() == double.is_ready()
        assert single.num_rows == double.num_rows == 3

    def test_reset_after_evidence_clears_decision(self):
        gate = CacheOnceGate(MultiKeyThresholdGate(threshold=0.1, comparator="score_above", expected_keys={2}))
        gate.reset(2)
        gate.update(torch.tensor([0.5, 0.0]), key=2)
        assert gate.is_ready()
        gate.reset(2)
        assert not gate.is_ready()
        assert not gate.open_rows().any()


class TestNoInstanceDefaults:
    """IR dataclasses never use instance defaults for object-valued fields."""

    @pytest.mark.parametrize("cls", [Intervention, TokenScope, Condition, WireForm])
    def test_object_defaults_use_default_factory(self, cls):
        for field_info in dataclasses.fields(cls):
            if field_info.default is dataclasses.MISSING:
                continue
            assert isinstance(field_info.default, (type(None), bool, int, float, str)), (
                f"{cls.__name__}.{field_info.name} has an object-valued instance default "
                f"{field_info.default!r}; use default_factory."
            )

    def test_default_gates_are_not_shared(self):
        transform = AdditiveTransform({0: torch.ones(1, H)})
        first = Intervention(layers=(0,), transform=transform)
        second = Intervention(layers=(0,), transform=transform)
        assert first.gate is not second.gate
        assert first.scope is not second.scope


class TestTokenScopeValidation:
    def test_last_k_requires_parameter(self):
        with pytest.raises(ValueError, match="last_k"):
            TokenScope("last_k")

    def test_from_position_requires_parameter(self):
        with pytest.raises(ValueError, match="from_position"):
            TokenScope("from_position")

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="scope kind"):
            TokenScope("prompt_only")

    def test_export_carries_parameters(self):
        assert TokenScope("last_k", last_k=4).export().params == {"k": 4}
        assert TokenScope("from_position", from_position=9).export().params == {"position": 9}


class TestInterventionWireKinds:
    def test_broadcast_additive_with_wrapper(self):
        transform = NormPreservingTransform(AdditiveTransform({3: torch.ones(1, H)}, strength=2.0))
        kinds = Intervention(layers=(3,), transform=transform).wire_kinds()
        assert kinds == InterventionKinds(
            transforms=frozenset({"additive"}),
            modifiers=frozenset({"norm_preserving"}),
            scopes=frozenset({"after_prompt"}),
            gates=frozenset(),
        )

    def test_positional_direction_is_hook_only(self):
        transform = AdditiveTransform({3: torch.ones(4, H)})
        assert Intervention(layers=(3,), transform=transform).wire_kinds() is None

    def test_layer_zero_input_edit_is_hook_only(self):
        transform = AdditiveTransform({0: torch.ones(1, H)})
        intervention = Intervention(layers=(0,), transform=transform, boundary="layer_input")
        assert intervention.wire_kinds() is None

    def test_norm_input_site_is_hook_only(self):
        transform = RotationTransform({3: torch.ones(2, H)})
        intervention = Intervention(layers=(3,), transform=transform, site="norm_input")
        assert intervention.wire_kinds() is None

    def test_head_additive_under_norm_preservation_is_hook_only(self):
        steering_vector = SteeringVector(
            model_type="test", directions={0: torch.ones(4, H // 4)}, num_heads=4, head_dim=H // 4,
        )
        transform = NormPreservingTransform(
            HeadAdditiveTransform(steering_vector, active_heads={0: {1}})
        )
        assert Intervention(layers=(0,), transform=transform).wire_kinds() is None

    def test_threshold_gate_is_hook_only(self):
        transform = AdditiveTransform({3: torch.ones(1, H)})
        gate = CacheOnceGate(MultiKeyThresholdGate(threshold=0.1, comparator="score_above", expected_keys={1}))
        condition = Condition(layer_ids=(1,), scorer=ProjectedCosineScorer({1: torch.ones(1, H)}))
        intervention = Intervention(layers=(3,), transform=transform, gate=gate, condition=condition)
        assert intervention.wire_kinds() is None

    def test_probe_gate_plans_cache_once(self):
        probe = _probe(layer_ids=(2,))
        transform = AdditiveTransform({3: torch.ones(1, H)})
        condition = Condition(layer_ids=(2,), scorer=ProbeContributionScorer(probe))
        intervention = Intervention(
            layers=(3,), transform=transform, gate=ProbeSumGate(probe), condition=condition,
        )
        kinds = intervention.wire_kinds()
        assert kinds is not None
        assert kinds.gates == frozenset({"cache_once", "probe_sum"})

    def test_combine_kinds_propagates_none(self):
        transform = AdditiveTransform({3: torch.ones(1, H)})
        exportable = Intervention(layers=(3,), transform=transform).wire_kinds()
        assert combine_kinds([exportable, None]) is None
        combined = combine_kinds([exportable, exportable])
        assert combined.transforms == frozenset({"additive"})


class TestBind:
    def test_selector_resolves_and_kinds_stay_stable(self):
        transform = AdditiveTransform({3: torch.ones(1, H)})
        intervention = Intervention(layers=FractionalDepthSelector(fraction=0.4), transform=transform)
        assert intervention.is_unbound
        bound = intervention.bind(None, None, layout=_layout(num_layers=8))
        assert bound.layers == (3,)
        assert not bound.is_unbound
        assert bound.wire_kinds() == intervention.wire_kinds()

    def test_coverage_is_validated(self):
        transform = AdditiveTransform({3: torch.ones(1, H)})
        intervention = Intervention(layers=(3, 4), transform=transform)
        with pytest.raises(ValueError, match="no direction for layer"):
            intervention.bind(None, None, layout=_layout())

    def test_coverage_opt_out(self):
        transform = AdditiveTransform({3: torch.ones(1, H)})
        intervention = Intervention(layers=(3, 4), transform=transform, require_coverage=False)
        bound = intervention.bind(None, None, layout=_layout())
        assert bound.layers == (3, 4)

    def test_out_of_range_layer_rejected(self):
        transform = AdditiveTransform({9: torch.ones(1, H)})
        intervention = Intervention(layers=(9,), transform=transform)
        with pytest.raises(ValueError, match="out of range"):
            intervention.bind(None, None, layout=_layout(num_layers=8))

    def test_scorer_boundary_mismatch_rejected(self):
        probe = _probe(layer_ids=(2,))
        transform = AdditiveTransform({3: torch.ones(1, H)})
        condition = Condition(layer_ids=(2,), scorer=ProbeContributionScorer(probe))
        intervention = Intervention(
            layers=(3,), transform=transform, gate=ProbeSumGate(probe), condition=condition,
            boundary="layer_input",
        )
        with pytest.raises(ValueError, match="expects features at"):
            intervention.bind(None, None, layout=_layout())

    def test_gate_source_fills_gate_and_condition(self):
        from aisteer360.algorithms.state_control._common.sources import ConditionPointSearch

        source = ConditionPointSearch(
            condition_vector=SteeringVector(model_type="test", directions={2: torch.ones(1, H)}),
            layer_ids=[2],
            threshold=0.05,
            comparator="larger",
        )
        transform = AdditiveTransform({3: torch.ones(1, H)})
        intervention = Intervention(layers=(3,), transform=transform, gate=source, boundary="layer_input")
        bound = intervention.bind(None, None, layout=_layout())
        assert isinstance(bound.gate, CacheOnceGate)
        assert isinstance(bound.gate.inner, MultiKeyThresholdGate)
        assert bound.condition is not None and bound.condition.layer_ids == (2,)
        assert source.resolved_point == {
            "layer_ids": [2], "threshold": 0.05, "comparator": "larger", "comparison_mode": "mean",
        }
        # the projected-cosine condition has no wire form, before or after binding
        assert intervention.wire_kinds() is None
        assert bound.wire_kinds() is None


class TestLowerInterventions:
    def test_additive_ops_one_per_layer_sharing_one_artifact(self):
        vector = torch.ones(1, H)
        transform = AdditiveTransform({2: vector, 3: vector}, strength=4.0)
        intervention = Intervention(layers=(2, 3), transform=transform, scope=TokenScope("after_prompt"))
        spec = lower_interventions([intervention], num_layers=8)
        assert spec is not None
        assert len(spec.ops) == 2
        assert [op["layers"] for op in spec.ops] == [[2], [3]]
        for op in spec.ops:
            assert op["transform"]["kind"] == "additive"
            assert op["transform"]["strength"] == 4.0
            assert op["scope"] == {"kind": "after_prompt"}
            assert op["gate"] is None
        assert len(spec.artifact_ids()) == 1
        assert len(spec.artifacts) == 1

    def test_modifiers_serialize_innermost_first(self):
        vector = torch.ones(1, H)
        transform = NormPreservingTransform(
            AlignmentAdaptiveTransform(
                AdditiveTransform({3: vector}, strength=2.0), {3: vector}, threshold=0.1,
            )
        )
        spec = lower_interventions(
            [Intervention(layers=(3,), transform=transform)], num_layers=8,
        )
        modifier_kinds = [m["kind"] for m in spec.ops[0]["transform"]["modifiers"]]
        assert modifier_kinds == ["alignment_adaptive", "norm_preserving"]

    def test_positional_additive_rejected(self):
        transform = AdditiveTransform({3: torch.ones(4, H)})
        spec = lower_interventions(
            [Intervention(layers=(3,), transform=transform)], num_layers=8,
        )
        assert spec is None

    def test_layer_input_boundary_maps_to_previous_wire_layer(self):
        transform = AdditiveTransform({3: torch.ones(1, H)})
        spec = lower_interventions(
            [Intervention(layers=(3,), transform=transform, boundary="layer_input", scope=TokenScope("all"))],
            num_layers=8,
        )
        assert spec.ops[0]["layers"] == [2]

    def test_layer_zero_input_edit_has_no_wire_form(self):
        transform = AdditiveTransform({0: torch.ones(1, H)})
        spec = lower_interventions(
            [Intervention(layers=(0,), transform=transform, boundary="layer_input")], num_layers=8,
        )
        assert spec is None

    def test_probe_gate_merges_condition_and_wraps_cache_once(self):
        probe = _probe(layer_ids=(2,))
        transform = AdditiveTransform({3: torch.ones(1, H)})
        condition = Condition(layer_ids=(2,), scorer=ProbeContributionScorer(probe))
        intervention = Intervention(
            layers=(3,), transform=transform, gate=ProbeSumGate(probe), condition=condition,
        )
        spec = lower_interventions([intervention], num_layers=8)
        gate = spec.ops[0]["gate"]
        assert gate["kind"] == "cache_once"
        inner = gate["inner"]
        assert inner["kind"] == "probe_sum"
        assert inner["condition_layers"] == [3]  # layer_output reads shift to l + 1
        assert inner["pooling"] == "mean"
        assert inner["artifact"] in spec.artifacts
        tensors = spec.artifacts[inner["artifact"]]
        assert set(tensors) == {"weights", "bias"}

    def test_explicit_cache_once_wrapper_is_preserved(self):
        probe = _probe(layer_ids=(2,))
        transform = AdditiveTransform({3: torch.ones(1, H)})
        condition = Condition(layer_ids=(2,), scorer=ProbeContributionScorer(probe))
        intervention = Intervention(
            layers=(3,), transform=transform, gate=CacheOnceGate(ProbeSumGate(probe)),
            condition=condition,
        )
        spec = lower_interventions([intervention], num_layers=8)
        gate = spec.ops[0]["gate"]
        assert gate["kind"] == "cache_once"
        assert gate["inner"]["kind"] == "probe_sum"

    def test_multi_intervention_op_order_follows_list_order(self):
        first = Intervention(
            layers=(3,), transform=AdditiveTransform({3: torch.ones(1, H)}, strength=1.0),
        )
        second = Intervention(
            layers=(2,), transform=DirectionalAblationTransform({2: torch.ones(1, H)}),
        )
        spec = lower_interventions([first, second], num_layers=8)
        assert [op["transform"]["kind"] for op in spec.ops] == ["additive", "directional_ablation"]

    def test_gate_and_scorer_both_exporting_tensors_is_an_error(self):
        probe = _probe(layer_ids=(2,))

        class _TensorExportingScorer(ProbeContributionScorer):
            def export(self):
                from aisteer360.algorithms.state_control._common.specs import WireForm

                return WireForm(kind="probe_sum", params={"pooling": "mean"},
                                tensors={"weights": torch.ones(1, H)})

        condition = Condition(layer_ids=(2,), scorer=_TensorExportingScorer(probe))
        intervention = Intervention(
            layers=(3,), transform=AdditiveTransform({3: torch.ones(1, H)}),
            gate=ProbeSumGate(probe), condition=condition,
        )
        with pytest.raises(ValueError, match="exactly one may own"):
            lower_interventions([intervention], num_layers=8)

    def test_condition_layers_follow_probe_order_on_the_wire(self):
        """The exported weight rows align with the probe's layer order, so the wire
        condition_layers follow the probe regardless of the condition's declaration order."""
        probe = _probe(layer_ids=(4, 2))
        condition = Condition(layer_ids=(2, 4), scorer=ProbeContributionScorer(probe))
        intervention = Intervention(
            layers=(3,), transform=AdditiveTransform({3: torch.ones(1, H)}),
            gate=ProbeSumGate(probe), condition=condition,
        )
        spec = lower_interventions([intervention], num_layers=8)
        inner = spec.ops[0]["gate"]["inner"]
        assert inner["condition_layers"] == [5, 3]  # probe order, mapped to wire indices

    def test_follower_probe_gate_lowers_without_a_condition(self):
        """The follower half of a shared-gate composition (probe gate, no condition) lowers;
        the probe supplies the evidence layers itself."""
        probe = _probe(layer_ids=(2,))
        intervention = Intervention(
            layers=(3,), transform=AdditiveTransform({3: torch.ones(1, H)}),
            gate=ProbeSumGate(probe),
        )
        assert intervention.wire_kinds() is not None
        spec = lower_interventions([intervention], num_layers=8)
        inner = spec.ops[0]["gate"]["inner"]
        assert inner["kind"] == "probe_sum"
        assert inner["condition_layers"] == [3]

    def test_round_trip_through_plugin_parser(self):
        from vllm_hook_plugins.core.schema import parse_intervention_spec

        probe = _probe(layer_ids=(2,))
        condition = Condition(layer_ids=(2,), scorer=ProbeContributionScorer(probe))
        intervention = Intervention(
            layers=(3, 4),
            transform=NormPreservingTransform(AdditiveTransform({3: torch.ones(1, H), 4: torch.ones(1, H)})),
            gate=ProbeSumGate(probe),
            condition=condition,
            scope=TokenScope("last_k", last_k=2),
        )
        spec = lower_interventions([intervention], num_layers=8)
        parsed = parse_intervention_spec(spec.to_wire(), num_layers=8)
        assert len(parsed.ops) == 2
        assert parsed.condition_layers() == frozenset({3})


class TestReviewRegressions:
    """Regression pins from the adversarial review of the seam landing."""

    def test_two_interventions_at_the_same_lowest_layer_elect_one_opener(self):
        from aisteer360.algorithms.state_control._common.model_layout import ModelLayout as ModulePaths
        from aisteer360.algorithms.state_control._common.runtime import build_hooks

        layout = ModulePaths(
            family="llama_style", layer_prefix="model.layers", num_layers=8,
            attn_suffix=".self_attn", oproj_suffix=".self_attn.o_proj",
            norm_attrs=("input_layernorm", "post_attention_layernorm"),
        )
        first = Intervention(layers=(3,), transform=AdditiveTransform({3: torch.ones(1, H)}))
        second = Intervention(layers=(3,), transform=AdditiveTransform({3: torch.ones(1, H)}))
        hooks = build_hooks([first, second], layout, torch.tensor([4]))
        assert len(hooks["forward"]) == 2  # both interventions hook; exactly one opener elected

    def test_layer_zero_head_additive_keeps_its_wire_form(self):
        """The o_proj site keeps its layer index on the wire, so layer 0 stays expressible."""
        head_dim = H // 4
        steering_vector = SteeringVector(
            model_type="test", directions={0: torch.ones(4, head_dim)},
            num_heads=4, head_dim=head_dim,
        )
        intervention = Intervention(
            layers=(0,),
            transform=HeadAdditiveTransform(steering_vector, active_heads={0: {1}}),
            boundary="layer_input",
        )
        kinds = intervention.wire_kinds()
        assert kinds is not None and "head_additive" in kinds.transforms
        spec = lower_interventions([intervention], num_layers=8)
        assert spec is not None
        assert spec.ops[0]["layers"] == [0]
