"""Phase-derived requirements for intervention controls.

Pins the three phase decisions of the derived `requirements()`: steer requires model-side work
exactly when the template carries unbound sources, generate offers the intervention-spec
alternative exactly when every component has a wire form, and score is in-process (remote
prompt-logprob scoring anchors token scopes at the request's prompt end). Also pins the eager
steer-time lowering failure naming the intervention and reason.
"""
import pytest
import torch

from aisteer360.algorithms.core.execution import BackendSpec, Capability
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.caa.control import CAA
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 16
LAYERS = 4

pytest.importorskip("vllm_hook_plugins")

SERVE_SPEC = BackendSpec(kind="vllm-serve", model="tiny", options={
    "base_url": "http://localhost:9", "hook_plugin": True,
})


def _vector(k: int = 1) -> SteeringVector:
    generator = torch.Generator().manual_seed(3)
    return SteeringVector(
        model_type="llama",
        directions={1: torch.randn(k, HIDDEN, generator=generator)},
    )


def _fit_caa() -> CAA:
    return CAA(data={"prompts": ["q"], "positives": ["a"], "negatives": ["b"]}, layer_id=1)


class TestPhaseVerdicts:

    def test_steer_phase_rejects_fitting_on_a_remote_pair(self):
        """A template carrying a fit source cannot steer against a capture-less remote pair."""
        pipeline = SteeringPipeline(controls=[_fit_caa()], lazy_init=True)
        report = pipeline.check(steer_backend=SERVE_SPEC, inference_backend=SERVE_SPEC)
        failures = report.failures_for("steer")
        assert len(failures) == 1
        assert failures[0].control == "CAA"
        assert "steering_vector" in failures[0].message

    def test_precomputed_template_steers_against_a_remote_pair(self):
        """A fully concrete configuration requires nothing at steer."""
        pipeline = SteeringPipeline(
            controls=[CAA(steering_vector=_vector(), layer_id=1)], lazy_init=True,
        )
        report = pipeline.check(steer_backend=SERVE_SPEC, inference_backend=SERVE_SPEC)
        assert report.supported("steer")
        assert report.supported("generate")

    def test_score_phase_rejects_spec_backend_by_name(self):
        """Scoring an intervention control on a spec backend fails at check, naming the control."""
        pipeline = SteeringPipeline(
            controls=[CAA(steering_vector=_vector(), layer_id=1)], lazy_init=True,
        )
        report = pipeline.check(steer_backend=SERVE_SPEC, inference_backend=SERVE_SPEC)
        failures = report.failures_for("score")
        assert len(failures) == 1
        assert failures[0].control == "CAA"
        assert "prompt" in failures[0].message

    def test_generate_offers_spec_alternative_only_with_a_wire_form(self):
        exportable = CAA(steering_vector=_vector(), layer_id=1)
        positional = CAA(steering_vector=_vector(k=3), layer_id=1)

        def offers_specs(control) -> bool:
            return any(
                Capability.INTERVENTION_SPECS in alternative.atoms
                for alternative in control.requirements().generate
            )

        assert offers_specs(exportable)
        assert not offers_specs(positional)


class TestEagerLoweringFailure:

    def test_lowering_failure_names_the_intervention_and_reason(self):
        """A configuration whose inexpressibility is artifact-dependent passes check() and
        fails at the eager steer-time lowering with the intervention named."""
        from aisteer360.algorithms.core.execution import UnsupportedOperationError

        class _LyingSource:
            """Declares a broadcast fit but resolves a positional vector."""

            steer_needs = "none"
            produces_positional = False

            def resolve(self, model, tokenizer, *, session=None):
                return _vector(k=3)

        from aisteer360.algorithms.state_control._common.specs import Intervention, TokenScope
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform
        from aisteer360.algorithms.state_control.base import InterventionControl

        class _DeclaredBroadcast(InterventionControl):
            Args = None
            hook_only_hint = "positional directions have no intervention-spec form"

            def _configure(self):
                self._template = (Intervention(
                    layers=(1,),
                    transform=AdditiveTransform(_LyingSource()),
                    scope=TokenScope("all"),
                ),)

        control = _DeclaredBroadcast()
        pipeline = SteeringPipeline(controls=[control], backend=SERVE_SPEC, lazy_init=True)
        pipeline.steer_backend = BackendSpec(kind="huggingface")
        pipeline.model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=2)
        pipeline.tokenizer = wordlevel_tokenizer()

        # check() consults construction-time facts, so the declared kinds pass
        assert pipeline.check(
            steer_backend=BackendSpec(kind="huggingface"), inference_backend=SERVE_SPEC,
        ).supported("generate")

        class _NullStager:
            _discovery = None

            def stage_artifacts(self, payloads):
                return None

        pipeline._backends[SERVE_SPEC] = _NullStager()
        with pytest.raises(UnsupportedOperationError) as excinfo:
            pipeline.steer()
        message = str(excinfo.value)
        assert "_DeclaredBroadcast" in message
        assert "intervention 0" in message
        assert "AdditiveTransform" in message
