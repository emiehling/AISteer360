"""Tests for `SteeringPipeline.steer()` backend threading and artifact hand-off (doc 04 §3, §6)."""
import torch

from aisteer360.backends.base import Artifact
from aisteer360.backends.huggingface.backend import HuggingFaceBackend
from aisteer360.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.structural_control.base import StructuralControl
from tests.conftest import hf_pipeline
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


class _RolloutInputControl(InputControl):
    """A mock optimizer-style input control that records the backend it was steered with."""

    Args = None
    supports_batching = True

    def __init__(self):
        self.received_backend = None
        self.steered = False

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

    def steer(self, model=None, tokenizer=None, backend=None, **kwargs):
        self.received_backend = backend
        self.steered = True


def test_steer_threads_backend_into_control():
    model = tiny_llama(num_layers=2, hidden=32, heads=4, vocab=100)
    control = _RolloutInputControl()
    pipeline = hf_pipeline(controls=[control], model=model, tokenizer=wordlevel_tokenizer())
    pipeline.steer()
    assert control.steered
    # the control received a generate-capable backend to route rollouts through
    assert control.received_backend is not None
    assert control.received_backend.capabilities.capabilities  # non-empty capability set


class _ModelReturningStructural(StructuralControl):
    """A structural control that returns a fresh model to be adopted."""

    Args = None

    def __init__(self, replacement):
        self._replacement = replacement

    def steer(self, model, tokenizer=None, backend=None, **kwargs):
        return self._replacement


def test_structural_control_model_adopted():
    replacement = tiny_llama(num_layers=2, hidden=32, heads=4, vocab=100)
    control = _ModelReturningStructural(replacement)
    # lazy backend; the structural control supplies the model
    from aisteer360.backends.huggingface.backend import HuggingFaceBackend

    backend = HuggingFaceBackend(lazy_init=True)
    backend.tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=[control], backend=backend)
    pipeline.steer()
    assert pipeline.model is replacement


def test_checkpoint_artifact_routed_to_backend():
    """A structural control returning an Artifact routes it to the backend's accept_artifact."""
    accepted = {}

    class _RecordingBackend(HuggingFaceBackend):
        def accept_artifact(self, artifact):
            accepted["artifact"] = artifact

    class _CheckpointStructural(StructuralControl):
        Args = None

        def steer(self, model, tokenizer=None, backend=None, **kwargs):
            return Artifact(kind="checkpoint", ref="/tmp/some-checkpoint")

    backend = _RecordingBackend(lazy_init=True)
    backend.adopt_model(tiny_llama(num_layers=2, hidden=32, heads=4, vocab=100))
    backend.tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=[_CheckpointStructural()], backend=backend)
    pipeline.steer()
    assert accepted["artifact"].kind == "checkpoint"
    assert accepted["artifact"].ref == "/tmp/some-checkpoint"
