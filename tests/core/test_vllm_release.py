"""Engine-gated tests for deterministic `VLLMBackend` release: boot->release->boot in one
process, idempotence, released-instance errors, and pipeline-level release with
reconstruct-on-next-use. The whole module skips when vLLM is not installed; running it requires a
GPU-capable environment with the `vllm` extra.

This is a separate module because the boot->release->boot test must not share a process-lifetime
engine with module-scoped fixtures from another file mid-test.
"""
import pytest

vllm = pytest.importorskip("vllm")

from aisteer360.algorithms.core.execution import (  # noqa: E402
    GenerationItem,
    GenerationParams,
    PreparedPrompt,
)
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline  # noqa: E402
from aisteer360.backends.vllm import VLLMBackend  # noqa: E402

TINY_MODEL = "JackFram/llama-68m"


def _spec():
    from aisteer360.algorithms.core.execution import BackendSpec

    return BackendSpec(
        kind="vllm",
        model=TINY_MODEL,
        options={
            "engine_kwargs": {
                "enforce_eager": True,
                "max_model_len": 512,
                "gpu_memory_utilization": 0.25,
            }
        },
    )


def _boot_or_skip():
    try:
        return VLLMBackend(_spec())
    except Exception as exception:
        pytest.skip(f"Could not boot the vLLM engine: {exception}")


def _generate_once(backend) -> list:
    item = GenerationItem(prompt=PreparedPrompt.from_text("The capital of France is"))
    with backend.open_session() as session:
        return session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))


def test_boot_release_boot():
    """Construct, generate, release, then construct a second engine with the same spec and
    generate again, in one process; both generations succeed."""
    first = _boot_or_skip()
    first_results = _generate_once(first)
    assert first_results[0].output.output_ids.shape[1] > 0
    first.release()

    second = VLLMBackend(_spec())
    try:
        second_results = _generate_once(second)
        assert second_results[0].output.output_ids.shape[1] > 0
    finally:
        second.release()


def test_release_idempotent():
    backend = _boot_or_skip()
    backend.release()
    backend.release()


def test_released_backend_raises():
    backend = _boot_or_skip()
    session = backend.open_session()  # opened before release
    backend.release()

    with pytest.raises(RuntimeError, match="was released"):
        backend.open_session()

    item = GenerationItem(prompt=PreparedPrompt.from_text("The capital of France is"))
    with pytest.raises(RuntimeError, match="was released"):
        session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))


def test_pipeline_release_on_vllm():
    """Steer, generate, release_backends(), then generate again; reconstruct-on-next-use boots a
    fresh engine and succeeds."""
    from aisteer360.algorithms.output_control.stopping_rules.control import StoppingRules

    pipeline = SteeringPipeline(
        controls=[StoppingRules(budget=6)],
        lazy_init=True,
        backend=_spec(),
        steer_backend="huggingface",
    )
    try:
        pipeline.steer()
        # steering runs on the in-process backend, so the engine boots inside the guard rather
        # than on the first generate() call
        pipeline._backend_for(pipeline._resolve_backend_pair()[1])
    except Exception as exception:
        pytest.skip(f"Could not boot the vLLM engine: {exception}")
    try:
        first = pipeline.generate(text="Once upon a time", max_new_tokens=8, do_sample=False)
        assert isinstance(first, str)

        pipeline.release_backends()
        assert pipeline._backends == {}

        second = pipeline.generate(text="Once upon a time", max_new_tokens=8, do_sample=False)
        assert isinstance(second, str)
    finally:
        pipeline.release_backends()
