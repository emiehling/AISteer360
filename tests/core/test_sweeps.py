"""Tests for the sweep layer (`algorithms/core/sweeps.py`): expansion, pre-flight, and the backend pool."""
import pytest

from steerability.algorithms.core.execution.contracts import BackendCapabilities, Capability
from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import BASELINE, Choice, Recipe, Template, entry
from steerability.algorithms.core.sweeps import (
    BackendPool,
    FactoryStats,
    PipelineFactory,
    expand_configurations,
    preflight,
)
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.pasta.control import PASTA

SERVE = BackendSpec(kind="vllm-serve", model="m", options={"base_url": "http://server:8000", "retry_backoff": 0.0})


class TestExpandConfigurations:
    def test_recipes_templates_and_sequences_flatten_in_order(self):
        template = Template((entry(SystemPrompt, text=Choice(("a", "b"))),))
        points = list(expand_configurations({
            "baseline": BASELINE,
            "prompted": [template, Recipe((entry(SystemPrompt, text="c"),))],
        }))
        assert [name for name, _ in points] == ["baseline", "prompted", "prompted", "prompted"]
        assert [point.assignment for _, point in points] == [
            {}, {"system_prompt.text": "a"}, {"system_prompt.text": "b"}, {},
        ]

    def test_repeated_recipes_in_one_pipeline_raise(self):
        recipe = Recipe((entry(SystemPrompt, text="a"),))
        with pytest.raises(ValueError, match="two points"):
            list(expand_configurations({"p": [recipe, Recipe((entry(SystemPrompt, text="a"),))]}))

    def test_an_empty_pipeline_raises(self):
        with pytest.raises(ValueError, match="BASELINE"):
            list(expand_configurations({"p": []}))


class TestPreflight:
    def test_unsupported_points_name_the_pipeline_and_recipe(self):
        recipe = Recipe((entry(PASTA, head_config={0: [0]}),))
        protocol = Protocol.create("m", backend=BackendSpec(kind="vllm", model="m"))
        messages = preflight([("pasta", next(iter(expand_configurations({"x": recipe})))[1])], protocol)
        assert messages and messages[0].startswith(f"pasta [{recipe.id}] PASTA (generate)")
        assert preflight([("baseline", next(iter(expand_configurations({"b": BASELINE})))[1])], protocol) == []


class _Backend:
    def __init__(self, in_process: bool = False):
        self.released = False
        self._atoms = frozenset({Capability.IN_PROCESS_TORCH}) if in_process else frozenset()

    def negotiated_capabilities(self):
        return BackendCapabilities(atoms=self._atoms)

    def release(self):
        self.released = True


class TestBackendPool:
    def test_one_backend_per_spec_and_artifacts(self):
        stats = FactoryStats()
        pool = BackendPool(stats)
        first = pool.get(SERVE, (), _Backend)
        assert pool.get(SERVE, (), _Backend) is first
        assert stats.engine_boots == 1
        assert pool.has_engine()

    def test_other_artifacts_replace_the_kept_backend(self, tmp_path):
        from steerability.algorithms.core.execution.payloads import CheckpointArtifact

        pool = BackendPool()
        first = pool.get(SERVE, (), _Backend)
        second = pool.get(SERVE, (CheckpointArtifact(path=str(tmp_path)),), _Backend)
        assert first.released and second is not first

    def test_backends_with_other_adapters_share_one_engine(self, tmp_path):
        from steerability.algorithms.core.execution.payloads import CheckpointArtifact, LoRAArtifact

        stats = FactoryStats()
        pool = BackendPool(stats=stats)
        adapters = [LoRAArtifact(path=str(tmp_path / name), base_model="m") for name in ("first", "second")]
        first = pool.get(spec=SERVE, artifacts=(adapters[0],), build=_Backend)
        assert pool.get(spec=SERVE, artifacts=(adapters[1],), build=_Backend) is first
        assert pool.get(spec=SERVE, artifacts=(), build=_Backend) is first
        checkpoint = CheckpointArtifact(path=str(tmp_path / "checkpoint"))
        assert pool.get(spec=SERVE, artifacts=(checkpoint, adapters[0]), build=_Backend) is not first
        assert stats.engine_boots == 2

    def test_release_all_counts_fit_releases(self):
        stats = FactoryStats()
        pool = BackendPool(stats)
        backend = pool.get(SERVE, (), _Backend)
        pool.release_all(for_fits=True)
        assert backend.released and not pool.has_engine()
        assert stats.engine_releases_for_fits == 1


def test_two_recipes_share_one_pooled_backend(monkeypatch, tmp_path):
    from steerability.backends.vllm import VLLMServeBackend
    from tests.core.test_vllm_serve_backend import _FakeServer
    from tests.utils.tiny_models import wordlevel_tokenizer

    server = _FakeServer()
    monkeypatch.setattr(
        VLLMServeBackend, "_request_json", lambda self, path, payload, expect_json=True: server.handle(path, payload)
    )
    monkeypatch.setattr("steerability.backends.vllm.backend._client_tokenizer",
                        lambda source, trust_remote_code=False: wordlevel_tokenizer())
    monkeypatch.setattr(
        "steerability.backends.vllm.backend._config_layout", lambda source, trust_remote_code=False: None
    )
    tokenizer_dir = tmp_path / "tokenizer"
    wordlevel_tokenizer().save_pretrained(tokenizer_dir)

    factory = PipelineFactory(Protocol.create("m", backend=SERVE, tokenizer=str(tokenizer_dir)))
    backends = []
    try:
        for text in ("be brief", "be kind"):
            with factory.steered(Recipe((entry(SystemPrompt, text=text),))) as steered:
                backends.append(steered.pipeline._backend_for(SERVE))
                assert steered.pipeline.generate(text="the cat", max_new_tokens=2, do_sample=False) is not None
        assert backends[0] is backends[1]
        assert factory.stats.engine_boots == 1
        assert sum(1 for path, _ in server.requests if path == "/version") == 1
    finally:
        factory.release()
    assert not factory.pool.has_engine()
