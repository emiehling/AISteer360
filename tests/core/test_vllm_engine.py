"""Engine-gated tests for `VLLMBackend`: prompt-only and driver pipelines on the offline
engine, greedy HF/vLLM parity, and structural checkpoint serving. The whole module skips when
vLLM is not installed; running it requires a GPU-capable environment with the `vllm` extra."""
import pytest

vllm = pytest.importorskip("vllm")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from aisteer360.algorithms.core.execution import (  # noqa: E402
    BackendSpec,
    GenerationItem,
    GenerationParams,
    PreparedPrompt,
    ScoringItem,
)
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline  # noqa: E402
from aisteer360.algorithms.output_control.stopping_rules.control import StoppingRules  # noqa: E402
from aisteer360.backends.vllm import VLLMBackend  # noqa: E402

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def engine_backend():
    spec = BackendSpec(
        kind="vllm",
        model=TINY_MODEL,
        options={"engine_kwargs": {"enforce_eager": True, "max_model_len": 512}},
    )
    try:
        return VLLMBackend(spec)
    except Exception as exception:
        pytest.skip(f"Could not boot the vLLM engine: {exception}")


class TestOfflineEngine:

    def test_prompt_only_generation(self, engine_backend):
        item = GenerationItem(prompt=PreparedPrompt.from_text("The capital of France is"))
        with engine_backend.open_session() as session:
            results = session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))
        output = results[0].output
        assert output.output_ids.shape[0] == 1
        assert output.output_ids.shape[1] > 0
        assert output.finish_reason in ("stop", "eos", "length")

    def test_greedy_parity_with_hf(self, engine_backend):
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        encoded = tokenizer("The sky is", return_tensors="pt")
        hf_full = model.generate(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
            max_new_tokens=8, do_sample=False,
        )
        hf_new = hf_full[0, encoded["input_ids"].size(1):].tolist()

        item = GenerationItem(prompt=PreparedPrompt.from_token_ids(encoded["input_ids"]))
        with engine_backend.open_session() as session:
            results = session.generate([item], GenerationParams(max_new_tokens=8, greedy=True))
        vllm_new = results[0].output.output_ids[0].tolist()
        assert vllm_new[: len(hf_new)] == hf_new[: len(vllm_new)]

    def test_stop_string_semantics(self, engine_backend):
        item = GenerationItem(prompt=PreparedPrompt.from_text("a b a b a b"))
        with engine_backend.open_session() as session:
            results = session.generate(
                [item],
                GenerationParams(max_new_tokens=16, greedy=True, stop_strings=("b",)),
            )
        output = results[0].output
        decoded = engine_backend.tokenizer.decode(output.output_ids[0], skip_special_tokens=True)
        if output.finish_reason == "stop":
            assert "b" in decoded  # ids returned as generated, stop text included

    def test_prompt_logprob_scoring(self, engine_backend):
        tokenizer = engine_backend.tokenizer
        prompt_ids = tokenizer("hello world", return_tensors="pt")["input_ids"]
        ref = prompt_ids[:, -2:]
        item = ScoringItem(
            prompt=PreparedPrompt.from_token_ids(prompt_ids), ref_output_ids=ref,
        )
        with engine_backend.open_session() as session:
            scored = session.score([item], GenerationParams())
        assert scored.shape == (1, 2)
        assert torch.isfinite(scored).all()

    def test_pipeline_end_to_end_with_stopping_rules(self):
        pipeline = SteeringPipeline(
            controls=[StoppingRules(budget=6)],
            lazy_init=True,
            backend=BackendSpec(
                kind="vllm",
                model=TINY_MODEL,
                options={"engine_kwargs": {"enforce_eager": True, "max_model_len": 512}},
            ),
            steer_backend="huggingface",
        )
        try:
            pipeline.steer()
        except Exception as exception:
            pytest.skip(f"Could not boot the vLLM engine: {exception}")
        out = pipeline.generate(text="Once upon a time", max_new_tokens=16, do_sample=False,
                                return_output=True)
        assert out.output_ids.shape[1] <= 6


@pytest.fixture(scope="module")
def plugin_backend():
    """Engine with the vLLM-Hook unified worker active and prefix caching enabled."""
    spec = BackendSpec(
        kind="vllm",
        model=TINY_MODEL,
        options={
            "hook_plugin": True,
            "engine_kwargs": {"max_model_len": 512, "enable_prefix_caching": True},
        },
    )
    try:
        backend = VLLMBackend(spec)
    except Exception as exception:
        pytest.skip(f"Could not boot the plugin engine: {exception}")
    if backend._discovery is None:
        pytest.skip("The engine served no vLLM-Hook discovery payload.")
    return backend


def _hf_reference(control_factory, prompt: str, max_new_tokens: int = 8):
    """Greedy continuation ids under the control's hooks on the in-process backend."""
    from aisteer360.backends.huggingface import HFBackend

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    control = control_factory()
    pipeline = SteeringPipeline(controls=[control], lazy_init=True)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()
    out = pipeline.generate(text=prompt, max_new_tokens=max_new_tokens, do_sample=False,
                            return_output=True)
    return out.output_ids[0].tolist(), control


def _steered_vector(model_ref: str, hidden: int, layers, k: int = 1, seed: int = 5):
    from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector

    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        model_type="llama",
        directions={lid: 4.0 * torch.randn(k, hidden, generator=generator) for lid in layers},
    )


class TestSpecParityOnEngine:
    """Greedy-decode parity per exported control (§8.2). Skips without a live plugin engine."""

    def _parity(self, plugin_backend, control_factory, prompt="The committee reviewed the plan"):
        reference_ids, _ = _hf_reference(control_factory, prompt)

        control = control_factory()
        pipeline = SteeringPipeline(
            controls=[control], lazy_init=True, backend=plugin_backend.spec,
            steer_backend="huggingface",
        )
        pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        pipeline.tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        pipeline._backends[plugin_backend.spec] = plugin_backend
        pipeline.steer()
        out = pipeline.generate(text=prompt, max_new_tokens=8, do_sample=False, return_output=True)
        engine_ids = out.output_ids[0].tolist()
        overlap = min(len(reference_ids), len(engine_ids))
        assert engine_ids[:overlap] == reference_ids[:overlap]

    def test_caa_parity(self, plugin_backend):
        hidden = plugin_backend._layout.hidden_size
        self._parity(
            plugin_backend,
            lambda: __import__(
                "aisteer360.algorithms.state_control.caa.control", fromlist=["CAA"]
            ).CAA(steering_vector=_steered_vector(TINY_MODEL, hidden, [1]), layer_id=1, multiplier=6.0),
        )

    def test_directional_ablation_parity(self, plugin_backend):
        hidden = plugin_backend._layout.hidden_size
        from aisteer360.algorithms.state_control.directional_ablation.control import (
            DirectionalAblation,
        )
        self._parity(
            plugin_backend,
            lambda: DirectionalAblation(
                steering_vector=_steered_vector(TINY_MODEL, hidden, [1, 2]), layer_ids=[1, 2],
            ),
        )

    def test_angular_steering_parity(self, plugin_backend):
        hidden = plugin_backend._layout.hidden_size
        from aisteer360.algorithms.state_control.angular_steering.control import AngularSteering
        self._parity(
            plugin_backend,
            lambda: AngularSteering(
                steering_vector=_steered_vector(TINY_MODEL, hidden, [1], k=2),
                target_degree=40.0, intervention_point="layer_output",
            ),
        )

    def test_steered_after_baseline_shared_prefix(self, plugin_backend):
        """The salting rule's regression alarm: a steered request after a baseline request over
        the same prompt must not reuse KV computed without the intervention."""
        from aisteer360.algorithms.state_control._common.specs import (
            Intervention,
            TokenScope,
            lower_interventions,
        )
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform
        from aisteer360.algorithms.core.execution import InterventionEntry

        hidden = plugin_backend._layout.hidden_size
        vector = _steered_vector(TINY_MODEL, hidden, [1])
        spec = lower_interventions(
            [Intervention(
                layers=(1,),
                transform=AdditiveTransform(vector.directions, strength=8.0),
                scope=TokenScope("all"),
            )],
            num_layers=plugin_backend._layout.num_layers,
        )
        prompt = PreparedPrompt.from_text("The committee reviewed the proposal carefully")
        params = GenerationParams(max_new_tokens=8, greedy=True)
        with plugin_backend.open_session() as session:
            baseline_first = session.generate([GenerationItem(prompt=prompt)], params)
            steered = session.generate(
                [GenerationItem(prompt=prompt, state_entries=(InterventionEntry(spec=spec),))],
                params,
            )
            baseline_again = session.generate([GenerationItem(prompt=prompt)], params)
        assert steered[0].output.output_ids.tolist() != baseline_first[0].output.output_ids.tolist()
        assert baseline_again[0].output.output_ids.tolist() == baseline_first[0].output.output_ids.tolist()

    def test_scored_vs_generated_scope_agreement(self, plugin_backend):
        """`after_prompt` scoring remaps to `from_position` at the original prompt length, so a
        reference scored under the spec matches in-process scoring under the same hooks."""
        from aisteer360.algorithms.state_control.caa.control import CAA

        hidden = plugin_backend._layout.hidden_size
        factory = lambda: CAA(
            steering_vector=_steered_vector(TINY_MODEL, hidden, [1]), layer_id=1,
            multiplier=6.0, token_scope="after_prompt",
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        prompt_ids = tokenizer("hello world example", return_tensors="pt")["input_ids"]
        ref_ids = tokenizer(" one two", return_tensors="pt", add_special_tokens=False)["input_ids"]

        hf_pipeline = SteeringPipeline(controls=[factory()], lazy_init=True)
        hf_pipeline.model = model
        hf_pipeline.tokenizer = tokenizer
        hf_pipeline.steer()
        hf_scores = hf_pipeline.compute_logprobs(prompt_ids, ref_output_ids=ref_ids)

        engine_pipeline = SteeringPipeline(
            controls=[factory()], lazy_init=True, backend=plugin_backend.spec,
            steer_backend="huggingface",
        )
        engine_pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        engine_pipeline.tokenizer = tokenizer
        engine_pipeline._backends[plugin_backend.spec] = plugin_backend
        engine_pipeline.steer()
        engine_scores = engine_pipeline.compute_logprobs(prompt_ids, ref_output_ids=ref_ids)
        assert torch.allclose(hf_scores, engine_scores, atol=5e-2, rtol=5e-2)

    def test_chunked_prefill_last_k_exactness(self, plugin_backend):
        """`last_k` selects absolute positions, so a long prompt under chunked prefill steers
        exactly the last k prompt rows plus decode rows (§3.4)."""
        from aisteer360.algorithms.state_control.caa.control import CAA

        hidden = plugin_backend._layout.hidden_size
        long_prompt = " ".join(["review"] * 96)
        factory = lambda: CAA(
            steering_vector=_steered_vector(TINY_MODEL, hidden, [1]), layer_id=1,
            multiplier=6.0, token_scope="last_k", last_k=3,
        )
        reference_ids, _ = _hf_reference(factory, long_prompt)

        control = factory()
        pipeline = SteeringPipeline(
            controls=[control], lazy_init=True, backend=plugin_backend.spec,
            steer_backend="huggingface",
        )
        pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        pipeline.tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        pipeline._backends[plugin_backend.spec] = plugin_backend
        pipeline.steer()
        out = pipeline.generate(text=long_prompt, max_new_tokens=8, do_sample=False,
                                return_output=True)
        engine_ids = out.output_ids[0].tolist()
        overlap = min(len(reference_ids), len(engine_ids))
        assert engine_ids[:overlap] == reference_ids[:overlap]


class TestCaptureOnEngine:
    """P3 capture and probe-path fixtures. Skip without a live plugin engine."""

    @pytest.mark.parametrize("location", ["layer_output", "layer_input"])
    @pytest.mark.parametrize("mode", ["all_tokens", "last_token"])
    def test_capture_parity_with_in_process_funnel(self, plugin_backend, mode, location):
        from aisteer360.algorithms.core.execution import BackendSpec
        from aisteer360.backends.huggingface import HFBackend

        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        prompts = [
            PreparedPrompt.from_text("The committee reviewed the proposal"),
            PreparedPrompt.from_text("A short prompt"),
        ]
        layers = [1, 2]

        hf_backend = HFBackend.adopt(
            BackendSpec(kind="huggingface"), lambda: model, lambda: tokenizer,
        )
        with hf_backend.open_session() as hf_session:
            reference = hf_session.capture(prompts, layers, mode, location=location)
        with plugin_backend.open_session() as session:
            captured = session.capture(prompts, layers, mode, location=location)

        assert captured.attention_mask.tolist() == reference.attention_mask.tolist()
        for layer in layers:
            assert torch.allclose(
                captured.hidden[layer].float(), reference.hidden[layer].float(),
                atol=5e-2, rtol=5e-2,
            )

    def test_vector_fitted_on_engine_steers_in_process(self, plugin_backend):
        from aisteer360.algorithms.core.internals.data import ContrastivePairs
        from aisteer360.algorithms.state_control._common.estimators import (
            MeanDifferenceEstimator,
        )
        from aisteer360.algorithms.state_control._common.specs import VectorTrainSpec

        pairs = ContrastivePairs(
            positives=["the committee approved it", "they agreed at once"],
            negatives=["the committee rejected it", "they refused at once"],
        )
        spec = VectorTrainSpec(method="mean_diff", accumulate="last_token", prompt_format="raw")
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)

        with plugin_backend.open_session() as session:
            remote_vector = MeanDifferenceEstimator().fit(
                None, tokenizer, data=pairs, spec=spec, session=session,
            )
        local_vector = MeanDifferenceEstimator().fit(model, tokenizer, data=pairs, spec=spec)
        for layer in local_vector.directions:
            assert torch.allclose(
                remote_vector.directions[layer], local_vector.directions[layer],
                atol=5e-2, rtol=5e-2,
            )

    def test_conditional_gate_open_vs_closed_matches_in_process(self, plugin_backend):
        """A probe-gated adapter fires on the gate-open prompt and stays inert on the
        gate-closed prompt, matching in-process decisions (P3.5)."""
        from aisteer360.algorithms.core.internals.probes import Probe
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform
        from aisteer360.algorithms.state_control.activation_adapter.control import (
            ActivationAdapter,
        )

        layout = plugin_backend._layout
        hidden = layout.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)

        open_prompt = "the committee approved the proposal"
        closed_prompt = "nothing to see here at all"
        enc_open = tokenizer(open_prompt, return_tensors="pt")
        enc_closed = tokenizer(closed_prompt, return_tensors="pt")

        # a probe whose weights separate the two prompts at layer 1's input
        from aisteer360.algorithms.core.internals.capture import layerwise_tokenwise_hidden
        hs_open = layerwise_tokenwise_hidden(model, dict(enc_open), location="layer_input")
        hs_closed = layerwise_tokenwise_hidden(model, dict(enc_closed), location="layer_input")
        weight = (hs_open[1].mean(dim=(0, 1)) - hs_closed[1].mean(dim=(0, 1))).float()
        weight = weight / weight.norm()
        score_open = float(hs_open[1].float().mean(dim=(0, 1)) @ weight)
        score_closed = float(hs_closed[1].float().mean(dim=(0, 1)) @ weight)
        bias = -(score_open + score_closed) / 2
        probe = Probe(
            model_type=getattr(model.config, "model_type", "unknown"),
            location="layer_input", pooling="mean", layer_ids=[1],
            weights={1: weight}, bias=bias, meta={},
        )

        generator = torch.Generator().manual_seed(9)
        vector = {2: 6.0 * torch.randn(1, hidden, generator=generator)}

        def factory():
            return ActivationAdapter(
                transform=AdditiveTransform(vector, strength=1.0),
                layer_ids=[2], hook_point="layer_input", token_scope="all",
                **probe.as_condition(allow_model_mismatch=True),
            )

        def run(backend_spec, backend=None):
            pipeline = SteeringPipeline(
                controls=[factory()], lazy_init=True, backend=backend_spec,
                steer_backend="huggingface",
            )
            pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
            pipeline.tokenizer = tokenizer
            if backend is not None:
                pipeline._backends[backend.spec] = backend
            pipeline.steer()
            return [
                pipeline.generate(text=prompt, max_new_tokens=8, do_sample=False, return_output=True)
                for prompt in (open_prompt, closed_prompt)
            ]

        hf_outputs = run("huggingface")
        engine_outputs = run(plugin_backend.spec, plugin_backend)
        for hf_out, engine_out in zip(hf_outputs, engine_outputs):
            hf_ids = hf_out.output_ids[0].tolist()
            engine_ids = engine_out.output_ids[0].tolist()
            overlap = min(len(hf_ids), len(engine_ids))
            assert engine_ids[:overlap] == hf_ids[:overlap]

    def test_routed_decoding_end_to_end_on_engine(self, plugin_backend):
        from aisteer360.algorithms.core.internals.data import ContrastivePairs
        from aisteer360.algorithms.core.internals.probes import (
            P,
            ProbeFitSpec,
            ProbeSetFit,
            RoutingRules,
            Rule,
        )
        from aisteer360.algorithms.output_control.routed_decoding import (
            RoutedDecoding,
            respond,
        )

        pairs = ContrastivePairs(
            positives=["the committee approved it"],
            negatives=["nothing to see here"],
        )
        control = RoutedDecoding(
            probes=ProbeSetFit(
                data={"topic": pairs},
                spec=ProbeFitSpec(method="mean_diff", pooling="mean", location="layer_input",
                                  prompt_format="raw", candidate_layers=[1]),
            ),
            rules=RoutingRules(rules=[Rule("topic", when=P("topic"), action=respond("ROUTED"))]),
        )
        pipeline = SteeringPipeline(
            controls=[control], lazy_init=True, backend=plugin_backend.spec,
            steer_backend="huggingface",
        )
        pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        pipeline.tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        pipeline._backends[plugin_backend.spec] = plugin_backend
        pipeline.steer()
        text = pipeline.generate(text="the committee approved it", max_new_tokens=8, do_sample=False)
        assert isinstance(text, str)


class TestConstraintParityOnEngine:
    """P4 parity fixture: one declarative source constrains identically on both arms."""

    def test_json_schema_constrained_parity(self, engine_backend):
        import json

        from aisteer360.algorithms.output_control.constrained_decoding import (
            ConstrainedDecoding,
        )

        pytest.importorskip("xgrammar")
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        prompt = "Return a JSON object:"

        def run(backend_spec, backend=None):
            control = ConstrainedDecoding(json_schema=schema, include_in_scoring=False)
            pipeline = SteeringPipeline(
                controls=[control], lazy_init=True, backend=backend_spec,
                steer_backend="huggingface",
            )
            pipeline.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
            pipeline.tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
            if backend is not None:
                pipeline._backends[backend.spec] = backend
            pipeline.steer()
            return pipeline.generate(text=prompt, max_new_tokens=24, do_sample=False)

        hf_text = run("huggingface")
        engine_text = run(engine_backend.spec, engine_backend)
        assert json.loads(engine_text) is not None
        assert json.loads(hf_text) is not None
        assert engine_text == hf_text
