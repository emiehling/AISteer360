"""Tests for fit keys, the fit cache, and the factory's use of it (`algorithms/core/fits.py`)."""
import pytest
import torch

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.fits import (
    FitCache,
    fit_cache_scope,
    fit_digests,
    resolved_id,
    source_access,
    stores_a_fit,
    weights_context,
)
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Recipe, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.input_control.few_shot.control import FewShot
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.cast.control import CAST, _BehaviorFit
from steerability.algorithms.state_control.common import sources
from steerability.algorithms.state_control.common.fit_specs import ConditionSearchSpec, VectorTrainSpec
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PAIRS = {"positives": ["yes indeed", "sure thing", "of course"], "negatives": ["no way", "never", "not at all"]}
OTHER_PAIRS = {"positives": ["happy", "glad"], "negatives": ["sad", "upset"]}


@pytest.fixture()
def fit_calls(monkeypatch):
    """Count the contrastive fits that run."""
    calls = []
    original = sources.ContrastiveFit._fit

    def counting(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sources.ContrastiveFit, "_fit", counting)
    return calls


def _caa(multiplier: float = 2.0, data=PAIRS) -> Recipe:
    return Recipe((entry(CAA, data=data, layer_id=1, multiplier=multiplier, train_spec={"method": "mean_diff"}),))


def _factory(tmp_path, **fields) -> tuple[PipelineFactory, Store]:
    store = Store.open(tmp_path / "store")
    return PipelineFactory(Protocol.create(MODEL, **fields), store=store, device_map=None), store


class TestKeys:
    def test_fit_digests_ignore_application_arguments_and_need_no_model(self):
        assert fit_digests(_caa(1.0)) == fit_digests(_caa(4.0))
        assert len(fit_digests(_caa())) == 1
        assert fit_digests(_caa(data=PAIRS)) != fit_digests(_caa(data=OTHER_PAIRS))
        assert fit_digests(Recipe((entry(SystemPrompt, text="a"),))) == ()

    def test_weights_context(self, tmp_path):
        assert weights_context(_caa()) == "base"
        (tmp_path / "adapter").mkdir()
        (tmp_path / "adapter" / "adapter_config.json").write_text("{}")
        with_lora = Recipe((entry(LoadLoRA, path=str(tmp_path / "adapter"), base_model=MODEL), *_caa().entries))
        assert weights_context(with_lora) != "base"

    def test_a_recipe_with_a_structural_control_stores_no_fit(self, tmp_path):
        cache = FitCache(Protocol.create(MODEL), "base", None)
        assert stores_a_fit(_caa().instantiate(), [], cache)
        (tmp_path / "adapter").mkdir()
        (tmp_path / "adapter" / "adapter_config.json").write_text("{}")
        with_lora = Recipe((entry(LoadLoRA, path=str(tmp_path / "adapter"), base_model=MODEL), *_caa().entries))
        assert not stores_a_fit(with_lora.instantiate(), [], cache)

    def test_resolved_id_depends_on_the_fits(self):
        recipe = _caa()
        assert resolved_id(recipe, {"k": "1"}) != resolved_id(recipe, {"k": "2"})
        assert resolved_id(recipe, {}) != recipe.id


class TestFitCache:
    def test_put_returns_the_earlier_value(self, tmp_path):
        store = Store.open(tmp_path)
        protocol = Protocol.create(MODEL)
        first, second = FitCache(protocol, "base", store), FitCache(protocol, "base", store)
        assert torch.equal(first.put("d", torch.ones(2), 1.0), torch.ones(2))
        assert torch.equal(second.put("d", torch.zeros(2), 1.0), torch.ones(2))
        assert torch.equal(second.get("d"), torch.ones(2))

    def test_a_stored_none_is_a_hit(self, tmp_path):
        cache = FitCache(Protocol.create(MODEL), "base", Store.open(tmp_path))
        assert not cache.has("d")
        assert cache.put("d", None, 0.0) is None
        assert cache.has("d") and cache.get("d") is None

    def test_get_returns_fresh_objects_and_records_use(self, tmp_path):
        cache = FitCache(Protocol.create(MODEL), "base", None)
        cache.put("d", {"x": torch.ones(2)}, 0.0)
        first, second = cache.get("d"), cache.get("d")
        assert first["x"] is not second["x"]
        assert list(cache.used()) == [cache.key("d")]

    def test_an_unencodable_value_is_process_local(self, tmp_path):
        cache = FitCache(Protocol.create(MODEL), "base", Store.open(tmp_path))
        value = object()
        assert cache.put("d", value, 0.0) is value
        assert not cache.has("d")
        assert cache.has_local()


class TestFactory:
    def test_steering_one_recipe_twice_fits_once(self, tmp_path, fit_calls):
        factory, store = _factory(tmp_path)
        try:
            with factory.steered(_caa()) as first:
                first_text = first.pipeline.generate(text="hello", max_new_tokens=3, do_sample=False)
            with factory.steered(_caa()) as second:
                second_text = second.pipeline.generate(text="hello", max_new_tokens=3, do_sample=False)
        finally:
            factory.release()
        assert len(fit_calls) == 1
        assert first.resolved_id == second.resolved_id
        assert first_text == second_text
        assert store.get_resolution(factory.protocol.id, _caa().id) == first.resolved_id
        assert (factory.stats.fits_run, factory.stats.fits_reused) == (1, 1)

    def test_a_new_multiplier_or_decoding_setting_does_not_refit(self, tmp_path, fit_calls):
        factory, store = _factory(tmp_path)
        try:
            with factory.steered(_caa(2.0)):
                pass
            with factory.steered(_caa(5.0)):
                pass
        finally:
            factory.release()
        sampled = PipelineFactory(
            Protocol.create(MODEL, generate_defaults={"temperature": 0.7}), store=store, device_map=None,
        )
        try:
            with sampled.steered(_caa(2.0)):
                pass
        finally:
            sampled.release()
        assert len(fit_calls) == 1

    def test_with_all_fits_cached_the_plan_has_no_fit(self, tmp_path):
        factory, store = _factory(tmp_path)
        try:
            with factory.steered(_caa()):
                pass
        finally:
            factory.release()
        control = _caa().instantiate()[0]
        cache = FitCache(factory.protocol, "base", store)
        assert control.steer_access() == ModelAccess.CAPTURE
        with fit_cache_scope(cache):
            assert control.steer_access() == ModelAccess.FACTS
            assert control.steer_fits() == ()
            assert source_access(control.declared_fit_sources()[0]) == ModelAccess.FACTS

    def test_changing_the_gate_search_keeps_the_behavior_fit(self, tmp_path, monkeypatch):
        behavior_calls = []
        original = _BehaviorFit.resolve

        def counting(self, *args, **kwargs):
            behavior_calls.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(_BehaviorFit, "resolve", counting)

        def recipe(step: float) -> Recipe:
            return Recipe((entry(
                CAST, behavior_data=PAIRS, behavior_fit=VectorTrainSpec(method="mean_diff", prompt_format="raw"),
                behavior_layer_ids=[1], condition_data=OTHER_PAIRS,
                condition_fit=VectorTrainSpec(method="mean_diff", prompt_format="raw", location="layer_input"),
                search=ConditionSearchSpec(candidate_layers=[0, 1], threshold_step=step),
            ),))

        factory, _ = _factory(tmp_path)
        try:
            with factory.steered(recipe(0.1)) as first:
                first_point = first.pipeline.state_controls[0].condition_point
            with factory.steered(recipe(0.05)):
                pass
            with factory.steered(recipe(0.1)) as again:
                again_point = again.pipeline.state_controls[0].condition_point
        finally:
            factory.release()
        assert len(behavior_calls) == 1
        assert again_point == first_point
        assert isinstance(again.pipeline.state_controls[0], CAST)

    def test_a_process_local_fit_steers_twice_without_a_resolution(self, tmp_path, monkeypatch):
        from steerability.algorithms.input_control.common.selectors.random import RandomSelector
        from steerability.algorithms.input_control.few_shot import selectors

        class PreparingSelector(RandomSelector):
            prepared = 0

            def prepare(self, model=None, tokenizer=None, data=None, **kwargs):
                PreparingSelector.prepared += 1

        monkeypatch.setitem(selectors.SELECTOR_REGISTRY, "preparing", PreparingSelector)
        recipe = Recipe((entry(
            FewShot, positive_example_pool=[{"prompt": "a", "response": "b"}], k_positive=1, selector="preparing",
        ),))
        assert recipe.instantiate()[0].steer_fits() == (("SelectorFit", "calibrated"),)
        factory, store = _factory(tmp_path)
        try:
            with factory.steered(recipe) as first:
                pass
            with factory.steered(recipe) as second:
                pass
        finally:
            factory.release()
        assert PreparingSelector.prepared == 2
        assert first.resolved_id != recipe.id
        assert store.get_resolution(factory.protocol.id, recipe.id) is None

    def test_prepare_skips_a_fit_that_is_not_stored(self, tmp_path, monkeypatch):
        from steerability.algorithms.input_control.common.selectors.random import RandomSelector
        from steerability.algorithms.input_control.few_shot import selectors

        class PreparingSelector(RandomSelector):
            prepared = 0

            def prepare(self, model=None, tokenizer=None, data=None, **kwargs):
                PreparingSelector.prepared += 1

        monkeypatch.setitem(selectors.SELECTOR_REGISTRY, "preparing", PreparingSelector)
        recipe = Recipe((entry(
            FewShot, positive_example_pool=[{"prompt": "a", "response": "b"}], k_positive=1, selector="preparing",
        ),))
        factory, _ = _factory(tmp_path)
        try:
            factory.prepare([recipe])
            assert PreparingSelector.prepared == 0
            with factory.steered(recipe):
                pass
        finally:
            factory.release()
        assert PreparingSelector.prepared == 1

    def test_prepare_fits_once_before_generation(self, tmp_path, fit_calls):
        factory, _ = _factory(tmp_path)
        try:
            factory.prepare([_caa(1.0), _caa(2.0), Recipe((entry(SystemPrompt, text="a"),))])
            assert len(fit_calls) == 1
            with factory.steered(_caa(3.0)):
                pass
        finally:
            factory.release()
        assert len(fit_calls) == 1


def test_sasa_served_from_the_cache_constructs_with_a_real_path(tmp_path):
    from pathlib import Path

    from steerability.algorithms.output_control.sasa.control import SASA

    data = {"pos": ["kind words", "nice day"], "neg": ["awful insult", "hateful remark"]}
    recipe = Recipe((entry(SASA, gen_wv_data=data, beta=2.0, prompt_format="raw"),))
    factory, _ = _factory(tmp_path)
    try:
        with factory.steered(recipe):
            pass
        with factory.steered(recipe) as served:
            control = served.pipeline.output_controls[0]
    finally:
        factory.release()
    assert isinstance(control.args.wv_path, str) and Path(control.args.wv_path).is_dir()
    assert control.args.gen_wv_data is None
