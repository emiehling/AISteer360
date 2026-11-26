"""Tests for fit keys, the fit cache, and the factory's use of it (`algorithms/core/fits.py`)."""
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import torch

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import CheckpointArtifact, LoRAArtifact
from steerability.algorithms.core.fits import (
    FitCache,
    control_fit_digests,
    fit_cache_scope,
    fit_digests,
    product_is_stored,
    product_weights,
    resolve_product,
    resolved_id,
    source_access,
    stored_product,
    stores_a_fit,
    weights_context,
)
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Recipe, RecipeError, entry
from steerability.algorithms.core.registry import REGISTRY, register_method
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.input_control.few_shot.control import FewShot
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.cast.control import CAST, _BehaviorFit
from steerability.algorithms.state_control.common import sources
from steerability.algorithms.state_control.common.fit_specs import ConditionSearchSpec, VectorTrainSpec
from steerability.algorithms.structural_control.base import StructuralControl
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PAIRS = {"positives": ["yes indeed", "sure thing", "of course"], "negatives": ["no way", "never", "not at all"]}
OTHER_PAIRS = {"positives": ["happy", "glad"], "negatives": ["sad", "upset"]}


@dataclass
class CountingAdapterArgs(BaseArgs):
    """Arguments of `CountingAdapter`.

    Attributes:
        seed: The seed of the adapter's weights, its one training argument.
        out_dir: An output location, which the fit identity leaves out.
    """

    seed: int = 0
    out_dir: str | None = None


class CountingAdapter(StructuralControl):
    """A structural control whose fit writes a LoRA adapter with weights drawn from `seed`, and records its fits.

    It resolves its product through the fit cache as a control outside the toolkit that trains does.
    """

    Args = CountingAdapterArgs
    fits: ClassVar[list] = []
    _product: LoRAArtifact | None = None
    _installed: bool = False

    def steer_access(self) -> ModelAccess:
        return ModelAccess.FACTS if product_is_stored(control=self) else ModelAccess.MODULE

    def steer_fits(self):
        return () if product_is_stored(control=self) else (("Adapter", "direction"),)

    def fit_identity(self):
        return {"seed": self.seed}

    def artifact_capability(self):
        return Capability.SERVE_LORA

    def export_artifact(self):
        if self._product is None and product_is_stored(control=self):
            self._product = _lora(stored_product(control=self))
        return self._product

    def export_state(self):
        product = self.export_artifact()
        return {} if product is None else {"artifact": product, "base_model": product.base_model}

    def frozen_form(self, state):
        return "structural_control/load_lora", {"path": state["artifact"], "base_model": state["base_model"],
                                                "merge": False}

    def steer(self, model, tokenizer=None, session=None, **kwargs):
        from peft import LoraConfig, get_peft_model

        trained = []

        def fit():
            CountingAdapter.fits.append(self)
            directory = tempfile.mkdtemp(prefix="counting-adapter-")
            torch.manual_seed(self.seed)
            config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
            adapted = get_peft_model(model, config)
            adapted.save_pretrained(directory)
            trained.append(adapted)
            return {"artifact": LoRAArtifact(path=directory, base_model=MODEL), "base_model": MODEL}

        if self._product is None:
            self._product = _lora(resolve_product(control=self, fit=fit))
        if trained:
            return trained[0]
        self._installed = True
        loader = LoadLoRA(path=self._product.path, base_model=self._product.base_model)
        return loader.steer(model=model, tokenizer=tokenizer, session=session)

    def restore(self, model):
        from peft import PeftModel

        return model.unload() if self._installed and isinstance(model, PeftModel) else None


@dataclass
class PassThroughArgs(BaseArgs):
    """Arguments of `PassThrough`.

    Attributes:
        label: A label that tells two entries apart and changes nothing.
    """

    label: str = ""


class PassThrough(StructuralControl):
    """A structural control that returns the model it receives and declares no fit."""

    Args = PassThroughArgs

    def steer_access(self) -> ModelAccess:
        return ModelAccess.FACTS

    def steer(self, model, tokenizer=None, session=None, **kwargs):
        return model


def _lora(state):
    artifact = state["artifact"]
    return LoRAArtifact(path=str(getattr(artifact, "path", artifact)), base_model=str(state["base_model"]))


@pytest.fixture()
def adapter():
    """Register `CountingAdapter` for one test and clear its record of fits."""
    register_method(
        category="structural",
        name="counting_adapter",
        control_cls=CountingAdapter,
        args_cls=CountingAdapterArgs,
    )
    CountingAdapter.fits = []
    try:
        yield CountingAdapter
    finally:
        REGISTRY["structural_control"].pop("counting_adapter", None)


@pytest.fixture()
def pass_through():
    """Register `PassThrough` for one test."""
    register_method(
        category="structural",
        name="pass_through",
        control_cls=PassThrough,
        args_cls=PassThroughArgs,
    )
    try:
        yield PassThrough
    finally:
        REGISTRY["structural_control"].pop("pass_through", None)


def _adapted(seed: int = 0, multiplier: float = 2.0) -> Recipe:
    return Recipe(entries=(entry(CountingAdapter, seed=seed), *_caa(multiplier).entries))


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

    def test_a_structural_product_is_a_stored_fit_and_a_loader_stores_none(self, tmp_path, adapter):
        cache = FitCache(Protocol.create(MODEL), "base", None)
        (tmp_path / "adapter").mkdir()
        (tmp_path / "adapter" / "adapter_config.json").write_text("{}")
        loader = entry(LoadLoRA, path=str(tmp_path / "adapter"), base_model=MODEL)
        assert stores_a_fit(controls=_caa().instantiate(), pending=[], cache=cache)
        assert stores_a_fit(controls=Recipe(entries=(loader, *_caa().entries)).instantiate(), pending=[], cache=cache)
        assert not stores_a_fit(controls=Recipe(entries=(loader,)).instantiate(), pending=[], cache=cache)
        trainer = Recipe(entries=(entry(CountingAdapter, seed=0),)).instantiate()
        assert stores_a_fit(controls=trainer, pending=[], cache=cache)
        product = {"artifact": LoRAArtifact(path=str(tmp_path / "adapter"), base_model=MODEL), "base_model": MODEL}
        cache.put(digest=control_fit_digests(control=trainer[0])[0], value=product, seconds=0.0)
        assert not stores_a_fit(controls=trainer, pending=[], cache=cache)

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


class TestProducts:
    def test_a_product_trains_once_per_fit_key(self, tmp_path, adapter):
        factory, _ = _factory(tmp_path)
        try:
            for recipe in (_adapted(0, 1.0), _adapted(0, 1.0), _adapted(0, 4.0), _adapted(1, 1.0)):
                with factory.steered(recipe=recipe):
                    pass
        finally:
            factory.release()
        assert [fit.seed for fit in CountingAdapter.fits] == [0, 1]
        assert factory.stats.structural_fits == 2

    def test_a_product_is_keyed_by_its_training_and_the_structural_entries_before_it(
        self, tmp_path, adapter, pass_through,
    ):
        factory, _ = _factory(tmp_path)
        recipes = [
            Recipe(entries=(entry(CountingAdapter, seed=0, out_dir="first"),)),
            Recipe(entries=(entry(CountingAdapter, seed=0, out_dir="second"),)),
            Recipe(entries=(entry(CountingAdapter, seed=0), entry(PassThrough, label="after"))),
        ]
        assert len({weights_context(recipe=recipe) for recipe in recipes}) == 3
        assert len(factory.pending_structural_fits(recipes=recipes)) == 1
        try:
            for recipe in recipes:
                with factory.steered(recipe=recipe):
                    pass
        finally:
            factory.release()
        assert len(CountingAdapter.fits) == 1 and factory.stats.structural_fits == 1
        before = Recipe(entries=(entry(PassThrough, label="before"), entry(CountingAdapter, seed=0)))
        assert len(factory.pending_structural_fits(recipes=[before])) == 1
        (digest,) = product_weights(recipe=recipes[0])
        assert product_weights(recipe=before)[digest] != product_weights(recipe=recipes[2])[digest] == "base"

    def test_a_recipe_that_contains_one_training_twice_is_refused(self, tmp_path, adapter):
        factory, _ = _factory(tmp_path)
        twice = Recipe(
            entries=(
                entry(CountingAdapter, name="first", seed=0),
                entry(CountingAdapter, name="second", seed=0),
            ),
        )
        with pytest.raises(RecipeError, match="twice"):
            factory.pending_structural_fits(recipes=[twice])
        other = Recipe(
            entries=(
                entry(CountingAdapter, name="first", seed=0),
                entry(CountingAdapter, name="second", seed=1),
            ),
        )
        assert len(factory.pending_structural_fits(recipes=[other])) == 2

    def test_the_yielded_pipeline_installs_the_stored_product_and_trains_nothing(self, tmp_path, adapter):
        factory, store = _factory(tmp_path)
        recipe = Recipe(entries=(entry(CountingAdapter, seed=0),))
        try:
            with factory.steered(recipe=recipe) as first:
                first_control = first.pipeline.structural_controls[0]
                first_text = first.pipeline.generate(text="hello", max_new_tokens=3, do_sample=False)
            with factory.steered(recipe=recipe) as second:
                second_control = second.pipeline.structural_controls[0]
                second_text = second.pipeline.generate(text="hello", max_new_tokens=3, do_sample=False)
        finally:
            factory.release()
        assert len(CountingAdapter.fits) == 1
        assert first_control not in CountingAdapter.fits and second_control not in CountingAdapter.fits
        assert first.resolved_id == second.resolved_id != recipe.id
        assert first_text == second_text
        product = Path(first_control.export_artifact().path)
        assert product.is_dir() and product.is_relative_to(store.artifacts.root)

    def test_the_fits_of_other_controls_run_on_the_stored_product(self, tmp_path, adapter, monkeypatch):
        factory, store = _factory(tmp_path)
        recipe = _adapted(0, 2.0)
        (key,) = factory.pending_structural_fits(recipes=[recipe])
        stored_at_fit = []
        original = sources.ContrastiveFit._fit

        def recording(self, *args, **kwargs):
            stored_at_fit.append(store.get_fit(fit_key=key) is not None)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(sources.ContrastiveFit, "_fit", recording)
        try:
            with factory.steered(recipe=recipe):
                pass
        finally:
            factory.release()
        assert stored_at_fit == [True] and len(CountingAdapter.fits) == 1

    def test_prepare_fits_the_product_first_and_steers_a_ladder_once(self, tmp_path, adapter, fit_calls):
        factory, _ = _factory(tmp_path)
        ladder = [_adapted(0, value) for value in (1.0, 2.0, 3.0)]
        try:
            factory.prepare(recipes=ladder)
            assert len(CountingAdapter.fits) == 1 and len(fit_calls) == 1
            assert factory.stats.pipelines == 1
            with factory.steered(recipe=ladder[2]):
                pass
        finally:
            factory.release()
        assert len(CountingAdapter.fits) == 1 and len(fit_calls) == 1

    def test_pending_structural_fits_give_the_products_the_store_does_not_hold(self, tmp_path, adapter):
        factory, _ = _factory(tmp_path)
        trained, other = _adapted(0, 1.0), Recipe(entries=(entry(CountingAdapter, seed=1),))
        loader = Recipe(entries=(entry(LoadLoRA, path=str(tmp_path / "adapter"), base_model=MODEL),))
        assert len(factory.pending_structural_fits(recipes=[trained, other, loader, _caa()])) == 2
        try:
            with factory.steered(recipe=trained):
                pass
        finally:
            factory.release()
        assert factory.pending_structural_fits(recipes=[trained, loader, _caa()]) == frozenset()
        assert len(factory.pending_structural_fits(recipes=[other])) == 1
        unstored = PipelineFactory(protocol=Protocol.create(model=MODEL), store=None, device_map=None)
        assert len(unstored.pending_structural_fits(recipes=[trained])) == 1

    def test_a_bundle_of_a_pipeline_served_from_the_store_loads_without_the_store(self, tmp_path, adapter):
        from steerability.spipe import SPipe

        factory, store = _factory(tmp_path)
        recipe = Recipe(entries=(entry(CountingAdapter, seed=3),))
        try:
            with factory.steered(recipe=recipe):
                pass
            with factory.steered(recipe=recipe) as served:
                text = served.pipeline.generate(text="hello", max_new_tokens=4, do_sample=False)
                bundle = served.pipeline.to_spipe(freeze=True).save(path=tmp_path / "bundle.spipe")
        finally:
            factory.release()
        assert len(CountingAdapter.fits) == 1
        store.close()
        shutil.rmtree(tmp_path / "store")
        loaded = SPipe.load(path=bundle).pipeline()
        assert type(loaded.structural_controls[0]).__name__ == "LoadLoRA"
        loaded.steer()
        assert loaded.generate(text="hello", max_new_tokens=4, do_sample=False) == text

    def test_a_stored_product_freezes_to_its_loader_after_a_round_trip_through_the_cache(self, tmp_path):
        from datasets import Dataset

        from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT

        product = tmp_path / "product"
        product.mkdir()
        (product / "adapter_config.json").write_text("{}")
        dataset = Dataset.from_dict({"text": ["hello world"]})
        store = Store.open(root=tmp_path / "store")
        cache = FitCache(protocol=Protocol.create(model=MODEL), weights="base", store=store)
        adapter_state = {"artifact": LoRAArtifact(path=str(product), base_model=MODEL), "base_model": MODEL}
        stored = cache.put(digest="adapter", value=adapter_state, seconds=0.0)
        assert isinstance(stored["artifact"], str)
        method, kwargs = SFT(train_dataset=dataset, use_peft=True).frozen_form(state=stored)
        assert method == "structural_control/load_lora" and kwargs["base_model"] == MODEL
        assert LoadLoRA(**kwargs).path == stored["artifact"]
        stored = cache.put(digest="checkpoint", value={"artifact": CheckpointArtifact(path=str(product))}, seconds=0.0)
        method, kwargs = SFT(train_dataset=dataset, use_peft=False).frozen_form(state=stored)
        assert method == "structural_control/load_checkpoint" and kwargs == {"path": stored["artifact"]}

    def test_the_merge_wrapper_under_a_cache_merges_into_a_directory_of_its_own(self, tmp_path, monkeypatch):
        pytest.importorskip("mergekit")
        from steerability.algorithms.structural_control.wrappers.mergekit import control as mergekit_control

        merges = []

        def merge(merge_config, out_path, options):
            merges.append(Path(out_path))
            Path(out_path, "config.json").write_text(json.dumps({"merged": True}))

        monkeypatch.setattr(mergekit_control.mk_merge, "run_merge", merge)
        out_path = tmp_path / "merged-model"
        out_path.mkdir()
        (out_path / "config.json").write_text(json.dumps({"stale": True}))
        config = {"merge_method": "linear", "models": [{"model": MODEL, "parameters": {"weight": 1.0}}],
                  "dtype": "float32"}
        store = Store.open(root=tmp_path / "store")
        cache = FitCache(protocol=Protocol.create(model=MODEL), weights="base", store=store)
        merged = mergekit_control.MergeKit(config_dict=config, out_path=str(out_path), load_merged=False)
        with fit_cache_scope(cache=cache):
            assert merged.steer_access() == ModelAccess.MODULE
            merged.steer(model=None)
        assert len(merges) == 1 and merges[0] != out_path and not merges[0].exists()
        product = Path(merged.export_artifact().path)
        assert json.loads((product / "config.json").read_text()) == {"merged": True}
        assert json.loads((out_path / "config.json").read_text()) == {"stale": True}
        again = mergekit_control.MergeKit(config_dict=config, out_path=str(out_path), load_merged=False)
        with fit_cache_scope(cache=cache):
            assert again.steer_access() == ModelAccess.FACTS and again.steer_fits() == ()
            again.steer(model=None)
        assert len(merges) == 1 and Path(again.export_artifact().path) == product

    def test_the_sft_wrapper_trains_once_through_the_factory(self, tmp_path, monkeypatch):
        from datasets import Dataset
        from trl import SFTTrainer

        from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT

        trainings = []
        original = SFTTrainer.train

        def counting(self, *args, **kwargs):
            trainings.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(SFTTrainer, "train", counting)
        recipe = Recipe(
            entries=(
                entry(
                    SFT,
                    train_dataset=Dataset.from_dict({"text": [f"hello world {index}" for index in range(4)]}),
                    use_peft=True,
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    logging_steps=100,
                    load_best_model_at_end=False,
                    max_length=32,
                ),
            ),
        )
        factory, store = _factory(tmp_path)
        try:
            with factory.steered(recipe=recipe) as first:
                first_text = first.pipeline.generate(text="hello", max_new_tokens=4, do_sample=False)
                state = first.pipeline.structural_controls[0].export_state()
            with factory.steered(recipe=recipe) as second:
                second_text = second.pipeline.generate(text="hello", max_new_tokens=4, do_sample=False)
        finally:
            factory.release()
        assert len(trainings) == 1 and first_text == second_text
        assert first.resolved_id == second.resolved_id
        assert isinstance(state["artifact"], LoRAArtifact) and state["base_model"] == MODEL
        assert Path(state["artifact"].path).is_relative_to(store.artifacts.root)


def _write_adapter(directory: Path, seed: int) -> str:
    """Write a LoRA adapter with weights drawn from `seed` for the tiny model."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL), config).save_pretrained(directory)
    return str(directory)


def test_recipes_with_installed_lora_adapters_use_the_shared_base_model(tmp_path, adapter, caplog):
    import logging

    factory, _ = _factory(tmp_path)
    loader = entry(LoadLoRA, path=_write_adapter(tmp_path / "adapter", seed=5), base_model=MODEL)
    recipes = {
        "prompt": Recipe(entries=(entry(SystemPrompt, text="a"),)),
        "trained": Recipe(entries=(entry(CountingAdapter, seed=0),)),
        "loader": Recipe(entries=(loader,)),
        "both": Recipe(entries=(loader, entry(SystemPrompt, text="b"))),
    }
    texts: dict[str, list[str]] = {}
    try:
        with caplog.at_level(logging.WARNING, logger="steerability.algorithms.core.sweeps"):
            for name in ("prompt", "trained", "loader", "trained", "both", "prompt"):
                with factory.steered(recipe=recipes[name]) as steered:
                    text = steered.pipeline.generate(text="hello", max_new_tokens=4, do_sample=False)
                    texts.setdefault(name, []).append(text)
    finally:
        factory.release()
    assert "Shared base model changed" not in caplog.text
    assert all(len(set(values)) == 1 for values in texts.values())
    assert len({values[0] for values in texts.values()}) == 4
    assert factory.stats.model_loads == 3


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
