"""The fit contract (spec section 8.1), checked over the registered controls on a tiny model.

Each control in `CONFIGS` has a minimal configuration, an argument change that changes what its fit
exports, and an application-only change. The checks:

- a fit-changing argument changes the fit digest, and an application-only argument does not;
- a control whose `steer()` runs the model or calls capture, generate, or score on its session
  declares a fit;
- two fits give the same exported content (controls whose fit samples from a model are exempt);
- `frozen_form(state)` depends only on the args and `state`;
- `export_state()` after a cache hit equals `export_state()` after the fit that filled the cache, for intervention
  controls and for the product of a structural control that trains;
- steering leaves the recipe of a control unchanged.

Every registered control is either configured here or listed in `NOT_CONFIGURED` with the reason.
"""
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.fits import (
    FitCache,
    apply_control_hits,
    control_fit_digests,
    fit_cache_scope,
    record_control_fits,
)
from steerability.algorithms.core.identity import content_id
from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec
from steerability.algorithms.core.internals.probes.probe_set import ProbeSetFit
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Recipe
from steerability.algorithms.core.registry import REGISTRY, resolve_method_key
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.base import InterventionControl
from steerability.algorithms.state_control.common.fit_specs import ConditionSearchSpec, VectorTrainSpec
from steerability.algorithms.state_control.common.sources import ContrastiveFit
from steerability.algorithms.state_control.common.transforms import AdditiveTransform
from steerability.backends.huggingface.session import ExclusiveSession
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

PAIRS = {"positives": ["the cat sat on", "the cat ran", "a cat"], "negatives": ["the dog sat", "dog ran", "a dog"]}
OTHER = {"positives": ["the mat", "on the mat"], "negatives": ["the dog", "a dog ran"]}
RAW = {"method": "mean_diff", "prompt_format": "raw"}
LABELED = {
    "positives": [f"the cat {i}" for i in range(6)], "negatives": [f"the dog {i}" for i in range(6)],
    "positive_groups": [i // 2 for i in range(6)], "negative_groups": [i // 2 for i in range(6)],
}


@dataclass(frozen=True)
class Config:
    """A minimal configuration of one control with a fit-changing and an application-only change."""

    base: dict
    fit_change: dict
    app_change: dict | None = None
    samples: bool = False


def _router():
    from steerability.algorithms.output_control.routed_decoding.actions import generate, respond
    from steerability.algorithms.output_control.routed_decoding.routing import P, Route, Router

    return Router(routes=[Route("r", when=P("topic"), action=respond("the mat"))], default_action=generate())


CONFIGS = {
    "state_control/caa": Config(
        {"data": PAIRS, "layer_id": 1, "train_spec": RAW}, {"data": OTHER}, {"multiplier": 3.0, "layer_id": 2},
    ),
    "state_control/act_add": Config(
        {"positive_prompt": "the cat", "negative_prompt": "the dog", "layer_id": 1},
        {"positive_prompt": "the mat"}, {"multiplier": 2.0, "alignment": 1},
    ),
    "state_control/angular_steering": Config(
        {"data": PAIRS, "train_spec": RAW, "target_degree": 90.0}, {"data": OTHER}, {"target_degree": 45.0},
    ),
    "state_control/directional_ablation": Config(
        {"data": PAIRS, "train_spec": RAW, "layer_ids": [1]}, {"data": OTHER}, {"alpha": 0.5, "layer_ids": [2]},
    ),
    "state_control/activation_adapter": Config(
        {"transform": AdditiveTransform(ContrastiveFit(data=PAIRS, method="mean_diff", prompt_format="raw")),
         "layer_ids": [1]},
        {"transform": AdditiveTransform(ContrastiveFit(data=OTHER, method="mean_diff", prompt_format="raw"))},
        {"layer_ids": [2], "token_scope": "after_prompt"},
    ),
    "state_control/cast": Config(
        {"behavior_data": PAIRS, "behavior_fit": VectorTrainSpec(method="mean_diff", prompt_format="raw"),
         "behavior_layer_ids": [1], "condition_data": OTHER,
         "condition_fit": VectorTrainSpec(method="mean_diff", prompt_format="raw", location="layer_input"),
         "search": ConditionSearchSpec(candidate_layers=[0, 1], threshold_step=0.1)},
        {"behavior_data": OTHER}, {"behavior_vector_strength": 3.0, "token_scope": "after_prompt"},
    ),
    "state_control/iti": Config(
        {"data": LABELED, "num_heads": 2}, {"data": {**LABELED, "positives": [f"a mat {i}" for i in range(6)]}},
        {"alpha": 3.0, "token_scope": "all"},
    ),
    "output_control/sasa": Config(
        {"gen_wv_data": {"pos": ["kind words", "nice day"], "neg": ["awful insult", "hateful remark"]}},
        {"gen_wv_data": {"pos": ["good", "great"], "neg": ["bad", "awful"]}}, {"beta": 3.0},
    ),
    "output_control/value_guidance": Config(
        {"value": {"kind": "subspace_margin", "data": {"positives": ["kind", "nice"], "negatives": ["bad", "mean"]}}},
        {"value": {"kind": "subspace_margin", "data": {"positives": ["good", "great"], "negatives": ["bad", "mean"]}}},
        {"beta": 3.0},
    ),
    "output_control/routed_decoding": Config(
        {"probes": ProbeSetFit(data={"topic": ContrastivePairs(**{k: PAIRS[k] for k in ("positives", "negatives")})},
                               spec=ProbeFitSpec(method="mean_diff", candidate_layers=[1])), "rules": _router()},
        {"probes": ProbeSetFit(data={"topic": ContrastivePairs(**{k: OTHER[k] for k in ("positives", "negatives")})},
                               spec=ProbeFitSpec(method="mean_diff", candidate_layers=[1]))},
        None,
    ),
    "input_control/few_shot": Config(
        {"positive_example_pool": [{"prompt": "a", "response": "b"}], "k_positive": 1},
        {}, {"k_positive": 0},
    ),
    "input_control/system_prompt": Config({"text": "be brief"}, {}, {"text": "be kind"}),
    "input_control/user_prefix": Config({"text": "please"}, {}, {"text": "kindly"}),
}

NOT_CONFIGURED = {
    "input_control/cpo": "trains a causal reward model on a downloaded embedding model; covered in test_cpo.py",
    "input_control/gepa": "needs a reflection model and a scored train set; its fit samples, see test_gepa.py",
    "input_control/prewrite": "needs a rewriter model; covered in test_prewrite.py",
    "output_control/best_of_n": "declares no fit; covered in test_best_of_n.py",
    "output_control/budget_forcing": "declares no fit; covered in test_budget_forcing.py",
    "output_control/constrained_decoding": "declares no fit; covered in test_constrained_decoding.py",
    "output_control/contrastive_decoding": "needs an auxiliary model; declares no fit",
    "output_control/contrastive_guidance": "needs auxiliary sources; declares no fit",
    "output_control/deal": "declares no fit; covered in test_deal.py",
    "output_control/dexperts": "needs expert models; declares no fit",
    "output_control/phased_decoding": "declares no fit; covered in test_phased_decoding.py",
    "output_control/rad": "needs a reward model; declares no fit",
    "output_control/search_decoding": "declares no fit; covered in test_search_driver.py",
    "output_control/stopping_rules": "declares no fit",
    "state_control/pasta": "a dict head map has no fit; HeadProfile fits are covered in test_pasta.py",
    "structural_control/load_checkpoint": "an artifact loader; receives the model by pipeline convention, fits nothing",
    "structural_control/load_lora": "an artifact loader; receives the model by pipeline convention, fits nothing",
    "structural_control/mergekit": "merges models; its product route is covered in tests/core/test_fits.py",
    "structural_control/sft": "trains on a dataset; covered below and in tests/core/test_fits.py",
    "structural_control/dpo": "trains on preference data with the product route of sft (TRLMixin)",
    "structural_control/grpo": "trains with a reward callable with the product route of sft (TRLMixin)",
    "structural_control/apo": "trains on preference data with the product route of sft (TRLMixin)",
}


def _registered() -> set[str]:
    return {f"{category}/{name}" for category, bucket in REGISTRY.items() for name in bucket}


def test_every_registered_control_is_configured_or_excused():
    assert _registered() <= set(CONFIGS) | set(NOT_CONFIGURED)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_llama(num_layers=4, hidden=32, heads=4)


def _control(method: str, **changes):
    config = CONFIGS[method]
    return resolve_method_key(method).control_cls(**{**config.base, **changes})


def _steer(control, model):
    pipeline = SteeringPipeline(controls=[control], model=model, tokenizer=wordlevel_tokenizer())
    pipeline.steer()
    return pipeline


@pytest.mark.parametrize("method", sorted(CONFIGS))
def test_fit_digests_follow_the_fit_inputs(method):
    config = CONFIGS[method]
    base = control_fit_digests(_control(method))
    if config.fit_change:
        assert control_fit_digests(_control(method, **config.fit_change)) != base
    if config.app_change is not None:
        assert control_fit_digests(_control(method, **config.app_change)) == base


@pytest.mark.parametrize("method", sorted(CONFIGS))
def test_model_side_work_is_declared_as_a_fit(method, model, monkeypatch):
    calls = []
    for name in ("capture", "generate", "score"):
        original = getattr(ExclusiveSession, name)

        def recording(self, *args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(ExclusiveSession, name, recording)
    forwards = []
    handle = model.register_forward_pre_hook(lambda module, args: forwards.append(1))
    try:
        control = _control(method)
        fits = control.steer_fits()
        _steer(control, model)
    finally:
        handle.remove()
    if calls or forwards:
        assert fits, f"{method} ran the model in steer() ({calls or 'forward'}) but declares no fit"


@pytest.mark.parametrize("method", sorted(CONFIGS))
def test_two_fits_export_the_same_content(method, model):
    if CONFIGS[method].samples:
        pytest.skip(f"{method}'s fit samples from the model")
    first, second = _control(method), _control(method)
    _steer(first, model)
    _steer(second, model)
    assert content_id(first.export_state()) == content_id(second.export_state())


@pytest.mark.parametrize("method", sorted(CONFIGS))
def test_frozen_form_depends_only_on_args_and_state(method, model):
    from steerability.algorithms.core.base_control import NotFreezableError

    steered = _control(method)
    _steer(steered, model)
    state = steered.export_state()
    try:
        expected = content_id(steered.frozen_form(state))
    except NotFreezableError:
        with pytest.raises(NotFreezableError):
            _control(method).frozen_form(state)
        return
    assert content_id(_control(method).frozen_form(state)) == expected


@pytest.mark.parametrize("method", sorted(method for method in CONFIGS
                                          if issubclass(resolve_method_key(method).control_cls, InterventionControl)))
def test_export_state_after_a_cache_hit_equals_the_fit(method, model):
    cache = FitCache(Protocol.create("tiny"), "base", None)
    results = []
    for _ in range(2):
        controls, pending = apply_control_hits([_control(method)], cache)
        with fit_cache_scope(cache):
            assert (controls[0].steer_access() == ModelAccess.FACTS) == bool(results)
            _steer(controls[0], model)
        record_control_fits(pending, cache)
        results.append(content_id(controls[0].export_state()))
    assert results[0] == results[1]


def test_a_structural_product_after_a_cache_hit_is_the_product_the_fit_stored(tmp_path):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steerability.algorithms.core.store import Store
    from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    store = Store.open(root=tmp_path / "store")
    cache = FitCache(protocol=Protocol.create(model=model_name), weights="base", store=store)
    dataset = Dataset.from_dict({"text": [f"the cat sat {index}" for index in range(4)]})
    results = []
    for _ in range(2):
        control = SFT(
            train_dataset=dataset,
            use_peft=True,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            logging_steps=100,
            load_best_model_at_end=False,
            max_length=16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        pipeline = SteeringPipeline(
            controls=[control],
            model=AutoModelForCausalLM.from_pretrained(model_name),
            tokenizer=tokenizer,
        )
        with fit_cache_scope(cache=cache):
            assert (control.steer_access() == ModelAccess.FACTS) == bool(results)
            assert bool(control.steer_fits()) == (not results)
            pipeline.steer()
        state = control.export_state()
        assert Path(state["artifact"].path).is_relative_to(store.artifacts.root)
        results.append(content_id(value=state))
    assert results[0] == results[1]


@pytest.mark.parametrize("method", sorted(CONFIGS))
def test_steering_leaves_the_recipe_unchanged(method, model):
    control = _control(method)
    before = Recipe.from_controls([control]).id
    _steer(control, model)
    assert Recipe.from_controls([control]).id == before
