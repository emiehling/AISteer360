"""Bundle identity: recipe ids from the recipe, entries as recipe JSON, the lock, and fit-digest staleness."""
import json

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.recipe import Recipe, entry
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.few_shot.control import FewShot
from steerability.algorithms.state_control.caa.control import CAA
from steerability.spipe import SPipe, SpipeStaleError

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

CAA_KWARGS = dict(
    data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]},
    train_spec={"method": "mean_diff", "accumulate": "last_token"},
    layer_id=1,
    multiplier=2.0,
)


def make_controls():
    few_shot = FewShot(
        directive="Answer formally.",
        positive_example_pool=[{"prompt": "hey", "response": "Good day."}],
        k_positive=1,
    )
    return [few_shot, CAA(**CAA_KWARGS)]


def test_entries_load_as_the_recipe_the_pipeline_was_built_from():
    recipe = Recipe((entry(CAA, **CAA_KWARGS), entry(
        FewShot, directive="Answer formally.", positive_example_pool=[{"prompt": "hey", "response": "Good day."}],
        k_positive=1,
    )))
    spipe = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=recipe.instantiate()).to_spipe(freeze=False)
    assert spipe.recipe_id == recipe.id == Recipe.from_controls(make_controls()).id
    assert spipe.recipe.id == recipe.id
    entries = {"schema": "recipe/1", "entries": [
        {"method": item["method"], "args": item["args"]} for item in spipe.manifest["controls"]
    ]}
    assert Recipe.from_json(entries).id == recipe.id
    assert spipe.protocol is None and spipe.manifest["lock"] is None


def test_recipe_id_does_not_include_the_model_ref():
    a = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=make_controls()).to_spipe(freeze=False)
    b = SteeringPipeline(model_name_or_path="org/other-model", controls=make_controls()).to_spipe(freeze=False)
    assert a.recipe_id == b.recipe_id
    assert a.manifest["model"]["ref"] != b.manifest["model"]["ref"]


def test_two_controls_of_one_method_save_and_load(tmp_path):
    second = dict(CAA_KWARGS, layer_id=0, multiplier=-1.0)
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[CAA(**CAA_KWARGS), CAA(**second)])
    spipe = pipeline.to_spipe(freeze=False)
    loaded = SPipe.load(spipe.save(tmp_path / "two.spipe"))
    controls = loaded.pipeline().state_controls
    assert [(control.layer_id, control.multiplier) for control in controls] == [(1, 2.0), (0, -1.0)]
    assert loaded.recipe_id == spipe.recipe_id


def test_disabled_controls_are_not_written():
    disabled = CAA(**CAA_KWARGS)
    disabled.enabled = False
    spipe = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[*make_controls()[:1], disabled]).to_spipe(
        freeze=False,
    )
    assert [item["method"] for item in spipe.manifest["controls"]] == ["input_control/few_shot"]
    assert all("enabled" not in item for item in spipe.manifest["controls"])


@pytest.fixture(scope="module")
def frozen_caa_dir(tmp_path_factory):
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pipeline = SteeringPipeline(
        model=model, tokenizer=tokenizer, controls=[CAA(**CAA_KWARGS)], model_name_or_path=TINY_MODEL,
    )
    pipeline.steer()
    return pipeline.to_spipe().save(tmp_path_factory.mktemp("spipe") / "caa_dir")


def _edit_manifest(directory, mutate):
    manifest = json.loads((directory / "spipe.json").read_text())
    mutate(manifest)
    (directory / "spipe.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))


def test_fit_digest_invariant_under_multiplier_edit(frozen_caa_dir):
    _edit_manifest(frozen_caa_dir, lambda m: m["controls"][0]["args"].__setitem__("multiplier", 42.0))
    SPipe.load(frozen_caa_dir)  # not stale


def test_fit_digest_sensitive_to_data_edit(frozen_caa_dir):
    def mutate(manifest):
        manifest["controls"][0]["args"]["data"]["fields"]["positives"] = ["EDITED", "kind b"]

    _edit_manifest(frozen_caa_dir, mutate)
    with pytest.raises(SpipeStaleError, match="fit digest|digests to"):
        SPipe.load(frozen_caa_dir)
    loaded = SPipe.load(frozen_caa_dir, allow_stale=True)
    report = loaded.verify()
    assert not report.ok
    assert any("digest" in message for message in report.errors)


def test_recorded_revision_is_applied_on_load(tmp_path):
    pipeline = SteeringPipeline(
        model_name_or_path=TINY_MODEL, controls=make_controls(), hf_model_kwargs={"revision": "abc"},
    )
    saved = pipeline.to_spipe(freeze=False).save(tmp_path / "pinned.spipe")
    loaded = SPipe.load(saved)
    assert loaded.manifest["model"]["revision"] == "abc"

    assert loaded.pipeline().hf_model_kwargs["revision"] == "abc"
    explicit = loaded.pipeline(hf_model_kwargs={"revision": "def", "torch_dtype": "float32"})
    assert explicit.hf_model_kwargs == {"revision": "def", "torch_dtype": "float32"}
    merged = loaded.pipeline(hf_model_kwargs={"torch_dtype": "float32"})
    assert merged.hf_model_kwargs == {"torch_dtype": "float32", "revision": "abc"}


def test_unpinned_bundle_adds_no_revision(tmp_path):
    saved = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=make_controls()).to_spipe(freeze=False).save(
        tmp_path / "unpinned.spipe"
    )
    assert "revision" not in SPipe.load(saved).pipeline().hf_model_kwargs


def test_a_frozen_lock_records_the_protocol(frozen_caa_dir):
    from steerability.algorithms.core.protocol import Protocol

    loaded = SPipe.load(frozen_caa_dir, allow_stale=True)
    lock = loaded.manifest["lock"]
    assert set(lock) == {"protocol_id", "protocol", "facts", "certificate"}
    assert lock["facts"]["model_fingerprint"]
    assert isinstance(loaded.protocol, Protocol)
    assert loaded.protocol.id == lock["protocol_id"]
    assert loaded.protocol.model == TINY_MODEL


def test_a_certificate_requires_a_frozen_bundle():
    from steerability.spipe import SpipeSaveError

    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=make_controls()[:1])
    with pytest.raises(SpipeSaveError, match="frozen"):
        pipeline.to_spipe(freeze=False, certificate={"schema": "certificate/3"})
    frozen = pipeline.to_spipe(freeze=True, certificate={"schema": "certificate/3", "note": "x"})
    assert frozen.manifest["lock"]["certificate"] == {"schema": "certificate/3", "note": "x"}
    with pytest.raises(SpipeSaveError, match="certificate/3"):
        pipeline.to_spipe(freeze=True, certificate={"note": "x"})
