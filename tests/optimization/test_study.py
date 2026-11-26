"""Tests for the study definition: axes, candidates, validation, and identity (`optimization/study.py`)."""
import dataclasses

import pytest

pytest.importorskip("inspect_ai")

from datasets import Dataset

from steerability.algorithms.core.identity import versioned
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Choice, Interval, RecipeError, Template, Text, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.output_control.best_of_n.control import BestOfN
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT
from steerability.optimization import optional, structures
from steerability.optimization.evaluation.plan import make_plan
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.measurement.pools import split_items
from steerability.optimization.study import Candidate, Constraint, Settings
from tests.optimization.helpers import (
    OTHER,
    PAIRS,
    StubFactory,
    SyntheticBenchmark,
    Truth,
    build_study,
    caa_template,
    prompt_template,
)

DATASET = Dataset.from_dict({"text": ["a formal reply", "another formal reply"]})
PROMPT = entry(SystemPrompt, name="prompt", text=Choice(values=("be brief", "be kind")))


def _quiet(recipe, item_id):
    return Truth()


def unversioned_scorer(prompt, continuations, params):
    return [0.0 for _ in continuations]


@versioned("1")
def versioned_scorer(prompt, continuations, params):
    return [0.0 for _ in continuations]


@pytest.fixture()
def store(tmp_path):
    return Store.open(tmp_path / "store")


def _with_space(store, space, **kwargs):
    return build_study(store, _quiet, space=space, **kwargs)


class TestValidate:
    def test_a_valid_study_passes(self, store):
        build_study(store, _quiet).validate(store)

    def test_a_study_with_a_loader_structure_validates_and_runs(self, store):
        lora = entry(LoadLoRA, name="lora", path=Choice(values=("adapter-a", "adapter-b")), base_model="synthetic")
        study = _with_space(store, structures(optional(lora), optional(PROMPT)))
        study.validate(store=store)
        study.objective.set_truth(truth=lambda recipe, item_id: Truth(gain=0.1))
        session = StudySession(study=study, store=store, factory=StubFactory(protocol=study.protocol))
        session.start()
        values = {"lora.path": "adapter-b", "prompt.text": "be kind"}
        assert session.evaluate(candidate=Candidate(structure="lora+prompt", values=values)).status == "accepted"
        assert session.evaluate(candidate=Candidate(structure="lora", values={"lora.path": "adapter-a"})).accepted

    def test_a_training_argument_left_as_the_solved_axis_points_to_searched(self, store):
        rate = Interval(low=1e-5, high=1e-4)
        sft = Template(entries=(entry(SFT, train_dataset=DATASET, use_peft=True, learning_rate=rate),))
        with pytest.raises(RecipeError, match="study.searched"):
            _with_space(store, {"lora": sft}).validate(store=store)
        _with_space(store, {"lora": sft}, searched={"sft.learning_rate"}).validate(store=store)

    def test_a_training_that_leaves_no_product_to_store_is_refused(self, store):
        merged = Template(entries=(entry(SFT, train_dataset=DATASET, use_peft=True, merge_lora_after_train=True),))
        with pytest.raises(ValueError, match="merged_output_dir"):
            _with_space(store, {"merged": merged}).validate(store=store)
        stored = Template(
            entries=(
                entry(
                    SFT,
                    train_dataset=DATASET,
                    use_peft=True,
                    merge_lora_after_train=True,
                    merged_output_dir="merged",
                ),
            ),
        )
        _with_space(store, {"merged": stored}).validate(store=store)

    def test_an_unversioned_callable_is_refused(self, store):
        unversioned = Template((entry(BestOfN, scorer=unversioned_scorer, n=Choice((2, 4))),))
        with pytest.raises(ValueError, match="version"):
            _with_space(store, {"best": unversioned}).validate(store)
        marked = Template((entry(BestOfN, scorer=versioned_scorer, n=Choice((2, 4))),))
        _with_space(store, {"best": marked}).validate(store)

    def test_two_ordered_axes_left_free_are_refused(self, store):
        template = caa_template(layer_id=Choice((1, 2), ordered=True))
        with pytest.raises(ValueError, match="ordered axes"):
            _with_space(store, {"caa": template}).validate(store)
        _with_space(store, {"caa": template}, searched={"caa.layer_id"}).validate(store)

    def test_a_solved_axis_that_changes_a_fit_is_refused_until_it_is_searched(self, store):
        template = caa_template(1.0, data=Choice((PAIRS, OTHER), ordered=True))
        with pytest.raises(RecipeError, match="study.searched"):
            _with_space(store, {"caa": template}).validate(store)
        _with_space(store, {"caa": template}, searched={"caa.data"}).validate(store)

    def test_searched_and_monotone_keys_must_be_axes(self, store):
        with pytest.raises(ValueError, match="searched"):
            build_study(store, _quiet, searched={"caa.unknown"}).validate(store)
        with pytest.raises(ValueError, match="monotone"):
            _with_space(store, {"prompt": prompt_template()}, monotone={"prompt.text"}).validate(store)
        build_study(store, _quiet, monotone={"caa.multiplier"}).validate(store)

    def test_a_discrete_solved_axis_has_at_most_twice_the_ladder_points(self, store):
        template = caa_template(Choice(tuple(float(value) for value in range(9)), ordered=True))
        with pytest.raises(ValueError, match="ladder_points"):
            _with_space(store, {"caa": template}).validate(store)

    def test_benchmark_names_must_be_distinct(self, store):
        study = build_study(store, _quiet)
        twin = SyntheticBenchmark("behavior", 3000, kind="capability", truth=_quiet, base=0.8)
        twin.prepare(store)
        pools = {**study.pools, twin.id: split_items(twin, held_out=2000, seed=0)}
        pools.pop(study.constraints[0].benchmark.id)
        named = dataclasses.replace(study, constraints=(Constraint(twin, 0.05),), pools=pools)
        with pytest.raises(ValueError, match="distinct"):
            named.validate(store)

    def test_an_unset_temperature_is_refused(self, store):
        study = build_study(store, _quiet)
        unset = dataclasses.replace(study, protocol=Protocol.create("synthetic-model"))
        with pytest.raises(ValueError, match="temperature"):
            unset.validate(store)

    def test_a_protocol_without_versions_is_refused(self, store):
        study = build_study(store, _quiet)
        bare = dataclasses.replace(study, protocol=dataclasses.replace(study.protocol, versions={}))
        with pytest.raises(ValueError, match="versions"):
            bare.validate(store)


class TestCandidates:
    def test_default_candidates_bind_for_every_structure(self, store):
        shared = caa_template(Interval(0.0, 2.0, axis="strength"),
                              layer_id=Choice((1, 2, 3), ordered=True, axis="strength"))
        space = {
            "caa": caa_template(),
            "prompt": prompt_template(),
            "text": Template((entry(SystemPrompt, text=Text(max_chars=200)),)),
            "shared": shared,
            "searched": caa_template(Interval(0.0, 2.0), layer_id=Choice((1, 2), ordered=True)),
        }
        study = _with_space(store, space, searched={"caa.layer_id"})
        for structure in space:
            plan = make_plan(study=study, candidate=study.default_candidate(structure=structure))
            assert plan.recipes
        assert study.default_candidate(structure="text").values == {"system_prompt.text": ""}
        assert study.default_candidate(structure="prompt").values == {"prompt.text": "be brief"}
        assert study.default_candidate(structure="caa").values == {}
        assert study.solved_axis("shared").key == "@strength"

    def test_candidates_that_bind_to_one_template_share_an_id(self, store):
        shared = caa_template(Interval(0.0, 2.0, axis="strength"),
                              layer_id=Choice((1, 2, 3), ordered=True, axis="strength"))
        study = _with_space(store, {"shared": shared}, searched={"@strength"})
        by_key = Candidate("shared", {"@strength": 1})
        by_members = Candidate("shared", {"caa.multiplier": 1.0, "caa.layer_id": 2})
        assert study.candidate_id(by_key) == study.candidate_id(by_members)
        floats = _with_space(store, {"caa": caa_template()}, searched={"caa.multiplier"})
        assert floats.candidate_id(Candidate("caa", {"caa.multiplier": 1})) == floats.candidate_id(
            Candidate("caa", {"caa.multiplier": 1.0}))

    def test_bind_rejects_missing_unknown_and_outside_values(self, store):
        study = _with_space(store, {"prompt": prompt_template(), "caa": caa_template()})
        with pytest.raises(RecipeError, match="does not bind"):
            study.bind(Candidate("prompt", {}))
        with pytest.raises(RecipeError, match="no proposer axis"):
            study.bind(Candidate("caa", {"caa.multiplier": 1.0}))
        with pytest.raises(RecipeError, match="outside"):
            study.bind(Candidate("prompt", {"prompt.text": "be loud"}))
        with pytest.raises(RecipeError, match="no structure"):
            study.bind(Candidate("missing", {}))
        assert study.bind(Candidate("prompt", {"prompt.text": "be kind"})).axes() == {}


class TestIdentity:
    def test_the_id_covers_the_settings_and_is_stable(self, store):
        study = build_study(store, _quiet)
        assert study.id == build_study(store, _quiet).id
        assert dataclasses.replace(study, settings=Settings(ladder_points=5)).id != study.id
        assert dataclasses.replace(study, seed=3).id != study.id
