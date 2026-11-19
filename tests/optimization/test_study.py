"""Tests for the study definition: axes, candidates, validation, and feasibility (`optimization/study.py`)."""
import dataclasses

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.identity import versioned
from steerability.algorithms.core.recipe import Choice, Entry, Interval, RecipeError, Template, Text, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.output_control.best_of_n.control import BestOfN
from steerability.optimization.evaluation.plan import make_plan
from steerability.optimization.measurement.pools import split_items
from steerability.optimization.study import Candidate, Constraint, Settings, Study, StudyInfeasibleError
from tests.optimization.helpers import (
    OTHER,
    PAIRS,
    SyntheticBenchmark,
    Truth,
    build_study,
    caa_template,
    prompt_template,
    protocol,
)


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

    def test_a_structural_entry_is_refused(self, store):
        lora = Template((Entry(method="structural_control/load_lora", args={"path": "adapter"}),))
        with pytest.raises(ValueError, match="structural"):
            _with_space(store, {"lora": lora}).validate(store, allow_infeasible=True)

    def test_an_unversioned_callable_is_refused(self, store):
        unversioned = Template((entry(BestOfN, scorer=unversioned_scorer, n=Choice((2, 4))),))
        with pytest.raises(ValueError, match="version"):
            _with_space(store, {"best": unversioned}).validate(store, allow_infeasible=True)
        marked = Template((entry(BestOfN, scorer=versioned_scorer, n=Choice((2, 4))),))
        _with_space(store, {"best": marked}).validate(store, allow_infeasible=True)

    def test_two_ordered_axes_left_free_are_refused(self, store):
        template = caa_template(layer_id=Choice((1, 2), ordered=True))
        with pytest.raises(ValueError, match="ordered axes"):
            _with_space(store, {"caa": template}).validate(store, allow_infeasible=True)
        _with_space(store, {"caa": template}, searched={"caa.layer_id"}).validate(store, allow_infeasible=True)

    def test_a_solved_axis_that_changes_a_fit_is_refused_until_it_is_searched(self, store):
        template = caa_template(1.0, data=Choice((PAIRS, OTHER), ordered=True))
        with pytest.raises(RecipeError, match="study.searched"):
            _with_space(store, {"caa": template}).validate(store, allow_infeasible=True)
        _with_space(store, {"caa": template}, searched={"caa.data"}).validate(store, allow_infeasible=True)

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
        pools = {**study.pools, twin.id: split_items(twin, sealed=2000, seed=0)}
        pools.pop(study.constraints[0].benchmark.id)
        named = dataclasses.replace(study, constraints=(Constraint(twin, 0.05),), pools=pools)
        with pytest.raises(ValueError, match="distinct"):
            named.validate(store)

    def test_an_infeasible_study_raises_unless_allowed(self, store):
        study = build_study(store, _quiet, capability_sealed=200)
        with pytest.raises(StudyInfeasibleError, match="guard band"):
            study.validate(store)
        study.validate(store, allow_infeasible=True)


class TestCandidates:
    def test_seed_candidates_bind_for_every_structure(self, store):
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
            plan = make_plan(study, study.seed_candidate(structure))
            assert plan.recipes
        assert study.seed_candidate("text").values == {"system_prompt.text": ""}
        assert study.seed_candidate("prompt").values == {"prompt.text": "be brief"}
        assert study.seed_candidate("caa").values == {}
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


class TestIdentityAndFeasibility:
    def test_the_id_covers_the_settings_and_is_stable(self, store):
        study = build_study(store, _quiet)
        assert study.id == build_study(store, _quiet).id
        assert dataclasses.replace(study, settings=Settings(ladder_items=128)).id != study.id
        assert dataclasses.replace(study, seed=3).id != study.id

    def test_the_gsm8k_example(self, store):
        capability = SyntheticBenchmark("gsm8k", 8792, kind="capability", truth=_quiet, base=0.8)
        behavior = SyntheticBenchmark("formality", 1000, kind="behavior", truth=_quiet, base=0.3)
        for benchmark in (capability, behavior):
            benchmark.prepare(store)
        pools = {capability.id: split_items(capability, sealed=5000, seed=0),
                 behavior.id: split_items(behavior, sealed=200, seed=0)}
        assert len(pools[capability.id].search) == 3792

        def study(margin):
            return Study(name="gsm8k", protocol=protocol(), space={"caa": caa_template()}, objective=behavior,
                         constraints=(Constraint(capability, margin),), pools=pools)

        assert study(0.03).feasibility() == []
        problems = study(0.02).feasibility()
        assert len(problems) == 2
        assert "2,656" in problems[0] and "1.40 points" in problems[1]
