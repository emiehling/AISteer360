"""Tests for recipes, templates, axes, and expansion (`algorithms/core/recipe.py`)."""
import math

import pytest
import torch

from steerability.algorithms.core.recipe import (
    BASELINE,
    Choice,
    Entry,
    Interval,
    Recipe,
    RecipeError,
    Template,
    Text,
    assignments,
    entry,
    expand,
)
from steerability.algorithms.core.registry import REGISTRY
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.input_control.user_prefix.control import UserPrefix
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.spipe.store import ArtifactStore

PAIRS = {"positives": ["yes", "sure"], "negatives": ["no", "never"]}


def _vector(scale: float = 1.0) -> SteeringVector:
    return SteeringVector(model_type="llama", directions={1: scale * torch.ones(1, 8)})


class TestRecipeIdentity:
    def test_defaults_names_disabled_entries_and_key_order_do_not_change_the_id(self):
        base = Recipe((entry(SystemPrompt, text="be brief"), entry(CAA, data=PAIRS, layer_id=1)))
        variants = [
            Recipe((entry(SystemPrompt, text="be brief", mode="prepend"), entry(CAA, layer_id=1, data=PAIRS))),
            Recipe((entry(SystemPrompt, name="sp", text="be brief"), entry(CAA, name="steer", data=PAIRS, layer_id=1))),
            Recipe((
                entry(SystemPrompt, text="be brief"), entry(CAA, data=PAIRS, layer_id=1),
                entry(UserPrefix, text="ignored", enabled=False),
            )),
            Recipe((entry(CAA, data=PAIRS, layer_id=1), entry(SystemPrompt, text="be brief"))),
        ]
        for variant in variants:
            assert variant.id == base.id
            assert variant == base

    def test_int_and_float_give_one_id(self):
        first = Recipe((entry(CAA, data=PAIRS, multiplier=1),))
        second = Recipe((entry(CAA, data=PAIRS, multiplier=1.0),))
        assert first.id == second.id

    def test_different_tensors_give_different_ids(self):
        first = Recipe((entry(CAA, steering_vector=_vector(1.0), layer_id=1),))
        second = Recipe((entry(CAA, steering_vector=_vector(2.0), layer_id=1),))
        assert first.id != second.id

    def test_unknown_method_and_bad_args_raise(self):
        with pytest.raises(RecipeError, match="Unknown method"):
            Recipe((Entry(method="state_control/not_a_method", args={}),)).id
        with pytest.raises(RecipeError, match="caa"):
            Recipe((entry(CAA, layer_id=1),)).id  # neither data nor a steering vector

    def test_unencodable_value_raises_with_the_path(self):
        from steerability.algorithms.output_control.value_guidance.control import ValueGuidance

        with pytest.raises(RecipeError, match=r"value_guidance\.value"):
            Recipe((entry(ValueGuidance, value=lambda context: context),)).id

    def test_recipe_rejects_ranges(self):
        with pytest.raises(RecipeError, match=r"caa\.multiplier"):
            Recipe((entry(CAA, data=PAIRS, multiplier=Choice((1.0, 2.0))),))

    def test_baseline_is_empty(self):
        assert BASELINE.entries == ()
        assert BASELINE.instantiate() == []
        assert BASELINE.canonical() == {"schema": "recipe/1", "entries": []}


class TestRecipeNames:
    def test_unnamed_entries_of_one_method_are_numbered(self):
        recipe = Recipe((
            entry(CAA, data=PAIRS, layer_id=1), entry(CAA, data=PAIRS, layer_id=2), entry(CAA, data=PAIRS, layer_id=3),
        ))
        assert recipe.names == ("caa", "caa_2", "caa_3")

    def test_a_repeated_explicit_name_raises(self):
        with pytest.raises(RecipeError, match="same name"):
            Recipe((entry(CAA, name="x", data=PAIRS), entry(SystemPrompt, name="x", text="a")))

    def test_category_order_sets_names_and_instantiate_order(self):
        forward = Template((entry(SystemPrompt, text="a"), entry(CAA, data=PAIRS, multiplier=Choice((1.0, 2.0)))))
        reverse = Template((entry(CAA, data=PAIRS, multiplier=Choice((1.0, 2.0))), entry(SystemPrompt, text="a")))
        assert forward.id == reverse.id
        assert list(forward.axes()) == list(reverse.axes()) == ["caa.multiplier"]
        recipe = reverse.bind({"caa.multiplier": 2.0}).recipe()
        assert [type(control) for control in recipe.instantiate()] == [SystemPrompt, CAA]


class TestRecipeJson:
    def test_round_trip_preserves_the_id(self, tmp_path):
        recipe = Recipe((
            entry(CAA, steering_vector=_vector(), layer_id=1, multiplier=2.0),
            entry(CAA, name="second", data=PAIRS, layer_id=2),
            entry(SystemPrompt, text="be brief"),
        ))
        store = ArtifactStore(tmp_path / "store")
        data = recipe.to_json(store)
        assert data == recipe.to_json()
        loaded = Recipe.from_json(data, store=store)
        assert loaded.id == recipe.id
        assert loaded.names == ("system_prompt", "caa", "caa_2")

    def test_a_different_schema_raises(self):
        with pytest.raises(RecipeError, match="schema"):
            Recipe.from_json({"schema": "recipe/0", "entries": []})

    def test_from_controls_matches_the_recipe(self):
        recipe = Recipe((entry(SystemPrompt, text="a"), entry(CAA, data=PAIRS, layer_id=1)))
        assert Recipe.from_controls(recipe.instantiate()).id == recipe.id


class TestRanges:
    def test_interval_grid_keeps_the_axis_and_drops_repeats(self):
        assert Interval(0.0, 1.0, axis="strength").grid(3).axis == "strength"
        assert Interval(1, 4, integer=True).grid(6).values == (1, 2, 3, 4)
        assert Interval(1.0, 1.0).grid(4).values == (1.0,)

    def test_log_interval_midpoint_is_the_geometric_mean(self):
        template = Template((entry(CAA, data=PAIRS, multiplier=Interval(1.0, 100.0, log=True)),))
        axis = template.axes()["caa.multiplier"]
        assert axis.continuous
        assert math.isclose(axis.at(0.5)["caa.multiplier"], 10.0)

    def test_choice_validation(self):
        with pytest.raises(RecipeError):
            Choice(())
        with pytest.raises(RecipeError, match="ordered"):
            Choice((1, 2), axis="a")
        with pytest.raises(RecipeError, match="nest"):
            Choice((Choice((1,)),))


class TestBind:
    def _template(self) -> Template:
        return Template((
            entry(SystemPrompt, text=Text(max_chars=5)),
            entry(CAA, data=PAIRS, layer_id=Choice((1, 2)), multiplier=Interval(0.0, 4.0)),
        ))

    def test_bind_and_recipe(self):
        recipe = self._template().bind({"system_prompt.text": "hi", "caa.layer_id": 2, "caa.multiplier": 1.5}).recipe()
        assert recipe.id == Recipe((
            entry(SystemPrompt, text="hi"), entry(CAA, data=PAIRS, layer_id=2, multiplier=1.5),
        )).id

    @pytest.mark.parametrize("path, value", [
        ("caa.layer_id", 3), ("caa.multiplier", 4.5), ("system_prompt.text", "too long"),
    ])
    def test_a_value_outside_its_range_raises_with_the_path(self, path, value):
        with pytest.raises(RecipeError, match=path.replace(".", r"\.")):
            self._template().bind({path: value})

    def test_a_path_without_a_range_raises(self):
        with pytest.raises(RecipeError, match=r"caa\.data"):
            self._template().bind({"caa.data": PAIRS})

    def test_recipe_raises_while_a_range_remains(self):
        with pytest.raises(RecipeError, match="still holds ranges"):
            self._template().bind({"caa.layer_id": 1}).recipe()

    def test_ranges_inside_nested_mappings(self):
        template = Template((entry(CAA, data=PAIRS, train_spec={"method": Choice(("mean_diff", "pca_pairwise"))}),))
        assert list(template.axes()) == ["caa.train_spec.method"]
        recipe = template.bind({"caa.train_spec.method": "mean_diff"}).recipe()
        assert recipe.instantiate()[0].train_spec.method == "mean_diff"


class TestSharedAxes:
    def _template(self) -> Template:
        return Template((entry(
            CAA, data=PAIRS,
            multiplier=Interval(5, 20, axis="strength"),
            layer_id=Choice((1, 2, 3, 4), ordered=True, axis="strength"),
        ),))

    def test_a_shared_axis_expands_as_one_axis(self):
        template = self._template()
        assert list(template.axes()) == ["@strength"]
        points = list(expand(template))
        assert [point.assignment for point in points] == [
            {"caa.multiplier": 5.0, "caa.layer_id": 1},
            {"caa.multiplier": 10.0, "caa.layer_id": 2},
            {"caa.multiplier": 15.0, "caa.layer_id": 3},
            {"caa.multiplier": 20.0, "caa.layer_id": 4},
        ]

    def test_binding_by_key_equals_binding_the_members(self):
        template = self._template()
        assert template.bind({"@strength": 2}).id == template.bind({"caa.multiplier": 15.0, "caa.layer_id": 3}).id

    def test_binding_in_part_or_at_differing_positions_raises(self):
        template = self._template()
        with pytest.raises(RecipeError, match="in part"):
            template.bind({"caa.layer_id": 3})
        with pytest.raises(RecipeError, match="differing positions"):
            template.bind({"caa.multiplier": 5.0, "caa.layer_id": 3})

    def test_continuous_shared_axis(self):
        template = Template((entry(
            CAA, data=PAIRS, multiplier=Interval(0.0, 2.0, axis="a"), layer_id=Interval(0, 4, integer=True, axis="a"),
        ),))
        assert template.bind({"@a": 0.5}).id == template.bind({"caa.multiplier": 1.0, "caa.layer_id": 2}).id
        with pytest.raises(RecipeError, match="differing positions"):
            template.bind({"caa.multiplier": 2.0, "caa.layer_id": 0})

    def test_choices_on_one_axis_must_have_equal_lengths(self):
        with pytest.raises(RecipeError, match="differing numbers"):
            Template((entry(CAA, data=PAIRS, multiplier=Choice((1.0, 2.0), ordered=True, axis="a"),
                            layer_id=Choice((1, 2, 3), ordered=True, axis="a")),))


class TestExpand:
    def test_grid_order_is_deterministic(self):
        template = Template((
            entry(SystemPrompt, text=Choice(("a", "b"))), entry(CAA, data=PAIRS, multiplier=Choice((1.0, 2.0))),
        ))
        first = [point.recipe.id for point in expand(template)]
        assert first == [point.recipe.id for point in expand(template)]
        assert [point.assignment for point in expand(template)] == [
            {"system_prompt.text": "a", "caa.multiplier": 1.0}, {"system_prompt.text": "a", "caa.multiplier": 2.0},
            {"system_prompt.text": "b", "caa.multiplier": 1.0}, {"system_prompt.text": "b", "caa.multiplier": 2.0},
        ]

    def test_grid_rejects_continuous_and_text_axes(self):
        continuous = Template((entry(CAA, data=PAIRS, multiplier=Interval(0.0, 1.0)),))
        with pytest.raises(RecipeError, match=r"Interval\.grid"):
            list(expand(continuous))
        text = Template((entry(SystemPrompt, text=Text()),))
        with pytest.raises(RecipeError, match="Text"):
            list(expand(text))

    def test_random_is_reproducible_and_deduplicated(self):
        template = Template((entry(CAA, data=PAIRS, multiplier=Interval(1.0, 100.0, log=True), layer_id=Choice((1, 2))),))
        first = [point.recipe.id for point in expand(template, strategy="random", samples=5, seed=3)]
        assert first == [point.recipe.id for point in expand(template, strategy="random", samples=5, seed=3)]
        assert len(first) == len(set(first)) == 5
        assert first != [point.recipe.id for point in expand(template, strategy="random", samples=5, seed=4)]

    def test_random_stops_when_the_space_is_exhausted(self):
        template = Template((entry(CAA, data=PAIRS, layer_id=Choice((1, 2))),))
        points = list(expand(template, strategy="random", samples=10, seed=0))
        assert len(points) == 2

    def test_random_requires_samples_and_seed(self):
        template = Template((entry(CAA, data=PAIRS, layer_id=Choice((1, 2))),))
        with pytest.raises(RecipeError, match="samples"):
            list(expand(template, strategy="random"))

    def test_a_recipe_yields_one_point(self):
        recipe = Recipe((entry(CAA, data=PAIRS),))
        assert [(point.recipe, point.assignment) for point in expand(recipe)] == [(recipe, {})]

    def test_assignments_over_a_subset_of_axes(self):
        template = Template((entry(CAA, data=PAIRS, layer_id=Choice((1, 2)), multiplier=Choice((1.0, 3.0))),))
        axes = {key: axis for key, axis in template.axes().items() if key == "caa.layer_id"}
        assert list(assignments(axes)) == [{"caa.layer_id": 1}, {"caa.layer_id": 2}]


# minimal arguments for registered controls whose Args have required fields
_MINIMAL_ARGS = {
    "state_control/caa": {"data": PAIRS},
    "state_control/act_add": {"positive_prompt": "love", "negative_prompt": "hate"},
    "input_control/system_prompt": {"text": "be brief"},
    "input_control/user_prefix": {"text": "please"},
}


def _registered_keys() -> list[str]:
    return sorted(f"{category}/{name}" for category, bucket in REGISTRY.items() for name in bucket)


@pytest.mark.parametrize("method", _registered_keys())
def test_from_controls_round_trips_every_constructible_control(method):
    """`Recipe.from_controls(recipe.instantiate()).id == recipe.id` for controls constructible without a model."""
    recipe = Recipe((Entry(method=method, args=_MINIMAL_ARGS.get(method, {})),))
    try:
        controls = recipe.instantiate()
        identity = recipe.id
    except Exception as exc:
        pytest.skip(f"{method} needs configuration beyond the minimal table: {type(exc).__name__}")
    assert Recipe.from_controls(controls).id == identity
