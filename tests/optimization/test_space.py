"""Tests for structure expansion and the support check of a space (`optimization/space.py`)."""
import dataclasses

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Choice, Interval, Point, RecipeError, Template, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import preflight
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.input_control.user_prefix.control import UserPrefix
from steerability.algorithms.state_control.act_add.control import ActAdd
from steerability.algorithms.state_control.caa.control import CAA
from steerability.optimization import Slot, check_space, one_of, optional, structures
from tests.optimization.helpers import PAIRS, RAW, Truth, build_study, protocol

PROMPT = entry(SystemPrompt, name="prompt", text=Choice(values=("be brief", "be formal")))
PREFIX = entry(UserPrefix, text=Choice(values=("please", "kindly")))
CAA_ENTRY = entry(
    CAA,
    data=PAIRS,
    train_spec=RAW,
    layer_id=Choice(values=(1, 2)),
    multiplier=Interval(low=0.0, high=3.0),
)
ACT_ADD = entry(ActAdd, positive_prompt="the cat", negative_prompt="the dog", layer_id=1)


def test_each_expanded_template_has_the_id_of_the_template_written_by_hand():
    space = structures(optional(PROMPT), optional(CAA_ENTRY))
    assert space["prompt"].id == Template(entries=(PROMPT,)).id
    assert space["caa"].id == Template(entries=(CAA_ENTRY,)).id
    assert space["prompt+caa"].id == Template(entries=(CAA_ENTRY, PROMPT)).id


@pytest.mark.parametrize(
    "slots, names",
    [
        ((PROMPT,), ["prompt"]),
        ((optional(PROMPT),), ["prompt"]),
        ((optional(PROMPT), optional(CAA_ENTRY)), ["prompt", "caa", "prompt+caa"]),
        ((optional(CAA_ENTRY), optional(PROMPT)), ["caa", "prompt", "prompt+caa"]),
        ((PROMPT, optional(CAA_ENTRY)), ["prompt", "prompt+caa"]),
        ((one_of(CAA_ENTRY, ACT_ADD),), ["caa", "act_add"]),
        ((optional(PROMPT), one_of(CAA_ENTRY, ACT_ADD)), ["caa", "act_add", "prompt+caa", "prompt+act_add"]),
        (
            (optional(PROMPT), optional(one_of(CAA_ENTRY, ACT_ADD))),
            ["prompt", "caa", "act_add", "prompt+caa", "prompt+act_add"],
        ),
        (
            (optional(PROMPT), optional(PREFIX), optional(CAA_ENTRY)),
            [
                "prompt",
                "user_prefix",
                "caa",
                "prompt+user_prefix",
                "prompt+caa",
                "user_prefix+caa",
                "prompt+user_prefix+caa",
            ],
        ),
    ],
)
def test_structures_are_ordered_by_size_with_each_slot_in_the_order_given(slots, names):
    assert list(structures(*slots)) == names


def test_entries_of_one_method_need_explicit_names():
    other = entry(CAA, data=PAIRS, train_spec=RAW, layer_id=2)
    with pytest.raises(RecipeError, match="explicit name="):
        structures(optional(CAA_ENTRY), optional(other))
    first = entry(CAA, name="first", data=PAIRS, train_spec=RAW, layer_id=1)
    second = entry(CAA, name="second", data=PAIRS, train_spec=RAW, layer_id=2)
    assert list(structures(optional(first), optional(second))) == ["first", "second", "first+second"]


def test_entries_and_combinations_that_resolve_to_one_name_are_refused():
    with pytest.raises(RecipeError, match="same name"):
        structures(optional(PROMPT), optional(entry(UserPrefix, name="prompt", text="x")))
    with pytest.raises(RecipeError, match="same name"):
        structures(optional(PROMPT), optional(PROMPT))
    joined = entry(SystemPrompt, name="a+b", text="x")
    first = entry(UserPrefix, name="a", text="y")
    second = entry(CAA, name="b", data=PAIRS, train_spec=RAW, layer_id=1)
    with pytest.raises(RecipeError, match="structure name 'a\\+b'"):
        structures(optional(joined), optional(first), optional(second))


def test_no_slot_and_a_slot_without_alternatives_are_refused():
    with pytest.raises(RecipeError, match="at least one slot"):
        structures()
    with pytest.raises(RecipeError, match="at least one alternative"):
        structures(Slot(alternatives=()))
    with pytest.raises(RecipeError, match="at least one alternative"):
        one_of()
    with pytest.raises(RecipeError, match="disabled"):
        optional(entry(SystemPrompt, text="x", enabled=False))


def test_the_operands_are_positional_entries_and_slots():
    with pytest.raises(TypeError):
        structures(slots=[PROMPT])
    with pytest.raises(TypeError):
        optional(item=PROMPT)
    with pytest.raises(TypeError, match="Entry and Slot"):
        structures("prompt")
    with pytest.raises(TypeError, match="Entry"):
        one_of(None)


def test_a_study_on_an_expanded_space_validates_and_its_axes_keep_their_keys(tmp_path):
    store = Store.open(root=tmp_path / "store")
    space = structures(optional(PROMPT), optional(CAA_ENTRY))
    study = build_study(store=store, truth=lambda recipe, item_id: Truth(), space=space)
    study.validate(store=store)
    keys = {structure: set(study.proposer_axes(structure=structure)) for structure in space}
    assert keys["prompt+caa"] == keys["prompt"] | keys["caa"] == {"prompt.text", "caa.layer_id"}
    assert study.solved_axis(structure="prompt+caa").key == "caa.multiplier"


ENGINE = BackendSpec(kind="vllm", model="synthetic-model")
DEFAULTS = {"prompt.text": "be brief", "caa.layer_id": 1, "caa.multiplier": 1.5}


def _engine_protocol():
    return Protocol.create(model="synthetic-model", backend=ENGINE, generate_defaults={"temperature": 0})


def test_check_space_gives_the_structures_a_backend_cannot_run_with_the_messages_of_preflight():
    space = structures(optional(PROMPT), optional(CAA_ENTRY))
    assert check_space(space=space, protocol=protocol()) == {}
    engine = _engine_protocol()
    problems = check_space(space=space, protocol=engine)
    assert list(problems) == ["caa", "prompt+caa"]
    for name, messages in problems.items():
        template = space[name]
        recipe = template.bind({path: value for path, value in DEFAULTS.items() if path in template.axes()}).recipe()
        assert messages == tuple(preflight(points=[(name, Point(recipe=recipe, assignment={}))], protocol=engine))
        assert all(" CAA (" in message for message in messages)
        assert any("CAA is unsupported at generate on backend kind 'vllm'" in message for message in messages)


def test_a_study_refuses_a_structure_its_backend_cannot_run(tmp_path):
    store = Store.open(root=tmp_path / "store")
    space = structures(optional(PROMPT), optional(CAA_ENTRY))
    study = build_study(store=store, truth=lambda recipe, item_id: Truth(), space=space)
    engine = dataclasses.replace(study, protocol=_engine_protocol())
    with pytest.raises(ValueError, match=r"cannot run structures \['caa', 'prompt\+caa'\]") as raised:
        engine.validate(store=store)
    assert "CAA is unsupported at generate on backend kind 'vllm'" in str(raised.value)
    problems = check_space(space=space, protocol=engine.protocol)
    supported = {name: template for name, template in space.items() if name not in problems}
    dataclasses.replace(engine, space=supported).validate(store=store)
