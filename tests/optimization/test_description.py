"""Tests for the description of a search for a language model (`search/description.py`)."""
import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.recipe import Choice
from steerability.algorithms.core.store import Store
from steerability.optimization.search.description import (
    candidate_document,
    describe_axes,
    describe_structure,
    describe_task,
    parse_document,
)
from steerability.optimization.study import Candidate
from tests.optimization.helpers import Truth, build_study, caa_template, prompt_template

GOAL = "The behavior score measures how formal the reply reads."


def _quiet(recipe, item_id):
    return Truth()


@pytest.fixture()
def study(tmp_path):
    space = {"prompt": prompt_template(), "caa": caa_template(layer_id=Choice(values=(1, 2)))}
    return build_study(store=Store.open(root=tmp_path / "store"), truth=_quiet, space=space)


def test_a_candidate_document_parses_back_to_its_candidate():
    candidate = Candidate(structure="prompt", values={"prompt.text": "be kind"})
    text = candidate_document(candidate=candidate)
    assert text == '{"structure": "prompt", "values": {"prompt.text": "be kind"}}'
    assert parse_document(text=f"  {text}\n") == candidate


@pytest.mark.parametrize(
    "text, message",
    [
        ("structure = caa", "JSON object"),
        ('{"structure": "caa", "values": {}} {"structure": "caa", "values": {}}', "more than one"),
        ("[1, 2]", "not an object"),
        ('{"structure": "caa", "values": {}, "notes": "x"}', "exactly the keys"),
        ('{"structure": 1, "values": {}}', "must be a string"),
    ],
)
def test_parse_document_rejects_malformed_documents(text, message):
    with pytest.raises(ValueError, match=message):
        parse_document(text=text)


def test_the_task_ends_with_the_goal_and_an_empty_goal_raises():
    task = describe_task(goal=f"  {GOAL}\n")
    assert task.endswith(GOAL) and "within a fixed margin" in task
    for goal in ("", "   ", None):
        with pytest.raises(ValueError, match="goal"):
            describe_task(goal=goal)


def test_a_structure_is_described_by_its_controls_and_its_solved_setting(study):
    assert describe_structure(study=study, structure="prompt") == (
        "The structure `prompt` contains the controls `prompt` (system_prompt)."
    )
    line = describe_structure(study=study, structure="caa")
    assert line.startswith("The structure `caa` contains the controls `caa`.")
    assert "The evaluator solves `caa.multiplier`" in line


def test_the_axes_give_their_ranges_and_help_texts(study):
    prompt = describe_axes(study=study, structure="prompt")
    assert prompt.startswith("- `prompt.text`: one of") and "System-prompt text set or merged" in prompt
    caa = describe_axes(study=study, structure="caa")
    assert caa.startswith("- `caa.layer_id`: one of 1, 2.") and "caa.multiplier" not in caa


def test_the_help_text_falls_back_to_the_args_docstring(tmp_path):
    searched = build_study(
        store=Store.open(root=tmp_path / "store"),
        truth=_quiet,
        space={"caa": caa_template()},
        searched={"caa.multiplier"},
    )
    axes = describe_axes(study=searched, structure="caa")
    assert "a number from 0 to 3" in axes and "Scaling factor for the steering vector" in axes


def test_the_descriptions_contain_no_item_text(study):
    texts = [describe_task(goal=GOAL)]
    for structure in study.space:
        texts.append(describe_structure(study=study, structure=structure))
        texts.append(describe_axes(study=study, structure=structure))
    for benchmark in study.benchmarks:
        assert not any(item_id in text for item_id in benchmark.item_ids() for text in texts)
