"""Tests for the built-in proposers (`search/proposers/`)."""
import logging

import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.recipe import Template, Text, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.optimization.evaluation.archive import ArchiveView
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.search.proposers import GridProposer, RandomProposer
from steerability.optimization.study import Candidate
from tests.optimization.helpers import (
    StubFactory,
    Truth,
    argument,
    build_study,
    caa_template,
    multiplier,
    prompt_template,
)

TEXTS = tuple(f"text {index}" for index in range(6))


def _truth(recipe, item_id):
    value = multiplier(recipe)
    if value is not None:
        return Truth(loss=0.0055 * value * value, gain=0.05 * value)
    text = argument(recipe, "input_control/system_prompt", "text")
    return Truth(loss=0.01 * TEXTS.index(text), gain=0.02 + 0.02 * TEXTS.index(text))


def _session(store, **kwargs):
    study = build_study(store, _truth, **kwargs)
    session = StudySession(study, store=store, factory=StubFactory(study.protocol))
    session.start()
    return session


@pytest.fixture()
def space():
    return {"caa": caa_template(), "prompt": prompt_template(TEXTS)}


def test_grid_and_random_proposers(tmp_path, space):
    session = _session(Store.open(tmp_path / "store"), space=space)
    grid = GridProposer(session.study)
    first = grid.ask(session.archive.view(), 3)
    assert first == [Candidate("caa", {}), Candidate("prompt", {"prompt.text": "text 0"}),
                     Candidate("prompt", {"prompt.text": "text 1"})]
    assert len(grid.ask(session.archive.view(), 10)) == 4
    assert grid.ask(session.archive.view(), 10) == []
    draws = RandomProposer(session.study, seed=3).ask(session.archive.view(), 5)
    assert draws == RandomProposer(session.study, seed=3).ask(session.archive.view(), 5)
    assert len({session.study.candidate_id(candidate) for candidate in draws}) == len(draws)
    session.evaluate(draws[0])
    assert session.study.candidate_id(draws[0]) not in {
        session.study.candidate_id(candidate) for candidate in RandomProposer(session.study, seed=3).ask(
            session.archive.view(), 5)
    }


def test_the_grid_requires_discrete_proposer_axes(tmp_path):
    session = _session(Store.open(tmp_path / "store"), space={"caa": caa_template()}, searched={"caa.multiplier"})
    with pytest.raises(ValueError, match="continuous"):
        GridProposer(session.study)


def test_the_random_proposer_warns_about_text_axes(tmp_path, space, caplog):
    text_space = {"caa": caa_template(), "text": Template((entry(SystemPrompt, text=Text(max_chars=200)),))}
    with_text = build_study(Store.open(tmp_path / "text"), _truth, space=text_space)
    with caplog.at_level(logging.WARNING, logger="steerability.optimization.search.proposers.random"):
        RandomProposer(with_text, seed=0)
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "'text'" in warnings[0] and "system_prompt.text" in warnings[0]
    caplog.clear()
    without_text = build_study(Store.open(tmp_path / "plain"), _truth, space=space)
    with caplog.at_level(logging.WARNING, logger="steerability.optimization.search.proposers.random"):
        RandomProposer(without_text, seed=0)
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_structures_restrict_the_grid_and_the_random_proposer(tmp_path, space):
    text_space = {**space, "text": Template(entries=(entry(SystemPrompt, name="free", text=Text(max_chars=200)),))}
    study = build_study(store=Store.open(root=tmp_path / "store"), truth=_truth, space=text_space)
    empty = ArchiveView(entries=())
    with pytest.raises(ValueError, match="Text range"):
        GridProposer(study=study)
    grid = GridProposer(study=study, structures=["prompt", "caa"])
    assert [candidate.structure for candidate in grid.ask(view=empty, n=100)] == ["caa"] + ["prompt"] * len(TEXTS)
    draws = RandomProposer(study=study, seed=0, structures=["prompt"]).ask(view=empty, n=4)
    assert len(draws) == 4 and {candidate.structure for candidate in draws} == {"prompt"}
    with pytest.raises(ValueError, match="not in the study"):
        GridProposer(study=study, structures=["missing"])
    with pytest.raises(ValueError, match="empty"):
        RandomProposer(study=study, seed=0, structures=[])
