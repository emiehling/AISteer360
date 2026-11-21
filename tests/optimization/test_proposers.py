"""Tests for the built-in proposers (`search/proposers/`)."""
import pytest

pytest.importorskip("inspect_ai")

from steerability.algorithms.core.store import Store
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
