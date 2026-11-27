"""Tests for weight changes in a study: the limits on structural fits and the weights of a certificate.

In the synthetic world a structural fit is a weights context other than `"base"` that the stub factory has not
prepared or steered yet. The `LoadLoRA` entries below give these contexts, one structural fit per adapter path.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from datasets import Dataset

from steerability.algorithms.core.execution.payloads import LoRAArtifact
from steerability.algorithms.core.fits import weights_context
from steerability.algorithms.core.recipe import Choice, Interval, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT
from steerability.optimization import Candidate, optional, structures
from steerability.optimization.evaluation.archive import ArchiveEntry, ArchiveView
from steerability.optimization.evaluation.service import EvaluationService
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.release.certify import certificate_weights, certify
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.openevolve import parse, program_text, system_message
from steerability.optimization.search.population import is_trained, trained_values
from steerability.optimization.search.proposers import GridProposer, RandomProposer
from steerability.spipe.codec import EncodeContext, encode
from tests.optimization.helpers import SMALL, FixedProposer, StubFactory, Truth, argument, build_study

ADAPTERS = ("adapter-a", "adapter-b", "adapter-c")
BASE = "synthetic-model"
DATASET = Dataset.from_dict({"text": ["a formal reply", "another formal reply"]})
TEXTS = ("be formal", "be brief", "be kind")


def _gain(recipe, item_id):
    return Truth(gain=0.1)


def _space(paths=ADAPTERS):
    lora = entry(LoadLoRA, name="lora", path=Choice(values=tuple(paths)), base_model=BASE)
    prompt = entry(SystemPrompt, name="prompt", text=Choice(values=("be formal", "be brief")))
    return structures(optional(lora), optional(prompt))


def _lora(path):
    return Candidate(structure="lora", values={"lora.path": path})


def _both(path, text):
    return Candidate(structure="lora+prompt", values={"lora.path": path, "prompt.text": text})


def _prompt(text):
    return Candidate(structure="prompt", values={"prompt.text": text})


@pytest.fixture()
def store(tmp_path):
    return Store.open(root=tmp_path / "store")


def _started(study, store, **limits):
    session = StudySession(study=study, store=store, factory=StubFactory(protocol=study.protocol), **limits)
    session.start()
    return session


def test_the_session_lets_through_its_structural_fits_and_stops_a_candidate_that_needs_another(store):
    study = build_study(store=store, truth=_gain, space=_space())
    assert list(study.space) == ["lora", "prompt", "lora+prompt"]
    session = _started(study, store, max_structural_fits=2)
    results = [session.evaluate(candidate=_lora(path)) for path in ADAPTERS]
    assert [result.status for result in results[:2]] == ["accepted", "accepted"]
    assert (results[2].status, results[2].reason, results[2].stage) == ("rejected", "budget", 1)
    assert session.archive.latest(candidate_id=study.candidate_id(candidate=_lora("adapter-c"))) is None
    assert session.structural_fits == 2 and not session.exhausted()
    assert session.evaluate(candidate=_both("adapter-a", "be formal")).status == "accepted"
    assert session.evaluate(candidate=_prompt("be brief")).status == "accepted"
    assert session.evaluate(candidate=_lora("adapter-c")).reason == "budget"


def test_a_candidate_stopped_in_a_batch_is_left_out_of_its_preparation(store):
    study = build_study(store=store, truth=_gain, space=_space(), settings=SMALL)
    session = _started(study, store, max_structural_fits=1)
    session.prepare(candidates=[_lora("adapter-a"), _lora("adapter-b"), _prompt("be formal")])
    prepared = session.factory.prepared[-1]
    recipe_of = {path: study.bind(candidate=_lora(path)).recipe().id for path in ("adapter-a", "adapter-b")}
    assert recipe_of["adapter-a"] in prepared and recipe_of["adapter-b"] not in prepared
    assert session.evaluate(candidate=_lora("adapter-b")).reason == "budget"


def test_a_phase_without_structural_fits_searches_on_the_stored_products(store):
    study = build_study(store=store, truth=_gain, space=_space())
    first = FixedProposer(study=study, candidates=[_lora("adapter-a")])
    second = FixedProposer(
        study=study,
        candidates=[_lora("adapter-b"), _both("adapter-a", "be formal"), _prompt("be brief")],
    )
    archive = run_study(
        study=study,
        phases=[SearchPhase(engine=first), SearchPhase(engine=second, structural_fits=0)],
        store=store,
        factory=StubFactory(protocol=study.protocol),
    )
    recorded = archive.view().ids
    assert study.candidate_id(candidate=_lora("adapter-b")) not in recorded
    for candidate in (_lora("adapter-a"), _both("adapter-a", "be formal"), _prompt("be brief")):
        assert study.candidate_id(candidate=candidate) in recorded
    assert [result.reason for _, result in second.told] == ["budget", None, None]


def test_a_run_limit_on_structural_fits_applies_to_every_phase(store):
    study = build_study(store=store, truth=_gain, space=_space())
    proposer = FixedProposer(study=study, candidates=[_lora(path) for path in ADAPTERS])
    archive = run_study(
        study=study,
        phases=[SearchPhase(engine=proposer)],
        store=store,
        factory=StubFactory(protocol=study.protocol),
        max_structural_fits=1,
    )
    assert len(archive.view()) == 1
    assert [result.reason for _, result in proposer.told] == [None, "budget", "budget"]


def test_the_service_lists_a_candidate_that_needs_a_structural_fit_beyond_the_limits(store):
    study = build_study(store=store, truth=_gain, space=_space())
    session = _started(study, store)
    service = EvaluationService(session=session, proposer="service", parse=parse, stage_deadline_seconds=600)
    program = program_text(study=study, candidate=_lora("adapter-b"))
    with session.allowance(structural_fits=0):
        reply = service.handle(request={"op": "full", "program": program})
    assert reply["artifacts"]["reason"] == "budget" and reply["metrics"]["combined_score"] == 0.0
    assert "needs a training or merge" in reply["artifacts"]["recent_unevaluated"]


def test_a_training_past_max_fit_seconds_is_recorded_as_failed_and_not_run_again(store):
    study = build_study(store=store, truth=_gain, space=_space())
    factory = StubFactory(protocol=study.protocol)
    slow = weights_context(recipe=study.bind(candidate=_lora("adapter-b")).recipe())
    factory.slow_weights.add(slow)
    session = StudySession(study=study, store=store, factory=factory, max_fit_seconds=60.0)
    session.start()
    failed = session.evaluate(candidate=_lora("adapter-b"))
    assert (failed.status, failed.stage, failed.reason) == ("rejected", 1, "fit_failed")
    session.prepare(candidates=[_both("adapter-b", "be formal"), _lora("adapter-c")])
    recipe_of = study.bind(candidate=_lora("adapter-c")).recipe().id
    assert factory.prepared[-1] == [recipe_of]
    again = session.evaluate(candidate=_both("adapter-b", "be formal"))
    assert (again.stage, again.reason) == (1, "fit_failed") and factory.timeouts == [slow]
    assert session.evaluate(candidate=_lora("adapter-a")).status == "accepted"


def _training_space(rate):
    sft = entry(SFT, name="sft", train_dataset=DATASET, use_peft=True, learning_rate=rate)
    prompt = entry(SystemPrompt, name="prompt", text=Choice(values=TEXTS))
    return structures(optional(sft), optional(prompt))


def _entry(study, *, structure, values, stage):
    """An archive entry of a candidate decided at `stage` (accepted at stage 5, a failed fit at stage 1)."""
    return ArchiveEntry(
        candidate_id=study.candidate_id(candidate=Candidate(structure=structure, values=values)),
        structure=structure,
        values=values,
        status="accepted" if stage == 5 else "rejected",
        stage=stage,
        reason=None if stage == 5 else "fit_failed",
        fitness=0.0,
        summary={},
    )


def test_the_view_says_whether_a_new_structural_fit_may_run(store):
    study = build_study(store=store, truth=_gain, space=_space())
    session = _started(study, store, max_structural_fits=2)
    assert session.view().fits_open
    with session.allowance(structural_fits=0):
        assert not session.view().fits_open
    session.evaluate(candidate=_lora("adapter-a"))
    assert session.view().fits_open and session.archive.view().fits_open
    session.evaluate(candidate=_lora("adapter-b"))
    assert not session.view().fits_open and session.view().ids == session.archive.view().ids


def test_trained_values_lists_what_any_structure_with_the_same_training_entry_trained(store):
    study = build_study(store=store, truth=_gain, space=_training_space(Choice(values=(1e-5, 1e-4, 1e-3))))
    assert study.training_entries(structure="sft+prompt") == study.training_entries(structure="sft")
    assert study.training_entries(structure="prompt") == {}
    decided = [
        ("sft", {"sft.learning_rate": 1e-4}, 5),
        ("sft+prompt", {"sft.learning_rate": 1e-3, "prompt.text": "be kind"}, 5),
        ("sft", {"sft.learning_rate": 1e-5}, 1),
        ("sft+prompt", {"sft.learning_rate": 1e-4, "prompt.text": "be brief"}, 5),
    ]
    entries = (_entry(study, structure=structure, values=values, stage=stage) for structure, values, stage in decided)
    view = ArchiveView(entries=tuple(entries))
    trained = trained_values(study=study, view=view, structure="sft+prompt")
    assert trained == {"sft": [{"sft.learning_rate": 1e-4}, {"sft.learning_rate": 1e-3}]}
    assert trained_values(study=study, view=view, structure="prompt") == {}
    candidate = Candidate(structure="sft+prompt", values={"sft.learning_rate": 1e-3, "prompt.text": "be formal"})
    assert is_trained(study=study, candidate=candidate, trained=trained)
    untrained = Candidate(structure="sft", values={"sft.learning_rate": 1e-5})
    assert not is_trained(study=study, candidate=untrained, trained=trained)
    prompt_only = Candidate(structure="prompt", values={"prompt.text": "be kind"})
    assert is_trained(study=study, candidate=prompt_only, trained={})


class _Recording(RandomProposer):
    """A random proposer that records, for each batch it is asked for, whether fits were open and what it was told."""

    def __init__(self, study, seed):
        super().__init__(study=study, seed=seed)
        self.batches: list[tuple[bool, list]] = []

    def ask(self, view, n):
        self.batches.append((view.fits_open, []))
        return super().ask(view=view, n=n)

    def tell(self, candidate, result):
        self.batches[-1][1].append(result)


def test_a_random_search_over_a_continuous_training_argument_runs_its_trainings_then_draws_trained_values(
    store, caplog,
):
    space = _training_space(Interval(low=1e-5, high=1e-3, log=True))
    study = build_study(store=store, truth=_gain, space=space, searched={"sft.learning_rate"})
    with caplog.at_level("WARNING", logger="steerability.optimization.study"):
        study.validate(store=store)
    assert "the proposer axis 'sft.learning_rate' is continuous and changes a training" in caplog.text
    factory = StubFactory(protocol=study.protocol)
    proposer = _Recording(study=study, seed=0)
    run_study(
        study=study,
        phases=[SearchPhase(engine=proposer, structural_fits=8)],
        store=store,
        factory=factory,
        max_candidates=40,
    )
    assert len(factory.fitted_weights - {"base"}) == 8
    assert proposer.batches[0][0]
    closed = [results for fits_open, results in proposer.batches if not fits_open]
    assert closed and sum(len(results) for results in closed) >= 8
    assert all(result.reason != "budget" for results in closed for result in results)


def test_a_grid_holds_back_untrained_values_while_fits_are_closed_and_proposes_them_once_open(store):
    study = build_study(store=store, truth=_gain, space=_training_space(Choice(values=(1e-5, 1e-4))))
    grid = GridProposer(study=study, structures=["sft", "sft+prompt"])
    trained = ArchiveView(entries=(_entry(study, structure="sft", values={"sft.learning_rate": 1e-4}, stage=5),))
    closed = ArchiveView(entries=trained.entries, fits_open=False)
    proposed = grid.ask(view=closed, n=10)
    assert {candidate.values["sft.learning_rate"] for candidate in proposed} == {1e-4}
    assert [candidate.structure for candidate in proposed] == ["sft+prompt"] * 3
    reopened = grid.ask(view=trained, n=10)
    assert [(candidate.structure, candidate.values["sft.learning_rate"]) for candidate in reopened] == [
        ("sft", 1e-5),
        *[("sft+prompt", 1e-5)] * 3,
    ]


def test_an_island_that_starts_with_fits_closed_is_told_the_trained_values(store):
    study = build_study(store=store, truth=_gain, space=_training_space(Choice(values=(1e-5, 1e-4))))
    trained = {"sft": [{"sft.learning_rate": 1e-4}]}
    message = system_message(study=study, structure="sft+prompt", goal="formal replies", trained=trained)
    assert "No new training or merge may run" in message
    assert '- `sft`: {"sft.learning_rate": 0.0001}' in message
    none_yet = system_message(study=study, structure="sft", goal="formal replies", trained={"sft": []})
    assert "none has been trained, so no candidate of this structure can be evaluated" in none_yet
    assert "No new training" not in system_message(study=study, structure="sft", goal="formal replies")


def _by_weights(recipe, item_id):
    path = argument(recipe, "structural_control/load_lora", "path")
    if path is None:
        return Truth(loss=0.01, gain=0.15)
    return Truth(gain=0.1) if path.endswith("adapter-a") else Truth(loss=0.02, gain=0.2)


def test_the_certificate_records_the_weights_of_each_structural_control(tmp_path, store):
    directories = []
    for name in ADAPTERS[:2]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "adapter_config.json").write_text(f'{{"name": "{name}"}}')
        directories.append(str(directory))
    study = build_study(store=store, truth=_by_weights, space=_space(directories))
    session = _started(study, store)
    for candidate in (_lora(directories[0]), _prompt("be formal"), _lora(directories[1])):
        assert session.evaluate(candidate=candidate).status == "accepted"
    certification = certify(session=session, bundle_dir=tmp_path / "bundles")
    assert len(certification.certified) == 3
    identity = {path: encode(LoRAArtifact(path=path, base_model=BASE), EncodeContext(store=None))["$artifact"]
                for path in directories}
    for certificate in certification.certificates.values():
        if certificate["structure"] == "prompt":
            assert certificate["weights"] == []
        else:
            expected = identity[certificate["values"]["lora.path"]]
            assert certificate["weights"] == [{"method": "structural_control/load_lora", "artifact_id": expected}]
    assert len(set(identity.values())) == 2
    disabled = LoadLoRA(path=directories[0], base_model=BASE)
    disabled.enabled = False
    pipeline = SimpleNamespace(structural_controls=[LoadLoRA(path=directories[1], base_model=BASE), disabled])
    assert certificate_weights(pipeline=pipeline) == [
        {"method": "structural_control/load_lora", "artifact_id": identity[directories[1]]},
    ]
