"""Tests for the agent proposer, with scripted models in place of a language model.

A scripted model is a list of replies, each an assistant message or a function of the episode's messages that returns
one. The resume test runs its second episode in another interpreter on the same run directory and store.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

from datasets import Dataset

from steerability.algorithms.core.recipe import Choice, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT
from steerability.optimization import Candidate, optional, structures
from steerability.optimization.evaluation.archive import Archive, ArchiveEntry, ArchiveView
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.proposers import AgentProposer
from steerability.optimization.search.proposers.agent import chat_model
from tests.optimization.helpers import StubFactory, Truth, build_study, prompt_template

TEXTS = ("be brief", "be formal", "be kind", "be exact", "be polite", "be warm")
GOAL = "The behavior score measures how formal the reply reads."
NOTE = "Formal and exact texts come first."
REPOSITORY = Path(__file__).resolve().parents[2]


def _gain(recipe, item_id):
    return Truth(gain=0.1)


def tool_call(*, name, **arguments):
    """One tool call of an assistant message."""
    return {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def proposal(*texts):
    """A `propose` call with one `prompt` candidate per text."""
    candidates = [{"structure": "prompt", "values": {"prompt.text": text}} for text in texts]
    return tool_call(name="propose", candidates=candidates)


def reply(*calls, content=""):
    """An assistant message with tool calls."""
    return {"content": content, "tool_calls": list(calls)}


class Scripted:
    """A model that answers each request with the next scripted reply and keeps the requests it was sent."""

    def __init__(self, *, replies):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def __call__(self, *, messages, tools):
        self.requests.append([dict(message) for message in messages])
        answer = self.replies.pop(0)
        return answer(messages) if callable(answer) else answer

    def tool_replies(self, *, request=-1):
        return [message["content"] for message in self.requests[request] if message["role"] == "tool"]


def _study(store):
    return build_study(store=store, truth=_gain, space={"prompt": prompt_template(texts=TEXTS)})


def _agent(study, tmp_path, model, **options):
    return AgentProposer(study=study, run_dir=tmp_path / "run", llm=model, goal=GOAL, **options)


def _entry(study, *, candidate, status, stage, reason, fitness, summary):
    return ArchiveEntry(
        candidate_id=study.candidate_id(candidate=candidate),
        structure=candidate.structure,
        values=candidate.values,
        status=status,
        stage=stage,
        reason=reason,
        fitness=fitness,
        summary=summary,
    )


def test_an_episode_reads_the_archive_takes_notes_and_proposes_a_batch_that_is_recorded(tmp_path):
    store = Store.open(root=tmp_path / "store")
    study = _study(store)
    model = Scripted(
        replies=[
            reply(tool_call(name="describe", structure="prompt"), tool_call(name="archive")),
            reply(tool_call(name="note", text=NOTE), proposal(*TEXTS[:4])),
        ],
    )
    archive = run_study(
        study=study,
        phases=[SearchPhase(engine=_agent(study, tmp_path, model), candidates=4)],
        store=store,
        factory=StubFactory(protocol=study.protocol),
    )
    assert {entry.values["prompt.text"] for entry in archive.view()} == set(TEXTS[:4])
    describe, counts = model.tool_replies()[:2]
    assert "`prompt.text`: one of" in describe and "For instance, a candidate of this structure is" in describe
    assert counts == "- `prompt`: 0 evaluated, 0 accepted"
    system, opening = model.requests[0]
    assert system["role"] == "system" and GOAL in system["content"]
    assert "This is the first episode. Candidates to start from:" in opening["content"]
    assert (tmp_path / "run" / "notes.md").read_text() == f"{NOTE}\n\n"
    (episode,) = sorted((tmp_path / "run" / "episodes").glob("*.json"))
    assert json.loads(episode.read_text())[-1]["content"] == "4 candidate(s) will be evaluated."


def test_a_proposal_lists_why_candidates_were_not_taken_and_the_episode_goes_on(tmp_path):
    store = Store.open(root=tmp_path / "store")
    space = {"prompt": prompt_template(texts=TEXTS), "other": prompt_template(texts=TEXTS)}
    study = build_study(store=store, truth=_gain, space=space)
    known = Candidate(structure="prompt", values={"prompt.text": "be brief"})
    rejected = _entry(
        study,
        candidate=known,
        status="rejected",
        stage=2,
        reason="no_behavior_gain",
        fitness=0.0,
        summary={"gain": -0.01},
    )
    refused = tool_call(
        name="propose",
        candidates=[
            {"structure": "other", "values": {"prompt.text": "be formal"}},
            {"structure": "prompt", "values": {"prompt.text": "be loud"}},
            {"structure": "prompt", "values": {"prompt.text": "be brief"}},
            "not a candidate",
        ],
    )
    lookup = tool_call(name="lookup", candidate={"structure": "prompt", "values": dict(known.values)})
    model = Scripted(
        replies=[
            reply(refused),
            reply(lookup, content="Let me look."),
            reply(tool_call(name="archive", structure="prompt")),
            reply(proposal("be formal", "be kind")),
        ],
    )
    agent = _agent(study, tmp_path, model, structures=["prompt"])
    proposed = agent.ask(view=ArchiveView(entries=(rejected,)), n=4)
    assert [candidate.values["prompt.text"] for candidate in proposed] == ["be formal", "be kind"]
    first, looked_up, digest = model.tool_replies(request=3)
    assert first.startswith("No candidate was taken:")
    assert "candidate 0: The structure 'other' is not one of ['prompt']." in first
    assert "candidate 1:" in first and "candidate 3:" in first
    assert "1 valid candidate(s) were evaluated or proposed before" in first
    assert json.loads(looked_up) == {
        "fitness": 0.0,
        "reason": "no_behavior_gain",
        "stage": 2,
        "status": "rejected",
        "summary": {"gain": -0.01},
    }
    assert '{"prompt.text":"be brief"} -> no_behavior_gain, gain -0.01' in digest and "- none yet" in digest


def test_an_episode_without_a_proposal_ends_at_its_step_limit_and_returns_no_candidate(tmp_path, caplog):
    store = Store.open(root=tmp_path / "store")
    study = _study(store)
    model = Scripted(replies=[{"content": "I am thinking."}] * 3)
    agent = _agent(study, tmp_path, model, max_steps=3)
    with caplog.at_level("WARNING", logger="steerability.optimization.search.proposers.agent"):
        assert agent.ask(view=ArchiveView(entries=()), n=4) == []
    assert "ended after 3 replies without a proposal" in caplog.text
    assert model.requests[-1][-1] == {
        "role": "user",
        "content": "Call a tool; `propose` with up to 4 candidates ends the episode.",
    }


def test_with_fits_closed_an_untrained_proposal_is_refused_and_describe_gives_the_trained_values(tmp_path):
    store = Store.open(root=tmp_path / "store")
    dataset = Dataset.from_dict({"text": ["a formal reply"]})
    sft = entry(SFT, name="sft", train_dataset=dataset, use_peft=True, learning_rate=Choice(values=(1e-5, 1e-4)))
    prompt = entry(SystemPrompt, name="prompt", text=Choice(values=TEXTS))
    study = build_study(store=store, truth=_gain, space=structures(optional(sft), optional(prompt)))
    trained = _entry(
        study,
        candidate=Candidate(structure="sft", values={"sft.learning_rate": 1e-4}),
        status="accepted",
        stage=5,
        reason=None,
        fitness=0.1,
        summary={},
    )
    untrained = {"structure": "sft+prompt", "values": {"sft.learning_rate": 1e-5, "prompt.text": "be kind"}}
    allowed = {"structure": "sft+prompt", "values": {"sft.learning_rate": 1e-4, "prompt.text": "be kind"}}
    describe_first = tool_call(name="describe", structure="sft+prompt")
    model = Scripted(
        replies=[
            reply(describe_first, tool_call(name="propose", candidates=[untrained])),
            reply(tool_call(name="propose", candidates=[allowed])),
        ],
    )
    agent = _agent(study, tmp_path, model, structures=["sft+prompt"])
    (proposed,) = agent.ask(view=ArchiveView(entries=(trained,), fits_open=False), n=2)
    assert proposed.values == allowed["values"]
    describe, refused = model.tool_replies(request=1)
    assert '- `sft`: {"sft.learning_rate": 0.0001}' in describe
    assert "it needs a new training, and no new training may run now" in refused
    assert "No new training or merge may run now" in model.requests[0][1]["content"]


def test_an_endpoint_configuration_calls_the_chat_completions_api(monkeypatch):
    openai = pytest.importorskip("openai")
    created = []

    class Message:
        def model_dump(self, exclude_none):
            return {"role": "assistant", "content": "hello"}

    class Client:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kwargs):
            created.append(kwargs)
            return type("Completion", (), {"choices": [type("Choice", (), {"message": Message()})()]})()

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-environment")
    model = chat_model(llm={"api_base": "http://localhost:8000/v1", "models": [{"name": "m"}], "temperature": 0.2})
    assert model(messages=[{"role": "user", "content": "hi"}], tools=[]) == {"role": "assistant", "content": "hello"}
    client, request = created
    assert client == {"base_url": "http://localhost:8000/v1", "api_key": "from-the-environment", "timeout": None}
    assert request == {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [], "temperature": 0.2}
    with pytest.raises(ValueError, match="api_base and a model name"):
        chat_model(llm={"models": [{"name": "m"}]})
    with pytest.raises(TypeError, match="llm must be a callable or a mapping"):
        chat_model(llm=3)


SECOND_EPISODE = '''
import sys

sys.path.insert(0, {repository!r})
from steerability.algorithms.core.store import Store
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.proposers import AgentProposer
from tests.optimization import test_agent
from tests.optimization.helpers import StubFactory


def second(messages):
    opening = messages[1]["content"]
    assert test_agent.NOTE in opening, opening
    assert "Evaluated candidates so far, by structure: `prompt`: 2." in opening, opening
    return test_agent.reply(test_agent.proposal("be polite", "be warm"))


store = Store.open(root={store!r})
study = test_agent._study(store)
agent = AgentProposer(
    study=study,
    run_dir={run_dir!r},
    llm=test_agent.Scripted(replies=[second]),
    goal=test_agent.GOAL,
)
run_study(
    study=study,
    phases=[SearchPhase(engine=agent, candidates=2)],
    store=store,
    factory=StubFactory(protocol=study.protocol),
)
store.close()
'''


def test_a_second_process_on_the_same_run_directory_and_store_continues_from_the_notes(tmp_path):
    store = Store.open(root=tmp_path / "store")
    study = _study(store)
    first = Scripted(replies=[reply(tool_call(name="note", text=NOTE)), reply(proposal("be formal", "be exact"))])
    run_study(
        study=study,
        phases=[SearchPhase(engine=_agent(study, tmp_path, first), candidates=2)],
        store=store,
        factory=StubFactory(protocol=study.protocol),
    )
    store.close()
    source = SECOND_EPISODE.format(
        repository=str(REPOSITORY),
        store=str(tmp_path / "store"),
        run_dir=str(tmp_path / "run"),
    )
    script = tmp_path / "second_episode.py"
    script.write_text(textwrap.dedent(source))
    completed = subprocess.run([sys.executable, str(script)], cwd=REPOSITORY, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-3000:]
    reopened = Store.open(root=tmp_path / "store")
    recorded = {entry.values["prompt.text"] for entry in Archive(study=_study(reopened), store=reopened).view()}
    assert recorded == {"be formal", "be exact", "be polite", "be warm"}
    assert len(list((tmp_path / "run" / "episodes").glob("*.json"))) == 2
