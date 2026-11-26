"""An end-to-end run of the OpenEvolve engine against a scripted language model, in the synthetic world.

The model is a local HTTP server that answers OpenEvolve's chat completions with scripted programs. Each island follows
its own script whatever order OpenEvolve schedules the islands in, since the system message of a request gives the
island's structure. The run checks the archive (the statuses and reasons of the scripted candidates) and OpenEvolve's
population in its last checkpoint.
"""
import http.server
import json
import re
import threading
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("openevolve")

from steerability.algorithms.core.recipe import Choice
from steerability.algorithms.core.store import Store
from steerability.optimization.search._openevolve_controller import latest_checkpoint
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.openevolve import BLOCK_END, BLOCK_START, OpenEvolveEngine, parse, program_text
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

GOAL = "The behavior score measures how formal the reply reads."
TEXTS = ("be brief", "be formal", "be kind")
PROMPT_TRUTHS = {"be brief": Truth(), "be formal": Truth(gain=0.1), "be kind": Truth(loss=0.08, gain=0.2)}


def _truth(recipe, item_id):
    value = multiplier(recipe)
    if value is None:
        return PROMPT_TRUTHS[argument(recipe, "input_control/system_prompt", "text")]
    if argument(recipe, "state_control/caa", "layer_id") == 3:
        return Truth()
    return Truth(loss=0.0055 * value * value, gain=0.05 * value)


def _prompt(text):
    return Candidate(structure="prompt", values={"prompt.text": text})


def _layer(layer_id):
    return Candidate(structure="caa", values={"caa.layer_id": layer_id})


def _malformed(structure):
    return f"# header\n{BLOCK_START}\n{{\"structure\": \"{structure}\", \"values\": \n{BLOCK_END}\n"


class ScriptedModel(http.server.BaseHTTPRequestHandler):
    """Answers each chat completion with the next program of the script of the island's structure."""

    scripts: dict[str, list[str]] = {}
    asked: dict[str, int] = {}
    lock = threading.Lock()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system = next(message["content"] for message in body["messages"] if message["role"] == "system")
        structure = re.search(r"The structure `([^`]+)`", system).group(1)
        with self.lock:
            count = self.asked.get(structure, 0)
            self.asked[structure] = count + 1
            script = self.scripts[structure]
            program = script[count % len(script)]
        reply = {
            "id": f"scripted-{structure}-{count}",
            "object": "chat.completion",
            "created": 0,
            "model": "scripted",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"```text\n{program}```"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def model():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ScriptedModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_an_openevolve_run_keeps_each_island_on_its_structure_with_accepted_programs_once(tmp_path, model):
    store = Store.open(root=tmp_path / "store")
    space = {"prompt": prompt_template(texts=TEXTS), "caa": caa_template(layer_id=Choice(values=(1, 2, 3)))}
    study = build_study(store=store, truth=_truth, space=space, monotone={"caa.multiplier"})

    def text(candidate):
        return program_text(study=study, candidate=candidate)

    ScriptedModel.asked = {}
    ScriptedModel.scripts = {
        "prompt": [text(_prompt("be formal")), text(_prompt("be kind")), text(_prompt("be formal")), text(_layer(2)),
                   _malformed("prompt")],
        "caa": [text(_layer(2)), text(_layer(3)), text(_layer(2)), text(_prompt("be formal")), _malformed("caa")],
    }
    run_dir = tmp_path / "run"
    engine = OpenEvolveEngine(
        run_dir=run_dir,
        llm={
            "api_base": f"http://127.0.0.1:{model.server_address[1]}/v1",
            "api_key": "test",
            "models": [{"name": "scripted"}],
        },
        iterations=12,
        goal=GOAL,
        checkpoint_interval=1,
    )
    archive = run_study(
        study=study,
        phases=[SearchPhase(engine=engine)],
        store=store,
        factory=StubFactory(protocol=study.protocol),
    )
    assert set(ScriptedModel.asked) == {"prompt", "caa"} and sum(ScriptedModel.asked.values()) >= 10

    expected = [
        (_prompt("be brief"), "rejected", "no_behavior_gain"),
        (_prompt("be formal"), "accepted", None),
        (_prompt("be kind"), "rejected", "not_confirmed"),
        (_layer(1), "accepted", None),
        (_layer(2), "accepted", None),
        (_layer(3), "rejected", "no_behavior_gain"),
    ]
    for candidate, status, reason in expected:
        result = archive.latest(candidate_id=study.candidate_id(candidate=candidate))
        assert result is not None and (result.status, result.reason) == (status, reason), candidate

    checkpoint = Path(latest_checkpoint(output_dir=run_dir / "openevolve"))
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    structures = [record["structure"] for record in json.loads((run_dir / "islands.json").read_text(encoding="utf-8"))]
    assert structures == ["prompt", "caa"] and len(metadata["islands"]) == 2
    for island, members in enumerate(metadata["islands"]):
        admitted = []
        for program_id in members:
            program = json.loads((checkpoint / "programs" / f"{program_id}.json").read_text(encoding="utf-8"))
            candidate = parse(program=program["code"])
            assert candidate.structure == structures[island]
            result = archive.latest(candidate_id=study.candidate_id(candidate=candidate))
            assert program["parent_id"] is None or (result.accepted and result.fitness > 0)
            assert "level" in program["metrics"]
            admitted.append(study.candidate_id(candidate=candidate))
        assert admitted and len(admitted) == len(set(admitted))
