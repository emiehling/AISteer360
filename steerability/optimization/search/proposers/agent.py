"""A proposer that asks a language model with tools for each batch of candidates."""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steerability.algorithms.core.recipe import RecipeError
from steerability.optimization.search.description import (
    candidate_document,
    describe_axes,
    describe_structure,
    describe_task,
    describe_trained_values,
    parse_document,
    rejection_digest,
)
from steerability.optimization.search.population import initial_population, is_trained, trained_values
from steerability.optimization.search.proposers.base import BaseProposer
from steerability.optimization.study import Candidate

if TYPE_CHECKING:
    from steerability.optimization.evaluation.archive import ArchiveView
    from steerability.optimization.study import Study

logger = logging.getLogger(__name__)

NOTES_FILE = "notes.md"
EPISODES_DIR = "episodes"
ARCHIVE_LINES = 5  # the default number of best entries and of rejections the `archive` tool lists

ChatModel = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Mapping[str, Any]]

_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "structure": {"type": "string", "description": "The structure's name."},
        "values": {"type": "object", "description": "One value per setting of the structure, keyed by path."},
    },
    "required": ["structure", "values"],
}
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "describe",
            "description": (
                "Describe a structure: its controls, its settings with their ranges, and the trained values a "
                "training may take when no new training may run. Without a structure, list the structures."
            ),
            "parameters": {"type": "object", "properties": {"structure": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archive",
            "description": (
                "Summarize the evaluated candidates of a structure: the best accepted ones and the latest "
                "rejections. Without a structure, give the counts of every structure."
            ),
            "parameters": {
                "type": "object",
                "properties": {"structure": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Give the result of one candidate, or say that it has not been evaluated.",
            "parameters": {
                "type": "object",
                "properties": {"candidate": _CANDIDATE_SCHEMA},
                "required": ["candidate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note",
            "description": "Append a note to the notes file, which every later episode begins from.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose",
            "description": (
                "Propose new candidates to evaluate. The episode ends once at least one of them is new and valid."
            ),
            "parameters": {
                "type": "object",
                "properties": {"candidates": {"type": "array", "items": _CANDIDATE_SCHEMA}},
                "required": ["candidates"],
            },
        },
    },
]


def chat_model(*, llm: ChatModel | Mapping[str, Any]) -> ChatModel:
    """The function that sends an episode's messages and tools to a model and returns its reply.

    A callable is returned unchanged. It is called as `llm(messages=..., tools=...)` with the messages and the tool
    schemas in the format of the OpenAI chat completions API, and it returns the assistant message as a mapping
    (`"content"` and `"tool_calls"`). A mapping
    configures an OpenAI-compatible endpoint with the keys `api_base`, `api_key` (or the `OPENAI_API_KEY` environment
    variable), the model's `name` (or the first entry of `models`, as in OpenEvolve's configuration), and optionally
    `temperature`, `max_tokens`, and `timeout`. A mapping requires the `openai` package, which the `optimize` extra
    installs.

    Args:
        llm: The callable or the endpoint configuration.

    Returns:
        The callable.

    Raises:
        TypeError: If `llm` is neither a callable nor a mapping.
        ValueError: If the mapping gives no `api_base` or no model name.
    """
    if callable(llm):
        return llm
    if not isinstance(llm, Mapping):
        raise TypeError(f"llm must be a callable or a mapping that configures an endpoint; got {type(llm).__name__}.")
    models = llm.get("models") or []
    name = llm.get("name") or (models[0].get("name") if models else None)
    if not llm.get("api_base") or not name:
        raise ValueError("An llm mapping needs an api_base and a model name (name, or the first entry of models).")
    options = {key: llm[key] for key in ("temperature", "max_tokens") if llm.get(key) is not None}
    from openai import OpenAI

    client = OpenAI(
        base_url=llm["api_base"],
        api_key=llm.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        timeout=llm.get("timeout"),
    )

    def call(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Mapping[str, Any]:
        completion = client.chat.completions.create(model=name, messages=messages, tools=tools, **options)
        return completion.choices[0].message.model_dump(exclude_none=True)

    return call


class AgentProposer(BaseProposer):
    """A proposer whose candidates come from a language model that reads the archive through tools.

    Each `ask(view, n)` is one episode, i.e., a conversation with the model that ends when it proposes at least one
    new valid candidate (at most `n`) or when it reaches `max_steps` replies. The model has five tools:

    - **describe**: the structures in scope, or one structure's controls and settings (`search/description.py`),
      with the trained values of its trainings when no new training may run (`trained_values`).
    - **archive**: the best accepted candidates and the latest rejections of a structure (`rejection_digest`).
    - **lookup**: the result of one candidate in the view.
    - **note**: text appended to `<run_dir>/notes.md`.
    - **propose**: candidates as JSON documents (`parse_document`), each bound by the study. A candidate outside
      `structures`, one that does not bind, one that is in the view, and, when no new training may run, one whose
      training has not been trained are refused, and the reply lists why.

    An episode begins from the notes file and the view only. Its first message contains the notes, the number of
    candidates of each structure in the view, and, when the notes file does not exist yet, the initial population
    (`initial_population`) as starting points. A run therefore resumes from the run directory and the store. Each
    episode's messages are written to `<run_dir>/episodes/<number>.json` and are not read back. Nothing the model
    writes is executed, since its tool calls are data. An episode that ends without a proposal returns no candidate,
    which ends the proposer's run.

    Args:
        study: The study.
        run_dir: The directory of the notes file and the episodes.
        llm: The model (see `chat_model`), i.e., a callable or an OpenAI-compatible endpoint configuration.
        goal: What the behavior score measures (see `describe_task`). It must not contain benchmark item text.
        structures: The structures to propose, or None for every structure of the study.
        max_steps: The largest number of model replies per episode.

    Raises:
        ValueError: If `goal` is empty, `max_steps` is not a positive integer, or `structures` is invalid.
        TypeError: If `llm` is neither a callable nor a mapping.
    """

    name = "agent"

    def __init__(
        self,
        *,
        study: Study,
        run_dir: str | Path,
        llm: ChatModel | Mapping[str, Any],
        goal: str,
        structures: Sequence[str] | None = None,
        max_steps: int = 16,
    ) -> None:
        super().__init__(study=study, structures=structures)
        describe_task(goal=goal)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError(f"max_steps must be a positive integer; got {max_steps!r}.")
        self.run_dir = Path(run_dir)
        self.goal = goal
        self.max_steps = max_steps
        self._model = chat_model(llm=llm)

    @property
    def notes_path(self) -> Path:
        """The notes file of the run directory."""
        return self.run_dir / NOTES_FILE

    def ask(self, view: ArchiveView, n: int) -> list[Candidate]:
        """Run one episode and return the candidates the model proposed, or none when it proposed none."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_message(n=n)},
            {"role": "user", "content": self._opening(view=view, n=n)},
        ]
        proposed: list[Candidate] = []
        for _ in range(self.max_steps):
            reply = dict(self._model(messages=messages, tools=TOOLS))
            calls = list(reply.get("tool_calls") or [])
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            if not calls:
                reminder = f"Call a tool; `propose` with up to {n} candidates ends the episode."
                messages.append({"role": "user", "content": reminder})
                continue
            for call in calls:
                text, proposed = self._tool(call=call, view=view, n=n)
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": text})
                if proposed:
                    break
            if proposed:
                break
        else:
            logger.warning("The agent's episode ended after %d replies without a proposal.", self.max_steps)
        self._write_episode(messages=messages)
        return proposed

    def _system_message(self, *, n: int) -> str:
        lines = "\n".join(f"- {describe_structure(study=self.study, structure=name)}" for name in self.structures)
        example = candidate_document(candidate=self.study.default_candidate(structure=self.structures[0]))
        paragraphs = [
            describe_task(goal=self.goal),
            f"The structures you may propose are:\n{lines}",
            f"A candidate is a JSON object such as {example}. Use `describe` for a structure's settings, `archive` and "
            "`lookup` to read the results so far, and `note` to keep what you learn, since the notes are the only "
            f"thing kept from one episode to the next. The episode ends when you call `propose` with up to {n} new "
            "candidates, which are then evaluated.",
        ]
        return "\n\n".join(paragraphs)

    def _opening(self, *, view: ArchiveView, n: int) -> str:
        paragraphs = []
        if self.notes_path.exists():
            notes = self.notes_path.read_text(encoding="utf-8").strip()
            paragraphs.append(f"Your notes from earlier episodes:\n\n{notes}" if notes else "Your notes file is empty.")
        else:
            population = initial_population(study=self.study, view=view, structures=self.structures)
            starts = "\n".join(
                f"- {candidate_document(candidate=candidate)}" for candidates in population.values()
                for candidate in candidates
            )
            paragraphs.append(f"This is the first episode. Candidates to start from:\n{starts}")
        evaluated = {structure: 0 for structure in self.structures}
        for entry in view:
            if entry.structure in evaluated:
                evaluated[entry.structure] += 1
        counts = ", ".join(f"`{structure}`: {count}" for structure, count in evaluated.items())
        paragraphs.append(f"Evaluated candidates so far, by structure: {counts}.")
        if not view.fits_open:
            paragraphs.append("No new training or merge may run now; `describe` gives the values that are trained.")
        paragraphs.append(f"Propose up to {n} candidates.")
        return "\n\n".join(paragraphs)

    def _tool(self, *, call: Mapping[str, Any], view: ArchiveView, n: int) -> tuple[str, list[Candidate]]:
        """The reply to one tool call, and the candidates it proposed (empty unless a proposal was taken)."""
        function = call.get("function") or {}
        name = function.get("name")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as error:
            return f"The arguments are not JSON ({error}).", []
        if not isinstance(arguments, dict):
            return "The arguments must be a JSON object.", []
        if name == "describe":
            return self._describe(structure=arguments.get("structure"), view=view), []
        if name == "archive":
            return self._archive(structure=arguments.get("structure"), limit=arguments.get("limit"), view=view), []
        if name == "lookup":
            return self._lookup(candidate=arguments.get("candidate"), view=view), []
        if name == "note":
            return self._note(text=arguments.get("text")), []
        if name == "propose":
            return self._propose(candidates=arguments.get("candidates"), view=view, n=n)
        return f"There is no tool {name!r}; the tools are {[tool['function']['name'] for tool in TOOLS]}.", []

    def _known(self, *, structure: Any) -> str | None:
        """The reason a structure cannot be used, or None."""
        if structure not in self.structures:
            return f"The structure {structure!r} is not one of {list(self.structures)}."
        return None

    def _describe(self, *, structure: Any, view: ArchiveView) -> str:
        if structure is None:
            return "\n".join(f"- {describe_structure(study=self.study, structure=name)}" for name in self.structures)
        problem = self._known(structure=structure)
        if problem is not None:
            return problem
        axes = describe_axes(study=self.study, structure=structure)
        paragraphs = [
            describe_structure(study=self.study, structure=structure),
            f"Its settings, keyed by path, are:\n{axes}" if axes else "It has no settings to propose.",
        ]
        if not view.fits_open:
            trained = describe_trained_values(trained=trained_values(study=self.study, view=view, structure=structure))
            if trained:
                paragraphs.append(trained)
        example = candidate_document(candidate=self.study.default_candidate(structure=structure))
        paragraphs.append(f"For instance, a candidate of this structure is {example}.")
        return "\n\n".join(paragraphs)

    def _archive(self, *, structure: Any, limit: Any, view: ArchiveView) -> str:
        if structure is None:
            lines = []
            for name in self.structures:
                entries = [entry for entry in view if entry.structure == name]
                accepted = [entry for entry in entries if entry.status == "accepted"]
                best = max((entry.fitness for entry in accepted), default=None)
                line = f"- `{name}`: {len(entries)} evaluated, {len(accepted)} accepted"
                lines.append(line + ("" if best is None else f", best fitness {best:.3g}"))
            return "\n".join(lines)
        problem = self._known(structure=structure)
        if problem is not None:
            return problem
        count = limit if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 else ARCHIVE_LINES
        best = [entry for entry in view.accepted() if entry.structure == structure][:count]
        lines = [f"The best accepted candidates of `{structure}`, highest fitness first:"]
        lines.extend(
            f"- {candidate_document(candidate=Candidate(structure=entry.structure, values=entry.values))} -> fitness "
            f"{entry.fitness:.3g}, summary {json.dumps(dict(entry.summary), sort_keys=True)}"
            for entry in best
        )
        if not best:
            lines.append("- none yet")
        rejections = rejection_digest(view=view, structure=structure, limit=count)
        lines.append(f"The latest rejections, newest first:\n{rejections}" if rejections else "No rejection yet.")
        return "\n".join(lines)

    def _lookup(self, *, candidate: Any, view: ArchiveView) -> str:
        try:
            parsed = parse_document(text=json.dumps(candidate))
            candidate_id = self.study.candidate_id(candidate=parsed)
        except (ValueError, TypeError, RecipeError) as error:
            return f"The candidate is invalid: {error}"
        entry = view.get(candidate_id)
        if entry is None:
            return "The candidate has not been evaluated."
        result = {
            "status": entry.status,
            "stage": entry.stage,
            "reason": entry.reason,
            "fitness": entry.fitness,
            "summary": dict(entry.summary),
        }
        return json.dumps(result, sort_keys=True)

    def _note(self, *, text: Any) -> str:
        if not isinstance(text, str) or not text.strip():
            return "A note must be a non-empty string."
        with self.notes_path.open("a", encoding="utf-8") as file:
            file.write(text.strip() + "\n\n")
        return "Noted."

    def _propose(self, *, candidates: Any, view: ArchiveView, n: int) -> tuple[str, list[Candidate]]:
        if not isinstance(candidates, list) or not candidates:
            return "`candidates` must be a non-empty list of candidates.", []
        valid: list[Candidate] = []
        problems: list[str] = []
        trained: dict[str, Any] = {}
        for index, item in enumerate(candidates):
            try:
                candidate = parse_document(text=json.dumps(item))
            except (ValueError, TypeError) as error:
                problems.append(f"candidate {index}: {error}")
                continue
            problem = self._known(structure=candidate.structure)
            if problem is None:
                try:
                    self.study.candidate_id(candidate=candidate)
                except RecipeError as error:
                    problem = str(error)
            if problem is None and not view.fits_open:
                if candidate.structure not in trained:
                    trained[candidate.structure] = trained_values(
                        study=self.study,
                        view=view,
                        structure=candidate.structure,
                    )
                if not is_trained(study=self.study, candidate=candidate, trained=trained[candidate.structure]):
                    problem = "it needs a new training, and no new training may run now (see `describe`)"
            if problem is not None:
                problems.append(f"candidate {index}: {problem}")
                continue
            valid.append(candidate)
        taken = self._take_new(candidates=valid, view=view, n=n)
        if len(taken) < len(valid):
            skipped = len(valid) - len(taken)
            problems.append(f"{skipped} valid candidate(s) were evaluated or proposed before, or beyond the first {n}")
        if not taken:
            return "No candidate was taken:\n" + "\n".join(f"- {problem}" for problem in problems), []
        text = f"{len(taken)} candidate(s) will be evaluated."
        if problems:
            text += " Not taken:\n" + "\n".join(f"- {problem}" for problem in problems)
        return text, taken

    def _write_episode(self, *, messages: list[dict[str, Any]]) -> None:
        directory = self.run_dir / EPISODES_DIR
        directory.mkdir(exist_ok=True)
        number = sum(1 for _ in directory.glob("*.json"))
        (directory / f"{number:04d}.json").write_text(json.dumps(messages, indent=1, default=str), encoding="utf-8")


__all__ = ["AgentProposer", "TOOLS", "chat_model"]
