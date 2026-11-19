"""Item-level evaluation: benchmarks, direct generation, and stored per-item evidence.

A benchmark defines items and how a completion is scored. `evaluate_items` measures a steered
pipeline on a selection of items and stores one evidence row per item, keyed by the protocol, the
resolved recipe, the benchmark, the item, and the replicate. Evidence is the resume mechanism: a
second call returns stored rows and generates nothing.

`PromptBenchmark` scores prompts with a `SampleScorer` and generates them directly
(`generate_requests`), storing each rollout by a key known before generation, so a changed scorer
re-scores stored completions. `InspectBenchmark` runs one Inspect task through `inspect_ai.eval`.
"""
from steerability.utils.optional import require

require("inspect_ai")
import hashlib
import importlib.metadata
import json
import logging
import random
import threading
import typing
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from steerability.algorithms.core.identity import content_id, derive_seed
from steerability.algorithms.core.scoring import SampleScorer
from steerability.algorithms.core.store import EvidenceRow, ItemRow, RolloutRow, ScoreRow, Store
from steerability.algorithms.core.utils.controls import runtime_kwargs_schema
from steerability.utils.rendering import has_chat_template, join_message_contents

if TYPE_CHECKING:
    from steerability.algorithms.core.sweeps import Steered

logger = logging.getLogger(__name__)

# `inspect_ai.eval` is called from one thread at a time
_EVAL_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class ItemResult:
    """The measurement of one item.

    Attributes:
        item_id: The item's id.
        value: The chosen scorer's value, on that scorer's scale (0.0 on an error row).
        status: `"ok"`, or `"error"` when generation or scoring failed after the retries.
        tokens: The tokens generated for the item, or None.
        completion: The text the scorers saw, or None on an error row.
        scores: Every scorer's raw value, by scorer name.
        rollout_key: The key of the stored rollout, or None for a row produced through Inspect.
        generated: Whether this call ran the model for the item; not stored.
    """

    item_id: str
    value: float
    status: Literal["ok", "error"]
    tokens: int | None
    completion: str | None
    scores: Mapping[str, Any]
    rollout_key: str | None = None
    generated: bool = False


class Benchmark(typing.Protocol):
    """A task to evaluate on: items, and how a completion of each item is scored.

    Attributes:
        name: The benchmark's name in frames and selections.
    """

    name: str

    @property
    def id(self) -> str:
        """The content id of everything that defines the items and their scoring."""
        ...

    def prepare(self, store: Store) -> None:
        """Write the benchmark's items to the store (once)."""
        ...

    def item_ids(self) -> tuple[str, ...]:
        """The item ids, in order, as written by `prepare`."""
        ...

    def groups(self) -> Mapping[str, str]:
        """Mapping from item id to group id; empty when the benchmark is ungrouped."""
        ...

    def run(self, steered: "Steered", item_ids: Sequence[str], *, store: Store, replicate: int) -> dict[str, ItemResult]:
        """Evaluate the given items without reading stored evidence.

        Returns a row for every item it completed and omits an item whose generation or scoring
        failed. Callers use `evaluate_items`.
        """
        ...


@dataclass(frozen=True)
class Request:
    """One prompt to generate directly.

    Attributes:
        item_id: The item the request belongs to.
        prompt: A string (one user turn) or a conversation of `{"role", "content"}` messages.
        generate: Per-item generation arguments, in `GenerateConfig` vocabulary.
        runtime_kwargs: Per-sample runtime kwargs.
    """

    item_id: str
    prompt: str | Sequence[Mapping[str, str]]
    generate: Mapping[str, Any] = field(default_factory=dict)
    runtime_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Generated:
    """One generated completion.

    Attributes:
        completion: The text a scorer sees (reasoning split off, stop strings applied).
        tokens: The tokens generated.
        finish_reason: The finish reason of the generation.
        rollout_key: The key of the stored rollout.
        generated: False when the rollout was already stored.
    """

    completion: str
    tokens: int
    finish_reason: str | None
    rollout_key: str
    generated: bool


def _messages(prompt: str | Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": str(message["role"]), "content": str(message["content"])} for message in prompt]


def _prompt_chars(prompt: Any) -> int:
    if isinstance(prompt, str):
        return len(prompt)
    return sum(len(str(message.get("content", ""))) for message in prompt)


@dataclass
class _Pending:
    request: Request
    key: str
    record: Any


def generate_requests(
    steered: "Steered",
    requests: Sequence[Request],
    *,
    store: Store,
    benchmark_generate: Mapping[str, Any],
    replicate: int,
) -> dict[str, Generated]:
    """Generate the completions of `requests` directly through the steered pipeline.

    Generation arguments are `protocol.generate_defaults`, then `benchmark_generate`, then the
    request's own, in `GenerateConfig` vocabulary; a later one wins. Static runtime kwargs come
    from the protocol's provider options. A sampled request's seed is `request_seed(derive_seed(
    protocol.seed, "replicate", replicate), prompt, runtime_kwargs)`, the seed per-request
    dispatch uses, so it does not depend on the benchmark or on batching.

    Each request's rollout key covers the protocol, the resolved recipe, the prompt, the runtime
    kwargs, the merged arguments, the seed, and the replicate. A stored rollout is returned without
    generating. Greedy requests are grouped by batch key, ordered by `(prompt length in characters,
    item id)` at replicate 0 and shuffled by `derive_seed(protocol.seed, "order", replicate)` at
    later replicates, and batched up to the provider's `max_batch_size` when the pipeline supports
    batching. Sampled requests generate one per call, concurrently on a backend that supports
    concurrent generation. A request whose generation fails is omitted from the result.

    Args:
        steered: The steered pipeline.
        requests: The requests.
        store: The store rollouts are written to.
        benchmark_generate: The benchmark's generation arguments.
        replicate: The replicate index.

    Returns:
        Mapping from item id to its generation.

    Raises:
        ValueError: If a request asks for more than one choice.
    """
    from inspect_ai.model import GenerateConfig

    from steerability.evaluation.batching import RequestAdmission, request_seed
    from steerability.evaluation.provider import CompletionAssembler, map_generate_config, output_token_count

    pipeline = steered.pipeline
    protocol = steered.protocol
    options = protocol.resolved_provider
    prompt_path = "messages" if has_chat_template(pipeline.tokenizer) else "text"
    schema = runtime_kwargs_schema(pipeline.controls)
    admission = RequestAdmission(
        pipeline,
        prompt_path=prompt_path,
        declared_scopes={name: entry["scope"] for name, entry in schema.items()},
        static_runtime_kwargs=options.runtime_kwargs,
        chat_template_kwargs=options.chat_template_kwargs,
    )
    assembler = CompletionAssembler(pipeline.tokenizer, options)
    pad_token_id = getattr(pipeline.tokenizer, "pad_token_id", None)
    base_seed = derive_seed(protocol.seed, "replicate", replicate)

    results: dict[str, Generated] = {}
    greedy: list[_Pending] = []
    sampled: list[_Pending] = []
    for request in requests:
        merged = {**protocol.generate_defaults, **benchmark_generate, **request.generate}
        config = GenerateConfig(**merged)
        gen_kwargs, extra_runtime, num_choices = map_generate_config(config, options, base_seed=base_seed)
        if num_choices > 1:
            raise ValueError(f"Item {request.item_id!r} asks for {num_choices} choices; direct generation makes one.")
        runtime_kwargs = {**extra_runtime, **request.runtime_kwargs}
        messages = _messages(request.prompt)
        prompt = messages if prompt_path == "messages" else join_message_contents(messages)
        seed = None
        if gen_kwargs.get("seed") is not None:
            seed = request_seed(int(gen_kwargs["seed"]), prompt, runtime_kwargs)
            gen_kwargs["seed"] = seed
        key = content_id({
            "protocol": protocol.id, "resolved": steered.resolved_id, "prompt": prompt,
            "runtime_kwargs": runtime_kwargs, "generate": merged, "seed": seed, "replicate": replicate,
        })
        stored = store.get_rollout(key)
        if stored is not None:
            results[request.item_id] = Generated(stored.completion, stored.tokens or 0, stored.finish_reason, key, False)
            continue
        record = admission.admit(prompt, gen_kwargs, runtime_kwargs, num_choices)
        (sampled if seed is not None or gen_kwargs.get("do_sample") else greedy).append(_Pending(request, key, record))

    def finish(pending: _Pending, output) -> None:
        stop_strings = tuple(pending.record.gen_kwargs.get("stop_strings", ()))
        completion = assembler.completions(output, stop_strings)[0]
        row = store.put_rollout(RolloutRow(
            rollout_key=pending.key, protocol_id=protocol.id, resolved_id=steered.resolved_id,
            completion=completion, finish_reason=output.finish_reasons[0],
            tokens=output_token_count(output, pad_token_id), created_at=_now(),
        ))
        results[pending.request.item_id] = Generated(row.completion, row.tokens or 0, row.finish_reason, row.rollout_key, True)

    def run_one(pending: _Pending) -> None:
        try:
            finish(pending, admission.run_batch([pending.record])[0])
        except Exception:
            logger.warning("Generation failed for item %r.", pending.request.item_id, exc_info=True)

    groups: dict[str, list[_Pending]] = {}
    for pending in greedy:
        groups.setdefault(pending.record.batch_key, []).append(pending)
    batch_size = options.max_batch_size if pipeline.supports_batching else 1
    for members in groups.values():
        if replicate == 0:
            members.sort(key=lambda pending: (_prompt_chars(pending.request.prompt), pending.request.item_id))
        else:
            members.sort(key=lambda pending: pending.request.item_id)
            random.Random(derive_seed(protocol.seed, "order", replicate)).shuffle(members)
        for start in range(0, len(members), batch_size):
            chunk = members[start:start + batch_size]
            try:
                outputs = admission.run_batch([pending.record for pending in chunk])
            except Exception:
                if len(chunk) == 1:
                    logger.warning("Generation failed for item %r.", chunk[0].request.item_id, exc_info=True)
                    continue
                for pending in chunk:
                    run_one(pending)
                continue
            for pending, output in zip(chunk, outputs):
                finish(pending, output)

    if sampled:
        if pipeline.supports_concurrent_generate and len(sampled) > 1:
            with ThreadPoolExecutor(max_workers=options.max_concurrency or 32) as executor:
                list(executor.map(run_one, sampled))
        else:
            for pending in sampled:
                run_one(pending)
    return results


def _check_versioned(scorer: Any) -> None:
    """Check that a scorer declares a version, which enters its benchmark's identity.

    Raises:
        ValueError: If the scorer declares no `__steerability_version__`.
    """
    if not isinstance(getattr(scorer, "__steerability_version__", None), str):
        raise ValueError(
            f"Scorer {getattr(scorer, '__qualname__', scorer)!r} has no version; decorate it with "
            "@versioned('<version>') from steerability.algorithms.core.identity so its benchmark has an identity."
        )


class PromptBenchmark:
    """Prompts scored by a `SampleScorer`, generated directly.

    The benchmark's id covers the item inputs, targets, groups, generation arguments, version,
    and the scorer's reference with its version, so the scorer must be a module-level callable
    marked with `versioned`. A string item is one user message, and a sequence is a conversation
    of `{"role", "content"}` messages. The scorer receives the completion and the row
    `{"input": <item input>, "reference": <target>}`, and its value is the item's value.

    Args:
        name: The benchmark's name.
        items: Mapping from item id to prompt.
        scorer: The versioned `SampleScorer`.
        targets: Mapping from item id to target, or None.
        groups: Mapping from item id to group id, or None.
        generate: Generation arguments in `GenerateConfig` vocabulary, or None.
        version: The benchmark's own version, changed to invalidate stored evidence.

    Raises:
        ValueError: If the scorer has no version, or `items` is empty.
        SpipeSaveError: If an input, target, or the scorer has no serialized form.
    """

    def __init__(
        self,
        name: str,
        items: Mapping[str, str | Sequence[Mapping[str, str]]],
        scorer: SampleScorer,
        *,
        targets: Mapping[str, Any] | None = None,
        groups: Mapping[str, str] | None = None,
        generate: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not items:
            raise ValueError("A PromptBenchmark needs at least one item.")
        self.name = name
        self.items = {str(key): value for key, value in items.items()}
        self.scorer = scorer
        self.targets = {str(key): value for key, value in (targets or {}).items()}
        self.groups_by_item = {str(key): str(value) for key, value in (groups or {}).items()}
        self.generate = dict(generate or {})
        self.version = version
        self.scorer_name = getattr(scorer, "__name__", "score")
        _check_versioned(scorer)
        self._id = content_id({
            "schema": "prompt_benchmark/1", "items": self.items, "targets": self.targets,
            "groups": self.groups_by_item, "generate": self.generate, "version": version, "scorer": scorer,
        })
        self._store: Store | None = None

    @property
    def id(self) -> str:
        """The benchmark's content id."""
        return self._id

    def prepare(self, store: Store) -> None:
        """Write the items to the store."""
        store.put_items(
            ItemRow(self.id, item_id, self.groups_by_item.get(item_id), {
                "id": item_id, "input": prompt, "target": self.targets.get(item_id), "metadata": {},
            })
            for item_id, prompt in self.items.items()
        )
        self._store = store

    def _require_prepared(self) -> Store:
        if self._store is None:
            raise RuntimeError(f"Benchmark {self.name!r} is not prepared; call prepare(store) first.")
        return self._store

    def item_ids(self) -> tuple[str, ...]:
        """The item ids, in order."""
        return tuple(row.item_id for row in self._require_prepared().get_items(self.id))

    def groups(self) -> Mapping[str, str]:
        """Mapping from item id to group id."""
        return {row.item_id: row.group_id for row in self._require_prepared().get_items(self.id) if row.group_id}

    def score(self, item_id: str, completion: str, *, store: Store) -> float:
        """The scorer's value for a completion of an item, through the score cache."""
        prompt = self.items[item_id]
        target = self.targets.get(item_id)
        key = content_id({"scorer": self.scorer, "input": prompt, "completion": completion, "target": target})
        stored = store.get_score(key)
        if stored is not None:
            return stored.value
        value = float(self.scorer(completion, {"input": prompt, "reference": target}))
        return store.put_score(ScoreRow(key, content_id(self.scorer), value, None)).value

    def run(self, steered: "Steered", item_ids: Sequence[str], *, store: Store, replicate: int) -> dict[str, ItemResult]:
        """Generate and score the given items; a failed item is omitted."""
        requests = [Request(item_id, self.items[item_id]) for item_id in item_ids]
        generations = generate_requests(
            steered, requests, store=store, benchmark_generate=self.generate, replicate=replicate,
        )
        results = {}
        for item_id, generation in generations.items():
            try:
                value = self.score(item_id, generation.completion, store=store)
            except Exception:
                logger.warning("Scoring failed for item %r.", item_id, exc_info=True)
                continue
            results[item_id] = ItemResult(
                item_id=item_id, value=value, status="ok", tokens=generation.tokens, completion=generation.completion,
                scores={self.scorer_name: value}, rollout_key=generation.rollout_key, generated=generation.generated,
            )
        return results


def _task_file(task: str) -> Path | None:
    """The local file a task reference names, or None for a registry reference."""
    candidate = Path(task.split("@", 1)[0])
    return candidate if candidate.suffix == ".py" and candidate.is_file() else None


def _jsonable(value: Any) -> Any:
    """A JSON form of an Inspect value (messages, targets, metadata)."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _model_role_identity(roles: Mapping[str, Any]) -> dict[str, str]:
    return {name: model if isinstance(model, str) else str(model) for name, model in roles.items()}


class InspectBenchmark:
    """One Inspect task, run through `inspect_ai.eval`.

    The benchmark's id covers the task reference, the SHA-256 of the task file when the reference
    is a file path (so editing a local task gives a new benchmark), the task arguments, the scorer
    key, the generation arguments, the model roles (by name), the group key, and the installed
    `inspect_ai` and `inspect_evals` versions. `prepare` loads the task and reads its dataset
    without running a model. A sample without an id takes its 1-based position.

    Args:
        task: The task reference (a registry name such as `"inspect_evals/gsm8k"` or a file path,
            optionally `"<file>@<task>"`).
        task_args: Task arguments, e.g. a grader model reference.
        scorer: The score key of an item's value: `"<scorer>"` for a scalar score or
            `"<scorer>/<key>"` for one key of a dict-valued score, converted with Inspect's
            `value_to_float`. None selects the task's first scorer.
        generate: `GenerateConfig` arguments passed to `eval`, over the protocol's defaults.
        model_roles: Inspect model roles (e.g. a grader), never the pipeline under evaluation.
        group_key: The sample metadata key holding each sample's group id, or None.

    Attributes:
        name: The task reference.
    """

    def __init__(
        self,
        task: str,
        *,
        task_args: Mapping[str, Any] | None = None,
        scorer: str | None = None,
        generate: Mapping[str, Any] | None = None,
        model_roles: Mapping[str, Any] | None = None,
        group_key: str | None = None,
    ) -> None:
        self.task = str(task)
        self.name = self.task
        self.task_args = dict(task_args or {})
        self.scorer = scorer
        self.generate = dict(generate or {})
        self.model_roles = dict(model_roles or {})
        self.group_key = group_key
        file = _task_file(self.task)
        self._id = content_id({
            "schema": "inspect_benchmark/1",
            "task": self.task,
            "task_file_sha256": hashlib.sha256(file.read_bytes()).hexdigest() if file is not None else None,
            "task_args": self.task_args,
            "scorer": scorer,
            "generate": self.generate,
            "model_roles": _model_role_identity(self.model_roles),
            "group_key": group_key,
            "versions": {"inspect_ai": _package_version("inspect-ai"), "inspect_evals": _package_version("inspect-evals")},
        })
        self._store: Store | None = None
        self._original_ids: dict[str, Any] = {}

    @property
    def id(self) -> str:
        """The benchmark's content id."""
        return self._id

    def _load_task(self):
        """The task object, loaded from its file or the registry with the task arguments."""
        from inspect_ai._eval.loader import create_file_tasks, load_file_tasks, load_tasks

        file = _task_file(self.task)
        if file is not None:
            name = self.task.partition("@")[2] or None
            if name is not None:
                load_file_tasks(file.absolute())
            tasks = create_file_tasks(file.absolute(), [name] if name else None, dict(self.task_args))
        else:
            tasks = load_tasks([self.task], dict(self.task_args))
        if len(tasks) != 1:
            raise ValueError(f"Task reference {self.task!r} resolves to {len(tasks)} tasks; name one task.")
        return tasks[0]

    def prepare(self, store: Store) -> None:
        """Load the task, read its dataset, and write its items to the store once."""
        if not store.has_items(self.id):
            task = self._load_task()
            rows = []
            for index, sample in enumerate(task.dataset):
                original = sample.id if sample.id is not None else index + 1
                metadata = dict(sample.metadata or {})
                group = metadata.get(self.group_key) if self.group_key is not None else None
                rows.append(ItemRow(self.id, str(original), None if group is None else str(group), {
                    "id": original, "input": _jsonable(sample.input), "target": _jsonable(sample.target),
                    "metadata": _jsonable(metadata),
                }))
            store.put_items(rows)
        self._store = store
        self._original_ids = {row.item_id: row.payload["id"] for row in store.get_items(self.id)}

    def _require_prepared(self) -> Store:
        if self._store is None:
            raise RuntimeError(f"Benchmark {self.name!r} is not prepared; call prepare(store) first.")
        return self._store

    def item_ids(self) -> tuple[str, ...]:
        """The item ids, in dataset order."""
        return tuple(row.item_id for row in self._require_prepared().get_items(self.id))

    def groups(self) -> Mapping[str, str]:
        """Mapping from item id to the sample's `group_key` metadata value."""
        return {row.item_id: row.group_id for row in self._require_prepared().get_items(self.id) if row.group_id}

    def _value(self, scores: Mapping[str, Any], converter) -> float:
        if not scores:
            raise ValueError("The sample has no scores.")
        if self.scorer is None:
            value = next(iter(scores.values()))
        else:
            name, _, key = self.scorer.partition("/")
            value = scores[name]
            if key:
                value = value[key]
        return float(converter(value))

    def run(self, steered: "Steered", item_ids: Sequence[str], *, store: Store, replicate: int) -> dict[str, ItemResult]:
        """Run the task on the given samples through `inspect_ai.eval`; a failed sample is omitted."""
        import inspect_ai
        from inspect_ai.scorer import value_to_float

        from steerability.evaluation.provider import as_inspect_model

        self._require_prepared()
        protocol = steered.protocol
        model = as_inspect_model(
            steered.pipeline, options=protocol.resolved_provider,
            base_seed=derive_seed(protocol.seed, "replicate", replicate), model_name=steered.recipe.id,
        )
        log_dir = store.logs / self.id / steered.resolved_id / f"replicate_{replicate}" / _now().replace(":", "-")
        kwargs: dict[str, Any] = {**protocol.generate_defaults, **self.generate}
        if self.model_roles:
            kwargs["model_roles"] = dict(self.model_roles)
        with _EVAL_LOCK:
            logs = inspect_ai.eval(
                self._load_task(), model=model,
                sample_id=[self._original_ids[item_id] for item_id in item_ids],
                display="none", log_dir=str(log_dir), epochs=1, **kwargs,
            )
        converter = value_to_float()
        results: dict[str, ItemResult] = {}
        for log in logs:
            for sample in log.samples or []:
                if sample.error is not None:
                    continue
                scores = {name: score.value for name, score in (sample.scores or {}).items()}
                try:
                    value = self._value(scores, converter)
                except (KeyError, TypeError, ValueError):
                    logger.warning("Sample %r has no %r score.", sample.id, self.scorer)
                    continue
                tokens = sum(usage.output_tokens for usage in (sample.model_usage or {}).values()) or None
                results[str(sample.id)] = ItemResult(
                    item_id=str(sample.id), value=value, status="ok", tokens=tokens,
                    completion=sample.output.completion if sample.output is not None else "",
                    scores=_jsonable(scores), rollout_key=None, generated=True,
                )
        return results


def _from_evidence(row: EvidenceRow, store: Store, generated: bool) -> ItemResult:
    detail = row.detail or {}
    completion = detail.get("completion")
    if completion is None and row.rollout_key is not None:
        rollout = store.get_rollout(row.rollout_key)
        completion = rollout.completion if rollout is not None else None
    return ItemResult(
        item_id=row.item_id, value=row.value, status=row.status, tokens=row.tokens,
        completion=None if row.status == "error" else completion, scores=detail.get("scores", {}),
        rollout_key=row.rollout_key, generated=generated,
    )


def evaluate_items(
    steered: "Steered",
    benchmark: Benchmark,
    item_ids: Sequence[str],
    *,
    store: Store,
    replicate: int = 0,
    retries: int = 2,
) -> dict[str, ItemResult]:
    """Measure a steered pipeline on items of a benchmark, storing one evidence row per item.

    Stored evidence rows for `(protocol, resolved recipe, benchmark, item, replicate)` are
    returned as they are. The benchmark runs the missing items, and the items it omits (or every
    item, when it raises) are run again up to `retries` more times. Items still missing get an
    error row (`status="error"`, `value=0.0`). `replicate` separates repeated measurements of one
    item: under sampling it selects a different seed, and under greedy decoding it repeats the
    generation in a different batch order.

    Args:
        steered: The steered pipeline.
        benchmark: The benchmark.
        item_ids: The items to measure.
        store: The store.
        replicate: The replicate index.
        retries: Further attempts for omitted items.

    Returns:
        Mapping from item id to its result, in the order of `item_ids`; `generated` is True on the
        rows whose generation this call ran.
    """
    protocol_id = steered.protocol.id
    item_ids = [str(item_id) for item_id in dict.fromkeys(item_ids)]
    stored = store.get_evidence(protocol_id, steered.resolved_id, benchmark.id, item_ids, replicate)
    generated: set[str] = set()
    missing = [item_id for item_id in item_ids if item_id not in stored]
    for attempt in range(1 + retries):
        if not missing:
            break
        try:
            results = benchmark.run(steered, missing, store=store, replicate=replicate)
        except Exception:
            logger.warning("Benchmark %r failed on %d item(s) (attempt %d).", benchmark.name, len(missing),
                           attempt + 1, exc_info=True)
            results = {}
        for item_id, result in results.items():
            if item_id not in missing:
                continue
            detail: dict[str, Any] = {"scores": dict(result.scores)}
            if result.rollout_key is None:
                detail["completion"] = result.completion
            stored[item_id] = store.put_evidence(EvidenceRow(
                protocol_id=protocol_id, resolved_id=steered.resolved_id, benchmark_id=benchmark.id,
                item_id=item_id, replicate=replicate, value=float(result.value), status="ok", tokens=result.tokens,
                rollout_key=result.rollout_key, detail=json.loads(json.dumps(detail, default=str)),
                created_at=_now(),
            ))
            if result.generated:
                generated.add(item_id)
        missing = [item_id for item_id in missing if item_id not in stored]
    for item_id in missing:
        stored[item_id] = store.put_evidence(EvidenceRow(
            protocol_id=protocol_id, resolved_id=steered.resolved_id, benchmark_id=benchmark.id, item_id=item_id,
            replicate=replicate, value=0.0, status="error", tokens=None, rollout_key=None, detail=None,
            created_at=_now(),
        ))
    return {item_id: _from_evidence(stored[item_id], store, item_id in generated) for item_id in item_ids}


__all__ = [
    "Benchmark",
    "Generated",
    "InspectBenchmark",
    "ItemResult",
    "PromptBenchmark",
    "Request",
    "evaluate_items",
    "generate_requests",
]
