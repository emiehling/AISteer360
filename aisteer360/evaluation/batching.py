"""Lock-leader batching collator for the steering-pipeline model provider.

Inspect issues one async request per sample, many concurrently; a `SteeringPipeline` is synchronous
and permits one in-flight generation, but accepts batched calls. The collator turns concurrent
requests into batched `pipeline.generate()` calls with no background task and no timer. One
`anyio.Lock` guards pipeline access: every request enqueues its record and then acquires the lock;
whoever holds it (the leader) takes its record plus the compatible queued records, dispatches one
batched call in a worker thread, and completes every member. A waiter that wakes with its record
already served returns without dispatching. The lock holder is the only caller of
`pipeline.generate`, so the one-in-flight invariant holds by construction.

An uncontended lock acquisition yields once to the event loop, so requests admitted in the same
scheduling window enqueue before the first leader takes its batch, and while a batch is in flight
newly arriving requests accumulate; the in-flight generation is the collation window. With a batch
ceiling of 1 the same code path degenerates to strict serialization.

Records share a dispatch only when the batched call is semantically identical to per-request calls.
The batch key digests the canonicalized call-scoped generation kwargs (seed and stop strings
included), the sorted key set of the record's per-sample runtime kwargs (values may differ per row;
keys may not), and a single-candidate flag; multi-candidate requests always form singleton
dispatches. Per-sample values never enter the key.

A failed multi-row dispatch triggers one poison-isolation pass: the members re-run serially, once,
so each record receives its own result or its own exception. Two consequences follow. First, the
serial pass regenerates the members that would have succeeded in the batch, so under sampling their
outputs differ from what the batch would have produced. Second, a batch that fails while every
member succeeds serially indicates a shape problem rather than a bad sample (the signature of a
row-scoped value whose per-row form is wrong); the collator logs a warning once per distinct
batch-level exception message, naming the batch size, so the misconfiguration surfaces instead of
degrading silently into serial throughput.

Reproducibility contract: which samples are evaluated is fixed by the suite, independent of batch
composition. A seeded dispatch carries `seed_scope` from `ProviderOptions` (default `"dispatch"`): on the
Hugging Face backend the dispatch decodes in one batched pass under one derived seed, so the dispatch is
reproducible as a whole while an individual sample's continuation depends on its dispatch-mates and row
position; under `"item"` scope each row derives its own seed and the session decodes rows one at a time. On
the vLLM backends per-request seeds execute inside one engine batch under either scope. Under the collator a
sample's dispatch membership and row index depend on async arrival timing, so bitwise reproducibility of
stochastic sampling is not preserved under concurrency under either scope; a bitwise-reproducible stochastic
run requires a batch ceiling of 1 and Inspect `max_connections=1`. Greedy decoding sidesteps seed sensitivity,
with the one caveat that padded-batch numerics can differ from single-item numerics on some kernels.
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

from aisteer360.utils.optional import require

require("inspect_ai")  # anyio arrives through the inspect extra
import anyio
import anyio.to_thread

from aisteer360.algorithms.core.identity import canonical_value
from aisteer360.algorithms.core.output import Output

if TYPE_CHECKING:
    from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

logger = logging.getLogger(__name__)


@dataclass(eq=False, slots=True)
class BatchRequest:
    """One admitted generation request, queued for a batched dispatch.

    Attributes:
        prompt: One conversation (a list of chat-message dicts) on the messages path, or one
            rendered string on the text path.
        gen_kwargs: The call-scoped generation kwargs mapped from the request's config.
        per_sample_runtime_kwargs: The request's per-sample runtime kwargs; every key is
            row-scoped by declaration.
        num_choices: Number of candidates requested; values above 1 dispatch as a singleton.
        batch_key: Digest governing which records may share a dispatch.
        done: Event set by the leader once the record holds its result.
        output: The record's `Output` on success, else None.
        error: The record's exception on failure, else None.
    """

    prompt: Any
    gen_kwargs: dict[str, Any]
    per_sample_runtime_kwargs: dict[str, Any]
    num_choices: int
    batch_key: str
    done: anyio.Event = field(default_factory=anyio.Event)
    output: Output | None = None
    error: BaseException | None = None


class LockLeaderCollator:
    """Collate concurrent async requests into batched `pipeline.generate()` calls.

    See the module docstring for the protocol, the batch-key semantics, poison isolation, and the
    reproducibility contract. `admit()` validates and builds a record without enqueueing; the
    async `serve()` enqueues it and runs the leader protocol.

    Args:
        pipeline: The steered pipeline; the lock holder is its only caller.
        max_batch_size: Dispatch ceiling; the caller passes the effective value (already clamped
            to 1 when the pipeline does not support batching).
        prompt_path: `"messages"` to dispatch conversations via `messages=`, `"text"` to dispatch
            rendered strings via `text=`.
        row_scoped_keys: Runtime-kwarg names declared `"row"`-scoped by the pipeline's enabled
            controls; the only names admissible per sample.
        static_runtime_kwargs: Call-scoped runtime kwargs applied to every dispatch,
            shallow-copied and never mutated.
        chat_template_kwargs: Forwarded to `apply_chat_template` on the messages path, or None.
    """

    def __init__(
        self,
        pipeline: "SteeringPipeline",
        *,
        max_batch_size: int,
        prompt_path: Literal["messages", "text"],
        row_scoped_keys: frozenset[str],
        static_runtime_kwargs: Mapping[str, Any],
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._max_batch_size = int(max_batch_size)
        self._prompt_path = prompt_path
        self._row_scoped_keys = frozenset(row_scoped_keys)
        self._static_runtime_kwargs = dict(static_runtime_kwargs)
        self._chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs is not None else None
        self._lock = anyio.Lock()
        self._queue: list[BatchRequest] = []
        self._closed = False
        self._batch_failures_logged: set[str] = set()

    @property
    def closed(self) -> bool:
        """Whether the collator has been closed."""
        return self._closed

    def close(self) -> None:
        """Refuse new admissions; an in-flight batch completes and delivers its results."""
        self._closed = True

    def admit(
        self,
        prompt: Any,
        gen_kwargs: Mapping[str, Any],
        per_sample_runtime_kwargs: Mapping[str, Any],
        num_choices: int,
    ) -> BatchRequest:
        """Validate one request and build its record, without enqueueing it.

        Args:
            prompt: The converted prompt (a conversation or a rendered string).
            gen_kwargs: The mapped call-scoped generation kwargs.
            per_sample_runtime_kwargs: The request's per-sample runtime kwargs.
            num_choices: Number of candidates requested (at least 1).

        Returns:
            The admitted `BatchRequest`.

        Raises:
            RuntimeError: If the collator is closed.
            ValueError: If a per-sample key is also supplied statically, or names a runtime kwarg
                that no enabled control declares `"row"`-scoped.
        """
        if self._closed:
            raise RuntimeError("The steering-pipeline provider is closed; build a new one.")
        for key in per_sample_runtime_kwargs:
            if key in self._static_runtime_kwargs:
                raise ValueError(
                    f"Runtime kwarg {key!r} is supplied both per sample (Sample.metadata) and statically "
                    "(ProviderOptions.runtime_kwargs); a call-scoped scalar and a row-aligned stream are "
                    "incoherent. Remove it from one tier."
                )
            if key not in self._row_scoped_keys:
                raise ValueError(
                    f"Per-sample runtime kwarg {key!r} is not declared 'row'-scoped by any enabled control. "
                    "Pass it as a static kwarg (ProviderOptions.runtime_kwargs), or declare "
                    "'scope': 'row' on the consuming control's RUNTIME_KWARGS_SCHEMA entry."
                )
        return BatchRequest(
            prompt=prompt,
            gen_kwargs=dict(gen_kwargs),
            per_sample_runtime_kwargs=dict(per_sample_runtime_kwargs),
            num_choices=max(1, int(num_choices)),
            batch_key=self._batch_key(gen_kwargs, per_sample_runtime_kwargs, num_choices),
        )

    @staticmethod
    def _batch_key(
        gen_kwargs: Mapping[str, Any],
        per_sample_runtime_kwargs: Mapping[str, Any],
        num_choices: int,
    ) -> str:
        """Digest of the record's dispatch-compatibility identity."""
        payload = json.dumps(
            {
                "gen_kwargs": canonical_value(dict(gen_kwargs)),
                "per_sample_keys": sorted(per_sample_runtime_kwargs),
                "single_candidate": num_choices <= 1,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def serve(self, record: BatchRequest) -> Output:
        """Enqueue one admitted record and run the leader protocol until it holds its result.

        Args:
            record: A record built by `admit()`.

        Returns:
            The record's `Output`.

        Raises:
            Exception: The record's own dispatch exception, re-raised on the requesting task.
        """
        self._queue.append(record)
        try:
            async with self._lock:
                if not record.done.is_set():
                    batch = self._take_batch(record)
                    try:
                        await anyio.to_thread.run_sync(self._dispatch, batch, abandon_on_cancel=False)
                    finally:
                        # anyio events are not thread-safe; set them on the event-loop thread
                        for member in batch:
                            member.done.set()
        finally:
            self._discard(record)
        if record.error is not None:
            raise record.error
        return record.output

    def _take_batch(self, record: BatchRequest) -> list[BatchRequest]:
        """Remove the leader's record plus compatible queued records, in queue order."""
        self._queue.remove(record)
        batch = [record]
        if record.num_choices == 1:
            capacity = self._max_batch_size - 1
            if capacity > 0:
                taken = [queued for queued in self._queue if queued.batch_key == record.batch_key][:capacity]
                for member in taken:
                    self._queue.remove(member)
                batch.extend(taken)
        return batch

    def _discard(self, record: BatchRequest) -> None:
        """Remove a record from the queue if still present (idempotent)."""
        try:
            self._queue.remove(record)
        except ValueError:
            pass

    def _dispatch(self, batch: list[BatchRequest]) -> None:
        """Run one batched pipeline call in the worker thread, filling each record's result slot.

        Never raises; a singleton failure lands on its record, and a multi-row failure triggers
        the serial poison-isolation pass.
        """
        try:
            outputs = self._run_batch(batch)
        except Exception as error:
            if len(batch) == 1:
                batch[0].error = error
                return
            for member in batch:
                try:
                    member.output = self._run_batch([member])[0]
                except Exception as member_error:
                    member.error = member_error
            if all(member.error is None for member in batch):
                message = f"{type(error).__name__}: {error}"
                if message not in self._batch_failures_logged:
                    self._batch_failures_logged.add(message)
                    logger.warning(
                        "A dispatch of %d requests failed (%s) while every member succeeded serially; "
                        "this is the signature of a runtime-kwarg value whose per-row form is wrong at "
                        "batch size > 1. Throughput degrades to serial until the shape is fixed.",
                        len(batch), message,
                    )
            return
        for member, output in zip(batch, outputs):
            member.output = output

    def _run_batch(self, batch: list[BatchRequest]) -> list[Output]:
        """Issue one `pipeline.generate` call for `batch`, returning one `Output` per record."""
        leader = batch[0]
        runtime_kwargs = dict(self._static_runtime_kwargs)
        for key in leader.per_sample_runtime_kwargs:
            runtime_kwargs[key] = [member.per_sample_runtime_kwargs[key] for member in batch]
        gen_kwargs = dict(leader.gen_kwargs)
        if self._chat_template_kwargs is not None:
            gen_kwargs["chat_template_kwargs"] = dict(self._chat_template_kwargs)
        if leader.num_choices > 1:
            prompt_kwargs = (
                {"messages": leader.prompt} if self._prompt_path == "messages" else {"text": leader.prompt}
            )
            output = self._pipeline.generate(
                runtime_kwargs=runtime_kwargs, return_output=True, n=leader.num_choices,
                **prompt_kwargs, **gen_kwargs,
            )
            return [output]
        prompts = [member.prompt for member in batch]
        prompt_kwargs = {"messages": prompts} if self._prompt_path == "messages" else {"text": prompts}
        outputs = self._pipeline.generate(
            runtime_kwargs=runtime_kwargs, return_output=True, **prompt_kwargs, **gen_kwargs,
        )
        return list(outputs)
