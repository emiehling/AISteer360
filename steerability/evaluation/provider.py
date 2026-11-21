"""Inspect AI model provider over a steered `SteeringPipeline`.

`as_inspect_model` wraps a steered pipeline as a generation-only Inspect model. Every request flows
through `pipeline.generate()`: on the messages path prompts enter as structured chat messages, so
the pipeline owns chat templating and message-phase input controls fire exactly as in deployment;
tokenizers without a chat template evaluate through the text path instead. Concurrent Inspect
requests collate into batched pipeline calls through the lock-leader collator in `batching`, or,
in per-request mode (`ProviderOptions.max_concurrency` on a pipeline that supports concurrent
generation), each request is dispatched as its own pipeline call.

The provider is generation-only by design. Logprob-based scoring (including prompt-logprob
perplexity) is structurally unsupported, since a pipeline's decoding-time controls are not
representable in teacher-forced likelihoods; tool calling and multimodal content are unsupported.
Unsupported requests raise at admission with an actionable message.
"""
from steerability.utils.optional import require

require("inspect_ai")
import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal, Mapping

import torch
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)

from steerability.algorithms.core.execution.spec import BackendSpec
from steerability.algorithms.core.output import Output, truncate_at_stop_strings
from steerability.algorithms.core.protocol import ProviderOptions
from steerability.algorithms.core.utils.assembly import resolve_decoding_driver
from steerability.algorithms.core.utils.controls import runtime_kwargs_schema
from steerability.evaluation.batching import LockLeaderCollator, PerRequestDispatcher
from steerability.utils.rendering import has_chat_template, join_message_contents
from steerability.utils.thinking import (
    ThinkingSplit,
    find_subsequence,
    resolve_split_mode,
    split_thinking,
    split_thinking_ids,
)

if TYPE_CHECKING:
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

logger = logging.getLogger(__name__)

# every GenerateConfig field belongs to exactly one class; a unit test pins the coverage so a new
# Inspect field fails at test time instead of being silently dropped
MAPPED_FIELDS: frozenset[str] = frozenset({
    "max_tokens", "stop_seqs", "temperature", "top_p", "top_k", "seed", "num_choices", "extra_body",
})
UPSTREAM_FIELDS: frozenset[str] = frozenset({
    "system_message", "max_retries", "timeout", "attempt_timeout", "stream_idle_timeout", "max_connections",
    "adaptive_connections", "fallback_models", "cache", "cache_prompt", "batch",
    "parallel_tool_calls", "internal_tools", "max_tool_output",
})
POLICY_FIELDS: frozenset[str] = frozenset({
    "best_of", "frequency_penalty", "presence_penalty", "logit_bias", "response_schema",
    "reasoning_effort", "reasoning_tokens", "reasoning_summary", "reasoning_history",
    "reasoning_mode", "verbosity", "effort", "extra_headers", "modalities",
})
REFUSED_FIELDS: frozenset[str] = frozenset({"logprobs", "top_logprobs", "prompt_logprobs"})

RUNTIME_KWARGS_EXTRA_BODY_KEY = "runtime_kwargs"

_STOP_REASON_MAP: dict[str | None, str] = {"length": "max_tokens", "stop": "stop", "eos": "stop"}


def convert_input(input: list) -> list[dict[str, str]]:
    """Convert Inspect chat messages to the `[{"role", "content"}, ...]` form the pipeline accepts.

    Reasoning content parts of earlier assistant turns are dropped.

    Args:
        input: The Inspect chat messages.

    Returns:
        The messages as role and content mappings.

    Raises:
        NotImplementedError: On tool messages, assistant tool calls, or non-text content parts.
    """
    messages: list[dict[str, str]] = []
    for message in input:
        if isinstance(message, ChatMessageTool) or message.role == "tool":
            raise NotImplementedError(
                "The conversation carries a tool message, but steering pipelines have no tool-use "
                "convention; agentic evaluation is unsupported. Choose a non-agentic task."
            )
        if message.role == "assistant" and getattr(message, "tool_calls", None):
            raise NotImplementedError(
                "An assistant message carries tool calls, but steering pipelines have no tool-use "
                "convention; agentic evaluation is unsupported. Choose a non-agentic task."
            )
        content = message.content
        if isinstance(content, str):
            text = content
        else:
            parts: list[str] = []
            for part in content:
                if isinstance(part, ContentText):
                    parts.append(part.text)
                elif isinstance(part, ContentReasoning):
                    logger.debug("Dropping a ContentReasoning part from the conversation history.")
                else:
                    raise NotImplementedError(
                        f"The conversation carries {type(part).__name__} content, which the pipeline's "
                        "text-only prompt surface cannot represent. Choose a text-only task."
                    )
            text = "".join(parts)
        messages.append({"role": message.role, "content": text})
    return messages


def _unsupported_param(name: str, value: Any, options: ProviderOptions, warned: set[str]) -> None:
    """Apply the unsupported-parameter policy for one non-None config field."""
    message = (
        f"GenerateConfig.{name}={value!r} has no mapping onto the steering-pipeline surface; "
        "honoring it silently is not possible and dropping it would change decoding semantics. "
        "Remove the parameter, or set ProviderOptions(on_unsupported_param='warn') to ignore it."
    )
    if options.on_unsupported_param == "raise":
        raise ValueError(message)
    if name not in warned:
        warned.add(name)
        warnings.warn(message, UserWarning)


def map_generate_config(
    config: GenerateConfig,
    options: ProviderOptions,
    *,
    base_seed: int | None = None,
    warned: set[str] | None = None,
) -> tuple[dict, dict, int]:
    """Classify and map one request's `GenerateConfig` onto pipeline generation kwargs.

    `temperature=0` maps to greedy decoding and drops any seed; a positive temperature samples
    with the config's seed, else `base_seed`, under `options.seed_scope`. An unset temperature
    leaves the backend's default sampling posture.

    Args:
        config: The request's config.
        options: The provider options, whose `on_unsupported_param` policy applies.
        base_seed: The seed of a sampled request whose config sets none.
        warned: The parameter names already warned about under the `"warn"` policy, updated in
            place; None warns every time.

    Returns:
        Tuple of (call-scoped gen kwargs, per-sample runtime kwargs, num_choices).

    Raises:
        NotImplementedError: If a logprob parameter is set.
        ValueError: If an unsupported parameter is set under the `"raise"` policy, or the
            reserved `extra_body` key is not a mapping.
    """
    warned = set() if warned is None else warned
    for name in REFUSED_FIELDS:
        if getattr(config, name) is not None:
            raise NotImplementedError(
                f"GenerateConfig.{name} requests log probabilities, but steering pipelines are "
                "evaluated generation-only: decoding-time controls are not representable in "
                "teacher-forced likelihoods. Use generation-based scorers."
            )
    extra_body = dict(config.extra_body or {})
    per_sample = extra_body.pop(RUNTIME_KWARGS_EXTRA_BODY_KEY, None) or {}
    if not isinstance(per_sample, Mapping):
        raise ValueError(
            f"extra_body[{RUNTIME_KWARGS_EXTRA_BODY_KEY!r}] must be a mapping of per-sample runtime "
            f"kwargs; got {type(per_sample).__name__}."
        )
    for name in POLICY_FIELDS:
        if getattr(config, name) is not None:
            _unsupported_param(name, getattr(config, name), options, warned)
    for key, value in extra_body.items():
        if value is not None:
            _unsupported_param(f"extra_body[{key!r}]", value, options, warned)

    gen_kwargs: dict[str, Any] = {}
    if config.max_tokens is not None:
        gen_kwargs["max_new_tokens"] = int(config.max_tokens)
    if config.stop_seqs:
        gen_kwargs["stop_strings"] = tuple(config.stop_seqs)

    temperature = config.temperature
    if temperature is None:
        if config.top_p is not None or config.top_k is not None or config.seed is not None or base_seed is not None:
            logger.debug(
                "temperature is unset, so the backend's default sampling posture applies; "
                "top_p/top_k/seed are not attached. Set temperature explicitly for a known posture."
            )
    elif temperature == 0:
        gen_kwargs["do_sample"] = False
        if config.top_p is not None or config.top_k is not None or config.seed is not None:
            logger.debug("Greedy decoding (temperature=0) drops top_p, top_k, and any seed.")
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        if config.top_p is not None:
            gen_kwargs["top_p"] = float(config.top_p)
        if config.top_k is not None:
            gen_kwargs["top_k"] = int(config.top_k)
        seed = config.seed if config.seed is not None else base_seed
        if seed is not None:
            gen_kwargs["seed"] = int(seed)
            gen_kwargs["seed_scope"] = options.seed_scope

    num_choices = int(config.num_choices) if config.num_choices is not None else 1
    return gen_kwargs, dict(per_sample), num_choices


def output_token_count(output: Output, pad_token_id: int | None) -> int:
    """The tokens generated for an `Output`: `generated_tokens` when set (every rollout a decoding
    driver generated), else the returned continuation's non-pad tokens."""
    if output.generated_tokens is not None:
        return int(output.generated_tokens)
    return _count_non_pad(output.output_ids, pad_token_id)


class CompletionAssembler:
    """Turn a pipeline `Output` into the text a scorer sees, row by row.

    Token ids are never modified. When reasoning tags are configured, each row is split into a
    reasoning part and an answer part, and only the answer is stop-string truncated. The split
    mode is resolved once against the tokenizer (`ProviderOptions.reasoning_split`):

        - `"text"`: the row is decoded, truncated at the first stop-string occurrence, then
          substring split. A stop string inside the reasoning cuts the text mid-thinking and the
          answer is empty.
        - `"tokens"`: the row ids are split, each segment is decoded, then the answer segment
          (only) is truncated at the first stop-string occurrence. The reasoning segment is kept
          verbatim.

    An opened but unclosed reasoning segment yields an empty answer with everything as
    reasoning, and one warning per output names the count of such rows.

    Args:
        tokenizer: The pipeline tokenizer.
        options: The provider options (reasoning tags and split settings).
    """

    def __init__(self, tokenizer, options: ProviderOptions) -> None:
        self._tokenizer = tokenizer
        self._options = options
        self._split_mode: Literal["text", "tokens"] | None = None
        self._close_ids: list[int] = []
        if options.reasoning_tags is not None:
            if options.reasoning_split == "auto":
                self._split_mode = resolve_split_mode(tokenizer, options.reasoning_tags)
            else:
                self._split_mode = options.reasoning_split
            if self._split_mode == "tokens":
                self._close_ids = tokenizer.encode(options.reasoning_tags[1], add_special_tokens=False)

    def splits(self, output: Output, stop_strings: tuple[str, ...] = ()) -> list[ThinkingSplit]:
        """One `ThinkingSplit` per row of `output.output_ids`; without reasoning tags the whole
        stop-truncated text is the answer."""
        texts = output.decode(self._tokenizer)
        result: list[ThinkingSplit] = []
        unclosed = 0
        for row, text in enumerate(texts):
            if self._options.reasoning_tags is None:
                answer = truncate_at_stop_strings(text, stop_strings) if stop_strings else text
                result.append(ThinkingSplit(thinking=None, answer=answer))
                continue
            split, closed = self._split_row(text, output.output_ids[row], stop_strings)
            if not closed:
                unclosed += 1
            result.append(split)
        if unclosed:
            logger.warning(
                "%d generation(s) opened a thinking segment without closing it (budget spent thinking); "
                "their answer segment is empty.",
                unclosed,
            )
        return result

    def completions(self, output: Output, stop_strings: tuple[str, ...] = ()) -> list[str]:
        """The answer text of each row, which is what a scorer sees."""
        return [split.answer for split in self.splits(output, stop_strings)]

    def _split_row(self, text: str, row_ids: torch.Tensor, stop_strings: tuple[str, ...]) -> tuple[ThinkingSplit, bool]:
        tags = self._options.reasoning_tags
        opened = self._options.reasoning_opened_at_start
        open_tag, close_tag = tags
        if self._split_mode == "tokens":
            ids = row_ids.tolist()
            split = split_thinking_ids(ids, self._tokenizer, tags, opened_at_start=opened)
            closed = split.thinking is None or find_subsequence(ids, self._close_ids) != -1
            answer = truncate_at_stop_strings(split.answer, stop_strings) if stop_strings else split.answer
            return ThinkingSplit(thinking=split.thinking, answer=answer), closed
        if stop_strings:
            text = truncate_at_stop_strings(text, stop_strings)
        closed = close_tag in text or not (open_tag in text or opened)
        return split_thinking(text, tags, opened_at_start=opened), closed


class SteeringPipelineModelAPI(ModelAPI):
    """Generation-only Inspect `ModelAPI` over an in-process steered `SteeringPipeline`.

    Constructed through `as_inspect_model`; the registry call convention is accepted so
    `ModelName` resolution works, but the provider cannot be built from a model name string.

    Args:
        model_name: Bare model name; Inspect renders the model as `steerability/<model_name>`.
        base_url: Accepted for the registry call convention; unused.
        api_key: Accepted for the registry call convention; unused.
        api_key_vars: Accepted for the registry call convention; unused.
        config: Accepted for the registry call convention; per-request configs arrive at
            `generate()`.
        pipeline: The steered `SteeringPipeline` to serve.
        options: Provider options; defaults to `ProviderOptions()`.
        base_seed: Attached to sampling dispatches whose `GenerateConfig` carries no seed.
        **model_args: Accepted for the registry call convention; unused.

    Raises:
        ValueError: If `pipeline` is None (the provider wraps an in-process pipeline and cannot
            be built from a model name; construct it through `as_inspect_model`), or the pipeline
            is not steered.
        TypeError: If `chat_template_kwargs` is set while the tokenizer has no chat template
            (the text path cannot take it).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        *,
        pipeline: "SteeringPipeline | None" = None,
        options: ProviderOptions | None = None,
        base_seed: int | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, config=GenerateConfig())
        if pipeline is None:
            raise ValueError(
                "The 'steerability' provider wraps an in-process SteeringPipeline and cannot be built from "
                "a model name. Construct it through steerability.evaluation.provider.as_inspect_model."
            )
        if not pipeline._is_steered:
            raise ValueError(
                "The pipeline is not steered. Call pipeline.steer() before as_inspect_model(pipeline)."
            )
        self._pipeline = pipeline
        self._options = options if options is not None else ProviderOptions()
        self._base_seed = base_seed
        self._warned_params: set[str] = set()

        self._prompt_path: Literal["messages", "text"] = (
            "messages" if has_chat_template(pipeline.tokenizer) else "text"
        )
        if self._prompt_path == "text":
            if self._options.chat_template_kwargs is not None:
                raise TypeError(
                    "chat_template_kwargs was set but the tokenizer has no chat template; the text "
                    "path cannot take it. Remove the option or use a chat-templated tokenizer."
                )
            warnings.warn(
                "The pipeline tokenizer has no chat template, so prompts are rendered to plain text "
                "and adapt_messages does not fire on this provider (token-level adapt still does).",
                UserWarning,
            )

        self._assembler = CompletionAssembler(pipeline.tokenizer, self._options)

        effective_max_batch = self._options.max_batch_size if pipeline.supports_batching else 1
        if effective_max_batch != self._options.max_batch_size:
            logger.info(
                "Pipeline does not support batching; clamping max_batch_size from %d to 1.",
                self._options.max_batch_size,
            )
        self._effective_max_batch = effective_max_batch

        self._per_request = self._options.max_concurrency is not None and pipeline.supports_concurrent_generate
        if self._options.max_concurrency is not None and not self._per_request:
            logger.info(
                "max_concurrency=%d is set, but the pipeline does not support concurrent generation (%s); "
                "requests are dispatched in batching mode.",
                self._options.max_concurrency, _concurrency_blocker(pipeline),
            )

        schema = runtime_kwargs_schema(pipeline.controls)
        declared_scopes = {name: entry["scope"] for name, entry in schema.items()}
        common: dict[str, Any] = dict(
            prompt_path=self._prompt_path,
            declared_scopes=declared_scopes,
            static_runtime_kwargs=self._options.runtime_kwargs,
            chat_template_kwargs=self._options.chat_template_kwargs,
        )
        self._collator: LockLeaderCollator | PerRequestDispatcher
        if self._per_request:
            self._collator = PerRequestDispatcher(
                pipeline, max_concurrency=int(self._options.max_concurrency), **common,
            )
        else:
            self._collator = LockLeaderCollator(pipeline, max_batch_size=effective_max_batch, **common)

    @property
    def prompt_path(self) -> Literal["messages", "text"]:
        """`"messages"` when the tokenizer has a chat template, `"text"` otherwise."""
        return self._prompt_path

    @property
    def dispatch_mode(self) -> Literal["batched", "per_request"]:
        """`"per_request"` when each request is dispatched as its own pipeline call, `"batched"` when
        the lock-leader collator combines requests into batched calls."""
        return "per_request" if self._per_request else "batched"

    @property
    def effective_max_batch(self) -> int:
        """The collator's dispatch ceiling after the batching clamp."""
        return self._effective_max_batch

    @property
    def inert_runtime_kwargs(self) -> frozenset[str]:
        """Runtime kwargs delivered on either tier that no enabled control of the pipeline declares."""
        return self._collator.inert_runtime_kwargs

    async def generate(
        self,
        input: list[ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool],
        tools: list,
        tool_choice: Any,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Serve one Inspect request through the collated pipeline.

        Raises:
            NotImplementedError: If the request supplies tools, tool messages, multimodal content,
                or logprob parameters.
            ValueError: If a `GenerateConfig` parameter cannot be honored (under the `"raise"`
                policy), or a per-sample runtime kwarg is also supplied statically or is declared
                `"call"`-scoped.
            RuntimeError: If the provider is closed.
        """
        if tools:
            raise NotImplementedError(
                "This task supplies tools, but steering pipelines have no tool-use convention; "
                "agentic evaluation is unsupported. Choose a non-agentic task."
            )
        messages = convert_input(input)
        gen_kwargs, per_sample_runtime_kwargs, num_choices = map_generate_config(
            config, self._options, base_seed=self._base_seed, warned=self._warned_params,
        )
        prompt: Any = messages
        if self._prompt_path == "text":
            # the construction warning announces the text path once per provider
            prompt = join_message_contents(messages)
        record = self._collator.admit(prompt, gen_kwargs, per_sample_runtime_kwargs, num_choices)
        output = await self._collator.serve(record)
        return self._assemble_model_output(output, stop_strings=gen_kwargs.get("stop_strings", ()))

    def _assemble_model_output(self, output: Output, *, stop_strings: tuple[str, ...]) -> ModelOutput:
        """Map one pipeline `Output` (one row per candidate) onto an Inspect `ModelOutput`.

        Each row's reasoning and answer come from the provider's `CompletionAssembler`. Usage
        counts the prompt's non-pad tokens as input tokens and `Output.generated_tokens` (every
        rollout a decoding driver generated) as output tokens when set, else the returned
        continuation's non-pad tokens.
        """
        tokenizer = self._pipeline.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        reasons = output.finish_reasons
        choices: list[ChatCompletionChoice] = []
        for row, split in enumerate(self._assembler.splits(output, stop_strings)):
            if split.thinking is not None:
                content: str | list = [ContentReasoning(reasoning=split.thinking), ContentText(text=split.answer)]
            else:
                content = split.answer
            choices.append(ChatCompletionChoice(
                message=ChatMessageAssistant(content=content, model=self.model_name, source="generate"),
                stop_reason=_STOP_REASON_MAP.get(reasons[row], "unknown"),
            ))
        input_tokens = _count_non_pad(output.adapted_input_ids, pad_token_id)
        returned_output_tokens = _count_non_pad(output.output_ids, pad_token_id)
        output_tokens = output_token_count(output, pad_token_id)
        usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return ModelOutput(
            model=self.model_name,
            choices=choices,
            usage=usage,
            metadata={"returned_output_tokens": returned_output_tokens},
        )

    def max_tokens(self) -> int | None:
        """Default `max_tokens` filled by Inspect when a request sets none."""
        return self._options.default_max_tokens

    def max_connections(self) -> int:
        """Advertised concurrency: `max_concurrency` in per-request mode, else the effective batch
        ceiling so batches fill."""
        if self._per_request:
            return int(self._options.max_concurrency)
        return self._effective_max_batch

    def connection_key(self) -> str:
        """Each provider instance owns its connection pool."""
        return f"steerability:{id(self)}"

    def should_retry(self, ex: Exception) -> bool:
        """Provider failures are deterministic, never rate limits."""
        return False

    def is_auth_failure(self, ex: Exception) -> bool:
        """The provider performs no authentication."""
        return False

    def tools_required(self) -> bool:
        """The provider never requires tool definitions."""
        return False

    def close(self) -> None:
        """Close the collator; further requests raise at admission."""
        self._collator.close()

    async def aclose(self) -> None:
        """Close the collator; further requests raise at admission."""
        self._collator.close()


@modelapi(name="steerability")
def steerability_provider() -> type[SteeringPipelineModelAPI]:
    """The registered provider behind `steerability/<model_name>`."""
    return SteeringPipelineModelAPI


def _concurrency_blocker(pipeline: "SteeringPipeline") -> str:
    """The condition that keeps `pipeline` from concurrent generation, for the fallback log."""
    driver = resolve_decoding_driver(getattr(pipeline, "output_controls", ()))
    if driver is not None:
        return f"decoding driver {type(driver).__name__} is enabled"
    backend = getattr(pipeline, "backend", None)
    kind = backend.kind if isinstance(backend, BackendSpec) else (backend or "huggingface")
    return f"backend kind {kind!r} runs in process"


def _count_non_pad(ids: torch.Tensor | None, pad_token_id: int | None) -> int:
    """Count non-pad positions in a token-id tensor (best effort where pad equals EOS)."""
    if ids is None:
        return 0
    if pad_token_id is None:
        return int(ids.numel())
    return int((ids != pad_token_id).sum().item())


def as_inspect_model(
    pipeline: "SteeringPipeline",
    *,
    options: ProviderOptions | None = None,
    base_seed: int | None = None,
    model_name: str = "steering-pipeline",
) -> Model:
    """Wrap a steered `SteeringPipeline` as an Inspect `Model`.

    The pipeline is an in-process object, so the provider is built with the pipeline in hand
    rather than resolved from a model-name string. Construction goes through the registered
    `steerability` provider factory, which attaches the registry info the `Model` reads to
    render `steerability/<model_name>`.

    Args:
        pipeline: The steered pipeline to serve.
        options: Provider options; defaults to `ProviderOptions()`.
        base_seed: Attached to sampling dispatches whose `GenerateConfig` carries no seed.
        model_name: Bare model name for logs and rendering.

    Returns:
        An `inspect_ai.model.Model` usable with `eval` and `eval_set`.

    Raises:
        ValueError: If the pipeline is not steered.
        TypeError: If `options.chat_template_kwargs` is set while the tokenizer has no chat
            template.

    Warns:
        UserWarning: If the tokenizer has no chat template, so prompts render to plain text and
            `adapt_messages` does not fire.
    """
    api = steerability_provider(
        model_name, pipeline=pipeline, options=options, base_seed=base_seed,
    )
    return Model(api, GenerateConfig())
