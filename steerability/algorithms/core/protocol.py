"""The measurement context of steered generation: base model, backend, decoding, and software versions.

A `Protocol` states everything outside a recipe that a measurement depends on. Its `id` scopes
every stored rollout and evidence row, and its `fit_context_id` scopes every cached fit. Both are
computed from declared fields only, so they are known before a model loads. `ProtocolFacts` are the
facts observed once a model is loaded (fingerprints, dtype, resolved commit); they are recorded on
first load and compared on later loads. `ProviderOptions` are the options of the evaluation
provider, which a protocol records as one of its fields.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from steerability.algorithms.core.execution.spec import KNOWN_BACKEND_KINDS, BackendSpec
from steerability.algorithms.core.identity import canonical_json, json_id
from steerability.utils.thinking import DEFAULT_THINK_TAGS

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

PROTOCOL_SCHEMA = "protocol/1"

# backend options that contain credentials; they never enter an identity
_CREDENTIAL_OPTIONS = ("api_key",)
_CREDENTIAL_SUFFIXES = ("_token", "_key")


class ProtocolMismatchError(RuntimeError):
    """Observed facts of a protocol conflict with the facts recorded for it earlier."""


def package_versions() -> dict[str, str]:
    """The versions of the toolkit and of the packages that determine its numerics.

    Returns:
        Mapping with `"steerability"`, `"transformers"`, `"torch"`, and `"python"`.
    """
    import torch
    import transformers

    import steerability

    return {
        "steerability": steerability.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def _is_credential(name: Any) -> bool:
    return isinstance(name, str) and (name in _CREDENTIAL_OPTIONS or name.endswith(_CREDENTIAL_SUFFIXES))


def _without_credentials(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_credentials(item) for key, item in value.items() if not _is_credential(key)}
    if isinstance(value, list):
        return [_without_credentials(item) for item in value]
    return value


def _backend_identity(spec: BackendSpec | None) -> dict | None:
    if spec is None:
        return None
    return {"kind": spec.kind, "model": spec.model, "options": _without_credentials(spec.options_dict())}


@dataclass(frozen=True)
class ProtocolFacts:
    """Facts about a protocol observed once its model is loaded.

    Attributes:
        model_fingerprint: The toolkit fingerprint of the loaded weights.
        tokenizer_fingerprint: A digest of the tokenizer's vocabulary and special tokens.
        chat_template_sha256: The SHA-256 of the tokenizer's chat template, or None when it has
            none.
        torch_dtype: The dtype of the loaded weights, without the `torch.` prefix.
        model_commit: The resolved Hub commit of the weights, when the config records one.
    """

    model_fingerprint: str | None = None
    tokenizer_fingerprint: str | None = None
    chat_template_sha256: str | None = None
    torch_dtype: str | None = None
    model_commit: str | None = None

    @classmethod
    def observe(cls, model: PreTrainedModel | None, tokenizer: PreTrainedTokenizerBase | None) -> ProtocolFacts:
        """The facts of a loaded model and tokenizer; each fact that cannot be read is None.

        Args:
            model: The loaded model, or None on a backend without one.
            tokenizer: The tokenizer, or None.

        Returns:
            The observed facts.
        """
        model_fingerprint = torch_dtype = model_commit = None
        if model is not None:
            from steerability.algorithms.core.internals.fingerprint import model_fingerprint as fingerprint_of

            try:
                model_fingerprint = fingerprint_of(model)
            except Exception:
                model_fingerprint = None
            dtype = getattr(model, "dtype", None)
            torch_dtype = str(dtype).removeprefix("torch.") if dtype is not None else None
            model_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
        tokenizer_fingerprint = template_digest = None
        if tokenizer is not None:
            try:
                vocabulary = sorted(tokenizer.get_vocab().items())
                special = sorted(getattr(tokenizer, "all_special_tokens", []) or [])
                payload = json.dumps({"vocab": vocabulary, "special": special}, separators=(",", ":"))
                tokenizer_fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            except Exception:
                tokenizer_fingerprint = None
            template = getattr(tokenizer, "chat_template", None)
            if isinstance(template, str) and template:
                template_digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        return cls(
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            chat_template_sha256=template_digest,
            torch_dtype=torch_dtype,
            model_commit=model_commit if isinstance(model_commit, str) else None,
        )

    def to_dict(self) -> dict[str, str | None]:
        """The facts as a plain mapping."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping | None) -> ProtocolFacts:
        """Facts from `to_dict` output (None gives no facts)."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in (data or {}).items() if key in known})

    def conflicts(self, other: ProtocolFacts) -> list[str]:
        """The facts that both record and that differ, as `"name: recorded != observed"` lines."""
        lines = []
        for f in dataclasses.fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if mine is not None and theirs is not None and mine != theirs:
                lines.append(f"{f.name}: {mine!r} != {theirs!r}")
        return lines

    def merged(self, other: ProtocolFacts) -> ProtocolFacts:
        """These facts with every missing fact filled from `other`.

        Raises:
            ProtocolMismatchError: If a fact recorded on both sides differs.
        """
        conflicts = self.conflicts(other)
        if conflicts:
            raise ProtocolMismatchError(
                "The loaded model does not match the facts recorded for this protocol: " + "; ".join(conflicts) + "."
            )
        return ProtocolFacts(**{
            f.name: getattr(self, f.name) if getattr(self, f.name) is not None else getattr(other, f.name)
            for f in dataclasses.fields(self)
        })


@dataclass(frozen=True, slots=True)
class ProviderOptions:
    """Options for the Inspect model provider that serves a steering pipeline.

    A `Protocol` records these options (`Protocol.provider`), and every benchmark run under the
    protocol passes them to the provider and to direct generation.

    Attributes:
        runtime_kwargs: Static runtime kwargs applied to every dispatch. The value of a `"call"`-scoped
            key is passed through unchanged. The value of a `"row"`-scoped key is one row's value, in
            the per-row form the consuming control expects, and it is broadcast to every row. A key that
            no enabled control declares is ignored (and logged once). A key that is also supplied per
            sample raises `ValueError` when the request is admitted. Static kwargs suit catalog tasks
            whose datasets have no steering columns, and kwargs that belong to the pipeline rather than
            to the sample.
        chat_template_kwargs: Keyword arguments forwarded to `apply_chat_template` on the messages
            path. Must be None when the tokenizer has no chat template, since the provider raises
            `TypeError` otherwise.
        max_batch_size: Maximum number of requests the collator combines into one dispatch. The
            provider uses 1 when the pipeline does not support batching.
        default_max_tokens: The value `ModelAPI.max_tokens()` returns, which Inspect uses when a
            request sets no `max_tokens`.
        reasoning_tags: The `(open_tag, close_tag)` pair used to split each generation into a
            reasoning part and an answer part, which lets scorers grade only the answer. Defaults to
            `("<think>", "</think>")`. None disables the split.
        reasoning_opened_at_start: Whether the chat template's generation prompt already opens the
            reasoning channel (some thinking-mode templates end the generation prompt with the open
            tag). A continuation that contains the close tag is split the same way under either
            setting. The setting affects only a continuation that contains neither tag (reasoning
            truncated before the close tag). Such a continuation is treated as unclosed reasoning when
            True and as an answer when False.
        reasoning_split: How the split finds the tags, resolved once against the pipeline tokenizer.
            `"text"` splits the decoded continuation by substring. `"tokens"` splits the continuation
            ids and decodes each segment. `"auto"` (default) picks `"text"` when decoding with
            `skip_special_tokens=True` keeps both tags, and `"tokens"` otherwise. Splitting on the ids
            keeps tags that are special tokens from being removed before the split.
        on_unsupported_param: `"raise"` (default) rejects a request with `ValueError` when it sets a
            `GenerateConfig` parameter the pipeline cannot honor. `"warn"` emits one `UserWarning` per
            parameter per provider and ignores the parameter. Log-probability parameters raise
            `NotImplementedError` under either setting.
        seed_scope: How the seed of a seeded sampling dispatch applies to its rows, forwarded to
            `pipeline.generate()`. The seed is the request's seed, or `base_seed` when the request sets
            none. With `"dispatch"` (default), the Hugging Face backend decodes the dispatch in one
            batched pass under one derived seed, and the batch is reproducible as a whole. Each
            `pipeline.generate()` call opens a new session on that backend. Every dispatch with the
            same seed therefore starts sampling from the same random state (common random numbers
            across prompts). With `"item"`, each row derives its own seed and the rows are decoded one
            at a time. The collator groups requests by arrival time, which makes per-row
            reproducibility unattainable under either setting. `"item"` therefore only makes decoding
            serial. The setting has no effect on the vLLM backends or in per-request mode.
        max_concurrency: Selects per-request mode when set (at least 1). When the pipeline's
            `supports_concurrent_generate` is True, each request is dispatched at once as its own
            single-prompt `pipeline.generate()` call in a worker thread, with at most
            `max_concurrency` calls in flight, and `max_connections()` returns `max_concurrency`.
            A seeded request then samples under a seed derived from the base seed and the request
            (its prompt and per-sample runtime kwargs), and a cancelled request aborts its
            generation. When the pipeline does not support concurrent generation, the provider
            logs at info level and keeps batching mode. None (default) selects batching mode.
    """

    runtime_kwargs: Mapping[str, Any] = field(default_factory=dict)
    chat_template_kwargs: Mapping[str, Any] | None = None
    max_batch_size: int = 8
    default_max_tokens: int = 1024
    reasoning_tags: tuple[str, str] | None = DEFAULT_THINK_TAGS
    reasoning_opened_at_start: bool = False
    reasoning_split: Literal["auto", "text", "tokens"] = "auto"
    on_unsupported_param: Literal["raise", "warn"] = "raise"
    seed_scope: Literal["item", "dispatch"] = "dispatch"
    max_concurrency: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_kwargs, Mapping):
            raise TypeError(f"runtime_kwargs must be a mapping; got {type(self.runtime_kwargs).__name__}.")
        if self.chat_template_kwargs is not None and not isinstance(self.chat_template_kwargs, Mapping):
            raise TypeError(
                f"chat_template_kwargs must be a mapping or None; got {type(self.chat_template_kwargs).__name__}."
            )
        if int(self.max_batch_size) < 1:
            raise ValueError(f"max_batch_size must be >= 1; got {self.max_batch_size}.")
        if int(self.default_max_tokens) < 1:
            raise ValueError(f"default_max_tokens must be >= 1; got {self.default_max_tokens}.")
        if self.reasoning_tags is not None:
            open_tag, close_tag = self.reasoning_tags
            if not (isinstance(open_tag, str) and open_tag and isinstance(close_tag, str) and close_tag):
                raise ValueError("reasoning_tags must be a pair of non-empty strings, or None.")
        if self.reasoning_split not in ("auto", "text", "tokens"):
            raise ValueError(
                f"reasoning_split must be 'auto', 'text', or 'tokens'; got {self.reasoning_split!r}."
            )
        if self.on_unsupported_param not in ("raise", "warn"):
            raise ValueError(f"on_unsupported_param must be 'raise' or 'warn'; got {self.on_unsupported_param!r}.")
        if self.seed_scope not in ("item", "dispatch"):
            raise ValueError(f"seed_scope must be 'item' or 'dispatch'; got {self.seed_scope!r}.")
        if self.max_concurrency is not None and int(self.max_concurrency) < 1:
            raise ValueError(f"max_concurrency must be >= 1 or None; got {self.max_concurrency}.")


@dataclass(frozen=True)
class Protocol:
    """The measurement context of steered generation.

    The fields determine the id (`id`) that scopes stored rollouts and evidence, except
    `provider.max_concurrency`, which changes throughput and not results. Credentials in backend
    options (`api_key`, and options whose name ends in `_token` or `_key`) are never part of the
    identity; a backend reads them from its spec or the environment. The fit context
    (`fit_context`, `fit_context_id`) is the part a fit can depend on, which excludes the provider,
    the decoding defaults, and the seed, so a change of decoding refits nothing.

    Attributes:
        model: Hub id or local path of the base model.
        model_revision: The Hub revision of the weights, or None.
        tokenizer: The tokenizer reference, or None for the model's own tokenizer.
        hf_model_kwargs: Load-time keyword arguments for in-process loads.
        backend: The backend spec, or None for the in-process Hugging Face backend.
        fit: The fit venue policy, `"auto"` or `"in_process"`.
        provider: The evaluation provider options, or None for `ProviderOptions()`.
        generate_defaults: Generation arguments applied under every benchmark's own.
        seed: The base seed of every derived sampling seed.
        versions: The software versions (`package_versions()`); `create` fills them.
    """

    model: str
    model_revision: str | None = None
    tokenizer: str | None = None
    hf_model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    backend: BackendSpec | None = None
    fit: Literal["auto", "in_process"] = "auto"
    provider: ProviderOptions | None = None
    generate_defaults: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError(f"model must be a non-empty string; got {self.model!r}.")
        if self.fit not in ("auto", "in_process"):
            raise ValueError(f"fit must be 'auto' or 'in_process'; got {self.fit!r}.")
        if self.backend is not None and not isinstance(self.backend, BackendSpec):
            raise TypeError(f"backend must be a BackendSpec or None; got {type(self.backend).__name__}.")
        if "revision" in self.hf_model_kwargs:
            raise ValueError("Give the weights revision as model_revision=, not inside hf_model_kwargs.")
        object.__setattr__(self, "hf_model_kwargs", dict(self.hf_model_kwargs))
        object.__setattr__(self, "generate_defaults", dict(self.generate_defaults))
        object.__setattr__(self, "versions", dict(self.versions))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "_identity", None)

    @classmethod
    def create(cls, model: str, **fields: Any) -> Protocol:
        """A protocol for `model` with `versions` filled from the installed packages.

        Args:
            model: Hub id or local path of the base model.
            **fields: The other fields.

        Returns:
            The protocol.
        """
        fields.setdefault("versions", package_versions())
        return cls(model=str(model), **fields)

    @classmethod
    def from_pipeline(cls, pipeline: SteeringPipeline, *, model: str | None = None) -> Protocol:
        """A protocol built from a pipeline's own fields, for callers that have none.

        The model reference is the pipeline's `model_name_or_path`, else the loaded model's
        `name_or_path`, else `model`. A `revision` in `hf_model_kwargs` becomes `model_revision`.

        Args:
            pipeline: The pipeline.
            model: The model reference to use when the pipeline has none.

        Raises:
            ValueError: If no model reference resolves.
        """
        model_ref = pipeline.model_name_or_path
        if model_ref is None and pipeline.model is not None:
            model_ref = getattr(pipeline.model, "name_or_path", None) or getattr(
                getattr(pipeline.model, "config", None), "_name_or_path", None
            )
        model_ref = model_ref or model
        if not model_ref:
            raise ValueError("The pipeline has no resolvable model reference for a protocol.")
        kwargs = dict(pipeline.hf_model_kwargs or {})
        revision = kwargs.pop("revision", None)
        backend = pipeline.backend
        if isinstance(backend, str):
            backend = None if backend == "huggingface" else BackendSpec(kind=backend, model=str(model_ref))
        if isinstance(backend, BackendSpec) and backend.kind == "huggingface":
            backend = None
        return cls.create(
            str(model_ref), model_revision=revision, tokenizer=pipeline.tokenizer_name_or_path,
            hf_model_kwargs=kwargs, backend=backend, fit=pipeline.fit,
        )

    @property
    def resolved_provider(self) -> ProviderOptions:
        """`provider`, or `ProviderOptions()` when it is None."""
        return self.provider if self.provider is not None else ProviderOptions()

    def _identity_payload(self) -> dict:
        provider = self.resolved_provider
        return {
            "schema": PROTOCOL_SCHEMA,
            "model": self.model,
            "model_revision": self.model_revision,
            "tokenizer": self.tokenizer,
            "hf_model_kwargs": dict(self.hf_model_kwargs),
            "backend": _backend_identity(self.backend),
            "fit": self.fit,
            "provider": {
                f.name: getattr(provider, f.name) for f in dataclasses.fields(provider) if f.name != "max_concurrency"
            },
            "generate_defaults": dict(self.generate_defaults),
            "seed": self.seed,
            "versions": dict(self.versions),
        }

    def identity(self) -> dict:
        """The canonical JSON form of every field except `provider.max_concurrency` and credentials.

        Returns:
            A JSON-serializable mapping, the input of `from_identity`.

        Raises:
            SpipeSaveError: If a field contains a value with no serialized form.
        """
        if self._identity is None:
            object.__setattr__(self, "_identity", json.loads(canonical_json(self._identity_payload())))
        return json.loads(json.dumps(self._identity))

    @classmethod
    def from_identity(cls, data: Mapping) -> Protocol:
        """Rebuild a protocol from `identity()` output; credentials are not restored.

        Raises:
            ValueError: If `data` is not a `protocol/1` identity.
        """
        from steerability.spipe.codec import DecodeContext, decode

        if not isinstance(data, Mapping) or data.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError(f"Expected a {PROTOCOL_SCHEMA!r} identity.")
        ctx = DecodeContext()
        backend = data.get("backend")
        spec = None
        if backend is not None:
            if backend.get("kind") not in KNOWN_BACKEND_KINDS:
                raise ValueError(f"Unknown backend kind {backend.get('kind')!r}.")
            spec = BackendSpec(kind=backend["kind"], model=backend.get("model"), options=backend.get("options") or {})
        provider_fields = dict(decode(data.get("provider") or {}, ctx))
        if provider_fields.get("reasoning_tags") is not None:
            provider_fields["reasoning_tags"] = tuple(provider_fields["reasoning_tags"])
        return cls(
            model=data["model"],
            model_revision=data.get("model_revision"),
            tokenizer=data.get("tokenizer"),
            hf_model_kwargs=decode(data.get("hf_model_kwargs") or {}, ctx),
            backend=spec,
            fit=data.get("fit", "auto"),
            provider=ProviderOptions(**provider_fields),
            generate_defaults=decode(data.get("generate_defaults") or {}, ctx),
            seed=int(data.get("seed", 0)),
            versions=dict(data.get("versions") or {}),
        )

    @property
    def id(self) -> str:
        """The content id of `identity()`."""
        return json_id(self.identity())

    def fit_context(self) -> dict:
        """The part of the identity a fit can depend on: the model, revision, tokenizer, load
        kwargs, backend, fit policy, and versions."""
        identity = self.identity()
        keys = ("model", "model_revision", "tokenizer", "hf_model_kwargs", "backend", "fit", "versions")
        return {"schema": "fit_context/1", **{key: identity[key] for key in keys}}

    @property
    def fit_context_id(self) -> str:
        """The content id of `fit_context()`."""
        return json_id(self.fit_context())

    @property
    def load_kwargs(self) -> dict[str, Any]:
        """The `hf_model_kwargs` an in-process load passes, with `model_revision` as `revision`."""
        kwargs = dict(self.hf_model_kwargs)
        if self.model_revision is not None:
            kwargs["revision"] = self.model_revision
        return kwargs

    def __hash__(self) -> int:
        return hash(self.id)


__all__ = ["Protocol", "ProtocolFacts", "ProtocolMismatchError", "ProviderOptions", "package_versions"]
