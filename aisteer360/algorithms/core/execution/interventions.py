"""Typed payloads for engine-hosted steering: `InterventionSpec` and `ProcessorSpec`."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from aisteer360.utils.optional import require


def _plain(value: Any) -> Any:
    """Recursively convert mappings and sequences to plain dicts and lists."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _collect_artifact_ids(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "artifact" and isinstance(item, str):
                found.add(item)
            else:
                _collect_artifact_ids(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_artifact_ids(item, found)


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """A serialized activation intervention for intervention-capable backends.

    Each op names its target layers, a transform (kind, scalar parameters, tensor payloads by
    artifact reference, and an ordered modifier list), a token scope, and an optional gate. Kind
    names are the advertised wire names; a worker rejects a spec containing a kind or field it
    does not list.

    Attributes:
        ops: The intervention ops, each a mapping with keys `"layers"`, `"transform"`,
            `"scope"`, and `"gate"`.
        artifacts: Tensor payloads keyed by the content-addressed artifact ids the ops
            reference, each a mapping from tensor name to a float32 contiguous CPU tensor.
            Sessions materialize these into the registry the serving engine reads before
            submission. Excluded from equality, the wire form, and the canonical form.
    """

    ops: tuple[Mapping[str, Any], ...] = ()
    artifacts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, compare=False)

    def to_wire(self) -> dict[str, Any]:
        """The plain-data wire form, `{"ops": [...]}`, with nested mappings and sequences
        converted to dicts and lists."""
        return _plain({"ops": list(self.ops)})

    def artifact_ids(self) -> tuple[str, ...]:
        """Sorted unique artifact ids referenced anywhere in the ops (transform payloads,
        modifiers, and gates, including nested inner gates)."""
        found: set[str] = set()
        _collect_artifact_ids(self.to_wire(), found)
        return tuple(sorted(found))

    def canonical(self) -> str:
        """The canonical serialization, the form hashed for cache salting and provenance.

        Delegates to `vllm_hook_plugins.core.canonical.canonical_bytes` (sorted keys, compact
        separators, UTF-8), so the toolkit and the plugin agree byte-for-byte on the canonical
        form of a spec.

        Raises:
            ModuleNotFoundError: If `vllm_hook_plugins` is not installed. The message names
                the `aisteer360[vllm]` extra.
            TypeError: If an op contains a value with no JSON form. Tensors belong in
                artifacts, never inline.
        """
        canonical = require("vllm_hook_plugins.core.canonical")
        return canonical.canonical_bytes(self.to_wire()).decode("utf-8")

    def salt(self) -> str:
        """The reference cache salt for requests carrying this spec.

        Delegates to `vllm_hook_plugins.core.canonical.request_salt` over the wire form and
        the referenced artifact ids. Returns the 64-char lowercase-hex digest.

        Raises:
            ModuleNotFoundError: If `vllm_hook_plugins` is not installed. The message names
                the `aisteer360[vllm]` extra.
        """
        canonical = require("vllm_hook_plugins.core.canonical")
        return canonical.request_salt(self.to_wire(), list(self.artifact_ids()))


@dataclass(frozen=True, slots=True)
class ProcessorSpec:
    """A serialized per-step logit processor for backends advertising engine-hosted logit math.

    Attributes:
        kind: The advertised processor kind name, e.g. `"constraint"`.
        params: Processor parameters.
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
