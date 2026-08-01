"""Typed payloads for engine-hosted steering: `InterventionSpec` and `ProcessorSpec`."""
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


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
    """

    ops: tuple[Mapping[str, Any], ...] = ()

    def canonical(self) -> str:
        """The canonical serialization (sorted keys), the form hashed for cache salting and
        provenance."""
        return json.dumps({"ops": [dict(op) for op in self.ops]}, sort_keys=True, default=str)


@dataclass(frozen=True, slots=True)
class ProcessorSpec:
    """A serialized per-step logit processor for backends advertising engine-hosted logit math.

    Attributes:
        kind: The advertised processor kind name, e.g. `"constraint"`.
        params: Processor parameters.
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
