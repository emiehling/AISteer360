"""Content identity and seed derivation.

`canonical_json` is the JSON of a value's identity-mode encoding by the spipe codec, and
`content_id` digests it. Every stored id in the toolkit (recipes, protocols, fits, rollouts,
benchmarks) is a `content_id`. A value with no serialized form raises `SpipeSaveError`, so an
identity is never computed from a partial description of a value. `derive_seed` derives every
seed the toolkit uses from a base seed and a path of labels.
"""
import hashlib
import json
from collections.abc import Callable
from typing import Any, TypeVar

ID_LENGTH = 16

_SEED_MASK = (1 << 63) - 1

F = TypeVar("F", bound=Callable[..., Any])


def canonical_json(value: Any) -> str:
    """The canonical JSON of a value: its identity-mode encoding, with sorted keys and compact separators.

    The encoding is the one the spipe codec writes in save mode, computed without writing any
    artifact (tensors, probes, memories, and directories reduce to their content ids). Floats
    are written as JSON numbers, so `1` and `1.0` differ.

    Args:
        value: The value to encode.

    Returns:
        The JSON string.

    Raises:
        SpipeSaveError: If the value, or a value inside it, has no serialized form (e.g., a
            lambda, a closure, a bound method, or an object of an unregistered type).
        ValueError: If the value contains a NaN or infinite float.
    """
    from steerability.spipe.codec import EncodeContext, encode

    encoded = encode(value, EncodeContext(store=None))
    return json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_id(value: Any, *, length: int = ID_LENGTH) -> str:
    """The first `length` hex characters of the SHA-256 of `canonical_json(value)`.

    Args:
        value: The value to identify.
        length: Number of hex characters to keep (at most 64).

    Returns:
        The hex digest prefix.

    Raises:
        SpipeSaveError: If the value has no serialized form.
    """
    from steerability.spipe.codec import EncodeContext, encode

    return json_id(encode(value, EncodeContext(store=None)), length=length)


def json_id(encoded: Any, *, length: int = ID_LENGTH) -> str:
    """The content id of a value already in its canonical encoded form.

    `json_id(encode(value))` equals `content_id(value)`. Recipes, protocols, and store keys whose
    canonical form is already built use it, since encoding an encoded form again would reject
    its `$`-prefixed tags.

    Args:
        encoded: A JSON-serializable value in the codec's encoded form.
        length: Number of hex characters to keep (at most 64).

    Returns:
        The hex digest prefix.
    """
    serialized = json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def derive_seed(*parts: int | str) -> int:
    """A seed in `[0, 2**63)` derived from a sequence of parts.

    The seed is the first eight bytes, read big-endian, of the SHA-256 of the parts joined by
    `":"`, masked to 63 bits. `derive_seed(base, label, index)` is the per-item seed the sessions
    derive for item `index` of an operation `label`.

    Args:
        *parts: The parts, usually a base seed followed by labels and indices.

    Returns:
        The derived seed.
    """
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


def versioned(version: str) -> Callable[[F], F]:
    """Mark a module-level callable with a version that enters its encoded identity.

    The decorator sets `__steerability_version__` on the callable. The codec encodes such a
    callable as `{"$ref": target, "version": version}`, so a recipe or benchmark that uses it
    changes identity when the version changes. Decoding with `allow_code=True` raises
    `SpipeCodeRefError` when the imported callable's version differs from the recorded one.

    Args:
        version: The version string.

    Returns:
        A decorator that returns the callable unchanged apart from the attribute.

    Raises:
        TypeError: If `version` is not a non-empty string.
    """
    if not isinstance(version, str) or not version:
        raise TypeError(f"version must be a non-empty string; got {version!r}.")

    def mark(function: F) -> F:
        function.__steerability_version__ = version
        return function

    return mark
