"""Exception types for the `.spipe` serialization format."""
from steerability.algorithms.core.base_control import NotFreezableError

__all__ = [
    "SpipeError",
    "SpipeFormatError",
    "SpipeSaveError",
    "SpipeIntegrityError",
    "SpipeStaleError",
    "SpipeCodeRefError",
    "NotFreezableError",
]


class SpipeError(Exception):
    """Base class for `.spipe` errors."""


class SpipeFormatError(SpipeError):
    """The manifest, archive, or artifact layout does not conform to the `spipe/1` format.

    Raised for the following conditions:

    - an unsupported format version, or a manifest field that is missing, mistyped, or unknown
    - a method key that does not match a registered method
    - a malformed archive
    - a malformed encoded value, or a `$dc` or `$ref` target that cannot be imported
    - args that a control or dataclass constructor rejects
    - a malformed artifact id
    - a symlink among the artifact entries of a bundle, or inside a tree payload
    """


class SpipeSaveError(SpipeError):
    """A pipeline or value cannot be serialized.

    Raised for unresolvable model references, unregistered control classes, live model or
    tokenizer objects inside args, lambdas and other unnameable callables, reserved `$`-prefixed
    mapping keys, values over the inline size limit, and controls that cannot freeze.
    """


class SpipeIntegrityError(SpipeError):
    """A stored artifact or data file is missing or does not match its recorded identity.

    Raised for the following conditions:

    - an artifact's bytes do not match its content-addressed id
    - a referenced artifact is not in the store, or no artifact store is available
    - an artifact's store sidecar is malformed, records a different id, or disagrees with the
      manifest record on the encoding or type
    - a `"path"` data reference points to a file that does not match its recorded digest
    """


class SpipeStaleError(SpipeError):
    """A frozen artifact's recorded fit digest does not match the current recipe.

    The recipe's fit-relevant fields were edited after freezing and the pinned artifacts no
    longer correspond to the recipe. Call `thaw()` and re-`steer()`, or pass
    `allow_stale=True` to load anyway.
    """


class SpipeCodeRefError(SpipeError):
    """Decoding a value requires running code that the loader was not permitted to run.

    Raised when a decoded value is one of the following and `SPipe.load()` was called without
    `allow_code=True`:

    - a `$ref` callable reference
    - a `$dc` class other than a toolkit enum or a data-only toolkit dataclass
    - an artifact whose payload contains pickled data

    `SPipe.instantiate_entry(lenient=True)` raises it for the last two even when the spipe was
    loaded with `allow_code=True`, and it decodes a `$ref` to a `CodeRef` instead. Calling a
    `CodeRef` also raises it.
    """
