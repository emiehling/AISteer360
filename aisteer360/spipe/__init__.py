"""`.spipe`: a portable serialization format for `SteeringPipeline`.

One format holds both the recipe (model reference plus controls-as-constructed) and, when
frozen, the resolution (fingerprints, resolved bindings, per-fit digests, and a
content-addressed artifact store). See `docs/concepts/spipe.md` for the model and the
sharing guidance.
"""
from aisteer360.spipe.codec import DataRef
from aisteer360.spipe.errors import (
    NotFreezableError,
    SpipeCodeRefError,
    SpipeError,
    SpipeFormatError,
    SpipeIntegrityError,
    SpipeSaveError,
    SpipeStaleError,
)
from aisteer360.spipe.spipe import SPipe, SpipeReport

__all__ = [
    "SPipe",
    "SpipeReport",
    "DataRef",
    "SpipeError",
    "SpipeFormatError",
    "SpipeSaveError",
    "SpipeIntegrityError",
    "SpipeStaleError",
    "SpipeCodeRefError",
    "NotFreezableError",
]
