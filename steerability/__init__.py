"""
Steerability toolkit.

The Steerability toolkit enables systematic control over language model behavior through four model
control surfaces: input, structural, state, and output. Methods can be composed into composite model operations (via
steering pipelines). Benchmarks enable comparison of steering pipelines on common use cases.
"""

import logging as _logging
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    __version__ = _distribution_version("steerability")
except _PackageNotFoundError:
    __version__ = "unknown"

_logging.getLogger(__name__).addHandler(_logging.NullHandler())
