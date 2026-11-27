"""The product of a structural control fitted in a process of its own, under a time limit.

`PipelineFactory` fits the product of a structural control (a training or a merge) in a pipeline of the recipe's
structural controls up to and including it. Given a time limit (`fit_seconds`), it fits a product whose structural
entries a new interpreter can rebuild in a child process instead (`fit_in_child`). The child is stopped once the limit
is reached, and its GPU memory comes back when it ends.

What the child receives is written to one JSON request file:

- the store's root and journal mode, the protocol's identity (`Protocol.identity`), and the factory's placement
  arguments;
- the structural entries up to the control as a recipe (`Recipe.to_json`, with its artifacts in the store's
  artifacts), the index of the entry to fit, and the fit key the parent expects for its product;
- the modules of the entries' control classes, which the child imports before it reads the recipe (a control defined
  outside the toolkit registers itself when its module is imported).

Credentials are not part of a protocol's identity, and the child reads them from the environment it inherits. A dataset
held in memory cannot be passed, since the codec records it by fingerprint only, and neither can a callable that the
child cannot import, e.g., one defined in a notebook (`transport_problems`). A dataset given as a `DataRef` to a file or
to a Hub dataset can be passed. The child stores the product in the store's fit cache, where the parent reads it. Its
output goes to the parent's standard output and error.

Run as `python -m steerability.algorithms.core.product_fit <request file>`.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from steerability.algorithms.core.recipe import Recipe
    from steerability.algorithms.core.sweeps import PipelineFactory

logger = logging.getLogger(__name__)

REQUEST_SCHEMA = "product_fit/1"
TERMINATE_SECONDS = 30.0


class ProductFitTimeout(RuntimeError):
    """The fit of a structural product in a child process ran past its time limit and was stopped; nothing is stored.

    Args:
        fit_key: The fit key of the product.
        seconds: The time limit.
    """

    def __init__(self, *, fit_key: str, seconds: float) -> None:
        super().__init__(f"The structural fit {fit_key} ran past its time limit of {seconds:g} seconds and stopped.")
        self.fit_key = fit_key
        self.seconds = seconds


def transport_problems(*, recipe: Recipe) -> list[str]:
    """Why a new interpreter cannot rebuild the entries of a recipe; empty when it can.

    An entry can be rebuilt when its control class and every callable in its arguments come from an importable module
    other than `__main__`, and its arguments contain no dataset held in memory (a `$data` reference of kind
    `"opaque"` in its encoded form).

    Args:
        recipe: The recipe.

    Returns:
        One message per problem.
    """
    from steerability.algorithms.core.registry import resolve_method_key

    names = [name for item, name in zip(recipe.entries, recipe.names) if item.enabled]
    problems: list[str] = []
    for name, encoded in zip(names, recipe.canonical()["entries"]):
        control_cls = resolve_method_key(key=encoded["method"]).control_cls
        module = control_cls.__module__
        if not _importable(module=module):
            problems.append(f"{name}: the control class {control_cls.__qualname__} is defined in {module!r}")
        problems.extend(f"{name}: {problem}" for problem in _value_problems(value=encoded["args"]))
    return problems


def _importable(*, module: str) -> bool:
    if module == "__main__":
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _value_problems(*, value: Any) -> list[str]:
    """The markers of an encoded value that a new interpreter cannot rebuild."""
    problems: list[str] = []
    if isinstance(value, Mapping):
        data = value.get("$data")
        if isinstance(data, Mapping) and data.get("kind", "opaque") == "opaque":
            problems.append(
                f"a dataset held in memory ({data.get('type')}); give it as a DataRef to a file or a Hub dataset",
            )
        target = value.get("$ref")
        if isinstance(target, str) and not _importable(module=target.partition(":")[0]):
            problems.append(f"the callable {target!r} cannot be imported by a new interpreter")
        for item in value.values():
            problems.extend(_value_problems(value=item))
    elif isinstance(value, list):
        for item in value:
            problems.extend(_value_problems(value=item))
    return problems


def fit_in_child(
    *,
    factory: PipelineFactory,
    recipe: Recipe,
    index: int,
    fit_key: str,
    seconds: float,
) -> int:
    """Fit the product of a recipe's structural entry in a child process, waiting at most `seconds`.

    Args:
        factory: The factory whose protocol, store, and placement the child uses; it must have a store.
        recipe: The structural entries up to and including the entry to fit.
        index: The index of the entry among the recipe's structural entries.
        fit_key: The fit key of the product, which the child checks against the key it computes.
        seconds: The time limit.

    Returns:
        The number of products the child fitted and stored (0 when another writer stored the product first).

    Raises:
        ProductFitTimeout: If the child runs past `seconds`; it is stopped, and nothing is stored.
        RuntimeError: If the child fails.
    """
    store = factory.store
    modules = sorted({_control_module(method=item.method) for item in recipe.entries if item.enabled})
    with tempfile.TemporaryDirectory(prefix="steerability-product-fit-") as scratch:
        request_path = Path(scratch) / "request.json"
        result_path = Path(scratch) / "result.json"
        request = {
            "schema": REQUEST_SCHEMA,
            "store": str(store.root),
            "journal_mode": store.journal_mode,
            "protocol": factory.protocol.identity(),
            "recipe": recipe.to_json(store=store.artifacts),
            "index": index,
            "fit_key": fit_key,
            "modules": modules,
            "device_map": factory.device_map,
            "trust_remote_code": factory.trust_remote_code,
            "result": str(result_path),
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        command = [sys.executable, "-m", "steerability.algorithms.core.product_fit", str(request_path)]
        logger.info("Fitting the structural product %s in a child process, for at most %g seconds.", fit_key, seconds)
        process = subprocess.Popen(command, env=dict(os.environ))
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            _stop(process=process)
            raise ProductFitTimeout(fit_key=fit_key, seconds=seconds) from None
        finally:
            if process.poll() is None:
                _stop(process=process)
        if process.returncode != 0:
            raise RuntimeError(
                f"The child process that fits the structural product {fit_key} exited with code "
                f"{process.returncode}; its output precedes this error.",
            )
        return int(json.loads(result_path.read_text(encoding="utf-8"))["products"])


def _control_module(*, method: str) -> str:
    from steerability.algorithms.core.registry import resolve_method_key

    return resolve_method_key(key=method).control_cls.__module__


def _stop(*, process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the product that a request file describes, in this process.

    Args:
        argv: The command-line arguments (the request file), or None for `sys.argv[1:]`.

    Returns:
        The exit code.

    Raises:
        RuntimeError: If the product's fit key in this process differs from the one the parent expects.
    """
    from steerability.algorithms.core.protocol import Protocol
    from steerability.algorithms.core.recipe import Recipe
    from steerability.algorithms.core.store import Store
    from steerability.algorithms.core.sweeps import PipelineFactory
    from steerability.spipe.codec import DecodeContext

    logging.basicConfig(level=logging.INFO)
    arguments = sys.argv[1:] if argv is None else list(argv)
    if len(arguments) != 1:
        logger.error("usage: python -m steerability.algorithms.core.product_fit <request file>")
        return 2
    request = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"Expected a {REQUEST_SCHEMA!r} request; got {request.get('schema')!r}.")
    for module in request["modules"]:
        importlib.import_module(module)
    store = Store.open(root=request["store"], journal_mode=request["journal_mode"])
    try:
        recipe = Recipe.from_json(
            data=request["recipe"],
            context=DecodeContext(store=store.artifacts, allow_code=True, data_mode="keep"),
        )
        factory = PipelineFactory(
            protocol=Protocol.from_identity(data=request["protocol"]),
            store=store,
            device_map=request["device_map"],
            trust_remote_code=request["trust_remote_code"],
        )
        products = factory.fit_product(recipe=recipe, index=request["index"], fit_key=request["fit_key"])
    finally:
        store.close()
    Path(request["result"]).write_text(json.dumps({"products": products}), encoding="utf-8")
    return 0


__all__ = ["ProductFitTimeout", "fit_in_child", "main", "transport_problems"]


if __name__ == "__main__":
    sys.exit(main())
