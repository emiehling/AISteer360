"""Steering-method introspection endpoint.

Returns metadata for every registered control: each method's static `args` (introspected from its
Args dataclass) and any per-inference `runtime_kwargs` it consumes (declared on the control class
via `RUNTIME_KWARGS_SCHEMA`). The frontend uses this to populate the library and render parameter
forms without hardcoding method-specific knowledge.
"""
from __future__ import annotations

import logging
import typing
from dataclasses import MISSING, fields, is_dataclass
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .auth import OwnerTokenHash

logger = logging.getLogger(__name__)

router = APIRouter(tags=["methods"])


class MethodFieldSpec(BaseModel):
    name: str
    type: str
    default: Any | None = None
    required: bool = False
    help: str | None = None


class MethodSpec(BaseModel):
    category: str
    method: str
    args: list[MethodFieldSpec] = Field(default_factory=list)
    runtime_kwargs: list[MethodFieldSpec] = Field(default_factory=list)


class MethodsResponse(BaseModel):
    methods: list[MethodSpec]


def _format_type(annotation: Any) -> str:
    """Render a type annotation as a stable string the frontend can dispatch on."""
    if annotation is type(None):
        return "None"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    if origin is typing.Union:
        return " | ".join(_format_type(a) for a in args)
    if origin in (list, tuple, set, frozenset):
        inner = ", ".join(_format_type(a) for a in args) if args else ""
        return f"{origin.__name__}[{inner}]" if inner else origin.__name__
    if origin is dict:
        if args:
            k, v = args
            return f"dict[{_format_type(k)}, {_format_type(v)}]"
        return "dict"
    if origin is typing.Literal:
        return "Literal[" + ", ".join(repr(a) for a in args) + "]"
    return str(annotation)


def _introspect_args(args_cls: type | None) -> list[MethodFieldSpec]:
    if args_cls is None or not is_dataclass(args_cls):
        return []
    try:
        hints = typing.get_type_hints(args_cls)
    except Exception as exc:
        logger.warning("get_type_hints failed for %s: %s", args_cls.__name__, exc)
        hints = {}
    out: list[MethodFieldSpec] = []
    for f in fields(args_cls):
        annotation = hints.get(f.name, f.type)
        if f.default is not MISSING:
            default: Any = f.default
            required = False
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            try:
                default = f.default_factory()  # type: ignore[misc]
            except Exception:
                default = None
            required = False
        else:
            default = None
            required = True
        out.append(
            MethodFieldSpec(
                name=f.name,
                type=_format_type(annotation),
                default=default,
                required=required,
                help=f.metadata.get("help") if f.metadata else None,
            )
        )
    return out


def _runtime_kwargs(control_cls: type) -> list[MethodFieldSpec]:
    schema = getattr(control_cls, "RUNTIME_KWARGS_SCHEMA", []) or []
    out: list[MethodFieldSpec] = []
    for entry in schema:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        out.append(
            MethodFieldSpec(
                name=entry["name"],
                type=str(entry.get("type") or ""),
                default=entry.get("default"),
                required=bool(entry.get("required", False)),
                help=entry.get("help"),
            )
        )
    return out


@router.get("/methods", response_model=MethodsResponse)
def get_methods(_: OwnerTokenHash) -> MethodsResponse:
    """List all registered steering methods with introspected args + runtime_kwargs metadata."""
    from aisteer360.algorithms.core.registry import REGISTRY

    methods: list[MethodSpec] = []
    for category, bucket in REGISTRY.items():
        for method_name, method in bucket.items():
            methods.append(
                MethodSpec(
                    category=category,
                    method=method_name,
                    args=_introspect_args(method.args_cls),
                    runtime_kwargs=_runtime_kwargs(method.control_cls),
                )
            )
    methods.sort(key=lambda m: (m.category, m.method))
    return MethodsResponse(methods=methods)
