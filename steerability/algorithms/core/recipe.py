"""Recipes and templates: the description of a steering pipeline.

A `Recipe` is an ordered tuple of `Entry` objects, each naming a registered steering method and its
constructor arguments. Its identity (`Recipe.id`) is the content id of a canonical form computed
from the method's `Args` dataclass, so key order, explicit defaults, entry names, disabled entries,
and the order of entries across categories do not change it. A `Template` is a recipe whose
arguments may contain ranges (`Choice`, `Interval`, `Text`). Ranges form axes, which `expand` and
`assignments` enumerate or sample, and `Template.bind` replaces ranges with values.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
import typing
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from steerability.algorithms.core.identity import json_id

if TYPE_CHECKING:
    from steerability.algorithms.core.base_control import BaseControl
    from steerability.spipe.codec import DecodeContext
    from steerability.spipe.store import ArtifactStore

RECIPE_SCHEMA = "recipe/1"
TEMPLATE_SCHEMA = "template/1"

CATEGORY_ORDER = ("structural_control", "input_control", "state_control", "output_control")

_POSITION_TOLERANCE = 1e-6


class RecipeError(ValueError):
    """A recipe or template is malformed, or a value lies outside its range."""


@dataclass(frozen=True, slots=True)
class Choice:
    """A range over an explicit set of values.

    Attributes:
        values: The values, at least one. Ranges do not nest, so a value is not a range.
        ordered: Whether the order of the values is meaningful. Only ordered choices may share an
            axis.
        axis: The name of a shared axis, or None for an axis of its own. Requires `ordered=True`.
    """

    values: tuple[Any, ...]
    ordered: bool = False
    axis: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.values, (str, bytes)) or not isinstance(self.values, Sequence):
            raise RecipeError(f"Choice values must be a sequence; got {type(self.values).__name__}.")
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise RecipeError("A Choice needs at least one value.")
        if any(isinstance(value, Range) for value in self.values):
            raise RecipeError("Ranges do not nest; a Choice value cannot be a range.")
        if self.axis is not None and not self.ordered:
            raise RecipeError(f"Choice on axis {self.axis!r} must be ordered (pass ordered=True).")


@dataclass(frozen=True, slots=True)
class Interval:
    """A range over a closed numeric interval.

    `at(t)` is `low + t * (high - low)`, or `low * (high / low) ** t` with `log=True`, rounded to the
    nearest integer (halves round up) with `integer=True`. An interval has no wrap-around, so an
    angle range stops short of the full turn, e.g., `Interval(0, 315)`.

    Attributes:
        low: The lower bound.
        high: The upper bound, at least `low`.
        log: Geometric spacing on a grid and log-uniform sampling; both bounds must be positive.
        integer: Whether the values are integers.
        axis: The name of a shared axis, or None for an axis of its own.
    """

    low: float
    high: float
    log: bool = False
    integer: bool = False
    axis: str | None = None

    def __post_init__(self) -> None:
        for bound in (self.low, self.high):
            if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound):
                raise RecipeError(f"Interval bounds must be finite numbers; got {bound!r}.")
        if self.high < self.low:
            raise RecipeError(f"Interval high ({self.high}) is below low ({self.low}).")
        if self.log and self.low <= 0:
            raise RecipeError(f"A log Interval needs positive bounds; got low={self.low}.")

    def at(self, fraction: float) -> float | int:
        """The value at a fraction in [0, 1] of the interval.

        Raises:
            RecipeError: If `fraction` is outside [0, 1].
        """
        if not 0.0 <= fraction <= 1.0:
            raise RecipeError(f"Interval fraction must be in [0, 1]; got {fraction}.")
        if self.log:
            value = self.low * (self.high / self.low) ** fraction
        else:
            value = self.low + fraction * (self.high - self.low)
        value = min(max(value, self.low), self.high)
        if self.integer:
            return int(math.floor(value + 0.5))
        return float(value)

    def grid(self, n: int) -> Choice:
        """An ordered `Choice` of `at(i / (n - 1))` for `i` in `range(n)`, keeping `axis`.

        A value equal to the one before it is dropped (integer intervals can produce repeats), so
        the result may have fewer than `n` values.

        Raises:
            RecipeError: If `n` is below 1.
        """
        if n < 1:
            raise RecipeError(f"Interval.grid needs n >= 1; got {n}.")
        values: list[float | int] = []
        for index in range(n):
            value = self.at(index / (n - 1) if n > 1 else 0.0)
            if not values or value != values[-1]:
                values.append(value)
        return Choice(tuple(values), ordered=True, axis=self.axis)

    def contains(self, value: Any) -> bool:
        """Whether `value` is a number within the interval (an integer when `integer` is set)."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if self.integer and not float(value).is_integer():
            return False
        return self.low - _POSITION_TOLERANCE <= value <= self.high + _POSITION_TOLERANCE

    def fraction_bounds(self, value: float) -> tuple[float, float]:
        """The fractions `t` in [0, 1] at which `at(t)` equals `value`, as a closed interval."""
        def inverse(target: float) -> float:
            if self.high == self.low:
                return 0.0
            if self.log:
                return math.log(target / self.low) / math.log(self.high / self.low)
            return (target - self.low) / (self.high - self.low)

        if not self.integer:
            fraction = inverse(float(value))
            return fraction - _POSITION_TOLERANCE, fraction + _POSITION_TOLERANCE
        low = max(value - 0.5, self.low)
        high = min(value + 0.5, self.high)
        return inverse(low) - _POSITION_TOLERANCE, inverse(high) + _POSITION_TOLERANCE


@dataclass(frozen=True, slots=True)
class Text:
    """A range over strings of bounded length. Text is never ordered and has no grid.

    Attributes:
        max_chars: The maximum length of a string in the range.
    """

    max_chars: int = 4000


Range = Choice | Interval | Text


def _is_range(value: Any) -> bool:
    return isinstance(value, (Choice, Interval, Text))


def _ordered(member: Range) -> bool:
    return isinstance(member, Interval) or (isinstance(member, Choice) and member.ordered)


def _same_value(left: Any, right: Any) -> bool:
    """Equality of two argument values, by canonical form for values without scalar equality."""
    scalar = (type(None), bool, int, float, str)
    if isinstance(left, scalar) and isinstance(right, scalar):
        return left == right and isinstance(left, bool) == isinstance(right, bool)
    from steerability.algorithms.core.identity import canonical_json

    try:
        return canonical_json(left) == canonical_json(right)
    except Exception:
        return left is right


@dataclass(frozen=True, slots=True)
class Axis:
    """A unit of variation: one range, or the ranges of one template that share an axis name.

    A discrete axis (any `Choice` member) has as many positions as its choices have values, and an
    `Interval` member contributes its value at each position's fraction. A continuous axis (only
    `Interval` members) has a position at every fraction.

    Attributes:
        key: The member's path, or `"@<name>"` for a shared axis.
        members: Mapping from member path to range, in path order.
        ordered: Whether the positions are ordered.
    """

    key: str
    members: Mapping[str, Range]
    ordered: bool

    @property
    def continuous(self) -> bool:
        """Whether every member is an `Interval`."""
        return all(isinstance(member, Interval) for member in self.members.values())

    @property
    def has_text(self) -> bool:
        """Whether a member is a `Text` range, which has no positions."""
        return any(isinstance(member, Text) for member in self.members.values())

    @property
    def count(self) -> int | None:
        """The number of positions of a discrete axis, or None for a continuous or text axis."""
        for member in self.members.values():
            if isinstance(member, Choice):
                return len(member.values)
        return None

    def at(self, fraction: float) -> dict[str, Any]:
        """The value of every member at a fraction of a continuous axis.

        Raises:
            RecipeError: If the axis is not continuous.
        """
        if not self.continuous:
            raise RecipeError(f"Axis {self.key!r} is not continuous; bind it by position index.")
        return {path: member.at(fraction) for path, member in self.members.items()}

    def at_index(self, index: int) -> dict[str, Any]:
        """The value of every member at a position index of a discrete axis.

        Raises:
            RecipeError: If the axis is not discrete or the index is out of range.
        """
        count = self.count
        if count is None or self.has_text:
            raise RecipeError(f"Axis {self.key!r} is not discrete; bind it by fraction.")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            raise RecipeError(f"Axis {self.key!r} has {count} position(s); got index {index!r}.")
        fraction = index / (count - 1) if count > 1 else 0.0
        return {
            path: member.values[index] if isinstance(member, Choice) else member.at(fraction)
            for path, member in self.members.items()
        }

    def default(self) -> dict[str, Any]:
        """The value of every member at the axis's default position.

        The default position is the first position of a discrete axis and the midpoint (fraction 0.5) of a continuous
        one. A `Text` member defaults to the empty string.

        Returns:
            Mapping from member path to value.
        """
        if self.has_text:
            return {path: "" for path in self.members}
        if self.continuous:
            return self.at(0.5)
        return self.at_index(0)

    def positions(self, n: int | None = None) -> tuple[dict[str, Any], ...]:
        """The positions of the axis in order, each a mapping from member path to value.

        A discrete axis ignores `n`. A continuous axis requires `n` and returns `at(i / (n - 1))`
        for each `i`, dropping a position equal to the one before it.

        Raises:
            RecipeError: If the axis has a `Text` member, or is continuous and `n` is None.
        """
        if self.has_text:
            raise RecipeError(f"Axis {self.key!r} is a Text range, which has no positions; bind it explicitly.")
        count = self.count
        if count is not None:
            return tuple(self.at_index(index) for index in range(count))
        if n is None:
            raise RecipeError(
                f"Axis {self.key!r} is continuous; give a grid size (or replace each member with Interval.grid(n))."
            )
        positions: list[dict[str, Any]] = []
        for index in range(n):
            position = self.at(index / (n - 1) if n > 1 else 0.0)
            if not positions or position != positions[-1]:
                positions.append(position)
        return tuple(positions)

    def index_of(self, values: Mapping[str, Any]) -> int:
        """The position index at which every member of a discrete axis takes its value in `values`.

        Raises:
            RecipeError: If a member's value is outside its range, or the members sit at differing
                positions.
        """
        candidates: set[int] | None = None
        for path, member in self.members.items():
            value = values[path]
            indices = {
                index for index in range(self.count)
                if _same_value(self.at_index(index)[path], value)
            }
            if not indices:
                raise RecipeError(f"{path}: {value!r} is not a position of axis {self.key!r}.")
            candidates = indices if candidates is None else candidates & indices
        if not candidates:
            raise RecipeError(f"Axis {self.key!r} is bound at differing positions: {dict(values)}.")
        return min(candidates)

    def check_fraction(self, values: Mapping[str, Any]) -> None:
        """Check that the members of a continuous axis take their values at one fraction.

        Raises:
            RecipeError: If the members sit at differing fractions.
        """
        low, high = 0.0 - _POSITION_TOLERANCE, 1.0 + _POSITION_TOLERANCE
        for path, member in self.members.items():
            member_low, member_high = member.fraction_bounds(values[path])
            low, high = max(low, member_low), min(high, member_high)
        if low > high:
            raise RecipeError(f"Axis {self.key!r} is bound at differing positions: {dict(values)}.")


@dataclass(frozen=True, slots=True, eq=False)
class Entry:
    """One control of a recipe or template.

    Attributes:
        method: The registry key `"<category>_control/<name>"`.
        args: The constructor arguments.
        enabled: Whether the control runs; a disabled entry is not part of the identity.
        name: A label for paths and frames, not part of the identity.
    """

    method: str
    args: Mapping[str, Any]
    enabled: bool = True
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or "/" not in self.method:
            raise RecipeError(f"Entry method must be a '<category>_control/<name>' key; got {self.method!r}.")
        if self.method.partition("/")[0] not in CATEGORY_ORDER:
            raise RecipeError(f"Entry method {self.method!r} names no steering category {list(CATEGORY_ORDER)}.")
        if not isinstance(self.args, Mapping):
            raise RecipeError(f"Entry args must be a mapping; got {type(self.args).__name__}.")
        object.__setattr__(self, "args", dict(self.args))

    @property
    def category(self) -> str:
        """The entry's category, e.g. `"state_control"`."""
        return self.method.partition("/")[0]


def entry(control: type | str, /, *, name: str | None = None, enabled: bool = True, **args: Any) -> Entry:
    """Build an `Entry` from a control class or a method key.

    Args:
        control: A registered control class (converted with `registry.method_key_for`) or a method
            key.
        name: A label for the entry.
        enabled: Whether the control runs.
        **args: The constructor arguments.

    Returns:
        The entry.

    Raises:
        RegistryError: If `control` is a class that is not registered.
    """
    if isinstance(control, type):
        from steerability.algorithms.core.registry import method_key_for

        control = method_key_for(control)
    return Entry(method=control, args=args, enabled=enabled, name=name)


def _ordered_entries(entries: Sequence[Entry]) -> tuple[Entry, ...]:
    """The entries in category order, keeping the given order within a category."""
    entries = tuple(entries)
    for item in entries:
        if not isinstance(item, Entry):
            raise RecipeError(f"Recipe entries must be Entry objects; got {type(item).__name__}.")
    return tuple(sorted(entries, key=lambda item: CATEGORY_ORDER.index(item.category)))


def _entry_names(entries: Sequence[Entry]) -> tuple[str, ...]:
    """The resolved names of ordered entries.

    Raises:
        RecipeError: If two entries resolve to one name.
    """
    names: list[str] = []
    unnamed_counts: dict[str, int] = {}
    for item in entries:
        if item.name is not None:
            names.append(item.name)
            continue
        base = item.method.partition("/")[2]
        count = unnamed_counts.get(item.method, 0) + 1
        unnamed_counts[item.method] = count
        names.append(base if count == 1 else f"{base}_{count}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RecipeError(f"Entries resolve to the same name(s) {duplicates}; give each entry a distinct name=.")
    return tuple(names)


def _child_path(path: str, key: Any, sequence: bool) -> str:
    return f"{path}[{key}]" if sequence else f"{path}.{key}"


def _walk_ranges(value: Any, path: str) -> Iterator[tuple[str, tuple, Range]]:
    """Yield `(path, key_path, range)` for every range in an argument value.

    Ranges are found at the top level and inside nested dicts, lists, and tuples. Component
    objects and dataclass instances are not searched.
    """
    def walk(current: Any, current_path: str, keys: tuple) -> Iterator[tuple[str, tuple, Range]]:
        if _is_range(current):
            yield current_path, keys, current
        elif isinstance(current, Mapping):
            for key, item in current.items():
                yield from walk(item, _child_path(current_path, key, False), (*keys, key))
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                yield from walk(item, _child_path(current_path, index, True), (*keys, index))

    yield from walk(value, path, ())


def _replace_at(value: Any, keys: tuple, replacement: Any) -> Any:
    """A copy of `value` with the item at `keys` replaced, copying only the containers on the path."""
    if not keys:
        return replacement
    head, rest = keys[0], keys[1:]
    if isinstance(value, Mapping):
        copied = dict(value)
        copied[head] = _replace_at(value[head], rest, replacement)
        return copied
    copied_list = list(value)
    copied_list[head] = _replace_at(value[head], rest, replacement)
    return tuple(copied_list) if isinstance(value, tuple) else copied_list


def _is_float_annotation(annotation: Any) -> bool:
    """Whether a field annotation is `float` or `float | None`."""
    if isinstance(annotation, str):
        normalized = annotation.replace(" ", "")
        return normalized in ("float", "float|None", "None|float", "Optional[float]", "typing.Optional[float]")
    if annotation is float:
        return True
    arguments = typing.get_args(annotation)
    return float in arguments and all(argument is float or argument is type(None) for argument in arguments)


def _float_fields(args_cls: type) -> set[str]:
    """The names of the dataclass fields annotated `float` or `float | None`."""
    try:
        hints = typing.get_type_hints(args_cls)
    except Exception:
        hints = {}
    return {
        field.name for field in dataclasses.fields(args_cls)
        if _is_float_annotation(hints.get(field.name, field.type))
    }


def _as_float(value: Any) -> Any:
    """`value` as a float when it is an int (not a bool), else unchanged."""
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _float_range(member: Range) -> Range:
    """A range whose bounds or choice values are converted with `_as_float`."""
    if isinstance(member, Interval) and not member.integer:
        return dataclasses.replace(member, low=float(member.low), high=float(member.high))
    if isinstance(member, Choice):
        return dataclasses.replace(member, values=tuple(_as_float(value) for value in member.values))
    return member


def _resolve(item: Entry, name: str):
    """The registered method of an entry.

    Raises:
        RecipeError: If the method key is unknown.
    """
    from steerability.algorithms.core.registry import RegistryError, resolve_method_key

    try:
        return resolve_method_key(item.method)
    except RegistryError as exc:
        raise RecipeError(f"{name}: {exc}") from exc


def _encode(value: Any, ctx, path: str) -> Any:
    """Encode one argument value with the codec, raising `RecipeError` with the argument path."""
    from steerability.spipe.codec import encode
    from steerability.spipe.errors import SpipeSaveError

    try:
        return encode(value, ctx, path)
    except SpipeSaveError as exc:
        raise RecipeError(str(exc)) from exc


def _encode_template_value(value: Any, ctx, path: str) -> Any:
    """Encode an argument value that may contain ranges, as `{"$range": {...}}` in place of each range."""
    if _is_range(value):
        kind = {Choice: "choice", Interval: "interval", Text: "text"}[type(value)]
        payload: dict[str, Any] = {"kind": kind}
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            payload[field.name] = _encode(list(item), ctx, f"{path}.{field.name}") if field.name == "values" \
                else item
        return {"$range": payload}
    if not any(True for _ in _walk_ranges(value, path)):
        return _encode(value, ctx, path)
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _encode_template_value(item, ctx, f"{path}.{key}") for key, item in value.items()}
        entries = [[key, _encode_template_value(item, ctx, f"{path}.{key}")] for key, item in value.items()]
        entries.sort(key=lambda pair: (type(pair[0]).__name__, repr(pair[0])))
        return {"$map": entries}
    return [_encode_template_value(item, ctx, f"{path}[{index}]") for index, item in enumerate(value)]


class _Described:
    """Shared behavior of `Recipe` and `Template`: ordering, names, identity, and equality."""

    entries: tuple[Entry, ...]

    def __post_init__(self) -> None:
        ordered = _ordered_entries(self.entries)
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "_names", _entry_names(ordered))
        object.__setattr__(self, "_canonical", None)
        object.__setattr__(self, "_id", None)

    @property
    def names(self) -> tuple[str, ...]:
        """The resolved entry names, in entry order."""
        return self._names

    def _canonical_entry(self, item: Entry, name: str, ctx) -> dict:
        raise NotImplementedError

    def _canonical_form(self, ctx, schema: str) -> dict:
        entries = []
        for item, name in zip(self.entries, self._names):
            if not item.enabled:
                continue
            entries.append({"method": item.method, "args": self._canonical_entry(item, name, ctx)})
        return {"schema": schema, "entries": entries}

    def canonical(self) -> dict:
        """The canonical form, computed once per instance.

        Returns:
            `{"schema": ..., "entries": [{"method": ..., "args": ...}, ...]}` over the enabled
            entries in category order.

        Raises:
            RecipeError: If a method key is unknown, the args do not construct the control's
                `Args`, or a value has no serialized form (the message gives the argument path).
        """
        if self._canonical is None:
            from steerability.spipe.codec import EncodeContext

            object.__setattr__(self, "_canonical", self._canonical_form(EncodeContext(store=None), self._schema))
        return self._canonical

    @property
    def id(self) -> str:
        """The content id of the canonical form."""
        if self._id is None:
            object.__setattr__(self, "_id", json_id(self.canonical()))
        return self._id

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.id))


@dataclass(frozen=True, eq=False)
class Recipe(_Described):
    """A concrete description of a steering pipeline: an ordered tuple of entries without ranges.

    Entries are ordered by category (structural, input, state, output), keeping the given order
    within a category, and each has a resolved name (`names`). Two recipes compare and hash by
    `id`. A value must not be changed after it has been used in a recipe, since the canonical form
    and the id are computed once.

    Attributes:
        entries: The entries.

    Raises:
        RecipeError: If an entry contains a range, two entries resolve to one name, or an entry is
            malformed.
    """

    entries: tuple[Entry, ...] = ()

    _schema: typing.ClassVar[str] = RECIPE_SCHEMA

    def __post_init__(self) -> None:
        super().__post_init__()
        for item, name in zip(self.entries, self._names):
            for key, value in item.args.items():
                for path, _, _ in _walk_ranges(value, f"{name}.{key}"):
                    raise RecipeError(f"{path}: a Recipe contains no ranges; use a Template and bind it.")

    def _canonical_entry(self, item: Entry, name: str, ctx) -> dict:
        method = _resolve(item, name)
        args_cls = method.control_cls.Args
        if args_cls is None:
            if item.args:
                raise RecipeError(f"{name}: {item.method} accepts no arguments; got {sorted(item.args)}.")
            return {}
        try:
            args = args_cls.validate(dict(item.args))
        except Exception as exc:
            raise RecipeError(f"{name}: the args do not construct {args_cls.__name__} ({type(exc).__name__}: {exc}).") \
                from exc
        float_fields = _float_fields(args_cls)
        encoded = {}
        for field in dataclasses.fields(args):
            if not field.init:
                continue
            value = getattr(args, field.name)
            if field.name in float_fields:
                value = _as_float(value)
            encoded[field.name] = _encode(value, ctx, f"{name}.{field.name}")
        return encoded

    def instantiate(self) -> list[BaseControl]:
        """One control per enabled entry, in entry order, built from the entry's args.

        Returns:
            The controls.

        Raises:
            RecipeError: If a method key is unknown.
        """
        controls = []
        for item, name in zip(self.entries, self._names):
            if not item.enabled:
                continue
            controls.append(_resolve(item, name).control_cls(**item.args))
        return controls

    def to_json(self, store: ArtifactStore | None = None) -> dict:
        """The canonical form, encoded in save mode into `store` when one is given.

        Entry names are labels and are not written, so a recipe read back has default names.

        Args:
            store: The artifact store that artifacts are written to, or None.

        Returns:
            The JSON form.
        """
        if store is None:
            return json.loads(json.dumps(self.canonical()))
        from steerability.spipe.codec import EncodeContext

        return self._canonical_form(EncodeContext(store=store), RECIPE_SCHEMA)

    @classmethod
    def from_json(
        cls,
        data: Mapping,
        *,
        store: ArtifactStore | None = None,
        allow_code: bool = False,
        context: DecodeContext | None = None,
    ) -> Recipe:
        """Read a recipe from its JSON form.

        Args:
            data: The JSON form from `to_json`.
            store: The artifact store that `$artifact` references are read from.
            allow_code: Whether decoding may import `$ref` callables and construct dataclasses
                outside the data-only set.
            context: A decoding context to use in place of `store` and `allow_code`, e.g. a
                lenient one (`code_mode="sentinel"`, `data_mode="keep"`) that imports and runs no
                code.

        Returns:
            The recipe.

        Raises:
            RecipeError: If the schema differs or an entry is malformed.
            SpipeCodeRefError: If decoding needs code that the context does not permit.
            SpipeFormatError: If an encoded value is malformed.
        """
        from steerability.spipe.codec import DecodeContext, decode

        if not isinstance(data, Mapping) or data.get("schema") != RECIPE_SCHEMA:
            found = data.get("schema") if isinstance(data, Mapping) else type(data).__name__
            raise RecipeError(f"Expected a {RECIPE_SCHEMA!r} recipe; got schema {found!r}.")
        ctx = context if context is not None else DecodeContext(store=store, allow_code=allow_code)
        entries = []
        for index, item in enumerate(data.get("entries") or []):
            if not isinstance(item, Mapping) or not isinstance(item.get("args"), Mapping):
                raise RecipeError(f"entries[{index}]: expected an object with 'method' and 'args'.")
            args = {key: decode(value, ctx, f"entries[{index}].args.{key}") for key, value in item["args"].items()}
            entries.append(Entry(method=item["method"], args=args))
        return cls(tuple(entries))

    @classmethod
    def from_controls(cls, controls: Sequence[BaseControl]) -> Recipe:
        """The recipe of constructed controls: each control's method key, the init fields of its
        `args`, and its `enabled` flag.

        Raises:
            RegistryError: If a control's class is not registered.
        """
        from steerability.algorithms.core.registry import method_key_for

        entries = []
        for control in controls:
            args = getattr(control, "args", None)
            values = {} if args is None else {
                field.name: getattr(args, field.name) for field in dataclasses.fields(args) if field.init
            }
            entries.append(Entry(
                method=method_key_for(type(control)), args=values, enabled=bool(getattr(control, "enabled", True)),
            ))
        return cls(tuple(entries))


@dataclass(frozen=True, eq=False)
class Template(_Described):
    """A recipe whose arguments may contain ranges.

    Ranges may sit at the top level of an entry's args or inside nested dicts, lists, and tuples,
    not inside component objects or dataclass instances. Each range is an axis of its own unless it
    names a shared `axis`, in which case the ranges with that name form one axis keyed
    `"@<name>"` and move together. Only ordered ranges (an `Interval`, or a `Choice` with
    `ordered=True`) may share an axis, and the `Choice` members of one axis must have the same
    number of values. Two templates compare and hash by `id`.

    Attributes:
        entries: The entries.

    Raises:
        RecipeError: If two entries resolve to one name, a shared axis mixes an unordered range,
            or its choices differ in length.
    """

    entries: tuple[Entry, ...]

    _schema: typing.ClassVar[str] = TEMPLATE_SCHEMA

    def __post_init__(self) -> None:
        super().__post_init__()
        ranges: dict[str, tuple[int, str, tuple, Range]] = {}
        for index, (item, name) in enumerate(zip(self.entries, self._names)):
            for key, value in item.args.items():
                for path, keys, member in _walk_ranges(value, f"{name}.{key}"):
                    ranges[path] = (index, key, keys, member)
        object.__setattr__(self, "_ranges", ranges)
        object.__setattr__(self, "_axes", self._build_axes())

    def _build_axes(self) -> dict[str, Axis]:
        grouped: dict[str, dict[str, Range]] = {}
        for path, (_, _, _, member) in self._ranges.items():
            axis_name = getattr(member, "axis", None)
            key = path if axis_name is None else f"@{axis_name}"
            grouped.setdefault(key, {})[path] = member
        axes = {}
        for key, members in grouped.items():
            if key.startswith("@"):
                unordered = [path for path, member in members.items() if not _ordered(member)]
                if unordered:
                    raise RecipeError(f"Axis {key!r}: only ordered ranges may share an axis; {unordered} are not.")
                counts = {len(member.values) for member in members.values() if isinstance(member, Choice)}
                if len(counts) > 1:
                    raise RecipeError(
                        f"Axis {key!r}: its Choice members have differing numbers of values {sorted(counts)}."
                    )
            axes[key] = Axis(key=key, members=dict(members), ordered=all(_ordered(m) for m in members.values()))
        return axes

    def _canonical_entry(self, item: Entry, name: str, ctx) -> dict:
        method = _resolve(item, name)
        args_cls = method.control_cls.Args
        if args_cls is None:
            if item.args:
                raise RecipeError(f"{name}: {item.method} accepts no arguments; got {sorted(item.args)}.")
            return {}
        known = {field.name: field for field in dataclasses.fields(args_cls) if field.init}
        unknown = sorted(set(item.args) - set(known))
        if unknown:
            raise RecipeError(f"{name}: {args_cls.__name__} has no field(s) {unknown}.")
        float_fields = _float_fields(args_cls)
        encoded = {}
        for field_name, field in known.items():
            if field_name in item.args:
                value = item.args[field_name]
            elif field.default is not dataclasses.MISSING:
                value = field.default
            elif field.default_factory is not dataclasses.MISSING:
                value = field.default_factory()
            else:
                raise RecipeError(f"{name}.{field_name}: the field has no default and is not given.")
            if field_name in float_fields:
                value = _float_range(value) if _is_range(value) else _as_float(value)
            encoded[field_name] = _encode_template_value(value, ctx, f"{name}.{field_name}")
        return encoded

    def axes(self) -> dict[str, Axis]:
        """The template's axes keyed by axis key, in entry order, then by the path of the first member."""
        return dict(self._axes)

    def bind(self, assignment: Mapping[str, Any]) -> Template:
        """Replace ranges with values.

        Keys are member paths, or the key of a shared axis bound with a position index (discrete
        axis) or a fraction in [0, 1] (continuous axis).

        Args:
            assignment: Mapping from path or axis key to value.

        Returns:
            A template with the bound ranges replaced.

        Raises:
            RecipeError: If a path has no range, a value is outside its range, or a shared axis
                is bound in part or at differing positions (the message gives the path or key).
        """
        values: dict[str, Any] = {}
        for key, value in assignment.items():
            if key.startswith("@"):
                axis = self._axes.get(key)
                if axis is None:
                    raise RecipeError(f"{key}: the template has no shared axis {key!r}.")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise RecipeError(f"{key}: bind a shared axis with a position index or a fraction; got {value!r}.")
                position = axis.at(float(value)) if axis.continuous else axis.at_index(value)
                for path, member_value in position.items():
                    if path in assignment:
                        raise RecipeError(f"{path}: bound both directly and through axis {key!r}.")
                    values[path] = member_value
            else:
                if key not in self._ranges:
                    raise RecipeError(f"{key}: the template has no range at this path.")
                values[key] = value

        for path, value in values.items():
            member = self._ranges[path][3]
            if isinstance(member, Choice):
                inside = any(_same_value(choice, value) for choice in member.values)
            elif isinstance(member, Interval):
                inside = member.contains(value)
            else:
                inside = isinstance(value, str) and len(value) <= member.max_chars
            if not inside:
                raise RecipeError(f"{path}: {value!r} is outside {member!r}.")

        for key, axis in self._axes.items():
            if not key.startswith("@"):
                continue
            bound = [path for path in axis.members if path in values]
            if not bound:
                continue
            if len(bound) != len(axis.members):
                missing = [path for path in axis.members if path not in values]
                raise RecipeError(f"{key}: shared axis is bound in part; bind {missing} too, or bind {key!r}.")
            member_values = {path: values[path] for path in axis.members}
            if axis.continuous:
                axis.check_fraction(member_values)
            else:
                axis.index_of(member_values)

        entries = list(self.entries)
        for path, value in values.items():
            index, arg, keys, _ = self._ranges[path]
            item = entries[index]
            args = dict(item.args)
            args[arg] = _replace_at(args[arg], keys, value)
            entries[index] = dataclasses.replace(item, args=args)
        return Template(tuple(entries))

    def recipe(self) -> Recipe:
        """The recipe of a template whose ranges are all bound.

        Raises:
            RecipeError: If a range remains.
        """
        if self._ranges:
            raise RecipeError(f"The template still has ranges at {sorted(self._ranges)}; bind them first.")
        return Recipe(self.entries)


@dataclass(frozen=True, slots=True)
class Point:
    """One point of a sweep.

    Attributes:
        recipe: The concrete recipe.
        assignment: Mapping from member path to the chosen value; empty for a recipe without axes.
    """

    recipe: Recipe
    assignment: Mapping[str, Any]


BASELINE = Recipe()


def _grid(axes: Mapping[str, Axis]) -> Iterator[dict[str, Any]]:
    positions = []
    for axis in axes.values():
        if axis.has_text:
            raise RecipeError(f"Axis {axis.key!r} is a Text range; a grid cannot enumerate text.")
        if axis.continuous:
            raise RecipeError(
                f"Axis {axis.key!r} is continuous; replace its intervals with Interval.grid(n) to enumerate a grid."
            )
        positions.append(axis.positions())
    import itertools

    for combination in itertools.product(*positions):
        merged: dict[str, Any] = {}
        for position in combination:
            merged.update(position)
        yield merged


def _draws(axes: Mapping[str, Axis], seed: int) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    for axis in axes.values():
        if axis.has_text:
            raise RecipeError(f"Axis {axis.key!r} is a Text range; random sampling cannot draw text.")
    while True:
        merged: dict[str, Any] = {}
        for axis in axes.values():
            if axis.continuous:
                merged.update(axis.at(rng.random()))
            else:
                merged.update(axis.at_index(rng.randrange(axis.count)))
        yield merged


def _sampled(draws: Iterator[dict[str, Any]], samples: int, identify) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for _ in range(10 * samples):
        assignment = next(draws)
        key = identify(assignment)
        if key in seen:
            continue
        seen.add(key)
        yield assignment
        if len(seen) >= samples:
            return


def _check_random(samples: int | None, seed: int | None) -> int:
    if samples is None or seed is None:
        raise RecipeError("strategy='random' requires samples= and seed=.")
    if samples < 1:
        raise RecipeError(f"samples must be >= 1; got {samples}.")
    return samples


def assignments(
    axes: Mapping[str, Axis],
    *,
    strategy: Literal["grid", "random"] = "grid",
    samples: int | None = None,
    seed: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Enumerate or sample assignments of axes, each mapping every member path to a value.

    The grid is the product of the axes in order, each contributing its positions; a continuous
    axis or a `Text` range raises. Random sampling draws every axis independently with
    `random.Random(seed)` (one index for a discrete axis, one fraction for a continuous axis, so
    a `log` interval is sampled log-uniformly), drops repeated assignments, and stops after
    `samples` assignments or `10 * samples` draws.

    Args:
        axes: The axes, e.g., a subset of `Template.axes()`.
        strategy: `"grid"` or `"random"`.
        samples: The number of assignments to sample (random only).
        seed: The sampling seed (random only).

    Yields:
        One assignment per point.

    Raises:
        RecipeError: For a grid over a continuous or text axis, random sampling over a text axis,
            or random sampling without `samples` and `seed`.
    """
    if strategy == "grid":
        yield from _grid(axes)
        return
    if strategy != "random":
        raise RecipeError(f"strategy must be 'grid' or 'random'; got {strategy!r}.")
    count = _check_random(samples, seed)
    from steerability.algorithms.core.identity import canonical_json

    yield from _sampled(_draws(axes, seed), count, lambda assignment: canonical_json(assignment))


def expand(
    source: Template | Recipe,
    *,
    strategy: Literal["grid", "random"] = "grid",
    samples: int | None = None,
    seed: int | None = None,
) -> Iterator[Point]:
    """The points of a template, or the one point of a recipe.

    Each point is `Point(template.bind(a).recipe(), a)` for an assignment `a` of the template's
    axes (see `assignments`). Random points are deduplicated by recipe id. A recipe, or a template
    without axes, yields one point with an empty assignment.

    Args:
        source: The template or recipe.
        strategy: `"grid"` or `"random"`.
        samples: The number of points to sample (random only).
        seed: The sampling seed (random only).

    Yields:
        The points, in a deterministic order.

    Raises:
        RecipeError: As for `assignments`, or if a bound value is invalid.
    """
    if isinstance(source, Recipe):
        yield Point(source, {})
        return
    axes = source.axes()
    if not axes:
        yield Point(source.recipe(), {})
        return
    if strategy == "grid":
        for assignment in assignments(axes, strategy="grid"):
            yield Point(source.bind(assignment).recipe(), assignment)
        return
    if strategy != "random":
        raise RecipeError(f"strategy must be 'grid' or 'random'; got {strategy!r}.")
    count = _check_random(samples, seed)
    points: dict[str, Point] = {}

    def identify(assignment: dict[str, Any]) -> str:
        recipe = source.bind(assignment).recipe()
        points.setdefault(recipe.id, Point(recipe, assignment))
        return recipe.id

    for assignment in _sampled(_draws(axes, seed), count, identify):
        yield points[identify(assignment)]


__all__ = [
    "Axis",
    "BASELINE",
    "Choice",
    "Entry",
    "Interval",
    "Point",
    "Range",
    "Recipe",
    "RecipeError",
    "Template",
    "Text",
    "assignments",
    "entry",
    "expand",
]
