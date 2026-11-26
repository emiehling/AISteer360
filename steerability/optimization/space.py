"""Structure expansion: which controls a pipeline may contain, declared as slots and expanded into a study's space.

A slot is one position of a pipeline with its alternatives. `optional` makes a control (or a slot) optional, and
`one_of` makes several controls exclusive. The call `structures(optional(prompt), optional(caa))` gives the
structures `prompt`, `caa`, and `prompt+caa`. The result is the mapping from structure name to `Template` that
`Study.space` takes, and nothing downstream depends on how it was built.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from steerability.algorithms.core.recipe import Entry, RecipeError, Template


@dataclass(frozen=True, kw_only=True)
class Slot:
    """The alternatives for one position of a pipeline; `None` stands for no control.

    Attributes:
        alternatives: The entries the position may contain, and `None` when it may contain no control.

    Raises:
        TypeError: If an alternative is neither an `Entry` nor None.
        RecipeError: If the slot has no alternative, or an entry is disabled.
    """

    alternatives: tuple[Entry | None, ...]

    def __post_init__(self) -> None:
        alternatives = tuple(self.alternatives)
        for item in alternatives:
            if item is not None and not isinstance(item, Entry):
                raise TypeError(f"A slot's alternatives are Entry objects or None; got {type(item).__name__}.")
            if item is not None and not item.enabled:
                raise RecipeError(
                    f"A slot's entries are enabled; the entry of {item.method} is disabled. A None alternative stands "
                    "for no control."
                )
        if not alternatives:
            raise RecipeError("A slot needs at least one alternative.")
        object.__setattr__(self, "alternatives", alternatives)


def one_of(*entries: Entry) -> Slot:
    """Exactly one of the entries.

    Args:
        *entries: The entries, in the order their structures are listed.

    Returns:
        The slot.

    Raises:
        TypeError: If an operand is not an `Entry`.
        RecipeError: If no entry is given, or an entry is disabled.
    """
    for item in entries:
        if not isinstance(item, Entry):
            raise TypeError(f"one_of takes Entry objects; got {type(item).__name__}.")
    return Slot(alternatives=entries)


def optional(item: Entry | Slot, /) -> Slot:
    """The entry, or one of the slot's alternatives, or no control.

    Args:
        item: An entry, or a slot whose alternatives the position may contain.

    Returns:
        The slot, with no control as its last alternative.

    Raises:
        TypeError: If `item` is neither an `Entry` nor a `Slot`.
    """
    if isinstance(item, Entry):
        return Slot(alternatives=(item, None))
    if isinstance(item, Slot):
        return Slot(alternatives=(*(entry for entry in item.alternatives if entry is not None), None))
    raise TypeError(f"optional takes an Entry or a Slot; got {type(item).__name__}.")


def structures(*slots: Entry | Slot) -> dict[str, Template]:
    """Every combination of the slots as a template, keyed by structure name.

    A bare `Entry` is a slot with one alternative, i.e., it is in every structure. The combinations are the Cartesian
    product of the slots without the empty combination, since the unsteered model is the study's baseline and not a
    structure. Each combination becomes a `Template` of its present entries, which the template orders by category
    (structural, input, state, output), and its name is `"+".join(template.names)` (e.g., `prompt+caa`, or `lora+caa`
    for a weight change and a steering vector). The product is enumerated with each slot's entries in the order
    given and no control last, and the structures are then sorted by their number of entries (a stable sort). No
    model or backend work is done, and whether a combination is supported is checked by `Study.validate`.

    Since an entry has one name in every combination, each of its axes has the same key in every structure that
    contains it (`caa.layer_id` in `caa` and in `prompt+caa`). A combination with two free ordered axes fails
    `Study.classify`, unless all but one are listed in `study.searched` (which applies to the whole study) or the two
    share an `axis=` name. Controls that exclude each other, such as two decoding drivers, belong in one `one_of`
    slot. Structures that contain the same weight-changing entry with the same values share its product, i.e., it is
    trained once for all of them.

    Args:
        *slots: The slots, each an `Entry` or a `Slot`.

    Returns:
        Mapping from structure name to template, in the order above.

    Raises:
        TypeError: If an operand is neither an `Entry` nor a `Slot`.
        RecipeError: If no slot is given, two entries share a method key and one of them has no explicit `name=`
            (unnamed entries of one method are numbered by their position in a template, and one entry would get
            different axis keys in different structures), two entries or two combinations resolve to the same name,
            or a slot is invalid (see `Slot`).
    """
    if not slots:
        raise RecipeError("structures needs at least one slot.")
    columns: list[list[Entry | None]] = []
    for slot in slots:
        if isinstance(slot, Entry):
            slot = Slot(alternatives=(slot,))
        elif not isinstance(slot, Slot):
            raise TypeError(f"structures takes Entry and Slot operands; got {type(slot).__name__}.")
        present = [item for item in slot.alternatives if item is not None]
        columns.append([*present, None] if len(present) < len(slot.alternatives) else present)
    _check_names(entries=[item for column in columns for item in column if item is not None])
    combinations = [
        [item for item in combination if item is not None] for combination in itertools.product(*columns)
    ]
    expanded: dict[str, Template] = {}
    for entries in sorted((entries for entries in combinations if entries), key=len):
        template = Template(entries=tuple(entries))
        name = "+".join(template.names)
        if name in expanded:
            raise RecipeError(f"Two combinations resolve to the structure name {name!r}; rename an entry.")
        expanded[name] = template
    return expanded


def _check_names(*, entries: list[Entry]) -> None:
    """Raise `RecipeError` unless every entry has one name in every combination and no two entries share a name."""
    by_method: dict[str, list[Entry]] = {}
    for item in entries:
        by_method.setdefault(item.method, []).append(item)
    for method, items in by_method.items():
        if len(items) > 1 and any(item.name is None for item in items):
            raise RecipeError(
                f"{len(items)} entries of {method} are given, and not all have an explicit name=. Unnamed entries of "
                "one method are numbered by their position in a template; give each a distinct name=."
            )
    names = [item.name if item.name is not None else item.method.partition("/")[2] for item in entries]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise RecipeError(f"Entries resolve to the same name(s) {repeated}; give each entry a distinct name=.")


__all__ = ["Slot", "one_of", "optional", "structures"]
