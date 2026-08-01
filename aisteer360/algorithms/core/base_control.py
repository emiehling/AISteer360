"""Shared base class for steering controls across all four categories."""
from abc import ABC
from dataclasses import fields

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.execution.capabilities import Capability
from aisteer360.algorithms.core.execution.requirements import Requirements, needs


class BaseControl(ABC):
    """Common construction and lifecycle for steering controls.

    Subclasses set `Args` to their hyperparameter dataclass; construction validates the arguments
    and mirrors every field onto the instance. A field whose name the subclass exposes as a
    `@property` is not mirrored (e.g. `CAST.condition_point`); the raw value stays reachable via
    `self.args.<name>`. Controls with `Args = None` accept no constructor arguments.

    Attributes:
        Args: The control's hyperparameter dataclass, or None for arg-free controls.
        RUNTIME_KWARGS_SCHEMA: Declarations for the per-call parameters the pipeline maps onto the
            control at inference time.
        enabled: Whether the control participates in the pipeline (identity controls set False).
        supports_batching: Whether the control processes a batched prompt in one call.
    """

    Args: type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = False

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is None:  # null / arg-free control
            if args or kwargs:
                raise TypeError(f"{type(self).__name__} accepts no constructor arguments.")
            self._configure()
            return

        self.args: BaseArgs = self.Args.validate(*args, **kwargs)

        # move fields to attributes, skipping any name the control exposes as a property
        # (e.g. CAST.condition_point); the raw value stays reachable via self.args.<name>
        for field in fields(self.args):
            if isinstance(getattr(type(self), field.name, None), property):
                continue
            setattr(self, field.name, getattr(self.args, field.name))

        self._configure()

    def _configure(self) -> None:
        """Post-construction hook, called after `Args` fields are mirrored onto the instance.

        Driver presets override this to map their mirrored args onto the fields their generic base
        reads, so subclasses never bypass a parent `__init__`. Default no-op.
        """
        pass

    def requirements(self) -> Requirements:
        """Backend requirements computed from this instance's configuration, per phase.

        The default requires `Capability.IN_PROCESS_TORCH` at generate and nothing at steer or
        score, which only the Hugging Face backend satisfies. A control with portable mechanisms
        overrides this to state weaker or alternative requirements. Configuration determines the
        result, so two configurations of one class may differ. Only enabled controls are
        consulted during support evaluation.

        Returns:
            The control's phase-keyed requirements.
        """
        return Requirements(generate=needs(Capability.IN_PROCESS_TORCH))

    def cleanup(self) -> None:
        """Release resources allocated during `steer()`.

        Override in subclasses that allocate GPU memory or other resources during steering. Default
        no-op.
        """
        pass
