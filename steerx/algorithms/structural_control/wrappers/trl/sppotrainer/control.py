from steerx.algorithms.structural_control.wrappers.trl.sppotrainer.args import (
    SPPOArgs,
)
from steerx.algorithms.structural_control.wrappers.trl.sppotrainer.base_mixin import (
    SPPOTrainerMixin,
)


class SPPO(SPPOTrainerMixin):
    """
    SPPO controller.
    """
    Args = SPPOArgs
