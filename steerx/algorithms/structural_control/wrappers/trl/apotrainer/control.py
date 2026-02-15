from steerx.algorithms.structural_control.wrappers.trl.apotrainer.args import (
    APOArgs,
)
from steerx.algorithms.structural_control.wrappers.trl.dpotrainer.base_mixin import (
    DPOTrainerMixin,
)


class APO(DPOTrainerMixin):
    """
    APO controller.
    """
    Args = APOArgs
