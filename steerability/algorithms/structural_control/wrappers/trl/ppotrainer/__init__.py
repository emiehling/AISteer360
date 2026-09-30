import logging

from steerability.algorithms.structural_control.wrappers.trl.ppotrainer.args import PPOArgs
from steerability.algorithms.structural_control.wrappers.trl.ppotrainer.base_mixin import (
    PPO_IMPORT_ERROR,
    PPO_IMPORT_HINT,
)
from steerability.algorithms.structural_control.wrappers.trl.ppotrainer.control import PPO

logger = logging.getLogger(__name__)

# discovery skips `ppo` when the TRL trainer is unavailable, since no `STEERING_METHOD` is exported
if PPO_IMPORT_ERROR is None:
    STEERING_METHOD = {
        "category": "structural_control",
        "name": "ppo",
        "control": PPO,
        "args": PPOArgs,
    }
else:
    logger.warning("Skipping ppo: %s", PPO_IMPORT_HINT.format(error=PPO_IMPORT_ERROR))
