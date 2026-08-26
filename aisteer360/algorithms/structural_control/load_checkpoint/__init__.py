from aisteer360.algorithms.structural_control.load_checkpoint.args import LoadCheckpointArgs
from aisteer360.algorithms.structural_control.load_checkpoint.control import LoadCheckpoint

STEERING_METHOD = {
    "category": "structural_control",
    "name": "load_checkpoint",
    "control": LoadCheckpoint,
    "args": LoadCheckpointArgs,
}
