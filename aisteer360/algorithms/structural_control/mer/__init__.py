from .args import MERArgs
from .control import MER

STEERING_METHOD = {
    "category": "structural_control",
    "name": "mer",
    "control": MER,
    "args": MERArgs,
}
