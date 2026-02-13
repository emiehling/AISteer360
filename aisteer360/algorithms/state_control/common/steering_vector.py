"""Steering vector: per-layer direction vectors learned by an estimator."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class SteeringVector:
    """Per-layer direction vectors for activation steering.

    Attributes:
        model_type: HuggingFace model_type string (e.g., "llama").
        directions: Mapping from layer_id to direction tensor of shape [H].
        explained_variances: Mapping from layer_id to explained variance scalar.
    """

    model_type: str
    directions: dict[int, torch.Tensor]
    explained_variances: dict[int, float]

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SteeringVector":
        """Move all direction tensors to device/dtype. Returns self for chaining."""
        self.directions = {
            k: v.to(device=device, dtype=dtype) if dtype else v.to(device=device)
            for k, v in self.directions.items()
        }
        return self

    def validate(self) -> None:
        """Validate that required fields are populated.

        Raises:
            ValueError: If model_type or directions are empty.
        """
        if not self.model_type:
            raise ValueError("model_type must be provided.")
        if not self.directions:
            raise ValueError("directions must not be empty.")

    def save(self, file_path: str) -> None:
        """Save the SteeringVector to a JSON file.

        Args:
            file_path: Path to save to. ".svec" extension added if not present.
        """
        if not file_path.endswith(".svec"):
            file_path += ".svec"
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = {
            "model_type": self.model_type,
            "directions": {str(k): v.tolist() for k, v in self.directions.items()},
            "explained_variances": {str(k): v for k, v in self.explained_variances.items()},
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
        logger.debug("Saved SteeringVector to %s", file_path)

    @classmethod
    def load(cls, file_path: str) -> "SteeringVector":
        """Load a SteeringVector from a JSON file.

        Args:
            file_path: Path to load from. ".svec" extension added if not present.

        Returns:
            Loaded SteeringVector instance.
        """
        if not file_path.endswith(".svec"):
            file_path += ".svec"
        with open(file_path) as f:
            data = json.load(f)
        directions = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in data["directions"].items()}
        explained_variances = {int(k): float(v) for k, v in data["explained_variances"].items()}
        logger.debug("Loaded SteeringVector from %s with layers %s", file_path, list(directions.keys()))
        return cls(model_type=data["model_type"], directions=directions, explained_variances=explained_variances)
