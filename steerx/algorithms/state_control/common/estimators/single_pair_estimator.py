"""Single-pair estimator for ActAdd steering vectors."""
import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..steering_vector import SteeringVector
from .base import BaseEstimator

logger = logging.getLogger(__name__)


class SinglePairEstimator(BaseEstimator[SteeringVector]):
    """Extracts per-token positional steering vectors from a single prompt pair.

    This is the estimator used by ActAdd. Given one positive prompt and one
    negative prompt, it computes the per-token activation difference at every
    layer (or a specified subset of layers), preserving the full positional
    structure of the contrast.

    Unlike MeanDifferenceEstimator (which averages over many pairs and collapses
    to a single direction), this produces a [T, H] direction matrix per layer,
    where T is the token length of the (padded) prompt pair.
    """

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        positive_prompt: str,
        negative_prompt: str,
        layer_ids: list[int] | None = None,
    ) -> SteeringVector:
        """Extract positional steering vector from a single prompt pair.

        Args:
            model: Model to extract hidden states from.
            tokenizer: Tokenizer for encoding the prompts.
            positive_prompt: Prompt representing the desired direction
                (e.g., "Love", "I talk about weddings constantly").
            negative_prompt: Prompt representing the opposite direction
                (e.g., "Hate", "I do not talk about weddings constantly").
            layer_ids: If provided, only compute directions for these layers.
                If None, compute for all layers.

        Returns:
            SteeringVector with [T, H] directions per layer.
        """
        device = next(model.parameters()).device
        model_type = getattr(model.config, "model_type", "unknown")

        logger.debug("Tokenizing prompt pair: positive=%r, negative=%r", positive_prompt, negative_prompt)

        # tokenize both prompts together for consistent padding
        enc = tokenizer(
            [positive_prompt, negative_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logger.debug("Running forward pass to extract hidden states")

        # forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
            )

        # outputs.hidden_states: tuple of (num_layers+1) tensors of shape [2, T, H]
        # index 0 is embedding output; 1..N are layer outputs
        directions: dict[int, torch.Tensor] = {}
        explained_variances: dict[int, float] = {}

        num_layers = len(outputs.hidden_states) - 1  # exclude embedding output
        logger.debug("Computing per-token difference for %d layers", num_layers)

        for layer_idx, hs in enumerate(outputs.hidden_states[1:]):
            if layer_ids is not None and layer_idx not in layer_ids:
                continue

            h_pos = hs[0]  # [T, H]
            h_neg = hs[1]  # [T, H]

            direction = (h_pos - h_neg).cpu().to(dtype=torch.float32)  # [T, H]

            directions[layer_idx] = direction
            # single-pair extraction doesn't produce explained variance; use 1.0 as sentinel
            explained_variances[layer_idx] = 1.0

        logger.debug("Finished fitting single-pair directions with T=%d tokens", direction.size(0))
        return SteeringVector(
            model_type=model_type,
            directions=directions,
            explained_variances=explained_variances,
        )