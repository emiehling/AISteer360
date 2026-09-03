"""Sequence-classification heads for the Granite and GraniteMoeHybrid architectures.

transformers ships no `AutoModelForSequenceClassification` head for the Granite families, so a Granite
causal-LM checkpoint cannot be loaded as a scalar-head reward model out of the box. These two classes
supply the head by mixing `GenericForSequenceClassification` (the pooled-last-token classifier the
Llama and Qwen heads use) with each family's `PreTrainedModel`, and `register_granite_sequence_classifiers`
registers them with `AutoModelForSequenceClassification`. A head shipped by a future transformers
version wins, since the registration is skipped when the mapping already covers the config.
"""
from __future__ import annotations

import logging

from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, AutoModelForSequenceClassification
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.models.granite.configuration_granite import GraniteConfig
from transformers.models.granite.modeling_granite import GranitePreTrainedModel
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridPreTrainedModel

logger = logging.getLogger(__name__)


class GraniteForSequenceClassification(GenericForSequenceClassification, GranitePreTrainedModel):
    """Sequence-classification head for the `granite` architecture."""


class GraniteMoeHybridForSequenceClassification(GenericForSequenceClassification, GraniteMoeHybridPreTrainedModel):
    """Sequence-classification head for the `granitemoehybrid` architecture."""


def register_granite_sequence_classifiers() -> None:
    """Register the Granite heads with `AutoModelForSequenceClassification` when transformers ships none.

    Idempotent, and a head shipped by a future transformers version wins: the registration for a
    config is skipped when `MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING` already covers it. When the lazy
    mapping does not support `in`, the `ValueError` `register` raises for an existing key is caught.
    """
    for config_cls, model_cls in (
        (GraniteConfig, GraniteForSequenceClassification),
        (GraniteMoeHybridConfig, GraniteMoeHybridForSequenceClassification),
    ):
        try:
            if config_cls in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING:
                continue
        except (KeyError, TypeError):
            pass
        try:
            AutoModelForSequenceClassification.register(config_cls, model_cls)
        except ValueError:
            logger.debug("Sequence-classification head already registered for %s", config_cls.__name__)
