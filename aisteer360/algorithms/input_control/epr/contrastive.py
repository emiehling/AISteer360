"""Stage 2 of EPR: DPR-style contrastive training of the dense retriever.

Two `HFEncoder` instances (input encoder + prompt encoder) are initialized from a shared base; their parameters are
independent (not weight-tied). For each batch of B anchors, one positive and one hard negative are sampled per anchor.
Anchors are encoded with the input encoder, demos with the prompt encoder. Loss is `-log softmax` across each anchor's
positive vs. (in-batch positives from other anchors + the explicit hard negatives).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from aisteer360.algorithms.input_control.epr.encoders import HFEncoder

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """Hyperparameters for the Stage 2 contrastive trainer."""

    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-5
    n_negatives_per_anchor: int = 1
    """Hard negatives per anchor, in addition to in-batch negatives."""
    max_length: int = 256
    warmup_steps: int = 100
    weight_decay: float = 0.01
    log_every: int = 50
    seed: int = 0


def train_contrastive_retriever(
    labeled_data: list[dict],
    base_encoder_name_or_path: str,
    pooling: Literal["cls", "mean"],
    config: ContrastiveConfig,
    device: str = "cpu",
    demo_template: str = "Input: {input}\nOutput: {output}",
) -> tuple[HFEncoder, HFEncoder]:
    """DPR-style contrastive training.

    Args:
        labeled_data: Output of `generate_contrastive_data`.
        base_encoder_name_or_path: Starting weights for both input and prompt encoders.
        pooling: `"cls"` or `"mean"`.
        config: Hyperparameters.
        device: Torch device string.
        demo_template: Template applied to each (positive/negative) demo before encoding.

    Returns:
        `(input_encoder, prompt_encoder)` — both freshly trained `HFEncoder` instances.
    """
    if not labeled_data:
        raise ValueError("labeled_data must be non-empty")

    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    input_encoder = HFEncoder(
        base_encoder_name_or_path,
        pooling=pooling,
        batch_size=config.batch_size,
        device=device,
        max_length=config.max_length,
        trainable=True,
    )
    prompt_encoder = HFEncoder(
        base_encoder_name_or_path,
        pooling=pooling,
        batch_size=config.batch_size,
        device=device,
        max_length=config.max_length,
        trainable=True,
    )

    params = list(input_encoder.model.parameters()) + list(prompt_encoder.model.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = _build_warmup_scheduler(optimizer, warmup_steps=config.warmup_steps)

    input_encoder.model.train()
    prompt_encoder.model.train()

    n_neg = max(0, int(config.n_negatives_per_anchor))
    step = 0
    final_epoch_losses: list[float] = []
    first_epoch_first_loss: float | None = None

    for epoch in range(config.epochs):
        indices = list(range(len(labeled_data)))
        rng.shuffle(indices)
        epoch_losses: list[float] = []

        for batch_start in range(0, len(indices), config.batch_size):
            batch_idx = indices[batch_start:batch_start + config.batch_size]
            if len(batch_idx) < 2:
                continue  # need at least two anchors for in-batch negatives

            anchors_text: list[str] = []
            positives_text: list[str] = []
            negatives_text: list[str] = []
            valid_batch_idx: list[int] = []

            for j in batch_idx:
                row = labeled_data[j]
                positives = row.get("positives") or []
                negatives = row.get("negatives") or []
                if not positives:
                    continue
                pos = rng.choice(positives)
                neg_samples: list[dict] = []
                if n_neg > 0:
                    if len(negatives) >= n_neg:
                        neg_samples = rng.sample(negatives, k=n_neg)
                    elif negatives:
                        neg_samples = [rng.choice(negatives) for _ in range(n_neg)]
                    else:
                        neg_samples = []
                if n_neg > 0 and not neg_samples:
                    continue

                valid_batch_idx.append(j)
                anchors_text.append(row["anchor"]["input"])
                positives_text.append(demo_template.format(**pos))
                for nsample in neg_samples:
                    negatives_text.append(demo_template.format(**nsample))

            if len(valid_batch_idx) < 2:
                continue

            anchor_emb = input_encoder.forward_torch(anchors_text)        # [B, D]
            pos_emb = prompt_encoder.forward_torch(positives_text)        # [B, D]
            if negatives_text:
                neg_emb = prompt_encoder.forward_torch(negatives_text)    # [B*n_neg, D]
                doc_emb = torch.cat([pos_emb, neg_emb], dim=0)            # [B + B*n_neg, D]
            else:
                doc_emb = pos_emb                                          # [B, D]

            logits = anchor_emb @ doc_emb.T  # [B, B + B*n_neg]
            targets = torch.arange(anchor_emb.shape[0], device=anchor_emb.device)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = float(loss.item())
            epoch_losses.append(loss_val)
            if first_epoch_first_loss is None:
                first_epoch_first_loss = loss_val
            step += 1
            if step % config.log_every == 0:
                logger.info("step %d epoch %d loss %.4f", step, epoch, loss_val)

        if epoch_losses:
            avg = sum(epoch_losses) / len(epoch_losses)
            logger.info("epoch %d avg loss %.4f over %d steps", epoch, avg, len(epoch_losses))
            final_epoch_losses = epoch_losses

    if first_epoch_first_loss is not None and final_epoch_losses:
        logger.info(
            "contrastive training done: initial loss %.4f → final-epoch avg %.4f",
            first_epoch_first_loss,
            sum(final_epoch_losses) / len(final_epoch_losses),
        )

    input_encoder.model.eval()
    prompt_encoder.model.eval()
    for p in input_encoder.model.parameters():
        p.requires_grad_(False)
    for p in prompt_encoder.model.parameters():
        p.requires_grad_(False)
    input_encoder.trainable = False
    prompt_encoder.trainable = False

    return input_encoder, prompt_encoder


def _build_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    """Linear warmup → constant. No decay; matches the simple-but-effective DPR setup."""
    warmup_steps = max(1, int(warmup_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
