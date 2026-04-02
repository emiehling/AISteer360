"""Angular direction estimator for learning orthonormal basis pairs."""
import logging
from typing import Literal, Sequence

import torch
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..specs import ContrastivePairs, VectorTrainSpec
from ..steering_vector import SteeringVector
from ..hook_utils import LAYERNORM_INPUT_MODULE, LAYERNORM_MID_MODULES, get_model_layer_list
from .base import BaseEstimator
from .contrastive_direction_estimator import _tokenize
from .utils import get_last_token_positions, select_at_positions

logger = logging.getLogger(__name__)

# Default sub-module suffixes whose *input* (captured via pre-hook) serves as
# a candidate activation for direction extraction.
DEFAULT_EXTRACT_MODULES: tuple[str, ...] = (
    *LAYERNORM_MID_MODULES,
    LAYERNORM_INPUT_MODULE,
)


# ------------------------------------------------------------------
# Activation extraction helpers
# ------------------------------------------------------------------


def _extract_activations(
    model: PreTrainedModel,
    enc: dict[str, torch.Tensor],
    extract_modules: Sequence[str],
    batch_size: int = 8,
) -> dict[tuple[int, str], torch.Tensor]:
    """Pre-hook named sub-modules to capture their input activations.

    Returns dict mapping ``(layer_idx, module_suffix)`` to ``[N, T, H]``
    CPU tensors.  The caller is responsible for selecting the desired
    token positions (e.g. via :func:`get_last_token_positions`).
    """
    _, layer_names = get_model_layer_list(model)
    num_layers = len(layer_names)
    # Extract the prefix from the first layer name (e.g. "model.layers" from "model.layers.0")
    prefix = layer_names[0].rsplit(".", 1)[0]
    module_dict = dict(model.named_modules())

    cache: dict[tuple[int, str], list[torch.Tensor]] = {}
    handles: list = []

    for layer_idx in range(num_layers):
        for mod_suffix in extract_modules:
            full_name = f"{prefix}.{layer_idx}.{mod_suffix}"
            if full_name in module_dict:

                def _make_hook(key: tuple[int, str]):
                    def hook_fn(module, args):
                        h = args[0] if isinstance(args, tuple) else args
                        cache.setdefault(key, []).append(h.detach().cpu())

                    return hook_fn

                handle = module_dict[full_name].register_forward_pre_hook(
                    _make_hook((layer_idx, mod_suffix))
                )
                handles.append(handle)

    # --- forward passes ---
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    N = input_ids.size(0)

    try:
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_ids = input_ids[start:end]
                batch_mask = (
                    attention_mask[start:end] if attention_mask is not None else None
                )
                model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                )
    finally:
        for h in handles:
            h.remove()

    # --- concatenate batches ---
    result: dict[tuple[int, str], torch.Tensor] = {}
    for key, tensors in cache.items():
        result[key] = torch.cat(tensors, dim=0)

    return result


# ------------------------------------------------------------------
# Estimator
# ------------------------------------------------------------------


class AngularDirectionEstimator(BaseEstimator[SteeringVector]):
    """Extracts orthonormal basis pairs for angular steering.

    For each ``(layer, module)`` pair, computes a candidate direction from
    the normalized mean difference of contrastive activations at the last
    token.  The best candidate (via ``strategy``) becomes the **global**
    ``first_direction`` shared by all layers; ``second_direction`` is the
    first PCA component across all candidates.

    Returns a :class:`SteeringVector` with ``K=2`` per layer.
    Orthonormalization is deferred to the consuming transform.

    Args:
        strategy: ``"max_sim"`` selects by highest mean pairwise cosine
            similarity; ``"max_norm"`` selects by largest candidate norm.
        extract_modules: Sub-module suffixes to pre-hook for activation
            capture.  Defaults cover LLaMA/Mistral/Qwen, Gemma, and the
            layer input.

    Reference:
        Vu & Nguyen, *Angular Steering*, `arXiv:2510.26243 <https://arxiv.org/abs/2510.26243>`_.
    """

    def __init__(
        self,
        strategy: Literal["max_sim", "max_norm"] = "max_sim",
        extract_modules: Sequence[str] = DEFAULT_EXTRACT_MODULES,
    ):
        self.strategy = strategy
        self.extract_modules = tuple(extract_modules)

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: ContrastivePairs,
        spec: VectorTrainSpec,
    ) -> SteeringVector:
        """Fit basis pairs from contrastive data.

        Only ``spec.batch_size`` is used; ``method`` and ``accumulate`` are
        ignored.
        """
        device = next(model.parameters()).device
        model_type = getattr(model.config, "model_type", "unknown")

        # build full texts
        if data.prompts is not None:
            pos_texts = [p + c for p, c in zip(data.prompts, data.positives)]
            neg_texts = [p + c for p, c in zip(data.prompts, data.negatives)]
        else:
            pos_texts = list(data.positives)
            neg_texts = list(data.negatives)

        logger.debug(
            "Tokenizing %d positive and %d negative examples",
            len(pos_texts),
            len(neg_texts),
        )

        enc_pos = _tokenize(tokenizer, pos_texts, device)
        enc_neg = _tokenize(tokenizer, neg_texts, device)

        # extract activations via pre-hooks on specified modules
        logger.debug(
            "Extracting hidden states with batch_size=%d, modules=%s",
            spec.batch_size,
            self.extract_modules,
        )
        acts_pos = _extract_activations(
            model, enc_pos, self.extract_modules, batch_size=spec.batch_size
        )
        acts_neg = _extract_activations(
            model, enc_neg, self.extract_modules, batch_size=spec.batch_size
        )

        num_pos = len(pos_texts)
        num_neg = len(neg_texts)

        # attention masks (CPU) for last-token selection
        attn_pos = enc_pos.get("attention_mask")
        attn_neg = enc_neg.get("attention_mask")
        if attn_pos is not None:
            attn_pos = attn_pos.cpu()
        if attn_neg is not None:
            attn_neg = attn_neg.cpu()

        # --- per-(layer, module) candidate directions ---
        candidate_directions: dict[tuple[int, str], torch.Tensor] = {}
        candidate_norms: dict[tuple[int, str], float] = {}

        for key in sorted(acts_pos.keys()):
            if key not in acts_neg:
                continue

            hp = acts_pos[key].float()  # [N, T, H]
            hn = acts_neg[key].float()

            # select last non-pad token
            pos_positions = get_last_token_positions(attn_pos, hp.size(1), num_pos)
            neg_positions = get_last_token_positions(attn_neg, hn.size(1), num_neg)
            hp_last = select_at_positions(hp, pos_positions)  # [N, H]
            hn_last = select_at_positions(hn, neg_positions)

            # per-sample L2-normalize
            hp_normed = hp_last / hp_last.norm(dim=-1, keepdim=True)
            hn_normed = hn_last / hn_last.norm(dim=-1, keepdim=True)

            # mean of normalized → normalize again
            hp_mean = hp_normed.mean(dim=0)
            hn_mean = hn_normed.mean(dim=0)
            hp_mean = hp_mean / hp_mean.norm()
            hn_mean = hn_mean / hn_mean.norm()

            # candidate direction — stored unnormalized so PCA captures
            # both direction and magnitude variation across layers
            diff = hp_mean - hn_mean
            candidate_norms[key] = float(diff.norm().item())
            candidate_directions[key] = diff

        if not candidate_directions:
            raise RuntimeError(
                "No candidate directions could be computed. "
                "Check that positive and negative data are non-empty."
            )

        # --- PCA across ALL candidates ---
        sorted_keys = sorted(candidate_directions.keys())
        all_candidates = torch.stack(
            [candidate_directions[k] for k in sorted_keys]
        )  # [num_candidates, H]
        pca = PCA()
        pca.fit(all_candidates.numpy())
        second_direction = torch.from_numpy(pca.components_[0]).float()

        # --- selection scores ---
        all_normed = all_candidates / (
            all_candidates.norm(dim=-1, keepdim=True) + 1e-8
        )
        pairwise = all_normed @ all_normed.T
        mean_cosine = pairwise.mean(dim=-1)  # [num_candidates]

        similarity_scores: dict[tuple[int, str], float] = {
            k: float(mean_cosine[i].item()) for i, k in enumerate(sorted_keys)
        }

        if self.strategy == "max_sim":
            scores = similarity_scores
        else:
            scores = candidate_norms

        # --- select best (layer, module) → global first_direction ---
        best_key = max(scores, key=scores.get)
        best_layer, best_module = best_key

        # normalize only the selected first_direction
        first_direction = candidate_directions[best_key]
        first_direction = first_direction / (first_direction.norm() + 1e-8)

        logger.info(
            "Angular steering: selected layer %d (%s) via %s (score=%.4f)",
            best_layer,
            best_module,
            self.strategy,
            scores[best_key],
        )

        # --- build SteeringVector with GLOBAL first_direction ---
        # All layers share the same (first_direction, second_direction) pair,
        # matching the reference implementation.
        num_layers = max(lid for (lid, _) in candidate_directions.keys()) + 1
        directions: dict[int, torch.Tensor] = {}
        for layer_id in range(num_layers):
            # [2, H]: row 0 = first_direction, row 1 = second_direction
            directions[layer_id] = torch.stack(
                [first_direction, second_direction], dim=0
            )

        # explained_variances keyed by layer_id (int): best score across
        # modules for each layer, so control.py auto-selection works.
        explained_variances: dict[int, float] = {}
        for (lid, _mod), score in scores.items():
            if lid not in explained_variances or score > explained_variances[lid]:
                explained_variances[lid] = score

        logger.debug(
            "Finished fitting angular directions for %d layers "
            "(%d candidates from %s, strategy=%s)",
            num_layers,
            len(candidate_directions),
            self.extract_modules,
            self.strategy,
        )
        return SteeringVector(
            model_type=model_type,
            directions=directions,
            explained_variances=explained_variances,
        )
