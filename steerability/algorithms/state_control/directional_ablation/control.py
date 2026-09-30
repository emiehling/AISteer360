"""Directional Ablation control: projects a learned direction out of the residual stream."""
from __future__ import annotations

from steerability.algorithms.state_control.base import InterventionControl
from steerability.algorithms.state_control.common.estimators import estimator_for
from steerability.algorithms.state_control.common.selectors import FractionalDepthSelector
from steerability.algorithms.state_control.common.sources import ContrastiveFit, LayerFilteredFit, _Precomputed
from steerability.algorithms.state_control.common.specs import CoveredLayers, Intervention, TokenScope
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.transforms import NormPreservingTransform, ProjectionTransform
from steerability.algorithms.state_control.common.transforms.base import unwrap_modifiers

from .args import DirectionalAblationArgs


class DirectionalAblation(InterventionControl):
    """Directional Ablation (feature removal via projection).

    Removes a learned feature direction from the residual stream at one or more layers during
    generation. At each steered position, the hidden state is updated as
    `h' = h - alpha * (d_hat^T h) d_hat`. This is the abliteration technique of Arditi et al.,
    which learns a direction as the difference in means over contrastive data and projects it out.

    The method operates in two phases:

    1. **Training (offline)**: extract residual activations for the contrastive pairs in `data` and
       take the mean difference (`train_spec.method="mean_diff"`) or a PCA direction
       (`"pca_pairwise"`, `"pca_center"`) as the feature direction. A precomputed direction, or a
       subspace of `K > 1` directions (orthonormalized before use), may be given directly as
       `steering_vector`.
    2. **Inference (online)**: at the output of each target layer, project the direction out of the
       residual stream at the positions selected by `token_scope`. `alpha = 1.0` fully removes the
       component (`h'.d_hat == 0`), and `alpha < 1.0` gives graded partial suppression.

    The target layers are `layer_ids` (or one layer at about 40% depth when None) intersected with
    the layers that have a direction, after the directions are restricted to `layer_range`.
    `steer()` raises `ValueError` when no target layer has a direction. Ablation is a projection,
    which is idempotent at `alpha=1` and reduces the norm of the hidden state.

    Reference:

    - "Refusal in Language Models Is Mediated by a Single Direction"
    Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
    [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)
    """

    Args = DirectionalAblationArgs
    supports_batching = True

    def _configure(self):
        if self.steering_vector is not None:
            inner = _Precomputed(self.steering_vector.clone())
        else:
            inner = ContrastiveFit(
                data=self.data,
                estimator=estimator_for(self.train_spec.method),
                estimator_kwargs={"spec": self.train_spec},
            )
        source = LayerFilteredFit(inner, layer_range=self.layer_range)

        transform = ProjectionTransform(source, alpha=self.alpha)
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)

        self._template = (Intervention(
            # heuristic default: single layer at ~40% depth (matches CAA)
            layers=CoveredLayers(
                within=tuple(sorted(set(self.layer_ids))) if self.layer_ids is not None
                       else FractionalDepthSelector(fraction=0.4)
            ),
            transform=transform,
            scope=TokenScope(self.token_scope, last_k=self.last_k, from_position=self.from_position),
        ),)

    @property
    def hook_only_hint(self) -> str:
        if self.alpha != 1.0:
            return "graded ablation (alpha < 1) has no intervention-spec form; run on the huggingface backend"
        return "subspace ablation has no intervention-spec form; run on the huggingface backend"

    @property
    def _layer_ids(self) -> list[int]:
        """The resolved target layers (empty before `steer()`)."""
        return list(self.interventions[0].layers) if self.interventions else []

    @property
    def _steering_vector(self) -> SteeringVector | None:
        """The bound steering artifact as a `SteeringVector` view (None before `steer()`)."""
        if not self.interventions:
            return None
        core, _ = unwrap_modifiers(self.interventions[0].transform)
        if getattr(core, "directions", None) is None:
            return None
        return SteeringVector(
            model_type="unknown",
            directions=core.directions,
            meta=core.artifact_meta or {},
        )
