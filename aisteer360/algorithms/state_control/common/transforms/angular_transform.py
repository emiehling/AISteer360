"""Angular steering transform — rotates activations in a learned 2D plane."""
import torch

from .base import BaseTransform


class AngularTransform(BaseTransform):
    """Rotates hidden states within a 2D steering plane.

    Given an orthonormal basis ``(b1, b2)`` spanning the steering plane, the
    transform projects each hidden state onto that plane and replaces the
    in-plane component with a rotated version:

        ``h' = h − P·h + ‖P·h‖ · v_θ``

    where:
        - ``P  = b1⊗b1ᵀ + b2⊗b2ᵀ``  is the projection onto the plane
        - ``v_θ = cos(θ)·b1 + sin(θ)·b2``  is the target direction
        - ``θ``  is ``target_degree`` converted to radians

    The out-of-plane component is preserved exactly; only the in-plane
    component's direction changes while its magnitude is held constant.

    **Adaptive mode** (``adaptive_mode=1``) adds per-token gating: a position
    is steered only when ``h·b1 > 0``, i.e. the activation already has a
    positive projection onto the first direction.  With ``adaptive_mode=0``
    all masked positions are steered unconditionally.

    Args:
        directions: Per-layer basis tensors.  Shape ``[2, H]`` per layer
            where ``directions[layer_id][0]`` is the first (raw) direction and
            ``directions[layer_id][1]`` is the second (raw) direction.
            Orthonormalization is performed internally and cached.
        target_degree: Rotation angle in degrees.  ``0`` aligns with the first
            direction; ``180`` is maximally opposite; ``90`` / ``270`` are
            perpendicular.
        adaptive_mode: ``0`` = always steer (non-adaptive).
            ``1`` = only steer positions whose activation has positive alignment
            with the first direction (recommended).
    """

    def __init__(
        self,
        directions: dict[int, torch.Tensor],
        target_degree: float = 180.0,
        adaptive_mode: int = 1,
    ):
        self.directions = directions
        self.target_degree = target_degree
        self.adaptive_mode = adaptive_mode
        # cache: (layer_id, device, dtype) → (b1, b2, proj_matrix)
        self._basis_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_basis(
        self,
        layer_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached ``(b1, b2, P)`` for *layer_id* on *device*/*dtype*."""
        key = (layer_id, device, dtype)
        if key not in self._basis_cache:
            d = self.directions[layer_id].to(device=device, dtype=dtype)
            # Gram–Schmidt orthonormalization
            b1 = d[0] / (d[0].norm() + 1e-8)
            b2 = d[1] - (d[1] @ b1) * b1
            b2 = b2 / (b2.norm() + 1e-8)
            proj = torch.outer(b1, b1) + torch.outer(b2, b2)
            self._basis_cache[key] = (b1, b2, proj)
        return self._basis_cache[key]

    # ------------------------------------------------------------------
    # BaseTransform interface
    # ------------------------------------------------------------------

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply angular rotation to hidden states.

        Args:
            hidden_states: Shape ``[B, T, H]``.
            layer_id: Which layer this is being applied at.
            token_mask: Shape ``[B, T]``.  True at positions eligible for
                steering (e.g. generated tokens only).
            **kwargs: Ignored.

        Returns:
            Modified hidden states, same shape as input.
        """
        if layer_id not in self.directions:
            return hidden_states

        device = hidden_states.device
        dtype = hidden_states.dtype
        b1, b2, proj = self._get_basis(layer_id, device, dtype)

        # target direction
        theta = torch.tensor(
            self.target_degree * torch.pi / 180.0,
            device=device,
            dtype=dtype,
        )
        v_theta = torch.cos(theta) * b1 + torch.sin(theta) * b2

        # project onto steering plane
        proj_h = hidden_states @ proj.T  # [B, T, H]
        r = proj_h.norm(dim=-1, keepdim=True)  # [B, T, 1]

        # rotated states
        steered = hidden_states - proj_h + r * v_theta  # [B, T, H]

        # build effective mask
        mask = token_mask  # [B, T]
        if self.adaptive_mode == 1:
            alignment = hidden_states @ b1  # [B, T]
            mask = mask & (alignment > 0)

        return torch.where(mask.unsqueeze(-1), steered, hidden_states)
