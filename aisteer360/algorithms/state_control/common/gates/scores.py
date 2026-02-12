"""Score functions used by gates to evaluate condition signals."""
import torch


@torch.no_grad()
def projected_cosine_similarity(
    hidden_state: torch.Tensor,
    projector: torch.Tensor,
) -> float:
    """Compute cosine similarity between a vector and its projection.

    This is the scoring function used by CAST for condition detection:
    project the hidden state through the condition subspace projector,
    apply tanh, then compute cosine similarity with the original.

    Args:
        hidden_state: Shape [H] - aggregated hidden state.
        projector: Shape [H, H] - outer-product projection matrix.

    Returns:
        Cosine similarity as a float.
    """
    projected = torch.tanh(projector @ hidden_state)
    sim = torch.dot(hidden_state, projected) / (
        hidden_state.norm() * projected.norm() + 1e-8
    )
    return float(sim.item())
