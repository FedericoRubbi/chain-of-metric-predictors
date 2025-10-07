import torch
from torch import Tensor


def scores_from_embeddings(h: Tensor, anchors: Tensor, similarity: str = 'cosine') -> Tensor:
    """Compute per-class scores G(x,y) given embeddings h (B, N) and anchors (C, N).
    For 'cosine', returns cosine similarity. For 'l2', returns negative squared L2 distances."""
    if similarity == 'cosine':
        h_norm = torch.nn.functional.normalize(h, p=2, dim=1)  # (B, N)
        # anchors are expected L2-normalized row-wise
        # Ensure both tensors have the same dtype for matrix multiplication
        anchors_T = anchors.T.to(h_norm.dtype)
        return h_norm @ anchors_T  # (B, C)
    elif similarity == 'l2':
        # ||h - e||^2 = ||h||^2 + ||e||^2 - 2 h·e
        # Ensure both tensors have the same dtype for matrix multiplication
        anchors_T = anchors.T.to(h.dtype)
        h2 = (h * h).sum(dim=1, keepdim=True)  # (B, 1)
        e2 = (anchors * anchors).sum(dim=1).unsqueeze(0).to(h.dtype)  # (1, C)
        dot = h @ anchors_T  # (B, C)
        return -(h2 + e2 - 2.0 * dot)
    else:
        raise ValueError(f"Unknown similarity: {similarity}")


def softmax_with_temperature(logits: Tensor, tau: float) -> Tensor:
    return torch.softmax(logits / tau, dim=-1)
