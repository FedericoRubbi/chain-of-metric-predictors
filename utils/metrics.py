import torch
from torch import Tensor
from typing import Tuple


def top1_accuracy(logits: Tensor, targets: Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def cross_entropy_from_probs(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """H(p, q) = - sum_y p_y * log(q_y). p and q are (B, C) probs (sum to 1)."""
    q = q.clamp_min(eps)
    return -(p * q.log()).sum(dim=1).mean()
