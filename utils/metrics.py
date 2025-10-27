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


def entropy_from_probs(p: Tensor, eps: float = 1e-12) -> Tensor:
    """Entropy H(p) for probabilities p with numerical stability."""
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=1).mean()


def kl_from_probs(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """KL divergence D_KL(p || q) for probabilities with stability."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=1).mean()


def js_from_probs(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """Jensen–Shannon divergence between two probability distributions."""
    m = 0.5 * (p + q)
    return 0.5 * (kl_from_probs(p, m, eps) + kl_from_probs(q, m, eps))


def symmetric_kl_from_probs(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """Symmetric KL: D_KL(p||q) + D_KL(q||p)."""
    return kl_from_probs(p, q, eps) + kl_from_probs(q, p, eps)
