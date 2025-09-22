from dataclasses import dataclass
import torch
import os
import numpy as np
from typing import Optional

@dataclass
class ClassAnchors:
    E: torch.Tensor  # (C, N), row i is anchor for class i

    def to(self, device):
        return ClassAnchors(self.E.to(device))

    @property
    def num_classes(self) -> int:
        return self.E.shape[0]

    @property
    def N(self) -> int:
        return self.E.shape[1]


def _orthonormal_rows(C: int, N: int, seed: int = 0) -> torch.Tensor:
    """Return C x N matrix with orthonormal rows (C <= N). Uses QR on Gaussian matrix.
    If C > N, falls back to normalized rows (not perfectly orthonormal)."""
    g = np.random.default_rng(seed).standard_normal(size=(N, C)).astype(np.float32)
    q, _ = np.linalg.qr(g, mode='reduced')  # q: (N, C) with orthonormal columns
    E = q.T  # (C, N) orthonormal rows
    if C > N:
        # Fallback: random normalized rows
        g2 = np.random.default_rng(seed + 1).standard_normal(size=(C, N)).astype(np.float32)
        E = g2 / (np.linalg.norm(g2, axis=1, keepdims=True) + 1e-9)
    return torch.from_numpy(E)


def build_or_load_anchors(path: str, C: int, N: int, seed: int = 0) -> ClassAnchors:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        E = torch.load(path, map_location='cpu')
        return ClassAnchors(E)
    E = _orthonormal_rows(C, N, seed=seed)
    # L2-normalize rows for cosine
    E = torch.nn.functional.normalize(E, p=2, dim=1)
    torch.save(E, path)
    return ClassAnchors(E)
