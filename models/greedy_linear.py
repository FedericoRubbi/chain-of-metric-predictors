from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn


@dataclass
class GreedyConfig:
    input_dim: int
    N: int
    layers: int
    similarity: str  # 'cosine' or 'l2'
    tau: float
    lambda_ace: float


class GreedyLinearNet(nn.Module):
    """Stack of Linear(+bias)+ReLU blocks of width N. First layer maps input_dim→N, others N→N.
    Exposes forward that returns all intermediate embeddings [h1, h2, ..., hL]."""
    def __init__(self, cfg: GreedyConfig):
        super().__init__()
        self.cfg = cfg
        mods: List[nn.Module] = []
        in_dim = cfg.input_dim
        # Build main Linear+ReLU stack
        for i in range(cfg.layers):
            mods.append(nn.Linear(in_dim, cfg.N, bias=True))
            mods.append(nn.ReLU(inplace=True))
            in_dim = cfg.N
        # remove final trailing ReLU? Keep it, as spec says Linear+ReLU for each layer.
        self.net = nn.Sequential(*mods)
        # For convenience, also keep references to each Linear and ReLU
        self.linears = nn.ModuleList([m for m in self.net if isinstance(m, nn.Linear)])
        self.relus = nn.ModuleList([m for m in self.net if isinstance(m, nn.ReLU)])

        # Residual stream: send input of each layer to its output (projection if needed)
        residuals: List[nn.Module] = []
        in_dim = cfg.input_dim
        for i in range(cfg.layers):
            if in_dim == cfg.N:
                residuals.append(nn.Identity())
            else:
                residuals.append(nn.Linear(in_dim, cfg.N, bias=False))
            in_dim = cfg.N
        self.residuals = nn.ModuleList(residuals)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.linears:
            nn.init.kaiming_uniform_(lin.weight, a=5**0.5)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    @torch.no_grad()
    def embeddings_no_grad(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Convenience for eval without grad: returns list of embeddings per layer (after ReLU)."""
        hs: List[torch.Tensor] = []
        h = x
        for i in range(1, self.cfg.layers + 1):
            h = self.forward_layer_from(h, i)
            hs.append(h)
        return hs

    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward returning [h1, h2, ..., hL] with grad enabled."""
        hs: List[torch.Tensor] = []
        h = x
        for i in range(1, self.cfg.layers + 1):
            h = self.forward_layer_from(h, i)
            hs.append(h)
        return hs

    def forward_layer_from(self, h_prev: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Compute h_i from given input h_{i-1} for specific layer index (1-based for ReLU outputs)."""
        # Each layer is (Linear_i, ReLU_i) pair in sequence
        lin = self.linears[layer_index - 1]
        relu = self.relus[layer_index - 1]
        res = self.residuals[layer_index - 1]
        return relu(lin(h_prev) + res(h_prev))
