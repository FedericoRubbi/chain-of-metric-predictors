from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn


@dataclass
class MlpConfig:
    input_dim: int
    N: int
    layers: int
    num_classes: int


class MlpBaseline(nn.Module):
    def __init__(self, cfg: MlpConfig):
        super().__init__()
        mods = []
        in_dim = cfg.input_dim
        for _ in range(cfg.layers):
            mods += [nn.Linear(in_dim, cfg.N, bias=True), nn.ReLU(inplace=True)]
            in_dim = cfg.N
        self.backbone = nn.Sequential(*mods)
        self.head = nn.Linear(cfg.N, cfg.num_classes, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning all intermediate embeddings."""
        embeddings = []
        h = x
        
        # Process through backbone layers
        for i, module in enumerate(self.backbone):
            h = module(h)
            # Collect embeddings after ReLU activations
            if isinstance(module, torch.nn.ReLU):
                embeddings.append(h)
        
        return embeddings
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)
