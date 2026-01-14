from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn


@dataclass
class MonoForwardConfig:
    input_dim: int
    N: int
    layers: int
    num_classes: int  # Added: Required for the local learnable projections


class MonoForwardNet(nn.Module):
    def __init__(self, cfg: MonoForwardConfig):
        super().__init__()
        self.cfg = cfg
        
        # 1. Main Backbone (Linear + ReLU)
        # Note: No residual stream is created here.
        self.linears = nn.ModuleList()
        self.relus = nn.ModuleList()
        
        in_dim = cfg.input_dim
        for _ in range(cfg.layers):
            self.linears.append(nn.Linear(in_dim, cfg.N, bias=True))
            self.relus.append(nn.ReLU(inplace=True))
            in_dim = cfg.N

        # 2. Local Classifiers (The Mono-Forward "Projection Matrices")
        # Each layer i has a learnable matrix M_i mapping hidden_dim -> num_classes
        self.classifiers = nn.ModuleList([
            nn.Linear(cfg.N, cfg.num_classes, bias=True) 
            for _ in range(cfg.layers)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_step(self, h_prev: torch.Tensor, layer_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes layer i.
        Returns:
            h_curr: Activation to pass to layer i+1
            logits: Predictions for local loss calculation at layer i
        """
        # 0-based indexing assumed for internal lists
        lin = self.linears[layer_index]
        relu = self.relus[layer_index]
        classifier = self.classifiers[layer_index]

        # 1. Standard Forward (No Norm, No Residual)
        # Mono-Forward uses raw activations.
        z = lin(h_prev)
        h_curr = relu(z)

        # 2. Local Prediction (Learnable Projection)
        # This replaces the fixed anchor similarity check.
        logits = classifier(h_curr)

        return h_curr, logits

    def forward_all(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Convenience method to get all activations and all local logits.
        Useful for validation/inference, though training is usually done loop-wise.
        """
        activations = []
        all_logits = []
        
        h = x
        for i in range(self.cfg.layers):
            h, logits = self.forward_step(h, i)
            activations.append(h)
            all_logits.append(logits)
            
        return activations, all_logits
