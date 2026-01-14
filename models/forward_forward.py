from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for torch.func availability (PyTorch 2.0+)
try:
    from torch.func import jacrev, vmap
except ImportError:
    try:
        from functorch import jacrev, vmap
    except ImportError:
        raise ImportError("Forward-Forward implementation requires PyTorch 2.0+ (torch.func) or functorch.")

def goodness(activation, theta=0.0, gamma=0.0):
    # Calculates goodness for a single sample (vmap handles batching)
    # energy is scalar per sample
    energy = torch.sum(activation ** 2) + gamma - theta
    inverse_goodness = (1 + torch.exp(-energy))
    return 1 / inverse_goodness

class FFBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 activation=None,
                 theta=2.0,
                 bias=False):
        super().__init__()

        self.theta = theta
        self.activation = activation if activation is not None else nn.LeakyReLU()
        
        self.W = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # Using Kaiming init consistent with other models in repo
        nn.init.kaiming_uniform_(self.W.weight, a=5**0.5)
        if self.W.bias is not None:
            nn.init.zeros_(self.W.bias)

    def forward(self, x):
        return self.activation(self.W(x))
    
    def forward_goodness(self, weight, x, gamma):
        # Functional forward for checking goodness (used in jacrev)
        # weight passed explicitly for jacrev
        x = F.linear(x, weight, bias=self.W.bias)
        x = self.activation(x)
        return goodness(x, theta=self.theta, gamma=gamma)
    
    def train_step(self, x, labels, lr=0.01, gamma=0.0):
        # Update weights based on Goodness gradient
        # labels: +1 for positive data, -1 for negative data
        
        # Use torch.func to get gradients of goodness w.r.t weights
        # vmap over batch dimension (dim 0 of x and gamma)
        # self.W.weight is constant across batch (in_dims=None)
        
        # Handle gamma shape: if it's (B,), vmap expects it.
        # If it's scalar 0, we might need to expand it or handle it.
        # Ideally gamma should be aligned with x.
        
        goodness_grad_fn = jacrev(self.forward_goodness)
        
        # dW calculation using vmap
        # gradients = vmap(goodness_grad_fn, (None, 0, 0))(self.W.weight, x, gamma) 
        # But wait, gamma needs to be (B,) if mapped.
        # If gamma is scalar 0.0, we need to handle that.
        
        if isinstance(gamma, (int, float)) and gamma == 0:
            # Create a tensor of zeros if gamma is scalar 0
            gamma_tensor = torch.zeros(x.shape[0], device=x.device)
        elif isinstance(gamma, torch.Tensor) and gamma.ndim == 0:
            gamma_tensor = gamma.expand(x.shape[0])
        else:
            gamma_tensor = gamma

        dW = vmap(goodness_grad_fn, in_dims=(None, 0, 0))(
            self.W.weight, x, gamma_tensor
        )
        
        # maximize goodness for positive labels, min for negative (by multiplying by label)
        # labels is (B,) or (B,1). dW is (B, Out, In).
        if labels.ndim == 1:
            labels = labels.view(-1, 1, 1)
        else:
             labels = labels.view(labels.shape[0], 1, 1)
             
        # Mean gradient across batch
        dW = torch.mean(dW * labels, dim=0)
        
        # Update weights (gradient ascent on goodness)
        # Note: Code snippet used: self.W.weight.data += lr * dW
        with torch.no_grad():
            self.W.weight.data += lr * dW


@dataclass
class ForwardForwardConfig:
    input_dim: int
    N: int
    layers: int
    num_classes: int
    theta: float = 2.0  # Threshold


class ForwardForwardNet(nn.Module):
    def __init__(self, cfg: ForwardForwardConfig):
        super().__init__()
        self.cfg = cfg
        self.theta = cfg.theta
        self.num_classes = cfg.num_classes
        
        self.layers = nn.ModuleList()
        
        # First layer concatenates input and one-hot labels
        # Layer 0: input_dim + num_classes -> N
        self.layers.append(FFBlock(
            in_dim=cfg.input_dim + cfg.num_classes,
            out_dim=cfg.N,
            theta=cfg.theta
        ))
        
        # Hidden layers: N -> N
        prev_dim = cfg.N
        for i in range(cfg.layers - 1):
            self.layers.append(FFBlock(
                in_dim=prev_dim,
                out_dim=cfg.N,
                theta=cfg.theta
            ))
            
    def forward(self, x: torch.Tensor, return_energy: bool = False):
        if return_energy:
            energies = []
            for layer in self.layers:
                x = layer(x)
                # energy = sum(x^2) per sample
                energies.append((x**2).sum(dim=-1))
            return x, energies
        else:
            for layer in self.layers:
                x = layer(x)
            return x

    def train_step_layer(self, x: torch.Tensor, y: torch.Tensor, layer_idx: int, lr: float = 0.01, gamma: torch.Tensor = 0.0):
        """
        Executes a training step for a specific layer.
        Generates negative samples by masking labels.
        """
        device = x.device
        
        # Prepare input with concatenated labels
        y_hot = F.one_hot(y, num_classes=self.num_classes).float()
        
        # Negative data generation:
        # Create an array like y, but not the same label
        y_neg_probs = torch.rand(y_hot.shape, device=device) * (1.0 - y_hot) # Mask out true class
        y_neg_idx = y_neg_probs.argmax(dim=1)
        y_neg = F.one_hot(y_neg_idx, num_classes=self.num_classes).float()

        # Create a random mask for selection (50% positive, 50% negative samples in batch)
        mask = torch.rand(y.shape[0], device=device) > 0.5
        
        # Construct mixed labels for input
        y_mixed = y_hot.clone()
        y_mixed[mask] = y_neg[mask]
            
        # pos_neg_lab: vector that is +1 for pos, -1 for neg samples
        # mask True -> y_neg used -> negative sample -> label -1
        # mask False -> y_hot used -> positive sample -> label +1
        pos_neg_lab = (mask.float() * -2) + 1

        # Concatenate x and mixed labels
        # Ensure x is flattened if needed, typically handled by caller or View
        x_in = torch.cat([x, y_mixed], dim=-1)

        # Forward through previous layers (they act as fixed feature extractors for current layer)
        # Using no_grad to save memory and ensure no graph building for previous layers
        with torch.no_grad():
            for i in range(layer_idx):
                x_in = self.layers[i](x_in)
        
        # Train step on current layer
        self.layers[layer_idx].train_step(x_in, pos_neg_lab, lr=lr, gamma=gamma)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict labels by testing goodness for all possible labels.
        accumulates goodness across all layers.
        """
        batch_size = x.size(0)
        device = x.device
        
        # Generate one-hots for all possible labels
        # labels: (num_classes, num_classes)
        labels = torch.arange(self.num_classes, device=device)
        labels_hot = F.one_hot(labels, num_classes=self.num_classes).float() # (C, C)
        
        # Expand for batch: (C, B, C)
        labels_hot = labels_hot.unsqueeze(1).repeat(1, batch_size, 1)
        
        # Expand x: (C, B, input_dim)
        x_expanded = x.unsqueeze(0).repeat(self.num_classes, 1, 1)
        
        # Concat: (C, B, input_dim + num_classes)
        x_in = torch.cat([x_expanded, labels_hot], dim=-1)
        
        # Flatten to (C*B, ...) for forward pass
        x_in_flat = x_in.view(-1, x_in.size(-1))
        
        # Compute energies
        _, energies_list = self.forward(x_in_flat, return_energy=True)
        # energies_list: List of (B*C,) tensors.
        
        # Sum energies across all layers
        total_energy = torch.stack(energies_list).sum(dim=0) # (B*C,)
        
        # Reshape back to (C, B)
        # x_in_flat was (C*B). total_energy has elements 0..B-1 for label 0, B..2B-1 for label 1.
        # view(C, B) matches this.
        total_energy = total_energy.view(self.num_classes, batch_size)
        
        # Argmax over Class dimension (dim 0)
        preds = total_energy.argmax(dim=0) # (B,)
        
        return preds
