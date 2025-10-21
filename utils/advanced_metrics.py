"""
Advanced metrics for comprehensive analysis of MLP and GMLP architectures.

This module implements all the metrics specified in the METRICS_IMPLEMENTATION_PLAN.md
to answer three key research questions:
1. Are later layers getting more predictive?
2. Are layers compressing?
3. Are layers genuinely different from previous layers?
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, mutual_info_score
from sklearn.decomposition import PCA


def cosine_alignment_metrics(embeddings: torch.Tensor, anchors: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute cosine alignment and margin metrics for layer embeddings.
    
    Args:
        embeddings: (B, N) layer embeddings
        anchors: (C, N) class anchors
        labels: (B,) ground truth labels
    
    Returns:
        Dict with 'alignment' and 'margin' metrics
    """
    # Ensure both tensors have the same dtype to avoid RuntimeError
    embeddings = embeddings.to(anchors.dtype)
    
    # Normalize embeddings and anchors
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # (B, N)
    anchors_norm = F.normalize(anchors, p=2, dim=1)  # (C, N)
    
    # Compute cosine similarities
    cos_sim = torch.mm(embeddings_norm, anchors_norm.t())  # (B, C)
    
    # Get similarities to correct anchors
    correct_sim = cos_sim.gather(1, labels.unsqueeze(1)).squeeze(1)  # (B,)
    
    # Get max similarity to incorrect anchors
    mask = torch.ones_like(cos_sim, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    incorrect_sim = cos_sim.masked_select(mask).view(cos_sim.size(0), -1).max(1)[0]  # (B,)
    
    alignment = correct_sim.mean().item()
    margin = (correct_sim - incorrect_sim).mean().item()
    
    return {'alignment': alignment, 'margin': margin}


def layerwise_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy for a single layer."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def layerwise_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute cross-entropy loss for a single layer."""
    ce_loss = F.cross_entropy(logits, labels, reduction='mean')
    return ce_loss.item()


def ace_regularizer(probs_current: torch.Tensor, probs_next: torch.Tensor) -> float:
    """
    Compute ACE regularizer between consecutive layers.
    
    Args:
        probs_current: (B, C) softmax probabilities from current layer
        probs_next: (B, C) softmax probabilities from next layer
    
    Returns:
        ACE regularizer value
    """
    eps = 1e-12
    probs_next_clamped = probs_next.clamp_min(eps)
    ace = -(probs_current * probs_next_clamped.log()).sum(dim=1).mean()
    return ace.item()


def mutual_information_estimate(inputs: torch.Tensor, outputs: torch.Tensor, bins: int = 50) -> float:
    """
    Estimate mutual information between inputs and outputs using histogram method.
    
    Args:
        inputs: (B, D) input vectors
        outputs: (B, N) output vectors
        bins: number of bins for histogram estimation
    
    Returns:
        Estimated mutual information
    """
    # Robust MI estimation with sanitization and degenerate-case handling
    try:
        # Flatten and detach for MI estimation
        inputs_flat = inputs.detach().view(inputs.size(0), -1).float()
        outputs_flat = outputs.detach().view(outputs.size(0), -1).float()

        X_in = inputs_flat.cpu().numpy()
        X_out = outputs_flat.cpu().numpy()

        # Replace NaN/Inf and clip extreme values to keep PCA stable
        X_in = np.nan_to_num(X_in, nan=0.0, posinf=1e6, neginf=-1e6)
        X_out = np.nan_to_num(X_out, nan=0.0, posinf=1e6, neginf=-1e6)

        if not np.isfinite(X_in).all() or not np.isfinite(X_out).all():
            return 0.0

        # First principal component for MI estimation
        pca_input = PCA(n_components=1).fit_transform(X_in)
        pca_output = PCA(n_components=1).fit_transform(X_out)

        a_min, a_max = float(np.min(pca_input)), float(np.max(pca_input))
        b_min, b_max = float(np.min(pca_output)), float(np.max(pca_output))

        # Guard degenerate ranges
        if not np.isfinite([a_min, a_max, b_min, b_max]).all():
            return 0.0
        if a_min == a_max or b_min == b_max:
            return 0.0

        # Ensure at least 2 bins and strictly monotonic edges
        bins = max(int(bins), 2)
        input_edges = np.linspace(a_min, a_max, bins)
        output_edges = np.linspace(b_min, b_max, bins)
        input_edges = np.unique(input_edges)
        output_edges = np.unique(output_edges)
        if input_edges.shape[0] < 2 or output_edges.shape[0] < 2:
            return 0.0

        input_discrete = np.digitize(pca_input.ravel(), input_edges)
        output_discrete = np.digitize(pca_output.ravel(), output_edges)

        mi = mutual_info_score(input_discrete, output_discrete)
        if not np.isfinite(mi):
            return 0.0
        return float(mi)
    except Exception:
        # Any numerical issue should not crash training; treat MI as 0
        return 0.0


class GaussianEntropyEstimator:
    """Estimator for Gaussian entropy proxy using EMA of activations."""
    
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha  # EMA decay rate
        self.running_mean = None
        self.running_var = None
    
    def update(self, activations: torch.Tensor) -> float:
        """
        Update running statistics and return entropy estimate.
        
        Args:
            activations: (B, N) layer activations
        
        Returns:
            Gaussian entropy estimate
        """
        mean = activations.mean(dim=0)  # (N,)
        var = activations.var(dim=0)  # (N,)
        
        if self.running_mean is None:
            self.running_mean = mean
            self.running_var = var
        else:
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * var
        
        # Gaussian entropy: H = 0.5 * sum(log(2*pi*e*sigma^2))
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * self.running_var).sum()
        return entropy.item()


def one_shot_linear_probe(embeddings: torch.Tensor, labels: torch.Tensor, lambda_reg: float = 1e-3) -> Dict[str, float]:
    """
    Perform one-shot linear probe using ridge regression for CLASSIFICATION.
    
    This metric measures how well a layer's representations can be classified using
    a simple linear classifier. It answers: "How predictive are this layer's features?"
    
    Args:
        embeddings: (B, N) layer embeddings
        labels: (B,) ground truth labels
        lambda_reg: regularization parameter
    
    Returns:
        Dict with 'accuracy' and 'f1_score' - classification performance metrics
    """
    # Convert to float32 and sanitize to avoid NaN/Inf issues from mixed precision
    X = embeddings.detach().float().cpu().numpy()
    y = labels.detach().cpu().numpy()
    
    # Replace NaN/Inf and clip extremes
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Optional light standardization to stabilize solver
    # Avoid zero-variance features by adding small epsilon
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    Xs = (X - mean) / std
    
    try:
        # Ridge regression for CLASSIFICATION: embeddings → labels
        clf = RidgeClassifier(alpha=lambda_reg)
        clf.fit(Xs, y)
        
        # Predictions
        y_pred = clf.predict(Xs)
        
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        return {'probe_accuracy': float(accuracy), 'f1_score': float(f1)}
    except Exception:
        # On any numerical error, return safe defaults
        return {'probe_accuracy': 0.0, 'f1_score': 0.0}


def participation_ratio(embeddings: torch.Tensor) -> float:
    """
    Compute participation ratio from activation covariance.
    
    Args:
        embeddings: (B, N) layer embeddings
    
    Returns:
        Participation ratio
    """
    # Convert to float32 for eigenvalue computation (linalg.eigvals doesn't support Half)
    embeddings = embeddings.float()
    
    # Center the data
    embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov = torch.mm(embeddings_centered.t(), embeddings_centered) / (embeddings.size(0) - 1)
    
    # Compute eigenvalues
    eigenvals = torch.linalg.eigvals(cov).real
    
    # Participation ratio
    sum_eigenvals = eigenvals.sum()
    sum_squared_eigenvals = (eigenvals ** 2).sum()
    
    pr = (sum_eigenvals ** 2) / sum_squared_eigenvals
    return pr.item()


def linear_cka(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> float:
    """
    Compute linear CKA between two sets of embeddings.
    
    Args:
        embeddings1: (B, N) first layer embeddings
        embeddings2: (B, N) second layer embeddings
    
    Returns:
        Linear CKA similarity
    """
    # Ensure both tensors have the same dtype
    embeddings1 = embeddings1.to(embeddings2.dtype)
    
    # Center the embeddings
    emb1_centered = embeddings1 - embeddings1.mean(dim=0, keepdim=True)
    emb2_centered = embeddings2 - embeddings2.mean(dim=0, keepdim=True)
    
    # Compute gram matrices
    gram1 = torch.mm(emb1_centered, emb1_centered.t())
    gram2 = torch.mm(emb2_centered, emb2_centered.t())
    
    # Compute CKA
    numerator = torch.trace(torch.mm(gram1, gram2))
    denominator = torch.sqrt(torch.trace(torch.mm(gram1, gram1)) * torch.trace(torch.mm(gram2, gram2)))
    
    cka = numerator / denominator
    return cka.item()


def ridge_regression_r2(embeddings_current: torch.Tensor, embeddings_next: torch.Tensor, lambda_reg: float = 1e-3) -> float:
    """
    Compute R² of ridge regression between consecutive layers for REGRESSION.
    
    This metric measures how much of the next layer's information is linearly
    predictable from the current layer. It answers: "How redundant are consecutive layers?"
    
    Args:
        embeddings_current: (B, N) current layer embeddings
        embeddings_next: (B, N) next layer embeddings
        lambda_reg: regularization parameter
    
    Returns:
        R² score - coefficient of determination (0 = no linear relationship, 1 = perfect linear relationship)
    """
    X = embeddings_current.cpu().numpy()
    y = embeddings_next.cpu().numpy()
    
    # Ridge regression for REGRESSION: layer_ℓ → layer_ℓ+1
    ridge = Ridge(alpha=lambda_reg)
    ridge.fit(X, y)
    
    # Predictions
    y_pred = ridge.predict(X)
    
    # R² score
    r2 = r2_score(y, y_pred, multioutput='uniform_average')
    return r2


# Convenience function to compute all metrics for a single layer
def compute_layer_metrics(
    embeddings: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    anchors: Optional[torch.Tensor] = None,
    embeddings_prev: Optional[torch.Tensor] = None,
    embeddings_next: Optional[torch.Tensor] = None,
    probs_current: Optional[torch.Tensor] = None,
    probs_next: Optional[torch.Tensor] = None,
    inputs: Optional[torch.Tensor] = None,
    entropy_estimator: Optional[GaussianEntropyEstimator] = None,
    lambda_reg: float = 1e-3
) -> Dict[str, float]:
    """
    Compute all available metrics for a single layer.
    
    Args:
        embeddings: (B, N) layer embeddings
        logits: (B, C) layer logits
        labels: (B,) ground truth labels
        anchors: (C, N) class anchors (optional)
        embeddings_prev: (B, N) previous layer embeddings (optional)
        embeddings_next: (B, N) next layer embeddings (optional)
        probs_current: (B, C) current layer probabilities (optional)
        probs_next: (B, C) next layer probabilities (optional)
        inputs: (B, D) input vectors (optional)
        entropy_estimator: GaussianEntropyEstimator instance (optional)
        lambda_reg: regularization parameter
    
    Returns:
        Dict with all computed metrics
    """
    metrics = {}
    
    # Basic metrics (always available)
    metrics['accuracy'] = layerwise_accuracy(logits, labels)
    metrics['cross_entropy'] = layerwise_cross_entropy(logits, labels)
    
    # Cosine alignment metrics (if anchors provided)
    if anchors is not None:
        cosine_metrics = cosine_alignment_metrics(embeddings, anchors, labels)
        metrics.update(cosine_metrics)
    
    # ACE regularizer (if probabilities provided)
    if probs_current is not None and probs_next is not None:
        metrics['ace_regularizer'] = ace_regularizer(probs_current, probs_next)
    
    # Mutual information (if inputs provided) with guard against numerical issues
    if inputs is not None:
        try:
            mi_val = mutual_information_estimate(inputs, embeddings)
            if np.isfinite(mi_val):
                metrics['mutual_information'] = mi_val
        except Exception:
            # Skip MI if it fails; other metrics remain
            pass
    
    # Gaussian entropy (if estimator provided)
    if entropy_estimator is not None:
        metrics['gaussian_entropy'] = entropy_estimator.update(embeddings)
    
    # One-shot linear probe (guarded)
    try:
        probe_metrics = one_shot_linear_probe(embeddings, labels, lambda_reg)
        metrics.update(probe_metrics)
    except Exception:
        pass
    
    # Participation ratio
    metrics['participation_ratio'] = participation_ratio(embeddings)
    
    # Linear CKA (if previous layer provided)
    if embeddings_prev is not None:
        metrics['linear_cka'] = linear_cka(embeddings_prev, embeddings)
    
    # Ridge regression R² (if next layer provided)
    if embeddings_next is not None:
        metrics['ridge_r2'] = ridge_regression_r2(embeddings, embeddings_next, lambda_reg)
    
    return metrics
