"""
Centralized metrics collection system for MLP and GMLP training.

This module provides a unified interface for collecting all metrics during training,
handling the complexity of metric computation across different architectures.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Any
from utils.advanced_metrics import (
    compute_layer_metrics, 
    GaussianEntropyEstimator,
    cosine_alignment_metrics,
    layerwise_accuracy,
    layerwise_cross_entropy,
    ace_regularizer,
    mutual_information_estimate,
    one_shot_linear_probe,
    participation_ratio,
    linear_cka,
    ridge_regression_r2
)
from utils.scores import scores_from_embeddings, softmax_with_temperature


class MetricsCollector:
    """Centralized metrics collection for both MLP and GMLP architectures."""
    
    def __init__(self, num_classes: int, N: int, similarity: str = 'cosine', tau: float = 1.0, lambda_reg: float = 1e-3):
        """
        Initialize metrics collector.
        
        Args:
            num_classes: Number of classes
            N: Embedding dimension
            similarity: Similarity type for anchor-based metrics ('cosine' or 'l2')
            tau: Temperature for softmax
            lambda_reg: Regularization parameter for ridge regression
        """
        self.num_classes = num_classes
        self.N = N
        self.similarity = similarity
        self.tau = tau
        self.lambda_reg = lambda_reg
        
        # Per-layer entropy estimators
        self.entropy_estimators: List[GaussianEntropyEstimator] = []
        
        # Anchors for cosine-based metrics
        self.anchors: Optional[torch.Tensor] = None
        
    def set_anchors(self, anchors: torch.Tensor):
        """Set class anchors for cosine-based metrics."""
        self.anchors = anchors
        
    def initialize_entropy_estimators(self, num_layers: int, alpha: float = 0.9):
        """Initialize entropy estimators for all layers."""
        self.entropy_estimators = [GaussianEntropyEstimator(alpha=alpha) for _ in range(num_layers)]
    
    def compute_layer_logits_and_probs(self, embeddings: torch.Tensor) -> tuple[Tensor, Tensor]:
        """Compute logits and probabilities from embeddings using anchors."""
        if self.anchors is None:
            raise ValueError("Anchors must be set before computing logits and probabilities")
        
        scores = scores_from_embeddings(embeddings, self.anchors, similarity=self.similarity)
        probs = softmax_with_temperature(scores, self.tau)
        return scores, probs
    
    def collect_mlp_metrics(
        self, 
        embeddings_list: List[torch.Tensor], 
        logits_list: List[torch.Tensor], 
        labels: torch.Tensor,
        inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Collect metrics for MLP architecture.
        
        Args:
            embeddings_list: List of (B, N) embeddings for each layer
            logits_list: List of (B, C) logits for each layer
            labels: (B,) ground truth labels
            inputs: (B, D) input vectors (optional)
        
        Returns:
            Dict with metrics for each layer
        """
        metrics = {}
        
        for i, (embeddings, logits) in enumerate(zip(embeddings_list, logits_list)):
            layer_key = f'layer_{i}'
            
            # Compute probabilities if anchors are available
            probs_current = None
            if self.anchors is not None:
                _, probs_current = self.compute_layer_logits_and_probs(embeddings)
            
            # Get previous and next layer embeddings
            embeddings_prev = embeddings_list[i-1] if i > 0 else None
            embeddings_next = embeddings_list[i+1] if i < len(embeddings_list) - 1 else None
            
            # Get probabilities for ACE regularizer
            probs_next = None
            if embeddings_next is not None and self.anchors is not None:
                _, probs_next = self.compute_layer_logits_and_probs(embeddings_next)
            
            # Get entropy estimator
            entropy_estimator = self.entropy_estimators[i] if i < len(self.entropy_estimators) else None
            
            # Compute all metrics for this layer
            layer_metrics = compute_layer_metrics(
                embeddings=embeddings,
                logits=logits,
                labels=labels,
                anchors=self.anchors if self.anchors is not None else None,
                embeddings_prev=embeddings_prev,
                embeddings_next=embeddings_next,
                probs_current=probs_current,
                probs_next=probs_next,
                inputs=inputs,
                entropy_estimator=entropy_estimator,
                lambda_reg=self.lambda_reg
            )
            
            metrics[layer_key] = layer_metrics
        
        return metrics
    
    def collect_gmlp_metrics(
        self, 
        embeddings_list: List[torch.Tensor], 
        labels: torch.Tensor,
        inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Collect metrics for GMLP architecture.
        
        Args:
            embeddings_list: List of (B, N) embeddings for each layer
            labels: (B,) ground truth labels
            inputs: (B, D) input vectors (optional)
        
        Returns:
            Dict with metrics for each layer
        """
        metrics = {}
        
        for i, embeddings in enumerate(embeddings_list):
            layer_key = f'layer_{i}'
            
            # Compute logits and probabilities using anchors
            logits, probs_current = self.compute_layer_logits_and_probs(embeddings)
            
            # Get previous and next layer embeddings
            embeddings_prev = embeddings_list[i-1] if i > 0 else None
            embeddings_next = embeddings_list[i+1] if i < len(embeddings_list) - 1 else None
            
            # Get probabilities for ACE regularizer
            probs_next = None
            if embeddings_next is not None:
                _, probs_next = self.compute_layer_logits_and_probs(embeddings_next)
            
            # Get entropy estimator
            entropy_estimator = self.entropy_estimators[i] if i < len(self.entropy_estimators) else None
            
            # Compute all metrics for this layer
            layer_metrics = compute_layer_metrics(
                embeddings=embeddings,
                logits=logits,
                labels=labels,
                anchors=self.anchors if self.anchors is not None else None,
                embeddings_prev=embeddings_prev,
                embeddings_next=embeddings_next,
                probs_current=probs_current,
                probs_next=probs_next,
                inputs=inputs,
                entropy_estimator=entropy_estimator,
                lambda_reg=self.lambda_reg
            )
            
            metrics[layer_key] = layer_metrics
        
        return metrics
    
    def collect_batch_metrics(
        self, 
        embeddings_list: List[torch.Tensor], 
        labels: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        architecture: str = 'gmlp'
    ) -> Dict[str, Any]:
        """
        Collect metrics for a single batch.
        
        Args:
            embeddings_list: List of (B, N) embeddings for each layer
            labels: (B,) ground truth labels
            inputs: (B, D) input vectors (optional)
            architecture: 'mlp' or 'gmlp'
        
        Returns:
            Dict with metrics for each layer
        """
        if architecture == 'mlp':
            # For MLP, we need logits from the model
            # This will be handled by the trainer
            raise NotImplementedError("MLP metrics collection requires logits from model")
        elif architecture == 'gmlp':
            return self.collect_gmlp_metrics(embeddings_list, labels, inputs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def get_summary_metrics(self, layer_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute summary metrics across all layers.
        
        Args:
            layer_metrics: Dict with metrics for each layer
        
        Returns:
            Dict with summary metrics
        """
        summary = {}
        
        # Extract metrics that can be summarized
        metrics_to_summarize = [
            'accuracy', 'cross_entropy', 'alignment', 'margin', 
            'ace_regularizer', 'mutual_information', 'gaussian_entropy',
            'participation_ratio', 'linear_cka', 'ridge_r2'
        ]
        
        for metric_name in metrics_to_summarize:
            values = []
            for layer_key, layer_data in layer_metrics.items():
                if metric_name in layer_data:
                    values.append(layer_data[metric_name])
            
            if values:
                summary[f'{metric_name}_mean'] = sum(values) / len(values)
                summary[f'{metric_name}_std'] = torch.tensor(values).std().item() if len(values) > 1 else 0.0
                summary[f'{metric_name}_min'] = min(values)
                summary[f'{metric_name}_max'] = max(values)
        
        # Special handling for probe metrics
        probe_accuracies = []
        probe_f1_scores = []
        for layer_key, layer_data in layer_metrics.items():
            if 'accuracy' in layer_data and 'f1_score' in layer_data:
                probe_accuracies.append(layer_data['accuracy'])
                probe_f1_scores.append(layer_data['f1_score'])
        
        if probe_accuracies:
            summary['probe_accuracy_mean'] = sum(probe_accuracies) / len(probe_accuracies)
            summary['probe_f1_mean'] = sum(probe_f1_scores) / len(probe_f1_scores)
        
        return summary
