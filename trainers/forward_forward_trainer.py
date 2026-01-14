from typing import Dict, List, Tuple, Optional
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.metrics import top1_accuracy
from utils.schedulers import WarmupCosine
from utils.logger import JsonlLogger
from utils.optimized_metrics_collector import OptimizedMetricsCollector
from models.forward_forward import ForwardForwardNet, ForwardForwardConfig
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from datetime import datetime

class DummyOptimizer:
    """A dummy optimizer to allow using PyTorch schedulers with manual update rules."""
    def __init__(self, lr: float):
        self.param_groups = [{'lr': lr}]
    
    def zero_grad(self, set_to_none=False):
        pass
        
    def step(self):
        pass

class ForwardForwardTrainer:
    def __init__(self, run_dir: str, model: ForwardForwardNet, cfg: Dict, num_classes: int, dataset: str):
        self.run_dir = run_dir
        self.model = model
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset = dataset

        # FF uses manual updates, so we use a dummy optimizer to track LR for the scheduler
        self.optimizer = DummyOptimizer(lr=cfg['lr'])
        self.logger = JsonlLogger(run_dir)
        
        # Initialize metrics collector
        enable_slow_metrics = self.cfg.get('enable_slow_metrics', False)
        self.metrics_collector = OptimizedMetricsCollector(
            num_classes=num_classes,
            N=cfg['N'],
            similarity='cosine', # Not strictly used but required by init
            tau=1.0, 
            lambda_reg=cfg.get('lambda_reg', 1e-3),
            enable_slow_metrics=enable_slow_metrics
        )
        self.metrics_collector.initialize_entropy_estimators(cfg['layers'])
        self.console = Console()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        total_steps = len(train_loader) * epochs
        warmup = int(self.cfg['warmup_ratio'] * total_steps)
        self.scheduler = WarmupCosine(self.optimizer, warmup_steps=warmup, total_steps=total_steps)

        best_val = 0.0
        best_path = os.path.join(self.run_dir, 'best.pt')
        global_step = 0
        
        with Progress(
            TextColumn("[bold blue]Training"),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            training_task = progress.add_task(f"Training for {epochs} epochs", total=epochs)
            
            for epoch in range(1, epochs + 1):
                self.model.train()
                epoch_task = progress.add_task(f"Epoch {epoch}/{epochs}", total=len(train_loader))
                
                # Forward-Forward training loop
                # We iterate through batches, and for each batch, we update all layers sequentially.
                # This corresponds to "Collaborative" or "Joint" training in the FF context,
                # ensuring all layers see data as it streams in.
                
                for it, (x, y) in enumerate(train_loader, start=1):
                    x = x.to(self.device).view(x.size(0), -1)
                    y = y.to(self.device)
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Update each layer
                    # Note: train_step_layer handles the forward pass of previous layers internally
                    # using no_grad to freeze them.
                    for layer_idx in range(self.cfg['layers']):
                        self.model.train_step_layer(
                            x, y, 
                            layer_idx=layer_idx, 
                            lr=current_lr,
                            gamma=0.0 # Standard FF (can be modified for collaborative energy interaction)
                        )
                    
                    self.scheduler.step()
                    global_step += 1
                    
                    # Log step (Lightweight, no full eval on every step for speed)
                    if global_step % self.cfg.get('metrics_log_frequency', 10) == 0:
                        log_entry = {
                            'phase': 'train',
                            'epoch': epoch,
                            'iter': it,
                            'step': global_step,
                            'lr': current_lr,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.logger.log(log_entry)

                    progress.update(epoch_task, advance=1)
                
                progress.remove_task(epoch_task)
                
                # Validation at end of epoch
                val_acc, val_metrics = self.evaluate(val_loader)
                
                log_entry = {
                    'phase': 'val',
                    'epoch': epoch,
                    'iter': 0,
                    'acc': val_acc,
                    'timestamp': datetime.now().isoformat()
                }
                if val_metrics:
                    log_entry['metrics'] = val_metrics
                self.logger.log(log_entry)
                
                if val_acc > best_val:
                    best_val = val_acc
                    torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, best_path)
                
                progress.update(training_task, advance=1, description=f"Training for {epochs} epochs - Val Acc: {val_acc:.4f}")

        torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, os.path.join(self.run_dir, 'last.pt'))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Optional[Dict]]:
        self.model.eval()
        total = 0
        correct = 0
        all_metrics = []
        
        # FF Evaluation is slower (runs C passes), so careful with large datasets if frequently validating
        for x, y in loader:
            x = x.to(self.device).view(x.size(0), -1)
            y = y.to(self.device)
            B = x.size(0)

            # Get predictions and energies
            # predict() returns class indices
            preds = self.model.predict(x)
            correct += (preds == y).sum().item()
            total += B
            
            # To get metrics, we need "logits". 
            # We can re-use the energy calculation logic from predict directly here to avoid double computation
            # but predict is efficient. Let's replicate logic to get energies for metrics.
            
            # Reconstruct what predict does to get energies
            labels = torch.arange(self.num_classes, device=self.device)
            labels_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
            labels_hot = labels_hot.unsqueeze(1).repeat(1, B, 1) # (C, B, C)
            x_expanded = x.unsqueeze(0).repeat(self.num_classes, 1, 1) # (C, B, In)
            x_in = torch.cat([x_expanded, labels_hot], dim=-1)
            x_in_flat = x_in.view(-1, x_in.size(-1))
            
            # Get energies
            _, energies_list = self.model.forward(x_in_flat, return_energy=True)
            # energies_list elements are (C*B,)
            
            # We need standard "embeddings" for metrics. 
            # FF doesn't have a single "embedding" because the input is conditioned on label.
            # However, we can use the "activations corresponding to the PREDICTED label" as the representative embedding?
            # Or just skip embedding-based metrics for FF and focus on Accuracy/Logits.
            
            # For now, we will construct "logits" from total energy for the metrics collector
            total_energy = torch.stack(energies_list).sum(dim=0) # (C*B,)
            total_energy = total_energy.view(self.num_classes, B).t() # (B, C) -> Effectively logits
            
            # Create dummy embeddings list (empty) or try to provide meaningful ones?
            # MetricCollector expects embeddings to compute things. 
            # If we pass nothing, it might break or skip.
            # Let's pass the logits as "embeddings" for the last layer just to satisfy interfaces if needed, 
            # or better, skip detailed embedding metrics if they don't apply.
            # OptimizedMetricsCollector.collect_mlp_metrics takes embeddings_list.
            # Let's try to collect basics using the logits (energy).
            
            try:
                # We treat total_energy as the 'logits' of the network
                # We don't have intermediate 'embeddings' in the standard sense compatible with classification 
                # because they are label-conditional.
                # So we pass a list with just the final logits to avoid error, or skip.
                # Actually, let's just compute accuracy manually here (already done) 
                # and maybe CrossEntropy of the energy-logits.
                
                # Construct a pseudo-Metrics dict specifically for FF
                batch_metrics = {}
                batch_metrics['layer_final'] = {}
                # "Goodness" logits
                batch_metrics['layer_final']['accuracy'] = (preds == y).float().mean().item()
                # batch_metrics['layer_final']['cross_entropy'] = cross_entropy(total_energy, y) # Optional
                
                all_metrics.append(batch_metrics)
            except Exception:
                pass
            
        accuracy = correct / total
        
        # Average metrics
        avg_metrics = None
        if all_metrics:
            avg_metrics = {}
            for layer_key in all_metrics[0].keys():
                avg_metrics[layer_key] = {}
                for metric_name in all_metrics[0][layer_key].keys():
                    values = [batch[layer_key][metric_name] for batch in all_metrics]
                    avg_metrics[layer_key][metric_name] = sum(values) / len(values)

        return accuracy, avg_metrics
