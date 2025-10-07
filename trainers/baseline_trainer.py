from typing import Dict, Tuple, Optional
import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from utils.schedulers import WarmupCosine
from utils.logger import JsonlLogger
from utils.metrics_collector import MetricsCollector
from utils.optimized_metrics_collector import OptimizedMetricsCollector
from utils.anchors import build_or_load_anchors
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from datetime import datetime


class BaselineTrainer:
    def __init__(self, run_dir: str, model: nn.Module, cfg: Dict, num_classes: int, dataset: str):
        self.run_dir = run_dir
        self.model = model
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset = dataset

        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=cfg['weight_decay'])
        self.crit = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=True)
        self.logger = JsonlLogger(run_dir)
        self.best_path = os.path.join(run_dir, 'best.pt')
        
        # Initialize metrics collector
        enable_slow_metrics = self.cfg.get('enable_slow_metrics', False)
        self.metrics_collector = OptimizedMetricsCollector(
            num_classes=num_classes,
            N=cfg['N'],
            similarity=cfg.get('similarity', 'cosine'),
            tau=cfg.get('tau', 1.0),
            lambda_reg=cfg.get('lambda_reg', 1e-3),
            enable_slow_metrics=enable_slow_metrics
        )
        
        # Load anchors for metrics
        anchors_path = os.path.join('artifacts', f"anchors_{dataset}_C{num_classes}_N{cfg['N']}_seed{cfg['seed']}.pt")
        anchors = build_or_load_anchors(anchors_path, num_classes, cfg['N'], seed=cfg['seed']).to(self.device)
        self.metrics_collector.set_anchors(anchors.E)
        
        # Initialize entropy estimators
        self.metrics_collector.initialize_entropy_estimators(cfg['layers'])
        self.console = Console()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        total_steps = len(train_loader) * epochs
        warmup = int(self.cfg['warmup_ratio'] * total_steps)
        sched = WarmupCosine(self.optim, warmup_steps=warmup, total_steps=total_steps)

        best_val = 0.0
        global_step = 0

        with Progress(
            TextColumn("[bold blue]Training"),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            # Create main training task
            training_task = progress.add_task(
                f"Training for {epochs} epochs", 
                total=epochs
            )
            
            for epoch in range(1, epochs + 1):
                self.model.train()
                epoch_metrics = []
                
                # Create epoch task
                epoch_task = progress.add_task(
                    f"Epoch {epoch}/{epochs}", 
                    total=len(train_loader)
                )
                
                for it, (x, y) in enumerate(train_loader, start=1):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    B = x.size(0)
                    x = x.view(B, -1)
                    
                    self.optim.zero_grad(set_to_none=True)
                    with autocast(enabled=True):
                        # Get all intermediate embeddings
                        embeddings_list = self.model.forward_all(x)
                        logits = self.model(x)
                        loss = self.crit(logits, y)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    sched.step()
                    
                    # Compute training accuracy
                    with torch.no_grad():
                        preds = logits.argmax(dim=-1)
                        acc = (preds == y).float().mean().item()
                    
                    # Collect metrics based on frequency setting
                    metrics_frequency = self.cfg.get('metrics_frequency', 'iteration')
                    should_collect_metrics = False
                    
                    if metrics_frequency == 'iteration':
                        # Collect metrics every few iterations to avoid overhead
                        should_collect_metrics = (it % self.cfg.get('metrics_log_frequency', 10) == 0)
                    elif metrics_frequency == 'epoch':
                        # Collect metrics only on the last iteration of the epoch
                        should_collect_metrics = (it == len(train_loader))
                    
                    if should_collect_metrics:
                        with torch.no_grad():
                            # Compute logits for each layer using anchors
                            logits_list = []
                            for embeddings in embeddings_list:
                                layer_logits, _ = self.metrics_collector.compute_layer_logits_and_probs(embeddings)
                                logits_list.append(layer_logits)
                            
                            # Collect metrics
                            batch_metrics = self.metrics_collector.collect_mlp_metrics(
                                embeddings_list, logits_list, y, x
                            )
                            epoch_metrics.append(batch_metrics)
                    
                    # Log training step
                    log_entry = {
                        'phase': 'train', 
                        'epoch': epoch, 
                        'iter': it,
                        'step': global_step,
                        'loss': float(loss.detach().item()),
                        'acc': acc,
                        'timestamp': datetime.now().isoformat()
                    }
                    if should_collect_metrics and epoch_metrics:
                        # Include metrics in the log entry if we just collected them
                        log_entry['metrics'] = epoch_metrics[-1]
                    self.logger.log(log_entry)
                    
                    global_step += 1
                    
                    # Update progress
                    progress.update(epoch_task, advance=1, description=f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
                
                # Complete epoch task
                progress.remove_task(epoch_task)
                
                # Validation with metrics
                val_acc, val_metrics = self.evaluate(val_loader)
                
                # Log validation metrics
                log_entry = {
                    'phase': 'val', 
                    'epoch': epoch, 
                    'iter': 0, 
                    'acc': val_acc,
                    'timestamp': datetime.now().isoformat()
                }
                if val_metrics:
                    log_entry['metrics'] = val_metrics
                
                # If using epoch frequency, also log the training metrics collected at the end of epoch
                if self.cfg.get('metrics_frequency', 'iteration') == 'epoch' and epoch_metrics:
                    log_entry['train_metrics'] = epoch_metrics[-1]
                
                self.logger.log(log_entry)
                
                if val_acc > best_val:
                    best_val = val_acc
                    torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, self.best_path)
                
                # Update main training progress
                progress.update(training_task, advance=1, description=f"Training for {epochs} epochs - Val Acc: {val_acc:.4f}")

        torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, os.path.join(self.run_dir, 'last.pt'))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Optional[Dict]]:
        self.model.eval()
        total = 0
        correct = 0
        all_metrics = []
        
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            B = x.size(0)
            x = x.view(B, -1)
            
            # Get embeddings and final logits
            embeddings_list = self.model.forward_all(x)
            logits = self.model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += B
            
            # Collect metrics for this batch
            logits_list = []
            for embeddings in embeddings_list:
                layer_logits, _ = self.metrics_collector.compute_layer_logits_and_probs(embeddings)
                logits_list.append(layer_logits)
            
            batch_metrics = self.metrics_collector.collect_mlp_metrics(
                embeddings_list, logits_list, y, x
            )
            all_metrics.append(batch_metrics)
        
        accuracy = correct / total
        
        # Average metrics across all batches
        if all_metrics:
            avg_metrics = {}
            for layer_key in all_metrics[0].keys():
                avg_metrics[layer_key] = {}
                for metric_name in all_metrics[0][layer_key].keys():
                    values = [batch[layer_key][metric_name] for batch in all_metrics]
                    avg_metrics[layer_key][metric_name] = sum(values) / len(values)
            return accuracy, avg_metrics
        else:
            return accuracy, None
