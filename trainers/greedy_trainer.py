from typing import Dict, List, Tuple, Optional
import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from utils.scores import scores_from_embeddings, softmax_with_temperature
from utils.metrics import cross_entropy_from_probs, top1_accuracy
from utils.anchors import build_or_load_anchors
from utils.schedulers import WarmupCosine
from utils.logger import JsonlLogger
from utils.metrics_collector import MetricsCollector
from utils.optimized_metrics_collector import OptimizedMetricsCollector
from models.greedy_linear import GreedyLinearNet, GreedyConfig
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from datetime import datetime


class GreedyTrainer:
    def __init__(self, run_dir: str, model: GreedyLinearNet, cfg: Dict, num_classes: int, dataset: str):
        self.run_dir = run_dir
        self.model = model
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset = dataset

        # Anchors (deterministic)
        anchors_path = os.path.join('artifacts', f"anchors_{dataset}_C{num_classes}_N{cfg['N']}_seed{cfg['seed']}.pt")
        self.anchors = build_or_load_anchors(anchors_path, num_classes, cfg['N'], seed=cfg['seed']).to(self.device)

        # Per-layer optimizers & schedulers
        self.optims: List[torch.optim.Optimizer] = []
        self.scheds: List[WarmupCosine] = []
        for idx, lin in enumerate(model.linears):
            # Include residual projection for the corresponding layer
            params = list(lin.parameters())
            if hasattr(model, 'residuals'):
                params += list(model.residuals[idx].parameters())
            opt = torch.optim.Adam(params, lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=cfg['weight_decay'])
            self.optims.append(opt)
        total_steps = None  # set during fit with len(train_loader) * epochs
        self.scaler = GradScaler(enabled=True)
        self.logger = JsonlLogger(run_dir)
        
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
        self.metrics_collector.set_anchors(self.anchors.E)
        self.metrics_collector.initialize_entropy_estimators(cfg['layers'])
        self.console = Console()

    def _make_targets(self, y: torch.Tensor) -> torch.Tensor:
        # one-hot targets p
        p = torch.zeros((y.size(0), self.num_classes), device=y.device, dtype=torch.float32)
        p.scatter_(1, y.view(-1, 1), 1.0)
        return p

    def _compute_layer_logits_and_probs(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = scores_from_embeddings(h, self.anchors.E, similarity=self.cfg['similarity'])  # (B, C)
        q = softmax_with_temperature(scores, self.cfg['tau'])
        return scores, q

    def _step_batch(self, batch, global_step: int):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        B = x.size(0)
        x = x.view(B, -1)

        p = self._make_targets(y)

        # We'll compute layer by layer, ensuring gradient isolation per layer
        # h0 is the input vector
        h_prev = x
        per_layer_acc = []
        per_layer_loss = []
        embeddings_list = []  # Store embeddings for metrics

        for i in range(1, self.cfg['layers'] + 1):
            # Detach input to keep independence from previous layer params
            h_in = h_prev.detach()

            # Forward current layer (with grad)
            with autocast(enabled=True):
                h_i = self.model.forward_layer_from(h_in, i)
                logits_i, q_i = self._compute_layer_logits_and_probs(h_i)
                
                # Store embeddings for metrics
                embeddings_list.append(h_i.detach())

                # For ACE we need q_{i+1}. If last layer, ACE term = 0.
                if i < self.cfg['layers']:
                    # Compute next layer output *without allowing gradients to flow back*
                    with torch.no_grad():
                        h_ip1 = self.model.forward_layer_from(h_i.detach(), i + 1)
                        _, q_ip1 = self._compute_layer_logits_and_probs(h_ip1)
                    ace = cross_entropy_from_probs(q_i, q_ip1)  # H(q_i, q_{i+1})
                    # Use per-layer ACE weight: layer i connecting to i+1 uses lambda_ace[i-1]
                    lambda_ace_i = self.cfg['lambda_ace'][i - 1]
                else:
                    ace = torch.tensor(0.0, device=self.device)
                    lambda_ace_i = 0.0

                ce = cross_entropy_from_probs(p, q_i)  # H(p, q_i)
                loss_i = ce - lambda_ace_i * ace

            # Optimize only layer i's Linear parameters
            self.optims[i - 1].zero_grad(set_to_none=True)
            self.scaler.scale(loss_i).backward()
            self.scaler.step(self.optims[i - 1])
            # No scheduler step here; we'll step per-iteration if created in fit()

            per_layer_loss.append(loss_i.detach().item())
            per_layer_acc.append(top1_accuracy(logits_i.detach(), y))

            # Prepare for next loop iteration's input
            h_prev = h_i.detach()

        self.scaler.update()

        # Collect metrics based on frequency setting
        metrics_frequency = self.cfg.get('metrics_frequency', 'iteration')
        should_collect_metrics = False
        
        if metrics_frequency == 'iteration':
            # Collect metrics every few iterations to avoid overhead
            should_collect_metrics = (global_step % self.cfg.get('metrics_log_frequency', 10) == 0)
        elif metrics_frequency == 'epoch':
            # For epoch frequency, we'll collect metrics at the end of each epoch
            # This will be handled in the fit() method
            should_collect_metrics = False
        
        batch_metrics = None
        if should_collect_metrics:
            with torch.no_grad():
                try:
                    batch_metrics = self.metrics_collector.collect_gmlp_metrics(
                        embeddings_list, y, x
                    )
                except Exception:
                    batch_metrics = None

        # Return last layer metrics + per-layer arrays + batch metrics
        return {
            'loss': sum(per_layer_loss) / len(per_layer_loss),
            'acc_last': per_layer_acc[-1],
            'acc_layers': per_layer_acc,
            'loss_layers': per_layer_loss,
            'batch_metrics': batch_metrics,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        total_steps = len(train_loader) * epochs
        warmup = int(self.cfg['warmup_ratio'] * total_steps)
        # Create schedulers now that we know total_steps
        self.scheds = [WarmupCosine(opt, warmup_steps=warmup, total_steps=total_steps) for opt in self.optims]

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
            # Create main training task
            training_task = progress.add_task(
                f"Training for {epochs} epochs", 
                total=epochs
            )
            
            for epoch in range(1, epochs + 1):
                self.model.train()
                epoch_metrics = None
                
                # Create epoch task
                epoch_task = progress.add_task(
                    f"Epoch {epoch}/{epochs}", 
                    total=len(train_loader)
                )
                
                for it, batch in enumerate(train_loader, start=1):
                    metrics = self._step_batch(batch, global_step)
                    # Step schedulers
                    for sch in self.scheds:
                        sch.step()
                    global_step += 1
                    
                    # Log training step
                    log_entry = {
                        'phase': 'train', 
                        'epoch': epoch, 
                        'iter': it, 
                        'step': global_step,
                        'loss': metrics['loss'], 
                        'acc': metrics['acc_last'],
                        'timestamp': datetime.now().isoformat()
                    }
                    if metrics['batch_metrics'] is not None:
                        log_entry['metrics'] = metrics['batch_metrics']
                    self.logger.log(log_entry)
                    
                    # Collect metrics at the end of epoch if using epoch frequency
                    if self.cfg.get('metrics_frequency', 'iteration') == 'epoch' and it == len(train_loader):
                        with torch.no_grad():
                            try:
                                # Re-run the last batch to collect metrics
                                x, y = batch
                                x = x.to(self.device)
                                y = y.to(self.device)
                                B = x.size(0)
                                x = x.view(B, -1)
                                
                                # Get embeddings for metrics collection
                                h_prev = x
                                embeddings_list = []
                                for i in range(1, self.cfg['layers'] + 1):
                                    h_i = self.model.forward_layer_from(h_prev, i)
                                    embeddings_list.append(h_i.detach())
                                    h_prev = h_i.detach()
                                
                                epoch_metrics = self.metrics_collector.collect_gmlp_metrics(
                                    embeddings_list, y, x
                                )
                            except Exception:
                                epoch_metrics = None
                    
                    # Update progress
                    progress.update(epoch_task, advance=1, description=f"Epoch {epoch}/{epochs} - Loss: {metrics['loss']:.4f}")
                
                # Complete epoch task
                progress.remove_task(epoch_task)
                
                # Validation (guarded)
                try:
                    val_acc_last, val_acc_layers, val_metrics = self.evaluate(val_loader)
                except Exception:
                    val_acc_last, val_acc_layers, val_metrics = 0.0, [0.0 for _ in range(self.cfg['layers'])], None
                
                # Log validation metrics
                log_entry = {
                    'phase': 'val', 
                    'epoch': epoch, 
                    'iter': 0, 
                    'acc': val_acc_last,
                    'timestamp': datetime.now().isoformat()
                }
                if val_metrics:
                    log_entry['metrics'] = val_metrics
                
                # If using epoch frequency, also log the training metrics collected at the end of epoch
                if self.cfg.get('metrics_frequency', 'iteration') == 'epoch' and epoch_metrics is not None:
                    log_entry['train_metrics'] = epoch_metrics
                
                self.logger.log(log_entry)

                # Checkpoint best
                if val_acc_last > best_val:
                    best_val = val_acc_last
                    torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, best_path)
                
                # Update main training progress
                progress.update(training_task, advance=1, description=f"Training for {epochs} epochs - Val Acc: {val_acc_last:.4f}")

        # Save final
        torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, os.path.join(self.run_dir, 'last.pt'))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, List[float], Optional[Dict]]:
        self.model.eval()
        device = self.device
        total = 0
        correct_layers = [0 for _ in range(self.cfg['layers'])]
        all_metrics = []
        
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)
            x = x.view(B, -1)

            h_prev = x
            embeddings_list = []
            
            for i in range(1, self.cfg['layers'] + 1):
                h_i = self.model.forward_layer_from(h_prev, i)
                logits_i, _ = self._compute_layer_logits_and_probs(h_i)
                preds = logits_i.argmax(dim=-1)
                correct_layers[i - 1] += (preds == y).sum().item()
                h_prev = h_i
                embeddings_list.append(h_i)
            
            total += B
            
            # Collect metrics for this batch (guarded)
            try:
                batch_metrics = self.metrics_collector.collect_gmlp_metrics(
                    embeddings_list, y, x
                )
                all_metrics.append(batch_metrics)
            except Exception:
                pass
        
        acc_layers = [c / total for c in correct_layers]
        
        # Average metrics across all batches
        avg_metrics = None
        if all_metrics:
            avg_metrics = {}
            for layer_key in all_metrics[0].keys():
                avg_metrics[layer_key] = {}
                for metric_name in all_metrics[0][layer_key].keys():
                    values = [batch[layer_key][metric_name] for batch in all_metrics]
                    avg_metrics[layer_key][metric_name] = sum(values) / len(values)
        
        return acc_layers[-1], acc_layers, avg_metrics
