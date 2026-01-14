from typing import Dict, List, Tuple, Optional
import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from utils.metrics import top1_accuracy
from utils.schedulers import WarmupCosine
from utils.logger import JsonlLogger
from utils.optimized_metrics_collector import OptimizedMetricsCollector
from models.mono_forward import MonoForwardNet, MonoForwardConfig
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from datetime import datetime


class MonoForwardTrainer:
    def __init__(self, run_dir: str, model: MonoForwardNet, cfg: Dict, num_classes: int, dataset: str):
        self.run_dir = run_dir
        self.model = model
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset = dataset

        # Per-layer optimizers & schedulers
        self.optims: List[torch.optim.Optimizer] = []
        self.scheds: List[WarmupCosine] = []
        
        # Mono-Forward: Optimizes layer i (Linear+ReLU) AND its local classifier together
        # gradients do not flow back to i-1, but do flow from classifier to the layer's linear block
        for i in range(cfg['layers']):
            params = list(model.linears[i].parameters()) + list(model.classifiers[i].parameters())
            opt = torch.optim.Adam(params, lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=cfg['weight_decay'])
            self.optims.append(opt)
            
        self.scaler = GradScaler(enabled=True)
        self.logger = JsonlLogger(run_dir)
        self.crit = nn.CrossEntropyLoss()
        
        # Initialize metrics collector
        enable_slow_metrics = self.cfg.get('enable_slow_metrics', False)
        self.metrics_collector = OptimizedMetricsCollector(
            num_classes=num_classes,
            N=cfg['N'],
            similarity=cfg.get('similarity', 'cosine'), # Unused for logits but required by init
            tau=cfg.get('tau', 1.0), # Unused
            lambda_reg=cfg.get('lambda_reg', 1e-3),
            enable_slow_metrics=enable_slow_metrics
        )
        # Note: We do NOT set anchors, as Mono-Forward doesn't use them.
        self.metrics_collector.initialize_entropy_estimators(cfg['layers'])
        self.console = Console()

    def _step_batch(self, batch, global_step: int):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        B = x.size(0)
        x = x.view(B, -1)

        # h0 is the input vector
        h_prev = x
        per_layer_acc = []
        per_layer_loss = []
        
        activations_list = []
        logits_list = []

        # Loop through layers 0 to L-1
        for i in range(self.cfg['layers']):
            # Detach input to keep independence from previous layer params
            h_in = h_prev.detach()

            # Forward current layer (with grad)
            with autocast(enabled=True):
                # model.forward_step returns (h_curr, logits)
                h_i, logits_i = self.model.forward_step(h_in, i)
                
                # Store for metrics
                activations_list.append(h_i.detach())
                logits_list.append(logits_i.detach())

                # Classification loss only
                loss_i = self.crit(logits_i, y)

            # Optimize layer i parameters (Backbone + Classifier)
            self.optims[i].zero_grad(set_to_none=True)
            self.scaler.scale(loss_i).backward()
            self.scaler.step(self.optims[i])
            
            per_layer_loss.append(loss_i.detach().item())
            per_layer_acc.append(top1_accuracy(logits_i.detach(), y))

            # Prepare for next loop iteration's input
            h_prev = h_i.detach()

        self.scaler.update()

        # Collect metrics based on frequency setting
        metrics_frequency = self.cfg.get('metrics_frequency', 'iteration')
        should_collect_metrics = False
        
        if metrics_frequency == 'iteration':
            should_collect_metrics = (global_step % self.cfg.get('metrics_log_frequency', 10) == 0)
        elif metrics_frequency == 'epoch':
            should_collect_metrics = False
        
        batch_metrics = None
        if should_collect_metrics:
            with torch.no_grad():
                try:
                    # adapting for MonoForward which produces logits directly
                    batch_metrics = self.metrics_collector.collect_mlp_metrics(
                        activations_list, logits_list, y, x
                    )
                except Exception:
                    batch_metrics = None

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
                                
                                # Make sure to get logits from forward_all
                                activations, all_logits = self.model.forward_all(x)
                                
                                epoch_metrics = self.metrics_collector.collect_mlp_metrics(
                                    activations, all_logits, y, x
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

            # MonoForward forward_all returns (activations, logits)
            activations, all_logits = self.model.forward_all(x)
            
            # Compute accuracy per layer
            for i, logits_i in enumerate(all_logits):
                preds = logits_i.argmax(dim=-1)
                correct_layers[i] += (preds == y).sum().item()
            
            total += B
            
            # Collect metrics for this batch (guarded)
            try:
                batch_metrics = self.metrics_collector.collect_mlp_metrics(
                    activations, all_logits, y, x
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
