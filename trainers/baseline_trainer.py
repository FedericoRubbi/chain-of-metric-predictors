from typing import Dict, Tuple
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from utils.schedulers import WarmupCosine
from utils.logger import JsonlLogger


class BaselineTrainer:
    def __init__(self, run_dir: str, model: nn.Module, cfg: Dict):
        self.run_dir = run_dir
        self.model = model
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=cfg['weight_decay'])
        self.crit = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=True)
        self.logger = JsonlLogger(run_dir)
        self.best_path = os.path.join(run_dir, 'best.pt')

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        total_steps = len(train_loader) * epochs
        warmup = int(self.cfg['warmup_ratio'] * total_steps)
        sched = WarmupCosine(self.optim, warmup_steps=warmup, total_steps=total_steps)

        best_val = 0.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            for it, (x, y) in enumerate(train_loader, start=1):
                x = x.to(self.device)
                y = y.to(self.device)
                B = x.size(0)
                x = x.view(B, -1)
                self.optim.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    logits = self.model(x)
                    loss = self.crit(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                sched.step()
                self.logger.log({'phase': 'train', 'epoch': epoch, 'iter': it, 'loss': float(loss.detach().item())})

            val_acc = self.evaluate(val_loader)
            self.logger.log({'phase': 'val', 'epoch': epoch, 'iter': 0, 'acc': val_acc})
            if val_acc > best_val:
                best_val = val_acc
                torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, self.best_path)

        torch.save({'model': self.model.state_dict(), 'cfg': self.cfg}, os.path.join(self.run_dir, 'last.pt'))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0
        correct = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            B = x.size(0)
            x = x.view(B, -1)
            logits = self.model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += B
        return correct / total
