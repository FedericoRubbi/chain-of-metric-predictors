import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosine(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, warmup_steps + 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = 0.5 * base_lr * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
