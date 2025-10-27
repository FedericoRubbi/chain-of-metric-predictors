import os
import json
from typing import Dict, Any
import numpy as np
try:
    import torch
except Exception:
    torch = None
from rich.console import Console
from rich.table import Table

class JsonlLogger:
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, 'log.jsonl')
        self.console = Console()

    def _to_serializable(self, obj: Any):
        """Recursively convert objects to JSON-serializable types."""
        # Torch tensors
        if torch is not None and isinstance(obj, torch.Tensor):
            if obj.dim() == 0:
                return obj.item()
            return obj.detach().cpu().tolist()
        # Numpy types
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Containers
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        return obj

    def log(self, rec: Dict[str, Any]):
        with open(self.path, 'a') as f:
            safe = self._to_serializable(rec)
            f.write(json.dumps(safe) + '\n')
        # Note: Removed console.print to prevent text spam during training
        # Progress bar updates are handled by Rich Progress in the trainers
