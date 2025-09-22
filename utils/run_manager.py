import os
from datetime import datetime
import pytz
import yaml
from typing import Dict, Any

class RunManager:
    """Creates run folders with names like '0001_2025-09-22_11-05-12' and saves params."""
    def __init__(self, base_dir: str = "runs", tz: str = "Europe/Rome"):
        self.base_dir = base_dir
        self.tz = pytz.timezone(tz)
        os.makedirs(self.base_dir, exist_ok=True)

    def _next_progressive(self, run_root: str) -> int:
        os.makedirs(run_root, exist_ok=True)
        existing = [d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d))]
        nums = []
        for name in existing:
            try:
                nums.append(int(name.split("_")[0]))
            except Exception:
                pass
        return (max(nums) + 1) if nums else 1

    def new_run(self, dataset: str, model: str, params: Dict[str, Any]) -> str:
        run_root = os.path.join(self.base_dir, dataset, model)
        progressive = self._next_progressive(run_root)
        timestamp = datetime.now(self.tz).strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{progressive:04d}_{timestamp}"
        run_dir = os.path.join(run_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        # Save params for traceability
        with open(os.path.join(run_dir, "params.yaml"), "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)
        return run_dir
