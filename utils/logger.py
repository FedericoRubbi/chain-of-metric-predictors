import os
import json
from typing import Dict, Any
from rich.console import Console
from rich.table import Table

class JsonlLogger:
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, 'log.jsonl')
        self.console = Console()

    def log(self, rec: Dict[str, Any]):
        with open(self.path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        # Note: Removed console.print to prevent text spam during training
        # Progress bar updates are handled by Rich Progress in the trainers
