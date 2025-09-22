import argparse
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from utils.seed import set_seed
from utils.run_manager import RunManager
from data.datasets import get_dataloaders
from models.greedy_linear import GreedyLinearNet, GreedyConfig
from trainers.greedy_trainer import GreedyTrainer
from models.mlp_baseline import MlpBaseline, MlpConfig
from trainers.baseline_trainer import BaselineTrainer

console = Console()


def print_config(params):
    table = Table(title="Training Configuration")
    for k in ["dataset", "model", "N", "layers", "similarity", "tau", "lambda_ace", "batch_size", "epochs", "lr", "weight_decay", "scheduler", "warmup_ratio", "num_workers", "seed"]:
        if k in params:
            table.add_row(k, str(params[k]))
    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 0))

    # Prepare data
    bundle = get_dataloaders(cfg['dataset'], batch_size=cfg['batch_size'], num_workers=cfg.get('num_workers', 4), seed=cfg.get('seed', 0))

    # Prepare run folder
    rm = RunManager()
    run_dir = rm.new_run(cfg['dataset'], cfg['model'], params={**cfg, 'num_classes': bundle.num_classes, 'input_dim': bundle.input_dim})

    console.rule("[bold green]Run Initialized")
    console.print(f"Run directory: [bold]{run_dir}[/bold]")
    print_config({**cfg, 'num_classes': bundle.num_classes, 'input_dim': bundle.input_dim})

    # Build & train
    if cfg['model'] == 'greedy':
        gcfg = GreedyConfig(
            input_dim=bundle.input_dim,
            N=cfg['N'],
            layers=cfg['layers'],
            similarity=cfg['similarity'],
            tau=cfg['tau'],
            lambda_ace=cfg['lambda_ace'],
        )
        model = GreedyLinearNet(gcfg)
        trainer = GreedyTrainer(run_dir, model, cfg, bundle.num_classes, cfg['dataset'])
        trainer.fit(bundle.train, bundle.val, epochs=cfg['epochs'])
        test_acc_last, test_acc_layers = trainer.evaluate(bundle.test)
        console.print(f"[bold cyan]Test Acc (last layer):[/bold cyan] {test_acc_last:.4f}")
        console.print(f"Per-layer test acc: {test_acc_layers}")

    elif cfg['model'] == 'mlp':
        mcfg = MlpConfig(
            input_dim=bundle.input_dim,
            N=cfg['N'],
            layers=cfg['layers'],
            num_classes=bundle.num_classes,
        )
        model = MlpBaseline(mcfg)
        trainer = BaselineTrainer(run_dir, model, cfg)
        trainer.fit(bundle.train, bundle.val, epochs=cfg['epochs'])
        test_acc = trainer.evaluate(bundle.test)
        console.print(f"[bold cyan]Test Acc:[/bold cyan] {test_acc:.4f}")
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

    console.rule("[bold green]Done")

if __name__ == '__main__':
    main()
