import argparse
import os
import yaml
import json
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import confusion_matrix, classification_report
from rich.console import Console
from data.datasets import get_dataloaders
from models.greedy_linear import GreedyLinearNet, GreedyConfig
from trainers.greedy_trainer import GreedyTrainer
from models.mlp_baseline import MlpBaseline, MlpConfig

console = Console()


def load_cfg_from_run(run_dir: str):
    params_path = os.path.join(run_dir, 'params.yaml')
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Run directory to evaluate (contains params.yaml, best.pt)')
    parser.add_argument('--ckpt', type=str, default='best.pt', help='Checkpoint file within run_dir (best.pt or last.pt)')
    parser.add_argument('--confusion', action='store_true', help='Save confusion matrix and per-class report')
    args = parser.parse_args()

    cfg = load_cfg_from_run(args.run_dir)

    bundle = get_dataloaders(cfg['dataset'], batch_size=cfg['batch_size'], num_workers=cfg.get('num_workers', 4), seed=cfg.get('seed', 0))

    ckpt_path = os.path.join(args.run_dir, args.ckpt)
    state = torch.load(ckpt_path, map_location='cpu')

    if cfg['model'] == 'greedy':
        gcfg = GreedyConfig(input_dim=bundle.input_dim, N=cfg['N'], layers=cfg['layers'], similarity=cfg['similarity'], tau=cfg['tau'], lambda_ace=cfg['lambda_ace'])
        model = GreedyLinearNet(gcfg)
        model.load_state_dict(state['model'])
        trainer = GreedyTrainer(args.run_dir, model, cfg, bundle.num_classes, cfg['dataset'])
        acc_last, acc_layers, test_metrics = trainer.evaluate(bundle.test)
        console.print(f"[bold cyan]TEST acc (last layer):[/bold cyan] {acc_last:.4f}")
        console.print(f"Per-layer acc: {acc_layers}")
        results = {'acc_last': acc_last, 'acc_layers': acc_layers}

        if args.confusion:
            # Compute confusion for the last layer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            ys = []
            preds = []
            with torch.no_grad():
                for x, y in bundle.test:
                    x = x.to(device)
                    y = y.to(device)
                    B = x.size(0)
                    x = x.view(B, -1)
                    h_prev = x
                    for i in range(1, cfg['layers'] + 1):
                        h_prev = model.forward_layer_from(h_prev, i)
                    logits, _ = trainer._compute_layer_logits_and_probs(h_prev)
                    preds.append(logits.argmax(dim=-1).cpu().numpy())
                    ys.append(y.cpu().numpy())
            y_true = np.concatenate(ys)
            y_pred = np.concatenate(preds)
            cm = confusion_matrix(y_true, y_pred)
            rep = classification_report(y_true, y_pred, output_dict=True)
            np.save(os.path.join(args.run_dir, 'confusion.npy'), cm)
            with open(os.path.join(args.run_dir, 'class_report.json'), 'w') as f:
                json.dump(rep, f, indent=2)
            console.print("Saved confusion.npy and class_report.json")

    elif cfg['model'] == 'mlp':
        mcfg = MlpConfig(input_dim=bundle.input_dim, N=cfg['N'], layers=cfg['layers'], num_classes=bundle.num_classes)
        model = MlpBaseline(mcfg)
        model.load_state_dict(state['model'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        ys = []
        preds = []
        with torch.no_grad():
            for x, y in bundle.test:
                x = x.to(device)
                y = y.to(device)
                B = x.size(0)
                x = x.view(B, -1)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += B
                ys.append(y.cpu().numpy())
                preds.append(pred.cpu().numpy())
        acc = correct / total
        console.print(f"[bold cyan]TEST acc:[/bold cyan] {acc:.4f}")
        results = {'acc': acc}

        if args.confusion:
            y_true = np.concatenate(ys)
            y_pred = np.concatenate(preds)
            cm = confusion_matrix(y_true, y_pred)
            rep = classification_report(y_true, y_pred, output_dict=True)
            np.save(os.path.join(args.run_dir, 'confusion.npy'), cm)
            with open(os.path.join(args.run_dir, 'class_report.json'), 'w') as f:
                json.dump(rep, f, indent=2)
            console.print("Saved confusion.npy and class_report.json")

    with open(os.path.join(args.run_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
