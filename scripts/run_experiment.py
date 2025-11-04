#!/usr/bin/env python3
"""
Experiment Orchestrator

Runs a series of training or simulation runs sequentially based on an
experiment specification YAML.

Features:
- Supports parameter sweeps (full grid) or a list of explicit configs
- Sequential execution to avoid VRAM exhaustion
- Writes per-run resolved configs for reproducibility
- Tracks experiment metadata, statuses, and discovered run directories
- Resume support to skip completed runs
"""

import argparse
import itertools
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

# Ensure project root on path for consistent CWD behavior
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint


console = Console()


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: str):
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_grid(sweep: dict):
    if not sweep:
        return [{}]
    keys = sorted(sweep.keys())
    values = [sweep[k] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def discover_new_run_dir(before_dirs: set, dataset: str, model: str) -> str:
    runs_root = Path("runs") / dataset / model
    if not runs_root.exists():
        return None
    after_dirs = {str(p) for p in runs_root.iterdir() if p.is_dir()}
    new_dirs = sorted(list(after_dirs - before_dirs), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if new_dirs:
        return new_dirs[0]
    # Fallback: most recent
    all_dirs = sorted([str(p) for p in runs_root.iterdir() if p.is_dir()], key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return all_dirs[0] if all_dirs else None


def list_existing_run_dirs(dataset: str, model: str) -> set:
    runs_root = Path("runs") / dataset / model
    if not runs_root.exists():
        return set()
    return {str(p) for p in runs_root.iterdir() if p.is_dir()}


def build_experiment_dir(name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path("experiments") / f"{name}_{timestamp}"
    ensure_dir(str(base))
    ensure_dir(str(base / "configs"))
    return str(base)


def save_metadata(meta_path: str, meta: dict):
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_metadata(meta_path: str):
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r') as f:
        return json.load(f)


def apply_overrides(base_cfg: dict, overrides: dict) -> dict:
    cfg = deepcopy(base_cfg)
    for k, v in (overrides or {}).items():
        cfg[k] = v
    return cfg


def print_overview(name: str, mode: str, exp_dir: str, total: int):
    console.rule("[bold blue]EXPERIMENT")
    table = Table(title="Experiment Overview", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Name", name)
    table.add_row("Mode", mode)
    table.add_row("Directory", exp_dir)
    table.add_row("Total Runs", str(total))
    console.print(table)


def run_single(mode: str, config_path: str, sim_flags: dict) -> int:
    import subprocess
    cmd = [sys.executable]
    if mode == "training":
        cmd += ["scripts/train.py", "--config", config_path]
    elif mode == "simulation":
        cmd += ["scripts/run_simulation.py", "--config", config_path]
        if sim_flags.get("show"):
            cmd.append("--show")
        if sim_flags.get("research_only"):
            cmd.append("--research_only")
        if sim_flags.get("no_confusion"):
            cmd.append("--no_confusion")
        if sim_flags.get("advanced"):
            cmd.append("--advanced")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run a sequential experiment from a spec YAML")
    parser.add_argument("--exp", required=True, help="Path to experiment YAML spec")
    parser.add_argument("--resume", action="store_true", help="Resume and skip completed runs")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N combinations")
    parser.add_argument("--name", type=str, default=None, help="Override experiment name")
    # Simulation-only flags (forwarded when mode: simulation)
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--research_only", action="store_true", help="Generate only research plots")
    parser.add_argument("--no_confusion", action="store_true", help="Skip confusion matrix")
    parser.add_argument("--advanced", action="store_true", help="Run advanced multi-run analysis")

    args = parser.parse_args()

    spec = load_yaml(args.exp)
    name = args.name or spec.get("name", "experiment")
    mode = spec.get("mode", "training")

    if mode not in {"training", "simulation"}:
        console.print(f"[bold red]Invalid mode in spec: {mode}[/bold red]")
        return 1

    exp_dir = build_experiment_dir(name)
    configs_dir = os.path.join(exp_dir, "configs")
    meta_path = os.path.join(exp_dir, "experiment.json")

    # Prepare run definitions
    runs = []  # list of dicts: {id, config_path, overrides, source_config}
    if spec.get("configs"):
        # Use explicit configs
        for idx, cfg_path in enumerate(spec["configs"]):
            base_cfg = load_yaml(cfg_path)
            # Apply fixed overrides if any
            resolved = apply_overrides(base_cfg, spec.get("fixed_overrides"))
            out_cfg_path = os.path.join(configs_dir, f"run_{idx:03d}.yaml")
            dump_yaml(resolved, out_cfg_path)
            runs.append({
                "id": idx,
                "config_path": out_cfg_path,
                "overrides": {},
                "source_config": cfg_path,
            })
    else:
        base_config_path = spec.get("base_config")
        if not base_config_path:
            console.print("[bold red]Spec must include either 'configs' or 'base_config'[/bold red]")
            return 1
        base_cfg = load_yaml(base_config_path)
        grid = generate_grid(spec.get("sweep", {}))
        if args.limit is not None:
            grid = grid[: args.limit]
        for idx, ov in enumerate(grid):
            resolved = apply_overrides(base_cfg, spec.get("fixed_overrides"))
            resolved = apply_overrides(resolved, ov)
            out_cfg_path = os.path.join(configs_dir, f"run_{idx:03d}.yaml")
            dump_yaml(resolved, out_cfg_path)
            runs.append({
                "id": idx,
                "config_path": out_cfg_path,
                "overrides": ov,
                "source_config": base_config_path,
            })

    # Initialize metadata
    meta = {
        "name": name,
        "mode": mode,
        "exp_spec": os.path.abspath(args.exp),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exp_dir": os.path.abspath(exp_dir),
        "total_runs": len(runs),
        "runs": [
            {
                "id": r["id"],
                "config_path": r["config_path"],
                "source_config": r["source_config"],
                "overrides": r["overrides"],
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "return_code": None,
                "run_dir": None,
            }
            for r in runs
        ],
    }
    save_metadata(meta_path, meta)

    print_overview(name, mode, exp_dir, len(runs))

    # Execute sequentially
    sim_flags = {
        "show": args.show,
        "research_only": args.research_only,
        "no_confusion": args.no_confusion,
        "advanced": args.advanced,
    }

    for i, run_item in enumerate(meta["runs"]):
        if args.resume and run_item.get("status") == "completed":
            continue

        cfg = load_yaml(run_item["config_path"])
        dataset = cfg.get("dataset")
        model = cfg.get("model")
        before = list_existing_run_dirs(dataset, model)

        console.print(Panel(f"▶️  [bold green]RUN {run_item['id']:03d}[/bold green] — {os.path.basename(run_item['config_path'])}", style="green"))
        run_item["status"] = "running"
        run_item["started_at"] = datetime.now().isoformat(timespec="seconds")
        save_metadata(meta_path, meta)

        rc = run_single(mode, run_item["config_path"], sim_flags)
        run_item["return_code"] = rc
        if rc == 0:
            # Discover produced run_dir
            found = discover_new_run_dir(before, dataset, model)
            run_item["run_dir"] = found
            run_item["status"] = "completed"
            run_item["completed_at"] = datetime.now().isoformat(timespec="seconds")
            console.print(f"[green]✓ Completed[/green] — run_dir: {found or 'unknown'}")
        else:
            run_item["status"] = "failed"
            run_item["completed_at"] = datetime.now().isoformat(timespec="seconds")
            console.print(f"[bold red]✗ Failed with code {rc}[/bold red]")

        save_metadata(meta_path, meta)

    # Summary table
    summary = Table(title="Experiment Summary", show_header=True, header_style="bold cyan")
    summary.add_column("Run")
    summary.add_column("Status")
    summary.add_column("Return")
    summary.add_column("Run Dir")
    for r in meta["runs"]:
        summary.add_row(
            f"{r['id']:03d}",
            r["status"],
            str(r["return_code"]) if r["return_code"] is not None else "-",
            r["run_dir"] or "-",
        )
    console.print(summary)
    console.rule("[bold green]EXPERIMENT COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


