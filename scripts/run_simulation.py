#!/usr/bin/env python3
"""
Comprehensive simulation script that automates the entire workflow:
1. Training a model with specified config
2. Running evaluation with confusion matrix
3. Generating comprehensive visualizations
4. Creating confusion matrix plots

This script eliminates the need to run multiple scripts manually.
"""

import argparse
import yaml
import sys
import os
import subprocess
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich import print as rprint

console = Console()


def print_simulation_header(config_path: str, run_dir: str):
    """Print a nice header for the simulation."""
    console.rule("[bold blue]COMPREHENSIVE SIMULATION")
    
    header_table = Table(title="Simulation Configuration", show_header=True, header_style="bold magenta")
    header_table.add_column("Parameter", style="cyan")
    header_table.add_column("Value", style="green")
    
    header_table.add_row("Config File", config_path)
    header_table.add_row("Run Directory", run_dir)
    header_table.add_row("Start Time", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print(header_table)
    console.print()


def load_and_print_config(config_path: str):
    """Load and display the configuration."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    config_table = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    # Key parameters to display
    key_params = [
        "dataset", "model", "N", "layers", "similarity", "tau", "lambda_ace", "ace_variant",
        "batch_size", "epochs", "lr", "weight_decay", "scheduler", "warmup_ratio", 
        "num_workers", "seed", "metrics_log_frequency"
    ]
    
    for param in key_params:
        if param in cfg:
            if param == "lambda_ace" and isinstance(cfg[param], list):
                config_table.add_row(param, f"[{', '.join(str(x) for x in cfg[param])}]")
            else:
                config_table.add_row(param, str(cfg[param]))
    
    console.print(config_table)
    console.print()
    
    return cfg


def run_training(config_path: str, run_dir: str):
    """Run the training script."""
    console.print(Panel("🚀 [bold green]STEP 1: TRAINING[/bold green]", style="green"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training model...", total=None)
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/train.py", 
                "--config", config_path
            ], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if result.returncode != 0:
                console.print(f"[bold red]Training failed![/bold red]")
                return False
            
            progress.update(task, description="✅ Training completed successfully!")
            return True
            
        except Exception as e:
            console.print(f"[bold red]Training error:[/bold red] {e}")
            return False


def run_evaluation(run_dir: str, confusion: bool = True):
    """Run the evaluation script."""
    console.print(Panel("📊 [bold blue]STEP 2: EVALUATION[/bold blue]", style="blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating model...", total=None)
        
        try:
            cmd = [sys.executable, "scripts/eval.py", "--run_dir", run_dir]
            if confusion:
                cmd.append("--confusion")
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if result.returncode != 0:
                console.print(f"[bold red]Evaluation failed![/bold red]")
                console.print(f"Error: {result.stderr}")
                return False
            
            progress.update(task, description="✅ Evaluation completed successfully!")
            console.print(f"[blue]Evaluation output:[/blue] {result.stdout}")
            return True
            
        except Exception as e:
            console.print(f"[bold red]Evaluation error:[/bold red] {e}")
            return False


def run_visualization(run_dir: str, show_plots: bool = False, research_only: bool = False):
    """Run the visualization scripts."""
    console.print(Panel("📈 [bold magenta]STEP 3: VISUALIZATION[/bold magenta]", style="magenta"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        # Basic visualization
        task1 = progress.add_task("Generating basic visualizations...", total=None)
        
        try:
            cmd = [sys.executable, "scripts/plot_metrics.py", "--run_dirs", run_dir]
            if show_plots:
                cmd.append("--show")
            if research_only:
                cmd.append("--comprehensive")
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if result.returncode != 0:
                console.print(f"[bold red]Basic visualization failed![/bold red]")
                console.print(f"Error: {result.stderr}")
                return False
            
            progress.update(task1, description="✅ Basic visualizations completed!")
            console.print(f"[magenta]Visualization output:[/magenta] {result.stdout}")
            
        except Exception as e:
            console.print(f"[bold red]Basic visualization error:[/bold red] {e}")
            return False
        
        # Confusion matrix visualization
        task2 = progress.add_task("Generating confusion matrix...", total=None)
        
        try:
            cmd = [sys.executable, "scripts/plot_metrics.py", "--run_dirs", run_dir, "--confusion", run_dir]
            if show_plots:
                cmd.append("--show")
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if result.returncode != 0:
                console.print(f"[bold yellow]Confusion matrix visualization failed (this is optional)[/bold yellow]")
                console.print(f"Note: {result.stderr}")
            else:
                progress.update(task2, description="✅ Confusion matrix completed!")
                console.print(f"[magenta]Confusion matrix output:[/magenta] {result.stdout}")
            
        except Exception as e:
            console.print(f"[bold yellow]Confusion matrix error (optional):[/bold yellow] {e}")
        
        return True


def run_advanced_visualization(run_dirs: list, show_plots: bool = False):
    """Run advanced visualization for multiple runs."""
    if len(run_dirs) < 2:
        return True
    
    console.print(Panel("🔬 [bold cyan]STEP 4: ADVANCED ANALYSIS[/bold cyan]", style="cyan"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating advanced visualizations...", total=None)
        
        try:
            cmd = [sys.executable, "scripts/plot_metrics.py", "--run_dirs"] + run_dirs + ["--comprehensive"]
            if show_plots:
                cmd.append("--show")
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if result.returncode != 0:
                console.print(f"[bold yellow]Advanced visualization failed (this is optional)[/bold yellow]")
                console.print(f"Note: {result.stderr}")
            else:
                progress.update(task, description="✅ Advanced visualizations completed!")
                console.print(f"[cyan]Advanced visualization output:[/cyan] {result.stdout}")
            
        except Exception as e:
            console.print(f"[bold yellow]Advanced visualization error (optional):[/bold yellow] {e}")
        
        return True


def generate_simulation_report(run_dir: str, config_path: str, start_time: float):
    """Generate a comprehensive simulation report."""
    end_time = time.time()
    duration = end_time - start_time
    
    console.print(Panel("📋 [bold yellow]SIMULATION REPORT[/bold yellow]", style="yellow"))
    
    # Check what files were generated
    run_path = Path(run_dir)
    generated_files = []
    
    if run_path.exists():
        for file_path in run_path.iterdir():
            if file_path.is_file():
                generated_files.append(file_path.name)
    
    report_table = Table(title="Generated Files", show_header=True, header_style="bold yellow")
    report_table.add_column("File Type", style="cyan")
    report_table.add_column("Files", style="green")
    
    # Categorize files
    categories = {
        "Training": ["params.yaml", "log.jsonl", "best.pt", "last.pt"],
        "Evaluation": ["test_results.json", "confusion.npy", "class_report.json"],
        "Visualization": [f for f in generated_files if f.endswith('.png')],
        "Other": [f for f in generated_files if f not in 
                 ["params.yaml", "log.jsonl", "best.pt", "last.pt", "test_results.json", 
                  "confusion.npy", "class_report.json"] and not f.endswith('.png')]
    }
    
    for category, files in categories.items():
        if files:
            report_table.add_row(category, ", ".join(files))
    
    console.print(report_table)
    
    # Summary
    summary_table = Table(title="Simulation Summary", show_header=True, header_style="bold yellow")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Config File", config_path)
    summary_table.add_row("Run Directory", run_dir)
    summary_table.add_row("Duration", f"{duration:.2f} seconds")
    summary_table.add_row("Total Files Generated", str(len(generated_files)))
    summary_table.add_row("Visualization Files", str(len([f for f in generated_files if f.endswith('.png')])))
    
    console.print(summary_table)
    
    # Next steps
    console.print(Panel("🎯 [bold green]NEXT STEPS[/bold green]", style="green"))
    console.print("1. Check the generated plots in the run directory")
    console.print("2. Review the evaluation results in test_results.json")
    console.print("3. Analyze the metrics data in log.jsonl")
    console.print("4. Use plot_metrics.py for multi-run comparisons")
    console.print()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive simulation script')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to YAML configuration file')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Specific run directory (optional, will be auto-generated if not provided)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively instead of saving only')
    parser.add_argument('--research_only', action='store_true',
                       help='Generate only research questions plots')
    parser.add_argument('--no_confusion', action='store_true',
                       help='Skip confusion matrix generation')
    parser.add_argument('--advanced', action='store_true',
                       help='Run advanced multi-run analysis (requires multiple runs)')
    parser.add_argument('--compare_runs', nargs='+', default=None,
                       help='Additional run directories to compare with')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Print header
    print_simulation_header(args.config, args.run_dir or "auto-generated")
    
    # Load and print config
    cfg = load_and_print_config(args.config)
    
    # Determine run directory
    if args.run_dir is None:
        # We need to get the run directory from the training script
        # This is a bit tricky since we need to run training first
        console.print("[yellow]Note: Run directory will be auto-generated during training[/yellow]")
        run_dir = None
    else:
        run_dir = args.run_dir
    
    # Step 1: Training
    if not run_training(args.config, run_dir or "auto"):
        console.print("[bold red]Simulation failed at training step![/bold red]")
        return 1
    
    # Get the actual run directory if it was auto-generated
    if run_dir is None:
        # Find the most recent run directory
        runs_dir = Path("runs")
        if runs_dir.exists():
            dataset = cfg['dataset']
            model = cfg['model']
            model_runs_dir = runs_dir / dataset / model
            if model_runs_dir.exists():
                # Get the most recent run
                run_dirs = sorted([d for d in model_runs_dir.iterdir() if d.is_dir()], 
                                key=lambda x: x.stat().st_mtime, reverse=True)
                if run_dirs:
                    run_dir = str(run_dirs[0])
                    console.print(f"[green]Auto-detected run directory: {run_dir}[/green]")
                else:
                    console.print("[bold red]Could not find run directory![/bold red]")
                    return 1
            else:
                console.print("[bold red]Could not find model runs directory![/bold red]")
                return 1
        else:
            console.print("[bold red]Could not find runs directory![/bold red]")
            return 1
    
    # Step 2: Evaluation
    if not run_evaluation(run_dir, confusion=not args.no_confusion):
        console.print("[bold red]Simulation failed at evaluation step![/bold red]")
        return 1
    
    # Step 3: Visualization
    if not run_visualization(run_dir, show_plots=args.show, research_only=args.research_only):
        console.print("[bold red]Simulation failed at visualization step![/bold red]")
        return 1
    
    # Step 4: Advanced visualization (if requested)
    if args.advanced or args.compare_runs:
        compare_runs = [run_dir]
        if args.compare_runs:
            compare_runs.extend(args.compare_runs)
        
        run_advanced_visualization(compare_runs, show_plots=args.show)
    
    # Generate final report
    generate_simulation_report(run_dir, args.config, start_time)
    
    console.rule("[bold green]🎉 SIMULATION COMPLETED SUCCESSFULLY! 🎉")
    return 0


if __name__ == '__main__':
    exit(main())
