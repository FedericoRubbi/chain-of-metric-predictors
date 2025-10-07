#!/usr/bin/env python3
"""
Background Training Script

This script allows you to run long training sessions in the background with proper logging
and process management. You can disconnect from SSH and the training will continue.

Usage:
    python scripts/run_background.py --config configs/my_config.yaml
    python scripts/run_background.py --config configs/my_config.yaml --simulation
    python scripts/run_background.py --config configs/my_config.yaml --simulation --show

Features:
- Runs training/simulation in background with nohup
- Saves process ID for monitoring
- Comprehensive logging to files
- Support for both training and simulation modes
- Automatic cleanup of old logs
"""

import argparse
import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()


def create_background_script(script_path: str, config_path: str, run_dir: str, 
                           additional_args: list = None, simulation: bool = False):
    """Create a background execution script."""
    
    # Determine the main script to run
    if simulation:
        main_script = "scripts/run_simulation.py"
        script_name = "simulation"
    else:
        main_script = "scripts/train.py"
        script_name = "training"
    
    # Create the background script content
    script_content = f"""#!/bin/bash
# Background {script_name} script
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

set -e  # Exit on any error

# Configuration
CONFIG_PATH="{config_path}"
RUN_DIR="{run_dir}"
SCRIPT_NAME="{script_name}"
LOG_FILE="{run_dir}/background_run.log"
PID_FILE="{run_dir}/background_run.pid"

# Create run directory if it doesn't exist
mkdir -p "$RUN_DIR"

# Function to cleanup on exit
cleanup() {{
    echo "Cleaning up background {script_name}..."
    rm -f "$PID_FILE"
    echo "Background {script_name} completed at $(date)" >> "$LOG_FILE"
}}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Log start
echo "Starting background {script_name} at $(date)" > "$LOG_FILE"
echo "Config: $CONFIG_PATH" >> "$LOG_FILE"
echo "Run directory: $RUN_DIR" >> "$LOG_FILE"
echo "Process ID: $$" >> "$LOG_FILE"
echo "PID: $$" > "$PID_FILE"

# Change to project directory
cd "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"

# Run the main script
echo "Executing: python3 {main_script} --config $CONFIG_PATH {' '.join(additional_args or [])}" >> "$LOG_FILE"
python3 {main_script} --config "$CONFIG_PATH" {' '.join(additional_args or [])} >> "$LOG_FILE" 2>&1

# Log completion
echo "Background {script_name} completed successfully at $(date)" >> "$LOG_FILE"
"""

    return script_content


def run_background_training(config_path: str, simulation: bool = False, 
                          additional_args: list = None, run_dir: str = None):
    """Run training or simulation in the background."""
    
    # Validate config file exists
    if not os.path.exists(config_path):
        console.print(f"[bold red]Error: Config file '{config_path}' not found![/bold red]")
        return False
    
    # Determine run directory
    if run_dir is None:
        # Create a timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_type = "simulation" if simulation else "training"
        run_dir = f"background_runs/{script_type}_{timestamp}"
    
    # Create run directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Create the background script
    script_content = create_background_script(
        "background_script.sh", config_path, run_dir, additional_args, simulation
    )
    
    script_path = os.path.join(run_dir, "background_script.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    # Create the nohup command
    log_file = os.path.join(run_dir, "nohup.log")
    pid_file = os.path.join(run_dir, "background_run.pid")
    
    # Run the background script
    cmd = [
        "nohup", "bash", script_path, 
        ">", log_file, "2>&1", "<", "/dev/null", "&"
    ]
    
    console.print(Panel(f"🚀 [bold green]Starting Background {'Simulation' if simulation else 'Training'}[/bold green]", style="green"))
    
    # Execute the command
    try:
        # Use shell=True to handle the redirection properly
        process = subprocess.Popen(
            f"nohup bash {script_path} > {log_file} 2>&1 < /dev/null &",
            shell=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Wait a moment for the process to start
        time.sleep(2)
        
        # Try to get the actual PID from the PID file
        actual_pid = None
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                actual_pid = f.read().strip()
        
        # Display information
        info_table = Table(title="Background Process Information", show_header=True, header_style="bold green")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Script Type", "Simulation" if simulation else "Training")
        info_table.add_row("Config File", config_path)
        info_table.add_row("Run Directory", run_dir)
        info_table.add_row("Process ID", actual_pid or "Unknown")
        info_table.add_row("Log File", log_file)
        info_table.add_row("PID File", pid_file)
        info_table.add_row("Background Script", script_path)
        
        console.print(info_table)
        
        console.print(Panel("📋 [bold blue]Monitoring Commands[/bold blue]", style="blue"))
        console.print(f"• Monitor progress: [cyan]tail -f {log_file}[/cyan]")
        console.print(f"• Check if running: [cyan]ps -p {actual_pid or 'PID'}[/cyan]")
        console.print(f"• Stop process: [cyan]kill {actual_pid or 'PID'}[/cyan]")
        console.print(f"• Check PID file: [cyan]cat {pid_file}[/cyan]")
        
        console.print(Panel("✅ [bold green]Background process started successfully![/bold green]", style="green"))
        console.print("You can now safely disconnect from SSH. The process will continue running.")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Error starting background process:[/bold red] {e}")
        return False


def list_background_processes():
    """List all running background processes."""
    console.print(Panel("📊 [bold blue]Background Processes[/bold blue]", style="blue"))
    
    background_runs_dir = Path("background_runs")
    if not background_runs_dir.exists():
        console.print("No background runs found.")
        return
    
    processes = []
    for run_dir in background_runs_dir.iterdir():
        if run_dir.is_dir():
            pid_file = run_dir / "background_run.pid"
            log_file = run_dir / "background_run.log"
            
            pid = None
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    pid = f.read().strip()
            
            # Check if process is still running
            running = False
            if pid:
                try:
                    subprocess.run(["ps", "-p", pid], check=True, capture_output=True)
                    running = True
                except subprocess.CalledProcessError:
                    running = False
            
            processes.append({
                'run_dir': str(run_dir),
                'pid': pid,
                'running': running,
                'log_file': str(log_file) if log_file.exists() else None
            })
    
    if not processes:
        console.print("No background processes found.")
        return
    
    # Display processes
    process_table = Table(title="Background Processes", show_header=True, header_style="bold blue")
    process_table.add_column("Run Directory", style="cyan")
    process_table.add_column("PID", style="green")
    process_table.add_column("Status", style="yellow")
    process_table.add_column("Log File", style="magenta")
    
    for proc in processes:
        status = "🟢 Running" if proc['running'] else "🔴 Stopped"
        process_table.add_row(
            proc['run_dir'],
            proc['pid'] or "Unknown",
            status,
            proc['log_file'] or "N/A"
        )
    
    console.print(process_table)


def stop_background_process(run_dir: str):
    """Stop a specific background process."""
    pid_file = os.path.join(run_dir, "background_run.pid")
    
    if not os.path.exists(pid_file):
        console.print(f"[bold red]PID file not found for {run_dir}[/bold red]")
        return False
    
    with open(pid_file, 'r') as f:
        pid = f.read().strip()
    
    try:
        # Send SIGTERM first
        os.kill(int(pid), signal.SIGTERM)
        console.print(f"[green]Sent SIGTERM to process {pid}[/green]")
        
        # Wait a bit
        time.sleep(2)
        
        # Check if still running
        try:
            subprocess.run(["ps", "-p", pid], check=True, capture_output=True)
            # Still running, send SIGKILL
            os.kill(int(pid), signal.SIGKILL)
            console.print(f"[yellow]Process {pid} still running, sent SIGKILL[/yellow]")
        except subprocess.CalledProcessError:
            console.print(f"[green]Process {pid} stopped successfully[/green]")
        
        # Clean up PID file
        os.remove(pid_file)
        return True
        
    except ProcessLookupError:
        console.print(f"[yellow]Process {pid} was not running[/yellow]")
        os.remove(pid_file)
        return True
    except Exception as e:
        console.print(f"[bold red]Error stopping process {pid}:[/bold red] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run training or simulation in background')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    parser.add_argument('--simulation', action='store_true',
                       help='Run simulation instead of just training')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Specific run directory (optional)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively (for simulation)')
    parser.add_argument('--research_only', action='store_true',
                       help='Generate only research questions plots (for simulation)')
    parser.add_argument('--no_confusion', action='store_true',
                       help='Skip confusion matrix generation (for simulation)')
    parser.add_argument('--advanced', action='store_true',
                       help='Run advanced multi-run analysis (for simulation)')
    parser.add_argument('--list', action='store_true',
                       help='List all background processes')
    parser.add_argument('--stop', type=str, default=None,
                       help='Stop a specific background process by run directory')
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_background_processes()
        return 0
    
    # Handle stop command
    if args.stop:
        if stop_background_process(args.stop):
            console.print(f"[green]Successfully stopped process in {args.stop}[/green]")
        else:
            console.print(f"[red]Failed to stop process in {args.stop}[/red]")
        return 0
    
    # Validate config file is provided for training/simulation
    if not args.config:
        console.print("[bold red]Error: --config is required for training/simulation![/bold red]")
        return 1
    
    # Prepare additional arguments for simulation
    additional_args = []
    if args.simulation:
        if args.show:
            additional_args.append("--show")
        if args.research_only:
            additional_args.append("--research_only")
        if args.no_confusion:
            additional_args.append("--no_confusion")
        if args.advanced:
            additional_args.append("--advanced")
    
    # Run background training/simulation
    if run_background_training(args.config, args.simulation, additional_args, args.run_dir):
        console.print(Panel("🎉 [bold green]Background process started successfully![/bold green]", style="green"))
        return 0
    else:
        console.print(Panel("❌ [bold red]Failed to start background process![/bold red]", style="red"))
        return 1


if __name__ == '__main__':
    exit(main())
