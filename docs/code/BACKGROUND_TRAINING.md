# Background Training Script

This script allows you to run long training sessions in the background with proper logging and process management. You can safely disconnect from SSH and the training will continue running.

## Features

- ✅ **Background Execution**: Runs training/simulation with `nohup` for SSH disconnection safety
- ✅ **Process Management**: Saves process IDs for monitoring and control
- ✅ **Comprehensive Logging**: All output logged to files with timestamps
- ✅ **Dual Mode Support**: Works with both training and simulation scripts
- ✅ **Process Monitoring**: List and stop running background processes
- ✅ **Automatic Cleanup**: Proper cleanup on process termination

## Quick Start

### Basic Training
```bash
# Run training in background
python3 scripts/run_background.py --config configs/greedy_mnist.yaml

# Or use the helper script
./bg_train.sh configs/greedy_mnist.yaml
```

### Full Simulation
```bash
# Run complete simulation (training + evaluation + visualization)
python3 scripts/run_background.py --config configs/mlp_cifar100.yaml --simulation

# With interactive plots
python3 scripts/run_background.py --config configs/mlp_cifar100.yaml --simulation --show
```

## Usage Examples

### 1. Basic Training
```bash
python3 scripts/run_background.py --config configs/my_config.yaml
```

### 2. Full Simulation
```bash
python3 scripts/run_background.py --config configs/my_config.yaml --simulation
```

### 3. Simulation with Options
```bash
python3 scripts/run_background.py --config configs/my_config.yaml --simulation \
    --show --research_only --advanced
```

### 4. Custom Run Directory
```bash
python3 scripts/run_background.py --config configs/my_config.yaml \
    --run_dir my_custom_run
```

## Process Management

### List Running Processes
```bash
python3 scripts/run_background.py --list
```

### Stop a Process
```bash
python3 scripts/run_background.py --stop background_runs/training_20250104_130000
```

### Monitor Progress
```bash
# Watch the main log
tail -f background_runs/training_20250104_130000/background_run.log

# Watch the nohup log
tail -f background_runs/training_20250104_130000/nohup.log
```

## File Structure

When you run a background process, it creates:

```
background_runs/
├── training_20250104_130000/          # Timestamped run directory
│   ├── background_script.sh           # Generated bash script
│   ├── background_run.pid             # Process ID file
│   ├── background_run.log             # Main execution log
│   ├── nohup.log                     # Nohup output log
│   └── [training outputs]            # All training/simulation outputs
```

## Command Line Options

### Main Options
- `--config CONFIG`: Path to YAML configuration file (required for training/simulation)
- `--simulation`: Run simulation instead of just training
- `--run_dir RUN_DIR`: Specific run directory (optional, auto-generated if not provided)

### Simulation Options
- `--show`: Show plots interactively instead of saving only
- `--research_only`: Generate only research questions plots
- `--no_confusion`: Skip confusion matrix generation
- `--advanced`: Run advanced multi-run analysis

### Process Management
- `--list`: List all background processes
- `--stop RUN_DIR`: Stop a specific background process by run directory

## Monitoring Commands

After starting a background process, you'll see output like:

```
🚀 Starting Background Training
┌──────────────────────────────────────────────────────────────────────────────┐
│ Background Process Information                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Property        │ Value                                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Script Type     │ Training                                                   │
│ Config File     │ configs/mnist_greedy.yaml                                  │
│ Run Directory   │ background_runs/training_20250104_130000                  │
│ Process ID      │ 12345                                                      │
│ Log File        │ background_runs/training_20250104_130000/background_run.log│
│ PID File        │ background_runs/training_20250104_130000/background_run.pid│
│ Background Script│ background_runs/training_20250104_130000/background_script.sh│
└──────────────────────────────────────────────────────────────────────────────┘

📋 Monitoring Commands
• Monitor progress: tail -f background_runs/training_20250104_130000/background_run.log
• Check if running: ps -p 12345
• Stop process: kill 12345
• Check PID file: cat background_runs/training_20250104_130000/background_run.pid
```

## Safety Features

- **SSH Disconnection Safe**: Uses `nohup` to prevent termination on SSH disconnect
- **Process Tracking**: Saves PID for monitoring and control
- **Automatic Cleanup**: Removes PID files on completion
- **Error Handling**: Proper error handling and logging
- **Signal Handling**: Graceful shutdown on SIGTERM/SIGINT

## Troubleshooting

### Process Not Starting
- Check that the config file exists and is valid
- Ensure you have write permissions in the project directory
- Check the nohup.log file for error messages

### Process Not Stopping
- Use `kill -9 PID` for forceful termination
- Check if the PID file exists and contains the correct process ID
- Use `ps aux | grep python` to find running processes

### Log Files Not Updating
- Check file permissions
- Ensure the process is still running with `ps -p PID`
- Check disk space availability

## Integration with Existing Scripts

This script works seamlessly with:
- `scripts/train.py` - Basic training
- `scripts/run_simulation.py` - Complete simulation workflow
- All existing configuration files
- All existing model types (greedy, mlp, etc.)

The background script automatically handles all the complexity of process management while preserving the full functionality of the underlying training and simulation scripts.
