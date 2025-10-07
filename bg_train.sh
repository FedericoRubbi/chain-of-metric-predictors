#!/bin/bash
# Quick background training launcher
# Usage: ./bg_train.sh <config_file> [simulation]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [simulation]"
    echo "Examples:"
    echo "  $0 configs/mnist_greedy.yaml"
    echo "  $0 configs/cifar100_mlp.yaml simulation"
    exit 1
fi

CONFIG_FILE="$1"
SIMULATION_FLAG=""

if [ "$2" = "simulation" ]; then
    SIMULATION_FLAG="--simulation"
fi

echo "🚀 Starting background training..."
echo "Config: $CONFIG_FILE"
echo "Mode: $([ -n "$SIMULATION_FLAG" ] && echo "Simulation" || echo "Training")"

python3 scripts/run_background.py --config "$CONFIG_FILE" $SIMULATION_FLAG

echo "✅ Background process started!"
echo "Use 'python3 scripts/run_background.py --list' to see running processes"
