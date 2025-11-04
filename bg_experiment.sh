#!/bin/bash
# Quick background experiment launcher
# Usage: ./bg_experiment.sh <experiment_spec.yaml> [simulation]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_spec.yaml> [simulation]"
    echo "Examples:"
    echo "  $0 experiments/lr_sweep_mnist_mlp.yaml"
    echo "  $0 experiments/grid_mnist_mlp.yaml simulation"
    exit 1
fi

EXP_FILE="$1"
SIMULATION_FLAG=""

if [ "$2" = "simulation" ]; then
    SIMULATION_FLAG="--simulation"
fi

echo "🚀 Starting background experiment..."
echo "Spec: $EXP_FILE"
echo "Mode: $([ -n "$SIMULATION_FLAG" ] && echo "Simulation" || echo "Training")"

python3 scripts/run_background.py --experiment "$EXP_FILE" $SIMULATION_FLAG

echo "✅ Background experiment started!"
echo "Use 'python3 scripts/run_background.py --list' to see running processes"


