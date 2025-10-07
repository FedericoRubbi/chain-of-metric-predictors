#!/bin/bash
# Quick script to validate log files in run directories
#
# Usage:
#   ./scripts/validate_logs.sh                  # Validate all runs
#   ./scripts/validate_logs.sh runs/cifar100    # Validate specific directory
#   ./scripts/validate_logs.sh --help           # Show help

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [DIRECTORY]"
    echo ""
    echo "Validate log files against the standardized format."
    echo ""
    echo "Arguments:"
    echo "  DIRECTORY    Optional path to run directory (default: runs/)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Validate all runs"
    echo "  $0 runs/cifar100      # Validate CIFAR-100 only"
    echo "  $0 runs/mnist         # Validate MNIST only"
    exit 0
fi

RUN_DIR="${1:-runs/}"

echo "============================================================"
echo "         Log Format Validation"
echo "============================================================"
echo ""

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory not found: $RUN_DIR"
    exit 1
fi

python3 tests/test_log_format.py "$RUN_DIR"

