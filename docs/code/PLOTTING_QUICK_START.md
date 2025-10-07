# Plotting Quick Start Guide

Quick reference for using the improved plotting system.

## Basic Usage

### Generate All Plots (Recommended)
```bash
# Single run
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0007_2025-10-03_19-08-01 --comprehensive

# Multiple runs
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0007_* --comprehensive

# With live display
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0007_* --comprehensive --show
```

### Generate Specific Plot Types

#### Basic Training Curves (Loss & Accuracy)
```bash
python scripts/plot_metrics.py --run_dirs <run_dir> --basic_curves
```

#### Specific Metric Comparison
```bash
# All layers
python scripts/plot_metrics.py --run_dirs <run_dir> --metric probe_accuracy

# Specific layer
python scripts/plot_metrics.py --run_dirs <run_dir> --metric probe_accuracy --layer layer_2

# Training phase
python scripts/plot_metrics.py --run_dirs <run_dir> --metric probe_accuracy --phase train
```

#### Timeline for Specific Metric & Layer
```bash
python scripts/plot_metrics.py --run_dirs <run_dir> --timeline probe_accuracy --layer layer_1
```

#### Architecture Comparison (MLP vs GMLP)
```bash
python scripts/plot_metrics.py --run_dirs runs/mnist/*/0007_* --architecture_comparison
```

#### Training vs Validation Comparison
```bash
python scripts/plot_metrics.py --run_dirs <run_dir> --train_val
```

#### All Metrics Overview
```bash
python scripts/plot_metrics.py --run_dirs <run_dir> --overview
```

#### Confusion Matrix
```bash
# First generate confusion matrix
python eval.py --run_dir <run_dir> --confusion

# Then plot it
python scripts/plot_metrics.py --run_dirs <run_dir> --confusion <run_dir>

# Normalized version
python scripts/plot_metrics.py --run_dirs <run_dir> --confusion <run_dir> --normalize
```

## Output Organization

Plots are automatically organized into subdirectories:

```
<run_dir>/plots/
├── accuracy/
│   ├── multi_run_prediction_accuracy_all_val.png
│   ├── layer_profile_accuracy_val.png
│   ├── timeline_accuracy_layer_0_val.png
│   ├── timeline_accuracy_layer_1_val.png
│   ├── timeline_accuracy_layer_2_val.png
│   └── timeline_accuracy_layer_3_val.png
├── probe_accuracy/
│   ├── multi_run_probe_accuracy_all_val.png
│   ├── layer_profile_probe_accuracy_val.png
│   ├── timeline_probe_accuracy_layer_0_val.png
│   ├── timeline_probe_accuracy_layer_1_val.png
│   ├── timeline_probe_accuracy_layer_2_val.png
│   └── timeline_probe_accuracy_layer_3_val.png
├── cross_entropy/
│   ├── multi_run_cross_entropy_all_val.png
│   ├── layer_profile_cross_entropy_val.png
│   ├── timeline_cross_entropy_layer_0_val.png
│   ├── timeline_cross_entropy_layer_1_val.png
│   ├── timeline_cross_entropy_layer_2_val.png
│   └── timeline_cross_entropy_layer_3_val.png
├── regularization/
│   ├── multi_run_alignment_all_val.png
│   ├── multi_run_margin_all_val.png
│   ├── multi_run_ace_regularizer_all_val.png
│   └── ...
├── entropy/
│   ├── multi_run_gaussian_entropy_all_val.png (log scale)
│   ├── multi_run_mutual_information_all_val.png
│   └── ...
└── general/
    ├── basic_training_curves.png
    ├── train_val_comparison_accuracy.png
    └── ...
```

## Key Features

### 1. Automatic Single/Multi-Run Detection
- **Single run**: Generates bar charts for layer profiles
- **Multiple runs**: Generates heatmaps for comparison

### 2. Log Scale for Entropy Metrics
All gaussian_entropy plots automatically use logarithmic y-axis for clarity.

### 3. Optimized Y-Axis
Bar charts automatically adjust y-axis to start from minimum value with margin, making differences more visible.

### 4. Complete Layer Coverage
Timeline plots now cover all layers (0, 1, 2, 3) for key metrics:
- accuracy
- probe_accuracy
- cross_entropy

### 5. Separated Accuracy Plots
- `multi_run_prediction_accuracy_all_val.png` - Prediction accuracy only
- `multi_run_probe_accuracy_all_val.png` - Probe accuracy only

## Common Workflows

### Analyzing a Single Training Run
```bash
cd /path/to/chain-of-metric-predictors
python scripts/plot_metrics.py \
    --run_dirs runs/mnist/greedy/0007_2025-10-03_19-08-01 \
    --comprehensive
```

### Comparing Multiple Runs
```bash
python scripts/plot_metrics.py \
    --run_dirs runs/mnist/greedy/0007_* \
    --comprehensive
```

### Comparing MLP vs GMLP
```bash
python scripts/plot_metrics.py \
    --run_dirs runs/mnist/mlp/0001_* runs/mnist/greedy/0007_* \
    --architecture_comparison
```

### Deep Dive on Specific Metric
```bash
# Overview across all layers
python scripts/plot_metrics.py --run_dirs <run_dir> --metric probe_accuracy

# Timeline for each layer
for layer in 0 1 2 3; do
    python scripts/plot_metrics.py --run_dirs <run_dir> \
        --timeline probe_accuracy --layer layer_$layer
done
```

## Tips

1. **Use wildcards** for multiple runs: `runs/mnist/greedy/0007_*`
2. **Check subdirectories** for organized output
3. **Use `--show`** during development to view plots immediately
4. **Use `--comprehensive`** for complete analysis
5. **Entropy metrics** automatically use log scale
6. **Bar charts** automatically optimize y-axis range

## Troubleshooting

### No plots generated
- Check that `log.jsonl` exists in run directory
- Verify metrics are present in log file
- Check console for error messages

### Missing timeline plots
- Ensure you're using `--comprehensive` flag
- Check that metrics exist for all layers

### Heatmaps look strange
- Single-run heatmaps automatically convert to bar charts
- Multi-run heatmaps require consistent layer structure

### Architecture comparison fails
- Need both MLP and GMLP/Greedy runs
- Check run names contain 'mlp' or 'greedy'/'gmlp'

