# Chain of Metric Predictors

This project implements neural network architectures for classification tasks using fully-connected linear layers with greedy training and comprehensive metric learning objectives.

## Architecture Overview

The project supports two main architectures:

- **MLP Baseline**: Standard multi-layer perceptron trained with backpropagation
- **Greedy MLP (GMLP)**: MLP trained with greedy backpropagation and metric learning objectives

### Key Features
- Sequence of fully-connected linear layers (with bias and ReLU activations)
- Each layer maps vectors of size N to vectors of size N (except first layer: input → N)
- Greedy training with individual optimizers per layer
- Metric learning with class anchors and temperature-scaled softmax
- ACE (Amended Cross Entropy) regularization between consecutive layers

## Research Questions

The system collects comprehensive metrics to answer:

1. **Are later layers getting more predictive?**
2. **Are layers compressing?**
3. **Are layers genuinely different from previous layers?**

## Quick Start

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (includes all plotting libraries)
pip install -r requirements.txt
```

### 🚀 Run Complete Simulation (Recommended)

Run training, evaluation, and visualization with a single command:

```bash
# Complete simulation
python scripts/run_simulation.py --config configs/greedy_mnist.yaml

# With interactive plots
python scripts/run_simulation.py --config configs/greedy_mnist.yaml --show

# Research questions analysis only
python scripts/run_simulation.py --config configs/greedy_mnist.yaml --research_only

# Compare multiple runs
python scripts/run_simulation.py --config configs/greedy_mnist.yaml --compare_runs runs/mnist/mlp/run1 runs/mnist/greedy/run2
```

**Features:**
- Automated workflow: training → evaluation → visualization
- Rich progress display with status updates
- Comprehensive reporting of generated files
- Error handling with informative messages
- Multi-run comparison capabilities

## Individual Scripts

### Script Overview

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `run_simulation.py` | **Complete workflow automation** | Training → Evaluation → Visualization in one command |
| `train.py` | **Model training** | Trains MLP or GMLP models with comprehensive metrics collection |
| `eval.py` | **Model evaluation** | Evaluates trained models and generates test results |
| `plot_metrics.py` | **🎯 Comprehensive visualization** | **ALL plot types**: training curves, confusion matrices, heatmaps, correlations, distributions, architecture comparison, layer progression, timeline analysis |

### Training
```bash
python scripts/train.py --config configs/greedy_mnist.yaml
python scripts/train.py --config configs/mlp_mnist.yaml
```
**What it does:** Trains models with specified configuration, collects comprehensive metrics during training, saves checkpoints and logs.

### Evaluation
```bash
python scripts/eval.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08 --confusion
```
**What it does:** Loads trained model, evaluates on test set, generates accuracy results and confusion matrix.

### Visualization

#### 🎯 Comprehensive Plotting (`plot_metrics.py`)

The enhanced plotting script provides complete visualization coverage for all training metrics:

```bash
# Generate ALL plots (comprehensive analysis)
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --comprehensive

# Basic training curves (loss, accuracy)
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --basic_curves

# Confusion matrix visualization
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --confusion runs/mnist/greedy/0001_2025-09-22_19-31-08

# Per-layer metric heatmaps
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --heatmap accuracy

# Training vs validation comparison
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --train_val

# All metrics overview
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --overview

# Metric distributions
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --distributions

# Architecture comparison (MLP vs GMLP)
python scripts/plot_metrics.py --run_dirs runs/mnist/mlp/run1 runs/mnist/greedy/run2 --architecture_comparison

# Metrics correlation matrix
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --correlation

# Layer progression analysis
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --layer_progression

# Timeline for specific metric
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --timeline accuracy --layer layer_0
```

**Available Plot Types:**
- **Basic Training Curves**: Loss and accuracy over training steps/epochs
- **Confusion Matrix**: Classification confusion matrices with normalization support
- **Per-Layer Heatmaps**: Metric values across layers and runs
- **Train vs Validation**: Overfitting analysis and comparison
- **All Metrics Overview**: Comprehensive view of all available metrics
- **Metric Distributions**: Statistical analysis of metric ranges
- **Architecture Comparison**: MLP vs GMLP performance comparison
- **Correlation Analysis**: Relationships between different metrics
- **Layer Progression**: How metrics evolve across layers
- **Timeline Analysis**: Metric evolution over training time

## Configuration

Configuration files specify:
- **Dataset**: mnist, cifar10, cifar100
- **Model**: greedy, mlp
- **Architecture**: N (hidden size), layers, similarity type
- **Training**: epochs, batch size, learning rate, scheduler
- **Regularization**: tau (temperature), lambda_ace
- **Metrics**: metrics_log_frequency (default: 5)

## Metrics Collection

The system collects comprehensive metrics during training that can be visualized with `plot_metrics.py`:

### 📊 Available Metrics for Visualization

| Metric | Description | Research Question |
|--------|-------------|-------------------|
| `accuracy` | Anchor-based accuracy using class anchors | Are later layers getting more predictive? |
| `probe_accuracy` | Linear probe accuracy (ridge regression) | Are later layers getting more predictive? |
| `cross_entropy` | Cross-entropy loss per layer | Are later layers getting more predictive? |
| `alignment` | Cosine alignment with class anchors | Are layers genuinely different? |
| `margin` | Classification margin from anchors | Are layers genuinely different? |
| `ace_regularizer` | ACE regularizer between consecutive layers | Are layers genuinely different? |
| `mutual_information` | Mutual information (input to layer outputs) | Are layers compressing? |
| `gaussian_entropy` | Gaussian entropy proxy (EMA-based) | Are layers compressing? |
| `f1_score` | F1 score from linear probe | Are later layers getting more predictive? |
| `linear_cka` | Linear CKA similarity between layers | Are layers genuinely different? |

### 🔬 Additional Metrics (when enabled)
- `participation_ratio` - Effective dimensionality (slow metric)
- `ridge_r2` - Ridge regression R² (layer-to-layer predictability)

**Metrics Logging Frequency:**
- `5` (default): Every 5 iterations
- `1`: Every iteration (detailed but slower)
- `20`: Every 20 iterations (faster but less detailed)

## Output Files

Each run creates a timestamped directory with:

### Training
- `params.yaml` - Complete configuration
- `log.jsonl` - Training logs with metrics
- `best.pt` - Best model checkpoint
- `last.pt` - Final model checkpoint

### Evaluation
- `test_results.json` - Test accuracy results
- `confusion.npy` - Confusion matrix
- `class_report.json` - Classification report

### Visualization (Generated by `plot_metrics.py`)

#### 🎯 Comprehensive Analysis
- `basic_training_curves.png` - Training loss and accuracy curves
- `all_metrics_overview_{phase}.png` - Complete overview of all metrics
- `metric_distributions_{phase}.png` - Statistical distributions of metrics
- `layer_progression_analysis.png` - How metrics evolve across layers

#### 🔍 Specific Analysis
- `heatmap_{metric}_{phase}.png` - Per-layer metric heatmaps
- `train_val_comparison_{metric}.png` - Training vs validation comparison
- `multi_run_{metric}_{layer}_{phase}.png` - Multi-run comparison
- `architecture_comparison_{metric}.png` - MLP vs GMLP comparison
- `correlation_matrix_{phase}.png` - Metrics correlation analysis
- `timeline_{metric}_{layer}.png` - Metric evolution over time

#### 📊 Confusion Matrices
- `confusion_matrix.png` - Classification confusion matrix
- `confusion_matrix_normalized.png` - Normalized confusion matrix

#### 📈 Legacy Plots (from `plot_curves.py`)
- `research_questions.png` - Research questions analysis
- `layerwise_{metric}_{phase}.png` - Layerwise metrics
- `metrics_heatmap_{phase}.png` - Metrics overview

## Project Structure

```
project/
├─ configs/                    # YAML configurations
├─ models/                     # Model implementations
├─ trainers/                   # Training logic
├─ utils/                      # Utilities (metrics, anchors, etc.)
├─ scripts/                    # CLI scripts
│   ├─ run_simulation.py     # Comprehensive simulation
│   ├─ train.py              # Training
│   ├─ eval.py               # Evaluation
│   ├─ plot_curves.py        # Basic visualization (legacy)
│   ├─ plot_metrics.py       # 🎯 Comprehensive visualization (primary)
│   └─ show_confusion.py     # Confusion matrix
├─ data/                      # Auto-downloaded datasets
└─ runs/                      # Training outputs
```

## 🚀 Quick Reference

### Most Common Commands

```bash
# Complete workflow
python scripts/run_simulation.py --config configs/greedy_mnist.yaml

# Generate all plots for a run
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --comprehensive

# Basic training curves only
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --basic_curves

# Compare MLP vs GMLP
python scripts/plot_metrics.py --run_dirs runs/mnist/mlp/run1 runs/mnist/greedy/run2 --architecture_comparison

# All metrics overview
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_2025-09-22_19-31-08 --overview
```

### Help and Options

```bash
# Show all plotting options
python scripts/plot_metrics.py --help

# Show simulation options
python scripts/run_simulation.py --help
```