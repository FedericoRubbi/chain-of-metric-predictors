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

# Install dependencies
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
| `plot_curves.py` | **Basic visualization** | Generates training curves and research questions analysis |
| `plot_metrics.py` | **Advanced analysis** | Multi-run comparison and statistical analysis |
| `show_confusion.py` | **Confusion matrix** | Visualizes classification confusion matrices |

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
```bash
# Basic plots
python scripts/plot_curves.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08

# Advanced analysis
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/run1 runs/mnist/greedy/run2 --comprehensive

# Confusion matrix
python scripts/show_confusion.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08
```
**What they do:**
- `plot_curves.py`: Generates training curves, layerwise metrics, and research questions analysis
- `plot_metrics.py`: Compares multiple runs, creates correlation matrices, and statistical analysis
- `show_confusion.py`: Creates confusion matrix visualizations with optional normalization

## Configuration

Configuration files specify:
- **Dataset**: mnist, cifar10, cifar100
- **Model**: greedy, mlp
- **Architecture**: N (hidden size), layers, similarity type
- **Training**: epochs, batch size, learning rate, scheduler
- **Regularization**: tau (temperature), lambda_ace
- **Metrics**: metrics_log_frequency (default: 5)

## Metrics Collection

The system collects 10 comprehensive metrics during training:

- **Cosine-to-label anchors** (alignment & margin)
- **Layerwise accuracy and cross-entropy**
- **ACE regularizer** (between consecutive layers)
- **Mutual information** (input to layer outputs)
- **Gaussian entropy proxy** (EMA-based estimation)
- **One-shot linear probe** (ridge regression classification)
- **Participation ratio** (effective dimensionality)
- **Linear CKA** (layer similarity)
- **Ridge regression R²** (layer-to-layer predictability)

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

### Visualization
- `basic_curves.png` - Training curves
- `research_questions.png` - Research questions analysis
- `layerwise_{metric}_{phase}.png` - Layerwise metrics
- `metrics_heatmap_{phase}.png` - Metrics overview
- `confusion.png` - Confusion matrix plot

### Advanced Analysis
- `multi_run_{metric}_{layer}_{phase}.png` - Multi-run comparison
- `architecture_comparison_{metric}.png` - MLP vs GMLP
- `correlation_matrix_{phase}.png` - Metrics correlation
- `layer_progression_analysis.png` - Statistical analysis

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
│   ├─ plot_curves.py        # Basic visualization
│   ├─ plot_metrics.py       # Advanced visualization
│   └─ show_confusion.py     # Confusion matrix
├─ data/                      # Auto-downloaded datasets
└─ runs/                      # Training outputs
```