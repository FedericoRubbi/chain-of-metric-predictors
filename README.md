# Chain of Metric Predictors

This project implements a novel architecture for classification tasks using a sequence of fully-connected linear layers with greedy training and metric learning objectives.

## Architecture Overview

The architecture consists of:
- A sequence of fully-connected linear layers (with bias and ReLU activations)
- Each layer maps vectors of size N to vectors of size N (except the first layer which maps input to N)
- Each layer is trained greedily with its own optimizer
- Uses metric learning with class anchors and temperature-scaled softmax
- Includes ACE (Amended Cross Entropy) regularization between consecutive layers

## Project Structure

```
project/
├─ data/                       # Auto-downloaded datasets
├─ configs/                    # YAML configs per dataset/model
├─ models/                     # Model implementations
├─ trainers/                   # Training logic
├─ utils/                      # Utilities (seed, run manager, anchors, etc.)
├─ scripts/
│   └─ train.py               # CLI training script
└─ requirements.txt
```

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run training with a specific configuration:

```bash
# MNIST with greedy architecture
python scripts/train.py --config configs/greedy_mnist.yaml

# CIFAR-10 with greedy architecture
python scripts/train.py --config configs/greedy_cifar10.yaml

# CIFAR-100 with greedy architecture
python scripts/train.py --config configs/greedy_cifar100.yaml

# MNIST with MLP baseline
python scripts/train.py --config configs/mlp_mnist.yaml
```

### Evaluation

Evaluate a trained model:

```bash
# Basic evaluation
python scripts/eval.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08

# Evaluation with confusion matrix
python scripts/eval.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08 --confusion

# Evaluate using last checkpoint instead of best
python scripts/eval.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08 --ckpt last.pt
```

### Visualization

Generate training curves and confusion matrices:

```bash
# Plot training curves
python scripts/plot_curves.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08

# Show plots interactively
python scripts/plot_curves.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08 --show

# Visualize confusion matrix
python scripts/show_confusion.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08

# Normalized confusion matrix
python scripts/show_confusion.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08 --normalize

# With custom class labels
python scripts/show_confusion.py --run_dir runs/mnist/greedy/0001_2025-09-22_19-31-08 --labels "0,1,2,3,4,5,6,7,8,9"
```

### Configuration

Each configuration file specifies:
- Dataset (mnist, cifar10, cifar100)
- Model type (greedy, mlp)
- Architecture parameters (N, layers, similarity type)
- Training parameters (epochs, batch size, learning rate)
- Regularization (tau, lambda_ace)

## Output Files

Each training run creates a timestamped directory with:

### Training Outputs
- `params.yaml`: Complete configuration used for the run
- `log.jsonl`: Training logs with loss and accuracy metrics
- `best.pt`: Best model checkpoint (by validation accuracy)
- `last.pt`: Final model checkpoint

### Evaluation Outputs (after running eval.py)
- `test_results.json`: Test accuracy results
- `confusion.npy`: Confusion matrix (numpy array)
- `class_report.json`: Detailed classification report

### Visualization Outputs (after running plot_curves.py)
- `plot_train_loss.png`: Training loss curve
- `plot_train_acc_last.png`: Training accuracy curve
- `plot_val_acc_last.png`: Validation accuracy curve
- `confusion.png`: Confusion matrix visualization (after running show_confusion.py)
