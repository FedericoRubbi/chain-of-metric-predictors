# Chain of Metric Predictors

This project implements a chosen architecture for classification tasks using a sequence of fully-connected linear layers with greedy training and metric learning objectives.

## Architecture Overview

The architecture consists of:
- A sequence of fully-connected linear layers (with bias and ReLU activations)
- Each layer maps vectors of size N to vectors of size N (except the first layer which maps input to N)
- Each layer is trained greedily with its own optimizer
- Uses metric learning with class anchors and temperature-scaled softmax
- Includes ACE (Amended Cross Entropy) regularization between consecutive layers


## Testing
### What architectures to test?
- baseline MLP trained with BP (MLP)
- baseline MLP trained with forward-forward algorithm (FF)
- baseline MLP trained with collaborative forward-forward (CFF)
- MLP trained with greedy BP (GMLP)


### What to test?
metrics provide insight for the following questions:
- are later layers getting more predictive?
- are layers compressing?
- are layers genuinely different from previous layers?

### What data to collect for each epoch?

MLP:
- cosine-to-label anchors (per layer) for alignment: \;\overline{\cos(z_\ell, a_{y})} and margin: \;\overline{\cos(z_\ell, a_{y}) - \max_{k\neq y}\cos(z_\ell, a_{k})}
- layerwise accuracy on test set
- layerwise cross-entropy with GT
- layerwise ACE regularizer, namely cross-entropy between consecutive softmax heads (averaged over a batch)
- functional entropy of each layer
- MI between input and each layer's output
- gaussian-entropy proxy of EMA of activations as \widehat{H}(Z_\ell)\propto \sum_i \tfrac12 \log \hat\sigma_i^2
- F1 score, accuracy One-shot linear probe (ridge, closed form), fit W = (X^\top X + \lambda I)^{-1}X^\top Y on one held-out minibatch
- participance ration from the activation covariance C on a minibatch (or diagonal approx): \mathrm{PR}=\frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}
- linear CKA (minibatch): \text{CKA}(Z_\ell,Z_{\ell+1}) = \frac{\|Z_\ell^\top Z_{\ell+1}\|F^2}{\|Z\ell^\top Z_\ell\|F \cdot \|Z{\ell+1}^\top Z_{\ell+1}\|_F}
- R^2 of ridge regression Z_{\ell+1}\approx Z_\ell W on a minibatch

GMLP:
- cosine-to-label anchors (per layer) for alignment: \;\overline{\cos(z_\ell, a_{y})} and margin: \;\overline{\cos(z_\ell, a_{y}) - \max_{k\neq y}\cos(z_\ell, a_{k})}
- layerwise accuracy on test set
- layerwise cross-entropy with GT
- layerwise ACE regularizer, namely cross-entropy between consecutive softmax heads (averaged over a batch)
- functional entropy of each layer
- MI between input and each layer's output
- gaussian-entropy proxy of EMA of activations as \widehat{H}(Z_\ell)\propto \sum_i \tfrac12 \log \hat\sigma_i^2
- F1 score, accuracy One-shot linear probe (ridge, closed form), fit W = (X^\top X + \lambda I)^{-1}X^\top Y on one held-out minibatch
- participance ration from the activation covariance C on a minibatch (or diagonal approx): \mathrm{PR}=\frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}
- linear CKA (minibatch): \text{CKA}(Z_\ell,Z_{\ell+1}) = \frac{\|Z_\ell^\top Z_{\ell+1}\|F^2}{\|Z\ell^\top Z_\ell\|F \cdot \|Z{\ell+1}^\top Z_{\ell+1}\|_F}
- R^2 of ridge regression Z_{\ell+1}\approx Z_\ell W on a minibatch

FF:
[TO BE DEFINED]

CFF:
[TO BE DEFINED]

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
