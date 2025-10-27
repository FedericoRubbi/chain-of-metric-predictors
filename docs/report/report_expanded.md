## Expanded Report: Greedy MLP vs Vanilla MLP in Chain of Metric Predictors

This document expands the LaTeX report by adding implementation-level details: datasets, training procedures, logging format, metrics actually collected, plotting and analysis utilities, configuration knobs, outputs, and practical guidance to reproduce and extend results.

### Datasets and Preprocessing

- **Supported datasets**: `mnist` (10 classes), `cifar10` (10), `cifar100` (100)
- **Input dimensions**:
  - MNIST: `1×28×28 → 784`
  - CIFAR-10/100: `3×32×32 → 3072`
- **Transforms**:
  - MNIST: `ToTensor + Normalize(mean=0.1307, std=0.3081)`
  - CIFAR: RandomCrop(32, padding=4) + RandomHorizontalFlip + `ToTensor + Normalize` (per-channel)
- **Splits**: training set is split into train/val with a fixed seed; test is separate.

## Architectures

### Vanilla MLP (Baseline)

- Stack of `Linear(+bias) + ReLU` blocks.
- First layer: `input_dim → N`; subsequent: `N → N`.
- Final classification head is the last `Linear` producing class logits.
- Training is end-to-end with a single optimizer and standard cross-entropy.

### Greedy MLP (GMLP)

- Same `Linear + ReLU` structure of width `N` for `L` layers, but no separate head.
- Fixed class anchors `E ∈ R^{C×N}` (loaded or deterministically generated per dataset/seed).
- At each layer `i`, embeddings `h_i` are compared to anchors using a similarity (cosine or negative L2) and converted to probabilities with temperature-scaled softmax `τ`.
- Loss per layer: `H(p, q_i) − λ_ACE · H(q_i, q_{i+1})`, where the regularizer vanishes at the last layer.
- Optimization is greedy: each layer has its own optimizer; inputs to the current layer are detached from previous gradients.

## Training Procedures

### Common

- **Device**: CUDA if available, otherwise CPU.
- **Optimizer**: Adam with configurable `lr`, `weight_decay`.
- **Scheduler**: Warmup + cosine decay (`WarmupCosine`) with total steps computed from epochs and loader length.
- **AMP**: Automatic mixed precision enabled.
- **Anchors**: Deterministic per dataset/seed; stored in `artifacts/anchors_{dataset}_C{num_classes}_N{N}_seed{seed}.pt`.

### BaselineTrainer specifics

- Uses a single optimizer for all parameters.
- Each step computes: `embeddings_list = model.forward_all(x)` for per-layer metrics, and `logits = model(x)` for loss/accuracy.
- Metrics collection frequency:
  - `metrics_frequency: iteration` → log every `metrics_log_frequency` steps.
  - `metrics_frequency: epoch` → log only on the last iteration of the epoch.
- Validation averages metrics across all batches and logs them.

### GreedyTrainer specifics

- Per-layer optimizers (`Adam`) and per-iteration scheduler steps for each optimizer.
- Greedy step: for layer `i`, compute `h_i` and local loss; detach inputs so only layer `i` updates.
- For ACE, compute `q_{i+1}` with `torch.no_grad()` when `i < L`.
- Logs last-layer accuracy for training; evaluation returns accuracies for all layers and averages metrics.

## Logging: Structure and Semantics

All trainers write JSON Lines (`.jsonl`) to `runs/.../log.jsonl`. Each line is a JSON object.

- **Base fields** (all entries):
  - `phase`: `"train" | "val"`
  - `epoch`: integer ≥ 1
  - `iter`: iteration within epoch (train: ≥1; val: always 0)
  - `timestamp`: ISO string
- **Training-only fields**:
  - `step`: global step counter (cumulative over epochs)
  - `loss`: training loss (float)
  - `acc`: training accuracy (baseline: overall; greedy: last layer)
- **Validation-only fields**:
  - `acc`: validation accuracy (baseline: overall; greedy: last layer)
- **Metrics fields** (optional depending on frequency):
  - `metrics`: per-layer metrics computed on current phase data
  - `train_metrics`: when `metrics_frequency: epoch`, validation entries also include the last epoch’s training metrics snapshot

Example (training with metrics):

```json
{
  "phase": "train",
  "epoch": 1,
  "iter": 400,
  "step": 400,
  "loss": 3.74,
  "acc": 0.125,
  "timestamp": "2025-10-04T14:06:36.155810",
  "metrics": {
    "layer_0": {"accuracy": 0.11, "cross_entropy": 4.54, "alignment": 0.06, "margin": -0.07}
  }
}
```

Notes:

- Older runs may omit `timestamp` (treated as a warning, not an error, by the validator).
- Validation logs must not include `loss`.

## Metrics: What Is Actually Collected

Metrics are computed per layer (`layer_0` … `layer_{L-1}`) by dedicated collectors. Some metrics are always on; some can be toggled for performance.

### Core classification metrics

- **accuracy**: Top-1 accuracy from anchor logits or model logits.
- **cross_entropy**: CE between logits and labels (mean over batch).

### Anchor/geometry metrics (require anchors)

- **alignment**: Mean cosine similarity between embeddings and the correct class anchor.
- **margin**: Mean difference between correct anchor similarity and the maximum incorrect anchor similarity.

### Regularization coupling

- **ace_regularizer**: `H(q_i, q_{i+1})` (cross-entropy between consecutive layer distributions); present when both `q_i` and `q_{i+1}` are computed.

### Information/compression proxies

- **mutual_information**: Histogram-based MI between inputs and layer embeddings (on PCA-reduced components).
- **gaussian_entropy**: EMA-based Gaussian entropy proxy per layer (non-negative, tracked by a per-layer estimator).

### Linear probe predictivity

- **probe_accuracy**: Accuracy of a one-shot RidgeClassifier trained and evaluated on the batch embeddings.
- **f1_score**: Weighted F1 of the probe.

### Inter-layer similarity/predictability

- **linear_cka**: Linear CKA of `layer_{i-1}` vs `layer_i` embeddings (NaN for `layer_0`).
- **ridge_r2**: R² of a ridge regression predicting `layer_{i+1}` from `layer_i` (enabled when slow metrics are on).

### Effective dimensionality (optional, slow)

- **participation_ratio**: Functional dimensionality based on eigen-spectrum of activation covariance.

### Performance controls

- `enable_slow_metrics: false` by default. When `true`, `participation_ratio` and `ridge_r2` are included (not recommended for every iteration due to cost).
- `metrics_frequency: iteration|epoch` and `metrics_log_frequency: int` gate how often metrics are computed.

## Evaluation

- Baseline: computes final logits and accuracy; collects per-layer metrics using `forward_all` + anchor logits; averages metrics across validation batches.
- Greedy: computes per-layer anchor logits; returns a list of accuracies per layer and averages the metrics across validation batches; last-layer accuracy is used for model selection.

## Plotting and Analysis

The primary analysis script is `scripts/plot_metrics.py`. It supports both single-run and multi-run views and organizes outputs into subdirectories by metric category.

- **Always useful**: basic training curves, per-metric timelines, train-vs-val comparisons, confusion matrices, all-metrics overview.
- **Multi-run**: per-layer heatmaps (auto-falls back to single-run layer profile), architecture comparison (MLP vs GMLP).
- **Special handling**: log scale for entropy; y-axis limits optimized for bar plots; separate prediction vs probe accuracy charts.

Key CLI examples:

```bash
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_... --comprehensive
python scripts/plot_metrics.py --run_dirs runs/mnist/greedy/0001_... --timeline accuracy --layer layer_0
python scripts/plot_metrics.py --run_dirs runs/mnist/mlp/run1 runs/mnist/greedy/run2 --architecture_comparison
```

## Configuration Surface

Configuration files (YAML) define dataset, model, architecture, training, regularization, and metrics cadence. Important keys:

- **Dataset**: `dataset: mnist|cifar10|cifar100`
- **Model**: `model: mlp|greedy`
- **Architecture**: `N` (width), `layers` (depth), `similarity: cosine|l2`
- **Training**: `epochs`, `batch_size`, `lr`, `weight_decay`, `warmup_ratio`
- **Greedy-only**: `tau` (temperature), `lambda_ace` (list of per-layer ACE weights), `ace_variant` (ce_i_next|ce_next_i|js|sym_kl)
- **Metrics**: `metrics_frequency: iteration|epoch`, `metrics_log_frequency`, `enable_slow_metrics`
- **Reproducibility**: `seed`

Example snippet:

```yaml
dataset: mnist
model: greedy
N: 256
layers: 4
similarity: cosine
tau: 1.0
lambda_ace: [1.0e-3, 1.0e-3, 1.0e-3]  # 3 values for 4 layers (connections: 1→2, 2→3, 3→4)
ace_variant: ce_i_next  # ce_i_next | ce_next_i | js | sym_kl
epochs: 20
batch_size: 128
lr: 1.0e-3
weight_decay: 1.0e-4
warmup_ratio: 0.03
metrics_frequency: iteration
metrics_log_frequency: 5
enable_slow_metrics: false
seed: 0
```

## Outputs per Run

Every run directory contains:

- Training: `params.yaml`, `log.jsonl`, `best.pt`, `last.pt`
- Evaluation: `test_results.json`, `confusion.npy`, `class_report.json`
- Visualization (generated): organized PNGs for curves, heatmaps, comparisons, timelines, confusion matrices, and overviews.

## Reproduction and Background Execution

- End-to-end automation: `python scripts/run_simulation.py --config configs/greedy_mnist.yaml`
- Background-safe execution with logging and process management: `python scripts/run_background.py --config ... [--simulation]`
- Log validation: `./scripts/validate_logs.sh` or `python3 tests/test_log_format.py runs/`

## Practical Notes and Caveats

- `gaussian_entropy` can be near zero early in training until estimators stabilize.
- `linear_cka` is undefined for `layer_0` (no previous layer); represented as NaN.
- Probe metrics can appear optimistic when evaluated on the same batch used for fitting (by design for speed/consistency across steps).
- Older logs may lack `timestamp`; tests treat it as an optional but recommended field.
- For fair architecture comparisons, align `N`, `layers`, optimizer hyperparameters, and data preprocessing across runs.

## How This Expands the Original Report

- Concretizes logging schema (fields, phases, frequencies) and where files live.
- Details the complete metric set, with when/how each is computed and toggles for cost control.
- Clarifies training loops for both trainers, including AMP and schedulers.
- Documents plotting capabilities, outputs, and recommended analyses linked to the research questions.
- Provides configuration examples and operational guidance for reproducibility.


