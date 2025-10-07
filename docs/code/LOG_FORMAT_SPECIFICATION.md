# Log Format Specification for Chain-of-Metric-Predictors

This document provides a detailed specification of the log format used by both the Baseline Trainer (MLP) and Greedy Trainer (GMLP) in the chain-of-metric-predictors project.

## Overview

Both trainers log training progress to JSONL (JSON Lines) files, where each line contains a JSON object representing a single log entry. The logs contain training metrics, validation metrics, and detailed layer-wise analysis metrics.

## Common Log Entry Structure

### Base Fields (All Entries)

| Field | Type | Description |
|-------|------|-------------|
| `phase` | string | Training phase: `"train"` or `"val"` |
| `epoch` | integer | Current epoch number (1-indexed) |
| `iter` | integer | Current iteration within epoch (1-indexed for train, 0 for validation) |
| `step` | integer | Global step counter (cumulative across epochs, train only) |
| `timestamp` | string | ISO format timestamp: `"YYYY-MM-DDTHH:MM:SS.ffffff"` |

### Training-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `loss` | float | Training loss value |
| `acc` | float | Training accuracy (final layer for Greedy, overall for Baseline) |

### Validation-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `acc` | float | Validation accuracy (final layer for Greedy, overall for Baseline) |

## Metrics Structure

### When Metrics Are Logged

**Both Trainers:**
- Training metrics: Logged based on `metrics_frequency` configuration
  - `"iteration"`: Every N iterations (controlled by `metrics_log_frequency`)
  - `"epoch"`: Only at the last iteration of each epoch
- Validation metrics: Always logged during validation
- **Note**: Validation entries may contain both `metrics` (validation) and `train_metrics` (training from end of epoch) when using `metrics_frequency: epoch`

### Metrics Field Structure

```json
{
  "metrics": {
    "layer_0": { /* metrics for layer 0 */ },
    "layer_1": { /* metrics for layer 1 */ },
    "layer_2": { /* metrics for layer 2 */ },
    "layer_3": { /* metrics for layer 3 */ }
  },
  "train_metrics": { /* same structure as metrics, but for training data */ }
}
```

## Layer Metrics Specification

Each layer contains the following metrics (when available):

### Core Classification Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `accuracy` | float | [0, 1] | Classification accuracy using anchor-based logits |
| `cross_entropy` | float | [0, ∞) | Cross-entropy loss between ground truth and predicted probabilities |
| `probe_accuracy` | float | [0, 1] | Accuracy of one-shot linear probe trained on embeddings |
| `f1_score` | float | [0, 1] | F1 score of the linear probe |

### Cosine Alignment Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `alignment` | float | [-1, 1] | Cosine similarity between embeddings and their correct class anchors |
| `margin` | float | [-∞, ∞) | Difference between correct and best incorrect anchor similarities |

### Information-Theoretic Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `mutual_information` | float | [0, ∞) | Mutual information between input and layer embeddings |
| `gaussian_entropy` | float | [0, ∞) | Gaussian entropy estimate of embedding distribution |

### Regularization Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `ace_regularizer` | float | [0, ∞) | ACE (Amended Cross-Entropy) regularizer term |

### Layer Comparison Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `linear_cka` | float | [0, 1] | Linear Centered Kernel Alignment with previous layer |

### Optional Slow Metrics (when `enable_slow_metrics: true`)

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `participation_ratio` | float | [0, ∞) | Participation ratio of embedding dimensions |
| `ridge_r2` | float | [-∞, 1] | R² score of ridge regression predicting next layer |

## Special Values

- `NaN`: Indicates computation failed or metric not applicable (e.g., `linear_cka` for first layer)
- `0.0`: Common for `gaussian_entropy` in early layers when entropy estimator not initialized

## Example Log Entries

### Baseline Trainer Training Entry (with metrics)
```json
{
  "phase": "train",
  "epoch": 1,
  "iter": 352,
  "step": 352,
  "loss": 3.775146484375,
  "acc": 0.115,
  "timestamp": "2025-10-04T14:07:14.504762",
  "metrics": {
    "layer_0": {
      "accuracy": 0.0,
      "cross_entropy": 4.60546875,
      "alignment": -0.0009142789640463889,
      "margin": -0.03752606734633446,
      "ace_regularizer": 4.60546875,
      "mutual_information": 2.6177624496457126,
      "gaussian_entropy": 3292.0,
      "probe_accuracy": 1.0,
      "f1_score": 1.0
    },
    "layer_1": {
      "accuracy": 0.0,
      "cross_entropy": 4.609375,
      "alignment": -0.002730128588154912,
      "margin": -0.04349743574857712,
      "ace_regularizer": 4.60546875,
      "mutual_information": 2.63580644723939,
      "gaussian_entropy": 0.0,
      "probe_accuracy": 1.0,
      "f1_score": 1.0,
      "linear_cka": NaN
    }
  }
}
```

### Baseline Trainer Validation Entry
```json
{
  "phase": "val",
  "epoch": 1,
  "iter": 0,
  "acc": 0.1142,
  "timestamp": "2025-10-04T14:08:39.625838",
  "metrics": {
    "layer_0": {
      "accuracy": 0.009765625,
      "cross_entropy": 4.605340921878815,
      "alignment": -0.0010244813387544128,
      "margin": -0.03804324548691511,
      "ace_regularizer": 4.605212724208831,
      "mutual_information": 2.168676972828517,
      "gaussian_entropy": 3450.811395263672,
      "probe_accuracy": 1.0,
      "f1_score": 1.0
    }
  },
  "train_metrics": {
    "layer_0": {
      "accuracy": 0.0,
      "cross_entropy": 4.60546875,
      "alignment": -0.0009142789640463889,
      "margin": -0.03752606734633446,
      "ace_regularizer": 4.60546875,
      "mutual_information": 2.6177624496457126,
      "gaussian_entropy": 3292.0,
      "probe_accuracy": 1.0,
      "f1_score": 1.0
    }
  }
}
```

### Greedy Trainer Training Entry
```json
{
  "phase": "train",
  "epoch": 1,
  "iter": 352,
  "step": 352,
  "loss": 3.7408908009529114,
  "acc": 0.125,
  "timestamp": "2025-10-04T14:06:36.155810"
}
```

### Greedy Trainer Validation Entry
```json
{
  "phase": "val",
  "epoch": 1,
  "iter": 0,
  "acc": 0.0882,
  "timestamp": "2025-10-04T14:07:20.504782",
  "metrics": {
    "layer_0": {
      "accuracy": 0.1076171875,
      "cross_entropy": 4.547021448612213,
      "alignment": 0.059319633059203625,
      "margin": -0.07621373739093543,
      "ace_regularizer": 4.484238481521606,
      "mutual_information": 2.1761252787736933,
      "gaussian_entropy": 2400.8355102539062,
      "probe_accuracy": 1.0,
      "f1_score": 1.0
    }
  },
  "train_metrics": {
    "layer_0": {
      "accuracy": 0.125,
      "cross_entropy": 4.539011478424072,
      "alignment": 0.06724878400564194,
      "margin": -0.06925731897354126,
      "ace_regularizer": 4.48207426071167,
      "mutual_information": 2.261972289201451,
      "gaussian_entropy": 2144.520263671875,
      "probe_accuracy": 1.0,
      "f1_score": 1.0
    }
  }
}
```

## Key Consistency Features

1. **Global Step Counter**: Both trainers include `step` field in training logs
2. **Accuracy Field**: Both trainers use `acc` field consistently (represents final layer accuracy for Greedy, overall accuracy for Baseline)
3. **Validation Logs**: Neither trainer includes `loss` field in validation entries
4. **Metrics Structure**: Both trainers use identical metrics structure with per-layer analysis
5. **Metrics Duplication**: Both trainers may include both `metrics` (validation) and `train_metrics` (training from end of epoch) in validation entries when using `metrics_frequency: epoch`

## Configuration Impact

The `metrics_frequency` configuration affects when metrics are collected:
- `"epoch"`: Metrics collected only at end of each epoch
- `"iteration"`: Metrics collected every N iterations (controlled by `metrics_log_frequency`)

The `enable_slow_metrics` configuration controls inclusion of computationally expensive metrics:
- `false` (default): Excludes `participation_ratio` and `ridge_r2`
- `true`: Includes all metrics (significantly slower)

## Notes for Analysis

1. **Consistent Structure**: Both trainers now use the same log structure for easy comparison
2. **Step Counter**: The `step` field appears only in training logs (not validation) and represents the global step counter across all epochs
3. **Accuracy Field**: The `acc` field represents the final/overall accuracy - final layer for Greedy, overall model for Baseline
4. **Metrics Separation**: Validation entries contain both `metrics` (validation) and `train_metrics` (training from end of epoch) when using `metrics_frequency: epoch` - use the appropriate field for your analysis
5. **NaN Values**: Common in `linear_cka` for first layer (no previous layer to compare)
6. **Zero Entropy**: `gaussian_entropy` often 0.0 in early epochs before entropy estimators are properly initialized
7. **Probe Metrics**: `probe_accuracy` and `f1_score` often 1.0 due to overfitting on small batches
8. **Layer Indexing**: Layers are 0-indexed (`layer_0`, `layer_1`, etc.)
