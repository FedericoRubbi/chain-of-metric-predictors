# Plot Analysis and Documentation

This document describes each plotting method in `plot_metrics.py`, what it's supposed to visualize, and the improvements made.

## Plot Organization

All plots are now organized into subdirectories by metric type:
- `accuracy/` - Prediction and probe accuracy plots
- `probe_accuracy/` - Probe-specific accuracy metrics
- `cross_entropy/` - Loss and cross-entropy metrics
- `regularization/` - ACE regularizer, margin, alignment metrics
- `entropy/` - Gaussian entropy and mutual information
- `compression/` - Participation ratio metrics
- `similarity/` - CKA and layer similarity metrics
- `general/` - Miscellaneous plots

## Current Plotting Methods

### 1. `plot_basic_training_curves()`
**Purpose**: Show training and validation loss/accuracy over time
**What it visualizes**: 
- Training loss vs epochs
- Validation loss vs epochs  
- Training accuracy vs epochs
- Validation accuracy vs epochs

**Status**: ✅ **GOOD** - This plot makes sense and provides useful information

### 2. `plot_multi_run_comparison(metric_name, layer_name, phase)`
**Purpose**: Compare a specific metric across multiple runs for a specific layer
**What it visualizes**:
- Line plot showing metric evolution over training steps/epochs
- Multiple runs overlaid for comparison
- X-axis: training steps/epochs
- Y-axis: metric value
- Different colors for different runs
- **Log scale** for gaussian_entropy metrics
- Plots saved to organized subdirectories

**Status**: ✅ **GOOD** - Enhanced with log scale for entropy and subdirectory organization

### 2b. `plot_prediction_accuracy_comparison(phase)`
**Purpose**: Plot prediction accuracy (non-probe) across all layers
**What it visualizes**:
- Separate plot for prediction accuracy only (excludes probe_accuracy)
- All layers overlaid for comparison
- X-axis: epochs (val) or steps (train)
- Y-axis: prediction accuracy

**Status**: ✅ **NEW** - Splits accuracy visualization for clarity

### 3. `plot_architecture_comparison(metric_name)`
**Purpose**: Compare MLP vs GMLP architectures
**What it visualizes**:
- Grouped bar chart of final train metric values per layer
- Separate bars for MLP and GMLP/Greedy groups with error bars (std)
- Shows which architecture performs better per layer
- **Y-axis starts from min value with margin** for better readability

**Status**: ✅ **GOOD** - Redesigned with grouped bars, error bars, and optimized y-axis

### 4. ~~`plot_metrics_correlation_matrix(phase)`~~ **REMOVED**
**Reason**: Correlation matrix was not providing actionable insights and cluttered the output.

**Status**: ❌ **REMOVED** - Method and all references deleted

### 5. ~~`plot_layer_progression_analysis()`~~ **REMOVED**
**Reason**: Boxplot visualization was not as informative as the dedicated layer profile and timeline plots.

**Status**: ❌ **REMOVED** - Method and all references deleted

### 6. `plot_metrics_timeline(metric_name, layer_name)`
**Purpose**: Show metric evolution over training time for a specific layer
**What it visualizes**:
- Line plot with training steps/epochs on X-axis
- Metric value on Y-axis
- Shows training dynamics for specific layer
- **Log scale** for gaussian_entropy metrics
- Plots saved to organized subdirectories
- **Now generated for all layers (0-3)** for key metrics (accuracy, probe_accuracy, cross_entropy)

**Status**: ✅ **ENHANCED** - Added log scale, subdirectories, and intermediate layer support

### 7. `plot_confusion_matrix(run_dir, normalize, labels)`
**Purpose**: Visualize classification performance
**What it visualizes**:
- Heatmap showing true vs predicted classes
- Diagonal = correct predictions
- Off-diagonal = misclassifications

**Status**: ✅ **GOOD** - Standard and useful confusion matrix

### 8. `plot_per_layer_metrics_heatmap(metric_name, phase)`
**Purpose**: Show metric values across layers and runs
**What it visualizes (multi-run)**:
- Heatmap with runs on Y-axis, layers on X-axis (layers aligned across runs)
- Color intensity = latest metric value
- Useful for comparing multiple runs
- Plots saved to organized subdirectories

**Single-run behavior**:
- Automatically falls back to `plot_single_run_layer_profile()` (bar chart across layers)

**Status**: ✅ **GOOD** - Multi-run only; single-run uses proper fallback

### 8b. `plot_single_run_layer_profile(metric_name, phase)`
**Purpose**: For single runs, show metric values across layers as bar chart
**What it visualizes**:
- Bar chart with layers on X-axis
- Latest metric value on Y-axis
- **Y-axis starts from min value with margin** for better readability
- Plots saved to organized subdirectories

**Status**: ✅ **GOOD** - Proper single-run alternative to heatmap with optimized y-axis

### 9. `plot_train_val_comparison(metric_name)`
**Purpose**: Compare training vs validation performance
**What it visualizes**:
- Side-by-side comparison of train vs val metrics
- **Now includes actual training accuracy** from train_data (main model)
- Shows per-layer validation metrics
- Shows overfitting patterns
- Multiple subplots for different layers

**Status**: ✅ **FIXED** - Now correctly plots training accuracy from train_data

### 10. `plot_all_metrics_overview(phase)`
**Purpose**: Show overview of all available metrics
**What it visualizes**:
- Grid of subplots, one per metric
- Each subplot shows metric evolution over training
- Comprehensive view of all metrics

**Status**: ✅ **GOOD** - Useful overview plot

### 11. ~~`plot_metric_distributions(phase)`~~ **REMOVED**
**Reason**: Histogram distributions were not providing significant insights beyond what's visible in other plots.

**Status**: ❌ **REMOVED** - Method and all references deleted

## Issues Identified

## Recent Improvements

### ✅ **All Issues Resolved**

1. **Y-axis optimization**: All bar plots now start from minimum value with margin for better readability
2. **Accuracy plot separation**: Probe accuracy and prediction accuracy now have separate plots
3. **Log scale for entropy**: Gaussian entropy plots now use log scale for clarity
4. **Correlation matrix removed**: Removed unused correlation matrix and all references
5. **Intermediate layer timelines**: Timeline plots now generated for all layers (0-3) for key metrics
6. **Organized subdirectories**: All plots organized into metric-specific subdirectories
7. **Single-run heatmaps fixed**: Replaced by single-run layer profile bar charts
8. **Multi-run detection**: Added via `self.is_multi_run` flag


### 📊 **Plot Categories**

**Always Useful** (regardless of single/multi run):
- `plot_basic_training_curves()`
- `plot_multi_run_comparison()` 
- `plot_metrics_timeline()`
- `plot_confusion_matrix()`
- `plot_train_val_comparison()`
- `plot_all_metrics_overview()`

**Only Useful with Multiple Runs**:
- `plot_per_layer_metrics_heatmap()` ✅ (auto-fallback to single-run profile)
- `plot_architecture_comparison()` (needs both MLP and GMLP runs)

## Research Questions Addressed

1. **Are later layers getting more predictive?**
   - ✅ `plot_multi_run_comparison()` for accuracy metrics
   - ✅ `plot_metrics_timeline()` for layer-specific trends
   - ✅ `plot_single_run_layer_profile()` for layer-wise comparison

2. **Are layers compressing?**
   - ✅ `plot_multi_run_comparison()` for entropy and mutual information
   - ✅ `plot_metrics_timeline()` with log scale for entropy metrics

3. **Are layers genuinely different from previous layers?**
   - ✅ `plot_multi_run_comparison()` for alignment/margin metrics
   - ✅ `plot_train_val_comparison()` for training dynamics

## Summary of Changes

All identified issues have been resolved:
- ✅ Bar plots optimized with proper y-axis limits
- ✅ Accuracy plots separated for clarity
- ✅ Gaussian entropy uses log scale
- ✅ Correlation matrix removed
- ✅ Timeline plots for all layers (0-3)
- ✅ Organized subdirectory structure
- ✅ Multi-run detection implemented
- ✅ Single-run heatmap fallback added

The plotting system is now production-ready with improved readability and organization.
