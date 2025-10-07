#!/usr/bin/env python3
"""
Comprehensive metrics plotting script for training analysis.

This script provides complete visualizations for all recorded metrics including:
- Basic training curves (loss, accuracy)
- Confusion matrices
- Per-layer metric analysis
- Validation vs training comparisons
- Metric correlation analysis
- Architecture comparisons
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings
import yaml
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AdvancedMetricsVisualizer:
    """Advanced metrics visualization with multi-run comparison capabilities."""
    
    def __init__(self, run_dirs: List[str], show_plots: bool = False):
        self.run_dirs = run_dirs
        self.show_plots = show_plots
        self.is_multi_run = len(run_dirs) > 1
        self.multi_run_data = {}
        self.comparison_dir = None
        
        # Load data from all runs
        for run_dir in run_dirs:
            self._load_run_data(run_dir)
        
        # Initialize color scheme
        self._init_color_scheme()
        
        # Setup comparison directory for multi-run plots
        if self.is_multi_run:
            self._setup_comparison_directory()
    
    def _load_run_data(self, run_dir: str):
        """Load data from a single run."""
        log_path = os.path.join(run_dir, 'log.jsonl')
        if not os.path.exists(log_path):
            print(f"Warning: No log.jsonl found in {run_dir}")
            return
        
        run_name = os.path.basename(run_dir)
        self.multi_run_data[run_name] = {
            'train_data': defaultdict(list),
            'val_data': defaultdict(list),
            'metrics_data': defaultdict(lambda: defaultdict(list))
        }
        
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self._parse_record(rec, run_name)
                except json.JSONDecodeError:
                    continue
    
    def _parse_record(self, rec: Dict[str, Any], run_name: str):
        """Parse a single log record for a specific run."""
        phase = rec.get('phase')
        epoch = rec.get('epoch')
        step = rec.get('step', rec.get('iter', 0))
        
        run_data = self.multi_run_data[run_name]
        
        if phase == 'train':
            run_data['train_data']['steps'].append(step)
            run_data['train_data']['epochs'].append(epoch)
            
            if 'loss' in rec and rec['loss'] is not None:
                run_data['train_data']['loss'].append(rec['loss'])
            # Standardized to 'acc' field for both trainers
            if 'acc' in rec and rec['acc'] is not None:
                run_data['train_data']['acc_last'].append(rec['acc'])
            
            if 'metrics' in rec and rec['metrics'] is not None:
                self._parse_metrics_data(rec['metrics'], step, epoch, 'train', run_name)
        
        elif phase == 'val':
            run_data['val_data']['epochs'].append(epoch)
            
            # Standardized to 'acc' field for both trainers
            if 'acc' in rec and rec['acc'] is not None:
                run_data['val_data']['acc_last'].append(rec['acc'])
            
            if 'metrics' in rec and rec['metrics'] is not None:
                self._parse_metrics_data(rec['metrics'], step, epoch, 'val', run_name)
    
    def _parse_metrics_data(self, metrics: Dict[str, Any], step: int, epoch: int, phase: str, run_name: str):
        """Parse rich metrics data for a specific run."""
        run_data = self.multi_run_data[run_name]
        for layer_key, layer_metrics in metrics.items():
            for metric_name, metric_value in layer_metrics.items():
                run_data['metrics_data'][phase][f"{layer_key}_{metric_name}"].append({
                    'step': step,
                    'epoch': epoch,
                    'value': metric_value
                })
    
    def _init_color_scheme(self):
        """Initialize color scheme: one base color per run, gradients for layers."""
        # Base colors for different runs (using distinct colors)
        base_colors = [
            '#e74c3c',  # Red
            '#3498db',  # Blue
            '#2ecc71',  # Green
            '#f39c12',  # Orange
            '#9b59b6',  # Purple
            '#1abc9c',  # Turquoise
            '#e67e22',  # Carrot
            '#34495e',  # Dark Blue-Gray
        ]
        
        self.run_colors = {}
        run_names = list(self.multi_run_data.keys())
        
        for i, run_name in enumerate(run_names):
            base_color = base_colors[i % len(base_colors)]
            self.run_colors[run_name] = base_color
    
    def _get_layer_color(self, run_name: str, layer_index: int, total_layers: int = 4) -> str:
        """
        Get color for a specific layer within a run using a gradient.
        
        Args:
            run_name: Name of the run
            layer_index: Index of the layer (0, 1, 2, 3)
            total_layers: Total number of layers
        
        Returns:
            Hex color string
        """
        base_color = self.run_colors.get(run_name, '#000000')
        
        # Convert hex to RGB
        base_color = base_color.lstrip('#')
        r, g, b = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Create gradient: lighter for layer 0, darker for layer 3
        # Interpolate between lighter (add white) and the base color
        if total_layers > 1:
            # Factor ranges from 0.3 (lightest) to 1.0 (darkest/base color)
            factor = 0.3 + (0.7 * layer_index / (total_layers - 1))
        else:
            factor = 1.0
        
        # Interpolate towards white for lighter colors
        white_blend = 1.0 - factor
        r = int(r * factor + 255 * white_blend)
        g = int(g * factor + 255 * white_blend)
        b = int(b * factor + 255 * white_blend)
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _setup_comparison_directory(self):
        """Setup a comparison directory for multi-run plots with incremental naming."""
        # Base directory for comparisons
        base_plots_dir = Path('plots')
        base_plots_dir.mkdir(exist_ok=True)
        
        # Find next available comparison number
        existing_comparisons = list(base_plots_dir.glob('comparison_*'))
        if existing_comparisons:
            # Extract numbers from existing comparison folders
            numbers = []
            for comp_dir in existing_comparisons:
                try:
                    num = int(comp_dir.name.split('_')[1])
                    numbers.append(num)
                except (ValueError, IndexError):
                    pass
            next_num = max(numbers, default=0) + 1 if numbers else 1
        else:
            next_num = 1
        
        # Create comparison directory with timestamp and incremental number
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.comparison_dir = base_plots_dir / f'comparison_{next_num:03d}_{timestamp}'
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        self._create_metadata_file()
        
        print(f"Multi-run plots will be saved to: {self.comparison_dir}")
    
    def _create_metadata_file(self):
        """Create metadata file with run information and configs."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'num_runs': len(self.run_dirs),
            'runs': []
        }
        
        for run_dir in self.run_dirs:
            run_info = {
                'run_directory': run_dir,
                'run_name': os.path.basename(run_dir)
            }
            
            # Try to load params.yaml if it exists
            params_file = os.path.join(run_dir, 'params.yaml')
            if os.path.exists(params_file):
                try:
                    with open(params_file, 'r') as f:
                        run_info['config'] = yaml.safe_load(f)
                except Exception as e:
                    run_info['config_error'] = str(e)
            else:
                run_info['config'] = 'params.yaml not found'
            
            metadata['runs'].append(run_info)
        
        # Save metadata as YAML
        metadata_file = self.comparison_dir / 'metadata.yaml'
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        
        print(f"Metadata saved to: {metadata_file}")
    
    def plot_multi_run_comparison(self, metric_name: str, layer_name: str = None, phase: str = 'val'):
        """Compare a specific metric across multiple runs."""
        plt.figure(figsize=(15, 8))
        
        # Check if we have data in the requested phase, fallback to 'val' if not
        available_phases = []
        for run_name, run_data in self.multi_run_data.items():
            for p in ['train', 'val']:
                if p in run_data['metrics_data'] and any(run_data['metrics_data'][p].values()):
                    available_phases.append(p)
        
        if not available_phases:
            print(f"No metrics data found in any phase")
            return
            
        # Use the requested phase if available, otherwise use the first available phase
        actual_phase = phase if phase in available_phases else available_phases[0]
        if actual_phase != phase:
            print(f"No data in {phase} phase, using {actual_phase} phase instead")
        
        for run_name, run_data in self.multi_run_data.items():
            if actual_phase not in run_data['metrics_data']:
                continue
            
            if layer_name:
                # Plot specific layer
                key = f"{layer_name}_{metric_name}"
                if key in run_data['metrics_data'][actual_phase] and run_data['metrics_data'][actual_phase][key]:
                    data = run_data['metrics_data'][actual_phase][key]
                    # Use epochs for validation data, steps for training data
                    x_values = [d['epoch'] for d in data] if actual_phase == 'val' else [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    # Extract layer index for color
                    try:
                        layer_idx = int(layer_name.split('_')[-1]) if layer_name.startswith('layer_') else 0
                    except ValueError:
                        layer_idx = 0
                    color = self._get_layer_color(run_name, layer_idx)
                    plt.plot(x_values, values, label=f'{run_name} ({layer_name})', 
                           marker='o', markersize=4, alpha=0.85, linewidth=2, color=color)
            else:
                # Plot all layers for this run - collect layers first to determine total
                layers_data = []
                for key, data in run_data['metrics_data'][actual_phase].items():
                    if key.endswith(f'_{metric_name}') and data:
                        layer = key.replace(f'_{metric_name}', '')
                        layers_data.append((layer, data))
                
                # Sort layers to ensure consistent ordering
                layers_data.sort(key=lambda x: x[0])
                total_layers = len(layers_data)
                
                for idx, (layer, data) in enumerate(layers_data):
                    # Use epochs for validation data, steps for training data
                    x_values = [d['epoch'] for d in data] if actual_phase == 'val' else [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    # Extract layer index for color gradient
                    try:
                        layer_idx = int(layer.split('_')[-1]) if layer.startswith('layer_') else idx
                    except ValueError:
                        layer_idx = idx
                    color = self._get_layer_color(run_name, layer_idx, total_layers)
                    plt.plot(x_values, values, label=f'{run_name} ({layer})', 
                           marker='o', markersize=4, alpha=0.85, linewidth=2, color=color)
        
        x_label = 'Epoch' if actual_phase == 'val' else 'Step'
        plt.xlabel(x_label)
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} Comparison Across Runs ({actual_phase.title()} Phase)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Apply log scale for gaussian_entropy
        # if 'gaussian_entropy' in metric_name.lower():
        #     plt.yscale('log')
        
        plt.tight_layout()
        
        filename = f'multi_run_{metric_name}_{layer_name or "all"}_{actual_phase}.png'
        subdir = self._get_metric_subdir(metric_name)
        self._save_plot(filename, subdir=subdir, is_multi_run=True)
    
    def plot_prediction_accuracy_comparison(self, phase: str = 'val'):
        """Plot prediction accuracy (non-probe accuracy) across all layers."""
        plt.figure(figsize=(15, 8))
        
        # Check if we have data in the requested phase
        available_phases = []
        for run_data in self.multi_run_data.values():
            for p in ['train', 'val']:
                if p in run_data['metrics_data'] and any(run_data['metrics_data'][p].values()):
                    available_phases.append(p)
        
        if not available_phases:
            print(f"No metrics data found in any phase")
            return
            
        actual_phase = phase if phase in available_phases else available_phases[0]
        if actual_phase != phase:
            print(f"No data in {phase} phase, using {actual_phase} phase instead")
        
        for run_name, run_data in self.multi_run_data.items():
            if actual_phase not in run_data['metrics_data']:
                continue
            
            # Plot all layers' prediction accuracy - collect layers first to determine total
            layers_data = []
            for key, data in run_data['metrics_data'][actual_phase].items():
                # Only plot 'accuracy' metric, not 'probe_accuracy'
                if key.endswith('_accuracy') and 'probe' not in key and data:
                    layer = key.replace('_accuracy', '')
                    layers_data.append((layer, data))
            
            # Sort layers to ensure consistent ordering
            layers_data.sort(key=lambda x: x[0])
            total_layers = len(layers_data)
            
            for idx, (layer, data) in enumerate(layers_data):
                x_values = [d['epoch'] for d in data] if actual_phase == 'val' else [d['step'] for d in data]
                values = [d['value'] for d in data]
                # Extract layer index for color gradient
                try:
                    layer_idx = int(layer.split('_')[-1]) if layer.startswith('layer_') else idx
                except ValueError:
                    layer_idx = idx
                color = self._get_layer_color(run_name, layer_idx, total_layers)
                plt.plot(x_values, values, label=f'{run_name} ({layer})', 
                       marker='o', markersize=4, alpha=0.85, linewidth=2, color=color)
        
        x_label = 'Epoch' if actual_phase == 'val' else 'Step'
        plt.xlabel(x_label)
        plt.ylabel('Prediction Accuracy')
        plt.title(f'Prediction Accuracy Comparison Across Runs ({actual_phase.title()} Phase)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'multi_run_prediction_accuracy_all_{actual_phase}.png'
        self._save_plot(filename, subdir='accuracy', is_multi_run=True)
    
    def plot_architecture_comparison(self, metric_name: str = 'accuracy'):
        """Compare final metric values between MLP and GMLP as grouped bars."""
        # Group runs by architecture
        mlp_runs = {}
        gmlp_runs = {}
            
        for run_name, run_data in self.multi_run_data.items():
            lower_name = run_name.lower()
            if 'mlp' in lower_name and 'greedy' not in lower_name and 'gmlp' not in lower_name:
                mlp_runs[run_name] = run_data
            elif 'greedy' in lower_name or 'gmlp' in lower_name:
                gmlp_runs[run_name] = run_data
        
        if not mlp_runs or not gmlp_runs:
            print("Need both MLP and GMLP runs for comparison")
            return
        
        # Determine available layers by union across groups (using train phase by default)
        layers_set = set()
        for group_runs in (mlp_runs, gmlp_runs):
            for run_data in group_runs.values():
                for key in run_data['metrics_data'].get('train', {}).keys():
                    if key.endswith(f'_{metric_name}'):
                        layers_set.add(key.replace(f'_{metric_name}', ''))
        if not layers_set:
            print(f"No training data found for metric {metric_name}")
            return
        layers = sorted(list(layers_set))
        
        # Collect final values per layer for each group
        mlp_values = []
        gmlp_values = []
        mlp_stds = []
        gmlp_stds = []
        
        for layer in layers:
            layer_key = f"{layer}_{metric_name}"
            mlp_layer_vals = []
            gmlp_layer_vals = []
            for run_data in mlp_runs.values():
                data = run_data['metrics_data'].get('train', {}).get(layer_key, [])
                if data:
                    mlp_layer_vals.append(data[-1]['value'])
            for run_data in gmlp_runs.values():
                data = run_data['metrics_data'].get('train', {}).get(layer_key, [])
                if data:
                    gmlp_layer_vals.append(data[-1]['value'])
            mlp_values.append(np.nanmean(mlp_layer_vals) if mlp_layer_vals else np.nan)
            gmlp_values.append(np.nanmean(gmlp_layer_vals) if gmlp_layer_vals else np.nan)
            mlp_stds.append(np.nanstd(mlp_layer_vals) if mlp_layer_vals else 0.0)
            gmlp_stds.append(np.nanstd(gmlp_layer_vals) if gmlp_layer_vals else 0.0)
        
        # Plot grouped bars
        x = np.arange(len(layers))
        width = 0.35
        plt.figure(figsize=(max(10, len(layers) * 1.2), 6))
        plt.bar(x - width/2, mlp_values, width, yerr=mlp_stds, label='MLP', alpha=0.8, capsize=4)
        plt.bar(x + width/2, gmlp_values, width, yerr=gmlp_stds, label='GMLP/Greedy', alpha=0.8, capsize=4)
        plt.xticks(x, layers, rotation=0)
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.xlabel('Layer')
        plt.title(f'Architecture Comparison (Final Train {metric_name.replace("_", " ").title()})')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        # Set y-axis limits with margin below minimum value
        all_values = mlp_values + gmlp_values
        finite_values = [v for v in all_values if not np.isnan(v)]
        if finite_values:
            min_val = min(finite_values)
            max_val = max(finite_values)
            range_val = max_val - min_val
            margin = max(range_val * 0.1, abs(min_val) * 0.05) if range_val > 0 else abs(min_val) * 0.1
            plt.ylim(bottom=min_val - margin)
        
        plt.tight_layout()
        subdir = self._get_metric_subdir(metric_name)
        self._save_plot(f'architecture_comparison_{metric_name}.png', subdir=subdir, is_multi_run=True)
    
    def plot_metrics_timeline(self, metric_name: str, layer_name: str = 'layer_0'):
        """Plot metric evolution over time for multiple runs."""
        # Check if we have data in any phase
        available_phases = []
        for run_data in self.multi_run_data.values():
            for p in ['train', 'val']:
                if p in run_data['metrics_data'] and any(run_data['metrics_data'][p].values()):
                    available_phases.append(p)
        
        if not available_phases:
            print(f"No metrics data found in any phase")
            return
            
        # Use validation phase if available, otherwise use the first available phase
        actual_phase = 'val' if 'val' in available_phases else available_phases[0]
        
        plt.figure(figsize=(15, 8))
        
        for run_name, run_data in self.multi_run_data.items():
            if actual_phase not in run_data['metrics_data']:
                continue
            
            key = f"{layer_name}_{metric_name}"
            if key in run_data['metrics_data'][actual_phase] and run_data['metrics_data'][actual_phase][key]:
                data = run_data['metrics_data'][actual_phase][key]
                # Use epochs for validation data, steps for training data
                x_values = [d['epoch'] for d in data] if actual_phase == 'val' else [d['step'] for d in data]
                values = [d['value'] for d in data]
                
                # Get color for this run and layer
                try:
                    layer_idx = int(layer_name.split('_')[-1]) if layer_name.startswith('layer_') else 0
                except ValueError:
                    layer_idx = 0
                color = self._get_layer_color(run_name, layer_idx)
                
                # Smooth the curve
                if len(values) > 10:
                    from scipy.ndimage import gaussian_filter1d
                    smoothed_values = gaussian_filter1d(values, sigma=1)
                    plt.plot(x_values, smoothed_values, label=run_name, alpha=0.85, linewidth=2, color=color)
                else:
                    plt.plot(x_values, values, label=run_name, alpha=0.85, linewidth=2, marker='o', markersize=4, color=color)
        
        x_label = 'Epoch' if actual_phase == 'val' else 'Step'
        plt.xlabel(x_label)
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} Timeline - {layer_name} ({actual_phase.title()} Phase)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Apply log scale for gaussian_entropy
        if 'gaussian_entropy' in metric_name.lower():
            plt.yscale('log')
        
        filename = f'timeline_{metric_name}_{layer_name}_{actual_phase}.png'
        subdir = self._get_metric_subdir(metric_name)
        # Timeline is multi-run if we have multiple runs
        self._save_plot(filename, subdir=subdir, is_multi_run=self.is_multi_run)
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report with all available plots."""
        print("Generating comprehensive metrics analysis report...")
        
        # Check which phase has data
        available_phases = []
        for run_data in self.multi_run_data.values():
            for p in ['train', 'val']:
                if p in run_data['metrics_data'] and any(run_data['metrics_data'][p].values()):
                    available_phases.append(p)
        
        if not available_phases:
            print("No metrics data found in any phase")
            return
            
        # Use validation phase if available, otherwise use the first available phase
        default_phase = 'val' if 'val' in available_phases else available_phases[0]
        print(f"Using {default_phase} phase for metrics analysis")
        
        # Basic training curves
        self.plot_basic_training_curves()
        
        # Multi-run comparisons for key metrics
        key_metrics = ['probe_accuracy', 'cross_entropy', 'alignment', 'margin', 
                      'ace_regularizer', 'mutual_information', 'gaussian_entropy']
        
        # Separate accuracy into probe_accuracy and prediction accuracy
        self.plot_multi_run_comparison('probe_accuracy', phase=default_phase)
        # Plot prediction accuracy (non-probe accuracy) separately
        self.plot_prediction_accuracy_comparison(phase=default_phase)
        
        for metric in key_metrics:
            self.plot_multi_run_comparison(metric, phase=default_phase)
            # Only generate heatmaps when we have multiple runs; otherwise produce a single-run layer profile
            if self.is_multi_run:
                self.plot_per_layer_metrics_heatmap(metric, phase=default_phase)
            else:
                self.plot_single_run_layer_profile(metric, phase=default_phase)
        
        # Architecture comparison (only meaningful with both architectures present)
        self.plot_architecture_comparison('accuracy')
        
        # Training vs validation comparison
        self.plot_train_val_comparison('accuracy')
        
        # All metrics overview
        self.plot_all_metrics_overview(default_phase)
        
        # Timeline analysis for key metrics across all layers (0, 1, 2, 3)
        timeline_metrics = ['accuracy', 'probe_accuracy', 'cross_entropy']
        for metric in timeline_metrics:
            # Generate timelines for all layers
            for layer_idx in range(4):  # layer_0 through layer_3
                self.plot_metrics_timeline(metric, f'layer_{layer_idx}')
        
        print("Comprehensive report generated successfully!")
    
    def plot_basic_training_curves(self):
        """Plot basic training curves: loss and accuracy over epochs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for run_name, run_data in self.multi_run_data.items():
            # Use base color for this run
            color = self.run_colors.get(run_name, '#000000')
            
            # Plot training loss
            if 'loss' in run_data['train_data'] and run_data['train_data']['loss']:
                axes[0].plot(run_data['train_data']['steps'], run_data['train_data']['loss'], 
                           label=f'{run_name} (train)', alpha=0.8, linewidth=2, color=color)
            
            # Plot training accuracy
            if 'acc_last' in run_data['train_data'] and run_data['train_data']['acc_last']:
                axes[1].plot(run_data['train_data']['steps'], run_data['train_data']['acc_last'], 
                           label=f'{run_name} (train)', alpha=0.8, linewidth=2, color=color)
            
            # Plot validation accuracy
            if 'acc_last' in run_data['val_data'] and run_data['val_data']['acc_last']:
                axes[2].plot(run_data['val_data']['epochs'], run_data['val_data']['acc_last'], 
                           label=f'{run_name} (val)', alpha=0.8, linewidth=2, marker='o', color=color)
        
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title('Validation Accuracy')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Hide the 4th subplot
        axes[3].set_visible(False)
        
        plt.suptitle('Basic Training Curves', fontsize=16)
        plt.tight_layout()
        self._save_plot('basic_training_curves.png', is_multi_run=self.is_multi_run)
    
    def plot_confusion_matrix(self, run_dir: str, normalize: bool = False, labels: List[str] = None):
        """Plot confusion matrix for a specific run."""
        cm_path = os.path.join(run_dir, 'confusion.npy')
        if not os.path.exists(cm_path):
            print(f"Confusion matrix not found: {cm_path}. Run eval.py --confusion first.")
            return
        
        cm = np.load(cm_path)
        if normalize:
            cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', square=True,
                   xticklabels=labels if labels else True, 
                   yticklabels=labels if labels else True)
        
        plt.title(f'Confusion Matrix{" (Normalized)" if normalize else ""}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if labels:
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save to plots subfolder
        plots_dir = os.path.join(run_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        output_path = os.path.join(plots_dir, f'confusion_matrix{"_normalized" if normalize else ""}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        if self.show_plots:
            plt.show()
        plt.close()
        print(f"Saved {output_path}")
    
    def plot_per_layer_metrics_heatmap(self, metric_name: str, phase: str = 'val'):
        """Create heatmap showing metric values across layers and runs."""
        # If single run, this heatmap is not informative; plot a layer profile instead
        if not self.is_multi_run:
            print("Single run detected. Generating layer profile instead of heatmap.")
            self.plot_single_run_layer_profile(metric_name, phase)
            return
        # Check if we have data in the requested phase, fallback to 'val' if not
        available_phases = []
        for run_data in self.multi_run_data.values():
            for p in ['train', 'val']:
                if p in run_data['metrics_data'] and any(run_data['metrics_data'][p].values()):
                    available_phases.append(p)
        
        if not available_phases:
            print(f"No metrics data found in any phase")
            return
            
        # Use the requested phase if available, otherwise use the first available phase
        actual_phase = phase if phase in available_phases else available_phases[0]
        if actual_phase != phase:
            print(f"No data in {phase} phase, using {actual_phase} phase instead")
        
        # Collect union of layers across runs
        layer_set = set()
        for run_data in self.multi_run_data.values():
            if actual_phase not in run_data['metrics_data']:
                continue
            for layer_key in run_data['metrics_data'][actual_phase].keys():
                if layer_key.endswith(f'_{metric_name}'):
                    layer_set.add(layer_key.replace(f'_{metric_name}', ''))
        layer_names = sorted(list(layer_set))
        if not layer_names:
            print(f"No data found for metric {metric_name} in phase {actual_phase}")
            return
        
        # Collect data aligned by layers
        data_matrix = []
        run_names = []
        for run_name, run_data in self.multi_run_data.items():
            if actual_phase not in run_data['metrics_data']:
                continue
            row = []
            for layer in layer_names:
                layer_key = f"{layer}_{metric_name}"
                data = run_data['metrics_data'][actual_phase].get(layer_key, [])
                if data:
                    row.append(data[-1]['value'])
                else:
                    row.append(np.nan)
            data_matrix.append(row)
            run_names.append(run_name)
        
        if not data_matrix:
            print(f"No data found for metric {metric_name} in phase {actual_phase}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(data_matrix, index=run_names, columns=layer_names)
        
        # Plot heatmap
        plt.figure(figsize=(max(8, len(layer_names) * 1.5), max(6, len(run_names) * 0.8)))
        
        sns.heatmap(df, annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': metric_name.replace('_', ' ').title()})
        
        plt.title(f'{metric_name.replace("_", " ").title()} Across Layers and Runs ({actual_phase.title()} Phase)')
        plt.xlabel('Layer')
        plt.ylabel('Run')
        plt.tight_layout()
        subdir = self._get_metric_subdir(metric_name)
        self._save_plot(f'heatmap_{metric_name}_{actual_phase}.png', subdir=subdir, is_multi_run=True)

    def plot_single_run_layer_profile(self, metric_name: str, phase: str = 'val'):
        """For a single run, plot latest metric value across layers as a bar chart."""
        # Determine which run to use (the first)
        if not self.multi_run_data:
            print("No run data available")
            return
        run_name = list(self.multi_run_data.keys())[0]
        run_data = self.multi_run_data[run_name]
        if phase not in run_data['metrics_data']:
            # Fallback to whichever is available
            available = [p for p in ['val', 'train'] if p in run_data['metrics_data'] and any(run_data['metrics_data'][p].values())]
            if not available:
                print("No metrics data found for single run")
                return
            phase = available[0]
        layer_values = {}
        for key, data in run_data['metrics_data'][phase].items():
            if key.endswith(f'_{metric_name}') and data:
                layer = key.replace(f'_{metric_name}', '')
                layer_values[layer] = data[-1]['value']
        if not layer_values:
            print(f"No data found for metric {metric_name} in phase {phase}")
            return
        layers = sorted(layer_values.keys())
        values = [layer_values[layer] for layer in layers]
        plt.figure(figsize=(max(10, len(layers) * 1.2), 6))
        plt.bar(layers, values, alpha=0.85)
        plt.xlabel('Layer')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} Across Layers ({phase.title()}) - {run_name}')
        
        # Set y-axis limits with margin below minimum value
        if values:
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val
            margin = max(range_val * 0.1, abs(min_val) * 0.05) if range_val > 0 else abs(min_val) * 0.1
            plt.ylim(bottom=min_val - margin)
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Organize into subdirectory based on metric type
        subdir = self._get_metric_subdir(metric_name)
        self._save_plot(f'layer_profile_{metric_name}_{phase}.png', subdir=subdir, is_multi_run=False)
    
    def plot_train_val_comparison(self, metric_name: str = 'accuracy'):
        """Compare training vs validation metrics."""
        # Check if we have data in both phases
        has_train = any(any(run_data['metrics_data'].get('train', {}).values()) 
                       for run_data in self.multi_run_data.values())
        has_val = any(any(run_data['metrics_data'].get('val', {}).values()) 
                     for run_data in self.multi_run_data.values())
        has_train_acc = any('acc_last' in run_data['train_data'] and run_data['train_data']['acc_last']
                     for run_data in self.multi_run_data.values())
        
        if not has_train and not has_val and not has_train_acc:
            print("No metrics data found in any phase")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        layers = ['layer_0', 'layer_1', 'layer_2', 'layer_3']
        
        for i, layer in enumerate(layers):
            if i >= len(axes):
                break
            
            ax = axes[i]
            try:
                layer_idx = int(layer.split('_')[-1]) if layer.startswith('layer_') else i
            except ValueError:
                layer_idx = i
            
            for run_name, run_data in self.multi_run_data.items():
                # Get color for this run and layer
                color = self._get_layer_color(run_name, layer_idx)
                
                # For accuracy metric, also plot the main training accuracy from train_data
                if metric_name == 'accuracy' and has_train_acc:
                    if 'acc_last' in run_data['train_data'] and run_data['train_data']['acc_last']:
                        train_steps = run_data['train_data']['steps']
                        train_acc = run_data['train_data']['acc_last']
                        ax.plot(train_steps, train_acc, label=f'{run_name} (train main)', 
                               alpha=0.7, linestyle='-', linewidth=2, color=color)
                
                # Per-layer training metrics
                if has_train and 'train' in run_data['metrics_data']:
                    train_key = f"{layer}_{metric_name}"
                    if train_key in run_data['metrics_data']['train'] and run_data['metrics_data']['train'][train_key]:
                        train_data = run_data['metrics_data']['train'][train_key]
                        train_steps = [d['step'] for d in train_data]
                        train_values = [d['value'] for d in train_data]
                        ax.plot(train_steps, train_values, label=f'{run_name} (train {layer})', 
                               alpha=0.8, linestyle='-', marker='o', markersize=2, color=color)
                
                # Validation data - use slightly darker version
                if has_val and 'val' in run_data['metrics_data']:
                    val_key = f"{layer}_{metric_name}"
                    if val_key in run_data['metrics_data']['val'] and run_data['metrics_data']['val'][val_key]:
                        val_data = run_data['metrics_data']['val'][val_key]
                        val_epochs = [d['epoch'] for d in val_data]
                        val_values = [d['value'] for d in val_data]
                        ax.plot(val_epochs, val_values, label=f'{run_name} (val {layer})', 
                               alpha=0.9, linestyle='--', marker='s', markersize=4, color=color)
            
            ax.set_title(f'{layer} - {metric_name.title()}')
            ax.set_xlabel('Step (train) / Epoch (val)')
            ax.set_ylabel(metric_name.title())
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training vs Validation: {metric_name.title()}', fontsize=16)
        plt.tight_layout()
        self._save_plot(f'train_val_comparison_{metric_name}.png', is_multi_run=self.is_multi_run)
    
    def plot_all_metrics_overview(self, phase: str = 'val'):
        """Create comprehensive overview of all metrics."""
        # Check if we have data in the requested phase, fallback to 'val' if not
        available_phases = []
        for run_data in self.multi_run_data.values():
            for p in ['train', 'val']:
                if p in run_data['metrics_data'] and any(run_data['metrics_data'][p].values()):
                    available_phases.append(p)
        
        if not available_phases:
            print(f"No metrics data found in any phase")
            return
            
        # Use the requested phase if available, otherwise use the first available phase
        actual_phase = phase if phase in available_phases else available_phases[0]
        if actual_phase != phase:
            print(f"No data in {phase} phase, using {actual_phase} phase instead")
        
        # Get all available metrics
        all_metrics = set()
        for run_data in self.multi_run_data.values():
            if actual_phase in run_data['metrics_data']:
                for key in run_data['metrics_data'][actual_phase].keys():
                    metric = key.split('_')[-1]
                    all_metrics.add(metric)
        
        all_metrics = sorted(list(all_metrics))
        
        if not all_metrics:
            print(f"No metrics found in {actual_phase} phase")
            return
        
        # Create subplots
        n_metrics = len(all_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, metric in enumerate(all_metrics):
            ax = axes[i]
            
            for run_name, run_data in self.multi_run_data.items():
                if actual_phase not in run_data['metrics_data']:
                    continue
                
                # Collect layers for this metric
                layers_data = []
                for key, data in run_data['metrics_data'][actual_phase].items():
                    if key.endswith(f'_{metric}') and data:
                        layer = key.replace(f'_{metric}', '')
                        layers_data.append((layer, data))
                
                # Sort layers
                layers_data.sort(key=lambda x: x[0])
                total_layers = len(layers_data)
                
                for idx, (layer, data) in enumerate(layers_data):
                    # Use epochs for validation data, steps for training data
                    x_values = [d['epoch'] for d in data] if actual_phase == 'val' else [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    # Extract layer index for color gradient
                    try:
                        layer_idx = int(layer.split('_')[-1]) if layer.startswith('layer_') else idx
                    except ValueError:
                        layer_idx = idx
                    color = self._get_layer_color(run_name, layer_idx, total_layers)
                    ax.plot(x_values, values, label=f'{run_name} ({layer})', 
                           alpha=0.85, linewidth=1.5, marker='o', markersize=2, color=color)
            
            ax.set_title(metric.replace('_', ' ').title())
            x_label = 'Epoch' if actual_phase == 'val' else 'Step'
            ax.set_xlabel(x_label)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(all_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'All Metrics Overview - {actual_phase.title()} Phase', fontsize=16)
        plt.tight_layout()
        self._save_plot(f'all_metrics_overview_{actual_phase}.png', is_multi_run=self.is_multi_run)
    
    def _get_metric_subdir(self, metric_name: str) -> str:
        """Determine subdirectory for organizing plots by metric type."""
        metric_categories = {
            'accuracy': ['accuracy'],
            'probe_accuracy': ['probe_accuracy'],
            'cross_entropy': ['cross_entropy', 'loss'],
            'regularization': ['ace_regularizer', 'margin', 'alignment'],
            'entropy': ['gaussian_entropy', 'mutual_information'],
            'compression': ['participation_ratio'],
            'similarity': ['linear_cka', 'cka'],
            'general': []  # catchall
        }
        
        for category, keywords in metric_categories.items():
            if any(keyword in metric_name.lower() for keyword in keywords):
                return category
        return 'general'
    
    def _save_plot(self, filename: str, subdir: str = None, is_multi_run: bool = None):
        """
        Save plot to appropriate directory.
        
        Args:
            filename: Name of the file to save
            subdir: Subdirectory within plots folder (e.g., 'accuracy', 'entropy')
            is_multi_run: If True, save to comparison directory. If False, save to first run's directory.
                         If None, auto-detect based on self.is_multi_run
        """
        # Auto-detect if not specified
        if is_multi_run is None:
            is_multi_run = self.is_multi_run
        
        if is_multi_run and self.comparison_dir:
            # Save to comparison directory for multi-run plots
            plots_dir = self.comparison_dir
            if subdir:
                plots_dir = plots_dir / subdir
            plots_dir.mkdir(parents=True, exist_ok=True)
            output_path = plots_dir / filename
        elif self.run_dirs:
            # Save to first run's directory for single-run plots
            plots_dir = Path(self.run_dirs[0]) / 'plots'
            if subdir:
                plots_dir = plots_dir / subdir
            plots_dir.mkdir(parents=True, exist_ok=True)
            output_path = plots_dir / filename
        else:
            print("No output directory available")
            return
        
        plt.savefig(str(output_path), bbox_inches='tight', dpi=300)
        if self.show_plots:
            plt.show()
        plt.close()
        print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive metrics visualization')
    parser.add_argument('--run_dirs', nargs='+', required=True, 
                       help='Paths to run directories for comparison')
    parser.add_argument('--show', action='store_true', 
                       help='Show plots instead of saving only')
    parser.add_argument('--metric', type=str, 
                       help='Plot specific metric only')
    parser.add_argument('--layer', type=str, 
                       help='Plot specific layer only')
    parser.add_argument('--phase', type=str, default='val', 
                       choices=['train', 'val'], help='Phase to plot')
    
    # Analysis types
    parser.add_argument('--comprehensive', action='store_true',
                       help='Generate comprehensive analysis report (all plots)')
    parser.add_argument('--basic_curves', action='store_true',
                       help='Plot basic training curves (loss, accuracy)')
    parser.add_argument('--confusion', type=str, metavar='RUN_DIR',
                       help='Plot confusion matrix for specific run')
    parser.add_argument('--heatmap', type=str, metavar='METRIC',
                       help='Create heatmap for specific metric')
    parser.add_argument('--train_val', action='store_true',
                       help='Compare training vs validation metrics')
    parser.add_argument('--overview', action='store_true',
                       help='Show overview of all metrics')
    parser.add_argument('--architecture_comparison', action='store_true',
                       help='Compare MLP vs GMLP architectures')
    parser.add_argument('--timeline', type=str, metavar='METRIC',
                       help='Plot timeline for specific metric')
    
    # Confusion matrix options
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize confusion matrix')
    parser.add_argument('--labels', type=str,
                       help='Comma-separated class labels for confusion matrix')
    
    args = parser.parse_args()
    
    try:
        visualizer = AdvancedMetricsVisualizer(args.run_dirs, args.show)
        
        if args.comprehensive:
            visualizer.generate_comprehensive_report()
        elif args.basic_curves:
            visualizer.plot_basic_training_curves()
        elif args.confusion:
            labels = args.labels.split(',') if args.labels else None
            visualizer.plot_confusion_matrix(args.confusion, args.normalize, labels)
        elif args.heatmap:
            visualizer.plot_per_layer_metrics_heatmap(args.heatmap, args.phase)
        elif args.train_val:
            visualizer.plot_train_val_comparison(args.metric or 'accuracy')
        elif args.overview:
            visualizer.plot_all_metrics_overview(args.phase)
        elif args.architecture_comparison:
            visualizer.plot_architecture_comparison(args.metric or 'accuracy')
        elif args.timeline:
            visualizer.plot_metrics_timeline(args.timeline, args.layer or 'layer_0')
        elif args.metric:
            visualizer.plot_multi_run_comparison(args.metric, args.layer, args.phase)
        else:
            print("Please specify an analysis type. Use --help for available options.")
            print("\nQuick start: --comprehensive (generates all plots)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
