#!/usr/bin/env python3
"""
Enhanced plotting script for comprehensive metrics visualization.

This script handles the rich metrics data collected during training,
providing detailed visualizations for all implemented metrics.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Any
from collections import defaultdict
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MetricsVisualizer:
    """Comprehensive metrics visualization for training runs."""
    
    def __init__(self, run_dir: str, show_plots: bool = False):
        self.run_dir = run_dir
        self.show_plots = show_plots
        self.log_path = os.path.join(run_dir, 'log.jsonl')
        
        # Data storage
        self.train_data = defaultdict(list)
        self.val_data = defaultdict(list)
        self.metrics_data = defaultdict(lambda: defaultdict(list))
        
        # Load and parse data
        self._load_data()
    
    def _load_data(self):
        """Load and parse log data."""
        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"No log.jsonl found in {self.run_dir}")
        
        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self._parse_record(rec)
                except json.JSONDecodeError:
                    continue
    
    def _parse_record(self, rec: Dict[str, Any]):
        """Parse a single log record."""
        phase = rec.get('phase')
        epoch = rec.get('epoch')
        step = rec.get('step', rec.get('iter', 0))
        
        if phase == 'train':
            self.train_data['steps'].append(step)
            self.train_data['epochs'].append(epoch)
            
            # Basic metrics
            if 'loss' in rec and rec['loss'] is not None:
                self.train_data['loss'].append(rec['loss'])
            if 'acc_last' in rec and rec['acc_last'] is not None:
                self.train_data['acc_last'].append(rec['acc_last'])
            
            # Rich metrics data
            if 'metrics' in rec and rec['metrics'] is not None:
                self._parse_metrics_data(rec['metrics'], step, epoch, 'train')
        
        elif phase == 'val':
            self.val_data['epochs'].append(epoch)
            
            # Basic metrics
            if 'acc_last' in rec and rec['acc_last'] is not None:
                self.val_data['acc_last'].append(rec['acc_last'])
            elif 'acc' in rec and rec['acc'] is not None:
                self.val_data['acc_last'].append(rec['acc'])
            
            # Rich metrics data
            if 'metrics' in rec and rec['metrics'] is not None:
                self._parse_metrics_data(rec['metrics'], step, epoch, 'val')
    
    def _parse_metrics_data(self, metrics: Dict[str, Any], step: int, epoch: int, phase: str):
        """Parse rich metrics data."""
        for layer_key, layer_metrics in metrics.items():
            for metric_name, metric_value in layer_metrics.items():
                self.metrics_data[phase][f"{layer_key}_{metric_name}"].append({
                    'step': step,
                    'epoch': epoch,
                    'value': metric_value
                })
    
    def plot_basic_curves(self):
        """Plot basic training curves (loss, accuracy)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Training loss
        if self.train_data['steps'] and self.train_data['loss']:
            axes[0].plot(self.train_data['steps'][:len(self.train_data['loss'])], 
                        self.train_data['loss'], alpha=0.7)
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Training Loss')
            axes[0].set_title('Training Loss')
            axes[0].grid(True, alpha=0.3)
        
        # Training accuracy
        if self.train_data['steps'] and self.train_data['acc_last']:
            axes[1].plot(self.train_data['steps'][:len(self.train_data['acc_last'])], 
                        self.train_data['acc_last'], alpha=0.7)
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Training Accuracy')
            axes[1].set_title('Training Accuracy (Last Layer)')
            axes[1].grid(True, alpha=0.3)
        
        # Validation accuracy
        if self.val_data['epochs'] and self.val_data['acc_last']:
            axes[2].plot(self.val_data['epochs'][:len(self.val_data['acc_last'])], 
                        self.val_data['acc_last'], marker='o', alpha=0.7)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Validation Accuracy')
            axes[2].set_title('Validation Accuracy (Last Layer)')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('basic_curves.png')
    
    def plot_layerwise_metrics(self, metric_name: str, phase: str = 'train'):
        """Plot a specific metric across all layers."""
        if phase not in self.metrics_data:
            print(f"No {phase} metrics data found")
            return
        
        # Find all layers for this metric
        layer_metrics = {}
        for key, data in self.metrics_data[phase].items():
            if key.endswith(f'_{metric_name}'):
                layer_name = key.replace(f'_{metric_name}', '')
                layer_metrics[layer_name] = data
        
        if not layer_metrics:
            print(f"No data found for metric: {metric_name}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot each layer
        for layer_name, data in sorted(layer_metrics.items()):
            if data:
                steps = [d['step'] for d in data]
                values = [d['value'] for d in data]
                plt.plot(steps, values, label=layer_name, marker='o', markersize=3, alpha=0.7)
        
        plt.xlabel('Step')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} Across Layers ({phase.title()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        self._save_plot(f'layerwise_{metric_name}_{phase}.png')
    
    def plot_metrics_heatmap(self, phase: str = 'train'):
        """Create a heatmap of metrics across layers."""
        if phase not in self.metrics_data:
            print(f"No {phase} metrics data found")
            return
        
        # Collect all metrics and layers
        metrics_names = set()
        layer_names = set()
        
        for key, data in self.metrics_data[phase].items():
            if data:
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    layer_name, metric_name = parts
                    metrics_names.add(metric_name)
                    layer_names.add(layer_name)
        
        if not metrics_names or not layer_names:
            print(f"No metrics data found for {phase}")
            return
        
        # Create matrix
        metrics_names = sorted(metrics_names)
        layer_names = sorted(layer_names)
        
        # Get latest values for each metric-layer combination
        matrix = np.zeros((len(metrics_names), len(layer_names)))
        
        for i, metric_name in enumerate(metrics_names):
            for j, layer_name in enumerate(layer_names):
                key = f"{layer_name}_{metric_name}"
                if key in self.metrics_data[phase] and self.metrics_data[phase][key]:
                    # Use the latest value
                    latest_value = self.metrics_data[phase][key][-1]['value']
                    matrix[i, j] = latest_value
        
        # Create heatmap
        plt.figure(figsize=(max(8, len(layer_names) * 1.5), max(6, len(metrics_names) * 0.8)))
        
        sns.heatmap(matrix, 
                   xticklabels=layer_names, 
                   yticklabels=metrics_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   cbar_kws={'label': 'Metric Value'})
        
        plt.title(f'Metrics Heatmap - {phase.title()} Phase')
        plt.xlabel('Layer')
        plt.ylabel('Metric')
        plt.tight_layout()
        
        self._save_plot(f'metrics_heatmap_{phase}.png')
    
    def plot_metric_evolution(self, metric_name: str, layer_name: str = None, phase: str = 'train'):
        """Plot the evolution of a specific metric over time."""
        if phase not in self.metrics_data:
            print(f"No {phase} metrics data found")
            return
        
        plt.figure(figsize=(12, 6))
        
        if layer_name:
            # Plot specific layer
            key = f"{layer_name}_{metric_name}"
            if key in self.metrics_data[phase] and self.metrics_data[phase][key]:
                data = self.metrics_data[phase][key]
                steps = [d['step'] for d in data]
                values = [d['value'] for d in data]
                plt.plot(steps, values, marker='o', markersize=3, alpha=0.7)
                plt.title(f'{metric_name.replace("_", " ").title()} - {layer_name} ({phase.title()})')
        else:
            # Plot all layers
            for key, data in self.metrics_data[phase].items():
                if key.endswith(f'_{metric_name}') and data:
                    layer = key.replace(f'_{metric_name}', '')
                    steps = [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    plt.plot(steps, values, label=layer, marker='o', markersize=3, alpha=0.7)
            plt.legend()
            plt.title(f'{metric_name.replace("_", " ").title()} Evolution ({phase.title()})')
        
        plt.xlabel('Step')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        
        filename = f'evolution_{metric_name}_{layer_name or "all"}_{phase}.png'
        self._save_plot(filename)
    
    def plot_research_questions(self):
        """Create plots specifically addressing the research questions."""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        # Question 1: Are later layers getting more predictive?
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_predictiveness(ax1)
        
        # Question 2: Are layers compressing?
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_compression(ax2)
        
        # Question 3: Are layers genuinely different?
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_layer_diversity(ax3)
        
        # Additional insights
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_layer_comparison(ax4)
        
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_metrics_summary(ax5)
        
        plt.suptitle('Research Questions Analysis', fontsize=16, y=0.98)
        self._save_plot('research_questions.png')
    
    def _plot_predictiveness(self, ax):
        """Plot metrics related to layer predictiveness."""
        # Use accuracy and one-shot probe accuracy
        metrics_to_plot = ['accuracy', 'f1_score']
        
        for metric in metrics_to_plot:
            for key, data in self.metrics_data['train'].items():
                if key.endswith(f'_{metric}') and data:
                    layer = key.replace(f'_{metric}', '')
                    steps = [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    ax.plot(steps, values, label=f'{layer} ({metric})', alpha=0.7)
        
        ax.set_title('Layer Predictiveness')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_compression(self, ax):
        """Plot metrics related to layer compression."""
        # Use participation ratio and gaussian entropy
        metrics_to_plot = ['participation_ratio', 'gaussian_entropy']
        
        for metric in metrics_to_plot:
            for key, data in self.metrics_data['train'].items():
                if key.endswith(f'_{metric}') and data:
                    layer = key.replace(f'_{metric}', '')
                    steps = [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    ax.plot(steps, values, label=f'{layer} ({metric})', alpha=0.7)
        
        ax.set_title('Layer Compression')
        ax.set_xlabel('Step')
        ax.set_ylabel('Compression Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_layer_diversity(self, ax):
        """Plot metrics related to layer diversity."""
        # Use linear CKA and ridge R2
        metrics_to_plot = ['linear_cka', 'ridge_r2']
        
        for metric in metrics_to_plot:
            for key, data in self.metrics_data['train'].items():
                if key.endswith(f'_{metric}') and data:
                    layer = key.replace(f'_{metric}', '')
                    steps = [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    ax.plot(steps, values, label=f'{layer} ({metric})', alpha=0.7)
        
        ax.set_title('Layer Diversity')
        ax.set_xlabel('Step')
        ax.set_ylabel('Diversity Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_layer_comparison(self, ax):
        """Plot comparison of key metrics across layers."""
        # Select key metrics for comparison
        key_metrics = ['accuracy', 'participation_ratio', 'alignment', 'margin']
        
        # Get latest values for each layer
        layer_data = defaultdict(dict)
        for metric in key_metrics:
            for key, data in self.metrics_data['train'].items():
                if key.endswith(f'_{metric}') and data:
                    layer = key.replace(f'_{metric}', '')
                    latest_value = data[-1]['value']
                    layer_data[layer][metric] = latest_value
        
        # Create bar plot
        layers = sorted(layer_data.keys())
        x = np.arange(len(layers))
        width = 0.2
        
        for i, metric in enumerate(key_metrics):
            values = [layer_data[layer].get(metric, 0) for layer in layers]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Metric Value')
        ax.set_title('Key Metrics Comparison Across Layers')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_summary(self, ax):
        """Plot summary of all metrics."""
        # Count available metrics
        metric_counts = defaultdict(int)
        for key, data in self.metrics_data['train'].items():
            if data:
                metric_name = key.split('_')[-1]
                metric_counts[metric_name] += 1
        
        # Create bar plot
        metrics = list(metric_counts.keys())
        counts = list(metric_counts.values())
        
        bars = ax.bar(metrics, counts)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Number of Layers')
        ax.set_title('Available Metrics Summary')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
    
    def _save_plot(self, filename: str):
        """Save plot to file."""
        output_path = os.path.join(self.run_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        if self.show_plots:
            plt.show()
        plt.close()
        print(f"Saved {output_path}")
    
    def generate_all_plots(self):
        """Generate all available plots."""
        print("Generating comprehensive metrics visualizations...")
        
        # Basic curves
        self.plot_basic_curves()
        
        # Research questions analysis
        if any(self.metrics_data.values()):
            self.plot_research_questions()
            
            # Layerwise metrics for key metrics
            key_metrics = ['accuracy', 'participation_ratio', 'alignment', 'margin', 
                          'gaussian_entropy', 'linear_cka', 'ridge_r2']
            
            for metric in key_metrics:
                self.plot_layerwise_metrics(metric, 'train')
                if 'val' in self.metrics_data:
                    self.plot_layerwise_metrics(metric, 'val')
            
            # Metrics heatmaps
            self.plot_metrics_heatmap('train')
            if 'val' in self.metrics_data:
                self.plot_metrics_heatmap('val')
            
            # Evolution plots for key metrics
            for metric in ['accuracy', 'participation_ratio', 'alignment']:
                self.plot_metric_evolution(metric, phase='train')
        
        print("All plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(description='Enhanced metrics visualization')
    parser.add_argument('--run_dir', type=str, required=True, help='Path to run directory')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving only')
    parser.add_argument('--metric', type=str, help='Plot specific metric only')
    parser.add_argument('--layer', type=str, help='Plot specific layer only')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'val'], 
                       help='Phase to plot (train or val)')
    parser.add_argument('--research_only', action='store_true', 
                       help='Generate only research questions plots')
    
    args = parser.parse_args()
    
    try:
        visualizer = MetricsVisualizer(args.run_dir, args.show)
        
        if args.research_only:
            visualizer.plot_research_questions()
        elif args.metric:
            if args.layer:
                visualizer.plot_metric_evolution(args.metric, args.layer, args.phase)
            else:
                visualizer.plot_layerwise_metrics(args.metric, args.phase)
        else:
            visualizer.generate_all_plots()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()