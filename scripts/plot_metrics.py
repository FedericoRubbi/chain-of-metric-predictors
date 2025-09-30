#!/usr/bin/env python3
"""
Specialized metrics plotting script for advanced analysis.

This script provides specialized visualizations for specific research questions
and metric comparisons across different runs.
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

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AdvancedMetricsVisualizer:
    """Advanced metrics visualization with multi-run comparison capabilities."""
    
    def __init__(self, run_dirs: List[str], show_plots: bool = False):
        self.run_dirs = run_dirs
        self.show_plots = show_plots
        self.multi_run_data = {}
        
        # Load data from all runs
        for run_dir in run_dirs:
            self._load_run_data(run_dir)
    
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
            if 'acc_last' in rec and rec['acc_last'] is not None:
                run_data['train_data']['acc_last'].append(rec['acc_last'])
            
            if 'metrics' in rec and rec['metrics'] is not None:
                self._parse_metrics_data(rec['metrics'], step, epoch, 'train', run_name)
        
        elif phase == 'val':
            run_data['val_data']['epochs'].append(epoch)
            
            if 'acc_last' in rec and rec['acc_last'] is not None:
                run_data['val_data']['acc_last'].append(rec['acc_last'])
            elif 'acc' in rec and rec['acc'] is not None:
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
    
    def plot_multi_run_comparison(self, metric_name: str, layer_name: str = None, phase: str = 'train'):
        """Compare a specific metric across multiple runs."""
        plt.figure(figsize=(15, 8))
        
        for run_name, run_data in self.multi_run_data.items():
            if phase not in run_data['metrics_data']:
                continue
            
            if layer_name:
                # Plot specific layer
                key = f"{layer_name}_{metric_name}"
                if key in run_data['metrics_data'][phase] and run_data['metrics_data'][phase][key]:
                    data = run_data['metrics_data'][phase][key]
                    steps = [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    plt.plot(steps, values, label=f'{run_name} ({layer_name})', 
                           marker='o', markersize=2, alpha=0.7)
            else:
                # Plot all layers for this run
                for key, data in run_data['metrics_data'][phase].items():
                    if key.endswith(f'_{metric_name}') and data:
                        layer = key.replace(f'_{metric_name}', '')
                        steps = [d['step'] for d in data]
                        values = [d['value'] for d in data]
                        plt.plot(steps, values, label=f'{run_name} ({layer})', 
                               marker='o', markersize=2, alpha=0.7)
        
        plt.xlabel('Step')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} Comparison Across Runs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'multi_run_{metric_name}_{layer_name or "all"}_{phase}.png'
        self._save_plot(filename)
    
    def plot_architecture_comparison(self, metric_name: str = 'accuracy'):
        """Compare metrics between different architectures (MLP vs GMLP)."""
        # Group runs by architecture
        mlp_runs = {}
        gmlp_runs = {}
        
        for run_name, run_data in self.multi_run_data.items():
            if 'mlp' in run_name.lower():
                mlp_runs[run_name] = run_data
            elif 'greedy' in run_name.lower() or 'gmlp' in run_name.lower():
                gmlp_runs[run_name] = run_data
        
        if not mlp_runs or not gmlp_runs:
            print("Need both MLP and GMLP runs for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot for each layer
        layers = ['layer_0', 'layer_1', 'layer_2', 'layer_3']
        
        for i, layer in enumerate(layers):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Plot MLP runs
            for run_name, run_data in mlp_runs.items():
                key = f"{layer}_{metric_name}"
                if key in run_data['metrics_data']['train'] and run_data['metrics_data']['train'][key]:
                    data = run_data['metrics_data']['train'][key]
                    steps = [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    ax.plot(steps, values, label=f'MLP ({run_name})', alpha=0.7, linestyle='-')
            
            # Plot GMLP runs
            for run_name, run_data in gmlp_runs.items():
                key = f"{layer}_{metric_name}"
                if key in run_data['metrics_data']['train'] and run_data['metrics_data']['train'][key]:
                    data = run_data['metrics_data']['train'][key]
                    steps = [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    ax.plot(steps, values, label=f'GMLP ({run_name})', alpha=0.7, linestyle='--')
            
            ax.set_title(f'{layer} - {metric_name.title()}')
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name.title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Architecture Comparison: {metric_name.title()}', fontsize=16)
        plt.tight_layout()
        self._save_plot(f'architecture_comparison_{metric_name}.png')
    
    def plot_metrics_correlation_matrix(self, phase: str = 'train'):
        """Create correlation matrix of metrics."""
        # Collect all metrics data
        all_metrics = defaultdict(list)
        
        for run_name, run_data in self.multi_run_data.items():
            if phase not in run_data['metrics_data']:
                continue
            
            for key, data in run_data['metrics_data'][phase].items():
                if data:
                    # Use latest value for each metric
                    latest_value = data[-1]['value']
                    all_metrics[key].append(latest_value)
        
        if not all_metrics:
            print(f"No metrics data found for {phase}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(max(12, len(corr_matrix.columns) * 0.8), 
                          max(10, len(corr_matrix.columns) * 0.8)))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title(f'Metrics Correlation Matrix - {phase.title()} Phase')
        plt.tight_layout()
        self._save_plot(f'correlation_matrix_{phase}.png')
    
    def plot_layer_progression_analysis(self):
        """Analyze how metrics progress across layers."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Key metrics for analysis
        metrics_to_analyze = [
            ('accuracy', 'Predictiveness'),
            ('participation_ratio', 'Compression'),
            ('alignment', 'Anchor Alignment'),
            ('margin', 'Classification Margin'),
            ('gaussian_entropy', 'Information Content'),
            ('linear_cka', 'Layer Similarity')
        ]
        
        for i, (metric, title) in enumerate(metrics_to_analyze):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Collect data for each layer
            layer_values = defaultdict(list)
            
            for run_name, run_data in self.multi_run_data.items():
                if 'train' not in run_data['metrics_data']:
                    continue
                
                for key, data in run_data['metrics_data']['train'].items():
                    if key.endswith(f'_{metric}') and data:
                        layer = key.replace(f'_{metric}', '')
                        # Use latest value
                        latest_value = data[-1]['value']
                        layer_values[layer].append(latest_value)
            
            # Plot box plots for each layer
            if layer_values:
                layers = sorted(layer_values.keys())
                values = [layer_values[layer] for layer in layers]
                
                bp = ax.boxplot(values, labels=layers, patch_artist=True)
                
                # Color boxes
                colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{title} Across Layers')
                ax.set_xlabel('Layer')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Layer Progression Analysis', fontsize=16)
        plt.tight_layout()
        self._save_plot('layer_progression_analysis.png')
    
    def plot_metrics_timeline(self, metric_name: str, layer_name: str = 'layer_0'):
        """Plot metric evolution over time for multiple runs."""
        plt.figure(figsize=(15, 8))
        
        for run_name, run_data in self.multi_run_data.items():
            if 'train' not in run_data['metrics_data']:
                continue
            
            key = f"{layer_name}_{metric_name}"
            if key in run_data['metrics_data']['train'] and run_data['metrics_data']['train'][key]:
                data = run_data['metrics_data']['train'][key]
                steps = [d['step'] for d in data]
                values = [d['value'] for d in data]
                
                # Smooth the curve
                if len(values) > 10:
                    from scipy.ndimage import gaussian_filter1d
                    smoothed_values = gaussian_filter1d(values, sigma=1)
                    plt.plot(steps, smoothed_values, label=run_name, alpha=0.8, linewidth=2)
                else:
                    plt.plot(steps, values, label=run_name, alpha=0.8, linewidth=2)
        
        plt.xlabel('Training Step')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} Timeline - {layer_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f'timeline_{metric_name}_{layer_name}.png'
        self._save_plot(filename)
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("Generating comprehensive metrics analysis report...")
        
        # Multi-run comparisons for key metrics
        key_metrics = ['accuracy', 'participation_ratio', 'alignment', 'margin']
        
        for metric in key_metrics:
            self.plot_multi_run_comparison(metric, phase='train')
        
        # Architecture comparison
        self.plot_architecture_comparison('accuracy')
        
        # Correlation analysis
        self.plot_metrics_correlation_matrix('train')
        
        # Layer progression analysis
        self.plot_layer_progression_analysis()
        
        # Timeline analysis for key metrics
        for metric in ['accuracy', 'participation_ratio']:
            self.plot_metrics_timeline(metric, 'layer_0')
            self.plot_metrics_timeline(metric, 'layer_3')
        
        print("Comprehensive report generated successfully!")
    
    def _save_plot(self, filename: str):
        """Save plot to the first run directory."""
        if self.run_dirs:
            output_path = os.path.join(self.run_dirs[0], filename)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            if self.show_plots:
                plt.show()
            plt.close()
            print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Advanced metrics visualization')
    parser.add_argument('--run_dirs', nargs='+', required=True, 
                       help='Paths to run directories for comparison')
    parser.add_argument('--show', action='store_true', 
                       help='Show plots instead of saving only')
    parser.add_argument('--metric', type=str, 
                       help='Plot specific metric only')
    parser.add_argument('--layer', type=str, 
                       help='Plot specific layer only')
    parser.add_argument('--phase', type=str, default='train', 
                       choices=['train', 'val'], help='Phase to plot')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Generate comprehensive analysis report')
    parser.add_argument('--architecture_comparison', action='store_true',
                       help='Compare MLP vs GMLP architectures')
    
    args = parser.parse_args()
    
    try:
        visualizer = AdvancedMetricsVisualizer(args.run_dirs, args.show)
        
        if args.comprehensive:
            visualizer.generate_comprehensive_report()
        elif args.architecture_comparison:
            visualizer.plot_architecture_comparison(args.metric or 'accuracy')
        elif args.metric:
            visualizer.plot_multi_run_comparison(args.metric, args.layer, args.phase)
        else:
            print("Please specify an analysis type (--comprehensive, --architecture_comparison, or --metric)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
