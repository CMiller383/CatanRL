#!/usr/bin/env python
"""
Standalone script for plotting AlphaZero Catan training metrics from saved model checkpoints.
This script can load one or multiple model files and generate visualizations of training progress.

Usage:
    python plot_training_metrics.py --model models/best_model.pt
    python plot_training_metrics.py --models models/model_iter_*.pt
    python plot_training_metrics.py --models models/model_iter_100.pt models/model_iter_200.pt models/model_iter_300.pt
"""

import argparse
import os
import torch
import numpy as np
import json
from datetime import datetime
import glob

def setup_plotting():
    """Ensure plotting libraries are installed and imported."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return True
    except ImportError:
        print("Plotly not found. Installing required packages...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "plotly", "kaleido"])
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            print("Successfully installed plotting packages.")
            return True
        except Exception as e:
            print(f"Error installing packages: {e}")
            print("Please install required packages manually: pip install plotly kaleido")
            return False

def load_model_metrics(model_path):
    """Load metrics from a saved model checkpoint."""
    try:
        # Print status
        print(f"Loading metrics from {model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract metrics
        metrics = checkpoint.get('metrics', {})
        iteration = checkpoint.get('iteration', 0)
        
        # Print basic info
        print(f"  Loaded checkpoint from iteration {iteration}")
        
        # If no metrics in checkpoint, initialize empty ones
        if not metrics:
            print(f"  Warning: No metrics found in checkpoint. Creating empty metrics.")
            metrics = {
                'iteration': [],
                'policy_loss': [],
                'value_loss': [],
                'total_loss': [],
                'win_rate': [],
                'avg_vp': [],
                'avg_game_length': [],
                'total_moves': []
            }
            
            # Add the current iteration if it's not a placeholder
            if iteration > 0:
                metrics['iteration'].append(iteration)
        
        return metrics, iteration
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None, None

def combine_metrics(all_metrics):
    """Combine metrics from multiple models into a single collection."""
    if not all_metrics:
        return {}
    
    # Initialize combined metrics with keys from first model
    combined = {key: [] for key in all_metrics[0].keys()}
    
    # Track seen iterations to avoid duplicates
    seen_iterations = set()
    
    # Merge all metrics
    for metrics in all_metrics:
        for i, iter_num in enumerate(metrics.get('iteration', [])):
            if iter_num not in seen_iterations:
                seen_iterations.add(iter_num)
                for key in combined.keys():
                    if key in metrics and i < len(metrics[key]):
                        combined[key].append(metrics[key][i])
    
    # Sort all metrics by iteration
    if combined.get('iteration'):
        sort_indices = np.argsort(combined['iteration'])
        for key in combined:
            combined[key] = [combined[key][i] for i in sort_indices]
    
    return combined

def calculate_statistics(metrics, window_size=5):
    """Calculate rolling statistics for metrics."""
    stats = {}
    
    # Only process if we have enough data
    if not metrics or 'iteration' not in metrics or len(metrics['iteration']) < window_size:
        return stats
    
    # Prepare iterations for stats
    iterations = metrics['iteration']
    window_iterations = []
    
    # Calculate windows of iterations (every window_size iterations)
    for i in range(0, len(iterations), window_size):
        if i + window_size <= len(iterations):
            window_start = iterations[i]
            window_end = iterations[i + window_size - 1]
            window_iterations.append((window_start, window_end))
    
    # Calculate statistics for each window
    stats['windows'] = []
    
    for start_idx in range(0, len(iterations), window_size):
        end_idx = min(start_idx + window_size, len(iterations))
        if end_idx - start_idx < 2:  # Need at least 2 points for meaningful stats
            continue
            
        window_stats = {
            'window_start': iterations[start_idx],
            'window_end': iterations[end_idx - 1],
            'metrics': {}
        }
        
        # Calculate stats for each metric
        for key in ['win_rate', 'avg_vp', 'avg_game_length', 'policy_loss', 'value_loss', 'total_loss']:
            if key in metrics:
                values = metrics[key][start_idx:end_idx]
                # Check if values are non-zero for win_rate, avg_vp, avg_game_length
                if key in ['win_rate', 'avg_vp', 'avg_game_length']:
                    values = [v for v in values if v > 0]
                
                if values:
                    window_stats['metrics'][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        stats['windows'].append(window_stats)
    
    return stats

def filter_non_zero_metrics(metrics, keys):
    """Filter out zero values from specific metrics."""
    result = {}
    for key in metrics:
        if key in keys:
            # For keys we want to filter, create paired data with non-zero values
            result[key] = []
            iterations = []
            for i, val in enumerate(metrics[key]):
                if val > 0:
                    result[key].append(val)
                    if i < len(metrics['iteration']):
                        iterations.append(metrics['iteration'][i])
            result[f"{key}_iterations"] = iterations
        else:
            # For other keys, keep as is
            result[key] = metrics[key]
    return result

def plot_metrics(metrics, output_dir='plots', filename_prefix='training_metrics'):
    """Plot the training metrics using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Please install with: pip install plotly kaleido")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of rows based on available metrics
    num_rows = 1
    
    # Create subplot
    subplot_titles = [
        'Training Losses', 
        'Performance Metrics', 
        'Game Length Metrics'
    ]
    
    fig = make_subplots(
        rows=num_rows, 
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1
    )
    
    # Subplot 1: Training Losses
    if all(k in metrics for k in ['iteration', 'policy_loss', 'value_loss', 'total_loss']):
        fig.add_trace(
            go.Scatter(
                x=metrics['iteration'], 
                y=metrics['policy_loss'],
                mode='lines+markers',
                name='Policy Loss'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=metrics['iteration'], 
                y=metrics['value_loss'],
                mode='lines+markers',
                name='Value Loss'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=metrics['iteration'], 
                y=metrics['total_loss'],
                mode='lines+markers',
                name='Total Loss'
            ),
            row=1, col=1
        )
    
    # # Filter metrics to remove zero values
    # filtered_metrics = filter_non_zero_metrics(metrics, ['win_rate', 'avg_vp', 'avg_game_length'])
    
    # # Subplot 2: Performance Metrics (only non-zero values)
    # if 'win_rate' in filtered_metrics and filtered_metrics['win_rate']:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=filtered_metrics['win_rate_iterations'],
    #             y=filtered_metrics['win_rate'],
    #             mode='lines+markers',
    #             name='Win Rate'
    #         ),
    #         row=2, col=1
    #     )
    
    # if 'avg_vp' in filtered_metrics and filtered_metrics['avg_vp']:
    #     # Scale VPs for comparison with win rate
    #     avg_vp_scaled = [vp / 10 for vp in filtered_metrics['avg_vp']]
    #     fig.add_trace(
    #         go.Scatter(
    #             x=filtered_metrics['avg_vp_iterations'],
    #             y=avg_vp_scaled,
    #             mode='lines+markers',
    #             name='Avg VP / 10'
    #         ),
    #         row=2, col=1
    #     )
    
    # # Subplot 3: Game Length Metrics (only non-zero values)
    # if 'avg_game_length' in filtered_metrics and filtered_metrics['avg_game_length']:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=filtered_metrics['avg_game_length_iterations'],
    #             y=filtered_metrics['avg_game_length'],
    #             mode='lines+markers',
    #             name='Avg Game Length (moves)'
    #         ),
    #         row=3, col=1
    #     )
    
    # Update layout
    # fig.update_layout(
    #     height=250 * num_rows,
    #     width=900,
    #     showlegend=True
    # )
    
    # Add axis labels
    fig.update_xaxes(title_text="Iteration", row=num_rows, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    # fig.update_yaxes(title_text="Performance", row=2, col=1)
    # fig.update_yaxes(title_text="Moves", row=3, col=1)
    
    # Save as HTML file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.html")
    fig.write_html(html_path)
    print(f"Saved interactive plot to {html_path}")
    
    # Also try saving as an image
    try:
        img_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.png")
        fig.write_image(img_path)
        print(f"Saved image to {img_path}")
    except Exception as e:
        print(f"Could not save image (requires kaleido): {e}")
    
    return html_path

def print_statistics(stats):
    """Print the calculated statistics in a readable format."""
    print("\n=== Training Progress Statistics ===")
    
    for window in stats.get('windows', []):
        start = window['window_start']
        end = window['window_end']
        print(f"\nIterations {start} to {end}:")
        
        for metric, values in window['metrics'].items():
            if metric == 'win_rate':
                print(f"  Win Rate: {values['mean']:.2f} ± {values['std']:.2f} (min: {values['min']:.2f}, max: {values['max']:.2f})")
            elif metric == 'avg_vp':
                print(f"  Avg VP: {values['mean']:.2f} ± {values['std']:.2f} (min: {values['min']:.2f}, max: {values['max']:.2f})")
            elif metric == 'avg_game_length':
                print(f"  Avg Game Length: {values['mean']:.1f} ± {values['std']:.1f} moves (min: {values['min']:.1f}, max: {values['max']:.1f})")
            elif metric == 'total_loss':
                print(f"  Total Loss: {values['mean']:.4f} ± {values['std']:.4f} (min: {values['min']:.4f}, max: {values['max']:.4f})")

def export_metrics(metrics, stats, output_dir='plots'):
    """Export metrics and statistics to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"metrics_export_{timestamp}.json")
    
    # Prepare export data
    export_data = {
        'raw_metrics': metrics,
        'statistics': stats,
        'export_time': timestamp
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported metrics and statistics to {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Plot AlphaZero Catan training metrics from model checkpoints")
    
    # Model selection arguments (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Path to a single model checkpoint")
    model_group.add_argument("--models", type=str, nargs='+', help="Paths to multiple model checkpoints")
    model_group.add_argument("--pattern", type=str, help="Glob pattern to select multiple models (e.g., 'models/model_iter_*.pt')")
    
    # Additional options
    parser.add_argument("--window", type=int, default=5, help="Window size for statistics calculation (default: 5)")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save outputs (default: plots)")
    parser.add_argument("--prefix", type=str, default="training_metrics", help="Prefix for output filenames")
    parser.add_argument("--include-zeros", action="store_true", help="Include zero values in performance metrics plots")
    
    args = parser.parse_args()
    
    # Check if plotting libraries are available
    if not setup_plotting():
        return
    
    # Collect model paths
    model_paths = []
    if args.model:
        model_paths = [args.model]
    elif args.models:
        model_paths = args.models
    elif args.pattern:
        model_paths = sorted(glob.glob(args.pattern))
    
    if not model_paths:
        print("No model files found!")
        return
    
    print(f"Processing {len(model_paths)} model checkpoint(s)...")
    
    # Load metrics from all models
    all_metrics = []
    for path in model_paths:
        metrics, iteration = load_model_metrics(path)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No metrics data found in the model checkpoint(s).")
        return
    
    # Combine metrics from all models
    combined_metrics = combine_metrics(all_metrics)
    
    # Calculate statistics
    stats = calculate_statistics(combined_metrics, window_size=args.window)
    
    # Print statistics
    print_statistics(stats)
    
    # Plot metrics
    plot_path = plot_metrics(combined_metrics, args.output_dir, args.prefix)
    
    # Export metrics and statistics
    export_metrics(combined_metrics, stats, args.output_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()