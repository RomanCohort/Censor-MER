"""
Experiment 4: Sparse Control Statistics (Full Implementation)
===============================================================
Analyze sparse control mechanism during training to demonstrate
parameter efficiency.

Usage:
    python experiments/exp4_sparse_control.py
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 4: Sparse Control Statistics")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Configuration
# =============================================================================

try:
    from main import Censor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# Training Simulation
# =============================================================================

def simulate_training(model, num_epochs=10, steps_per_epoch=50):
    """Simulate training and collect sparse control statistics."""

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = []
    total_steps = num_epochs * steps_per_epoch

    print(f"\nSimulating {num_epochs} epochs ({total_steps} steps)...")

    for epoch in range(num_epochs):
        epoch_stats = []

        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step

            # Forward pass
            video = torch.randn(2, 3, 16, 224, 224).to(device)
            optimizer.zero_grad()
            out = model(video)
            loss = out.sum()
            loss.backward()
            optimizer.step()

            # Collect stats every 10 steps
            if step % 10 == 0:
                stats = collect_sparse_stats(model)
                stats['step'] = global_step
                stats['epoch'] = epoch
                epoch_stats.append(stats)

        # End of epoch summary
        if epoch_stats:
            frozen_ratios = [s['total_frozen_ratio'] for s in epoch_stats]
            print(f"  Epoch {epoch+1}: Frozen {np.mean(frozen_ratios)*100:.1f}%")

        history.extend(epoch_stats)

    return history


def collect_sparse_stats(model):
    """Collect sparse control statistics from model."""

    stats = {
        'stages': {},
        'total_neurons': 0,
        'total_frozen': 0,
        'total_frozen_ratio': 0,
        'total_usage_mean': 0,
    }

    if not hasattr(model, 'sparse_control') or model.sparse_control is None:
        return stats

    neuron_map = {'fast': 512, 'slow': 768, 'fusion': 1024}

    for name, ctrl in model.sparse_control.sparse_controllers.items():
        if hasattr(ctrl, 'get_sparse_stats'):
            ctrl_stats = ctrl.get_sparse_stats()
            frozen_ratio = ctrl_stats.get('frozen_ratio', 0)
            usage_mean = ctrl_stats.get('usage_mean', 0)

            neurons = neuron_map.get(name, 1024)
            frozen = int(neurons * frozen_ratio)

            stats['stages'][name] = {
                'neurons': neurons,
                'frozen_ratio': frozen_ratio,
                'frozen_count': frozen,
                'usage_mean': usage_mean,
            }

            stats['total_neurons'] += neurons
            stats['total_frozen'] += frozen

    if stats['total_neurons'] > 0:
        stats['total_frozen_ratio'] = stats['total_frozen'] / stats['total_neurons']

    return stats


# =============================================================================
# Analysis
# =============================================================================

def analyze_history(history):
    """Analyze training history."""

    if not history:
        return None

    # Extract trends
    steps = [h['step'] for h in history]
    frozen_ratios = [h['total_frozen_ratio'] for h in history]

    # Stage-wise analysis
    stage_trends = {}
    stages = ['fast', 'slow', 'fusion']

    for stage in stages:
        ratios = [h['stages'].get(stage, {}).get('frozen_ratio', 0) for h in history]
        if ratios:
            stage_trends[stage] = {
                'initial': ratios[0],
                'final': ratios[-1],
                'mean': np.mean(ratios),
                'std': np.std(ratios),
            }

    return {
        'steps': steps,
        'frozen_ratio_trend': frozen_ratios,
        'initial_frozen_ratio': frozen_ratios[0] if frozen_ratios else 0,
        'final_frozen_ratio': frozen_ratios[-1] if frozen_ratios else 0,
        'mean_frozen_ratio': np.mean(frozen_ratios),
        'stage_trends': stage_trends,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    # Create model with sparse control
    print("\nInitializing model with sparse control...")
    model = Censor(verbose=False, enable_sparse_control=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Initial stats
    print("\nInitial state:")
    initial_stats = collect_sparse_stats(model)
    print(f"  Frozen ratio: {initial_stats['total_frozen_ratio']*100:.1f}%")

    # Simulate training
    history = simulate_training(model, num_epochs=10, steps_per_epoch=50)

    # Final stats
    print("\nFinal state:")
    final_stats = collect_sparse_stats(model)
    print(f"  Frozen ratio: {final_stats['total_frozen_ratio']*100:.1f}%")

    # Calculate effective parameters
    active_ratio = 1 - final_stats['total_frozen_ratio']
    effective_params = total_params * active_ratio

    print("\n" + "=" * 60)
    print("Parameter Efficiency Analysis")
    print("=" * 60)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Hard frozen neurons: {final_stats['total_frozen']} ({final_stats['total_frozen_ratio']*100:.1f}%)")
    print(f"Active neurons: {final_stats['total_neurons'] - final_stats['total_frozen']} ({active_ratio*100:.1f}%)")
    print(f"Effective parameters: ~{effective_params/1e6:.2f}M")
    print("=" * 60)

    # Per-stage breakdown
    print("\nPer-stage statistics:")
    for stage, s in final_stats['stages'].items():
        print(f"  {stage}: {s['frozen_ratio']*100:.1f}% frozen, usage={s['usage_mean']:.3f}")

    # Analyze trends
    analysis = analyze_history(history)

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results = {
        'date': datetime.now().isoformat(),
        'device': str(device),
        'total_params': total_params,
        'effective_params': effective_params,
        'initial_stats': initial_stats,
        'final_stats': final_stats,
        'analysis': analysis,
        'history': history,
    }

    with open(output_dir / 'exp4_sparse_control.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary text
    with open(output_dir / 'exp4_sparse_control.txt', 'w') as f:
        f.write(f"Sparse Control Statistics\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Total params: {total_params/1e6:.2f}M\n")
        f.write(f"Effective params: {effective_params/1e6:.2f}M\n")
        f.write(f"Frozen ratio: {final_stats['total_frozen_ratio']*100:.1f}%\n\n")

        f.write("Per-stage:\n")
        for stage, s in final_stats['stages'].items():
            f.write(f"  {stage}: frozen={s['frozen_ratio']*100:.1f}%, usage={s['usage_mean']:.3f}\n")

        f.write("\nTrend:\n")
        if analysis:
            f.write(f"  Initial frozen: {analysis['initial_frozen_ratio']*100:.1f}%\n")
            f.write(f"  Final frozen: {analysis['final_frozen_ratio']*100:.1f}%\n")

    print(f"\nSaved to: {output_dir / 'exp4_sparse_control.json'}")
    print(f"Summary: {output_dir / 'exp4_sparse_control.txt'}")


if __name__ == '__main__':
    main()