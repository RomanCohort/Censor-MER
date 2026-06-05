"""
Experiment 4: Sparse Control Statistics
========================================
Analyze sparse control mechanism for parameter efficiency.

Usage:
    python experiments/exp4_sparse_control.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 4: Sparse Control Statistics")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    from main import Censor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create model with sparse control
print("\nCreating model with sparse control...")
model = Censor(verbose=False, enable_sparse_control=True)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Simulate training
print("\nSimulating training (300 steps)...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_steps = 300
log_interval = 50

for step in range(num_steps):
    video = torch.randn(2, 3, 16, 224, 224).to(device)
    optimizer.zero_grad()
    out = model(video)
    loss = out.sum()
    loss.backward()
    optimizer.step()

    if (step + 1) % log_interval == 0:
        print(f"  Step {step + 1}/{num_steps}")

# Collect statistics
print("\n" + "=" * 60)
print("Sparse Control Statistics")
print("=" * 60)

final_stats = {}
if hasattr(model, 'sparse_control') and model.sparse_control:
    for name, ctrl in model.sparse_control.sparse_controllers.items():
        if hasattr(ctrl, 'get_sparse_stats'):
            final_stats[name] = ctrl.get_sparse_stats()

# Calculate effective parameters
neuron_map = {'fast': 512, 'slow': 768, 'fusion': 1024}
total_neurons = 0
frozen_neurons = 0

for name, stats in final_stats.items():
    frozen_ratio = stats.get('frozen_ratio', 0)
    neurons = neuron_map.get(name, 1024)

    total_neurons += neurons
    frozen_neurons += int(neurons * frozen_ratio)

    print(f"\n{name}:")
    print(f"  Neurons: {neurons}")
    print(f"  Frozen: {frozen_ratio*100:.1f}%")
    print(f"  Active: {(1-frozen_ratio)*100:.1f}%")
    print(f"  Usage mean: {stats.get('usage_mean', 0):.3f}")

# Summary
active_ratio = 1 - frozen_neurons / total_neurons
effective_params = total_params * active_ratio

print("\n" + "-" * 60)
print("Summary:")
print(f"  Total neurons: {total_neurons}")
print(f"  Frozen neurons: {frozen_neurons} ({frozen_neurons/total_neurons*100:.1f}%)")
print(f"  Active neurons: {total_neurons - frozen_neurons} ({active_ratio*100:.1f}%)")
print(f"  Total params: {total_params/1e6:.2f}M")
print(f"  Effective params: ~{effective_params/1e6:.2f}M")
print("-" * 60)

# Save
output_dir = Path(__file__).parent.parent / 'results'
output_dir.mkdir(exist_ok=True)

with open(output_dir / 'exp4_sparse_control.txt', 'w') as f:
    f.write(f"Sparse Control Statistics\n")
    f.write(f"Date: {datetime.now()}\n\n")
    f.write(f"Total params: {total_params/1e6:.2f}M\n")
    f.write(f"Frozen ratio: {frozen_neurons/total_neurons*100:.1f}%\n")
    f.write(f"Active ratio: {active_ratio*100:.1f}%\n")
    f.write(f"Effective params: ~{effective_params/1e6:.2f}M\n\n")

    f.write("Per-stage stats:\n")
    for name, stats in final_stats.items():
        f.write(f"\n{name}:\n")
        for k, v in stats.items():
            f.write(f"  {k}: {v}\n")

print(f"\nSaved to: {output_dir / 'exp4_sparse_control.txt'}")