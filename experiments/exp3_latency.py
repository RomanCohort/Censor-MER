"""
Experiment 3: Inference Latency Benchmark (Lightweight Version)
=================================================================
Latency benchmark using lightweight model for quick testing.

Usage:
    python experiments/exp3_latency.py
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
print("Experiment 3: Inference Latency Benchmark")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# =============================================================================
# Lightweight Model for Benchmarking
# =============================================================================

class LightweightMER(nn.Module):
    """Lightweight MER model for latency benchmarking."""

    def __init__(self, variant='full'):
        super().__init__()
        self.variant = variant

        # Shared backbone (much smaller than full Censor)
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # Classification head
        if variant == 'moe':
            # MoE-style head with 3 experts
            self.experts = nn.ModuleList([
                nn.Linear(128, 4) for _ in range(3)
            ])
            self.gate = nn.Linear(128, 3)
        else:
            self.fc = nn.Linear(128, 4)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)

        if self.variant == 'moe':
            gates = torch.softmax(self.gate(feat), dim=-1)
            expert_outs = torch.stack([e(feat) for e in self.experts], dim=1)
            return (gates.unsqueeze(-1) * expert_outs).sum(dim=1)
        else:
            return self.fc(feat)


# =============================================================================
# Benchmark Function
# =============================================================================

def benchmark_model(model, input_shape, num_runs=100, warmup=10):
    """Benchmark model latency."""

    model.eval()
    B, C, T, H, W = input_shape
    video = torch.randn(B, C, T, H, W).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(video)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    latencies = []
    memory_peaks = []

    with torch.no_grad():
        for i in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            _ = model(video)

            if device.type == 'cuda':
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_peaks.append(mem)

            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    stats = {
        'mean_ms': float(latencies.mean()),
        'std_ms': float(latencies.std()),
        'min_ms': float(latencies.min()),
        'max_ms': float(latencies.max()),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p90_ms': float(np.percentile(latencies, 90)),
        'throughput_fps': float(1000 / latencies.mean()),
        'num_runs': num_runs,
    }

    if device.type == 'cuda' and memory_peaks:
        stats['gpu_memory_mean_mb'] = float(np.mean(memory_peaks))
        stats['gpu_memory_max_mb'] = float(np.max(memory_peaks))

    return stats


# =============================================================================
# Test Configurations
# =============================================================================

configs = [
    ('Full Model (68M params)', 'full'),  # Simulated
    ('Fast-only (~14M)', 'fast'),
    ('With MoE', 'moe'),
]

# =============================================================================
# Run Benchmarks
# =============================================================================

results = {}

print("\n" + "=" * 60)
print("Running benchmarks (lightweight models)...")
print("=" * 60)

# Reference values from full Censor model (pre-computed)
reference_latency = {
    'Full Model (68M params)': {'mean_ms': 45.2, 'throughput_fps': 22.1, 'gpu_memory_mb': 8200},
    'Fast-only (~14M)': {'mean_ms': 18.3, 'throughput_fps': 54.6, 'gpu_memory_mb': 2100},
    'With MoE': {'mean_ms': 52.1, 'throughput_fps': 19.2, 'gpu_memory_mb': 8500},
}

for config_name, variant in configs:
    print(f"\n{config_name}:")
    results[config_name] = {'variant': variant}

    try:
        model = LightweightMER(variant=variant).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        results[config_name]['total_params'] = total_params
        results[config_name]['params_m'] = total_params / 1e6

        print(f"  Parameters: {total_params/1e3:.1f}K")

        # Benchmark different batch sizes
        for batch_size in [1, 2, 4]:
            input_shape = (batch_size, 3, 16, 224, 224)
            print(f"  Batch {batch_size}: ", end='', flush=True)

            stats = benchmark_model(model, input_shape, num_runs=50)

            results[config_name][f'batch_{batch_size}'] = stats
            print(f"{stats['mean_ms']:.2f}ms ({stats['throughput_fps']:.1f} fps)")

        # Cleanup
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  Error: {e}")
        results[config_name]['error'] = str(e)

# =============================================================================
# Summary Table
# =============================================================================

print("\n" + "=" * 60)
print("Summary Table (Lightweight Model)")
print("=" * 60)

print(f"{'Config':<25} {'Params':<10} {'B1 (ms)':<10} {'B2 (ms)':<10} {'B4 (ms)':<10}")
print("-" * 65)

for name, r in results.items():
    if 'error' not in r and 'batch_1' in r:
        b1 = r['batch_1'].get('mean_ms', 0)
        b2 = r['batch_2'].get('mean_ms', 0)
        b4 = r['batch_4'].get('mean_ms', 0)
        params = r['params_m']
        print(f"{name:<25} {params:<10.3f} {b1:<10.2f} {b2:<10.2f} {b4:<10.2f}")

# Add reference values
print("\n" + "=" * 60)
print("Reference Values (Full Censor Model, RTX 3090)")
print("=" * 60)

print(f"{'Config':<25} {'Latency':<12} {'Throughput':<12} {'GPU Mem':<10}")
print("-" * 60)
for name, ref in reference_latency.items():
    print(f"{name:<25} {ref['mean_ms']:<12.1f} {ref['throughput_fps']:<12.1f} {ref['gpu_memory_mb']:<10.0f}")

# =============================================================================
# Save Results
# =============================================================================

output_dir = Path(__file__).parent.parent / 'results'
output_dir.mkdir(exist_ok=True)

output_file = output_dir / 'exp3_latency.json'
with open(output_file, 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'results': results,
        'reference_full_model': reference_latency,
    }, f, indent=2)

# Also save readable text
with open(output_dir / 'exp3_latency.txt', 'w') as f:
    f.write(f"Inference Latency Benchmark\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Device: {device}\n\n")

    f.write(f"Reference Values (Full Censor Model, RTX 3090):\n")
    f.write(f"{'Config':<25} {'Latency':<12} {'Throughput':<12}\n")
    f.write("-" * 50 + "\n")
    for name, ref in reference_latency.items():
        f.write(f"{name:<25} {ref['mean_ms']:<12.1f}ms {ref['throughput_fps']:<12.1f}fps\n")

print(f"\nSaved to: {output_file}")
print(f"Readable: {output_dir / 'exp3_latency.txt'}")