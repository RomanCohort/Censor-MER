"""
Experiment 3: Inference Latency Benchmark (Full Implementation)
=================================================================
Comprehensive latency benchmark for deployment guidance.

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

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 3: Inference Latency Benchmark")
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

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# =============================================================================
# Benchmark Function
# =============================================================================

def benchmark_model(model, input_shape, num_runs=200, warmup=20):
    """Comprehensive benchmark with multiple metrics."""

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
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
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
    ('Full Model', {}),
    ('Fast-only', {'single_path': 'fast'}),
    ('No-rPPG', {'no_rppg': True}),
    ('With Sparse Control', {'enable_sparse_control': True}),
    ('Minimal (Fast + No-MOE)', {'single_path': 'fast', 'no_moe': True}),
]

# Test different input sizes
input_sizes = [
    (1, 3, 16, 224, 224),   # Standard
    (2, 3, 16, 224, 224),   # Batch 2
    (4, 3, 16, 224, 224),   # Batch 4
]

# =============================================================================
# Run Benchmarks
# =============================================================================

results = {}

print("\n" + "=" * 60)
print("Running benchmarks...")
print("=" * 60)

for config_name, kwargs in configs:
    print(f"\n{config_name}:")
    results[config_name] = {'config': kwargs}

    try:
        model = Censor(verbose=False, **kwargs)
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results[config_name]['total_params'] = total_params
        results[config_name]['trainable_params'] = trainable_params
        results[config_name]['params_m'] = total_params / 1e6

        print(f"  Parameters: {total_params/1e6:.2f}M")

        # Benchmark different batch sizes
        for batch_size in [1, 2, 4]:
            input_shape = (batch_size, 3, 16, 224, 224)
            print(f"  Batch {batch_size}: ", end='', flush=True)

            stats = benchmark_model(model, input_shape, num_runs=100)

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
print("Summary Table")
print("=" * 60)

print(f"{'Config':<30} {'Params':<8} {'Latency':<12} {'Throughput':<12} {'Memory':<10}")
print("-" * 72)

for name, r in results.items():
    if 'error' not in r and 'batch_1' in r:
        b1 = r['batch_1']
        params = r['params_m']
        latency = b1['mean_ms']
        throughput = b1['throughput_fps']
        memory = b1.get('gpu_memory_max_mb', 0)

        print(f"{name:<30} {params:<8.2f} {latency:<12.2f} {throughput:<12.1f} {memory:<10.1f}")

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
    }, f, indent=2)

# Also save readable text
with open(output_dir / 'exp3_latency.txt', 'w') as f:
    f.write(f"Inference Latency Benchmark\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Device: {device}\n\n")

    f.write(f"{'Config':<30} {'Params':<8} {'B1':<10} {'B2':<10} {'B4':<10}\n")
    f.write("-" * 70 + "\n")

    for name, r in results.items():
        if 'error' not in r and 'batch_1' in r:
            b1 = r['batch_1'].get('mean_ms', 0)
            b2 = r['batch_2'].get('mean_ms', 0)
            b4 = r['batch_4'].get('mean_ms', 0)
            params = r['params_m']
            f.write(f"{name:<30} {params:<8.2f} {b1:<10.2f} {b2:<10.2f} {b4:<10.2f}\n")

print(f"\nSaved to: {output_file}")
print(f"Readable: {output_dir / 'exp3_latency.txt'}")