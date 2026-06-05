"""
Experiment 3: Inference Latency Benchmark
==========================================
Benchmark inference latency for deployment guidance.

Usage:
    python experiments/exp3_latency.py
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
print("Experiment 3: Inference Latency Benchmark")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    from main import Censor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configurations
configs = [
    ('Full Model', {}),
    ('Fast-only', {'single_path': 'fast'}),
    ('No-rPPG', {'no_rppg': True}),
    ('With Sparse Control', {'enable_sparse_control': True}),
]

B, C, T, H, W = 1, 3, 16, 224, 224
num_runs = 100
warmup = 10

print(f"\nInput shape: ({B}, {C}, {T}, {H}, {W})")
print(f"Benchmark: {num_runs} runs, {warmup} warmup")

results = {}

for name, kwargs in configs:
    print(f"\n{name}:")

    try:
        model = Censor(verbose=False, **kwargs)
        model = model.to(device)
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parameters: {params:.2f}M")

        # Warmup
        video = torch.randn(B, C, T, H, W).to(device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(video)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(video)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)

        latencies = np.array(latencies)

        results[name] = {
            'params': params,
            'mean_ms': latencies.mean(),
            'std_ms': latencies.std(),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'throughput': 1000 / latencies.mean(),
        }

        if device.type == 'cuda':
            mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            results[name]['gpu_mem_mb'] = mem
            print(f"  GPU Memory: {mem:.1f} MB")

        print(f"  Latency: {latencies.mean():.2f} ± {latencies.std():.2f} ms")
        print(f"  P95: {np.percentile(latencies, 95):.2f} ms")
        print(f"  Throughput: {results[name]['throughput']:.1f} fps")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    except Exception as e:
        print(f"  Error: {e}")
        results[name] = {'error': str(e)}

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"{'Config':<25} {'Params':<10} {'Latency':<15} {'Throughput':<12}")
print("-" * 60)
for name, r in results.items():
    if 'mean_ms' in r:
        print(f"{name:<25} {r['params']:<10.2f} {r['mean_ms']:<15.2f} {r['throughput']:<12.1f}")

# Save
output_dir = Path(__file__).parent.parent / 'results'
output_dir.mkdir(exist_ok=True)

with open(output_dir / 'exp3_latency.txt', 'w') as f:
    f.write(f"Inference Latency Benchmark\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Input: ({B}, {C}, {T}, {H}, {W})\n\n")
    for name, r in results.items():
        f.write(f"{name}:\n")
        for k, v in r.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

print(f"\nSaved to: {output_dir / 'exp3_latency.txt'}")