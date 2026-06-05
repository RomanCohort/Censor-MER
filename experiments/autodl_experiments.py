"""
Censor -- AutoDL Experiment Scripts
====================================
Three experiments for paper revision:
1. OFF-ApexNet LOSO reproduction for fair SOTA comparison
2. rPPG signal quality validation (heart rate estimation)
3. Inference latency benchmark (ms/frame)

Usage on AutoDL:
    python experiments/autodl_experiments.py --exp 1  # OFF-ApexNet baseline
    python experiments/autodl_experiments.py --exp 2  # rPPG validation
    python experiments/autodl_experiments.py --exp 3  # Latency benchmark
    python experiments/autodl_experiments.py --exp all # Run all
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Experiment 1: OFF-ApexNet LOSO Reproduction
# =============================================================================

def experiment_1_offapexnet_loso():
    """
    Reproduce OFF-ApexNet under same LOSO protocol for fair SOTA comparison.

    OFF-ApexNet architecture:
    - Two-stream: RGB + Optical Flow
    - Backbone: VGG-16 (pretrained on ImageNet)
    - Fusion: Concatenation + FC layers
    - Output: 4-class (happiness, surprise, disgust, repression)

    Expected result: ~87.64% (from original paper on CASME II)
    Our LOSO result: TBD
    """
    print("=" * 60)
    print("Experiment 1: OFF-ApexNet LOSO Reproduction")
    print("=" * 60)

    try:
        from torchvision.models import vgg16
        from dataset import CASME2Dataset
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure dataset.py and CASME II data are available")
        return None

    class OFFApexNet(nn.Module):
        """OFF-ApexNet: Two-stream VGG-16 for MER."""

        def __init__(self, num_classes=4):
            super().__init__()
            # RGB stream
            vgg_rgb = vgg16(pretrained=True)
            self.rgb_features = vgg_rgb.features

            # Optical flow stream (2-channel input)
            vgg_flow = vgg16(pretrained=True)
            # Modify first conv for 2-channel input
            first_conv = vgg_flow.features[0]
            vgg_flow.features[0] = nn.Conv2d(2, 64, kernel_size=3, padding=1)
            # Initialize with average of RGB weights
            with torch.no_grad():
                vgg_flow.features[0].weight.data = first_conv.weight.data[:, :2].clone()
            self.flow_features = vgg_flow.features

            # Fusion classifier
            self.classifier = nn.Sequential(
                nn.Linear(512 * 2, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes),
            )

        def forward(self, rgb_frame, flow_frame):
            # Extract features
            rgb_feat = self.rgb_features(rgb_frame)
            rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)

            flow_feat = self.flow_features(flow_frame)
            flow_feat = flow_feat.view(flow_feat.size(0), -1)

            # Fusion
            fused = torch.cat([rgb_feat, flow_feat], dim=1)
            out = self.classifier(fused)
            return out

    # LOSO evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load CASME II dataset
    # Note: You need to implement the CASME2Dataset class or use existing loader
    print("Loading CASME II dataset...")
    print("Note: Using apex frames only (OFF-ApexNet protocol)")

    # Placeholder for actual implementation
    # results = run_loso_cv(model_class=OFFApexNet, dataset=..., device=device)

    # Results template
    results = {
        'fold_accuracies': [],
        'mean_accuracy': 0.0,
        'std_accuracy': 0.0,
        'per_class_f1': {},
    }

    print("\n[PLACEHOLDER] Implement LOSO cross-validation with:")
    print("  - 24 folds (excluding sub13, sub22)")
    print("  - Apex frame extraction per sample")
    print("  - Optical flow computation (TV-L1)")
    print("  - VGG-16 two-stream architecture")
    print("  - Same hyperparameters as Censor for fair comparison")

    print("\nExpected output format:")
    print("  Mean Accuracy: XX.XX% ± YY.YY%")
    print("  Per-fold results saved to: results/offapexnet_loso_folds.csv")

    return results


# =============================================================================
# Experiment 2: rPPG Signal Quality Validation
# =============================================================================

def experiment_2_rppg_validation():
    """
    Validate rPPG signal quality via heart rate estimation.

    Methods:
    1. Extract rPPG signals from CASME II videos using CHROM
    2. Compute heart rate from frequency domain (FFT peak in 0.5-4 Hz)
    3. Compare against expected physiological range (60-100 BPM normal)

    Metrics:
    - Signal-to-Noise Ratio (SNR)
    - Peak detection success rate
    - Heart rate distribution (should be in physiological range)
    """
    print("=" * 60)
    print("Experiment 2: rPPG Signal Quality Validation")
    print("=" * 60)

    try:
        from model.preprocessing import rPPGExtractor
        from scipy import signal
        from scipy.fft import fft, fftfreq
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install scipy: pip install scipy")
        return None

    def compute_hr_from_rppg(rppg_signal, fps=200):
        """Compute heart rate from rPPG signal using FFT."""
        # Bandpass filter (0.5-4 Hz = 30-240 BPM)
        nyquist = fps / 2
        low = 0.5 / nyquist
        high = 4.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, rppg_signal)

        # FFT
        n = len(filtered)
        yf = fft(filtered)
        xf = fftfreq(n, 1/fps)

        # Find peak in valid range (0.5-4 Hz)
        mask = (xf >= 0.5) & (xf <= 4.0)
        power = np.abs(yf[mask])
        freqs = xf[mask]

        if len(power) == 0:
            return None, 0.0

        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]
        hr_bpm = peak_freq * 60  # Convert Hz to BPM

        # SNR: peak power / mean power
        snr = power[peak_idx] / (np.mean(power) + 1e-8)

        return hr_bpm, snr

    # Placeholder for actual implementation
    print("Loading CASME II videos for rPPG extraction...")
    print("Computing heart rate estimates...")

    # Results template
    results = {
        'num_samples': 0,
        'hr_estimates': [],
        'snr_values': [],
        'hr_mean': 0.0,
        'hr_std': 0.0,
        'snr_mean': 0.0,
        'valid_hr_rate': 0.0,  # % of samples with HR in 60-100 BPM
    }

    print("\n[PLACEHOLDER] Implement rPPG validation with:")
    print("  - Load CASME II video frames")
    print("  - Apply CHROM method (model/preprocessing.py:rPPGExtractor)")
    print("  - Compute HR via FFT peak detection")
    print("  - Report: mean HR, SNR, valid HR rate")

    print("\nExpected output format:")
    print("  Mean Heart Rate: XX.X ± XX.X BPM")
    print("  Mean SNR: X.XX dB")
    print("  Valid HR Rate: XX.X% (60-100 BPM range)")
    print("  Results saved to: results/rppg_validation.json")

    return results


# =============================================================================
# Experiment 3: Inference Latency Benchmark
# =============================================================================

def experiment_3_latency_benchmark():
    """
    Benchmark inference latency for deployment guidance.

    Metrics:
    - Forward pass time (ms/frame)
    - GPU memory usage (MB)
    - Throughput (frames/second)

    Tested configurations:
    1. Full model (68.35M params)
    2. Fast-only (12.85M params) - for resource-constrained deployment
    3. No-rPPG (57.90M params) - ablation
    """
    print("=" * 60)
    print("Experiment 3: Inference Latency Benchmark")
    print("=" * 60)

    try:
        from main import Censor
        import torchvision.transforms as transforms
    except ImportError as e:
        print(f"Import error: {e}")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    def benchmark_model(model, input_shape, num_runs=100, warmup=10):
        """Benchmark model inference latency."""
        model.eval()

        # Create dummy input
        video = torch.randn(*input_shape).to(device)
        flow = torch.randn(input_shape[0], 2, *input_shape[2:]).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(video, flow)

        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(video, flow)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        return {
            'mean_ms': latencies.mean(),
            'std_ms': latencies.std(),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'throughput': 1000 / latencies.mean(),  # frames/sec
        }

    # Test configurations
    configs = [
        ('Full Model', None),
        ('Fast-only', 'fast_only'),
        ('No-rPPG', 'no_rppg'),
    ]

    results = {}
    batch_size = 1
    input_shape = (batch_size, 3, 16, 224, 224)

    print(f"\nInput shape: {input_shape}")
    print(f"Number of runs: 100 (warmup: 10)")

    for name, ablation in configs:
        print(f"\nBenchmarking {name}...")

        # Create model
        try:
            model = Censor(num_classes=4, ablation=ablation)
            model = model.to(device)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {num_params / 1e6:.2f}M")

            # Benchmark
            metrics = benchmark_model(model, input_shape)
            results[name] = {
                'num_params': num_params,
                **metrics
            }

            # GPU memory
            if device.type == 'cuda':
                mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                results[name]['gpu_memory_mb'] = mem_mb
                print(f"  GPU Memory: {mem_mb:.1f} MB")

            print(f"  Latency: {metrics['mean_ms']:.2f} ± {metrics['std_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput']:.1f} frames/sec")

            # Cleanup
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        except Exception as e:
            print(f"  Error: {e}")
            results[name] = None

    # Summary
    print("\n" + "=" * 60)
    print("Latency Benchmark Summary")
    print("=" * 60)
    print(f"{'Config':<20} {'Params (M)':<12} {'Latency (ms)':<15} {'Throughput':<12}")
    print("-" * 60)
    for name, metrics in results.items():
        if metrics:
            print(f"{name:<20} {metrics['num_params']/1e6:<12.2f} "
                  f"{metrics['mean_ms']:<15.2f} {metrics['throughput']:<12.1f}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'latency_benchmark.txt', 'w') as f:
        f.write(f"Inference Latency Benchmark\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Input shape: {input_shape}\n\n")
        for name, metrics in results.items():
            if metrics:
                f.write(f"{name}:\n")
                for k, v in metrics.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

    print(f"\nResults saved to: {output_dir / 'latency_benchmark.txt'}")

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='AutoDL Experiments for Paper Revision')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='Experiment to run (1=OFF-ApexNet, 2=rPPG, 3=Latency)')
    args = parser.parse_args()

    print("Censor - AutoDL Experiments")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment: {args.exp}")
    print()

    if args.exp in ['1', 'all']:
        experiment_1_offapexnet_loso()
        print()

    if args.exp in ['2', 'all']:
        experiment_2_rppg_validation()
        print()

    if args.exp in ['3', 'all']:
        experiment_3_latency_benchmark()
        print()

    print("=" * 60)
    print("Experiments completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
