"""
Experiment 2: rPPG Signal Quality Validation
=============================================
Validate rPPG signal quality via heart rate estimation.

Usage:
    python experiments/exp2_rppg_validation.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 2: rPPG Signal Quality Validation")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install scipy: pip install scipy")
    sys.exit(1)

def compute_hr_from_rppg(rppg_signal, fps=200):
    """Compute heart rate from rPPG signal using FFT."""
    if len(rppg_signal) < 40:
        return None, 0.0

    nyquist = fps / 2
    low = 0.5 / nyquist
    high = min(4.0 / nyquist, 0.99)

    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, rppg_signal)
    except Exception:
        return None, 0.0

    n = len(filtered)
    yf = fft(filtered)
    xf = fftfreq(n, 1/fps)

    mask = (xf >= 0.5) & (xf <= 4.0) & (xf > 0)
    if not mask.any():
        return None, 0.0

    power = np.abs(yf[mask])
    freqs = xf[mask]

    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    hr_bpm = peak_freq * 60

    snr = power[peak_idx] / (np.mean(power) + 1e-8)

    return hr_bpm, snr


print("\nSimulating rPPG extraction on CASME II-like conditions:")
print("  - fps: 200")
print("  - duration: 80 frames (400ms window)")
print("  - Expected HR: 60-100 BPM")

# Generate synthetic rPPG signals
np.random.seed(42)
num_samples = 100
fps = 200
duration_frames = 80

hr_estimates = []
snr_values = []
errors = []

print(f"\nTesting on {num_samples} synthetic signals...")
for i in range(num_samples):
    true_hr = np.random.uniform(60, 100)
    cardiac_freq = true_hr / 60

    t = np.arange(duration_frames) / fps
    cardiac = np.sin(2 * np.pi * cardiac_freq * t) * 0.5
    noise = np.random.randn(duration_frames) * 0.1
    motion = np.sin(2 * np.pi * 0.2 * t) * 0.05

    rppg_signal = cardiac + noise + motion

    hr, snr = compute_hr_from_rppg(rppg_signal, fps)
    if hr is not None:
        hr_estimates.append(hr)
        snr_values.append(snr)
        errors.append(abs(hr - true_hr))

# Results
hr_mean = np.mean(hr_estimates) if hr_estimates else 0
hr_std = np.std(hr_estimates) if hr_estimates else 0
snr_mean = np.mean(snr_values) if snr_values else 0
error_mean = np.mean(errors) if errors else 0
valid_rate = len(hr_estimates) / num_samples * 100

print("\n" + "=" * 60)
print("Results:")
print(f"  Valid samples: {len(hr_estimates)}/{num_samples} ({valid_rate:.1f}%)")
print(f"  Mean HR: {hr_mean:.1f} ± {hr_std:.1f} BPM")
print(f"  Mean SNR: {snr_mean:.2f}")
print(f"  Estimation error: {error_mean:.1f} BPM (MAE)")
print("=" * 60)

# Important note
print("\n[IMPORTANT NOTE]")
print("ME duration (40-200ms = 8-40 frames at 200fps) is shorter than")
print("rPPG window (40+ frames for stable HR).")
print("This means rPPG contribution may reflect chromatic features,")
print("not validated HR signals.")

# Save results
output_dir = Path(__file__).parent.parent / 'results'
output_dir.mkdir(exist_ok=True)

with open(output_dir / 'exp2_rppg_validation.txt', 'w') as f:
    f.write(f"rPPG Signal Quality Validation\n")
    f.write(f"Date: {datetime.now()}\n\n")
    f.write(f"Simulated conditions:\n")
    f.write(f"  fps: {fps}\n")
    f.write(f"  window: {duration_frames} frames ({duration_frames/fps*1000:.0f}ms)\n")
    f.write(f"  samples: {num_samples}\n\n")
    f.write(f"Results:\n")
    f.write(f"  valid_rate: {valid_rate:.1f}%\n")
    f.write(f"  hr_mean: {hr_mean:.1f} BPM\n")
    f.write(f"  hr_std: {hr_std:.1f} BPM\n")
    f.write(f"  snr_mean: {snr_mean:.2f}\n")
    f.write(f"  error_mae: {error_mean:.1f} BPM\n\n")
    f.write(f"Note: ME duration < rPPG window, contribution may be chromatic features.\n")

print(f"\nSaved to: {output_dir / 'exp2_rppg_validation.txt'}")