"""
Experiment 2: rPPG Signal Quality Validation
=============================================
Validate rPPG signal quality using existing rPPGExtractor from Censor.

Uses:
    - model.preprocessing.rPPGExtractor for signal extraction
    - CASME II videos for real data testing

Data path: /root/autodl-tmp/data/CASME2

Usage:
    python experiments/exp2_rppg_validation.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 2: rPPG Signal Quality Validation")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Configuration
# =============================================================================

CASME2_PATH = '/root/autodl-tmp/data/CASME2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# Import existing rPPG extractor
# =============================================================================

try:
    from model.preprocessing import rPPGExtractor
    print("Using rPPGExtractor from model/preprocessing.py")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
except ImportError:
    print("Warning: scipy not installed, using synthetic test only")
    signal = None
    fft = None


# =============================================================================
# HR Estimation from rPPG signal
# =============================================================================

def estimate_hr_from_signal(rppg_signal, fps=200):
    """
    Estimate heart rate from extracted rPPG signal.

    Args:
        rppg_signal: (T,) or (3, T) tensor/array
        fps: frame rate

    Returns:
        hr_bpm: estimated heart rate
        snr_db: signal-to-noise ratio
    """
    if signal is None:
        return None, 0.0

    # Convert to numpy
    if torch.is_tensor(rppg_signal):
        rppg_signal = rppg_signal.detach().cpu().numpy()

    # Take mean across channels if needed
    if rppg_signal.ndim > 1:
        rppg_signal = rppg_signal.mean(axis=0)

    if len(rppg_signal) < 40:
        return None, 0.0

    # Bandpass filter
    nyquist = fps / 2
    low = max(0.67 / nyquist, 0.01)  # 40 BPM
    high = min(4.0 / nyquist, 0.99)  # 240 BPM

    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, rppg_signal)
    except Exception:
        return None, 0.0

    # FFT
    n = len(filtered)
    yf = fft(filtered)
    xf = fftfreq(n, 1/fps)

    # Find peak in valid range
    mask = (xf >= 0.67) & (xf <= 4.0) & (xf > 0)
    if not mask.any():
        return None, 0.0

    power = np.abs(yf[mask])
    freqs = xf[mask]

    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    hr_bpm = peak_freq * 60

    # SNR
    snr_db = 10 * np.log10(power[peak_idx] / (np.mean(power) + 1e-8))

    return hr_bpm, snr_db


# =============================================================================
# Test with real data
# =============================================================================

def test_with_casme2(data_path, max_videos=20):
    """Test rPPG extraction on CASME II videos."""

    import cv2

    video_dir = Path(data_path) / 'cropped'
    if not video_dir.exists():
        video_dir = Path(data_path)

    # Find video files
    video_files = []
    for ext in ['*.avi', '*.mp4', '*.mkv']:
        video_files.extend(video_dir.rglob(ext))

    if not video_files:
        print(f"No videos found in {video_dir}")
        return None

    print(f"Found {len(video_files)} videos, testing {min(max_videos, len(video_files))}")

    # Create rPPG extractor
    rppg_extractor = rPPGExtractor().to(device)
    rppg_extractor.eval()

    results = []

    for i, video_path in enumerate(video_files[:max_videos]):
        print(f"\n[{i+1}] {video_path.name}")

        try:
            # Load video
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 200

            frames = []
            while len(frames) < 100:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            cap.release()

            if len(frames) < 20:
                print(f"  Too few frames: {len(frames)}")
                continue

            print(f"  Frames: {len(frames)}, FPS: {fps:.1f}")

            # Convert to tensor (B, C, T, H, W)
            video_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
            video_tensor = video_tensor.to(device)

            # Extract rPPG
            with torch.no_grad():
                rppg_heatmap = rppg_extractor(video_tensor)  # (1, 3, T, H, W)

            # Get signal (average over spatial dims)
            rppg_signal = rppg_heatmap[0].mean(dim=-1).mean(dim=-1)  # (3, T)
            rppg_signal = rppg_signal.mean(dim=0)  # (T,)

            # Estimate HR
            hr, snr = estimate_hr_from_signal(rppg_signal, fps)

            if hr is None:
                print(f"  HR estimation failed")
                continue

            in_range = 40 <= hr <= 200
            print(f"  HR: {hr:.1f} BPM, SNR: {snr:.1f} dB")
            print(f"  Physiological range: {'YES' if in_range else 'NO'}")

            results.append({
                'video': video_path.name,
                'frames': len(frames),
                'fps': fps,
                'hr_bpm': hr,
                'snr_db': snr,
                'in_range': in_range,
            })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return results


# =============================================================================
# Synthetic test (fallback)
# =============================================================================

def synthetic_test(num_samples=100):
    """Synthetic rPPG test when no real data available."""

    print("\nRunning synthetic test...")

    rppg_extractor = rPPGExtractor().to(device)
    rppg_extractor.eval()

    results = []

    for i in range(num_samples):
        true_hr = np.random.uniform(60, 100)
        t = np.arange(80) / 200
        cardiac = np.sin(2 * np.pi * true_hr/60 * t) * 0.1

        # Create synthetic video
        video = torch.randn(1, 3, 80, 224, 224).to(device) * 0.1
        video += torch.from_numpy(cardiac).float().view(1, 1, -1, 1, 1).to(device) * 0.5

        with torch.no_grad():
            rppg = rppg_extractor(video)

        rppg_signal = rppg[0].mean(dim=-1).mean(dim=-1).mean(dim=0)
        hr, snr = estimate_hr_from_signal(rppg_signal, 200)

        if hr is not None:
            results.append({
                'true_hr': true_hr,
                'est_hr': hr,
                'error': abs(hr - true_hr),
            })

    if results:
        errors = [r['error'] for r in results]
        print(f"MAE: {np.mean(errors):.1f} BPM")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    # Check data
    if Path(CASME2_PATH).exists():
        print(f"CASME II found: {CASME2_PATH}")
        results = test_with_casme2(CASME2_PATH, max_videos=50)

        if results:
            hr_list = [r['hr_bpm'] for r in results]
            snr_list = [r['snr_db'] for r in results]
            in_range = sum(r['in_range'] for r in results)

            summary = {
                'mode': 'real',
                'num_videos': len(results),
                'hr_mean': np.mean(hr_list),
                'hr_std': np.std(hr_list),
                'snr_mean': np.mean(snr_list),
                'physiological_rate': in_range / len(results) * 100,
            }
        else:
            print("No valid results from real data, running synthetic test")
            results = synthetic_test()
            summary = {'mode': 'synthetic', 'num_samples': len(results)}
    else:
        print(f"CASME II not found at {CASME2_PATH}")
        print("Running synthetic test...")
        results = synthetic_test()
        summary = {'mode': 'synthetic', 'num_samples': len(results)}

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'exp2_rppg_validation.json'
    with open(output_file, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)

    print(f"\nSaved to: {output_file}")


if __name__ == '__main__':
    main()