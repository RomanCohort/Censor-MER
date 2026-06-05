"""
Experiment 2: rPPG Signal Quality Validation (Full Implementation)
====================================================================
Extract rPPG from real CASME II videos and validate signal quality.

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

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 2: rPPG Signal Quality Validation")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Configuration
# =============================================================================

CASME2_PATH = '/root/autodl-tmp/data/CASME2'

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    import cv2
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install scipy opencv-python")
    sys.exit(1)


# =============================================================================
# rPPG Extraction (CHROM Method)
# =============================================================================

def extract_rppg_chrom(frames, fps=200):
    """
    Extract rPPG signal using CHROM method.

    Args:
        frames: list of (H, W, 3) RGB frames
        fps: frame rate

    Returns:
        rppg_signal: (T,) normalized chrominance signal
    """
    if len(frames) < 10:
        return None

    signals = []
    for frame in frames:
        # Convert to float
        rgb = frame.astype(np.float32) / 255.0

        # Average over face region (simplified - use center)
        h, w = rgb.shape[:2]
        roi = rgb[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]

        # Mean RGB
        r_mean = roi[:, :, 0].mean()
        g_mean = roi[:, :, 1].mean()
        b_mean = roi[:, :, 2].mean()

        signals.append([r_mean, g_mean, b_mean])

    signals = np.array(signals)  # (T, 3)

    # CHROM: normalized chrominance
    xs = signals[:, 0] - signals[:, 1]
    ys = signals[:, 0] + signals[:, 1] - 2 * signals[:, 2]

    # Normalize
    xs = xs / (np.std(xs) + 1e-8)
    ys = ys / (np.std(ys) + 1e-8)

    # Combine
    alpha = np.std(xs) / (np.std(ys) + 1e-8)
    rppg = xs - alpha * ys

    return rppg


def compute_hr_fft(rppg_signal, fps=200):
    """Compute heart rate from rPPG using FFT."""
    if rppg_signal is None or len(rppg_signal) < 40:
        return None, 0.0, 0.0

    # Bandpass filter (0.67-4 Hz = 40-240 BPM)
    nyquist = fps / 2
    low = 0.67 / nyquist
    high = min(4.0 / nyquist, 0.99)

    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, rppg_signal)
    except Exception:
        return None, 0.0, 0.0

    # FFT
    n = len(filtered)
    yf = fft(filtered)
    xf = fftfreq(n, 1/fps)

    # Find peak
    mask = (xf >= 0.67) & (xf <= 4.0) & (xf > 0)
    if not mask.any():
        return None, 0.0, 0.0

    power = np.abs(yf[mask])
    freqs = xf[mask]

    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    hr_bpm = peak_freq * 60

    # SNR
    snr_db = 10 * np.log10(power[peak_idx] / (np.mean(power) + 1e-8))

    # Quality score
    quality = np.std(filtered) / (np.mean(np.abs(filtered)) + 1e-8)

    return hr_bpm, snr_db, quality


# =============================================================================
# Video Processing
# =============================================================================

def process_casme2_videos(data_root, max_videos=50):
    """Process CASME II videos and extract rPPG."""

    video_dir = Path(data_root) / 'videos'
    if not video_dir.exists():
        video_dir = Path(data_root)

    video_files = list(video_dir.glob('*.avi')) + list(video_dir.glob('*.mp4'))

    if len(video_files) == 0:
        print(f"No videos found in {video_dir}")
        return None

    print(f"Found {len(video_files)} videos, processing {min(max_videos, len(video_files))}")

    results = []

    for i, video_path in enumerate(video_files[:max_videos]):
        print(f"\n[{i+1}/{min(max_videos, len(video_files))}] {video_path.name}")

        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 200
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Read frames
            frames = []
            while len(frames) < 100:  # Max 100 frames
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            if len(frames) < 20:
                print(f"  Too few frames: {len(frames)}")
                continue

            print(f"  Frames: {len(frames)}, FPS: {fps}")

            # Extract rPPG
            rppg = extract_rppg_chrom(frames, fps)

            if rppg is None:
                print(f"  rPPG extraction failed")
                continue

            # Compute HR
            hr, snr, quality = compute_hr_fft(rppg, fps)

            if hr is None:
                print(f"  HR estimation failed")
                continue

            # Check physiological range
            in_range = 40 <= hr <= 200

            print(f"  HR: {hr:.1f} BPM, SNR: {snr:.1f} dB, Quality: {quality:.3f}")
            print(f"  Physiological: {'YES' if in_range else 'NO'}")

            results.append({
                'video': video_path.name,
                'frames': len(frames),
                'fps': fps,
                'hr_bpm': hr,
                'snr_db': snr,
                'quality': quality,
                'in_physiological_range': in_range,
            })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    # Check data
    if not Path(CASME2_PATH).exists():
        print(f"\nError: CASME II not found at {CASME2_PATH}")
        print("\nRunning synthetic validation instead...")

        # Synthetic fallback
        np.random.seed(42)
        num_samples = 100

        hr_list = []
        snr_list = []

        for i in range(num_samples):
            true_hr = np.random.uniform(60, 100)
            t = np.arange(80) / 200
            rppg = np.sin(2 * np.pi * true_hr/60 * t) + np.random.randn(80) * 0.1
            hr, snr, _ = compute_hr_fft(rppg, 200)
            if hr is not None:
                hr_list.append(hr)
                snr_list.append(snr)

        results = {
            'mode': 'synthetic',
            'num_samples': len(hr_list),
            'hr_mean': np.mean(hr_list),
            'hr_std': np.std(hr_list),
            'snr_mean': np.mean(snr_list),
        }

    else:
        print(f"CASME II found: {CASME2_PATH}")
        video_results = process_casme2_videos(CASME2_PATH, max_videos=100)

        if video_results:
            hr_list = [r['hr_bpm'] for r in video_results]
            snr_list = [r['snr_db'] for r in video_results]
            in_range_count = sum(r['in_physiological_range'] for r in video_results)

            results = {
                'mode': 'real',
                'num_videos': len(video_results),
                'hr_mean': np.mean(hr_list),
                'hr_std': np.std(hr_list),
                'snr_mean': np.mean(snr_list),
                'snr_std': np.std(snr_list),
                'physiological_rate': in_range_count / len(video_results) * 100,
                'video_results': video_results,
            }
        else:
            results = {'mode': 'failed', 'error': 'No valid videos processed'}

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    print(f"Samples: {results.get('num_videos', results.get('num_samples', 0))}")
    print(f"Mean HR: {results['hr_mean']:.1f} ± {results['hr_std']:.1f} BPM")
    print(f"Mean SNR: {results['snr_mean']:.1f} dB")
    if 'physiological_rate' in results:
        print(f"Physiological range: {results['physiological_rate']:.1f}%")

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'exp2_rppg_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {output_file}")


if __name__ == '__main__':
    main()