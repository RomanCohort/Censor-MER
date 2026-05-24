"""
Censor -- Precompute & Cache Preprocessing
===========================================
Precomputes optical flow and rPPG for all CASME2 samples,
caching them to disk so training doesn't recompute every epoch.

Usage:
    python precompute_cache.py --data_root /root/autodl-tmp/data/CASME2
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from dataset_frames import FrameSequenceDataset
from config.defaults import DATA_CONFIG


def compute_optical_flow(frames):
    """
    Compute TV-L1 optical flow for frame sequence.

    Args:
        frames: (T, H, W, 3) RGB uint8

    Returns:
        flow: (2, T, H, W) float32, x+y channels
    """
    T, H, W, C = frames.shape
    flow_channels = np.zeros((2, T, H, W), dtype=np.float32)

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
        tau=0.25, lamda=0.15, theta=0.3
    )

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for t in range(1, T):
        curr_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        flow = tvl1.calc(prev_gray, curr_gray, None)
        if flow is not None:
            flow_channels[0, t] = flow[:, :, 0]  # x
            flow_channels[1, t] = flow[:, :, 1]  # y
        prev_gray = curr_gray

    return flow_channels


def compute_rppg(frames):
    """
    Simplified rPPG extraction using CHROM method.

    Args:
        frames: (T, H, W, 3) RGB uint8

    Returns:
        rppg: (3, T, H, W) float32
    """
    T, H, W, C = frames.shape
    # Normalize to [0, 1]
    frames_f = frames.astype(np.float32) / 255.0

    # Spatial average of face region (center crop)
    h_start, h_end = H // 4, 3 * H // 4
    w_start, w_end = W // 4, 3 * W // 4
    face_region = frames_f[:, h_start:h_end, w_start:w_end, :]

    # Average RGB over spatial dims -> (T, 3)
    rgb_mean = face_region.mean(axis=(1, 2))

    # CHROM: project to pulse signal
    xs = 3 * rgb_mean[:, 0] - 2 * rgb_mean[:, 1]
    ys = 1.5 * rgb_mean[:, 0] + rgb_mean[:, 1] - 1.5 * rgb_mean[:, 2]

    # Bandpass filter (simplified: moving average detrend)
    window = 5
    if T > window:
        xs_smooth = np.convolve(xs, np.ones(window)/window, mode='same')
        ys_smooth = np.convolve(ys, np.ones(window)/window, mode='same')
        xs = xs - xs_smooth
        ys = ys - ys_smooth

    # Normalize
    alpha = np.std(xs) / max(np.std(ys), 1e-8)
    pulse = xs - alpha * ys
    pulse = (pulse - pulse.mean()) / max(pulse.std(), 1e-8)

    # Broadcast pulse signal to spatial dims
    rppg = np.zeros((3, T, H, W), dtype=np.float32)
    for c in range(3):
        for t in range(T):
            rppg[c, t] = pulse[t] * 0.1  # Scale down as modulation

    # Add original RGB as base
    rppg += frames_f.permute(3, 0, 1, 2) if hasattr(frames_f, 'permute') else np.transpose(frames_f, (3, 0, 1, 2))

    return rppg


def compute_rppg_simple(frames):
    """
    Simplified rPPG: just use normalized RGB + temporal difference.

    Args:
        frames: (T, H, W, 3) RGB uint8

    Returns:
        rppg: (3, T, H, W) float32
    """
    frames_f = frames.astype(np.float32) / 255.0
    rppg = np.transpose(frames_f, (3, 0, 1, 2)).copy()  # (3, T, H, W)

    # Add temporal difference as pulse hint
    if frames_f.shape[0] > 1:
        temporal_diff = np.zeros_like(rppg)
        temporal_diff[:, 1:, :, :] = rppg[:, 1:, :, :] - rppg[:, :-1, :, :]
        rppg = rppg + 0.1 * temporal_diff

    return rppg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--H', type=int, default=224)
    parser.add_argument('--W', type=int, default=224)
    parser.add_argument('--skip_flow', action='store_true', help='Skip optical flow (slow)')
    args = parser.parse_args()

    cache_dir = os.path.join(args.data_root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Load dataset
    dataset = FrameSequenceDataset(
        data_root=args.data_root,
        split='train',
        T=args.T, H=args.H, W=args.W,
        augment=False,
        temporal_jitter=False,
    )

    # Also load val
    val_dataset = FrameSequenceDataset(
        data_root=args.data_root,
        split='val',
        T=args.T, H=args.H, W=args.W,
        augment=False,
        temporal_jitter=False,
    )

    all_samples = list(dataset.samples['video_path']) + list(val_dataset.samples['video_path'])
    all_samples = list(set(all_samples))  # deduplicate

    print(f"[Cache] Total unique samples: {len(all_samples)}")
    print(f"[Cache] Output directory: {cache_dir}")
    print(f"[Cache] Skip optical flow: {args.skip_flow}")

    for idx, sample_path in enumerate(tqdm(all_samples, desc="Caching")):
        cache_path = os.path.join(cache_dir, sample_path.replace('/', '_') + '.npz')

        if os.path.exists(cache_path):
            continue  # Already cached

        try:
            # Load raw frames
            frame_dir = os.path.join(dataset.cropped_dir, sample_path)
            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

            frames = []
            for f in frame_files:
                img = cv2.imread(os.path.join(frame_dir, f))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (args.W, args.H))
                    frames.append(img)

            if len(frames) == 0:
                continue

            frames = np.stack(frames, axis=0)  # (T_orig, H, W, 3)

            # Sample T frames (even sampling)
            T = args.T
            if len(frames) <= T:
                indices = np.arange(T) % len(frames)
            else:
                indices = np.linspace(0, len(frames) - 1, T).astype(int)
            frames = frames[indices]  # (T, H, W, 3)

            # Compute rPPG
            rppg = compute_rppg_simple(frames)  # (3, T, H, W)

            save_dict = {
                'rgb': np.transpose(frames, (3, 0, 1, 2)).astype(np.float32) / 255.0,  # (3, T, H, W)
                'rppg': rppg,
            }

            # Compute optical flow (slow)
            if not args.skip_flow:
                flow = compute_optical_flow(frames)  # (2, T, H, W)
                save_dict['flow'] = flow

            np.savez_compressed(cache_path, **save_dict)

        except Exception as e:
            print(f"\n[Error] {sample_path}: {e}")
            continue

    # Print stats
    cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]
    print(f"\n[Cache] Done! Cached {len(cached_files)} / {len(all_samples)} samples")
    print(f"[Cache] Directory: {cache_dir}")

    # Check size
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cached_files)
    print(f"[Cache] Total size: {total_size / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
