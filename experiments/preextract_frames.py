"""
Pre-extract all CASME II frames to a single .npy memory-mapped file.
This eliminates per-epoch JPEG decoding overhead (the main CPU bottleneck).

Usage:
  python experiments/preextract_frames.py --dataset casme2
  python experiments/preextract_frames.py --dataset casme2 --include-others  # 5-class (249 samples)
  python experiments/preextract_frames.py --dataset samm
  python experiments/preextract_frames.py --dataset smic

Output:
  /root/autodl-tmp/data/CASME2/preextracted.npz  (or similar)
  Contains: frames (N, C, T, H, W), labels (N,), subjects (N,)
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_PATHS = {
    'casme2': '/root/autodl-tmp/data/CASME2',
    'smic':   '/root/SMIC_all_cropped',
    'samm':   '/root/data/SAMM/SAMM',
}


def preextract(dataset_name, data_root, include_others=False):
    print(f"Pre-extracting {dataset_name} from {data_root}")
    if include_others:
        print(f"  Mode: 5-class (including 'others')")

    if dataset_name == 'casme2':
        from dataset_frames import FrameSequenceDataset
        # Load ALL samples (no train/val split) -- LOSO handles splitting
        ds = FrameSequenceDataset(data_root, split='train', face_align=False,
                                  val_ratio=0.0, include_others=include_others)
    elif dataset_name == 'samm':
        from dataset_samm import SAMMDataset
        ds = SAMMDataset(data_root, face_align=False, val_ratio=0.0)
    elif dataset_name == 'smic':
        from dataset_smic import SMICDataset
        ds = SMICDataset(data_root, face_align=False, val_ratio=0.0)
    else:
        raise ValueError(dataset_name)

    # Get all samples (not just train split)
    if hasattr(ds, 'samples') and hasattr(ds.samples, 'iloc'):
        n_samples = len(ds.samples)
        subjects = [ds.samples.iloc[i]['subject'] for i in range(n_samples)]
        labels = [int(ds.samples.iloc[i]['me_label']) for i in range(n_samples)]
    else:
        n_samples = len(ds)
        subjects = ['unknown'] * n_samples
        labels = [0] * n_samples

    print(f"Total samples: {n_samples}")

    # Extract all frames
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    all_frames = []
    all_labels = []
    all_subjects = []
    failed = 0

    for idx, batch in enumerate(loader):
        try:
            if isinstance(batch, (list, tuple)):
                x = batch[0]  # (1, C, T, H, W)
                y = batch[1]
            elif isinstance(batch, dict):
                x = batch.get('video', batch[0])
                y = batch.get('label', batch[1])
            else:
                raise ValueError(f"Unknown batch type: {type(batch)}")

            # Ensure (C, T, H, W)
            if x.dim() == 5 and x.shape[0] == 1:
                x = x.squeeze(0)

            all_frames.append(x.numpy())
            all_labels.append(int(y) if torch.is_tensor(y) else int(y))
            all_subjects.append(subjects[idx] if idx < len(subjects) else 'unknown')
        except Exception as e:
            print(f"  Failed sample {idx}: {e}")
            failed += 1

    print(f"Extracted: {len(all_frames)} samples, {failed} failed")

    if len(all_frames) == 0:
        print("No samples extracted, aborting.")
        return

    # Stack and save
    frames = np.stack(all_frames, axis=0)  # (N, C, T, H, W)
    labels = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects)

    output_path = Path(data_root) / 'preextracted.npz'
    np.savez_compressed(
        output_path,
        frames=frames,
        labels=labels,
        subjects=subjects,
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved to: {output_path}")
    print(f"  frames: {frames.shape}, dtype={frames.dtype}")
    print(f"  labels: {labels.shape}")
    print(f"  subjects: {len(set(subjects))} unique")
    print(f"  file size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['casme2', 'samm', 'smic'])
    parser.add_argument('--include-others', action='store_true',
                        help='Include "others" class for 5-class classification (CASME2 only)')
    args = parser.parse_args()

    data_root = DATA_PATHS[args.dataset]
    if not Path(data_root).exists():
        print(f"[ERROR] Data not found: {data_root}")
        return

    preextract(args.dataset, data_root, include_others=args.include_others)


if __name__ == '__main__':
    main()