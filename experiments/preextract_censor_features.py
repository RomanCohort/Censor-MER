"""
Pre-extract Censor backbone features for all samples.
After running this, exp7/exp8 can train fusion heads directly on cached
features without loading Censor model or doing JPEG decoding.

Usage:
  python experiments/preextract_censor_features.py --dataset casme2
  python experiments/preextract_censor_features.py --dataset samm
  python experiments/preextract_censor_features.py --dataset smic

Output:
  {data_root}/censor_features.npz
    - fast_features: (N, 512)
    - slow_features: (N, 768)
    - labels: (N,)
    - subjects: (N,)
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


def preextract_features(dataset_name, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load pre-extracted frames
    npz_path = Path(data_root) / 'preextracted.npz'
    if not npz_path.exists():
        print(f"[ERROR] No preextracted.npz at {npz_path}")
        print("  Run 'python experiments/preextract_frames.py --dataset {dataset_name}' first!")
        return

    from experiments.preextracted_dataset import PreextractedDataset, build_loso_splits
    ds = PreextractedDataset(str(npz_path))

    # Load Censor model
    from main import Censor
    print("Loading Censor model...")
    censor = Censor(pretrained_backbone=True, verbose=False).to(device)
    censor.eval()

    # Extract features
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    all_fast = []
    all_slow = []
    all_labels = []
    all_subjects = []

    print("Extracting Censor features...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x = batch['video']
                y = batch['label']

            # Ensure (B, C, T, H, W)
            if x.dim() == 5 and x.shape[-1] in (3, 6):
                x = x.permute(0, 4, 1, 2, 3).contiguous()
            if x.dim() == 5 and x.shape[1] > 3:
                x = x[:, :3]

            x = x.to(device)

            try:
                fast_feat = censor.extract_fast_features(x)  # (B, 512)
                slow_feat = censor.extract_slow_features(x)  # (B, 768)
            except AttributeError:
                # Fallback: use full forward and split
                out = censor(x)
                fast_feat = out.get('fast_features', torch.zeros(x.size(0), 512, device=device))
                slow_feat = out.get('slow_features', torch.zeros(x.size(0), 768, device=device))

            all_fast.append(fast_feat.cpu().numpy())
            all_slow.append(slow_feat.cpu().numpy())
            all_labels.extend(y.tolist() if torch.is_tensor(y) else list(y))

    # Get subjects
    all_subjects = ds.subjects[:len(all_labels)]

    fast_arr = np.concatenate(all_fast, axis=0)
    slow_arr = np.concatenate(all_slow, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)
    subjects_arr = np.array(all_subjects)

    # Save
    output_path = Path(data_root) / 'censor_features.npz'
    np.savez_compressed(
        output_path,
        fast_features=fast_arr,
        slow_features=slow_arr,
        labels=labels_arr,
        subjects=subjects_arr,
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved to: {output_path}")
    print(f"  fast_features: {fast_arr.shape}")
    print(f"  slow_features: {slow_arr.shape}")
    print(f"  labels: {labels_arr.shape}, unique: {len(set(labels_arr.tolist()))}")
    print(f"  subjects: {len(set(subjects_arr))} unique")
    print(f"  file size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['casme2', 'samm', 'smic'])
    args = parser.parse_args()

    data_root = DATA_PATHS[args.dataset]
    if not Path(data_root).exists():
        print(f"[ERROR] Data not found: {data_root}")
        return

    preextract_features(args.dataset, data_root)


if __name__ == '__main__':
    main()