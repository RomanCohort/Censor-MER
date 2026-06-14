"""
Pre-extracted Dataset for fast LOSO training.
Loads from preextracted.npz instead of JPEG frames -- eliminates CPU bottleneck.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path


class PreextractedDataset(Dataset):
    """Load pre-extracted frames from .npz file. Zero JPEG decoding overhead."""

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.frames = data['frames']    # (N, C, T, H, W) float32
        self.labels = data['labels']    # (N,) int64
        self.subjects = list(data['subjects'])  # (N,) str

        print(f"[PreextractedDataset] Loaded {len(self.labels)} samples from {npz_path}")
        print(f"  frames shape: {self.frames.shape}")
        print(f"  unique subjects: {len(set(self.subjects))}")
        print(f"  unique labels: {len(set(self.labels.tolist()))}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.frames[idx]).float()  # (C, T, H, W)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def get_subject(self, idx):
        return self.subjects[idx]


def build_loso_splits(dataset):
    """Build LOSO splits from PreextractedDataset."""
    from collections import defaultdict

    subjects = sorted(set(dataset.subjects))
    CASME2_EXCLUDED = ['sub13', 'sub22']

    # Exclude problematic subjects if present
    subjects = [s for s in subjects if s not in CASME2_EXCLUDED]

    subj_to_idx = defaultdict(list)
    for i in range(len(dataset)):
        subj = dataset.subjects[i]
        if subj in subjects:
            subj_to_idx[subj].append(i)

    splits = []
    for subj in subjects:
        test_idx = subj_to_idx[subj]
        train_idx = [i for s, idxs in subj_to_idx.items() if s != subj for i in idxs]
        splits.append((train_idx, test_idx, subj))

    return splits, subjects