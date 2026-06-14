"""
Censor Feature Dataset for fast fusion head training.
Loads cached Censor backbone features from censor_features.npz.
Zero JPEG decoding, zero Censor model forward pass overhead.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CensorFeatureDataset(Dataset):
    """Load pre-extracted Censor features from .npz file.

    __getitem__ returns (fast_feat, slow_feat, label) — same format
    that exp7 fusion heads expect.
    """

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.fast_features = data['fast_features']   # (N, 512)
        self.slow_features = data['slow_features']   # (N, 768)
        self.labels = data['labels']                  # (N,) int64
        self.subjects = list(data['subjects'])        # (N,) str

        # DataFrame for get_loso_splits() compatibility
        self.samples = pd.DataFrame({
            'subject': self.subjects,
            'me_label': self.labels.tolist(),
        })

        print(f"[CensorFeatureDataset] Loaded {len(self.labels)} samples from {npz_path}")
        print(f"  fast_features: {self.fast_features.shape}")
        print(f"  slow_features: {self.slow_features.shape}")
        print(f"  unique subjects: {len(set(self.subjects))}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fast = torch.from_numpy(self.fast_features[idx]).float()   # (512,)
        slow = torch.from_numpy(self.slow_features[idx]).float()   # (768,)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return fast, slow, label


def build_loso_splits(dataset):
    """Build LOSO splits from CensorFeatureDataset."""
    from collections import defaultdict

    subjects = sorted(set(dataset.subjects))
    CASME2_EXCLUDED = ['sub13', 'sub22']
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