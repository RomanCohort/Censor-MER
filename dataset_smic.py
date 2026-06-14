"""
Censor -- SMIC Dataset Loader
==============================
Loads SMIC (Spontaneous Micro-expression Corpus) HS subset.

SMIC structure (no annotation file -- labels from directory names):
    SMIC_all_cropped/
    HS/
      s1/
        micro/
          positive/  -> s1_po_01/reg_image*.bmp  (happiness)
          negative/  -> s1_ne_01/reg_image*.bmp  (disgust/sadness)
          surprise/  -> s1_sur_01/reg_image*.bmp
        non_micro/
          s1_n1/reg_image*.bmp  (excluded by default)

SMIC-HS: 16 subjects, ~164 micro-expression samples, 3-class
  positive: ~51, negative: ~70, surprise: ~43

Emotion mapping to CASME2 5-class:
  positive -> happiness (0)
  negative -> disgust (2)
  surprise -> surprise (1)
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from pathlib import Path

from config.defaults import DATA_CONFIG
from dataset_frames import align_face_sequence


# SMIC emotion mapping (3-class -> CASME2 5-class)
SMIC_EMOTION_MAP = {
    'positive': 0,   # happiness
    'negative': 2,   # disgust (SMIC negative approx CASME2 disgust/sadness)
    'surprise': 1,   # surprise
}

SMIC_EMOTION_NAMES = ['positive', 'negative', 'surprise']


class SMICDataset(Dataset):
    """
    Dataset for loading SMIC-HS micro-expression frame sequences.

    Unlike CASME2, SMIC has no annotation file -- labels are inferred from
    directory names (positive/negative/surprise). SMIC uses .bmp frames
    instead of .jpg, and has no onset/apex/offset annotations.
    """

    EMOTION_NAMES = SMIC_EMOTION_NAMES

    def __init__(self, data_root, split='train', T=None, H=None, W=None,
                 augment=None, temporal_jitter=None, val_ratio=0.2, seed=42,
                 face_align=True, loso_fold=None, loso_subjects=None,
                 subset='HS', use_non_micro=False):
        self.data_root = data_root
        self.split = split
        self.T = T or DATA_CONFIG['T']
        self.H = H or DATA_CONFIG['H']
        self.W = W or DATA_CONFIG['W']
        self.augment = augment if augment is not None else DATA_CONFIG['augment']
        self.temporal_jitter = temporal_jitter if temporal_jitter is not None else DATA_CONFIG['temporal_jitter']
        self.face_align = face_align
        self.use_non_micro = use_non_micro

        self.subset_dir = os.path.join(data_root, subset)

        # Build sample list from directory structure
        self.samples = self._build_sample_list()
        print(f"[SMICDataset] Loaded {len(self.samples)} samples from {self.subset_dir}")

        # Split into train/val
        if loso_fold is not None:
            self._split_loso(loso_fold, loso_subjects)
        else:
            self._split_dataset(val_ratio, seed)

        # ImageNet normalization
        self.mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(3, 1, 1, 1)
        self.std = torch.tensor(DATA_CONFIG['normalize_std']).view(3, 1, 1, 1)

    def _build_sample_list(self):
        """
        Build sample list by scanning SMIC directory structure.

        SMIC labels are embedded in directory names:
            micro/positive/ -> label 0 (happiness)
            micro/negative/ -> label 2 (disgust)
            micro/surprise/ -> label 1 (surprise)
            non_micro/      -> label 4 (others) [optional]
        """
        samples = []

        for subject_dir in sorted(os.listdir(self.subset_dir)):
            subject_path = os.path.join(self.subset_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            # Process micro-expression samples
            micro_dir = os.path.join(subject_path, 'micro')
            if os.path.exists(micro_dir):
                for emotion_dir in sorted(os.listdir(micro_dir)):
                    emotion_path = os.path.join(micro_dir, emotion_dir)
                    if not os.path.isdir(emotion_path):
                        continue

                    me_label = SMIC_EMOTION_MAP.get(emotion_dir, 4)

                    for sample_dir in sorted(os.listdir(emotion_path)):
                        sample_path = os.path.join(emotion_path, sample_dir)
                        if not os.path.isdir(sample_path):
                            continue

                        frame_files = sorted([f for f in os.listdir(sample_path)
                                              if f.endswith('.bmp') or f.endswith('.jpg')])
                        num_frames = len(frame_files)

                        if num_frames == 0:
                            continue

                        samples.append({
                            'video_path': f"{subject_dir}/micro/{emotion_dir}/{sample_dir}",
                            'subject': subject_dir,
                            'filename': sample_dir,
                            'me_label': me_label,
                            'emotion': emotion_dir,
                            'onset': 0,
                            'apex': num_frames // 2,  # estimate apex at middle
                            'offset': num_frames - 1,
                            'num_frames': num_frames,
                            'action_units': '',
                        })

            # Optionally include non-micro-expression samples
            if self.use_non_micro:
                non_micro_dir = os.path.join(subject_path, 'non_micro')
                if os.path.exists(non_micro_dir):
                    for sample_dir in sorted(os.listdir(non_micro_dir)):
                        sample_path = os.path.join(non_micro_dir, sample_dir)
                        if not os.path.isdir(sample_path):
                            continue

                        frame_files = sorted([f for f in os.listdir(sample_path)
                                              if f.endswith('.bmp') or f.endswith('.jpg')])
                        num_frames = len(frame_files)

                        if num_frames == 0:
                            continue

                        samples.append({
                            'video_path': f"{subject_dir}/non_micro/{sample_dir}",
                            'subject': subject_dir,
                            'filename': sample_dir,
                            'me_label': 4,
                            'emotion': 'non_micro',
                            'onset': 0,
                            'apex': num_frames // 2,
                            'offset': num_frames - 1,
                            'num_frames': num_frames,
                            'action_units': '',
                        })

        return pd.DataFrame(samples)

    def _split_dataset(self, val_ratio, seed):
        """Split dataset into train and validation sets."""
        random.seed(seed)
        subjects = sorted(self.samples['subject'].unique())
        val_subjects = set(random.sample(subjects, int(len(subjects) * val_ratio)))

        if self.split == 'train':
            self.samples = self.samples[~self.samples['subject'].isin(val_subjects)]
        else:
            self.samples = self.samples[self.samples['subject'].isin(val_subjects)]

        print(f"[SMICDataset] {self.split}: {len(self.samples)} samples")

    def _split_loso(self, fold, loso_subjects=None):
        """Leave-One-Subject-Out split."""
        if loso_subjects is not None:
            subjects = sorted(loso_subjects)
        else:
            subjects = sorted(self.samples['subject'].unique())

        if fold >= len(subjects):
            raise ValueError(f"LOSO fold {fold} >= {len(subjects)} subjects")

        val_subject = subjects[fold]
        print(f"[SMICDataset] LOSO fold {fold}/{len(subjects)}: "
              f"val_subject={val_subject}")

        if self.split == 'train':
            self.samples = self.samples[self.samples['subject'] != val_subject]
        else:
            self.samples = self.samples[self.samples['subject'] == val_subject]

        print(f"[SMICDataset] {self.split}: {len(self.samples)} samples "
              f"(subject {val_subject})")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, sample_path):
        """
        Load frame sequence from SMIC directory.

        SMIC uses .bmp format (uncompressed).

        Args:
            sample_path (str): Relative path like "s1/positive/s1_po_01"

        Returns:
            frames (np.ndarray): (T_orig, H, W, 3) RGB uint8
        """
        frame_dir = os.path.join(self.subset_dir, sample_path)

        # Get sorted frame files (SMIC uses .bmp)
        frame_files = sorted([f for f in os.listdir(frame_dir)
                              if f.endswith('.bmp') or f.endswith('.jpg')])

        if len(frame_files) == 0:
            raise RuntimeError(f"No frames found in {frame_dir}")

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(frame_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        if len(frames) == 0:
            raise RuntimeError(f"Could not read any frames from {frame_dir}")

        return np.stack(frames, axis=0)  # (T_orig, H, W, 3)

    def _sample_frames(self, frames, onset=0, apex=0, offset=0):
        """
        Sample T frames with apex-centered strategy.

        SMIC has no onset/apex/offset annotations, so we estimate:
            apex ~ middle frame (peak of micro-expression)
        """
        total = len(frames)
        T = self.T

        if total <= T:
            indices = np.arange(T) % total
            return frames[indices]

        # Use estimated apex (middle frame)
        apex_idx = min(total - 1, apex) if apex > 0 else total // 2
        onset_idx = max(0, apex_idx - total // 4)
        offset_idx = min(total - 1, apex_idx + total // 4)

        if self.split == 'train' and self.temporal_jitter:
            # Random window around estimated apex
            half_T = T // 2
            start = max(0, apex_idx - half_T)
            first_indices = np.linspace(start, apex_idx, half_T).astype(int)

            remaining = T - half_T
            end = min(total - 1, apex_idx + remaining)
            second_indices = np.linspace(apex_idx, end, remaining).astype(int)

            indices = np.concatenate([first_indices, second_indices])
        else:
            # Deterministic: 40% before, 20% apex, 40% after
            n_before = int(T * 0.4)
            n_apex = max(1, T - 2 * n_before)
            n_after = T - n_before - n_apex

            before_indices = np.linspace(onset_idx, apex_idx, n_before + 1).astype(int)[:n_before]
            apex_indices = np.full(n_apex, apex_idx, dtype=int)
            after_indices = np.linspace(apex_idx, offset_idx, n_after + 1).astype(int)[1:n_after + 1]

            indices = np.concatenate([before_indices, apex_indices, after_indices])

        # Ensure exactly T indices
        if len(indices) < T:
            indices = np.concatenate([indices, np.full(T - len(indices), indices[-1], dtype=int)])
        elif len(indices) > T:
            indices = indices[:T]

        indices = np.clip(indices, 0, total - 1)
        return frames[indices]

    def _align_frames(self, frames):
        """Apply gaze stabilization reflex (face alignment) + resize."""
        if not self.face_align:
            resized = []
            for frame in frames:
                resized.append(cv2.resize(frame, (self.W, self.H)))
            return np.stack(resized, axis=0)

        aligned = align_face_sequence(frames, target_size=(self.W, self.H))
        return aligned

    def _augment_frames(self, frames_tensor):
        """Apply augmentation to video tensor."""
        if not self.augment or self.split != 'train':
            return frames_tensor

        # Horizontal flip
        if random.random() < 0.5:
            frames_tensor = frames_tensor.flip(-1)

        # Color jitter
        if random.random() < 0.3:
            b = 1.0 + random.uniform(-0.1, 0.1)
            c = 1.0 + random.uniform(-0.1, 0.1)
            frames_tensor = frames_tensor * b
            mean = frames_tensor.mean(dim=(2, 3), keepdim=True)
            frames_tensor = (frames_tensor - mean) * c + mean
            frames_tensor = frames_tensor.clamp(0, 1)

        # Saturation jitter
        if random.random() < 0.2:
            gray = frames_tensor.mean(dim=0, keepdim=True).expand_as(frames_tensor)
            alpha = random.uniform(0.5, 1.0)
            frames_tensor = alpha * frames_tensor + (1 - alpha) * gray

        # Random temporal crop
        if random.random() < 0.3:
            T = frames_tensor.shape[1]
            if T > 4:
                crop_len = random.randint(T // 2, T)
                start = random.randint(0, T - crop_len)
                cropped = frames_tensor[:, start:start + crop_len, :, :]
                indices = torch.linspace(0, crop_len - 1, T).long()
                frames_tensor = cropped[:, indices, :, :]

        # Random erasing
        if random.random() < 0.15:
            C, T, H, W = frames_tensor.shape
            eh = random.randint(H // 8, H // 4)
            ew = random.randint(W // 8, W // 4)
            eh_start = random.randint(0, H - eh)
            ew_start = random.randint(0, W - ew)
            frames_tensor[:, :, eh_start:eh_start + eh, ew_start:ew_start + ew] = 0

        return frames_tensor

    def __getitem__(self, idx):
        """
        Get a sample.

        Returns:
            video (torch.Tensor): (C, T, H, W) normalized
            me_label (int): Emotion label (mapped to CASME2 5-class)
            au_label (torch.Tensor): AU labels (zeros for SMIC, no AU annotations)
        """
        sample = self.samples.iloc[idx]

        # Load frames
        frames = self._load_frames(sample['video_path'])

        # Get estimated onset/apex/offset
        onset = int(sample.get('onset', 0)) if pd.notna(sample.get('onset', 0)) else 0
        apex = int(sample.get('apex', 0)) if pd.notna(sample.get('apex', 0)) else 0
        offset = int(sample.get('offset', 0)) if pd.notna(sample.get('offset', 0)) else 0

        # Sample T frames with apex-centered strategy
        frames = self._sample_frames(frames, onset=onset, apex=apex, offset=offset)

        # Align + resize
        frames = self._align_frames(frames)

        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

        # Normalize
        frames_tensor = (frames_tensor - self.mean) / self.std

        # Augment
        frames_tensor = self._augment_frames(frames_tensor)

        # Labels
        me_label = int(sample['me_label'])
        au_label = torch.zeros(28, dtype=torch.float32)  # SMIC has no AU annotations

        return frames_tensor, me_label, au_label


def get_smic_dataloaders(data_root, batch_size=8, T=16, H=224, W=224,
                         num_workers=4, val_ratio=0.2, seed=42,
                         face_align=True, loso_fold=None,
                         subset='HS', use_non_micro=False):
    """Create train and validation dataloaders for SMIC-HS."""
    train_dataset = SMICDataset(
        data_root=data_root,
        split='train',
        T=T, H=H, W=W,
        augment=True,
        temporal_jitter=True,
        val_ratio=val_ratio,
        seed=seed,
        face_align=face_align,
        loso_fold=loso_fold,
        subset=subset,
        use_non_micro=use_non_micro,
    )

    val_dataset = SMICDataset(
        data_root=data_root,
        split='val',
        T=T, H=H, W=W,
        augment=False,
        temporal_jitter=False,
        val_ratio=val_ratio,
        seed=seed,
        face_align=face_align,
        loso_fold=loso_fold,
        subset=subset,
        use_non_micro=use_non_micro,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_smic_subjects(data_root, subset='HS'):
    """Get sorted list of SMIC subjects for LOSO."""
    subset_dir = os.path.join(data_root, subset)
    if not os.path.exists(subset_dir):
        return None
    subjects = sorted([d for d in os.listdir(subset_dir)
                       if os.path.isdir(os.path.join(subset_dir, d))])
    return subjects


if __name__ == '__main__':
    # Test SMIC dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='D:/360Downloads/SMIC_all_cropped')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    print("Testing SMICDataset...")

    train_loader, val_loader = get_smic_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    for batch_idx, (videos, me_labels, au_labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Videos: {videos.shape}, dtype={videos.dtype}")
        print(f"  ME labels: {me_labels.shape}")
        print(f"  AU labels: {au_labels.shape}")
        print(f"  ME label values: {me_labels.tolist()}")
        break

    print("\n[Success] SMIC dataset test passed!")
