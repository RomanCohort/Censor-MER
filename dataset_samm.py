"""
Censor -- SAMM Dataset Loader
==============================
Loads SAMM (Spontaneous Micro-expression Corpus) dataset.

SAMM structure:
    SAMM/
    ├── 006/
    │   ├── 006_1_2/          -> subject_AU_emotion
    │   │   ├── 006_05562.jpg -> subject_frameNumber.jpg
    │   │   ├── 006_05563.jpg
    │   │   └── ...
    │   └── ...
    ├── 007/
    └── SAMM_Micro_FACS_Codes_v2.xlsx

Naming convention: {subject}_{AU}_{emotion}
  AU codes: 1-44 (FACS Action Units)
  Emotion codes: 1-7 (mapped below)

SAMM: 32 subjects, ~159 micro-expression samples
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


# SAMM emotion code mapping (from SAMM paper)
# Code -> (name, unified_4class_label)
SAMM_EMOTION_MAP = {
    1: ('Happiness', 0),
    2: ('Surprise', 1),
    3: ('Disgust', 2),
    4: ('Fear', 2),        # merge into disgust (4-class)
    5: ('Repression', 3),
    6: ('Sadness', 2),     # merge into disgust
    7: ('Anger', 2),       # merge into disgust
}

SAMM_EMOTION_NAMES = ['Happiness', 'Surprise', 'Disgust', 'Repression']


class SAMMDataset(Dataset):
    """
    Dataset for loading SAMM micro-expression frame sequences.

    SAMM uses {subject}_{AU}_{emotion} naming convention and has an Excel
    annotation file with onset/apex/offset frame numbers.
    """

    EMOTION_NAMES = SAMM_EMOTION_NAMES

    def __init__(self, data_root, split='train', T=None, H=None, W=None,
                 augment=None, temporal_jitter=None, val_ratio=0.2, seed=42,
                 face_align=True, loso_fold=None, loso_subjects=None):
        self.data_root = data_root
        self.split = split
        self.T = T or DATA_CONFIG['T']
        self.H = H or DATA_CONFIG['H']
        self.W = W or DATA_CONFIG['W']
        self.augment = augment if augment is not None else DATA_CONFIG['augment']
        self.temporal_jitter = temporal_jitter if temporal_jitter is not None else DATA_CONFIG['temporal_jitter']
        self.face_align = face_align

        # SAMM data is in SAMM/SAMM/ subdirectory
        self.samm_dir = os.path.join(data_root, 'SAMM') if os.path.exists(os.path.join(data_root, 'SAMM')) else data_root

        # Load labels from Excel
        self.samples = self._load_annotations()
        print(f"[SAMMDataset] Loaded {len(self.samples)} samples from {self.samm_dir}")

        # Split
        if loso_fold is not None:
            self._split_loso(loso_fold, loso_subjects)
        else:
            self._split_dataset(val_ratio, seed)

        # ImageNet normalization
        self.mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(3, 1, 1, 1)
        self.std = torch.tensor(DATA_CONFIG['normalize_std']).view(3, 1, 1, 1)

    def _load_annotations(self):
        """Load SAMM annotations from Excel file."""
        excel_path = None
        for candidate in [
            os.path.join(self.samm_dir, 'SAMM_Micro_FACS_Codes_v2.xlsx'),
            os.path.join(self.data_root, 'SAMM_Micro_FACS_Codes_v2.xlsx'),
        ]:
            if os.path.exists(candidate):
                excel_path = candidate
                break

        if excel_path is not None:
            samples = self._load_from_excel(excel_path)
            if len(samples) > 0:
                return samples
            print(f"[SAMMDataset] Excel parsing returned 0 samples, falling back to directory scan")

        return self._load_from_directory()

    def _load_from_excel(self, excel_path):
        """Load annotations from SAMM Excel file."""
        print(f"[SAMMDataset] Loading annotations from: {excel_path}")
        df = pd.read_excel(excel_path)

        samples = []
        for idx, row in df.iterrows():
            # Parse subject and filename from row
            # SAMM Excel columns vary; try common column names
            subject = None
            filename = None

            # Try different column name conventions
            for col in ['Subject', 'subject', 'Subject_ID', 'Participant']:
                if col in df.columns and pd.notna(row.get(col)):
                    subject = str(int(row[col])).zfill(3) if isinstance(row[col], (int, float)) else str(row[col])
                    break

            for col in ['Filename', 'filename', 'Video', 'Clip']:
                if col in df.columns and pd.notna(row.get(col)):
                    filename = str(row[col])
                    break

            # If no subject from Excel, try to parse from filename
            if filename and not subject:
                parts = filename.split('_')
                if len(parts) >= 1:
                    subject = parts[0].zfill(3)

            if not subject or not filename:
                continue

            # Parse emotion code from filename: {subject}_{AU}_{emotion}
            parts = filename.split('_')
            if len(parts) >= 3:
                try:
                    emotion_code = int(parts[-1])
                except ValueError:
                    continue
            else:
                continue

            # Map to unified 4-class
            if emotion_code not in SAMM_EMOTION_MAP:
                continue

            emotion_name, me_label = SAMM_EMOTION_MAP[emotion_code]

            # Check frame directory exists
            frame_dir = os.path.join(self.samm_dir, subject, filename)
            if not os.path.exists(frame_dir):
                continue

            frame_files = sorted([f for f in os.listdir(frame_dir)
                                  if f.endswith('.jpg') or f.endswith('.bmp')])
            num_frames = len(frame_files)
            if num_frames == 0:
                continue

            # Parse onset/apex/offset from Excel
            def safe_int(val):
                if pd.isna(val):
                    return 0
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    return 0

            onset = 0
            apex = num_frames // 2
            offset = num_frames - 1

            for col in ['Onset', 'OnsetFrame', 'Onset Frame']:
                if col in df.columns:
                    onset = safe_int(row.get(col))
                    break
            for col in ['Apex', 'ApexFrame', 'Apex Frame']:
                if col in df.columns:
                    apex = safe_int(row.get(col))
                    break
            for col in ['Offset', 'OffsetFrame', 'Offset Frame']:
                if col in df.columns:
                    offset = safe_int(row.get(col))
                    break

            # Parse AU labels
            au_str = ''
            for col in ['AU', 'ActionUnits', 'Action Units', 'FACS']:
                if col in df.columns and pd.notna(row.get(col)):
                    au_str = str(row[col])
                    break

            samples.append({
                'video_path': f"{subject}/{filename}",
                'subject': subject,
                'filename': filename,
                'me_label': me_label,
                'emotion': emotion_name,
                'onset': onset,
                'apex': apex if apex > 0 else num_frames // 2,
                'offset': offset if offset > 0 else num_frames - 1,
                'num_frames': num_frames,
                'action_units': au_str,
            })

        return pd.DataFrame(samples)

    def _load_from_directory(self):
        """
        Fallback: build sample list from directory structure.
        Filename format: {subject}_{AU}_{emotion}
        """
        samples = []

        for subject_dir in sorted(os.listdir(self.samm_dir)):
            subject_path = os.path.join(self.samm_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            for sample_dir in sorted(os.listdir(subject_path)):
                sample_path = os.path.join(subject_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue

                # Parse: {subject}_{AU}_{emotion}
                parts = sample_dir.split('_')
                if len(parts) < 3:
                    continue

                try:
                    emotion_code = int(parts[-1])
                except ValueError:
                    continue

                if emotion_code not in SAMM_EMOTION_MAP:
                    continue

                emotion_name, me_label = SAMM_EMOTION_MAP[emotion_code]

                frame_files = sorted([f for f in os.listdir(sample_path)
                                      if f.endswith('.jpg') or f.endswith('.bmp')])
                num_frames = len(frame_files)
                if num_frames == 0:
                    continue

                samples.append({
                    'video_path': f"{subject_dir}/{sample_dir}",
                    'subject': subject_dir,
                    'filename': sample_dir,
                    'me_label': me_label,
                    'emotion': emotion_name,
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

        print(f"[SAMMDataset] {self.split}: {len(self.samples)} samples")

    def _split_loso(self, fold, loso_subjects=None):
        """Leave-One-Subject-Out split."""
        if loso_subjects is not None:
            subjects = sorted(loso_subjects)
        else:
            subjects = sorted(self.samples['subject'].unique())

        if fold >= len(subjects):
            raise ValueError(f"LOSO fold {fold} >= {len(subjects)} subjects")

        val_subject = subjects[fold]
        print(f"[SAMMDataset] LOSO fold {fold}/{len(subjects)}: "
              f"val_subject={val_subject}")

        if self.split == 'train':
            self.samples = self.samples[self.samples['subject'] != val_subject]
        else:
            self.samples = self.samples[self.samples['subject'] == val_subject]

        print(f"[SAMMDataset] {self.split}: {len(self.samples)} samples "
              f"(subject {val_subject})")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, sample_path):
        """Load frame sequence from SAMM directory."""
        frame_dir = os.path.join(self.samm_dir, sample_path)

        frame_files = sorted([f for f in os.listdir(frame_dir)
                              if f.endswith('.jpg') or f.endswith('.bmp')])

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

        return np.stack(frames, axis=0)

    def _sample_frames(self, frames, onset=0, apex=0, offset=0):
        """Sample T frames with apex-centered strategy."""
        total = len(frames)
        T = self.T

        if total <= T:
            indices = np.arange(T) % total
            return frames[indices]

        apex_idx = min(total - 1, apex - 1) if apex > 0 else total // 2
        onset_idx = max(0, onset - 1) if onset > 0 else max(0, apex_idx - total // 4)
        offset_idx = min(total - 1, offset - 1) if offset > 0 else min(total - 1, apex_idx + total // 4)

        if apex_idx <= onset_idx:
            apex_idx = min(onset_idx + total // 4, total - 1)
        if offset_idx <= apex_idx:
            offset_idx = min(apex_idx + total // 4, total - 1)

        if self.split == 'train' and self.temporal_jitter:
            half_T = T // 2
            onset_apex_len = apex_idx - onset_idx
            if onset_apex_len >= half_T:
                first_indices = np.linspace(onset_idx, apex_idx, half_T).astype(int)
            else:
                start = max(0, apex_idx - half_T)
                first_indices = np.linspace(start, apex_idx, half_T).astype(int)

            remaining = T - half_T
            after_apex = total - apex_idx - 1
            if after_apex >= remaining:
                second_indices = np.linspace(apex_idx, min(apex_idx + after_apex, total - 1), remaining).astype(int)
            else:
                second_indices = np.linspace(apex_idx, total - 1, remaining).astype(int)

            indices = np.concatenate([first_indices, second_indices])
        else:
            n_before = int(T * 0.4)
            n_apex = max(1, T - 2 * n_before)
            n_after = T - n_before - n_apex

            before_indices = np.linspace(onset_idx, apex_idx, n_before + 1).astype(int)[:n_before]
            apex_indices = np.full(n_apex, apex_idx, dtype=int)
            after_indices = np.linspace(apex_idx, offset_idx, n_after + 1).astype(int)[1:n_after + 1]

            indices = np.concatenate([before_indices, apex_indices, after_indices])

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

        if random.random() < 0.5:
            frames_tensor = frames_tensor.flip(-1)

        if random.random() < 0.3:
            b = 1.0 + random.uniform(-0.1, 0.1)
            c = 1.0 + random.uniform(-0.1, 0.1)
            frames_tensor = frames_tensor * b
            mean = frames_tensor.mean(dim=(2, 3), keepdim=True)
            frames_tensor = (frames_tensor - mean) * c + mean
            frames_tensor = frames_tensor.clamp(0, 1)

        if random.random() < 0.2:
            gray = frames_tensor.mean(dim=0, keepdim=True).expand_as(frames_tensor)
            alpha = random.uniform(0.5, 1.0)
            frames_tensor = alpha * frames_tensor + (1 - alpha) * gray

        if random.random() < 0.3:
            T = frames_tensor.shape[1]
            if T > 4:
                crop_len = random.randint(T // 2, T)
                start = random.randint(0, T - crop_len)
                cropped = frames_tensor[:, start:start + crop_len, :, :]
                indices = torch.linspace(0, crop_len - 1, T).long()
                frames_tensor = cropped[:, indices, :, :]

        if random.random() < 0.15:
            C, T, H, W = frames_tensor.shape
            eh = random.randint(H // 8, H // 4)
            ew = random.randint(W // 8, W // 4)
            eh_start = random.randint(0, H - eh)
            ew_start = random.randint(0, W - ew)
            frames_tensor[:, :, eh_start:eh_start + eh, ew_start:ew_start + ew] = 0

        return frames_tensor

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]

        frames = self._load_frames(sample['video_path'])

        onset = int(sample.get('onset', 0)) if pd.notna(sample.get('onset', 0)) else 0
        apex = int(sample.get('apex', 0)) if pd.notna(sample.get('apex', 0)) else 0
        offset = int(sample.get('offset', 0)) if pd.notna(sample.get('offset', 0)) else 0

        frames = self._sample_frames(frames, onset=onset, apex=apex, offset=offset)
        frames = self._align_frames(frames)

        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        frames_tensor = (frames_tensor - self.mean) / self.std
        frames_tensor = self._augment_frames(frames_tensor)

        me_label = int(sample['me_label'])

        # Parse AU labels
        au_label = torch.zeros(28, dtype=torch.float32)
        au_str = str(sample.get('action_units', ''))
        if au_str and au_str != 'nan' and au_str != '':
            for part in au_str.replace(' ', '').split('+'):
                part = part.strip()
                if part.startswith('L') or part.startswith('R'):
                    part = part[1:]
                if part.startswith('AU'):
                    part = part[2:]
                try:
                    au_id = int(part)
                    if 1 <= au_id <= 28:
                        au_label[au_id - 1] = 1.0
                except ValueError:
                    pass

        return frames_tensor, me_label, au_label


def get_samm_dataloaders(data_root, batch_size=8, T=16, H=224, W=224,
                         num_workers=4, val_ratio=0.2, seed=42,
                         face_align=True, loso_fold=None):
    """Create train and validation dataloaders for SAMM."""
    train_dataset = SAMMDataset(
        data_root=data_root,
        split='train',
        T=T, H=H, W=W,
        augment=True,
        temporal_jitter=True,
        val_ratio=val_ratio,
        seed=seed,
        face_align=face_align,
        loso_fold=loso_fold,
    )

    val_dataset = SAMMDataset(
        data_root=data_root,
        split='val',
        T=T, H=H, W=W,
        augment=False,
        temporal_jitter=False,
        val_ratio=val_ratio,
        seed=seed,
        face_align=face_align,
        loso_fold=loso_fold,
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


def get_samm_subjects(data_root):
    """Get sorted list of SAMM subjects for LOSO."""
    samm_dir = os.path.join(data_root, 'SAMM') if os.path.exists(os.path.join(data_root, 'SAMM')) else data_root
    if not os.path.exists(samm_dir):
        return None
    subjects = sorted([d for d in os.listdir(samm_dir)
                       if os.path.isdir(os.path.join(samm_dir, d)) and d[0].isdigit()])
    return subjects


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/data/SAMM')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    print("Testing SAMMDataset...")

    train_loader, val_loader = get_samm_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    for batch_idx, (videos, me_labels, au_labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Videos: {videos.shape}, dtype={videos.dtype}")
        print(f"  ME labels: {me_labels.shape}")
        print(f"  AU labels: {au_labels.shape}")
        print(f"  ME label values: {me_labels.tolist()}")
        break

    print("\n[Success] SAMM dataset test passed!")
