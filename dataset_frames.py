"""
Censor -- Frame Sequence Dataset Loader

Supports loading micro-expression samples from preprocessed frame sequences.
Includes gaze stabilization reflex (face alignment) and LOSO cross-validation.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from pathlib import Path

from config.defaults import DATA_CONFIG


# =============================================================================
# Gaze Stabilization Reflex -- Face Alignment
# =============================================================================
# In biology, the vestibulo-ocular reflex stabilizes gaze during head movement.
# Analogously, face alignment stabilizes the facial coordinate frame across
# subjects, reducing inter-subject variation so the model focuses on
# expression dynamics rather than identity.

# Canonical eye positions for 224x224 alignment target
# Based on average face proportions: eyes at ~30% from top, ~30%/70% horizontal
REFERENCE_LANDMARKS = np.array([
    [0.315, 0.36],   # Left eye center
    [0.685, 0.36],   # Right eye center
], dtype=np.float32)


def _detect_eye_centers(frame):
    """
    Detect eye centers using mediapipe Face Mesh.

    Args:
        frame: (H, W, 3) RGB uint8

    Returns:
        eye_centers: (2, 2) [left_eye, right_eye] normalized coords, or None
    """
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3,
        ) as face_mesh:
            results = face_mesh.process(frame)
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            # Left eye: landmark 33 (inner) and 133 (outer) -> center
            left_x = (landmarks.landmark[33].x + landmarks.landmark[133].x) / 2
            left_y = (landmarks.landmark[33].y + landmarks.landmark[133].y) / 2
            # Right eye: landmark 362 (inner) and 263 (outer) -> center
            right_x = (landmarks.landmark[362].x + landmarks.landmark[263].x) / 2
            right_y = (landmarks.landmark[362].y + landmarks.landmark[263].y) / 2

            return np.array([
                [left_x, left_y],
                [right_x, right_y],
            ], dtype=np.float32)
    except ImportError:
        return None
    except Exception:
        return None


def _compute_alignment_transform(src_points, dst_points):
    """
    Compute similarity transform (rotation + scale + translation) to align
    src_points to dst_points using least-squares.

    Args:
        src_points: (N, 2) source points
        dst_points: (N, 2) destination points

    Returns:
        M: (2, 3) affine transformation matrix
    """
    return cv2.estimateAffinePartial2D(src_points, dst_points)[0]


def align_face(frame, target_size=(224, 224)):
    """
    Align a face frame using gaze stabilization reflex.

    Detects eye positions and applies a similarity transform to warp the face
    into a canonical coordinate frame. This mimics the vestibulo-ocular reflex
    that stabilizes retinal images during head movement.

    Args:
        frame: (H, W, 3) RGB uint8
        target_size: (W, H) output size

    Returns:
        aligned: (H, W, 3) RGB uint8, aligned face
    """
    h, w = frame.shape[:2]
    eye_centers = _detect_eye_centers(frame)

    if eye_centers is None:
        # Fallback: no alignment possible, just resize
        return cv2.resize(frame, target_size)

    # Scale reference landmarks to target pixel coordinates
    dst_pixels = REFERENCE_LANDMARKS.copy()
    dst_pixels[:, 0] *= target_size[0]  # x * W
    dst_pixels[:, 1] *= target_size[1]  # y * H

    # Source eye centers in pixel coordinates
    src_pixels = eye_centers.copy()
    src_pixels[:, 0] *= w
    src_pixels[:, 1] *= h

    # Compute similarity transform
    M = _compute_alignment_transform(src_pixels, dst_pixels)

    if M is None:
        return cv2.resize(frame, target_size)

    # Apply warp
    aligned = cv2.warpAffine(
        frame, M, target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def align_face_sequence(frames, target_size=(224, 224)):
    """
    Align a sequence of face frames using the first frame's eye positions.

    Using the first frame's transform for the entire sequence ensures temporal
    consistency -- analogous to how the vestibulo-ocular reflex maintains a
    stable gaze reference across rapid micro-expression changes.

    Args:
        frames: (T, H, W, 3) RGB uint8
        target_size: (W, H) output size

    Returns:
        aligned: (T, H, W, 3) RGB uint8
    """
    if len(frames) == 0:
        return frames

    h, w = frames.shape[1], frames.shape[2]
    eye_centers = _detect_eye_centers(frames[0])

    if eye_centers is None:
        # No alignment possible, just resize all
        return np.stack([cv2.resize(f, target_size) for f in frames], axis=0)

    # Compute transform from first frame
    dst_pixels = REFERENCE_LANDMARKS.copy()
    dst_pixels[:, 0] *= target_size[0]
    dst_pixels[:, 1] *= target_size[1]

    src_pixels = eye_centers.copy()
    src_pixels[:, 0] *= w
    src_pixels[:, 1] *= h

    M = _compute_alignment_transform(src_pixels, dst_pixels)

    if M is None:
        return np.stack([cv2.resize(f, target_size) for f in frames], axis=0)

    # Apply same transform to all frames (temporal consistency)
    aligned = []
    for frame in frames:
        warped = cv2.warpAffine(
            frame, M, target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned.append(warped)

    return np.stack(aligned, axis=0)


class FrameSequenceDataset(Dataset):
    """
    Dataset for loading preprocessed micro-expression frame sequences.

    Expected structure:
        data_root/
        ├── cropped/
        │   ├── sub01/
        │   │   ├── EP02_01f/
        │   │   │   ├── reg_img001.jpg
        │   │   │   ├── reg_img002.jpg
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        ├── labels.csv (generated by convert_casme2.py)
        └── CASME2-coding-20190701.xlsx (original)

    Labels.csv format:
        video_path, subject, filename, me_label, emotion, onset, apex, offset, num_frames
    """

    EMOTION_NAMES_4CLASS = [
        'happiness', 'surprise', 'disgust', 'repression'
    ]

    EMOTION_NAMES_5CLASS = [
        'happiness', 'surprise', 'disgust', 'repression', 'others'
    ]

    # Always exclude these (too few samples: sadness=4, fear=2)
    EXCLUDE_EMOTIONS_ALWAYS = {'sadness', 'fear', 'anger', 'contempt'}

    def __init__(self, data_root, split='train', T=None, H=None, W=None,
                 augment=None, temporal_jitter=None, val_ratio=0.2, seed=42,
                 face_align=True, loso_fold=None, loso_subjects=None,
                 include_others=False):
        """
        Args:
            data_root (str): Root directory of CASME2 data
            split (str): 'train' or 'val'
            T (int): Number of frames to sample (default from config)
            H (int): Target height (default from config)
            W (int): Target width (default from config)
            augment (bool): Apply data augmentation
            temporal_jitter (bool): Random temporal sampling during training
            val_ratio (float): Validation split ratio (ignored if loso_fold set)
            seed (int): Random seed for splitting
            face_align (bool): Apply gaze stabilization reflex (face alignment)
            loso_fold (int): LOSO fold index (0-based). If set, uses
                Leave-One-Subject-Out: fold-th subject is val, rest are train.
            loso_subjects (list): Override subject list for LOSO
        """
        self.data_root = data_root
        self.split = split
        self.T = T or DATA_CONFIG['T']
        self.H = H or DATA_CONFIG['H']
        self.W = W or DATA_CONFIG['W']
        self.augment = augment if augment is not None else DATA_CONFIG['augment']
        self.temporal_jitter = temporal_jitter if temporal_jitter is not None else DATA_CONFIG['temporal_jitter']
        self.face_align = face_align
        self.include_others = include_others

        # Set emotion names and exclude set based on include_others
        if include_others:
            self.EMOTION_NAMES = self.EMOTION_NAMES_5CLASS
            self.EXCLUDE_EMOTIONS = self.EXCLUDE_EMOTIONS_ALWAYS
            self.num_classes = 5
        else:
            self.EMOTION_NAMES = self.EMOTION_NAMES_4CLASS
            self.EXCLUDE_EMOTIONS = self.EXCLUDE_EMOTIONS_ALWAYS | {'others'}
            self.num_classes = 4

        self.cropped_dir = os.path.join(data_root, 'cropped')

        # Load labels
        labels_path = os.path.join(data_root, 'labels.csv')
        if not os.path.exists(labels_path):
            # Convert from Excel if labels.csv doesn't exist
            print(f"[FrameSequenceDataset] labels.csv not found, converting from Excel...")
            self._convert_excel_to_csv(data_root)
            labels_path = os.path.join(data_root, 'labels.csv')

        self.samples = pd.read_csv(labels_path)
        print(f"[FrameSequenceDataset] Loaded {len(self.samples)} samples from {labels_path}")

        # Split into train/val
        if loso_fold is not None:
            self._split_loso(loso_fold, loso_subjects)
        else:
            self._split_dataset(val_ratio, seed)

        # ImageNet normalization (shape: (C, 1, 1, 1) for broadcasting over (C, T, H, W))
        self.mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(3, 1, 1, 1)
        self.std = torch.tensor(DATA_CONFIG['normalize_std']).view(3, 1, 1, 1)

    def _convert_excel_to_csv(self, data_root):
        """Convert CASME2 Excel to standard CSV format."""
        excel_path = os.path.join(data_root, 'CASME2-coding-20190701.xlsx')
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        df = pd.read_excel(excel_path)
        df.columns = ['Subject', 'Filename', 'Unnamed2', 'OnsetFrame', 'ApexFrame',
                      'OffsetFrame', 'Unnamed6', 'ActionUnits', 'Emotion']

        emotion_map_4class = {
            'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3,
        }

        emotion_map_5class = {
            'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 'others': 4,
        }

        emotion_map = emotion_map_5class if self.include_others else emotion_map_4class

        samples = []
        for idx, row in df.iterrows():
            subject = f"sub{int(row['Subject']):02d}"
            filename = row['Filename']
            frame_dir = os.path.join(self.cropped_dir, subject, filename)

            if not os.path.exists(frame_dir):
                continue

            emotion = str(row['Emotion']).strip().lower() if pd.notna(row['Emotion']) else 'others'

            # Skip excluded emotions (sadness, fear, anger, contempt)
            if emotion in self.EXCLUDE_EMOTIONS:
                continue

            me_label = emotion_map.get(emotion, -1)
            if me_label == -1:
                continue

            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
            num_frames = len(frame_files)

            # Safe int conversion -- CASME2 has invalid values like '/' in frame columns
            def safe_int(val):
                if pd.isna(val):
                    return 0
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    return 0

            samples.append({
                'video_path': f"{subject}/{filename}",
                'subject': subject,
                'filename': filename,
                'me_label': me_label,
                'emotion': emotion,
                'onset': safe_int(row['OnsetFrame']),
                'apex': safe_int(row['ApexFrame']),
                'offset': safe_int(row['OffsetFrame']),
                'num_frames': num_frames,
                'action_units': str(row['ActionUnits']) if pd.notna(row['ActionUnits']) else '',
            })

        samples_df = pd.DataFrame(samples)
        samples_df.to_csv(os.path.join(data_root, 'labels.csv'), index=False)
        print(f"[FrameSequenceDataset] Converted {len(samples)} samples to labels.csv")

    def _split_dataset(self, val_ratio, seed):
        """Split dataset into train and validation sets."""
        random.seed(seed)
        subjects = sorted(self.samples['subject'].unique())
        val_subjects = set(random.sample(subjects, int(len(subjects) * val_ratio)))

        if self.split == 'train':
            self.samples = self.samples[~self.samples['subject'].isin(val_subjects)]
        else:
            self.samples = self.samples[self.samples['subject'].isin(val_subjects)]

        print(f"[FrameSequenceDataset] {self.split}: {len(self.samples)} samples")

    def _split_loso(self, fold, loso_subjects=None):
        """
        Leave-One-Subject-Out (LOSO) split.

        In MER, LOSO is the standard evaluation protocol: train on all subjects
        except one, test on the held-out subject. This tests generalization to
        unseen identities -- critical for real-world deployment.

        Args:
            fold (int): Fold index (0-based). The fold-th subject is held out.
            loso_subjects (list): Override subject list
        """
        if loso_subjects is not None:
            subjects = sorted(loso_subjects)
        else:
            subjects = sorted(self.samples['subject'].unique())

        if fold >= len(subjects):
            raise ValueError(f"LOSO fold {fold} >= {len(subjects)} subjects")

        val_subject = subjects[fold]
        print(f"[FrameSequenceDataset] LOSO fold {fold}/{len(subjects)}: "
              f"val_subject={val_subject}")

        if self.split == 'train':
            self.samples = self.samples[self.samples['subject'] != val_subject]
        else:
            self.samples = self.samples[self.samples['subject'] == val_subject]

        print(f"[FrameSequenceDataset] {self.split}: {len(self.samples)} samples "
              f"(subject {val_subject})")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, sample_path):
        """
        Load frame sequence from directory.

        Args:
            sample_path (str): Relative path like "sub01/EP02_01f"

        Returns:
            frames (np.ndarray): (T_orig, H, W, 3) RGB uint8
        """
        frame_dir = os.path.join(self.cropped_dir, sample_path)

        # Get sorted frame files
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

        if len(frame_files) == 0:
            raise RuntimeError(f"No frames found in {frame_dir}")

        # Load frames
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
        Sample T frames with apex-centered strategy (retinal adaptation).

        The retina adapts to baseline (onset) and is most sensitive at the
        peak change (apex). This sampling emphasizes the onset→apex transition
        where micro-expression information is concentrated.

        Args:
            frames (np.ndarray): (T_orig, H, W, 3)
            onset (int): Onset frame index (1-based in CASME2)
            apex (int): Apex frame index (1-based in CASME2)
            offset (int): Offset frame index (1-based in CASME2)

        Returns:
            sampled (np.ndarray): (T, H, W, 3)
        """
        total = len(frames)
        T = self.T

        if total <= T:
            indices = np.arange(T) % total
            return frames[indices]

        # Convert 1-based CASME2 frame indices to 0-based
        onset_idx = max(0, onset - 1) if onset > 0 else 0
        apex_idx = min(total - 1, apex - 1) if apex > 0 else total // 2
        offset_idx = min(total - 1, offset - 1) if offset > 0 else total - 1

        # Ensure valid range
        if apex_idx <= onset_idx:
            apex_idx = min(onset_idx + total // 4, total - 1)
        if offset_idx <= apex_idx:
            offset_idx = min(apex_idx + total // 4, total - 1)

        if self.split == 'train' and self.temporal_jitter:
            # Training: random window centered around apex
            # Sample more frames from onset→apex (the informative part)
            half_T = T // 2

            # First half: onset→apex transition (most important)
            onset_apex_len = apex_idx - onset_idx
            if onset_apex_len >= half_T:
                # Evenly sample from onset to apex
                first_indices = np.linspace(onset_idx, apex_idx, half_T).astype(int)
            else:
                # Not enough frames, pad from before onset
                start = max(0, apex_idx - half_T)
                first_indices = np.linspace(start, apex_idx, half_T).astype(int)

            # Second half: apex→offset or after apex
            remaining = T - half_T
            after_apex = total - apex_idx - 1
            if after_apex >= remaining:
                second_indices = np.linspace(apex_idx, min(apex_idx + after_apex, total - 1), remaining).astype(int)
            else:
                second_indices = np.linspace(apex_idx, total - 1, remaining).astype(int)

            indices = np.concatenate([first_indices, second_indices])
        else:
            # Validation: deterministic apex-centered sampling
            # Distribute frames: 40% before apex, 20% at apex, 40% after apex
            n_before = int(T * 0.4)
            n_apex = max(1, T - 2 * n_before)
            n_after = T - n_before - n_apex

            before_indices = np.linspace(onset_idx, apex_idx, n_before + 1).astype(int)[:n_before]
            apex_indices = np.full(n_apex, apex_idx, dtype=int)
            after_indices = np.linspace(apex_idx, offset_idx, n_after + 1).astype(int)[1:n_after + 1]

            indices = np.concatenate([before_indices, apex_indices, after_indices])

        # Ensure we have exactly T indices
        if len(indices) < T:
            indices = np.concatenate([indices, np.full(T - len(indices), indices[-1], dtype=int)])
        elif len(indices) > T:
            indices = indices[:T]

        indices = np.clip(indices, 0, total - 1)
        return frames[indices]

    def _resize_frames(self, frames):
        """Resize frames to target resolution."""
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (self.W, self.H)))
        return np.stack(resized, axis=0)

    def _align_frames(self, frames):
        """
        Apply gaze stabilization reflex (face alignment) to frame sequence.

        Uses the first frame's eye positions to compute a similarity transform,
        then applies it to all frames for temporal consistency. This is
        analogous to the vestibulo-ocular reflex stabilizing retinal images.

        Args:
            frames: (T_orig, H, W, 3) RGB uint8

        Returns:
            aligned: (T_orig, H, W, 3) RGB uint8, aligned and resized
        """
        if not self.face_align:
            return self._resize_frames(frames)

        # Align using first frame's landmarks, output at target resolution
        aligned = align_face_sequence(frames, target_size=(self.W, self.H))
        return aligned

    def _augment_frames(self, frames_tensor):
        """
        Apply augmentation to video tensor.

        Args:
            frames_tensor (torch.Tensor): (C, T, H, W) float [0, 1]

        Returns:
            augmented (torch.Tensor): (C, T, H, W)
        """
        if not self.augment or self.split != 'train':
            return frames_tensor

        # Horizontal flip
        if random.random() < 0.5:
            frames_tensor = frames_tensor.flip(-1)

        # Color jitter (brightness and contrast)
        if random.random() < 0.3:
            b = 1.0 + random.uniform(-0.1, 0.1)
            c = 1.0 + random.uniform(-0.1, 0.1)
            frames_tensor = frames_tensor * b
            mean = frames_tensor.mean(dim=(2, 3), keepdim=True)
            frames_tensor = (frames_tensor - mean) * c + mean
            frames_tensor = frames_tensor.clamp(0, 1)

        # Stronger color jitter: saturation and hue
        if random.random() < 0.2:
            # Convert to grayscale and mix
            gray = frames_tensor.mean(dim=0, keepdim=True).expand_as(frames_tensor)
            alpha = random.uniform(0.5, 1.0)
            frames_tensor = alpha * frames_tensor + (1 - alpha) * gray

        # Random temporal crop (stronger jitter)
        if random.random() < 0.3:
            T = frames_tensor.shape[1]
            if T > 4:
                crop_len = random.randint(T // 2, T)
                start = random.randint(0, T - crop_len)
                cropped = frames_tensor[:, start:start + crop_len, :, :]
                # Resize back to T frames
                indices = torch.linspace(0, crop_len - 1, T).long()
                frames_tensor = cropped[:, indices, :, :]

        # Random erasing (simulate occlusion)
        if random.random() < 0.15:
            C, T, H, W = frames_tensor.shape
            eh = random.randint(H // 8, H // 4)
            ew = random.randint(W // 8, W // 4)
            eh_start = random.randint(0, H - eh)
            ew_start = random.randint(0, W - ew)
            frames_tensor[:, :, eh_start:eh_start + eh, ew_start:ew_start + ew] = 0

        # Temporal dropout: randomly zero out 1-2 frames (simulate frame loss)
        if random.random() < 0.2:
            C, T, H, W = frames_tensor.shape
            num_drop = random.randint(1, 2)
            drop_indices = random.sample(range(T), min(num_drop, T))
            for di in drop_indices:
                frames_tensor[:, di, :, :] = 0

        # Temporal speed perturbation: speed up or slow down by small factor
        if random.random() < 0.15:
            C, T, H, W = frames_tensor.shape
            speed_factor = random.uniform(0.8, 1.2)
            new_T = int(T / speed_factor)
            if new_T >= 4 and new_T != T:
                indices = torch.linspace(0, T - 1, new_T).long()
                resampled = frames_tensor[:, indices, :, :]
                # Resize back to T frames
                indices_back = torch.linspace(0, new_T - 1, T).long()
                frames_tensor = resampled[:, indices_back, :, :]

        # Elastic deformation: simulate subtle facial muscle variation
        # (biomimetic: facial muscles deform elastically, not rigidly)
        if random.random() < 0.15:
            C, T, H, W = frames_tensor.shape
            # Generate random displacement fields
            alpha = random.uniform(8, 15)  # deformation magnitude
            sigma = random.uniform(3, 5)   # smoothing sigma
            dx = np.random.randn(H, W) * alpha
            dy = np.random.randn(H, W) * alpha
            dx = cv2.GaussianBlur(dx, (0, 0), sigma).astype(np.float32)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma).astype(np.float32)
            # Apply same deformation to all frames (consistent spatial warp)
            for t in range(T):
                frame = frames_tensor[:, t, :, :].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                # Remap coordinates
                x, y = np.meshgrid(np.arange(W), np.arange(H))
                map_x = (x + dx).astype(np.float32)
                map_y = (y + dy).astype(np.float32)
                warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                frames_tensor[:, t, :, :] = torch.from_numpy(warped).permute(2, 0, 1)

        return frames_tensor

    def __getitem__(self, idx):
        """
        Get a sample.

        Returns:
            video (torch.Tensor): (C, T, H, W) normalized
            me_label (int): Emotion label (0-7)
            au_label (torch.Tensor): Placeholder AU labels (zeros)
        """
        sample = self.samples.iloc[idx]

        # Load frames
        frames = self._load_frames(sample['video_path'])  # (T_orig, H, W, 3)

        # Get onset/apex/offset for apex-centered sampling
        onset = int(sample.get('onset', 0)) if pd.notna(sample.get('onset', 0)) else 0
        apex = int(sample.get('apex', 0)) if pd.notna(sample.get('apex', 0)) else 0
        offset = int(sample.get('offset', 0)) if pd.notna(sample.get('offset', 0)) else 0

        # Sample T frames with apex-centered strategy
        frames = self._sample_frames(frames, onset=onset, apex=apex, offset=offset)

        # Align (gaze stabilization reflex) + resize
        frames = self._align_frames(frames)  # (T, H, W, 3)

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames_tensor = torch.from_numpy(frames).float() / 255.0  # [0, 1]
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

        # Normalize
        frames_tensor = (frames_tensor - self.mean) / self.std

        # Augment
        frames_tensor = self._augment_frames(frames_tensor)

        # Labels
        me_label = int(sample['me_label'])

        # Parse AU labels from action_units string
        au_label = torch.zeros(28, dtype=torch.float32)
        au_str = str(sample.get('action_units', ''))
        if au_str and au_str != 'nan' and au_str != '':
            # CASME2 AU format: "4+7+L10" or "12" or "4+7+L10+R12"
            for part in au_str.replace(' ', '').split('+'):
                part = part.strip()
                # Remove L/R prefix (left/right side indicator)
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


def get_casme2_dataloaders(data_root, batch_size=8, T=16, H=224, W=224,
                           num_workers=4, val_ratio=0.2, seed=42,
                           face_align=True, loso_fold=None):
    """
    Create train and validation dataloaders for CASME2.

    Args:
        data_root (str): Path to CASME2 data
        batch_size (int): Batch size
        T (int): Number of frames
        H, W (int): Spatial resolution
        num_workers (int): DataLoader workers
        val_ratio (float): Validation ratio (ignored if loso_fold set)
        seed (int): Random seed
        face_align (bool): Apply gaze stabilization reflex (face alignment)
        loso_fold (int): LOSO fold index. If set, uses Leave-One-Subject-Out.

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    train_dataset = FrameSequenceDataset(
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

    val_dataset = FrameSequenceDataset(
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


def get_loso_subjects(data_root):
    """
    Get sorted list of subjects for LOSO cross-validation.

    Returns:
        subjects: sorted list of subject IDs
    """
    labels_path = os.path.join(data_root, 'labels.csv')
    if not os.path.exists(labels_path):
        # Trigger Excel -> CSV conversion by instantiating dataset briefly
        print(f"[get_loso_subjects] labels.csv not found, triggering conversion...")
        ds = FrameSequenceDataset(data_root, split='train', face_align=False)
        labels_path = os.path.join(data_root, 'labels.csv')
    if not os.path.exists(labels_path):
        return None
    samples = pd.read_csv(labels_path)
    return sorted(samples['subject'].unique())


if __name__ == '__main__':
    # Test dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    print("Testing FrameSequenceDataset...")

    train_loader, val_loader = get_casme2_dataloaders(
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

    print("\n[Success] Dataset test passed!")


# =============================================================================
# Dict-format Dataset Wrappers (for train_cross.py compatibility)
# =============================================================================
# These wrappers return dict format {'frames', 'me_label', 'au_label', 'subject'}
# instead of tuple format (video, me_label, au_label), enabling ConcatDataset
# across CASME2, SMIC, and SAMM with unified label mapping.

class CASME2FrameDataset(Dataset):
    """
    CASME2 dataset wrapper returning dict format with unified emotion mapping.

    Supports custom emotion_map for cross-dataset joint training.
    """
    EXCLUDE_EMOTIONS = {'sadness', 'fear', 'anger', 'contempt', 'others'}

    def __init__(self, root_dir, T=16, H=224, W=224, augment=True,
                 emotion_map=None, exclude_negative=True, face_align=True):
        """
        Args:
            root_dir: CASME2 data root (contains cropped/ and labels.csv)
            T, H, W: Frame sampling parameters
            augment: Apply data augmentation
            emotion_map: Custom emotion mapping dict. If None, uses default 4-class.
            exclude_negative: Exclude samples with label=-1 (others class)
            face_align: Apply gaze stabilization reflex
        """
        self.root_dir = root_dir
        self.T = T
        self.H = H
        self.W = W
        self.augment = augment
        self.face_align = face_align
        self.emotion_map = emotion_map or {
            'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3,
        }
        self.exclude_negative = exclude_negative

        self.cropped_dir = os.path.join(root_dir, 'cropped')

        # Load labels
        labels_path = os.path.join(root_dir, 'labels.csv')
        if not os.path.exists(labels_path):
            # Convert from Excel
            from scripts.convert_casme2 import convert_casme2_to_standard
            convert_casme2_to_standard(root_dir)
            labels_path = os.path.join(root_dir, 'labels.csv')

        self.samples = pd.read_csv(labels_path)

        # Apply emotion mapping and filtering
        if emotion_map is not None:
            # Remap labels using custom emotion_map
            valid_indices = []
            for idx, row in self.samples.iterrows():
                emotion = str(row.get('emotion', '')).strip().lower()
                new_label = emotion_map.get(emotion, -1)
                if exclude_negative and new_label == -1:
                    continue
                self.samples.at[idx, 'me_label'] = new_label
                valid_indices.append(idx)
            self.samples = self.samples.loc[valid_indices].reset_index(drop=True)
        else:
            # Default: exclude 'others' class
            self.samples = self.samples[
                self.samples['me_label'] != -1
            ].reset_index(drop=True)

        # Store subjects for LOSO
        self.subjects = sorted(self.samples['subject'].unique())

        # ImageNet normalization
        self.mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(3, 1, 1, 1)
        self.std = torch.tensor(DATA_CONFIG['normalize_std']).view(3, 1, 1, 1)

        print(f"[CASME2FrameDataset] {len(self.samples)} samples, {len(self.subjects)} subjects")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]

        # Load frames
        frame_dir = os.path.join(self.cropped_dir, sample['video_path'])
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        frames = []
        for ff in frame_files:
            frame = cv2.imread(os.path.join(frame_dir, ff))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        if len(frames) == 0:
            # Return dummy
            frames = np.zeros((self.T, self.H, self.W, 3), dtype=np.uint8)

        frames = np.stack(frames, axis=0)  # (T_orig, H, W, 3)

        # Sample T frames with apex-centered strategy
        onset = int(sample.get('onset', 0)) if pd.notna(sample.get('onset', 0)) else 0
        apex = int(sample.get('apex', 0)) if pd.notna(sample.get('apex', 0)) else 0
        offset = int(sample.get('offset', 0)) if pd.notna(sample.get('offset', 0)) else 0

        # Use FrameSequenceDataset's _sample_frames logic
        total = len(frames)
        T = self.T
        if total <= T:
            indices = np.arange(T) % total
            frames = frames[indices]
        else:
            onset_idx = max(0, onset - 1) if onset > 0 else 0
            apex_idx = min(total - 1, apex - 1) if apex > 0 else total // 2
            offset_idx = min(total - 1, offset - 1) if offset > 0 else total - 1
            if apex_idx <= onset_idx:
                apex_idx = min(onset_idx + total // 4, total - 1)
            if offset_idx <= apex_idx:
                offset_idx = min(apex_idx + total // 4, total - 1)

            n_before = int(T * 0.4)
            n_apex = max(1, T - 2 * n_before)
            n_after = T - n_before - n_apex
            before = np.linspace(onset_idx, apex_idx, n_before + 1).astype(int)[:n_before]
            at_apex = np.full(n_apex, apex_idx, dtype=int)
            after = np.linspace(apex_idx, offset_idx, n_after + 1).astype(int)[1:n_after + 1]
            indices = np.concatenate([before, at_apex, after])
            if len(indices) < T:
                indices = np.concatenate([indices, np.full(T - len(indices), indices[-1], dtype=int)])
            indices = indices[:T]
            indices = np.clip(indices, 0, total - 1)
            frames = frames[indices]

        # Resize + align
        if self.face_align:
            frames = align_face_sequence(frames, target_size=(self.W, self.H))
        else:
            frames = np.stack([cv2.resize(f, (self.W, self.H)) for f in frames], axis=0)

        # To tensor
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        frames_tensor = (frames_tensor - self.mean) / self.std

        # Augment
        if self.augment:
            frames_tensor = self._augment(frames_tensor)

        # Labels
        me_label = int(sample['me_label'])
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

        return {
            'frames': frames_tensor,
            'me_label': me_label,
            'au_label': au_label,
            'subject': sample.get('subject', ''),
        }

    def get_subject(self, idx):
        """Get subject ID for LOSO."""
        return self.samples.iloc[idx].get('subject', '')

    def _augment(self, frames_tensor):
        """Apply augmentation to video tensor (C, T, H, W)."""
        # Horizontal flip
        if random.random() < 0.5:
            frames_tensor = frames_tensor.flip(-1)
        # Color jitter
        if random.random() < 0.3:
            b = 1.0 + random.uniform(-0.1, 0.1)
            frames_tensor = frames_tensor * b
            frames_tensor = frames_tensor.clamp(0, 1)
        # Random erasing
        if random.random() < 0.15:
            C, T, H, W = frames_tensor.shape
            eh = random.randint(H // 8, H // 4)
            ew = random.randint(W // 8, W // 4)
            eh_start = random.randint(0, H - eh)
            ew_start = random.randint(0, W - ew)
            frames_tensor[:, :, eh_start:eh_start + eh, ew_start:ew_start + ew] = 0
        return frames_tensor


class SMICFrameDataset(Dataset):
    """
    SMIC dataset wrapper returning dict format with unified emotion mapping.
    """
    def __init__(self, root_dir, T=16, H=224, W=224, augment=True,
                 emotion_map=None, exclude_negative=True, face_align=True):
        self.root_dir = root_dir
        self.T = T
        self.H = H
        self.W = W
        self.augment = augment
        self.face_align = face_align

        # SMIC emotion mapping
        default_map = {'positive': 0, 'negative': 2, 'surprise': 1}
        self.emotion_map = emotion_map or default_map

        # Find SMIC directory
        self.smic_dir = root_dir
        for candidate in [os.path.join(root_dir, 'SMIC_all_cropped'),
                          os.path.join(root_dir, 'HS'),
                          root_dir]:
            if os.path.exists(candidate):
                hs_dir = os.path.join(candidate, 'HS')
                if os.path.exists(hs_dir):
                    self.smic_dir = hs_dir
                    break

        # Build sample list
        self.samples = self._build_samples()
        if exclude_negative:
            self.samples = [s for s in self.samples if s['me_label'] >= 0]

        # Store subjects
        self.subjects = sorted(set(s['subject'] for s in self.samples))

        # Normalization
        self.mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(3, 1, 1, 1)
        self.std = torch.tensor(DATA_CONFIG['normalize_std']).view(3, 1, 1, 1)

        print(f"[SMICFrameDataset] {len(self.samples)} samples, {len(self.subjects)} subjects")

    def _build_samples(self):
        samples = []
        for subject_dir in sorted(os.listdir(self.smic_dir)):
            subject_path = os.path.join(self.smic_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            micro_dir = os.path.join(subject_path, 'micro')
            if not os.path.exists(micro_dir):
                continue
            for emotion_dir in sorted(os.listdir(micro_dir)):
                emotion_path = os.path.join(micro_dir, emotion_dir)
                if not os.path.isdir(emotion_path):
                    continue
                me_label = self.emotion_map.get(emotion_dir, -1)
                for sample_dir in sorted(os.listdir(emotion_path)):
                    sample_path = os.path.join(emotion_path, sample_dir)
                    if not os.path.isdir(sample_path):
                        continue
                    frame_files = sorted([f for f in os.listdir(sample_path)
                                          if f.endswith('.bmp') or f.endswith('.jpg')])
                    if len(frame_files) == 0:
                        continue
                    samples.append({
                        'path': sample_path,
                        'subject': subject_dir,
                        'me_label': me_label,
                        'emotion': emotion_dir,
                        'num_frames': len(frame_files),
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_dir = sample['path']
        frame_files = sorted([f for f in os.listdir(frame_dir)
                              if f.endswith('.bmp') or f.endswith('.jpg')])
        frames = []
        for ff in frame_files:
            frame = cv2.imread(os.path.join(frame_dir, ff))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        if len(frames) == 0:
            frames = np.zeros((self.T, self.H, self.W, 3), dtype=np.uint8)
        frames = np.stack(frames, axis=0)

        # Sample T frames (center around estimated apex)
        total = len(frames)
        T = self.T
        if total <= T:
            indices = np.arange(T) % total
            frames = frames[indices]
        else:
            apex_idx = total // 2
            n_before = int(T * 0.4)
            n_apex = max(1, T - 2 * n_before)
            n_after = T - n_before - n_apex
            before = np.linspace(0, apex_idx, n_before + 1).astype(int)[:n_before]
            at_apex = np.full(n_apex, apex_idx, dtype=int)
            after = np.linspace(apex_idx, total - 1, n_after + 1).astype(int)[1:n_after + 1]
            indices = np.concatenate([before, at_apex, after])
            indices = indices[:T]
            indices = np.clip(indices, 0, total - 1)
            frames = frames[indices]

        # Resize + align
        if self.face_align:
            frames = align_face_sequence(frames, target_size=(self.W, self.H))
        else:
            frames = np.stack([cv2.resize(f, (self.W, self.H)) for f in frames], axis=0)

        # To tensor
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        frames_tensor = (frames_tensor - self.mean) / self.std

        if self.augment:
            if random.random() < 0.5:
                frames_tensor = frames_tensor.flip(-1)

        return {
            'frames': frames_tensor,
            'me_label': sample['me_label'],
            'au_label': torch.zeros(28, dtype=torch.float32),
            'subject': sample['subject'],
        }

    def get_subject(self, idx):
        return self.samples[idx]['subject']


class SAMMFrameDataset(Dataset):
    """
    SAMM dataset wrapper returning dict format with unified emotion mapping.
    """
    def __init__(self, root_dir, T=16, H=224, W=224, augment=True,
                 emotion_map=None, exclude_negative=True, face_align=True):
        self.root_dir = root_dir
        self.T = T
        self.H = H
        self.W = W
        self.augment = augment
        self.face_align = face_align

        # SAMM emotion mapping
        default_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 2, 6: 2, 7: 4}
        self.emotion_map = emotion_map or default_map

        # Find SAMM directory
        self.samm_dir = root_dir
        for candidate in [os.path.join(root_dir, 'SAMM'), root_dir]:
            if os.path.exists(candidate):
                self.samm_dir = candidate
                break

        # Build sample list from directory structure
        self.samples = self._build_samples()
        if exclude_negative:
            self.samples = [s for s in self.samples if s['me_label'] >= 0]

        self.subjects = sorted(set(s['subject'] for s in self.samples))

        self.mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(3, 1, 1, 1)
        self.std = torch.tensor(DATA_CONFIG['normalize_std']).view(3, 1, 1, 1)

        print(f"[SAMMFrameDataset] {len(self.samples)} samples, {len(self.subjects)} subjects")

    def _build_samples(self):
        samples = []
        for subject_dir in sorted(os.listdir(self.samm_dir)):
            subject_path = os.path.join(self.samm_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            # Skip non-subject directories
            if not subject_dir.replace('_', '').isdigit():
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

                me_label = self.emotion_map.get(emotion_code, -1)

                frame_files = sorted([f for f in os.listdir(sample_path)
                                      if f.endswith('.jpg') or f.endswith('.bmp')])
                if len(frame_files) == 0:
                    continue

                samples.append({
                    'path': sample_path,
                    'subject': subject_dir,
                    'me_label': me_label,
                    'emotion_code': emotion_code,
                    'num_frames': len(frame_files),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_dir = sample['path']
        frame_files = sorted([f for f in os.listdir(frame_dir)
                              if f.endswith('.jpg') or f.endswith('.bmp')])
        frames = []
        for ff in frame_files:
            frame = cv2.imread(os.path.join(frame_dir, ff))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        if len(frames) == 0:
            frames = np.zeros((self.T, self.H, self.W, 3), dtype=np.uint8)
        frames = np.stack(frames, axis=0)

        # Sample T frames
        total = len(frames)
        T = self.T
        if total <= T:
            indices = np.arange(T) % total
            frames = frames[indices]
        else:
            apex_idx = total // 2
            n_before = int(T * 0.4)
            n_apex = max(1, T - 2 * n_before)
            n_after = T - n_before - n_apex
            before = np.linspace(0, apex_idx, n_before + 1).astype(int)[:n_before]
            at_apex = np.full(n_apex, apex_idx, dtype=int)
            after = np.linspace(apex_idx, total - 1, n_after + 1).astype(int)[1:n_after + 1]
            indices = np.concatenate([before, at_apex, after])
            indices = indices[:T]
            indices = np.clip(indices, 0, total - 1)
            frames = frames[indices]

        # Resize + align
        if self.face_align:
            frames = align_face_sequence(frames, target_size=(self.W, self.H))
        else:
            frames = np.stack([cv2.resize(f, (self.W, self.H)) for f in frames], axis=0)

        # To tensor
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        frames_tensor = (frames_tensor - self.mean) / self.std

        if self.augment:
            if random.random() < 0.5:
                frames_tensor = frames_tensor.flip(-1)

        return {
            'frames': frames_tensor,
            'me_label': sample['me_label'],
            'au_label': torch.zeros(28, dtype=torch.float32),
            'subject': sample['subject'],
        }

    def get_subject(self, idx):
        return self.samples[idx]['subject']