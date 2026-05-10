# =============================================================================
# Censor -- Real Micro-Expression Dataset Loader
# =============================================================================
# Supports CASME II, SAMM, SMIC-HS, MMEW, and CAS(ME)³ formats.
#
# All public MER datasets require signed license agreements:
#   CASME II: http://casme.psych.ac.cn/casme/c2
#   SAMM:     https://www.mmu.ac.uk (contact A. Davison)
#   SMIC:     https://www.oulu.fi
#   MMEW:     https://github.com/benxianyeteam/MMEW-Dataset
#   CAS(ME)³: http://melab.psych.ac.cn
#
# Dataset folder structure (expected after preparation):
#   ./data/<dataset_name>/
#     videos/
#       video1.avi
#       video2.avi
#       ...
#     labels.csv
#     subjects.csv
# =============================================================================

import os
import csv
import random
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config.defaults import DATA_CONFIG


# =============================================================================
# Frame-level Transformations
# =============================================================================

class CenterCrop:
    """Center crop retaining face_crop_ratio of each dimension."""
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def __call__(self, img):
        h, w = img.shape[-2:]
        crop_h, crop_w = int(h * self.ratio), int(w * self.ratio)
        y0, x0 = (h - crop_h) // 2, (w - crop_w) // 2
        return img[..., y0:y0 + crop_h, x0:x0 + crop_w]


class VideoResize:
    """Resize video tensor to (T, H, W)."""
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, video):
        """video: np.ndarray (T, H, W, C) or torch.Tensor (T, C, H, W)."""
        if isinstance(video, np.ndarray):
            resized = np.stack([cv2.resize(f, self.size) for f in video], axis=0)
            return resized
        # torch tensor: use interpolation
        C, T, H, W = video.shape
        return torch.nn.functional.interpolate(
            video, size=self.size, mode='bilinear', align_corners=False
        )


class VideoNormalize:
    """Normalize video using ImageNet mean/std."""
    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(mean or DATA_CONFIG['normalize_mean']).view(-1, 1, 1, 1)
        self.std = torch.tensor(std or DATA_CONFIG['normalize_std']).view(-1, 1, 1, 1)

    def __call__(self, video):
        return (video - self.mean) / self.std


class VideoRandomHorizontalFlip:
    """Random horizontal flip with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        if random.random() < self.p:
            return video.flip(-1)
        return video


class VideoColorJitter:
    """Simple color jitter for video tensors."""
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, video):
        """video: (C, T, H, W), C=RGB order."""
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video).float()
        C, T, H, W = video.shape
        # Brightness
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        video = video * b
        # Contrast
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = video.mean(dim=(2, 3), keepdim=True)
        video = (video - mean) * c + mean
        # Clamp
        video = video.clamp(0, 1)
        return video


# =============================================================================
# Optical Flow Computation
# =============================================================================

def compute_tvl1_flow_np(frame0, frame1, tau=0.25, lmbda=0.15, theta=0.3):
    """
    Compute TV-L1 optical flow between two grayscale frames using OpenCV.

    Args:
        frame0, frame1 (np.ndarray): Grayscale frames, shape (H, W), dtype=np.float32
    Returns:
        flow (np.ndarray): Flow field, shape (H, W, 2), dtype=np.float32
    """
    try:
        from cv2.optflow import createOptFlow_DualTVL1
        dtvl = createOptFlow_DualTVL1()
        dtvl.setTau(tau)
        dtvl.setLambda(lmbda)
        dtvl.setTheta(theta)
        dtvl.setWarpingsNumber(5)
        dtvl.setEpsilon(0.01)
        flow = dtvl.calc(frame0, frame1, None)
    except ImportError:
        # Fallback: gradient-based approximation
        diff = frame1 - frame0
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        dx = cv2.filter2D(diff, -1, sobel_x)
        dy = cv2.filter2D(diff, -1, sobel_y)
        mag = np.sqrt(dx**2 + dy**2 + 1e-8)
        flow_x = dx / (mag + 1.0) * tau
        flow_y = dy / (mag + 1.0) * tau
        flow = np.stack([flow_x, flow_y], axis=-1)
    return flow


def compute_video_optical_flow(video_frames):
    """
    Compute optical flow for all consecutive frame pairs in a video.

    Args:
        video_frames (np.ndarray): (T, H, W, C), RGB uint8 [0,255]
    Returns:
        flow_stack (np.ndarray): (T-1, H, W, 2) flow fields, float32
    """
    T = video_frames.shape[0]
    flows = []
    for t in range(T - 1):
        # Convert to grayscale
        gray0 = cv2.cvtColor(video_frames[t], cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray1 = cv2.cvtColor(video_frames[t + 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        flow = compute_tvl1_flow_np(gray0, gray1)
        flows.append(flow)
    return np.stack(flows, axis=0)  # (T-1, H, W, 2)


# =============================================================================
# Dataset Class
# =============================================================================

class MERDataset(Dataset):
    """
    Micro-Expression Recognition Dataset.

    Loads video frames from disk, performs face crop / center crop, resizes,
    applies augmentation, optionally computes optical flow, and returns
    normalized video tensor + labels.

    Expected annotation CSV format (columns):
        video_path, subject, me_label, au_01, au_02, ..., au_28

    Args:
        root_dir (str): Root directory containing 'videos/' and 'labels.csv'
        split (str): 'train', 'val', or 'test'
        T (int): Number of temporal frames to sample
        H, W (int): Spatial resolution
        augment (bool): Enable data augmentation
        compute_flow (bool): Compute optical flow during loading
        temporal_jitter (bool): Random temporal sampling
    """

    # FACS AU mapping for 28-dimensional AU labels
    AU_NAMES = [
        'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07',
        'AU09', 'AU10', 'AU11', 'AU12', 'AU14', 'AU15',
        'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26',
        'AU27', 'AU28',
    ] + [f'AU{i:02d}' for i in [3, 8, 13, 16, 18, 19, 21, 22]]

    ME_CATEGORIES = ["Happiness", "Sadness", "Surprise", "Fear", "Anger", "Disgust", "Contempt"]

    def __init__(self, root_dir, split='train', T=None, H=None, W=None,
                 augment=None, compute_flow=None, temporal_jitter=None):
        self.root_dir = root_dir
        self.split = split  # 'train', 'val', 'test'
        cfg = DATA_CONFIG

        self.T = T or cfg['T']
        self.H = H or cfg['H']
        self.W = W or cfg['W']
        self.augment = augment if augment is not None else cfg['augment']
        self.compute_flow = compute_flow if compute_flow is not None else cfg['compute_flow_on_the_fly']
        self.temporal_jitter = temporal_jitter if temporal_jitter is not None else cfg['temporal_jitter']

        # Load annotation CSV
        csv_path = os.path.join(root_dir, 'labels.csv')
        if not os.path.exists(csv_path):
            # Fallback: check for annotations.csv
            csv_path = os.path.join(root_dir, 'annotations.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"No labels.csv or annotations.csv found in {root_dir}.\n"
                f"Expected columns: video_path,subject,me_label,au_01,au_02,...,au_28"
            )

        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {
                    'video_path': os.path.join(root_dir, 'videos', row['video_path'].strip()),
                    'subject': row.get('subject', '').strip(),
                    'me_label': int(row.get('me_label', 0)),
                }
                # Parse AU labels (28 binary values)
                au = np.zeros(28, dtype=np.float32)
                for i in range(28):
                    key = f'au_{i+1:02d}'
                    if key in row:
                        au[i] = float(row[key].strip())
                sample['au_label'] = au
                self.samples.append(sample)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {csv_path}")

        print(f"[MERDataset] Loaded {len(self.samples)} samples ({split}) from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path):
        """
        Load video frames from file.

        Args:
            video_path (str): Path to .avi or .mp4 file
        Returns:
            frames (np.ndarray): (T, H, W, 3), RGB uint8 [0,255]
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Could not read video: {video_path}")

        return np.stack(frames, axis=0)  # (T, H, W, 3)

    def _sample_frames(self, frames):
        """
        Sample T frames from the video.
        If temporal_jitter is enabled in train mode, randomly selects start point.
        Otherwise, evenly samples T frames.
        """
        total = len(frames)
        T = self.T

        if total <= T:
            # Repeat frames if video is too short
            indices = np.arange(T) % total
        elif self.split == 'train' and self.temporal_jitter:
            # Random temporal jitter: sample a contiguous segment
            max_start = total - T
            start = np.random.randint(0, max_start + 1)
            indices = np.arange(start, start + T)
        else:
            # Even sampling
            indices = np.linspace(0, total - 1, T).astype(int)

        return frames[indices]

    def _center_crop(self, frames):
        """Center crop the frame retaining face_crop_ratio."""
        ratio = DATA_CONFIG['face_crop_ratio']
        h, w = frames.shape[1:3]
        crop_h, crop_w = int(h * ratio), int(w * ratio)
        y0, x0 = (h - crop_h) // 2, (w - crop_w) // 2
        return frames[:, y0:y0 + crop_h, x0:x0 + crop_w]

    def _augment_video(self, video):
        """
        Apply data augmentation to video tensor.

        Args:
            video (torch.Tensor): (C, T, H, W), float [0,1]
        Returns:
            augmented (torch.Tensor): (C, T, H, W), float [0,1]
        """
        if not self.augment or self.split != 'train':
            return video

        # Horizontal flip
        if random.random() < 0.5:
            video = video.flip(-1)

        # Color jitter
        b = 1.0 + random.uniform(-0.1, 0.1)
        c = 1.0 + random.uniform(-0.1, 0.1)
        video = video * b
        mean = video.mean(dim=(2, 3), keepdim=True)
        video = (video - mean) * c + mean
        video = video.clamp(0, 1)

        return video

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load video
        frames = self._load_video(sample['video_path'])  # (T_orig, H, W, 3)

        # Center crop (face region)
        frames = self._center_crop(frames)  # (T_orig, H_crop, W_crop, 3)

        # Resize spatial dims to (H, W)
        frames_rsz = np.stack([cv2.resize(f, (self.W, self.H)) for f in frames], axis=0)
        # frames_rsz: (T_orig, H, W, 3), RGB uint8

        # Sample T frames
        frames_sampled = self._sample_frames(frames_rsz)  # (T, H, W, 3)

        # Convert to float [0, 1]
        video = frames_sampled.astype(np.float32) / 255.0

        # Compute optical flow if requested (for validation/debugging)
        # Note: During training, flow is computed inside the Censor model.
        # This flow is returned as raw for potential external use.
        flow = None
        if self.compute_flow:
            flow = compute_video_optical_flow(frames_sampled)  # (T-1, H, W, 2)

        # Convert to torch: (C, T, H, W)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()  # (3, T, H, W)

        # Augmentation
        video = self._augment_video(video)

        # Normalize
        video = self._normalize(video)

        return video, sample['me_label'], sample['au_label']

    def _normalize(self, video):
        """Apply ImageNet normalization."""
        mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(-1, 1, 1, 1)
        std = torch.tensor(DATA_CONFIG['normalize_std']).view(-1, 1, 1, 1)
        return (video - mean) / std


# =============================================================================
# Dataset Inference Visualizer
# =============================================================================

def visualize_sample(dataset, idx=0):
    """Visualize a single sample from the dataset."""
    import matplotlib.pyplot as plt

    video, me_label, au_label = dataset[idx]

    print(f"Video shape: {video.shape}")  # (3, T, H, W)
    print(f"ME label: {me_label} ({MERDataset.ME_CATEGORIES[me_label] if me_label < 7 else 'Unknown'})")
    print(f"AU labels: {au_label}")
    active_aus = [i for i in range(28) if au_label[i] > 0.5]
    print(f"Active AUs: {active_aus}")

    # Display first 4 frames
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        frame = video[:, i, :, :].permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array(DATA_CONFIG['normalize_mean'])
        std = np.array(DATA_CONFIG['normalize_std'])
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        ax.imshow(frame)
        ax.set_title(f"Frame {i}")
        ax.axis('off')
    plt.tight_layout()
    return fig


# =============================================================================
# Dataset Splits
# =============================================================================

def get_dataset_splits(root_dir, dataset_name='casme2', val_ratio=0.2, seed=42):
    """
    Get train/val datasets with optional Leave-One-Subject-Out.

    Args:
        root_dir (str): Root directory of the dataset
        dataset_name (str): Dataset name ('casme2', 'samm', 'smic', 'mmew')
        val_ratio (float): Ratio of validation samples (ignored for LOSO)
        seed (int): Random seed
    Returns:
        train_dataset, val_dataset
    """
    # Currently uses random split (not LOSO)
    # LOSO requires per-subject folds which depends on annotation format
    full_dataset = MERDataset(root_dir, split='train')

    # Random split
    rng = torch.Generator().manual_seed(seed)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=rng
    )

    return train_dataset, val_dataset


def create_dataloader(dataset, batch_size=2, shuffle=True, num_workers=0):
    """Create DataLoader for a MERDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test MER Dataset')
    parser.add_argument('--root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--dataset', type=str, default='casme2', help='Dataset name')
    parser.add_argument('--test-only', action='store_true', help='Run test without training')
    args = parser.parse_args()

    data_dir = os.path.join(args.root, args.dataset)
    print(f"Testing dataset from: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        print("Please prepare the dataset first using prepare_data.py")
        exit(0)

    try:
        dataset = MERDataset(data_dir, split='train')
        print(f"Dataset size: {len(dataset)}")
        video, me_label, au_label = dataset[0]
        print(f"Sample 0: video={video.shape}, me_label={me_label}, au_label={au_label.shape}")

        viz_fig = visualize_sample(dataset, 0)
        viz_fig.savefig('dataset_sample.png')
        print("Sample visualization saved to dataset_sample.png")

        # Test dataloader
        loader = create_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))
        print(f"Batch: videos={batch[0].shape}, labels={batch[1].shape}, au={batch[2].shape}")
        print("Dataset test PASSED!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo prepare the dataset, place files in the following structure:")
        print(f"  {data_dir}/")
        print(f"    videos/")
        print(f"      subject01_video1.avi")
        print(f"      subject01_video2.avi")
        print(f"      ...")
        print(f"    labels.csv  (columns: video_path,subject,me_label,au_01,...,au_28)")
        print(f"\nOr use --synthetic_data with train.py for testing.")
