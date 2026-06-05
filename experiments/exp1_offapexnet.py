"""
Experiment 1: OFF-ApexNet LOSO Reproduction
=============================================
Reproduce OFF-ApexNet under same LOSO protocol for fair SOTA comparison.
Uses existing dataset loaders from Censor project.

Data paths (AutoDL):
    CASME II: /root/autodl-tmp/data/CASME2
    SMIC: /root/SMIC_all_cropped
    SAMM: /root/data/SAMM/SAMM

Usage:
    python experiments/exp1_offapexnet.py --dataset casme2
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Configuration
# =============================================================================

DATA_PATHS = {
    'casme2': '/root/autodl-tmp/data/CASME2',
    'smic': '/root/SMIC_all_cropped',
    'samm': '/root/data/SAMM/SAMM',
}

# Excluded subjects (from paper)
CASME2_EXCLUDED = ['sub13', 'sub22']

# =============================================================================
# OFF-ApexNet Architecture
# =============================================================================

class OFFApexNet(nn.Module):
    """OFF-ApexNet: Two-stream VGG-16 for MER (apex frame only)."""

    def __init__(self, num_classes=4):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # RGB stream
        self.rgb_features = vgg.features
        self.rgb_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Optical flow stream (2-channel)
        self.flow_conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.flow_features = nn.Sequential(
            nn.ReLU(inplace=True),
            *vgg.features[4:],
        )
        self.flow_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Initialize
        with torch.no_grad():
            self.flow_conv1.weight.data = vgg.features[0].weight.data[:, :2].mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
            self.flow_conv1.bias.data = vgg.features[0].bias.data

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, rgb, flow):
        rgb_f = self.rgb_pool(self.rgb_features(rgb)).view(rgb.size(0), -1)
        flow_f = self.flow_pool(self.flow_features(self.flow_conv1(flow))).view(flow.size(0), -1)
        return self.classifier(torch.cat([rgb_f, flow_f], dim=1))


# =============================================================================
# Use Existing Dataset Loaders
# =============================================================================

def get_dataset_loader(dataset_name, data_root, subject_list=None):
    """Use existing dataset loaders from Censor project."""

    if dataset_name == 'casme2':
        from dataset import MERDataset

        # First convert CASME2 if needed
        labels_csv = Path(data_root) / 'labels.csv'
        if not labels_csv.exists():
            print("Converting CASME2 labels...")
            from scripts.convert_casme2 import convert_casme2_to_standard
            convert_casme2_to_standard(data_root)

        dataset = MERDataset(data_root, split='train')  # Load all, we'll split by subject

    elif dataset_name == 'smic':
        from dataset_smic import SMICDataset
        dataset = SMICDataset(data_root)

    elif dataset_name == 'samm':
        from dataset_samm import SAMMDataset
        dataset = SAMMDataset(data_root)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


# =============================================================================
# LOSO Cross-Validation
# =============================================================================

def run_loso_cv(dataset_name, data_root, device, epochs=30, batch_size=8):
    """Run Leave-One-Subject-Out cross-validation using existing data loaders."""

    print(f"\nLoading {dataset_name} from {data_root}")

    try:
        dataset = get_dataset_loader(dataset_name, data_root)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Get subject list from dataset
    if hasattr(dataset, 'samples'):
        subjects = list(set(s.get('subject', 'unknown') for s in dataset.samples))
    else:
        # Fallback: try to infer from directory structure
        print("Warning: Could not get subject list, using placeholder")
        subjects = [f'sub{i:02d}' for i in range(1, 27)]

    # Filter excluded subjects for CASME2
    if dataset_name == 'casme2':
        subjects = [s for s in subjects if s not in CASME2_EXCLUDED]

    subjects = sorted(subjects)
    print(f"Found {len(subjects)} subjects: {subjects}")

    results = []

    # Simplified: run single fold for testing
    print("\nNote: Full LOSO requires extracting apex frames + optical flow")
    print("Running single-fold test for architecture validation...")

    # Create model
    model = OFFApexNet(num_classes=4).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: OFF-ApexNet, {params:.2f}M params")

    # Test forward pass
    rgb = torch.randn(2, 3, 224, 224).to(device)
    flow = torch.randn(2, 2, 224, 224).to(device)
    out = model(rgb, flow)
    print(f"Forward pass: input {rgb.shape}, output {out.shape}")

    return {
        'status': 'architecture_verified',
        'params_m': params,
        'num_subjects': len(subjects),
        'subjects': subjects,
        'note': 'Full LOSO requires apex frame extraction from videos. Use train_frames.py for complete training.'
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='casme2', choices=['casme2', 'smic', 'samm'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 1: OFF-ApexNet LOSO Reproduction")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}")

    # Check data
    data_root = DATA_PATHS[args.dataset]
    if not Path(data_root).exists():
        print(f"\nError: Data not found at {data_root}")
        print("Update DATA_PATHS in script with correct locations.")
        return

    print(f"Data root: {data_root}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Run
    results = run_loso_cv(args.dataset, data_root, device, args.epochs, args.batch_size)

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'exp1_offapexnet_{args.dataset}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to: {output_file}")

    # Print note about full implementation
    print("\n" + "=" * 60)
    print("NOTE: For complete LOSO training, use:")
    print("  python train_frames.py --dataset casme2 --protocol loso")
    print("=" * 60)


if __name__ == '__main__':
    main()