"""
Experiment 1: OFF-ApexNet LOSO Reproduction
=============================================
Reproduce OFF-ApexNet under same LOSO protocol for fair SOTA comparison.

Usage:
    python experiments/exp1_offapexnet.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Experiment 1: OFF-ApexNet LOSO Reproduction")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    from torchvision.models import vgg16, VGG16_Weights
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install torchvision: pip install torchvision")
    sys.exit(1)

class OFFApexNet(nn.Module):
    """OFF-ApexNet: Two-stream VGG-16 for MER (apex frame only)."""

    def __init__(self, num_classes=4):
        super().__init__()
        # RGB stream (single apex frame)
        vgg_rgb = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.rgb_features = vgg_rgb.features
        self.rgb_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Optical flow stream (2-channel)
        self.flow_features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *vgg_rgb.features[7:],
        )
        self.flow_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Initialize flow conv
        with torch.no_grad():
            rgb_conv1 = vgg_rgb.features[0]
            self.flow_features[0].weight.data = rgb_conv1.weight.data[:, :2].mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
            self.flow_features[0].bias.data = rgb_conv1.bias.data

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

    def forward(self, rgb_apex, flow_apex):
        rgb_feat = self.rgb_features(rgb_apex)
        rgb_feat = self.rgb_pool(rgb_feat).view(rgb_feat.size(0), -1)

        flow_feat = self.flow_features(flow_apex)
        flow_feat = self.flow_pool(flow_feat).view(flow_feat.size(0), -1)

        fused = torch.cat([rgb_feat, flow_feat], dim=1)
        return self.classifier(fused)


# Architecture test
print("\nTesting OFF-ApexNet architecture...")
model = OFFApexNet(num_classes=4)
rgb = torch.randn(1, 3, 224, 224)
flow = torch.randn(1, 2, 224, 224)
out = model(rgb, flow)
params = sum(p.numel() for p in model.parameters()) / 1e6

print(f"  RGB input: {rgb.shape}")
print(f"  Flow input: {flow.shape}")
print(f"  Output: {out.shape}")
print(f"  Parameters: {params:.2f}M")

# Check CASME II data
data_root = Path(__file__).parent.parent / 'data' / 'CASME_II'
for path in ['/root/autodl-tmp/data/CASME_II', '/data/CASME_II', './data/CASME_II']:
    if Path(path).exists():
        data_root = Path(path)
        break

print("\n" + "=" * 60)
if data_root.exists():
    print(f"CASME II found at: {data_root}")
    print("TODO: Implement full LOSO cross-validation")
else:
    print("CASME II data NOT found")
    print("Please place CASME II at ./data/CASME_II/")
    print("Required structure:")
    print("  videos/")
    print("  labels.csv")
print("=" * 60)

# Save architecture info
output_dir = Path(__file__).parent.parent / 'results'
output_dir.mkdir(exist_ok=True)

with open(output_dir / 'exp1_offapexnet.txt', 'w') as f:
    f.write(f"OFF-ApexNet Architecture Test\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Parameters: {params:.2f}M\n")
    f.write(f"RGB input: (B, 3, 224, 224)\n")
    f.write(f"Flow input: (B, 2, 224, 224)\n")
    f.write(f"Output: (B, 4)\n")
    f.write(f"CASME II path: {data_root}\n")
    f.write(f"Status: Architecture verified\n")

print(f"\nSaved to: {output_dir / 'exp1_offapexnet.txt'}")