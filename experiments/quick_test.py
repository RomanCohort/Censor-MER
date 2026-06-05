"""
Quick Test Script for AutoDL Experiments
=========================================
Tests each experiment component independently for faster debugging.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Quick Test Script")
print("=" * 60)

# Test 1: Import Censor
print("\n[Test 1] Import Censor...")
try:
    from main import Censor
    print("  ✓ Censor imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create Censor instances
print("\n[Test 2] Create Censor instances...")
configs = [
    ('Full Model', {}),
    ('Fast-only', {'single_path': 'fast'}),
    ('No-rPPG', {'no_rppg': True}),
]

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

models = {}
for name, kwargs in configs:
    try:
        model = Censor(verbose=False, **kwargs)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ {name}: {params:.2f}M params")
        models[name] = model
    except Exception as e:
        print(f"  ✗ {name}: {e}")

# Test 3: Forward pass
print("\n[Test 3] Forward pass test...")
B, C, T, H, W = 1, 3, 16, 224, 224
video = torch.randn(B, C, T, H, W)
print(f"  Input shape: {video.shape}")

for name, model in models.items():
    try:
        model.eval()
        with torch.no_grad():
            start = time.time()
            out = model(video)
            elapsed = (time.time() - start) * 1000
        print(f"  ✓ {name}: output shape {out.shape}, {elapsed:.1f}ms")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

# Test 4: OFF-ApexNet architecture
print("\n[Test 4] OFF-ApexNet architecture...")
try:
    from torchvision.models import vgg16, VGG16_Weights

    class OFFApexNet(torch.nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.rgb_features = vgg.features
            self.flow_features = torch.nn.Sequential(
                torch.nn.Conv2d(2, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                *vgg.features[4:],  # Skip first conv+relu
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7 * 2, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(4096, num_classes),
            )

        def forward(self, rgb, flow):
            rgb_f = self.rgb_features(rgb).view(rgb.size(0), -1)
            flow_f = self.flow_features(flow).view(flow.size(0), -1)
            return self.classifier(torch.cat([rgb_f, flow_f], 1))

    model = OFFApexNet()
    rgb = torch.randn(1, 3, 224, 224)
    flow = torch.randn(1, 2, 224, 224)
    out = model(rgb, flow)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ OFF-ApexNet: {params:.2f}M params, output {out.shape}")
except Exception as e:
    print(f"  ✗ OFF-ApexNet: {e}")

# Test 5: rPPG HR estimation
print("\n[Test 5] rPPG HR estimation...")
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq

    def compute_hr(rppg, fps=200):
        nyquist = fps / 2
        b, a = signal.butter(4, [0.5/nyquist, 4.0/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, rppg)
        n = len(filtered)
        yf = fft(filtered)
        xf = fftfreq(n, 1/fps)
        mask = (xf >= 0.5) & (xf <= 4.0)
        power = np.abs(yf[mask])
        freqs = xf[mask]
        hr = freqs[np.argmax(power)] * 60
        return hr

    # Synthetic test
    fps = 200
    t = np.arange(80) / fps
    true_hr = 75
    rppg = np.sin(2 * np.pi * true_hr/60 * t) + np.random.randn(80) * 0.1
    est_hr = compute_hr(rppg, fps)
    print(f"  ✓ rPPG HR: true={true_hr} BPM, estimated={est_hr:.1f} BPM")
except ImportError:
    print("  ✗ scipy not installed: pip install scipy")
except Exception as e:
    print(f"  ✗ rPPG HR: {e}")

# Test 6: Dataset loader
print("\n[Test 6] Dataset loader...")
try:
    from dataset import MERDataset
    print("  ✓ MERDataset imported (requires CASME II data to instantiate)")
except Exception as e:
    print(f"  ✗ Dataset: {e}")

print("\n" + "=" * 60)
print("Quick test completed!")
print("=" * 60)
