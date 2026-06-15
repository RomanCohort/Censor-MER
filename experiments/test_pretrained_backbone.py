"""
Test if Censor backbone weights are actually loaded from pretrained.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import Censor

print("=" * 60)
print("Censor Backbone Pretrained Check")
print("=" * 60)

# Create model with pretrained backbone
model = Censor(pretrained_backbone=True, verbose=True)

# Check fast pathway (R3D-18)
fast_path = model.fast_pathway
print("\n--- Fast Pathway (R3D-18) ---")
print(f"Type: {type(fast_path).__name__}")

# Check if weights differ from random init
for name, param in fast_path.named_parameters():
    if 'conv' in name and 'weight' in name:
        w = param.data
        print(f"  {name}: shape={w.shape}, mean={w.mean().item():.4f}, std={w.std().item():.4f}")
        # Kinetics pretrained weights typically have non-zero mean
        if abs(w.mean().item()) > 0.01:
            print(f"    -> Likely PRETRAINED (non-zero mean)")
        else:
            print(f"    -> Likely RANDOM INIT (zero-ish mean)")
        break

# Check slow pathway (Swin3D-T)
slow_path = model.slow_pathway
print("\n--- Slow Pathway (Swin3D-T) ---")
print(f"Type: {type(slow_path).__name__}")

for name, param in slow_path.named_parameters():
    if 'patch_embed' in name and 'weight' in name:
        w = param.data
        print(f"  {name}: shape={w.shape}, mean={w.mean().item():.4f}, std={w.std().item():.4f}")
        if abs(w.mean().item()) > 0.01:
            print(f"    -> Likely PRETRAINED")
        else:
            print(f"    -> Likely RANDOM INIT")
        break

print("\n" + "=" * 60)
print("Conclusion:")
print("If weights show non-zero mean/std, backbone is pretrained.")
print("If weights are zero-ish, backbone failed to load pretrained.")
print("=" * 60)