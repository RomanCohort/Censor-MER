"""
Quick sanity check for all supplementary experiments.
Run this FIRST on AutoDL to verify setup before running full experiments.

Usage:
  python experiments/sanity_check.py

Expected: All checks pass in ~2 minutes.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check(condition, name, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    return condition


def main():
    print("=" * 60)
    print("CENSOR Supplementary Experiments - Sanity Check")
    print("=" * 60)

    import torch
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checks_passed = 0
    checks_total = 0

    # 1. GPU check
    print("\n1. Hardware")
    checks_total += 2
    checks_passed += check(torch.cuda.is_available(), "CUDA available")
    if torch.cuda.is_available():
        checks_passed += check(
            torch.cuda.get_device_properties(0).total_memory > 10e9,
            "GPU VRAM >= 10GB",
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        checks_passed += 0

    # 2. Data check
    print("\n2. Data Paths")
    data_paths = {
        'CASME II': '/root/autodl-tmp/data/CASME2',
        'SAMM': '/root/data/SAMM/SAMM',
        'SMIC': '/root/SMIC_all_cropped',
    }
    for name, path in data_paths.items():
        checks_total += 1
        checks_passed += check(os.path.isdir(path), f"{name} exists", path)

    # 3. Import check
    print("\n3. Module Imports")
    import_tests = [
        ("config.defaults", "Configuration"),
        ("model.preprocessing", "Preprocessing modules"),
        ("model.backbones", "Backbone networks"),
        ("model.attention", "Attention modules"),
        ("model.fusion", "Fusion module"),
        ("model.moe_head", "MoE gating"),
        ("model.biomimetic_enhance", "Sparse control"),
        ("dataset", "CASME II dataset"),
        ("dataset_samm", "SAMM dataset"),
        ("dataset_smic", "SMIC dataset"),
    ]
    for module_name, desc in import_tests:
        checks_total += 1
        try:
            __import__(module_name)
            checks_passed += check(True, desc, module_name)
        except ImportError as e:
            checks_passed += check(False, desc, str(e))

    # 4. Censor model import (from main.py)
    print("\n4. Censor Model")
    checks_total += 2
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "main_module", str(Path(__file__).parent.parent / "main.py")
        )
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        Censor = main_module.Censor
        checks_passed += check(True, "Censor class importable", "main.py")

        # Test instantiation
        try:
            model = Censor(pretrained_backbone=False, verbose=False)
            checks_passed += check(True, "Censor instantiation", f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
        except Exception as e:
            checks_passed += check(False, "Censor instantiation", str(e)[:80])
    except Exception as e:
        checks_passed += check(False, "Censor class importable", str(e)[:80])
        checks_passed += check(False, "Censor instantiation", "skipped")

    # 5. Multi-scale 3D ResNet
    print("\n5. Multi-scale 3D ResNet (SOTA reproduction)")
    checks_total += 2
    try:
        from experiments.exp1b_multiscale_3d_resnet import MultiScale3DResNet
        checks_passed += check(True, "MultiScale3DResNet importable")

        try:
            model = MultiScale3DResNet(num_classes=4, pretrained=False)
            x = torch.randn(2, 3, 16, 224, 224)
            if device.type == 'cuda':
                model = model.to(device)
                x = x.to(device)
            with torch.no_grad():
                out = model(x)
            checks_passed += check(
                out.shape == (2, 4),
                "Forward pass",
                f"input {x.shape} -> output {out.shape}"
            )
        except Exception as e:
            checks_passed += check(False, "Forward pass", str(e)[:80])
    except ImportError as e:
        checks_passed += check(False, "MultiScale3DResNet importable", str(e)[:80])
        checks_passed += check(False, "Forward pass", "skipped")

    # 6. Alternative fusion modules
    print("\n6. Alternative Fusion Modules")
    alt_fusions = ['ConcatFusion', 'AttentionFusion', 'FeatureEnsemble']
    for cls_name in alt_fusions:
        checks_total += 1
        try:
            cls = getattr(
                __import__('experiments.exp7_deep_ablation', fromlist=[cls_name]),
                cls_name
            )
            model = cls(num_classes=4)
            fast = torch.randn(2, 512)
            slow = torch.randn(2, 768)
            out = model(fast, slow)
            checks_passed += check(
                out.shape == (2, 4),
                cls_name,
                f"({2},{512})+({2},{768}) -> {out.shape}"
            )
        except Exception as e:
            checks_passed += check(False, cls_name, str(e)[:80])

    # 7. Checkpoint existence
    print("\n7. Saved Checkpoints")
    ckpt_dirs = [
        '/root/autodl-tmp/data/checkpoints',
        '/root/autodl-tmp/checkpoints',
        '/root/checkpoints',
    ]
    found_ckpt = False
    for d in ckpt_dirs:
        if os.path.isdir(d):
            pt_files = list(Path(d).glob('*.pt'))
            if pt_files:
                found_ckpt = True
                checks_total += 1
                checks_passed += check(True, "Checkpoint found", str(pt_files[:3]))
                break
    if not found_ckpt:
        checks_total += 1
        checks_passed += check(False, "Checkpoint found",
                                "Need to upload trained checkpoint")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)

    if checks_passed >= checks_total - 2:
        print("\nReady to run supplementary experiments!")
        print("Run: nohup bash experiments/run_supplementary.sh &")
    else:
        print("\nSome checks failed. Fix issues before running experiments.")
        print("Common fixes:")
        print("  - Upload data to /root/autodl-tmp/data/CASME2 etc.")
        print("  - Install missing packages: pip install sklearn scipy")
        print("  - Upload checkpoint to /root/autodl-tmp/data/checkpoints/")


if __name__ == '__main__':
    main()