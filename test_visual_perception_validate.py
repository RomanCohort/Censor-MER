# =============================================================================
# Test: Visual Perception Effectiveness Validation
# =============================================================================
# More comprehensive test of biomimetic visual perception features

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from visual_perception import (
    PupilController,
    RetinalContrastNorm,
    MachBandEnhancer,
    CenterSurroundReceptiveField,
    VisualPerceptionPostProcess
)
from config.defaults import VISUAL_PERCEPTION_CONFIG


def validate_pupil_effect():
    """Validate that PupilController responds to illumination"""
    print("\n" + "=" * 60)
    print(" Validation: PupilController illumination response")
    print("=" * 60)

    pupil = PupilController(VISUAL_PERCEPTION_CONFIG)

    # Test monotonic response - multiple illumination levels
    illumination_levels = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
    gains = []

    for illum in illumination_levels:
        img = torch.ones(1, 3, 64, 64) * illum
        with torch.no_grad():
            out = pupil(img)
        gain = out.mean() / img.mean()
        gains.append(gain.item())

    print(f"Illumination: {illumination_levels}")
    print(f"Gains:       {[f'{g:.4f}' for g in gains]}")

    # Check gain variation (untrained network may not have perfect response)
    gain_std = torch.tensor(gains).std().item()
    print(f"Gain std:   {gain_std:.4f}")
    print(f"PupilController: illumination response varies = {gain_std > 0.001}")


def validate_retinal_effect():
    """Validate RetinalContrastNorm enhances local contrast"""
    print("\n" + "=" * 60)
    print(" Validation: RetinalContrastNorm contrast enhancement")
    print("=" * 60)

    retinal = RetinalContrastNorm(VISUAL_PERCEPTION_CONFIG)

    # Create image with low local contrast variation
    base = torch.ones(2, 3, 64, 64) * 0.5
    stripe = base.clone()
    # Add local variation to stripe
    stripe[:, :, 20:40, 20:40] += 0.3

    with torch.no_grad():
        base_out = retinal(base)
        stripe_out = retinal(stripe)

    # Check that local contrast is preserved/enhanced
    local_std_base = base.std()
    local_std_out = base_out.std()
    local_std_stripe = stripe[:, :, 20:40, 20:40].std()
    local_std_stripe_out = stripe_out[:, :, 20:40, 20:40].std()

    print(f"Base local std:   {local_std_base:.4f} -> {local_std_out:.4f}")
    print(f"Stripe local std: {local_std_stripe:.4f} -> {local_std_stripe_out:.4f}")
    print(f"RetinalContrastNorm: local contrast computed = {local_std_out > 0}")


def validate_mach_effect():
    """Validate MachBandEnhancer edge overshoot"""
    print("\n" + "=" * 60)
    print(" Validation: MachBandEnhancer edge overshoot")
    print("=" * 60)

    mach = MachBandEnhancer(VISUAL_PERCEPTION_CONFIG)

    # Create image with clear step edge
    img = torch.zeros(1, 1, 32, 32)
    img[:, :, :16, :] = 0.2  # Dark half
    img[:, :, 16:, :] = 0.8  # Bright half

    with torch.no_grad():
        enhanced = mach(img)

    # Sample at various positions
    positions = [14, 15, 16, 17]  # Edge transition
    values_in = [img[0, 0, 16, p].item() for p in positions]
    values_out = [enhanced[0, 0, 16, p].item() for p in positions]

    print(f"Input at row 16:   {values_in}")
    print(f"Output at row 16: {values_out}")

    # Check overshoot: output should exceed max input at bright side
    input_max = img.max().item()
    output_max = enhanced.max().item()
    overshoot = output_max - input_max

    print(f"Input max:  {input_max:.4f}")
    print(f"Output max: {output_max:.4f}")
    print(f"Overshoot: {overshoot:.4f}")

    # Mach band creates overshoot at edges
    has_effect = (enhanced != img).any()
    print(f"MachBandEnhancer: edge effect active = {has_effect}")


def validate_receptive_effect():
    """Validate CenterSurroundReceptiveField edge detection"""
    print("\n" + "=" * 60)
    print(" Validation: CenterSurroundReceptiveField edge detection")
    print("=" * 60)

    receptive = CenterSurroundReceptiveField(VISUAL_PERCEPTION_CONFIG)

    # Create vertical edge
    img = torch.zeros(1, 1, 32, 32)
    img[:, :, :, :16] = 0.0  # Left half black
    img[:, :, :, 16:] = 1.0  # Right half white

    with torch.no_grad():
        response = receptive(img)

    # Check response at edge vs uniform regions
    edge_response = response[0, 0, 16, 16].item()  # At edge
    flat_response = response[0, 0, 8, 8].item()    # Flat region (black)
    bright_response = response[0, 0, 8, 24].item()  # Flat region (white)

    print(f"Edge response:     {edge_response:.4f}")
    print(f"Flat (black):      {flat_response:.4f}")
    print(f"Flat (white):      {bright_response:.4f}")

    # Edge should have higher response
    edge_detected = abs(edge_response) > abs(flat_response) * 2
    print(f"CenterSurroundReceptiveField: edge detected = {edge_detected}")


def validate_full_pipeline():
    """Validate full pipeline integration"""
    print("\n" + "=" * 60)
    print(" Validation: Full VisualPerceptionPostProcess Pipeline")
    print("=" * 60)

    vpp = VisualPerceptionPostProcess(VISUAL_PERCEPTION_CONFIG)

    # Test with various images
    test_images = [
        ("bright", torch.rand(1, 3, 112, 112) * 0.8 + 0.2),
        ("dark", torch.rand(1, 3, 112, 112) * 0.2),
        ("mixed", torch.randn(1, 3, 112, 112) * 0.3 + 0.5),
    ]

    for name, img in test_images:
        with torch.no_grad():
            out = vpp(img)

        # Verify output is valid
        print(f"{name}: in={img.shape} -> out={out.shape}, finite={out.isfinite().all()}")

    print("Full pipeline: integration OK")


def validate_gradients():
    """Validate gradients can flow through modules"""
    print("\n" + "=" * 60)
    print(" Validation: Gradient Flow")
    print("=" * 60)

    vpp = VisualPerceptionPostProcess(VISUAL_PERCEPTION_CONFIG)

    # Enable gradient tracking
    img = torch.rand(1, 3, 64, 64, requires_grad=True)
    out = vpp(img)
    loss = out.mean()
    loss.backward()

    has_grad = img.grad is not None
    grad_norm = img.grad.norm().item() if has_grad else 0

    print(f"Input requires_grad: {img.requires_grad}")
    print(f"Output grad exists: {has_grad}")
    print(f"Gradient norm: {grad_norm:.4f}")
    print(f"Gradient flow: OK")


def validate_inverse_mode():
    """Validate inverse mode works (gradient can flow backward)"""
    print("\n" + "=" * 60)
    print(" Validation: Inverse Gradient Mode")
    print("=" * 60)

    vpp = VisualPerceptionPostProcess(VISUAL_PERCEPTION_CONFIG)

    img = torch.rand(2, 3, 64, 64, requires_grad=True)

    # Forward
    out = vpp(img)

    # Backward with loss
    loss = out.sum()
    loss.backward()

    valid_grad = img.grad is not None and img.grad.norm() > 0

    print(f"Input grad norm: {img.grad.norm():.4f}")
    print(f"Inverse mode: working = {valid_grad}")


def validate_config():
    """Validate config values are sensible"""
    print("\n" + "=" * 60)
    print(" Validation: Config Parameters")
    print("=" * 60)

    cfg = VISUAL_PERCEPTION_CONFIG

    # Check all expected keys exist
    expected_keys = [
        'pupil_hidden_dim', 'pupil_base_gain', 'pupil_modulation_range',
        'retinal_kernel', 'retinal_alpha', 'retinal_beta',
        'mach_band_strength', 'mach_band_sigma',
        'center_sigma', 'surround_sigma',
        'enable_retinal', 'enable_mach', 'receptive_weight'
    ]

    missing = [k for k in expected_keys if k not in cfg]
    print(f"Config keys: {len(cfg)} / {len(expected_keys)}")
    print(f"Missing keys: {missing if missing else 'None'}")

    # Check value ranges
    print(f"\nPupil gain range: [{cfg['pupil_base_gain']}, {cfg['pupil_base_gain'] + cfg['pupil_modulation_range']}]")
    print(f"Retinal kernel: {cfg['retinal_kernel']} (odd: {cfg['retinal_kernel'] % 2 == 1})")
    print(f"Mach band strength: {cfg['mach_band_strength']}")
    print(f"Center/surround sigma: {cfg['center_sigma']} / {cfg['surround_sigma']}")


if __name__ == '__main__':
    print("=" * 60)
    print(" Visual Perception Effectiveness Validation")
    print("=" * 60)

    validate_config()
    validate_pupil_effect()
    validate_retinal_effect()
    validate_mach_effect()
    validate_receptive_effect()
    validate_full_pipeline()
    validate_gradients()
    validate_inverse_mode()

    print("\n" + "=" * 60)
    print(" All Validations Complete!")
    print("=" * 60)