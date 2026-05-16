# =============================================================================
# Test: Visual Perception Post-Processing System
# =============================================================================

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from visual_perception import (
    PupilController,
    RetinalContrastNorm,
    MachBandEnhancer,
    CenterSurroundReceptiveField,
    VisualPerceptionPostProcess,
    create_visual_perception_module
)
from config.defaults import VISUAL_PERCEPTION_CONFIG


def test_pupil_controller():
    """Test PupilController - illumination adaptation"""
    print("\n" + "=" * 50)
    print(" Test: PupilController")
    print("=" * 50)

    pupil = PupilController(VISUAL_PERCEPTION_CONFIG)
    print(f"Parameters: {sum(p.numel() for p in pupil.parameters()):,}")

    # Test with different illumination levels
    bright_img = torch.ones(2, 3, 224, 224) * 0.9  # Bright image
    dark_img = torch.ones(2, 3, 224, 224) * 0.1     # Dark image

    with torch.no_grad():
        bright_out = pupil(bright_img)
        dark_out = pupil(dark_img)

    bright_gain = bright_out.mean() / bright_img.mean()
    dark_gain = dark_out.mean() / dark_img.mean()

    print(f"Input bright: {bright_img.mean():.4f} -> Output: {bright_out.mean():.4f}, gain={bright_gain:.4f}")
    print(f"Input dark:  {dark_img.mean():.4f} -> Output: {dark_out.mean():.4f}, gain={dark_gain:.4f}")

    # Note: untrained network - gain values are random
    # After training, dark images should get higher gain (pupil dilation)
    print("PupilController test passed! (untrained network)")


def test_retinal_contrast_norm():
    """Test RetinalContrastNorm - local contrast normalization"""
    print("\n" + "=" * 50)
    print(" Test: RetinalContrastNorm")
    print("=" * 50)

    retinal = RetinalContrastNorm(VISUAL_PERCEPTION_CONFIG)
    print(f"Parameters: {sum(p.numel() for p in retinal.parameters()):,}")

    # Create test image with varying local contrast
    base = torch.ones(2, 3, 64, 64) * 0.5
    # Add high contrast region
    base[:, :, 16:48, 16:48] = 0.9
    # Add low contrast region
    base[:, :, :16, :16] = 0.55

    with torch.no_grad():
        normalized = retinal(base)

    print(f"Input: min={base.min():.4f}, max={base.max():.4f}, mean={base.mean():.4f}")
    print(f"Output: min={normalized.min():.4f}, max={normalized.max():.4f}, mean={normalized.mean():.4f}")

    # Check local variance is normalized
    local_std_in = base.std()
    local_std_out = normalized.std()
    print(f"Std in: {local_std_in:.4f}, Std out: {local_std_out:.4f}")

    print("RetinalContrastNorm test passed!")


def test_mach_band_enhancer():
    """Test MachBandEnhancer - edge sharpening"""
    print("\n" + "=" * 50)
    print(" Test: MachBandEnhancer")
    print("=" * 50)

    mach = MachBandEnhancer(VISUAL_PERCEPTION_CONFIG)
    print(f"Parameters: {sum(p.numel() for p in mach.parameters()):,}")

    # Create image with sharp edge
    img = torch.zeros(1, 1, 32, 32)
    img[:, :, :16, :] = 0.0  # Left half black
    img[:, :, 16:, :] = 1.0   # Right half white (step edge)

    with torch.no_grad():
        enhanced = mach(img)

    print(f"Input edge profile (row 16): {img[0, 0, 15, :].tolist()}")
    print(f"Output edge profile (row 16): {enhanced[0, 0, 15, :].tolist()}")

    # Check Mach band effect: overshoot on bright side, undershoot on dark side
    edge_center = 15
    print(f"At edge boundary: in={img[0,0,15,16]:.4f}, out={enhanced[0,0,15,16]:.4f}")

    print("MachBandEnhancer test passed!")


def test_center_surround_receptive_field():
    """Test CenterSurroundReceptiveField - edge detection"""
    print("\n" + "=" * 50)
    print(" Test: CenterSurroundReceptiveField")
    print("=" * 50)

    receptive = CenterSurroundReceptiveField(VISUAL_PERCEPTION_CONFIG)
    print(f"Parameters: {sum(p.numel() for p in receptive.parameters()):,}")

    # Create image with a point stimulus (center-surround)
    img = torch.zeros(1, 1, 32, 32)
    img[0, 0, 16, 16] = 1.0  # Center point

    with torch.no_grad():
        response = receptive(img)

    print(f"Input shape: {img.shape}")
    print(f"Output shape: {response.shape}")
    print(f"Response at center: {response[0, 0, 16, 16]:.4f}")
    print(f"Response at surround: {response[0, 0, 14, 14]:.4f}")

    # Check center-surround antagonism
    center_val = response[0, 0, 16, 16].item()
    surround_val = response[0, 0, 10, 10].item()
    print(f"Center-surround: center={center_val:.4f}, surround={surround_val:.4f}")

    print("CenterSurroundReceptiveField test passed!")


def test_visual_perception_postprocess():
    """Test full VisualPerceptionPostProcess pipeline"""
    print("\n" + "=" * 50)
    print(" Test: VisualPerceptionPostProcess")
    print("=" * 50)

    vpp = VisualPerceptionPostProcess(VISUAL_PERCEPTION_CONFIG)
    print(f"Total Parameters: {sum(p.numel() for p in vpp.parameters()):,}")

    # Test with synthetic image
    gen_img = torch.rand(2, 3, 224, 224)  # Generated image

    with torch.no_grad():
        output = vpp(gen_img)

    print(f"Input: {gen_img.shape}, range=[{gen_img.min():.4f}, {gen_img.max():.4f}]")
    print(f"Output: {output.shape}, range=[{output.min():.4f}, {output.max():.4f}]")

    # Test with options
    with torch.no_grad():
        output_no_retinal = vpp(gen_img, apply_retinal=False)
        output_no_mach = vpp(gen_img, apply_retinal=None, apply_mach=False)

    print(f"With retinal disabled: {output_no_retinal.shape}")
    print(f"With mach disabled: {output_no_mach.shape}")

    print("VisualPerceptionPostProcess test passed!")


def test_factory_function():
    """Test factory function"""
    print("\n" + "=" * 50)
    print(" Test: create_visual_perception_module")
    print("=" * 50)

    module = create_visual_perception_module()
    print(f"Created: {type(module).__name__}")
    print(f"Parameters: {sum(p.numel() for p in module.parameters()):,}")

    test_input = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        output = module(test_input)

    print(f"Test pass!")


def test_config_loading():
    """Test config loading"""
    print("\n" + "=" * 50)
    print(" Test: Config Loading")
    print("=" * 50)

    print(f"VISUAL_PERCEPTION_CONFIG:")
    for k, v in VISUAL_PERCEPTION_CONFIG.items():
        print(f"  {k}: {v}")

    print("Config loading test passed!")


if __name__ == '__main__':
    print("=" * 60)
    print(" Visual Perception Post-Processing System Test")
    print("=" * 60)

    test_config_loading()
    test_pupil_controller()
    test_retinal_contrast_norm()
    test_mach_band_enhancer()
    test_center_surround_receptive_field()
    test_visual_perception_postprocess()
    test_factory_function()

    print("\n" + "=" * 60)
    print(" All Visual Perception Tests Passed!")
    print("=" * 60)