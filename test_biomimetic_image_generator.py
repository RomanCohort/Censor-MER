# =============================================================================
# Test: Biomimetic Image Generator
# =============================================================================

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model.biomimetic_image_generator import (
    BiomimeticImageGenerator,
    LightShadowGenerator,
    BiomimeticGenerationPipeline,
    AUToIllumination,
    IlluminativeRenderer,
    create_biomimetic_generator
)
from config.defaults import VISUAL_PERCEPTION_CONFIG, AU_DECODER_CONFIG, DATA_CONFIG


def test_au_to_illumination():
    """Test AU to illumination mapping"""
    print("\n" + "=" * 50)
    print(" Test: AUToIllumination")
    print("=" * 50)

    au_to_illum = AUToIllumination(num_aus=28)
    print(f"Parameters: {sum(p.numel() for p in au_to_illum.parameters()):,}")

    # Test with random AU intensities
    au_intensities = torch.rand(2, 16, 28)  # (B, T, num_aus)

    with torch.no_grad():
        illum_params = au_to_illum(au_intensities)

    print(f"Input AU: {au_intensities.shape}")
    print(f"Output: {illum_params.shape}")
    print(f"Illum params: {illum_params[0].tolist()}")

    print("AUToIllumination passed!")


def test_illuminative_renderer():
    """Test illumination rendering"""
    print("\n" + "=" * 50)
    print(" Test: IlluminativeRenderer")
    print("=" * 50)

    renderer = IlluminativeRenderer(image_size=224)

    # Test with base image
    image = torch.rand(2, 3, 224, 224)  # RGB in [0, 1]
    illum_params = torch.rand(2, 4)  # [scale, shadow, rim, ambient]

    with torch.no_grad():
        lit_image = renderer(image, illum_params)

    print(f"Input: {image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
    print(f"Output: {lit_image.shape}, range=[{lit_image.min():.3f}, {lit_image.max():.3f}]")

    print("IlluminativeRenderer passed!")


def test_light_shadow_generator():
    """Test light/shadow generator"""
    print("\n" + "=" * 50)
    print(" Test: LightShadowGenerator")
    print("=" * 50)

    ls_gen = LightShadowGenerator(image_size=224)
    print(f"Parameters: {sum(p.numel() for p in ls_gen.parameters()):,}")

    # Test with image and AU regions
    image = torch.rand(2, 3, 224, 224)
    au_regions = torch.rand(2, 3)  # forehead, cheek, mouth

    with torch.no_grad():
        lit_image = ls_gen(image, au_regions)

    print(f"Input: {image.shape}")
    print(f"Output: {lit_image.shape}")

    print("LightShadowGenerator passed!")


def test_biomimetic_image_generator():
    """Test main image generator"""
    print("\n" + "=" * 50)
    print(" Test: BiomimeticImageGenerator")
    print("=" * 50)

    # Create generator with config
    config = {
        'fast_dim': 512,
        'slow_dim': 768,
        'fused_dim': 1024,
        'latent_dim': 512,
    }
    generator = BiomimeticImageGenerator(config)
    print(f"Parameters: {sum(p.numel() for p in generator.parameters()):,}")

    # Test with dual-pathway features
    fast_feat = torch.randn(2, 512)  # (B, 512)
    slow_feat = torch.randn(2, 768)  # (B, 768)

    with torch.no_grad():
        # With dual-pathway
        image1 = generator(fast_feat=fast_feat, slow_feat=slow_feat, au_intensities=None, apply_visual_perception=False)
        print(f"Dual-pathway: {image1.shape}, range=[{image1.min():.3f}, {image1.max():.3f}]")

        # With AU
        au_intensities = torch.rand(2, 16, 28)
        image2 = generator(fast_feat=fast_feat, slow_feat=slow_feat, au_intensities=au_intensities, apply_visual_perception=True)
        print(f"With AU: {image2.shape}, range=[{image2.min():.3f}, {image2.max():.3f}]")

    print("BiomimeticImageGenerator passed!")


def test_full_pipeline():
    """Test full generation pipeline"""
    print("\n" + "=" * 50)
    print(" Test: BiomimeticGenerationPipeline")
    print("=" * 50)

    pipeline = BiomimeticGenerationPipeline()
    print(f"Total Parameters: {sum(p.numel() for p in pipeline.parameters()):,}")

    # Test forward with dual-pathway features
    fast_feat = torch.randn(2, 512)
    slow_feat = torch.randn(2, 768)
    au_intensities = torch.rand(2, 16, 28)

    with torch.no_grad():
        image = pipeline(fast_feat, slow_feat, au_intensities)

    print(f"Fast input: {fast_feat.shape}")
    print(f"Slow input: {slow_feat.shape}")
    print(f"Output: {image.shape}")
    print(f"Range: [{image.min():.3f}, {image.max():.3f}]")

    print("BiomimeticGenerationPipeline passed!")


def test_factory():
    """Test factory function"""
    print("\n" + "=" * 50)
    print(" Test: create_biomimetic_generator")
    print("=" * 50)

    generator = create_biomimetic_generator()
    print(f"Created: {type(generator).__name__}")

    # Test with dual-pathway features
    fast_feat = torch.randn(1, 512)
    slow_feat = torch.randn(1, 768)

    with torch.no_grad():
        image = generator(fast_feat, slow_feat)

    print(f"Output: {image.shape}")

    print("Factory test passed!")


def test_config():
    """Test configuration loading"""
    print("\n" + "=" * 50)
    print(" Test: Config Loading")
    print("=" * 50)

    print(f"AU_DECODER_CONFIG num_aus: {AU_DECODER_CONFIG.get('num_aus')}")
    print(f"DATA_CONFIG: T={DATA_CONFIG.get('T')}, H={DATA_CONFIG.get('H')}")
    print(f"VISUAL_PERCEPTION_CONFIG: {list(VISUAL_PERCEPTION_CONFIG.keys())[:5]}...")

    print("Config test passed!")


if __name__ == '__main__':
    print("=" * 60)
    print(" Biomimetic Image Generator Test")
    print("=" * 60)

    test_config()
    test_au_to_illumination()
    test_illuminative_renderer()
    test_light_shadow_generator()
    test_biomimetic_image_generator()
    test_full_pipeline()
    test_factory()

    print("\n" + "=" * 60)
    print(" All Tests Passed!")
    print("=" * 60)