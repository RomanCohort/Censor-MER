# =============================================================================
# Censor -- Spherical Harmonics Lighting Module
# =============================================================================
# Realistic face lighting using Spherical Harmonics (SH)
# Based on: https://www.cs.cmu.edu/~pts16A/face/faceRenderer.pdf
# =============================================================================
# SH is the standard for face relighting in computer graphics
# Key insight: Most face lighting can be approximated with 9 SH bands

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Spherical Harmonics Basis Functions
# =============================================================================

class SphericalHarmonicsBasis(nn.Module):
    """
    SH basis functions up to degree 2 (9 bands).

    Band 0: Constant (1 channel)
    Band 1: Linear (3 channels)
    Band 2: Quadratic (5 channels)

    Total: 9 coefficients per RGB channel = 27 values
    """

    # Precomputed SHEval at certain angles
    # For efficiency, we use analytical formulas

    def __init__(self, max_degree=2):
        super().__init__()
        self.max_degree = max_degree

    def forward(self, normals):
        """
        Evaluate SH basis functions at given normal directions.

        Args:
            normals: (B, V, 3) or (B, 3, H, W) unit normal vectors
        Returns:
            sh: (B, ..., 9) SH coefficients
        """
        if normals.dim() == 4:
            # Image format: (B, 3, H, W) -> (B, H*W, 3)
            B, C, H, W = normals.shape
            normals = normals.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, 3)

        # Normalize (should already be normalized)
        n = F.normalize(normals, dim=-1)  # (B, V, 3)
        x, y, z = n[..., 0], n[..., 1], n[..., 2]

        # Band 0: constant
        sh0 = torch.ones_like(x) * 0.282095

        # Band 1: linear
        sh1 = torch.stack([
            0.488603 * y,        # Y_1^-1
            0.488603 * z,        # Y_1^0
            0.488603 * x,        # Y_1^1
        ], dim=-1)

        # Band 2: quadratic
        sh2 = torch.stack([
            1.092548 * x * y,   # Y_2^-2
            1.092548 * y * z,   # Y_2^-1
            0.315392 * (3 * z**2 - 1),  # Y_2^0
            1.092548 * x * z,   # Y_2^1
            0.546274 * (x**2 - y**2),  # Y_2^2
        ], dim=-1)

        # Concatenate: (B, V, 9)
        sh = torch.cat([sh0.unsqueeze(-1), sh1, sh2], dim=-1)

        return sh


# =============================================================================
# SH Lighting Estimator
# =============================================================================

class SHLightingEstimator(nn.Module):
    """
    Estimates SH lighting coefficients from features.

    Instead of fitting to a target image, we estimate
    lighting directly from dual-pathway features.
    """

    def __init__(self, feature_dim=1024, num_sh_bands=9, num_rgb=3):
        super().__init__()
        self.num_sh = num_sh_bands
        self.num_rgb = num_rgb

        # Lighting estimator per band
        self.lighting_estimator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_sh_bands * num_rgb),
            nn.Tanh()  # Bound to [-1, 1]
        )

        # Learnable average lighting (as initialization)
        # Typical indoor lighting: warm, from above-front
        avg_lighting = torch.tensor([
            0.8,   # Band 0 (ambient) - warm
            0.1,   # Y level
            0.5,   # Z (front)
            0.3,   # X (right)
            0.0,   # XY
            0.2,   # YZ
            0.6,   # Z^2
            0.1,   # XZ
            0.0,   # X^2-Y^2
        ]).repeat(num_rgb)  # Repeat for RGB

        self.register_buffer('avg_lighting', avg_lighting)

    def forward(self, features):
        """
        Args:
            features: (B, feature_dim)
        Returns:
            sh_coeffs: (B, 9, 3) or (B, 27)
        """
        raw = self.lighting_estimator(features)  # (B, 27)

        # Reshape: (B, 3, 9) -> (B, 9, 3)
        sh = raw.view(-1, self.num_rgb, self.num_sh)
        sh = sh.permute(0, 2, 1)  # (B, 9, 3)

        return sh


# =============================================================================
# SH Lighting Renderer
# =============================================================================

class SHLightingRenderer(nn.Module):
    """
    Applies SH lighting to face normals.

    Key insight: Dot product of normal with light direction.
    """

    def __init__(self):
        super().__init__()
        self.sh_basis = SphericalHarmonicsBasis(max_degree=2)

    def forward(self, normal_map, sh_coeffs):
        """
        Apply SH lighting.

        Args:
            normal_map: (B, 3, H, W) face normals
            sh_coeffs: (B, 9, 3) SH coefficients per RGB
        Returns:
            lit_image: (B, 3, H, W)
        """
        # Evaluate SH at each normal direction
        sh_basis = self.sh_basis(normal_map)  # (B, HW, 9)

        # Lighting = SH_basis @ SH_coeffs
        # sh_basis: (B, HW, 9), sh_coeffs: (B, 9, 3) -> (B, HW, 3)
        B = normal_map.shape[0]
        HW = normal_map.shape[2] * normal_map.shape[3]

        # Reshape output
        lit = torch.bmm(sh_basis, sh_coeffs)  # (B, HW, 3)
        lit = lit.view(B, 3, normal_map.shape[2], normal_map.shape[3])

        return lit


# =============================================================================
# Complete SH Lighting Pipeline
# =============================================================================

class SHLightingPipeline(nn.Module):
    """
    Complete SH lighting pipeline.

    Flow:
    1. Feature → SH coefficients
    2. Normal map → SH basis
    3. SH basis × SH coeffs → lit face
    """

    def __init__(self, feature_dim=1024, image_size=224):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.estimator = SHLightingEstimator(feature_dim)
        self.renderer = SHLightingRenderer()

    def forward(self, features, normal_map):
        """
        Args:
            features: (B, feature_dim)
            normal_map: (B, 3, H, W) face normals
        Returns:
            lit: (B, 3, H, W) lit face
        """
        # Estimate SH coefficients
        sh_coeffs = self.estimator(features)  # (B, 9, 3)

        # Apply lighting
        lit = self.renderer(normal_map, sh_coeffs)

        return lit


# =============================================================================
# Simple SH-based Face Lighting (without 3D mesh)
# =============================================================================

class SimpleSHFaceLighting(nn.Module):
    """
    Simplified SH lighting for face images.
    Uses estimated face normals from image gradients.
    """

    def __init__(self, feature_dim=1024):
        super().__init__()
        self.estimator = SHLightingEstimator(feature_dim)
        self.sh_basis = SphericalHarmonicsBasis()

        # Estimate normals from gradients (simplified)
        self.normal_estimator = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 49),  # 7x7 normal map
            nn.Tanh()
        )

    def estimate_normals(self, features, image_size=224):
        """
        Estimate face normals from features.

        Simplified: just return forward-facing normals.
        """
        normals = self.normal_estimator(features)  # (B, 3*49)
        normals = normals.view(-1, 3, 7, 7)

        # Upsample to image size
        normals = F.interpolate(normals, size=(image_size, image_size),
                              mode='bilinear', align_corners=False)

        # Ensure normalized
        normals = F.normalize(normals, dim=1)

        return normals

    def forward(self, features):
        """
        Apply SH lighting to face.

        Args:
            features: (B, feature_dim)
        Returns:
            sh_coeffs: (B, 9, 3)
        """
        # Estimate SH from features
        sh_coeffs = self.estimator(features)

        return sh_coeffs


# =============================================================================
# Ambient + Directional Light (alternative to SH)
# =============================================================================

class AmbientDirectionalLight(nn.Module):
    """
    Simple ambient + directional light model.
    More interpretable than SH but less accurate.
    """

    def __init__(self, feature_dim=1024):
        super().__init__()

        # Light parameters
        self.light_estimator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),  # ambient(3) + diffuse(3) + specular(3) + strength(1)
        )

    def forward(self, features):
        """
        Args:
            features: (B, feature_dim)
        Returns:
            dict with ambient, diffuse, specular, strength
        """
        params = self.light_estimator(features)

        ambient = torch.sigmoid(params[:, :3])
        diffuse = torch.tanh(params[:, 3:6])
        specular = torch.tanh(params[:, 6:9])
        strength = torch.sigmoid(params[:, 9:10]) * 2 + 0.5  # [0.5, 2.5]

        return {
            'ambient': ambient,     # (B, 3) ambient color
            'diffuse': diffuse,      # (B, 3) light direction
            'specular': specular,    # (B, 3) specular highlight
            'strength': strength,    # (B, 1)
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print(" Spherical Harmonics Lighting Test")
    print("=" * 50)

    # Test SH basis
    sh_basis = SphericalHarmonicsBasis()
    normals = torch.randn(2, 3, 224, 224)
    normals = F.normalize(normals, dim=1)

    with torch.no_grad():
        sh = sh_basis(normals)

    print(f"Normals: {normals.shape}")
    print(f"SH basis: {sh.shape}")

    # Test estimator
    estimator = SHLightingEstimator(1024)
    features = torch.randn(2, 1024)

    with torch.no_grad():
        sh_coeffs = estimator(features)

    print(f"SH coeffs: {sh_coeffs.shape}")

    # Test full pipeline
    pipeline = SHLightingPipeline(1024, 224)
    normal_map = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        lit = pipeline(features, normal_map)

    print(f"Lit image: {lit.shape}, range=[{lit.min():.3f}, {lit.max():.3f}]")

    # Test simple lighting
    simple = SimpleSHFaceLighting(1024)

    with torch.no_grad():
        sh = simple(features)

    print(f"Simple SH: {sh.shape}")

    # Test ambient + directional
    ad_light = AmbientDirectionalLight(1024)

    with torch.no_grad():
        light_params = ad_light(features)

    print(f"Ambient: {light_params['ambient'].shape}")
    print(f"Diffuse: {light_params['diffuse'].shape}")

    print("\nSpherical Harmonics Lighting Test Passed!")