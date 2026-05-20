# =============================================================================
# Censor -- AU Attention Map Utilities
# =============================================================================
# Standalone AU (Action Unit) spatial attention maps for focusing on
# facial regions important for micro-expression detection.
#
# Based on FACS (Facial Action Coding System):
#   - Brows: AU1 (Inner Brow Raiser), AU2 (Outer Brow Raiser), AU4 (Brow Lowerer)
#   - Eyes: AU5 (Upper Lid Raiser), AU6 (Cheek Raiser), AU7 (Lid Tightener)
#   - Nose: AU9 (Nose Wrinkler)
#   - Mouth: AU10,12,14,15,17,20,23-28
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class AULandmarkAttention(nn.Module):
    """
    AU-based landmark attention map generator.

    Creates spatial attention maps emphasizing facial regions where
    Action Units typically occur.

    Usage:
        au_attn = AULandmarkAttention(size=224)
        attention_map = au_attn()  # (1, 1, H, W)
        # Apply as: features = features * attention_map
    """

    def __init__(self, size=224, device='cpu'):
        super().__init__()
        self.size = size

        # Pre-computed AU landmarks (normalized coordinates)
        # Derived from average facial landmark positions
        self.register_buffer('_au_map', self._create_au_landmark_map(size))

    def _create_au_landmark_map(self, size):
        """Create AU landmark attention map."""
        # Coordinate grids
        y = torch.arange(size, dtype=torch.float32)
        x = torch.arange(size, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        # Normalize to [-1, 1]
        xx_norm = (xx / size * 2 - 1)
        yy_norm = (yy / size * 2 - 1)

        # AU region centers and widths (approximate)
        # Format: (center_y, center_x, sigma_y, sigma_x, weight)
        au_regions = [
            # Brows (top)
            (-0.7, -0.3, 0.08, 0.25, 1.0),   # AU1,2,4
            (-0.7,  0.3, 0.08, 0.25, 1.0),   # AU1,2,4 (right)
            # Eyes (upper-mid)
            (-0.3, -0.25, 0.1, 0.15, 1.2),   # AU5,6,7 (left)
            (-0.3,  0.25, 0.1, 0.15, 1.2),   # AU5,6,7 (right)
            # Nose (center)
            (0.0, 0.0, 0.08, 0.06, 0.8),    # AU9
            # Mouth (lower)
            (0.35, -0.15, 0.12, 0.1, 1.0),   # AU10,12,14,15 (left)
            (0.35,  0.15, 0.12, 0.1, 1.0),   # AU10,12,14,15 (right)
            (0.45, 0.0, 0.08, 0.12, 0.9),    # AU17,20,23-25 (center)
        ]

        # Combine all AU regions
        attention = torch.zeros_like(xx_norm)
        for cy, cx, sy, sx, w in au_regions:
            gaussian = torch.exp(
                -((yy_norm - cy)**2 / (2 * sy**2) + (xx_norm - cx)**2 / (2 * sx**2))
            )
            attention = attention + w * gaussian

        # Normalize to [0, 1]
        attention = attention / (attention.max() + 1e-8)

        return attention.view(1, 1, size, size)

    def forward(self, size=None):
        """
        Get AU attention map.

        Args:
            size (int, optional): Output size. If None, use init size.
        Returns:
            attention (torch.Tensor): (1, 1, H, W)
        """
        if size is not None and size != self.size:
            # Resize
            attention = F.interpolate(
                self._au_map,
                size=(size, size),
                mode='bilinear',
                align_corners=False
            )
        else:
            attention = self._au_map
        return attention


class AUMaskedAttention(nn.Module):
    """
    AU attention with masking capability.

    Applies AU attention and optionally masks non-AU regions.
    """

    def __init__(self, size=224, mask_threshold=0.1):
        super().__init__()
        self.size = size
        self.mask_threshold = mask_threshold
        self.au_attention = AULandmarkAttention(size)

    def forward(self, x, apply_mask=False):
        """
        Apply AU attention to features.

        Args:
            x (torch.Tensor): Features, (B, C, H, W) or (B, C, T, H, W)
            apply_mask (bool): If True, mask out low-attention regions
        Returns:
            attended (torch.Tensor): Same shape as input
            attention (torch.Tensor): Attention map, resized to match input
        """
        # Get AU attention map resized to input features
        if x.dim() == 5:
            # Video: (B, C, T, H, W)
            _, _, _, H, W = x.shape
            attn = F.interpolate(
                self.au_attention(),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            attn = attn.unsqueeze(2)  # (1, 1, 1, H, W)
        else:
            # Image/feature: (B, C, H, W)
            _, _, H, W = x.shape
            attn = F.interpolate(
                self.au_attention(),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        attended = x * attn

        if apply_mask:
            mask = (attn > self.mask_threshold).float()
            attended = attended * mask

        return attended, attn


def create_au_attention_map(size=224):
    """
    Standalone function to create AU attention map.

    Args:
        size (int): Output resolution
    Returns:
        attention (torch.Tensor): (1, 1, H, W)
    """
    au = AULandmarkAttention(size)
    return au()


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print('=== Testing AU Attention ===')

    # Test standalone function
    attn = create_au_attention_map(224)
    print(f'Attention map: {attn.shape}')
    print(f'Max: {attn.max():.4f}, Min: {attn.min():.4f}')

    # Test as module
    au_module = AULandmarkAttention(size=224)
    attn2 = au_module()
    print(f'\nModule output: {attn2.shape}')

    # Test with features
    masker = AUMaskedAttention(size=224)
    features = torch.randn(2, 512, 14, 14)
    masked, attn = masker(features)
    print(f'\nInput: {features.shape}')
    print(f'Masked: {masked.shape}')
    print(f'Attention: {attn.shape}')

    print('\n=== AU Attention Ready ===')