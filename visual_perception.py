# =============================================================================
# Censor -- Visual Perception Post-Processing Modules
# =============================================================================
# Implements biomimetic visual perception mechanisms for enhanced image realism:
#   1. PupilController: Pupil adjustment based on illumination
#   2. RetinalContrastNorm: Local contrast normalization (retinal adaptation)
#   3. MachBandEnhancer: Edge overshoot for sharper transitions
#   4. CenterSurroundReceptiveField: DoG-based edge detection
# =============================================================================
#
# Biological basis:
#   - Pupil: Controls light intake (2-8mm diameter range)
#   - Retina: Adaptive gain control (Weber-Fechner law)
#   - Mach bands: Overshoot at luminance edges
#   - Ganglion cells: Center-surround antagonism (DoG receptive field)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.defaults import VISUAL_PERCEPTION_CONFIG


class PupilController(nn.Module):
    """
    Biomimetic pupil controller.

    Simulates human pupil adjustment: constricts in bright light,
    dilates in dark conditions to regulate light intake.

    Architecture:
        Input image (B, C, H, W)
          -> Global illumination estimation (mean brightness)
          -> FC(1 -> hidden) -> ReLU -> FC(hidden -> 1) -> Sigmoid
          -> Pupil dilation factor (0~1)
          -> Modulated output
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or VISUAL_PERCEPTION_CONFIG

        hidden_dim = cfg.get('pupil_hidden_dim', 64)

        # Illumination -> dilation mapping
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Base gain and modulation range
        self.base_gain = cfg.get('pupil_base_gain', 0.8)
        self.modulation_range = cfg.get('pupil_modulation_range', 0.4)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image, shape (B, C, H, W)
        Returns:
            torch.Tensor: Pupil-adjusted image
        """
        # Estimate global illumination (mean brightness per image)
        illumination = x.mean(dim=[1, 2, 3], keepdim=True)  # (B, 1, 1, 1)

        # Predict pupil dilation factor
        dilation = self.sigmoid(self.fc2(F.relu(self.fc1(illumination.mean(dim=[1], keepdim=True)))))  # (B, 1, 1, 1)

        # Apply gain: base + dilation * modulation_range
        gain = self.base_gain + dilation * self.modulation_range

        return x * gain


class RetinalContrastNorm(nn.Module):
    """
    Biomimetic retinal contrast normalization.

    Simulates retinal adaptive gain control: enhances contrast in dark regions,
    compresses contrast in bright regions (Weber-Fechner law approximation).

    Uses local mean and standard deviation for normalization.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or VISUAL_PERCEPTION_CONFIG

        kernel_size = cfg.get('retinal_kernel', 9)
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.padding = self.kernel_size // 2
        self.epsilon = 1e-4
        self.alpha = cfg.get('retinal_alpha', 0.5)  # Normalization strength
        self.beta = cfg.get('retinal_beta', 0.0)       # Bias term

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image, shape (B, C, H, W)
        Returns:
            torch.Tensor: Contrast-normalized image
        """
        # Compute local mean
        mean = F.avg_pool2d(x, self.kernel_size, 1, self.padding)

        # Compute local variance = E[X²] - E[X]²
        sqr_mean = F.avg_pool2d(x ** 2, self.kernel_size, 1, self.padding)
        var = (sqr_mean - mean ** 2).clamp(min=self.epsilon)
        std = var.sqrt()

        # Normalized output: (x - mean) / std * alpha + beta
        normalized = self.alpha * (x - mean) / (std + self.epsilon) + self.beta

        return normalized


class MachBandEnhancer(nn.Module):
    """
    Biomimetic Mach band enhancer.

    Simulates Mach band effect: overestimation of brightness on the
    side of edges closer to light, underestimation on the darker side.
    This creates sharper perceived edges.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or VISUAL_PERCEPTION_CONFIG

        self.edge_strength = cfg.get('mach_band_strength', 0.3)
        self.sigma = cfg.get('mach_band_sigma', 2.0)

        # Store number of channels for kernel creation
        self.registered_channels = None

    def _create_kernels(self, channels):
        """Create derivative kernels for given number of channels"""
        dx_kernel = torch.tensor([[[[-1, 1]]]], dtype=torch.float32).expand(channels, 1, 1, 2)
        dy_kernel = torch.tensor([[[[-1], [1]]]], dtype=torch.float32).expand(channels, 1, 2, 1)

        self.register_buffer('_dx_kernel', dx_kernel)
        self.register_buffer('_dy_kernel', dy_kernel)
        self.registered_channels = channels

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image, shape (B, C, H, W)
        Returns:
            torch.Tensor: Edge-enhanced image
        """
        # Create kernels if needed (first call)
        if self.registered_channels != x.size(1):
            self._create_kernels(x.size(1))

        # Compute first-order derivatives with depthwise convolution (groups = channels)
        dx = F.conv2d(x, self._dx_kernel, padding=(0, 1), groups=x.size(1))  # Output: B, C, H, W+1
        dy = F.conv2d(x, self._dy_kernel, padding=(1, 0), groups=x.size(1))  # Output: B+1, C, H, W

        # Trim to match input size
        dx = dx[:, :, :, :x.size(3)]
        dy = dy[:, :, :x.size(2), :]

        # Mach band overshoot: sign(dx) * |dx| creates overshoot/undershoot
        mach_effect = self.edge_strength * (dx.sign() * dx.abs() + dy.sign() * dy.abs())

        # Apply with smooth falloff near edges (simplified: just add directly)
        enhanced = x + mach_effect

        return enhanced.clamp(0, 1) if x.max() <= 1 else enhanced


class CenterSurroundReceptiveField(nn.Module):
    """
    Biomimetic center-surround receptive field.

    Simulates retinal ganglion cells with DoG (Difference of Gaussian) receptive fields:
    - ON-center: excitatory center, inhibitory surround
    - OFF-center: inhibitory center, excitatory surround

    Used for edge detection and contrast computation.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or VISUAL_PERCEPTION_CONFIG

        self.center_sigma = cfg.get('center_sigma', 1.5)
        self.surround_sigma = cfg.get('surround_sigma', 3.0)

        # Store for initialization later
        self.registered_channels = None

        # Initialize kernels with default (will be recreated on first forward)
        # Create dummy kernels for registration
        dummy_x = self._dog_kernel(self.center_sigma, self.surround_sigma, axis=0)
        self.register_buffer('_dog_x', dummy_x)
        self.register_buffer('_dog_y', self._dog_kernel(self.center_sigma, self.surround_sigma, axis=1))

    def _create_kernels(self, channels):
        """Create DoG kernels for given number of channels"""
        dog_x = self._dog_kernel(self.center_sigma, self.surround_sigma, axis=0)
        dog_y = self._dog_kernel(self.center_sigma, self.surround_sigma, axis=1)

        # Expand to match input channels
        dog_x = dog_x.expand(channels, 1, -1, 1)
        dog_y = dog_y.expand(channels, 1, 1, -1)

        # Remove old buffers if exist
        if hasattr(self, '_dog_x'):
            del self._dog_x
        if hasattr(self, '_dog_y'):
            del self._dog_y

        self.register_buffer('_dog_x', dog_x)
        self.register_buffer('_dog_y', dog_y)
        self.registered_channels = channels

    def _gaussian_2d(self, sigma, size=7):
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size) - size // 2
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def _dog_kernel(self, center_s, surround_s, axis=0):
        """Create 1D DoG kernel (center - surround)"""
        size = max(int(6 * surround_s), 7)
        if size % 2 == 0:
            size += 1

        coords = torch.arange(size) - size // 2

        # Center Gaussian
        center = torch.exp(-coords ** 2 / (2 * center_s ** 2))
        center = center / center.sum()

        # Surround Gaussian
        surround = torch.exp(-coords ** 2 / (2 * surround_s ** 2))
        surround = surround / surround.sum()

        # DoG = center - surround
        dog = center - surround

        if axis == 0:
            return dog.view(1, 1, -1, 1)
        else:
            return dog.view(1, 1, 1, -1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image, shape (B, C, H, W)
        Returns:
            torch.Tensor: DoG response (edge/contrast response)
        """
        # Create kernels if needed (first call)
        if self.registered_channels != x.size(1):
            self._create_kernels(x.size(1))

        # Calculate padding for same padding
        pad_x = self._dog_x.shape[2] // 2
        pad_y = self._dog_y.shape[3] // 2

        # Apply separable DoG filter with depthwise convolution
        response = F.conv2d(x, self._dog_x, padding=(pad_x, 0), groups=x.size(1))
        response = F.conv2d(response, self._dog_y, padding=(0, pad_y), groups=x.size(1))

        return response


class VisualPerceptionPostProcess(nn.Module):
    """
    Integrated biomimetic visual perception post-processing pipeline.

    Combines all visual perception modules:
    1. PupilController - illumination adaptation
    2. RetinalContrastNorm - local contrast normalization
    3. MachBandEnhancer - edge sharpening
    4. CenterSurroundReceptiveField - edge detection (residual connection)
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or VISUAL_PERCEPTION_CONFIG

        self.pupil = PupilController(cfg)
        self.retinal = RetinalContrastNorm(cfg)
        self.mach = MachBandEnhancer(cfg)
        self.receptive = CenterSurroundReceptiveField(cfg)

        # Options
        self.apply_retinal = cfg.get('enable_retinal', True)
        self.apply_mach = cfg.get('enable_mach', True)
        self.receptive_weight = cfg.get('receptive_weight', 0.1)

    def forward(self, generated_image, apply_retinal=None, apply_mach=None):
        """
        Args:
            generated_image (torch.Tensor): Generated image from model
            apply_retinal (bool, optional): Override retinal normalization
            apply_mach (bool, optional): Override Mach band enhancement
        Returns:
            torch.Tensor: Post-processed image
        """
        x = generated_image

        # 1. Pupil adaptation
        x = self.pupil(x)

        # 2. Retinal contrast normalization
        if apply_retinal is None:
            apply_retinal = self.apply_retinal
        if apply_retinal:
            x = self.retinal(x)

        # 3. Mach band enhancement
        if apply_mach is None:
            apply_mach = self.apply_mach
        if apply_mach:
            x = self.mach(x)

        # 4. Center-surround edge response (residual)
        edge_response = self.receptive(x)
        x = x + self.receptive_weight * edge_response

        return x


# =============================================================================
# Utility functions
# =============================================================================

def create_visual_perception_module(config=None):
    """Factory function to create the post-process module"""
    return VisualPerceptionPostProcess(config or VISUAL_PERCEPTION_CONFIG)