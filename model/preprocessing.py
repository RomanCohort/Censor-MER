# =============================================================================
# Censor -- Biomimetic Preprocessing Module (Enhanced)
# =============================================================================
# Simulates biological preprocessing:
#   1. SaliencyDetector: Foveal sampling via Gaussian pyramid + spatial prior
#   2. rPPGExtractor: Remote photoplethysmography blood-flow signal extraction
#   3. TVL1OpticalFlow: Real TV-L1 optical flow via OpenCV DualTVL1
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.defaults import PREPROCESS_CONFIG


# =============================================================================
# SaliencyDetector -- Fovea-Inspired Spatial Saliency
# =============================================================================
# Biological motivation: Human fovea has highest cone density (~1-2° visual angle),
# peripheral vision has lower resolution. The fovea directs attention to facial
# action units (AU) regions (eyes, mouth, nose).
#
# Mathematical formulation:
#   S(x,y) = sum_{l=0}^{L-1} w_l * G_sigma(x,y) * I_l(x,y)
# where I_l is the l-th Gaussian pyramid level,
# G_sigma is a center-biased spatial prior (Gaussian),
# w_l = 1/2^l are level weights (higher weight for finer resolution).
# =============================================================================

class SaliencyDetector(nn.Module):
    """
    Fovea-inspired saliency detector using Gaussian pyramid and spatial prior.

    The detector simulates the human retinal fovea by building a multi-scale
    Gaussian pyramid and combining it with a center-biased spatial prior map.
    This suppresses background noise and highlights facial regions of interest.

    Architecture:
        Input (B, C, T, H, W)
          -> Gaussian Pyramid (4 levels)
          -> Center-biased Spatial Prior (centered on face region)
          -> Weighted fusion -> Saliency Map (B, 1, T, H, W)
    """

    def __init__(self, pyramid_levels=None, gaussian_sigma=None, center_bias_strength=None):
        super().__init__()
        self.pyramid_levels = pyramid_levels or PREPROCESS_CONFIG['pyramid_levels']
        self.gaussian_sigma = gaussian_sigma or PREPROCESS_CONFIG['gaussian_sigma']
        self.center_bias = center_bias_strength or PREPROCESS_CONFIG['center_bias_strength']

        # Build Gaussian blur kernel for pyramid construction
        # Kernel size = 2*ceil(3*sigma) + 1
        kernel_size = int(2 * np.ceil(3 * self.gaussian_sigma) + 1)
        kernel = self._gaussian_kernel(kernel_size, self.gaussian_sigma)  # (K, K)
        self.register_buffer('gaussian_kernel', kernel)

        # Learnable weights for multi-level fusion
        self.fusion_weights = nn.Parameter(torch.ones(self.pyramid_levels) / self.pyramid_levels)

    @staticmethod
    def _gaussian_kernel(size, sigma):
        """Generate a 2D Gaussian kernel."""
        ax = torch.arange(size) - size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def _gaussian_blur_2d(self, x):
        """
        Apply 2D Gaussian blur to spatial dims of a 5D tensor (per-channel, same kernel).

        Args:
            x (torch.Tensor): (B, C, T, H, W)
        Returns:
            blurred (torch.Tensor): (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        # Merge batch+temporal dims for conv2d: (B*T, C, H, W)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        # 2D Gaussian kernel: (C, C, K, K) shared across spatial dims
        # Replicate kernel for each input channel
        kernel = self.gaussian_kernel[None, None, :, :]  # (1, 1, K, K)
        kernel_C = kernel.expand(C, -1, -1, -1)  # (C, 1, K, K)
        padding = kernel.shape[-1] // 2
        # Depthwise conv: each channel gets its own Gaussian blur
        blurred = F.conv2d(x_2d, kernel_C, padding=padding, groups=C)
        # Restore shape: (B*T, C, H, W) -> (B, T, C, H, W) -> (B, C, T, H, W)
        _, _, H_out, W_out = blurred.shape
        blurred = blurred.view(B, T, C, H_out, W_out).permute(0, 2, 1, 3, 4)
        return blurred

    def _build_gaussian_pyramid(self, x):
        """
        Build Gaussian pyramid levels.
        Input:  (B, C, T, H, W)
        Output: list of (B, C, T, H_l, W_l) where level l has resolution H/2^l, W/2^l
        """
        pyramid = []
        current = x
        for level in range(self.pyramid_levels):
            if level > 0:
                # Apply Gaussian blur then downsample
                current = self._gaussian_blur_2d(current)
                # Downsample by factor 2 (spatial only)
                current = F.avg_pool3d(current, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            pyramid.append(current)
        return pyramid

    def _create_spatial_prior(self, h, w, device, dtype):
        """
        Create center-biased 2D Gaussian spatial prior.
        Mathematical: P(x,y) = exp(-((x-mx)^2/(2*sx^2) + (y-my)^2/(2*sy^2)))
        where mx=W/2, my=H/2, sx=W/6, sy=H/6 (face-center bias).
        """
        x = torch.arange(w, device=device, dtype=dtype)
        y = torch.arange(h, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        mx, my = w / 2, h / 2
        sx, sy = w / 6, h / 6  # face-center bias

        prior = torch.exp(
            -((xx - mx)**2 / (2 * sx**2)) - ((yy - my)**2 / (2 * sy**2))
        )
        # Normalize to [0, 1]
        prior = (prior - prior.min()) / (prior.max() - prior.min() + 1e-8)
        prior = prior * self.center_bias
        return prior.view(1, 1, 1, h, w)  # (1, 1, 1, H, W)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw RGB video, shape (B, C, T, H, W)
        Returns:
            saliency_map (torch.Tensor): Spatial prior map, shape (B, 1, T, H, W)
        """
        print(f"[SaliencyDetector] Input: {x.shape}")
        B, C, T, H, W = x.shape

        # Build Gaussian pyramid
        pyramid = self._build_gaussian_pyramid(x)

        # Compute fusion weights (softmax over levels)
        fusion_w = F.softmax(self.fusion_weights, dim=0)

        # Reconstruct saliency map by upsampling and fusing pyramid levels
        saliency = torch.zeros_like(x[:, :1])  # (B, 1, T, H, W)

        for level, level_feat in enumerate(pyramid):
            # Upsample to original resolution
            _, _, _, H_l, W_l = level_feat.shape
            upsampled = F.interpolate(
                level_feat.mean(dim=1, keepdim=True),  # (B, 1, T, H_l, W_l)
                size=(T, H, W),
                mode='trilinear',
                align_corners=False
            )
            saliency = saliency + fusion_w[level] * upsampled

        # Apply spatial prior (center-biased Gaussian)
        spatial_prior = self._create_spatial_prior(H, W, x.device, x.dtype)
        saliency = saliency * spatial_prior

        # Normalize per-frame (avoid in-place ops for gradient tracking)
        saliency_normalized = torch.zeros_like(saliency)
        for b in range(B):
            for t in range(T):
                frame = saliency[b, 0, t]
                saliency_normalized[b, 0, t] = (frame - frame.mean()) / (frame.std() + 1e-8)
        saliency = saliency_normalized

        print(f"[SaliencyDetector] Output: {saliency.shape}")
        return saliency


# =============================================================================
# rPPGExtractor -- Remote Photoplethysmography Signal
# =============================================================================
# Biological motivation: Blood oxygen saturation changes cause subtle skin color
# fluctuations driven by the cardiac cycle (~0.5-4.0 Hz). This simulates reading
# "blushing" (anger/happiness) or "pallor" (fear/sadness) from facial color.
#
# Mathematical formulation:
#   rPPG(t) = sum_{c in {R,G,B}} alpha_c * I_c(t)
#   rPPG_filtered(t) = sum_{tau=-K}^{K} h(tau) * rPPG(t-tau)
# where alpha_c are learned chrominance projection weights,
# h is a learned FIR temporal bandpass filter (0.5-4.0 Hz cardiac range).
# =============================================================================

class rPPGExtractor(nn.Module):
    """
    Remote photoplethysmography (rPPG) blood-flow signal extractor.

    Extracts subtle blood oxygen saturation changes from facial skin color
    fluctuations using chrominance decomposition and temporal bandpass filtering.

    Architecture:
        Input (B, C, T, H, W)
          -> Spatial averaging over H,W -> (B, 3, T) time series
          -> Temporal bandpass Conv1d -> (B, 3, T)
          -> Chrominance projection -> (B, 3, T)
          -> Spatial broadcast -> (B, 3, T, H, W)
    """

    def __init__(self, window_size=None, bandpass_low=None, bandpass_high=None):
        super().__init__()
        self.window_size = window_size or PREPROCESS_CONFIG['rppg_window_size']
        self.bandpass_low = bandpass_low or PREPROCESS_CONFIG['rppg_bandpass_low']
        self.bandpass_high = bandpass_high or PREPROCESS_CONFIG['rppg_bandpass_high']

        # Temporal bandpass filter: 1D conv that acts as FIR filter on time dimension
        # This learns the bandpass characteristics (0.5-4.0 Hz cardiac range)
        kernel_size = self.window_size * 2 + 1
        self.temporal_conv = nn.Conv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
            groups=3  # Depthwise: each RGB channel processed independently
        )

        # Chrominance projection: maps filtered RGB signals to blood-flow signal
        self.projection = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.temporal_conv.weight)
        nn.init.xavier_uniform_(self.projection[0].weight)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw RGB video, shape (B, 3, T, H, W)
        Returns:
            rppg_heatmap (torch.Tensor): Blood-flow signal heatmap, shape (B, 3, T, H, W)
        """
        print(f"[rPPGExtractor] Input: {x.shape}")
        B, C, T, H, W = x.shape

        # Step 1: Spatial averaging to get per-frame time series
        # Aggregate over spatial dimensions to get color changes over time
        mean_pool = x.mean(dim=[-1, -2])  # (B, 3, T)

        # Step 2: Temporal bandpass filtering (learned FIR)
        # This extracts the cardiac frequency range (0.5-4.0 Hz)
        filtered = self.temporal_conv(mean_pool)  # (B, 3, T)

        # Step 3: Chrominance projection
        # Project from RGB space to blood-flow signal basis
        proj = self.projection(filtered.transpose(1, 2))  # (B, T, 3)
        rppg_signal = proj.transpose(1, 2)  # (B, 3, T)

        # Step 4: Spatial broadcast
        # Expand (B, 3, T) -> (B, 3, T, H, W) by broadcasting across spatial dims.
        # This simulates skin-color fluctuation visible across the entire face.
        rppg_heatmap = rppg_signal[:, :, :, None, None]  # (B, 3, T, 1, 1)
        rppg_heatmap = rppg_heatmap.expand(-1, -1, -1, H, W)  # (B, 3, T, H, W)

        print(f"[rPPGExtractor] Output: {rppg_heatmap.shape}")
        return rppg_heatmap


# =============================================================================
# TVL1OpticalFlow -- Real TV-L1 via OpenCV DualTVL1
# =============================================================================
# Mathematical formulation (TV-L1):
#   min_u  integral( |grad u| + lambda * |rho(u)| ) dx
# where u = (u_x, u_y) is the flow field,
# rho(u) = I_1(x+u) - I_0(x) is the brightness constancy error.
# Solved via primal-dual algorithm with duality-based TV denoising.
#
# This implementation uses OpenCV's DualTVL1 algorithm, which is one of the
# most accurate classical optical flow methods.
# =============================================================================

class TVL1OpticalFlow(nn.Module):
    """
    Real TV-L1 optical flow computation via OpenCV DualTVL1.

    The TV-L1 (Total Variation L1) optical flow algorithm is a variational method
    that minimizes the L1 data term (robust to outliers) with total variation
    regularization. It provides accurate flow estimation for facial micro-expressions.

    Args:
        frame0 (torch.Tensor): Previous frame, shape (B, C, H, W)
        frame1 (torch.Tensor): Current frame, shape (B, C, H, W)
    Returns:
        flow (torch.Tensor): Optical flow, shape (B, 2, H, W)
            - channel 0: x-displacement (horizontal)
            - channel 1: y-displacement (vertical)
    """

    def __init__(self, tau=None, lmbda=None, theta=None):
        super().__init__()
        self.tau = tau or PREPROCESS_CONFIG['tvl1_tau']
        self.lmbda = lmbda or PREPROCESS_CONFIG['tvl1_lambda']
        self.theta = theta or PREPROCESS_CONFIG['tvl1_theta']

        # Initialize OpenCV DualTVL1 solver
        self._init_opencv_flow()

        print(f"[TVL1OpticalFlow] Initialized with OpenCV DualTVL1")
        print(f"  tau={self.tau}, lambda={self.lmbda}, theta={self.theta}")

    def _init_opencv_flow(self):
        """Initialize OpenCV DualTVL1 optical flow solver."""
        try:
            from cv2.optflow import createOptFlow_DualTVL1
            self.dtvl = createOptFlow_DualTVL1()
            self.dtvl.setTau(self.tau)
            self.dtvl.setLambda(self.lmbda)
            self.dtvl.setTheta(self.theta)
            # Number of warping iterations (default 5)
            self.dtvl.setWarpingsNumber(5)
            # Epsilon for convergence (default 0.01)
            self.dtvl.setEpsilon(0.01)
            self._opencv_available = True
        except ImportError:
            print(f"[TVL1OpticalFlow] WARNING: OpenCV DualTVL1 not available.")
            print(f"[TVL1OpticalFlow] Falling back to gradient-based placeholder.")
            self._opencv_available = False

            # Fallback: Sobel kernels for gradient computation
            sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32)
            self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
            self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _compute_opencv_flow(self, frame0, frame1):
        """
        Compute optical flow using OpenCV DualTVL1.

        Args:
            frame0 (torch.Tensor): Previous frame, shape (B, C, H, W), RGB float [0,1]
            frame1 (torch.Tensor): Current frame, shape (B, C, H, W), RGB float [0,1]
        Returns:
            flow (torch.Tensor): Flow field, shape (B, 2, H, W)
        """
        B = frame0.shape[0]
        flows = []

        for b in range(B):
            # Convert to numpy: (H, W, C) -> RGB
            img0 = frame0[b].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
            img1 = frame1[b].permute(1, 2, 0).detach().cpu().numpy()

            # Convert to grayscale (TV-L1 works on grayscale)
            gray0 = np.mean(img0, axis=2)  # (H, W)
            gray1 = np.mean(img1, axis=2)

            # Compute flow
            flow_np = self.dtvl.calc(gray0.astype(np.float32), gray1.astype(np.float32), None)
            # flow_np shape: (H, W, 2) where channels are (x, y) displacement

            # Convert back to tensor
            flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1)  # (2, H, W)
            flows.append(flow_tensor)

        return torch.stack(flows, dim=0).to(frame0.device)  # (B, 2, H, W)

    def _sobel_gradient(self, x):
        """Compute spatial gradients using Sobel operator (fallback)."""
        B, C, H, W = x.shape
        dx = F.conv2d(
            x.view(B * C, 1, H, W),
            self.sobel_x,
            padding=1, groups=1
        ).view(B, C, H, W).mean(dim=1)  # (B, H, W)
        dy = F.conv2d(
            x.view(B * C, 1, H, W),
            self.sobel_y,
            padding=1, groups=1
        ).view(B, C, H, W).mean(dim=1)  # (B, H, W)
        return dx, dy

    def forward(self, frame0, frame1):
        """
        Compute optical flow between two consecutive frames.

        Args:
            frame0 (torch.Tensor): Previous frame, shape (B, C, H, W)
            frame1 (torch.Tensor): Current frame, shape (B, C, H, W)
        Returns:
            flow (torch.Tensor): Estimated flow field, shape (B, 2, H, W)
                - channel 0: x-displacement (horizontal)
                - channel 1: y-displacement (vertical)
        """
        print(f"[TVL1OpticalFlow] Inputs: {frame0.shape}, {frame1.shape}")

        if self._opencv_available:
            flow = self._compute_opencv_flow(frame0, frame1)
            print(f"[TVL1OpticalFlow] Output: {flow.shape} (OpenCV DualTVL1)")
        else:
            # Fallback: gradient-based placeholder
            diff = frame1 - frame0
            dx, dy = self._sobel_gradient(diff)
            flow_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
            flow_x = dx / (flow_mag + 1.0) * self.tau
            flow_y = dy / (flow_mag + 1.0) * self.tau
            flow = torch.stack([flow_x, flow_y], dim=1)
            print(f"[TVL1OpticalFlow] Output: {flow.shape} (gradient-based fallback)")

        return flow


def compute_tvl1_flow_opencv(frame0_np, frame1_np, tau=None, lambda_=None, theta=None):
    """
    Standalone helper function: Real TV-L1 implementation using OpenCV.

    Args:
        frame0_np (np.ndarray): Previous frame in BGR format, shape (H, W, 3)
        frame1_np (np.ndarray): Current frame in BGR format, shape (H, W, 3)
        tau (float): Time step
        lambda_ (float): Smoothness weight
        theta (float): Angular coefficient
    Returns:
        flow (np.ndarray): Optical flow, shape (H, W, 2)
    """
    from cv2.optflow import createOptFlow_DualTVL1
    dtvl = createOptFlow_DualTVL1()
    if tau is not None:
        dtvl.setTau(tau)
    if lambda_ is not None:
        dtvl.setLambda(lambda_)
    if theta is not None:
        dtvl.setTheta(theta)
    flow = dtvl.calc(frame0_np.astype(np.float32), frame1_np.astype(np.float32), None)
    return flow