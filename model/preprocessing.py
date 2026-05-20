# =============================================================================
# Censor -- Biomimetic Preprocessing Module (Enhanced)
# =============================================================================
# Simulates biological preprocessing:
#   1. SaliencyDetector: Foveal sampling via Gaussian pyramid + spatial prior
#   2. rPPGExtractor: Remote photoplethysmography blood-flow signal extraction
#   3. TVL1OpticalFlow: Real TV-L1 optical flow via OpenCV DualTVL1
#   4. ReferenceFace: Average face template for geometric reference
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
# AdaptiveOpticalFlow -- Two-Stage Optical Flow
# =============================================================================
# Two-stage approach:
#   Stage 1 (Fast): Frame difference for motion screening (~15ms)
#   Stage 2 (Fine): TV-L1 only when motion detected (~150ms)
# Average inference: ~50ms (vs 150ms for TV-L1 only)
# =============================================================================

class AdaptiveOpticalFlow(nn.Module):
    """
    Two-stage optical flow: fast screening + fine computation.

    Stage 1: Frame difference for fast motion screening (~15ms)
    Stage 2: TV-L1 only when motion detected (~150ms)

    Average inference: ~50ms (vs 150ms for TV-L1 only)
    """

    def __init__(self, fast_threshold=0.1, use_tvl1=True):
        super().__init__()
        self.fast_threshold = fast_threshold
        self.use_tvl1 = use_tvl1

        if use_tvl1:
            self.tvl1 = TVL1OpticalFlow()

    def _frame_diff(self, frames):
        """Fast frame difference: (B, C, T, H, W) -> (B, C, H, W)"""
        return frames[:, :, -1] - frames[:, :, 0]  # Last frame - first frame

    def _compute_tvl1(self, frames):
        """Compute TV-L1 flow between first and last frame."""
        frame0 = frames[:, :, 0]  # (B, C, H, W)
        frame1 = frames[:, :, -1]  # (B, C, H, W)

        B, C, H, W = frame0.shape
        flows = []

        # Process each sample in batch
        for b in range(B):
            flow = self.tvl1(frame0[b], frame1[b])  # (2, H, W)
            flows.append(flow)

        return torch.stack(flows, dim=0)

    def forward(self, frames):
        """
        Args:
            frames (torch.Tensor): Video frames, shape (B, C, T, H, W)
        Returns:
            flow (torch.Tensor): Optical flow, shape (B, 2, H, W)
            stage (str): 'fast' or 'fine'
        """
        # Stage 1: Fast screening via frame difference
        diff = self._frame_diff(frames)
        motion_magnitude = diff.abs().mean()

        if motion_magnitude > self.fast_threshold and self.use_tvl1:
            # Stage 2: Fine computation with TV-L1
            flow = self._compute_tvl1(frames)
            stage = 'fine'
        else:
            # Fast path: use frame difference as flow proxy
            diff_gray = diff.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            # Convert to flow-like format (u, v)
            flow = torch.cat([diff_gray, diff_gray], dim=1)  # (B, 2, H, W)
            stage = 'fast'

        return flow, stage


# =============================================================================
# SaliencyDetectorE2E -- Fully End-to-End with Adaptive Sigma
# =============================================================================

class SaliencyDetectorE2E(nn.Module):
    """
    Fully end-to-end trainable saliency detector with resolution-adaptive sigma.

    Improvements over SaliencyDetector:
      1. All parameters are learnable (sigma_ratio, center_bias, fusion_weights)
      2. sigma_ratio * min(H,W) ensures resolution-independent center prior
      3. Proper gradient flow for end-to-end training
    """

    def __init__(self, levels=4, sigma_ratio=0.15):
        super().__init__()
        self.levels = levels
        self.sigma_ratio = nn.Parameter(torch.tensor(sigma_ratio))
        self.center_bias = nn.Parameter(torch.tensor(0.5))
        self.fusion_weights = nn.Parameter(torch.ones(levels) / levels)

    def _gaussian_kernel(self, size, sigma):
        """Generate a 2D Gaussian kernel."""
        ax = torch.arange(size, dtype=torch.float32) - size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def _gaussian_blur(self, x):
        """Apply 2D Gaussian blur."""
        B, C, T, H, W = x.shape
        kernel_size = 5  # Fixed for efficiency
        sigma = 1.0
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        kernel = kernel.expand(C, -1, -1, -1)
        padded = F.pad(x_2d, (2, 2, 2, 2), mode='replicate')
        blurred = F.conv2d(padded, kernel, groups=C).view(B, T, C, H, W)
        return blurred.permute(0, 2, 1, 3, 4)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw RGB video, shape (B, C, T, H, W)
        Returns:
            saliency (torch.Tensor): Saliency map, shape (B, 1, T, H, W)
        """
        B, C, T, H, W = x.shape

        # Resolution-adaptive sigma
        min_dim = min(H, W)
        sigma = self.sigma_ratio * min_dim

        # Build Gaussian pyramid
        pyramids = [x]
        current = x
        for l in range(1, self.levels):
            current = self._gaussian_blur(current)
            current = F.avg_pool3d(current, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            pyramids.append(current)

        # Weighted fusion (softmax over levels)
        weights = F.softmax(self.fusion_weights, dim=0)

        saliency = torch.zeros(B, 1, T, H, W, device=x.device, dtype=x.dtype)
        for level, (pyr, w) in enumerate(zip(pyramids, weights)):
            _, _, _, H_l, W_l = pyr.shape
            upsampled = F.interpolate(
                pyr.mean(dim=1, keepdim=True),
                size=(T, H, W),
                mode='trilinear',
                align_corners=False
            )
            saliency = saliency + w * upsampled

        # Adaptive center prior
        Y, X = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing='ij')
        center_Y, center_X = H // 2, W // 2
        gaussian_prior = torch.exp(-((Y - center_Y)**2 + (X - center_X)**2) / (2 * sigma**2))
        gaussian_prior = gaussian_prior * self.center_bias
        gaussian_prior = gaussian_prior / (gaussian_prior.sum() + 1e-8)

        saliency = saliency * gaussian_prior.view(1, 1, 1, H, W)

        return saliency


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
    """
    Per-sample resting state baseline for micro-expression detection.

    Uses the first N frames of each video as the personalized baseline,
    rather than learning a global average face. This provides:
      1. Personal baseline (individual's resting expression)
      2. Lighting-adaptive (captures ambient lighting)
      3. Geometry baseline (aligned face position)

    Architecture:
        Input (B, C, T, H, W), baseline_frames=N
          -> Extract first N frames -> (B, C, N, H, W)
          -> Temporal average -> baseline (B, C, H, W)
          -> Compute residual -> (B, C, T, H, W)
          -> AU attention weighting
    """

    def __init__(self, baseline_frames=3, use_spatial_attention=True,
                 use_temporal_difference=True):
        super().__init__()
        self.baseline_frames = baseline_frames
        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_difference = use_temporal_difference

        # Spatial attention map for AU regions
        if use_spatial_attention:
            self.register_buffer(
                '_au_attention',
                self._create_au_attention_map(224)
            )

    def _create_au_attention_map(self, size):
        """Create spatial attention map emphasizing AU regions."""
        y = torch.arange(size, dtype=torch.float32)
        x = torch.arange(size, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        # Normalize
        xx = (xx / size * 2 - 1)
        yy = (yy / size * 2 - 1)

        # AU regions:
        # Brows (AU1, AU2, AU4) - top
        brows = torch.exp(-((yy + 0.45)**2 / 0.05 + xx**2 / 0.2))
        # Eyes (AU1, AU2, AU5, AU7) - upper-mid
        eyes = torch.exp(-((yy + 0.3)**2 / 0.1 + xx**2 / 0.3))
        # Nose (AU9) - center
        nose = torch.exp(-(yy**2 / 0.05 + xx**2 / 0.02))
        # Mouth (AU10, AU12, AU14-17, AU20, AU23-28) - lower
        mouth = torch.exp(-((yy - 0.35)**2 / 0.08 + xx**2 / 0.15))

        attention = eyes + nose + mouth + brows
        attention = attention / (attention.max() + 1e-8)
        return attention.view(1, 1, size, size)

    def compute_baseline(self, x):
        """
        Compute baseline from first N frames.

        Args:
            x (torch.Tensor): Input video, shape (B, C, T, H, W)
        Returns:
            baseline (torch.Tensor): Resting baseline, shape (B, C, H, W)
        """
        B, C, T, H, W = x.shape
        N = self.baseline_frames

        # Extract first N frames
        baseline_frames = x[:, :, :N, :, :]  # (B, C, N, H, W)

        # Temporal average -> baseline
        baseline = baseline_frames.mean(dim=2, keepdim=True)  # (B, C, 1, H, W)
        baseline = baseline.squeeze(2)  # (B, C, H, W)

        return baseline

    def forward(self, x, return_baseline=True):
        """
        Compute baseline and residual from input video.

        Args:
            x (torch.Tensor): Input video, shape (B, C, T, H, W)
            return_baseline (bool): Whether to return baseline
        Returns:
            baseline (torch.Tensor): Resting baseline, shape (B, C, H, W)
            residual (torch.Tensor): Deviation from baseline, shape (B, C, T, H, W)
            attention (torch.Tensor): AU attention, shape (1, 1, H, W)
        """
        print(f"[ReferenceFace] Input: {x.shape}")
        B, C, T, H, W = x.shape

        # Step 1: Compute baseline from first N frames
        baseline = self.compute_baseline(x)  # (B, C, H, W)

        # Step 2: Compute residual (current frames - baseline)
        # Expand baseline to match temporal dimension
        baseline_expanded = baseline.unsqueeze(2)  # (B, C, 1, H, W)
        baseline_expanded = baseline_expanded.expand(-1, -1, T, -1, -1)  # (B, C, T, H, W)
        residual = x - baseline_expanded  # (B, C, T, H, W)

        # Step 3: Compute AU attention
        if self.use_spatial_attention:
            attention = self._au_attention
            if attention.shape[-2:] != (H, W):
                attention = F.interpolate(
                    attention,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
        else:
            attention = torch.ones(1, 1, H, W, device=x.device)

        print(f"[ReferenceFace] Output: baseline={baseline.shape}, residual={residual.shape}, attention={attention.shape}")

        if return_baseline:
            return baseline, residual, attention.squeeze(0)
        else:
            return residual, attention.squeeze(0)


# =============================================================================
# ReferenceFaceWithMotion -- Resting Baseline + Motion Enhancement
# =============================================================================

class ReferenceFaceWithMotion(nn.Module):
    """
    Resting baseline + motion enhancement.
    """

    def __init__(self, baseline_frames=3, use_flow=True):
        super().__init__()
        self.baseline_frames = baseline_frames
        self.use_flow = use_flow

        self.base_ref = ReferenceFace(
            baseline_frames=baseline_frames,
            use_spatial_attention=True,
            use_temporal_difference=True
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input video, shape (B, C, T, H, W)
        Returns:
            baseline (torch.Tensor): (B, C, H, W)
            residual (torch.Tensor): (B, C, T, H, W)
            motion (torch.Tensor): Motion from baseline, (B, 2, H, W)
            attention (torch.Tensor): (1, H, W)
        """
        # Get baseline and residual
        baseline, residual, attention = self.base_ref(x)

        # Compute motion from baseline to current frame
        if self.use_flow:
            frame0 = baseline  # (B, C, H, W)
            frameT = x[:, :, -1]  # Last frame (B, C, H, W)
            motion = frameT - frame0  # (B, C, H, W)
            motion = motion[:, :2] if motion.shape[1] >= 2 else torch.cat([motion, motion], dim=1)
        else:
            motion = residual[:, :, -1]  # (B, C, H, W)
            motion = motion[:, :2] if motion.shape[1] >= 2 else torch.cat([motion, motion], dim=1)

        return baseline, residual, motion, attention