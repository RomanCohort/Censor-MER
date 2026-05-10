# =============================================================================
# Censor -- Fusiform-Amygdala Circuit Attention
# =============================================================================
# Implements the brain-inspired attention modulation circuit:
#   1. Amygdala: Fast threat detection -> Attention Prior Map
#   2. FFA (Fusiform Face Area): Feature recalibration between pathways
#   3. CASANet: 3D Contextual Attention Saliency-Aware Network
#      -> Inverted triangle region + Apex frame detection
# =============================================================================
#
# Neuroanatomical basis:
#   The fusiform gyrus (FFA) is specialized for face processing.
#   The amygdala provides emotional gating signals.
#   Together they form a "what + where" circuit for emotional faces.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.defaults import AMYGDALA_CONFIG, FFA_CONFIG, CASA_CONFIG


# =============================================================================
# Amygdala -- Emotion Salience Attention Prior Generator
# =============================================================================
# Biological analogy: The amygdala receives fast, crude visual input from the
# subcortical pathway (superior colliculus -> pulvinar) and generates an
# "emotional salience" signal that modulates cortical processing. This is the
# "emotion before cognition" mechanism - we react before we consciously perceive.
#
# Mathematical formulation:
#   M(x,y) = sigma( W_2 * ReLU( W_1 * f_fast + b_1 ) + b_2 )
# where f_fast is the Fast pathway output (512-dim),
# M is a 14x14 spatial attention prior map (APM).
# =============================================================================

class Amygdala(nn.Module):
    """
    Amygdala-inspired attention prior generator.

    Receives Fast pathway features (B, 512) and generates a spatial
    attention prior map (B, 1, 14, 14) highlighting high-motion facial regions
    (eye corners, nostril wings) for downstream cortical gating.

    Architecture:
        Input (B, 512)
          -> FC(512 -> 256) -> ReLU
          -> FC(256 -> 14*14)
          -> Reshape (B, 1, 14, 14) -> Sigmoid
          -> Output (B, 1, 14, 14) Attention Prior Map
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or AMYGDALA_CONFIG

        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.output_h, self.output_w = cfg['output_spatial']

        # Feature compression: Fast pathway -> salience vector
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Spatial expansion: salience vector -> 2D attention map
        output_neurons = self.output_h * self.output_w
        self.fc2 = nn.Linear(self.hidden_dim, output_neurons)
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, fast_feat):
        """
        Args:
            fast_feat (torch.Tensor): Fast pathway output, shape (B, 512)
        Returns:
            apm (torch.Tensor): Attention Prior Map, shape (B, 1, 14, 14)
        """
        print(f"[Amygdala] Input: {fast_feat.shape}")

        # Feature compression
        h = self.fc1(fast_feat)  # (B, 256)
        h = self.relu(h)

        # Spatial expansion
        h = self.fc2(h)  # (B, 14*14=196)
        apm = self.sigmoid(h.view(-1, 1, self.output_h, self.output_w))  # (B, 1, 14, 14)

        print(f"[Amygdala] Attention Prior Map: {apm.shape}")
        return apm


# =============================================================================
# FFA (Feature Fusion Attention) -- Mutual Pathway Recalibration
# =============================================================================
# Biological analogy: The Fusiform Face Area (FFA) integrates information
# from multiple visual pathways through reciprocal connections, reweighting
# features based on their relevance to face processing.
#
# Implements channel-wise recalibration between fast (512) and slow (768)
# features using a squeeze-and-excitation (SE) mechanism in the joint space.
#
# Mathematical formulation:
#   s = sigma( W_2 * ReLU( W_1 * [f_fast; f_slow] + b_1 ) + b_2 )
#   f_fast_gated = s[:512] * f_fast
#   f_slow_gated = s[512:] * f_slow
# =============================================================================

class FFA(nn.Module):
    """
    FFA (Feature Fusion Attention) module.

    Implements mutual channel recalibration between Fast and Slow pathways
    using an SE-style squeeze-and-excitation gate over the joint feature space.

    Architecture:
        Input: fast (B, 512), slow (B, 768)
          -> Concat -> (B, 1280)
          -> FC(1280 -> 80) -> ReLU (squeeze)
          -> FC(80 -> 1280) -> Sigmoid (excitation)
          -> Gate: multiply weights with features
          -> Split back to fast (B, 512) and slow (B, 768)
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or FFA_CONFIG

        fast_dim = cfg['fast_dim']
        slow_dim = cfg['slow_dim']
        joint_dim = fast_dim + slow_dim
        reduction = cfg['reduction_ratio']
        squeeze_dim = max(joint_dim // reduction, 16)

        # Squeeze: compress joint feature space
        self.squeeze = nn.Linear(joint_dim, squeeze_dim)
        self.relu = nn.ReLU(inplace=True)

        # Excitation: expand to channel-wise gating weights
        self.excitation = nn.Linear(squeeze_dim, joint_dim)
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        nn.init.kaiming_normal_(self.squeeze.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.squeeze.bias, 0)
        nn.init.xavier_uniform_(self.excitation.weight)
        nn.init.constant_(self.excitation.bias, 0)

    def forward(self, fast_feat, slow_feat):
        """
        Args:
            fast_feat (torch.Tensor): Fast pathway output, shape (B, 512)
            slow_feat (torch.Tensor): Slow pathway output, shape (B, 768)
        Returns:
            fast_gated (torch.Tensor): Gated fast features, shape (B, 512)
            slow_gated (torch.Tensor): Gated slow features, shape (B, 768)
        """
        print(f"[FFA] Inputs: fast={fast_feat.shape}, slow={slow_feat.shape}")

        # Joint representation
        joint = torch.cat([fast_feat, slow_feat], dim=1)  # (B, 1280)

        # Squeeze-and-Excitation
        s = self.squeeze(joint)  # (B, 80)
        s = self.relu(s)
        gating = self.excitation(s)  # (B, 1280)
        gating = self.sigmoid(gating)

        # Gate features
        fast_dim = fast_feat.shape[1]
        gate_fast = gating[:, :fast_dim]
        gate_slow = gating[:, fast_dim:]

        fast_gated = fast_feat * gate_fast
        slow_gated = slow_feat * gate_slow

        print(f"[FFA] Outputs: fast_gated={fast_gated.shape}, slow_gated={slow_gated.shape}")
        return fast_gated, slow_gated


# =============================================================================
# CASANet -- 3D Contextual Attention Saliency-Aware Network
# =============================================================================
# Purpose: Spatio-temporal attention specifically for micro-expression.
# Two functions:
#   1. Attend to the "inverted triangle" region (eyes + nose bridge + mouth)
#      where micro-expressions concentrate.
#   2. Detect the apex frame (frame with maximum expression intensity).
#
# Mathematical formulation:
#   A) Inverted triangle attention:
#      A_spatial(x,y) = Softmax( M[x,y] )
#      where M is a learnable mask with Gaussian initialization biased toward
#      the upper face center (inverted triangle shape).
#
#   B) Apex frame detection via temporal self-attention:
#      For each spatial position, compute attention across T_frames:
#      apex_score[t] = sum_{t'} Softmax(Q @ K^T / sqrt(d_k))[t, t'] * V[t']
#      The frame with the highest attention aggregation is the apex.
# =============================================================================

class CASANet(nn.Module):
    """
    CASANet -- Contextual Attention Saliency-Aware Network.

    Spatio-temporal attention module for micro-expression recognition.
    Applies inverted-triangle spatial attention + temporal apex detection.

    Architecture:
        Input: (B, 768, T_s, H_s, W_s) -- slow pathway spatial feature map
          -> Part A: Inverted Triangle Spatial Attention (learned mask)
          -> Part B: Temporal Self-Attention for Apex Detection
          -> Output: (B, 768, T_s, H_s, W_s) attended features
          -> Output: (B, T_s) apex frame scores
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or CASA_CONFIG

        embed_dim = cfg['embed_dim']
        num_heads = cfg['num_heads']
        ffn_dim = cfg.get('ffn_dim', embed_dim * 4)
        pyramid_h, pyramid_w = cfg['pyramid_size'][1], cfg['pyramid_size'][2]

        # =====================================================================
        # Part A: Inverted Triangle Spatial Attention Mask
        # =====================================================================
        # Learnable spatial mask initialized with inverted triangle prior
        # Broader at top (eye region), narrower at bottom (mouth)
        self.spatial_mask = nn.Parameter(torch.zeros(1, 1, pyramid_h, pyramid_w), requires_grad=True)
        self._init_triangle_prior(pyramid_h, pyramid_w)

        # =====================================================================
        # Part B: Temporal Self-Attention for Apex Detection
        # =====================================================================
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False,
            dropout=0.1
        )
        self.temp_proj = nn.Linear(embed_dim, embed_dim)
        self.temp_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(0.1)
        )
        nn.init.xavier_uniform_(self.temp_proj.weight)
        nn.init.constant_(self.temp_proj.bias, 0)

        # Output projection
        self.output_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.output_proj.weight, mode='fan_out', nonlinearity='relu')

    def _init_triangle_prior(self, h, w):
        """
        Initialize spatial mask with inverted triangle prior.
        Values are higher in the upper-center face region (eyes, nose bridge)
        and taper down toward the jawline.

        Mathematical: M(y,x) = -( (x-cx)² / w(y)² + (y-cy)² / 0.01 )
        where w(y) = 0.3 * (1 - |y-cy|_norm * 0.5) narrows toward bottom.
        """
        with torch.no_grad():
            cx, cy = w / 2, h / 2
            for y in range(h):
                for x in range(w):
                    rel_y = (y - cy) / (h)  # normalized vertical position
                    rel_x = (x - cx) / (w)  # normalized horizontal position
                    # Width narrows linearly from top to bottom (inverted triangle)
                    width_at_y = 0.3 * (1 - abs(rel_y) * 0.5)
                    # Gaussian-like score in inverted triangle region
                    score = -((rel_x / (width_at_y + 0.01))**2 + (rel_y / 0.6)**2) / 0.1
                    self.spatial_mask[0, 0, y, x] = score
        print(f"[CASANet] Triangle prior initialized: {self.spatial_mask.shape}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Slow pathway feature map, shape (B, 768, T_s, H_s, W_s)
        Returns:
            attended (torch.Tensor): Attended features, shape (B, 768, T_s, H_s, W_s)
            apex_scores (torch.Tensor): Apex frame scores, shape (B, T_s)
        """
        print(f"[CASANet] Input: {x.shape}")
        B, C, T_s, H_s, W_s = x.shape

        # =====================================================================
        # Part A: Inverted Triangle Spatial Attention
        # =====================================================================
        # Interpolate spatial mask to match feature map resolution
        spatial_weights = self.spatial_mask
        if H_s != spatial_weights.shape[2] or W_s != spatial_weights.shape[3]:
            spatial_weights = F.interpolate(
                spatial_weights, size=(H_s, W_s), mode='bilinear', align_corners=False
            )

        # Apply softmax over spatial dimensions to get attention distribution
        attn_weights = F.softmax(spatial_weights.view(1, 1, -1), dim=2).view(1, 1, H_s, W_s)

        # Apply spatial attention: attended = x * attention_mask + x (residual)
        spatial_attended = x * attn_weights.unsqueeze(2).expand(-1, -1, T_s, -1, -1)
        # Residual connection
        spatial_attended = spatial_attended + x
        print(f"[CASANet] After spatial attention: {spatial_attended.shape}")

        # =====================================================================
        # Part B: Temporal Self-Attention for Apex Detection
        # =====================================================================
        # Prepare for temporal attention: aggregate spatial information
        # (B, 768, T_s, H_s, W_s) -> (B, H_s*W_s, T_s, 768)
        attended_perm = spatial_attended.permute(0, 3, 4, 2, 1)  # (B, H_s, W_s, T_s, 768)
        attended_flat = attended_perm.reshape(B * H_s * W_s, T_s, C)  # (B*H*W, T_s, 768)

        # Temporal self-attention
        # Sequence dim = time, batch dim = B*H*W
        h_t = attended_flat.permute(1, 0, 2)  # (T_s, B*H*W, 768)
        attn_out, raw_attn_weights = self.temporal_attn(h_t, h_t, h_t)
        # raw_attn_weights: (B*H*W, T_s, T_s) - not needed separately

        # Residual + FFN
        h_t = self.temp_norm(h_t + attn_out)
        h_t = h_t + self.ffn(h_t)
        h_t = h_t.permute(1, 0, 2)  # (B*H*W, T_s, 768)
        temporal_attended = h_t.reshape(B, H_s, W_s, T_s, C).permute(0, 4, 3, 1, 2)  # (B, 768, T_s, H_s, W_s)

        # Compute apex frame scores
        # Higher temporal attention aggregation -> more likely apex frame
        apex_scores = temporal_attended.mean(dim=[-1, -2, 1])  # (B, T_s)

        # Output projection
        attended = self.output_proj(temporal_attended)  # (B, 768, T_s, H_s, W_s)

        print(f"[CASANet] Output features: {attended.shape}")
        print(f"[CASANet] Apex scores: {apex_scores.shape}")

        return attended, apex_scores