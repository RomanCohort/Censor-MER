# =============================================================================
# Censor -- Biomimetic Image Generator (Full Integration + Anti-Overfitting)
# =============================================================================
# Generates realistic facial images with biomimetic light/shadow and micro-expression.
#
# Architecture:
#   1. Dual-Pathway Fusion: Fast(512) + Slow(768) -> Fused(1024)
#   2. Sparse Control: 防过拟合 (剪枝/Dropout/L2)
#   3. Emotion-Aware: Amygdala fast attention -> emotional lighting
#   4. Spatial-Aware: CASANet attention -> region-based lighting
#   5. Bio-Signals: rPPG -> skin tone rendering
#   6. Visual Conditioning: AU features -> illumination
#   7. Feature Decoding: Fused features -> latent image features
#   8. Upsampling: Latent -> RGB image
#   9. Light/Shadow: Biomimetic effects
#   10. Micro-expression: Temporal dynamics
# =============================================================================
# Biological basis:
#   - Pupil: Controls light intake (illumination)
#   - Retina: Adaptive gain (local contrast)
#   - Mach bands: Edge overshoot (sharp transitions)
#   - Ganglion cells: Center-surround (edge detection)
#   - rPPG: Blood flow -> skin tone
#   - Saliency: Visual attention
#   - Amygdala: Emotional response
#   - CASANet: Spatial attention
#   - LongTermMemorySparseControl: Synaptic pruning (防过拟合)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.defaults import (
    VISUAL_PERCEPTION_CONFIG,
    AU_DECODER_CONFIG,
    DATA_CONFIG,
    FFA_CONFIG,
    AMYGDALA_CONFIG,
    CASA_CONFIG,
    SPARSE_CONTROL_CONFIG,
)


class AUToIllumination(nn.Module):
    """
    Maps Action Unit intensities to illumination parameters.
    Simulates how facial muscle activations affect perceived lighting.

    AU positions influence shadow regions:
    - AU4 (Brow Lowerer): forehead shadow
    - AU6 (Cheek Raiser): cheek illumination
    - AU10, AU12: mouth region lighting
    """

    def __init__(self, num_aus=28):
        super().__init__()
        self.num_aus = num_aus

        # AU region groupings
        self.forehead_aus = [4]  # AU4
        self.cheek_aus = [6, 7]  # AU6, AU7
        self.mouth_aus = [10, 12, 14, 15, 17, 20, 23, 24, 25, 26, 27]  # Multiple mouth AUs
        self.eye_aus = [1, 2, 5]  # AU1, AU2, AU5

        # FC to illumination parameters
        # Output: [illumination_scale, shadow_intensity, rim_light, ambient]
        self.illum_proj = nn.Sequential(
            nn.Linear(num_aus, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()  # All outputs in [0, 1]
        )

    def forward(self, au_intensities):
        """
        Args:
            au_intensities: (B, T, num_aus) or (B, num_aus)
        Returns:
            illum_params: (B, 4) or (B, T, 4)
        """
        # Handle both temporal and non-temporal
        has_time = au_intensities.dim() == 3

        if has_time:
            B, T, _ = au_intensities.shape
            # Use peak AU intensity across time
            au_peak = au_intensities.max(dim=1)[0]  # (B, num_aus)
        else:
            au_peak = au_intensities

        illum_params = self.illum_proj(au_peak)  # (B, 4)

        if has_time:
            illum_params = illum_params.unsqueeze(1).expand(-1, T, -1)

        return illum_params


class IlluminativeRenderer(nn.Module):
    """
    Renders illumination effects based on AU-derived parameters.
    Simulates studio lighting based on facial muscle positions.
    """

    def __init__(self, image_size=224):
        super().__init__()
        self.image_size = image_size

        # Create coordinate grids
        self.register_buffer('coords', self._create_coords())
        self.register_buffer('center_mask', self._create_center_mask())

    def _create_coords(self):
        """Create normalized coordinate grid [-1, 1]"""
        h = w = self.image_size
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx, yy], dim=0)  # (2, H, W)
        return coords.unsqueeze(0)  # (1, 2, H, W)

    def _create_center_mask(self):
        """Create face center mask (oval)"""
        coords = self.coords  # (1, 2, H, W)
        x, y = coords[:, 0], coords[:, 1]

        # Ellipse for face center (slightly wider than tall)
        dist = (x / 0.7) ** 2 + y ** 2
        mask = (dist < 1.0).float()  # (1, H, W)
        return mask

    def forward(self, image, illum_params):
        """
        Apply illumination to image.

        Args:
            image: (B, C, H, W) RGB, should be in [0, 1]
            illum_params: (B, 4) or (B, T, 4) - [scale, shadow, rim, ambient]
        Returns:
            illuminated: (B, C, H, W)
        """
        B, C, H, W = image.shape

        # Handle temporal input - use mean across time
        if illum_params.dim() == 3:
            illum_params = illum_params.mean(dim=1)  # (B, 4)

        # Unpack parameters
        illum_scale = illum_params[:, 0:1].view(B, 1, 1, 1)  # Overall brightness
        shadow = illum_params[:, 1:2].view(B, 1, 1, 1)  # Shadow depth
        rim = illum_params[:, 2:3].view(B, 1, 1, 1)  # Rim light intensity
        ambient = illum_params[:, 3:4].view(B, 1, 1, 1)  # Ambient

        illuminated = image.clone()

        # 1. Overall illumination
        illuminated = illuminated * illum_scale

        # 2. Center vs edge illumination (Rembrandt-style)
        center_mask = self.center_mask  # (1, H, W)
        center_mask = center_mask.expand(B, -1, -1)  # (B, H, W)

        # Brighten center (key light)
        illuminated = illuminated + (1 - center_mask.unsqueeze(1)) * 0.1 * illum_scale

        # 3. Rim light on edges
        rim_mask = 1 - center_mask  # Edges
        illuminated = illuminated + rim_mask.unsqueeze(1) * rim * 0.3

        # 4. Ambient fill
        illuminated = illuminated + ambient * 0.2

        # 5. Shadow in center (subtle depth)
        illuminated = illuminated * center_mask.unsqueeze(1) * (1 - shadow * 0.1) + \
                      illuminated * (1 - center_mask.unsqueeze(1))

        return illuminated.clamp(0, 1)


# =============================================================================
# Dual-Pathway Fusion Module (Fast + Slow)
# =============================================================================

class DualPathwayFusion(nn.Module):
    """
    Dual-Pathway Fusion: Fast(512) + Slow(768) -> Fused(1024)

    Combines:
    - Fast Pathway: 光流特征 (运动信息)
    - Slow Pathway: RGB+rPPG特征 (外观+脉搏信息)

    Uses SE-style attention with optional DTN tension gating.
    """

    def __init__(self, fast_dim=512, slow_dim=768, output_dim=1024, use_dtn=False):
        super().__init__()
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.output_dim = output_dim
        self.use_dtn = use_dtn

        # Joint dimension
        joint_dim = fast_dim + slow_dim

        # SE-style squeeze
        self.squeeze = nn.Linear(joint_dim, max(joint_dim // 16, 32))
        self.relu = nn.ReLU(inplace=True)
        self.excitation = nn.Linear(max(joint_dim // 16, 32), joint_dim)
        self.sigmoid = nn.Sigmoid()

        # Optional: DTN tension (biomimetic)
        if use_dtn:
            self.tension_fast = nn.Linear(fast_dim, 1)
            self.tension_slow = nn.Linear(slow_dim, 1)
            self.dtn_weight = nn.Parameter(torch.tensor(0.3))

        # Output projection
        self.output_proj = nn.Linear(joint_dim, output_dim)

        print(f"[DualPathwayFusion] Fast={fast_dim} + Slow={slow_dim} -> Output={output_dim}, DTN={use_dtn}")

    def forward(self, fast_feat, slow_feat):
        """
        Args:
            fast_feat: (B, 512) - Fast pathway features (光流)
            slow_feat: (B, 768) - Slow pathway features (RGB+rPPG)
        Returns:
            fused: (B, output_dim) - Fused features
        """
        B = fast_feat.shape[0]

        # SE gating
        joint = torch.cat([fast_feat, slow_feat], dim=1)
        s = self.squeeze(joint)
        s = self.relu(s)
        se_gate = self.excitation(s)
        se_gate = self.sigmoid(se_gate)

        # Apply gating
        gated = joint * se_gate

        # DTN tension (optional)
        if self.use_dtn:
            tension_f = self.tension_fast(fast_feat)
            tension_s = self.tension_slow(slow_feat)
            tension = torch.cat([tension_f, tension_s], dim=1)
            tension_gate = torch.sigmoid(tension)
            dtn_w = self.dtn_weight.sigmoid()
            gated = gated * (1 - dtn_w) + gated * tension_gate * dtn_w

        # Project to output dimension
        fused = self.output_proj(gated)

        return fused


# Import anti-overfitting mechanisms from biomimetic_enhance
from model.biomimetic_enhance import (
    LongTermMemorySparseControl,
    NeuronUsageTracker,
    HardFreezePath,
    SoftDecayPath,
    GrowthFactorSignal,
)

# =============================================================================
# Emotion-Aware Lighting Module (基于Amygdala)
# =============================================================================

class EmotionalLighting(nn.Module):
    """
    基于杏仁核(Amygdala)机制的情绪感知光照模块。

    将快速通路情绪反应映射到光照参数：
    - 情绪强烈 -> 光照对比度增加
    - 情绪平稳 -> 柔和光照
    - 情绪激活 -> 高光强调
    """

    def __init__(self, input_dim=512):
        super().__init__()
        self.input_dim = input_dim

        # 情绪反应 -> 光照参数
        # 输出: [intensity, contrast, warmth, saturation]
        self.emotion_to_light = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, fast_feat):
        """
        Args:
            fast_feat: (B, 512) - Fast pathway features
        Returns:
            light_params: (B, 4) - [intensity, contrast, warmth, saturation]
        """
        return self.emotion_to_light(fast_feat)


# =============================================================================
# Spatial Attention Lighting (基于CASANet)
# =============================================================================

class SpatialAttentionLighting(nn.Module):
    """
    基于CASANet的空间注意力光照模块。

    使用倒三角空间注意力：
    - 眼睛/鼻子区域 -> 高亮
    - 边缘区域 -> 柔和
    """

    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim

        # 空间注意力权重
        self.register_buffer('spatial_prior', self._create_triangular_prior())

        # 注意力到光照
        self.attn_to_illum = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [center_weight, rim_weight, ambient]
            nn.Sigmoid()
        )

    def _create_triangular_prior(self):
        """创建倒三角空间先验 (眼睛>鼻子>嘴巴)"""
        h, w = 7, 7  # 对应下采样后的尺寸
        prior = torch.zeros(h, w)

        for i in range(h):
            for j in range(w):
                # 眼睛在顶部，嘴巴在底部
                row_weight = 1.0 - (i / h) * 0.5
                # 中心更亮
                col_dist = abs(j - w / 2) / (w / 2)
                col_weight = 1.0 - col_dist * 0.3
                prior[i, j] = row_weight * col_weight

        prior = prior / prior.max()
        return prior.unsqueeze(0)  # (1, H, W)

    def forward(self, spatial_feat):
        """
        Args:
            spatial_feat: (B, 768, 7, 7) - Slow pathway spatial features
        Returns:
            illum_weights: (B, 3) - [center, rim, ambient]
        """
        B = spatial_feat.shape[0]

        # 全局池化 + 空间先验加权
        weighted = spatial_feat * self.spatial_prior.view(1, 1, 7, 7)
        pooled = weighted.sum(dim=[2, 3]) / (7 * 7)  # (B, 768)

        return self.attn_to_illum(pooled)


# =============================================================================
# Bio-Signal Lighting (基于rPPG)
# =============================================================================

class BioSignalLighting(nn.Module):
    """
    基于rPPG的生物信号光照模块。

    将脉搏信号映射到面部肤色：
    - 高血流量 -> 红润面色
    - 低血流量 -> 苍白面色
    """

    def __init__(self):
        super().__init__()

        # rPPG强度 -> 肤色参数
        # 输入: 假设rPPG特征维度
        # 输出: [skin_rouge, skin_green, skin_blue, blood_flow]
        self.rppg_to_skin = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )

    def forward(self, rppg_signal):
        """
        Args:
            rppg_signal: (B, T) or scalar - rPPG signal
        Returns:
            skin_params: (B, 4) - [rouge, green, blue, blood_flow]
        """
        # 处理多维度输入
        if rppg_signal.dim() > 1:
            # 取峰值或均值
            rppg_peak = rppg_signal.max(dim=1)[0].mean(dim=1, keepdim=True)
        else:
            rppg_peak = rppg_signal.unsqueeze(0) if rppg_signal.dim() == 0 else rppg_signal

        return self.rppg_to_skin(rppg_peak)


# =============================================================================
# Integrated Biomimetic Image Generator (Full Integration)
# =============================================================================

class BiomimeticImageGenerator(nn.Module):
    """
    完全集成的仿生图像生成器。

    整合所有识别机制：
    1. Dual-Pathway Fusion: Fast(512) + Slow(768) -> Fused(1024)
    2. EmotionalLighting (Amygdala): 情绪 -> 光照
    3. SpatialAttentionLighting (CASANet): 空间 -> 光照区域
    4. BioSignalLighting (rPPG): 脉搏 -> 肤色
    5. AU-to-Illumination: AU -> 光照参数
    6. VisualPerception: 生物视觉后处理
    7. Micro-expression: 时序动态

    输出: 逼真面部图像 + 光影 + 微表情
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}

        # Get dimensions from configs
        au_cfg = AU_DECODER_CONFIG
        data_cfg = DATA_CONFIG
        ffa_cfg = FFA_CONFIG
        amygdala_cfg = AMYGDALA_CONFIG
        casa_cfg = CASA_CONFIG
        sparse_cfg = SPARSE_CONTROL_CONFIG

        self.num_aus = au_cfg.get('num_aus', 28)
        self.temporal_steps = data_cfg.get('T', 16)
        self.image_size = data_cfg.get('H', 224)
        self.width = data_cfg.get('W', 224)

        # ============================================================
        # Dual-Pathway Configuration
        # ============================================================
        self.fast_dim = cfg.get('fast_dim', ffa_cfg.get('fast_dim', 512))   # Fast Pathway
        self.slow_dim = cfg.get('slow_dim', ffa_cfg.get('slow_dim', 768))  # Slow Pathway
        self.fused_dim = cfg.get('fused_dim', 1024)

        # ============================================================
        # 新增: 防过拟合机制 (Sparse Control)
        # ============================================================
        self.enable_sparse_control = cfg.get('enable_sparse_control', True)

        if self.enable_sparse_control:
            # 创建稀疏控制器
            sparse_config = sparse_cfg.copy()
            sparse_config['dim'] = self.fused_dim
            self.sparse_control = LongTermMemorySparseControl(sparse_config)
            print("[BiomimeticImageGenerator] Sparse control enabled")

        # ============================================================
        # 新增: 识别机制集成
        # ============================================================
        # 1. 情绪感知光照 (Amygdala)
        self.emotional_lighting = EmotionalLighting(input_dim=self.fast_dim)

        # 2. 空间注意力光照 (CASANet)
        self.spatial_lighting = SpatialAttentionLighting(embed_dim=self.slow_dim)

        # 3. 生物信号光照 (rPPG) - 可选
        self.bio_lighting = BioSignalLighting()

        # ============================================================
        # Dual-Pathway Fusion
        # ============================================================
        self.pathway_fusion = DualPathwayFusion(
            fast_dim=self.fast_dim,
            slow_dim=self.slow_dim,
            output_dim=self.fused_dim
        )

        # Input feature dimension (from fusion)
        self.input_dim = self.fused_dim

        # Latent dimension for image generation
        self.latent_dim = cfg.get('latent_dim', 512)

        # =================================================================
        # Stage 1: Feature to Latent
        # =================================================================
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(inplace=True),
        )

        # =================================================================
        # Stage 2: Latent to Feature Map (upsampling)
        # =================================================================
        # Use transposed conv for upsampling
        # latent (B, 512) -> feature map (B, 128, 14, 14)
        self.latent_to_feature = nn.Sequential(
            nn.Linear(self.latent_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 28 -> 56
        self.upsample4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 56 -> 112
        self.upsample5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 112 -> 224

        # =================================================================
        # Stage 3: Feature to RGB
        # =================================================================
        self.to_rgb = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # =================================================================
        # Stage 4: AU to Illumination
        # =================================================================
        self.au_to_illum = AUToIllumination(self.num_aus)
        self.illuminator = IlluminativeRenderer(self.image_size)

        # =================================================================
        # Stage 5: Visual Perception Post-Processing
        # =================================================================
        from visual_perception import VisualPerceptionPostProcess
        self.visual_perception = VisualPerceptionPostProcess(VISUAL_PERCEPTION_CONFIG)

        # =================================================================
        # Stage 6: Micro-expression Overlay
        # =================================================================
        self.micro_expr_overlay = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        print(f"[BiomimeticImageGenerator] Initialized: fast={self.fast_dim}, slow={self.slow_dim}, fused={self.fused_dim}")

    def forward(self, fast_feat=None, slow_feat=None, au_intensities=None, apply_visual_perception=True,
              rppg_signal=None, emotion_fast_feat=None, spatial_slow_feat=None):
        """
        Generate realistic facial image from dual-pathway features.

        Args:
            fast_feat: (B, fast_dim=512) OR fused_feat (B, 1024)
            slow_feat: (B, slow_dim=768) OR None
            au_intensities: (B, T, num_aus) or (B, num_aus) AU intensities (optional)
            apply_visual_perception: bool, apply biomimetic post-processing
            rppg_signal: (B, T) or None - rPPG signal for skin tone
            emotion_fast_feat: (B, 512) or None - Fast features for情绪光照
            spatial_slow_feat: (B, 768, 7, 7) or None - Spatial features for 空间光照

        Returns:
            generated_image: (B, C, H, W) RGB image in [0, 1]
        """
        # ============================================================
        # Handle input: dual-pathway or fused
        # ============================================================
        if slow_feat is not None:
            # Dual-pathway input: Fast + Slow
            fused_feat = self.pathway_fusion(fast_feat, slow_feat)
            print(f"[BiomimeticImageGenerator] Fused dual-pathway: {fused_feat.shape}")

            # 使用双通道特征用于光照模块
            _emotion_feat = fast_feat if emotion_fast_feat is None else emotion_fast_feat
            _spatial_feat = (None, slow_feat)  # 占位，后续处理
        elif fast_feat is not None:
            # Assume single fused input
            fused_feat = fast_feat
            _emotion_feat = None
            _spatial_feat = None
        else:
            raise ValueError("Must provide either fast_feat or (fast_feat, slow_feat)")

        B = fused_feat.shape[0]

        # =================================================================
        # Stage 1: Multi-Source Lighting
        # =================================================================
        # 1) Emotional lighting (Amygdala-based)
        if _emotion_feat is not None:
            emotion_light = self.emotional_lighting(_emotion_feat)  # (B, 4)
            print(f"[BiomimeticImageGenerator] Emotion lighting: {emotion_light[0].tolist()}")

        # 2) Spatial attention lighting (CASANet-based)
        if spatial_slow_feat is not None:
            spatial_light = self.spatial_lighting(spatial_slow_feat)  # (B, 3)
            print(f"[BiomimeticImageGenerator] Spatial lighting: {spatial_light[0].tolist()}")

        # 3) Bio-signal lighting (rPPG-based)
        if rppg_signal is not None:
            bio_light = self.bio_lighting(rppg_signal)  # (B, 4)
            print(f"[BiomimeticImageGenerator] Bio lighting: {bio_light[0].tolist()}")

        # =================================================================
        # ============================================================
        # Apply Sparse Control (防过拟合) - 在特征融合后
        # ============================================================
        if self.enable_sparse_control and hasattr(self, 'sparse_control'):
            # 应用到fused_feat而不是latent
            fused_feat, latent_stats = self.sparse_control(fused_feat)
            print(f"[BiomimeticImageGenerator] Sparse control: stats={latent_stats}")

        # =================================================================
        # Stage 2: Feature Encoding
        # =================================================================
        latent = self.feature_encoder(fused_feat)  # (B, latent_dim)

        # =================================================================
        # Stage 3: Latent to Feature Map
        # =================================================================
        feat_map = self.latent_to_feature(latent)  # (B, 512*7*7)
        feat_map = feat_map.view(B, 512, 7, 7)  # (B, 512, 7, 7)

        # Upsampling chain
        feat_map = F.relu(self.upsample1(feat_map))  # (B, 256, 14, 14)
        feat_map = F.relu(self.upsample2(feat_map))  # (B, 128, 28, 28)
        feat_map = F.relu(self.upsample3(feat_map))  # (B, 64, 56, 56)
        feat_map = F.relu(self.upsample4(feat_map))  # (B, 32, 112, 112)
        feat_map = F.relu(self.upsample5(feat_map))  # (B, 16, 224, 224)

        # =================================================================
        # Stage 4: To RGB
        # =================================================================
        image = self.to_rgb(feat_map)  # (B, 3, 224, 224)
        print(f"[BiomimeticImageGenerator] Generated base image: {image.shape}")

        # =================================================================
        # Stage 5: AU-based Illumination (if AU provided)
        # =================================================================
        if au_intensities is not None:
            illum_params = self.au_to_illum(au_intensities)  # (B, 4)
            image = self.illuminator(image, illum_params)  # Apply lighting
            print(f"[BiomimeticImageGenerator] Applied illumination")

        # =================================================================
        # Stage 5: Visual Perception Post-Processing
        # =================================================================
        if apply_visual_perception:
            image = self.visual_perception(image)
            print(f"[BiomimeticImageGenerator] Applied visual perception")

        # =================================================================
        # Stage 6: Micro-expression Enhancement
        # =================================================================
        # Temporal difference for micro-expression dynamics
        if au_intensities is not None and au_intensities.dim() == 3:
            # Temporal: compute subtle change overlay
            if au_intensities.shape[1] > 1:
                # Frame difference
                diff = au_intensities[:, 1:] - au_intensities[:, :-1]  # (B, T-1, num_aus)
                # Use peak difference for overlay intensity
                diff_strength = diff.abs().max(dim=1)[0].mean(dim=1, keepdim=True)  # (B, 1)
                # Apply subtle overlay
                overlay = self.micro_expr_overlay(image * 1.1) - 0.05  # Enhance subtle differences
                # Blend based on expression strength
                image = image + (overlay - image) * diff_strength.view(B, 1, 1, 1) * 0.1

        print(f"[BiomimeticImageGenerator] Final output: {image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
        return image.clamp(0, 1)


class LightShadowGenerator(nn.Module):
    """
    Simplified Light/Shadow Generator.

    Generates lighting effects for facial images without full generation.
    Best used as a post-processing module.
    """

    def __init__(self, image_size=224):
        super().__init__()
        self.image_size = image_size

        # Light direction estimation (simple)
        self.light_encoder = nn.Sequential(
            nn.Linear(3, 16),  # Assume 3 AU groups
            nn.ReLU(),
            nn.Linear(16, 2),  # light_x, light_y in [-1, 1]
            nn.Tanh()
        )

        # Coordinate grids
        self.register_buffer('coords', self._create_coords())

    def _create_coords(self):
        h = w = self.image_size
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx, yy], dim=0)
        return coords.unsqueeze(0)

    def forward(self, image, au_regions):
        """
        Apply directional lighting.

        Args:
            image: (B, C, H, W) RGB in [0, 1]
            au_regions: (B, 3) regional AU activities [forehead, cheek, mouth]
        """
        B, C, H, W = image.shape

        # Estimate light direction
        light_dir = self.light_encoder(au_regions)  # (B, 2)

        # Create lighting field
        coords = self.coords.expand(B, -1, -1, -1)  # (B, 2, H, W)
        light_field = coords - light_dir.view(B, 2, 1, 1)  # Distance from light
        light_intensity = 1 - light_field.pow(2).sum(dim=1, keepdim=True) * 0.5  # (B, 1, H, W)

        # Apply lighting
        lit_image = image * light_intensity

        return lit_image.clamp(0, 1)


# =============================================================================
# Integration: Full Generation Pipeline
# =============================================================================

class BiomimeticGenerationPipeline(nn.Module):
    """
    Full biomimetic generation pipeline with dual-pathway support.

    Combines:
    - Dual-Pathway Fusion: Fast(512) + Slow(768) -> Fused(1024)
    - AU decoder + Illumination + Image Generation + Visual Perception
    """

    def __init__(self, config=None):
        super().__init__()

        # Main generator
        self.generator = BiomimeticImageGenerator(config)

        # Optional: separate light/shadow module
        self.light_shadow = LightShadowGenerator()

        print("[BiomimeticGenerationPipeline] Initialized with dual-pathway")

    def forward(self, fast_feat, slow_feat, au_intensities=None):
        """
        Generate final image from dual-pathway features.

        Args:
            fast_feat: (B, 512) - Fast pathway features (光流)
            slow_feat: (B, 768) - Slow pathway features (RGB+rPPG)
            au_intensities: (B, T, 28) or (B, 28)

        Returns:
            image: (B, 3, 224, 224)
        """
        return self.generator(fast_feat=fast_feat, slow_feat=slow_feat, au_intensities=au_intensities)


# =============================================================================
# Factory Function
# =============================================================================

def create_biomimetic_generator(config=None):
    """Create the biomimetic image generator"""
    return BiomimeticGenerationPipeline(config)