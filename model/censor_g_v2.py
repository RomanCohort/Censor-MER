# Censor-G v2: Visual Cortex-Inspired ME Generation
# =============================================================================
# 核心创新：融合视觉中枢的分层处理概念
#
# 架构：
#   V1层：AU显著性筛选（快速识别哪些AU需要精细处理）
#   V2层：AU交互矩阵（协同/对抗建模）
#   V3层：AU时间动力学（快肌/慢肌的不同时间曲线）
#   V4层：局部运动场生成（每个AU控制特定面部区域）
#   IT层：运动场融合与最终生成
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# AU常量定义（基于FACS）
# =============================================================================

# 17个AU及其肌肉类型
AU_INFO = {
    'AU1': {'name': 'Inner Brow Raiser', 'muscle_type': 'fast', 'region': 'eyebrow_inner'},
    'AU2': {'name': 'Outer Brow Raiser', 'muscle_type': 'fast', 'region': 'eyebrow_outer'},
    'AU4': {'name': 'Brow Lowerer', 'muscle_type': 'fast', 'region': 'eyebrow'},
    'AU5': {'name': 'Upper Lid Raiser', 'muscle_type': 'fast', 'region': 'eye_upper'},
    'AU6': {'name': 'Cheek Raiser', 'muscle_type': 'mixed', 'region': 'cheek'},
    'AU7': {'name': 'Lid Tightener', 'muscle_type': 'fast', 'region': 'eye'},
    'AU9': {'name': 'Nose Wrinkler', 'muscle_type': 'fast', 'region': 'nose'},
    'AU10': {'name': 'Upper Lip Raiser', 'muscle_type': 'mixed', 'region': 'lip_upper'},
    'AU12': {'name': 'Lip Corner Puller', 'muscle_type': 'mixed', 'region': 'mouth_corner'},
    'AU14': {'name': 'Dimpler', 'muscle_type': 'slow', 'region': 'mouth_corner'},
    'AU15': {'name': 'Lip Corner Depressor', 'muscle_type': 'slow', 'region': 'mouth_corner'},
    'AU17': {'name': 'Chin Raiser', 'muscle_type': 'slow', 'region': 'chin'},
    'AU20': {'name': 'Lip Stretcher', 'muscle_type': 'mixed', 'region': 'mouth'},
    'AU23': {'name': 'Lip Tightener', 'muscle_type': 'slow', 'region': 'mouth'},
    'AU24': {'name': 'Lip Pressor', 'muscle_type': 'slow', 'region': 'mouth'},
    'AU25': {'name': 'Lips Part', 'muscle_type': 'mixed', 'region': 'mouth'},
    'AU26': {'name': 'Jaw Drop', 'muscle_type': 'slow', 'region': 'jaw'},
}

# AU索引映射
AU_INDEX = {au: i for i, au in enumerate(sorted(AU_INFO.keys()))}

# AU区域定义（像素坐标范围，基于224x224图像）
AU_REGIONS = {
    'eyebrow_inner': {'center': (112, 60), 'radius': 20},
    'eyebrow_outer': {'center': (80, 55), 'radius': 15},
    'eyebrow': {'center': (112, 60), 'radius': 30},
    'eye_upper': {'center': (100, 80), 'radius': 25},
    'eye': {'center': (100, 80), 'radius': 20},
    'cheek': {'center': (90, 100), 'radius': 30},
    'nose': {'center': (112, 100), 'radius': 15},
    'lip_upper': {'center': (112, 130), 'radius': 15},
    'mouth_corner': {'center': (90, 140), 'radius': 20},
    'mouth': {'center': (112, 140), 'radius': 25},
    'chin': {'center': (112, 160), 'radius': 20},
    'jaw': {'center': (112, 180), 'radius': 30},
}


# =============================================================================
# V1层：AU显著性筛选
# =============================================================================

class V1AUSaliency(nn.Module):
    """
    V1层：AU显著性筛选。

    快速识别哪些AU需要精细处理。
    借鉴Civis Lucri-Faber的AdaptiveVisualAttention两阶段机制。

    处理流程：
      1. 输入AU激活向量 (17维)
      2. 快速评分：哪些AU显著（> threshold）
      3. 精细处理的AU进入V2层
      4. 不显著的AU直接使用模板运动场
    """

    def __init__(self, num_au=17, threshold=0.1):
        super().__init__()

        self.num_au = num_au
        self.threshold = threshold

        # AU显著性评分器
        self.saliency_scorer = nn.Sequential(
            nn.Linear(num_au, 32),
            nn.ReLU(),
            nn.Linear(32, num_au),
            nn.Sigmoid()
        )

        # 预定义的模板运动场（用于不显著的AU）
        self.template_fields = self._create_template_fields()

    def forward(self, au_activation):
        """
        Args:
            au_activation (torch.Tensor): AU激活，shape (B, 17)

        Returns:
            significant_au (torch.Tensor): 显著AU索引
            saliency_scores (torch.Tensor): 显著性评分
            fast_path_mask (torch.Tensor): 快速路径掩码
        """
        B = au_activation.shape[0]

        # 计算显著性评分
        saliency_scores = self.saliency_scorer(au_activation)  # (B, 17)

        # 识别显著AU
        significant_mask = saliency_scores > self.threshold  # (B, 17)

        # 快速路径：不显著的AU使用模板
        fast_path_mask = ~significant_mask

        return {
            'significant_mask': significant_mask,
            'saliency_scores': saliency_scores,
            'fast_path_mask': fast_path_mask,
        }

    def _create_template_fields(self):
        """创建模板运动场（用于不显著的AU）。"""
        # 每个AU的基本运动方向模板
        templates = {}

        for au, info in AU_INFO.items():
            idx = AU_INDEX[au]
            region = info['region']
            region_info = AU_REGIONS[region]

            # 创建简单的平移模板
            templates[idx] = {
                'region': region_info,
                'direction': self._get_default_direction(au),
            }

        return templates

    def _get_default_direction(self, au):
        """获取AU的默认运动方向。"""
        directions = {
            'AU1': (0, -5),   # 内眉上扬
            'AU2': (0, -5),   # 外眉上扬
            'AU4': (0, 3),    # 眉降
            'AU5': (0, -3),   # 眼睑上扬
            'AU6': (0, -2),   # 颧骨上扬
            'AU12': (3, -3),  # 嘴角外上扬
            'AU14': (-2, 0),  # 嘴角收紧
            'AU17': (0, -3),  # 下颏上扬
        }
        return directions.get(au, (0, 0))


# =============================================================================
# V2层：AU交互矩阵
# =============================================================================

class V2AUInteraction(nn.Module):
    """
    V2层：AU协同/对抗建模。

    基于FACS文献设计AU之间的相互作用：
      - 协同：某些AU组合会增强效果（如AU6+AU12=真诚微笑）
      - 对抗：某些AU组合会互相抑制（如AU12+AU14=压抑微笑）
    """

    def __init__(self, num_au=17):
        super().__init__()

        self.num_au = num_au

        # 协同矩阵（可学习，但有预定义初始值）
        self.synergy_matrix = nn.Parameter(self._init_synergy_matrix())

        # 对抗矩阵（可学习，但有预定义初始值）
        self.antagonism_matrix = nn.Parameter(self._init_antagonism_matrix())

        # 交互强度调节（可学习）
        self.interaction_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, au_activation, significant_mask=None):
        """
        Args:
            au_activation (torch.Tensor): AU激活，shape (B, 17)
            significant_mask (torch.Tensor): 显著性掩码，可选

        Returns:
            effective_au (torch.Tensor): 交互后的有效AU激活
        """
        # 计算协同效果
        synergy_effect = torch.matmul(au_activation, self.synergy_matrix)

        # 计算对抗效果
        antagonism_effect = torch.matmul(au_activation, self.antagonism_matrix)

        # 组合
        effective_au = au_activation + self.interaction_scale * (synergy_effect - antagonism_effect)

        # 裁剪到 [0, 1]
        effective_au = torch.clamp(effective_au, 0, 1)

        # 如果有显著性掩码，只对显著AU应用交互
        if significant_mask is not None:
            effective_au = effective_au * significant_mask.float() + \
                          au_activation * (~significant_mask).float()

        return effective_au

    def _init_synergy_matrix(self):
        """
        初始化协同矩阵。

        基于FACS情感映射：
          - Happiness: AU6 + AU12协同
          - Surprise: AU1 + AU2 + AU5协同
        """
        matrix = torch.zeros(self.num_au, self.num_au)

        # AU6 + AU12 协同（真诚微笑）
        matrix[AU_INDEX['AU6'], AU_INDEX['AU12']] = 0.3
        matrix[AU_INDEX['AU12'], AU_INDEX['AU6']] = 0.3

        # AU1 + AU2 协同（惊讶眉）
        matrix[AU_INDEX['AU1'], AU_INDEX['AU2']] = 0.2
        matrix[AU_INDEX['AU2'], AU_INDEX['AU1']] = 0.2

        # AU5 + AU25 协同（惊讶张眼张嘴）
        matrix[AU_INDEX['AU5'], AU_INDEX['AU25']] = 0.15

        # AU4 + AU9 协同（厌恶皱眉皱鼻）
        matrix[AU_INDEX['AU4'], AU_INDEX['AU9']] = 0.2

        return matrix

    def _init_antagonism_matrix(self):
        """
        初始化对抗矩阵。

        基于FACS：
          - AU12 + AU14 对抗（嘴角上扬 vs 嘴角收紧）
          - AU4 + AU2 对抗（眉降 vs 眉扬）
        """
        matrix = torch.zeros(self.num_au, self.num_au)

        # AU12 + AU14 对抗（压抑微笑）
        matrix[AU_INDEX['AU12'], AU_INDEX['AU14']] = 0.4
        matrix[AU_INDEX['AU14'], AU_INDEX['AU12']] = 0.3

        # AU4 + AU2 对抗（悲伤眉 vs 惊讶眉）
        matrix[AU_INDEX['AU4'], AU_INDEX['AU2']] = 0.3
        matrix[AU_INDEX['AU2'], AU_INDEX['AU4']] = 0.25

        # AU15 + AU12 对抗（嘴角下垂 vs 嘴角上扬）
        matrix[AU_INDEX['AU15'], AU_INDEX['AU12']] = 0.35

        return matrix


# =============================================================================
# V3层：AU时间动力学
# =============================================================================

class V3AUTemporalDynamics(nn.Module):
    """
    V3层：基于肌肉生理学的AU时间动力学。

    不同AU有不同的时间特性：
      - 快肌纤维（眼、眉）：快速收缩，适合微表情的快速起始
      - 慢肌纤维（下巴）：缓慢收缩，适合持续表情
      - 混合肌（嘴角）：中等速度

    借鉴神经科学中的肌肉纤维类型差异。
    """

    def __init__(self, num_au=17, num_frames=16):
        super().__init__()

        self.num_au = num_au
        self.num_frames = num_frames

        # AU肌肉时间参数（可学习，有预定义初始值）
        self.au_time_params = nn.Parameter(self._init_time_params())

    def forward(self, au_activation, num_frames=None):
        """
        Args:
            au_activation (torch.Tensor): AU激活，shape (B, 17)
            num_frames (int): 输出帧数

        Returns:
            au_temporal_field (torch.Tensor): AU时间场，shape (B, 17, T)
        """
        T = num_frames or self.num_frames
        B = au_activation.shape[0]

        # 为每个AU生成时间曲线
        temporal_curves = []

        for au_idx in range(self.num_au):
            # 获取该AU的时间参数
            onset_speed = self.au_time_params[au_idx, 0]
            apex_duration = self.au_time_params[au_idx, 1]
            decay_speed = self.au_time_params[au_idx, 2]

            # 生成曲线
            curve = self._generate_single_curve(T, onset_speed, apex_duration, decay_speed)
            temporal_curves.append(curve)

        # 组合
        temporal_curves = torch.stack(temporal_curves, dim=0)  # (17, T)
        temporal_curves = temporal_curves.unsqueeze(0).expand(B, -1, -1)  # (B, 17, T)

        # 应用AU激活强度
        au_temporal_field = au_activation.unsqueeze(-1) * temporal_curves  # (B, 17, T)

        return au_temporal_field

    def _generate_single_curve(self, T, onset_speed, apex_duration, decay_speed):
        """
        生成单个AU的时间曲线。

        onset_speed: 起始速度（越大越快）
        apex_duration: apex持续时间比例
        decay_speed: 衰减速度（越大越快）
        """
        curve = torch.zeros(T)

        # 计算各阶段边界
        onset_end = int(T * (0.3 - 0.1 * onset_speed))  # onset速度影响onset时长
        apex_end = int(T * (onset_end / T + apex_duration))

        # Onset阶段
        for t in range(onset_end):
            progress = t / onset_end
            # 使用onset_speed调节曲线形状
            curve[t] = progress ** (1 - onset_speed * 0.5)

        # Apex阶段
        curve[onset_end:apex_end] = 1.0

        # Decay阶段
        for t in range(apex_end, T):
            progress = (t - apex_end) / (T - apex_end)
            curve[t] = 1 - progress ** (1 + decay_speed * 0.5)

        return curve

    def _init_time_params(self):
        """
        初始化AU时间参数。

        参数：[onset_speed, apex_duration, decay_speed]
        """
        params = torch.zeros(self.num_au, 3)

        # 快肌（眼、眉）：快速起始，快速衰减
        fast_au = ['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU9']
        for au in fast_au:
            idx = AU_INDEX[au]
            params[idx] = torch.tensor([0.8, 0.15, 0.7])  # 快起始，短apex，快衰减

        # 混合肌（嘴角）：中等速度
        mixed_au = ['AU6', 'AU10', 'AU12', 'AU20', 'AU25']
        for au in mixed_au:
            idx = AU_INDEX[au]
            params[idx] = torch.tensor([0.5, 0.2, 0.5])

        # 慢肌（下巴）：缓慢运动
        slow_au = ['AU14', 'AU15', 'AU17', 'AU23', 'AU24', 'AU26']
        for au in slow_au:
            idx = AU_INDEX[au]
            params[idx] = torch.tensor([0.3, 0.25, 0.4])  # 慢起始，长apex，慢衰减

        return params


# =============================================================================
# V4层：局部运动场生成
# =============================================================================

class V4LocalMotionField(nn.Module):
    """
    V4层：AU→局部运动场。

    每个AU生成特定面部区域的局部warping场，
    而不是全局关键点位移。

    借鉴Civis的视觉金字塔概念：
      - Level 0: 精细运动场（高分辨率）
      - Level 1: 中等运动场
      - Level 2: 粗运动场（低分辨率）
    """

    def __init__(self, num_au=17, image_size=224, pyramid_levels=3):
        super().__init__()

        self.num_au = num_au
        self.image_size = image_size
        self.pyramid_levels = pyramid_levels

        # AU区域参数
        self.au_region_params = self._create_region_params()

        # 运动场生成器（每个AU一个小网络）
        self.field_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 + pyramid_levels, 32),  # AU强度 + 时间因子
                nn.ReLU(),
                nn.Linear(32, 2 * self._get_field_size()),
            )
            for _ in range(num_au)
        ])

    def forward(self, au_temporal_field, time_idx=None):
        """
        Args:
            au_temporal_field (torch.Tensor): AU时间场，shape (B, 17, T)
            time_idx (int): 当前帧索引

        Returns:
            motion_field (torch.Tensor): 运动场，shape (B, 2, H, W)
        """
        B = au_temporal_field.shape[0]

        if time_idx is None:
            T = au_temporal_field.shape[2]
            time_idx = T // 2  # 默认用apex帧

        # 获取当前帧的AU激活
        au_frame = au_temporal_field[:, :, time_idx]  # (B, 17)

        # 初始化运动场
        motion_field = torch.zeros(B, 2, self.image_size, self.image_size)

        # 为每个AU生成局部运动场
        for au_idx in range(self.num_au):
            # AU强度
            au_intensity = au_frame[:, au_idx].unsqueeze(-1)  # (B, 1)

            # 金字塔因子
            pyramid_factors = torch.ones(B, self.pyramid_levels)

            # 生成运动场参数
            field_params = self.field_generators[au_idx](
                torch.cat([au_intensity, pyramid_factors], dim=-1)
            )  # (B, 2*field_size)

            # 转换为局部运动场
            local_field = self._create_local_field(au_idx, field_params, au_intensity)

            # 添加到全局运动场（区域加权）
            motion_field = motion_field + local_field

        return motion_field

    def _create_local_field(self, au_idx, field_params, intensity):
        """创建局部运动场。"""
        B = intensity.shape[0]

        # 获取AU区域
        region = AU_INFO[list(AU_INDEX.keys())[au_idx]]['region']
        region_info = AU_REGIONS[region]

        center = region_info['center']
        radius = region_info['radius']

        # 创建权重mask（Gaussian）
        mask = self._create_gaussian_mask(center, radius)

        # 解析运动参数
        dx = field_params[:, 0] * intensity.squeeze(-1)  # (B,)
        dy = field_params[:, 1] * intensity.squeeze(-1)

        # 创建运动场
        field = torch.zeros(B, 2, self.image_size, self.image_size)

        for b in range(B):
            field[b, 0, :, :] = dx[b] * mask  # x方向运动
            field[b, 1, :, :] = dy[b] * mask  # y方向运动

        return field

    def _create_gaussian_mask(self, center, radius):
        """创建Gaussian权重mask。"""
        H, W = self.image_size, self.image_size
        cx, cy = center

        # 创建坐标网格
        y = torch.arange(H).float()
        x = torch.arange(W).float()

        # Gaussian
        mask = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * radius**2))

        return mask

    def _get_field_size(self):
        """获取运动场大小。"""
        return self.image_size // 4  # 使用1/4分辨率

    def _create_region_params(self):
        """创建AU区域参数。"""
        params = {}
        for au, info in AU_INFO.items():
            idx = AU_INDEX[au]
            region = info['region']
            params[idx] = AU_REGIONS[region]
        return params


# =============================================================================
# IT层：运动场融合与生成
# =============================================================================

class ITMotionFusion(nn.Module):
    """
    IT层：运动场融合与最终生成。

    融合所有AU的局部运动场：
      1. 处理区域重叠冲突
      2. 平滑过渡
      3. Warping生成
    """

    def __init__(self, image_size=224):
        super().__init__()

        self.image_size = image_size

        # 冲突解决网络
        self.conflict_resolver = nn.Sequential(
            nn.Conv2d(2 * 17, 32, 3, 1, 1),  # 输入：所有AU的运动场
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, 1),  # 输出：融合后的运动场
        )

    def forward(self, au_motion_fields, neutral_face):
        """
        Args:
            au_motion_fields (list): 各AU的运动场列表
            neutral_face (torch.Tensor): 中性脸，shape (B, C, H, W)

        Returns:
            generated_frame (torch.Tensor): 生成的帧
        """
        B, C, H, W = neutral_face.shape

        # 合并所有AU运动场
        all_fields = torch.stack(au_motion_fields, dim=1)  # (B, 17, 2, H, W)
        all_fields = all_fields.view(B, 2 * 17, H, W)

        # 融合运动场
        fused_field = self.conflict_resolver(all_fields)  # (B, 2, H, W)

        # Warping
        generated = self._warp_image(neutral_face, fused_field)

        return generated

    def _warp_image(self, image, motion_field):
        """使用运动场warp图像。"""
        B, C, H, W = image.shape

        # 创建采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W)
        )
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)
        grid = grid.expand(B, -1, -1, -1).to(image.device)

        # 添加运动场（归一化）
        motion_normalized = motion_field.permute(0, 2, 3, 1) / (H / 2)
        sampling_grid = grid + motion_normalized

        # Warp
        warped = F.grid_sample(image, sampling_grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)

        return warped


# =============================================================================
# 完整的Censor-G v2架构
# =============================================================================

class CensorGv2(nn.Module):
    """
    Censor-G v2: 视觉中枢启发的微表情生成。

    分层处理：
      V1 → AU显著性筛选
      V2 → AU交互矩阵
      V3 → AU时间动力学
      V4 → 局部运动场生成
      IT → 运动场融合与生成
    """

    def __init__(
        self,
        num_au=17,
        num_frames=16,
        image_size=224,
        threshold=0.1,
    ):
        super().__init__()

        self.num_au = num_au
        self.num_frames = num_frames
        self.image_size = image_size

        # V1层：AU显著性筛选
        self.v1_saliency = V1AUSaliency(num_au, threshold)

        # V2层：AU交互
        self.v2_interaction = V2AUInteraction(num_au)

        # V3层：AU时间动力学
        self.v3_temporal = V3AUTemporalDynamics(num_au, num_frames)

        # V4层：局部运动场
        self.v4_motion_field = V4LocalMotionField(num_au, image_size)

        # IT层：融合与生成
        self.it_fusion = ITMotionFusion(image_size)

    def forward(self, neutral_face, au_activation, emotion_class=None):
        """
        生成微表情视频。

        Args:
            neutral_face (torch.Tensor): 中性脸，shape (B, C, H, W)
            au_activation (torch.Tensor): AU激活，shape (B, 17)
            emotion_class (torch.Tensor): 情感类别（可选）

        Returns:
            generated_video (torch.Tensor): 生成的视频，shape (B, C, T, H, W)
        """
        B, C, H, W = neutral_face.shape
        T = self.num_frames

        # V1: 显著性筛选
        v1_output = self.v1_saliency(au_activation)
        significant_mask = v1_output['significant_mask']

        # V2: AU交互
        effective_au = self.v2_interaction(au_activation, significant_mask)

        # V3: 时间动力学
        au_temporal_field = self.v3_temporal(effective_au, T)  # (B, 17, T)

        # 生成每一帧
        generated_frames = []

        for t in range(T):
            # V4: 局部运动场
            motion_field = self.v4_motion_field(au_temporal_field, time_idx=t)

            # IT: 融合与warping
            frame = self.it_fusion._warp_image(neutral_face, motion_field)
            generated_frames.append(frame)

        # 组合为视频
        generated_video = torch.stack(generated_frames, dim=2)  # (B, C, T, H, W)

        return generated_video

    def generate_with_intensity(self, neutral_face, emotion_class, intensity):
        """
        根据情感和强度生成AU配置并生成视频。

        Args:
            neutral_face (torch.Tensor): 中性脸
            emotion_class (torch.Tensor): 情感类别
            intensity (torch.Tensor): 强度参数

        Returns:
            generated_video (torch.Tensor): 生成的视频
        """
        # 情感→AU映射
        au_config = self._emotion_to_au(emotion_class)

        # 应用强度
        au_activation = au_config * intensity.unsqueeze(-1)

        # 生成
        return self.forward(neutral_face, au_activation, emotion_class)

    def _emotion_to_au(self, emotion_class):
        """情感类别→默认AU配置。"""
        B = emotion_class.shape[0]
        au_config = torch.zeros(B, self.num_au)

        for b in range(B):
            emotion = emotion_class[b].item()

            if emotion == 0:  # Happiness
                au_config[b, AU_INDEX['AU6']] = 0.6
                au_config[b, AU_INDEX['AU12']] = 0.8
                au_config[b, AU_INDEX['AU25']] = 0.2

            elif emotion == 1:  # Surprise
                au_config[b, AU_INDEX['AU1']] = 0.7
                au_config[b, AU_INDEX['AU2']] = 0.7
                au_config[b, AU_INDEX['AU5']] = 0.8
                au_config[b, AU_INDEX['AU25']] = 0.5

            elif emotion == 2:  # Disgust
                au_config[b, AU_INDEX['AU4']] = 0.5
                au_config[b, AU_INDEX['AU9']] = 0.7
                au_config[b, AU_INDEX['AU10']] = 0.4
                au_config[b, AU_INDEX['AU17']] = 0.3

            elif emotion == 3:  # Repression
                au_config[b, AU_INDEX['AU14']] = 0.6
                au_config[b, AU_INDEX['AU17']] = 0.4
                au_config[b, AU_INDEX['AU4']] = 0.3

        return au_config.to(emotion_class.device)


# =============================================================================
# Demo
# =============================================================================

def demo_censor_g_v2():
    """Demo Censor-G v2。"""
    print("\n" + "="*60)
    print("Censor-G v2 Demo: Visual Cortex-Inspired ME Generation")
    print("="*60)

    # 创建模型
    model = CensorGv2(num_au=17, num_frames=16, image_size=224)

    # 创建输入
    B, C, H, W = 2, 3, 224, 224
    neutral_face = torch.randn(B, C, H, W)
    au_activation = torch.rand(B, 17)
    emotion_class = torch.tensor([0, 1])  # Happiness, Surprise

    # 测试生成
    print("\n[Test 1] Direct AU generation")
    generated = model(neutral_face, au_activation)
    print(f"  Generated video shape: {generated.shape}")

    print("\n[Test 2] Emotion + Intensity generation")
    intensity = torch.tensor([0.8, 0.6])
    generated = model.generate_with_intensity(neutral_face, emotion_class, intensity)
    print(f"  Generated video shape: {generated.shape}")

    # 测试各层
    print("\n[Test 3] Layer-wise outputs")
    v1_output = model.v1_saliency(au_activation)
    print(f"  V1: significant_mask = {v1_output['significant_mask'][0]}")

    effective_au = model.v2_interaction(au_activation)
    print(f"  V2: effective_au difference = {(effective_au - au_activation).abs().mean():.4f}")

    temporal_field = model.v3_temporal(effective_au)
    print(f"  V3: temporal_field shape = {temporal_field.shape}")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_censor_g_v2()