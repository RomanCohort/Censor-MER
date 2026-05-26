# =============================================================================
# Censor-G: Real Image Generation Network
# =============================================================================
# 实现真正的微表情图像生成网络
#
# 架构设计（借鉴FOMM）：
#   1. Keypoint Detector: 检测面部关键点（68点）
#   2. Motion Estimator: AU激活 → 关键点位移 → 密集运动场
#   3. Image Generator: 使用运动场warp中性脸生成微表情帧
#   4. Discriminator: GAN判别器（可选）
#
# 与FOMM的区别：
#   - FOMM: 视频→关键点检测→运动估计→生成
#   - Censor-G: AU激活→SNN处理→关键点位移→运动场→生成
#
# 关键创新：
#   - AU直接控制运动（不需要从驱动视频提取）
#   - SNN机制调节运动的时间和强度
#   - 神经科学启发的运动模式
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Part 1: AU到关键点位移的映射
# =============================================================================
# 每个AU激活对应特定面部区域的运动
# 参考：FACS (Facial Action Coding System) 文献
# =============================================================================

# AU到关键点索引的映射（基于68点面部标注）
AU_KEYPOINT_MAPPING = {
    # AU1: Inner Brow Raiser → 眉毛内侧上移
    'AU1': {
        'keypoints': [17, 18, 19],  # 左眉内侧点
        'direction': (0, -5),       # 向上移动
        'scale': 0.8,
    },
    # AU2: Outer Brow Raiser → 眉毛外侧上移
    'AU2': {
        'keypoints': [20, 21, 22],  # 左眉外侧点
        'direction': (0, -4),
        'scale': 0.7,
        'mirror': [26, 27, 28],     # 右眉对应点
    },
    # AU4: Brow Lowerer → 眉毛下移
    'AU4': {
        'keypoints': [17, 18, 19, 20, 21, 22],
        'direction': (0, 3),
        'scale': 0.6,
        'mirror': [26, 27, 28, 29, 30, 31],
    },
    # AU5: Upper Lid Raiser → 上眼睑上移
    'AU5': {
        'keypoints': [37, 38, 43, 44],  # 上眼睑点
        'direction': (0, -2),
        'scale': 0.9,
    },
    # AU6: Cheek Raiser → 脸颊上移（笑眼）
    'AU6': {
        'keypoints': [33, 34, 35],  # 左脸颊
        'direction': (0, -3),
        'scale': 0.5,
        'mirror': [43, 44, 45],
    },
    # AU12: Lip Corner Puller → 嘴角上移（微笑）
    'AU12': {
        'keypoints': [48, 49],  # 左嘴角
        'direction': (3, -4),
        'scale': 0.85,
        'mirror': [54, 55],
    },
    # AU14: Dimpler → 嘴角内侧收缩（压抑）
    'AU14': {
        'keypoints': [48, 49, 54, 55],
        'direction': (-1.5, 0),
        'scale': 0.6,
    },
    # AU15: Lip Corner Depressor → 嘴角下移
    'AU15': {
        'keypoints': [48, 49, 54, 55],
        'direction': (0, 3),
        'scale': 0.5,
    },
    # AU17: Chin Raiser → 下唇上移
    'AU17': {
        'keypoints': [57, 58, 59, 60, 61],  # 下唇区域
        'direction': (0, -3),
        'scale': 0.4,
    },
    # AU25: Lips Part → 嘴唇张开
    'AU25': {
        'keypoints': [62, 63, 64, 65],  # 上下唇接触点
        'direction': (0, 5),  # 下唇下移
        'scale': 0.7,
    },
    # AU26: Jaw Drop → 下颌下移
    'AU26': {
        'keypoints': [8, 9, 10, 11, 12, 13, 14, 15, 16],  # 下颌线
        'direction': (0, 8),
        'scale': 0.5,
    },
}

# AU索引映射
AU_INDEX = {
    'AU1': 0, 'AU2': 1, 'AU4': 2, 'AU5': 3, 'AU6': 4, 'AU7': 5,
    'AU9': 6, 'AU10': 7, 'AU12': 8, 'AU14': 9, 'AU15': 10, 'AU17': 11,
    'AU20': 12, 'AU23': 13, 'AU24': 14, 'AU25': 15, 'AU26': 16
}


# =============================================================================
# Part 2: 关键点检测器
# =============================================================================

class KeypointDetector(nn.Module):
    """
    面部关键点检测器

    使用预训练模型或简单CNN检测68个面部关键点
    """

    def __init__(self, num_keypoints=68, image_size=224):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.image_size = image_size

        # 简化的关键点检测网络
        # 实际使用时可以替换为预训练模型（如dlib或MediaPipe）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # 关键点预测头
        self.keypoint_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 2),  # (x, y) for each keypoint
        )

        # 初始化为中性脸的标准关键点位置
        self.register_buffer('neutral_keypoints', self._create_neutral_keypoints())

    def _create_neutral_keypoints(self):
        """创建中性脸的标准关键点位置"""
        # 基于68点面部标注的标准位置
        keypoints = []

        # 下颌 (0-16)
        keypoints.extend([
            [0.10, 0.50], [0.12, 0.55], [0.15, 0.60], [0.18, 0.65],
            [0.22, 0.70], [0.27, 0.75], [0.32, 0.78], [0.38, 0.80],
            [0.50, 0.82],  # 下颌中点
            [0.62, 0.80], [0.68, 0.78], [0.73, 0.75], [0.78, 0.70],
            [0.82, 0.65], [0.85, 0.60], [0.88, 0.55], [0.90, 0.50],
        ])  # 17 points

        # 眉鼻连接 (17-27)
        keypoints.extend([
            [0.15, 0.25], [0.18, 0.23], [0.21, 0.22], [0.25, 0.23], [0.28, 0.25],  # 左眉 17-21
            [0.72, 0.25], [0.75, 0.23], [0.79, 0.22], [0.82, 0.23], [0.85, 0.25],  # 右眉 22-26
            [0.50, 0.30],  # 鼻根 27
        ])  # 11 points

        # 鼻子 (28-35)
        keypoints.extend([
            [0.50, 0.35], [0.50, 0.40], [0.50, 0.45],  # 鼻梁
            [0.45, 0.42], [0.50, 0.43], [0.55, 0.42],  # 鼻翼
            [0.42, 0.45], [0.58, 0.45],  # 鼻翼外侧
        ])  # 8 points

        # 眼睛 (36-47)
        keypoints.extend([
            [0.20, 0.35], [0.22, 0.33], [0.24, 0.33], [0.26, 0.35],
            [0.24, 0.37], [0.22, 0.37],  # 左眼 36-41
            [0.74, 0.35], [0.76, 0.33], [0.78, 0.33], [0.80, 0.35],
            [0.78, 0.37], [0.76, 0.37],  # 右眼 42-47
        ])  # 12 points

        # 嘴巴外轮廓 (48-59)
        keypoints.extend([
            [0.35, 0.60], [0.38, 0.58], [0.42, 0.57], [0.46, 0.57],
            [0.50, 0.58], [0.54, 0.57], [0.58, 0.57], [0.62, 0.58],
            [0.65, 0.60], [0.65, 0.62],  # 左嘴角到右嘴角
        ])  # 10 points

        # 嘴巴内轮廓 (60-67)
        keypoints.extend([
            [0.62, 0.65], [0.58, 0.66], [0.54, 0.67], [0.50, 0.68],
            [0.46, 0.67], [0.42, 0.66], [0.38, 0.65], [0.35, 0.62],
        ])  # 8 points

        # 总计: 17 + 11 + 8 + 12 + 10 + 8 = 66 points
        # 需要补充2个点（通常是眼睛中心的额外标记）
        keypoints.extend([
            [0.23, 0.35], [0.77, 0.35],  # 左右眼睛中心额外点
        ])  # 2 points → total 68

        # 转换为tensor
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

        # 验证数量
        assert keypoints_tensor.shape[0] == 68, f"Expected 68 keypoints, got {keypoints_tensor.shape[0]}"

        # 缩放到图像尺寸
        keypoints_tensor = keypoints_tensor * self.image_size

        return keypoints_tensor

    def forward(self, image):
        """
        检测面部关键点

        Args:
            image: (B, C, H, W) 输入图像

        Returns:
            keypoints: (B, 68, 2) 关键点坐标 (x, y)
        """
        B = image.shape[0]

        # 编码
        features = self.encoder(image)

        # 预测关键点偏移
        offset = self.keypoint_head(features)  # (B, 136)
        offset = offset.view(B, 68, 2)

        # 基础关键点 + 偏移
        keypoints = self.neutral_keypoints.unsqueeze(0) + offset * 10

        # 限制在图像范围内
        keypoints = torch.clamp(keypoints, 0, self.image_size)

        return keypoints

    def get_neutral_keypoints(self):
        """获取中性脸关键点"""
        return self.neutral_keypoints.clone()


# =============================================================================
# Part 3: 运动场估计器
# =============================================================================

class MotionFieldEstimator(nn.Module):
    """
    运动场估计器

    从AU激活和关键点位置估计密集运动场：
      1. AU → 关键点位移
      2. 关键点位移 → 局部运动（使用泰勒展开）
      3. 局部运动 → 密集运动场（稀疏到密集插值）
    """

    def __init__(self, num_keypoints=68, num_au=17, image_size=224, skip_smoothing=False):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_au = num_au
        self.image_size = image_size

        # 运动放大系数
        # PHASE 2 FIX: 从50降到5，避免过度放大与损失函数冲突
        # 原始值50过大，与损失函数的0.3限制形成矛盾
        self.motion_scale = 5.0

        # 是否跳过平滑网络（直接用关键点运动）
        self.skip_smoothing = skip_smoothing

        # 关键点检测器
        self.keypoint_detector = KeypointDetector(num_keypoints, image_size)

        # AU到关键点位移的权重（可学习）
        # 每个AU影响多个关键点
        self.au_to_keypoint_weights = nn.Parameter(
            self._init_au_to_keypoint_weights()
        )

        # 稀疏到密集插值网络
        self.sparse_to_dense = nn.Sequential(
            nn.Conv2d(2, 32, 7, 1, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 7, 1, 3),
        )

    def _init_au_to_keypoint_weights(self):
        """初始化AU到关键点位移的权重矩阵"""
        # 权重矩阵: (num_au, num_keypoints, 2)
        weights = torch.zeros(self.num_au, self.num_keypoints, 2)

        # 基于AU_KEYPOINT_MAPPING初始化
        # 重要：乘以motion_scale放大运动
        for au_name, mapping in AU_KEYPOINT_MAPPING.items():
            au_idx = AU_INDEX[au_name]
            kp_indices = mapping['keypoints']
            direction = mapping['direction']
            scale = mapping['scale'] * self.motion_scale  # ← 放大！

            for kp_idx in kp_indices:
                if kp_idx < self.num_keypoints:
                    weights[au_idx, kp_idx, 0] = direction[0] * scale
                    weights[au_idx, kp_idx, 1] = direction[1] * scale

            # 镜像点（右侧）
            if 'mirror' in mapping:
                for kp_idx in mapping['mirror']:
                    if kp_idx < self.num_keypoints:
                        weights[au_idx, kp_idx, 0] = -direction[0] * scale  # 水平镜像
                        weights[au_idx, kp_idx, 1] = direction[1] * scale

        return weights

    def forward(self, neutral_face, au_activation, keypoint_displacement=None):
        """
        估计运动场

        Args:
            neutral_face: (B, C, H, W) 中性脸图像
            au_activation: (B, 17) AU激活强度
            keypoint_displacement: (B, 68, 2) 可选的直接关键点位移

        Returns:
            motion_field: (B, 2, H, W) 密集运动场 (dx, dy)
            keypoints_displaced: (B, 68, 2) 位移后的关键点
        """
        B = au_activation.shape[0]

        # 获取中性脸关键点
        neutral_keypoints = self.keypoint_detector.get_neutral_keypoints()
        neutral_keypoints = neutral_keypoints.unsqueeze(0).expand(B, -1, -1)

        # 计算关键点位移
        if keypoint_displacement is None:
            # AU → 关键点位移
            # au_activation: (B, 17)
            # au_to_keypoint_weights: (17, 68, 2)

            # 扩展维度
            au_exp = au_activation.unsqueeze(-1).unsqueeze(-1)  # (B, 17, 1, 1)
            weights_exp = self.au_to_keypoint_weights.unsqueeze(0)  # (1, 17, 68, 2)

            # 位移计算
            displacement = (au_exp * weights_exp).sum(dim=1)  # (B, 68, 2)

        else:
            displacement = keypoint_displacement

        # 位移后的关键点
        keypoints_displaced = neutral_keypoints + displacement

        # 稀疏关键点位移 → 密集运动场
        motion_field = self._sparse_to_dense_motion(
            neutral_keypoints, keypoints_displaced
        )

        return motion_field, keypoints_displaced

    def _sparse_to_dense_motion(self, source_kp, target_kp):
        """
        稀疏关键点位移 → 密集运动场

        使用基于距离的插值：
          motion(x,y) = Σ displacement_i * weight_i(x,y)
          weight_i(x,y) = exp(-||p - kp_i||² / σ²)
        """
        B, N, _ = source_kp.shape
        H = W = self.image_size

        # 创建坐标网格
        y_coords = torch.linspace(0, H-1, H, device=source_kp.device)
        x_coords = torch.linspace(0, W-1, W, device=source_kp.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 扩展为 (B, H, W)
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        # 计算位移
        displacement = target_kp - source_kp  # (B, N, 2)

        # 增加sigma扩大影响范围（从20到40）
        sigma = 40.0  # 影响范围

        motion_field_x = torch.zeros(B, H, W, device=source_kp.device)
        motion_field_y = torch.zeros(B, H, W, device=source_kp.device)

        for i in range(N):
            # 关键点位置
            kp_x = source_kp[:, i, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            kp_y = source_kp[:, i, 1].unsqueeze(-1).unsqueeze(-1)

            # 距离权重
            dist_sq = (grid_x - kp_x)**2 + (grid_y - kp_y)**2
            weight = torch.exp(-dist_sq / (sigma**2))

            # 位移贡献
            motion_field_x += displacement[:, i, 0].unsqueeze(-1).unsqueeze(-1) * weight
            motion_field_y += displacement[:, i, 1].unsqueeze(-1).unsqueeze(-1) * weight

        # 组合为运动场
        motion_field = torch.stack([motion_field_x, motion_field_y], dim=1)  # (B, 2, H, W)

        # 通过卷积网络平滑（可选）
        if self.skip_smoothing:
            # 直接使用关键点运动，不经过压缩网络
            motion_field = motion_field
        else:
            smoothed = self.sparse_to_dense(motion_field)

            # 重要：不要让卷积网络压缩运动幅度
            original_mag = motion_field.abs().mean()
            smoothed_mag = smoothed.abs().mean()

            if smoothed_mag > 0.001 and original_mag > smoothed_mag:
                # 如果smoothed被压缩了，放大回去
                scale_factor = original_mag / (smoothed_mag + 1e-6)
                motion_field = smoothed * scale_factor.clamp(max=100)
            else:
                motion_field = smoothed

        return motion_field


# =============================================================================
# Part 4: 图像生成器
# =============================================================================

class ImageGenerator(nn.Module):
    """
    图像生成器

    使用运动场warp中性脸生成微表情帧
    """

    def __init__(self, image_size=224):
        super().__init__()

        self.image_size = image_size

        # 可选的纹理增强网络
        self.texture_enhancer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, neutral_face, motion_field, enhance_texture=False):
        """
        使用运动场生成图像

        Args:
            neutral_face: (B, C, H, W) 中性脸
            motion_field: (B, 2, H, W) 运动场
            enhance_texture: 是否增强纹理细节

        Returns:
            generated: (B, C, H, W) 生成的微表情帧
        """
        # 使用grid_sample进行运动warp
        generated = self._warp_image(neutral_face, motion_field)

        # 可选的纹理增强
        if enhance_texture:
            generated = generated + self.texture_enhancer(generated)

        return generated

    def _warp_image(self, image, motion_field):
        """
        使用运动场warp图像

        Args:
            image: (B, C, H, W)
            motion_field: (B, 2, H, W) - dx, dy

        Returns:
            warped: (B, C, H, W)
        """
        B, C, H, W = image.shape

        # 创建标准网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=2)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # 运动场标准化（像素坐标 → -1到1）
        motion_normalized = motion_field.permute(0, 2, 3, 1)  # (B, H, W, 2)
        motion_normalized[:, :, :, 0] = motion_normalized[:, :, :, 0] / (W / 2)
        motion_normalized[:, :, :, 1] = motion_normalized[:, :, :, 1] / (H / 2)

        # 应用运动
        sampling_grid = grid + motion_normalized
        sampling_grid = torch.clamp(sampling_grid, -1, 1)

        # Grid sample
        warped = F.grid_sample(
            image, sampling_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        return warped


# =============================================================================
# Part 5: 完整生成器
# =============================================================================

class CensorGGenerator(nn.Module):
    """
    Censor-G完整图像生成器

    整合：
      - SNN机制（AU处理）
      - 运动场估计
      - 图像生成

    输入：
      - neutral_face: 中性脸图像
      - au_activation: AU激活强度

    输出：
      - generated_video: 生成的微表情视频序列
    """

    def __init__(self, num_au=17, num_keypoints=68, num_frames=16, image_size=224, skip_smoothing=True):
        super().__init__()

        self.num_au = num_au
        self.num_frames = num_frames
        self.image_size = image_size

        # 运动场估计器（默认跳过平滑网络）
        self.motion_estimator = MotionFieldEstimator(
            num_keypoints=num_keypoints,
            num_au=num_au,
            image_size=image_size,
            skip_smoothing=skip_smoothing
        )

        # 图像生成器
        self.image_generator = ImageGenerator(image_size)

        # SNN时间处理（简化版）
        # 用于调节AU强度随时间变化
        self.time_modulator = nn.Sequential(
            nn.Linear(num_au + 1, 64),  # AU + time index
            nn.ReLU(),
            nn.Linear(64, num_au),
            nn.Sigmoid()
        )

        # 时间曲线模板
        self.register_buffer('temporal_curve', self._create_temporal_curve(num_frames))

    def _create_temporal_curve(self, num_frames):
        """创建微表情时间曲线模板"""
        T = num_frames

        # Onset阶段（0-30%）
        onset_end = int(T * 0.3)

        # Apex阶段（30-50%）
        apex_end = int(T * 0.5)

        # Offset阶段（50-100%）

        curve = torch.zeros(T)

        # Onset: 上升
        for t in range(onset_end):
            curve[t] = t / onset_end

        # Apex: 峰值保持
        for t in range(onset_end, apex_end):
            curve[t] = 1.0

        # Offset: 下降
        for t in range(apex_end, T):
            curve[t] = 1.0 - (t - apex_end) / (T - apex_end)

        return curve

    def forward(self, neutral_face, au_activation, emotion_class=None):
        """
        生成微表情视频序列

        Args:
            neutral_face: (B, C, H, W) 中性脸
            au_activation: (B, 17) AU激活强度
            emotion_class: (B,) 可选的情感类别（用于调整时间曲线）

        Returns:
            generated_video: (B, C, T, H, W) 生成的视频
            motion_fields: List[Tensor] 各帧的运动场
        """
        B = neutral_face.shape[0]
        T = self.num_frames

        generated_frames = []
        motion_fields = []

        for t in range(T):
            # 时间调制
            time_intensity = self.temporal_curve[t]

            # AU强度随时间变化
            au_t = au_activation * time_intensity

            # 运动场估计
            motion_field, keypoints = self.motion_estimator(neutral_face, au_t)

            # PHASE 2 FIX: 自适应缩放（而非强制放大）
            # 目标：运动幅度在3-30像素范围内（微表情可见范围）
            TARGET_MIN_MOTION = 3.0  # 从5降到3
            TARGET_MAX_MOTION = 30.0  # 新增上限

            motion_mag = motion_field.abs().mean()

            if motion_mag < TARGET_MIN_MOTION and motion_mag > 0.01:
                # 运动太小，适度放大到最小可见幅度
                scale_factor = TARGET_MIN_MOTION / (motion_mag + 1e-6)
                scale_factor = scale_factor.clamp(max=5.0)  # 从100降到5
                motion_field = motion_field * scale_factor
            elif motion_mag > TARGET_MAX_MOTION:
                # 运动太大，压缩到最大幅度
                scale_factor = TARGET_MAX_MOTION / (motion_mag + 1e-6)
                motion_field = motion_field * scale_factor

            # 图像生成
            frame = self.image_generator(neutral_face, motion_field)

            generated_frames.append(frame)
            motion_fields.append(motion_field)

        # 组合为视频
        generated_video = torch.stack(generated_frames, dim=2)  # (B, C, T, H, W)

        return generated_video, motion_fields


# =============================================================================
# Part 6: GAN判别器（可选）
# =============================================================================

class VideoDiscriminator(nn.Module):
    """
    视频判别器（用于GAN训练）

    判断生成的视频是否真实
    """

    def __init__(self, num_frames=16, image_size=224):
        super().__init__()

        self.num_frames = num_frames

        # 3D卷积网络
        self.discriminator = nn.Sequential(
            nn.Conv3d(3, 32, (4, 4, 4), (2, 2, 2), (1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, (4, 4, 4), (2, 2, 2), (1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, (4, 4, 4), (2, 2, 2), (1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, video):
        """
        判断视频真假

        Args:
            video: (B, C, T, H, W)

        Returns:
            logits: (B, 1)
        """
        return self.discriminator(video)


# =============================================================================
# Part 7: FOMM Baseline（简化版）
# =============================================================================

class FOMMBaseline(nn.Module):
    """
    First Order Motion Model简化版（作为baseline对比）
    """

    def __init__(self, num_keypoints=10, image_size=224):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.image_size = image_size

        # 关键点检测器（简化版）
        self.kp_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.kp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, num_keypoints * 2),
        )

        # 运动生成器
        self.motion_generator = nn.Sequential(
            nn.Linear(num_keypoints * 2 + 17, 256),  # keypoints + AU
            nn.ReLU(),
            nn.Linear(256, 2 * image_size * image_size),
        )

        # 图像生成器
        self.generator = ImageGenerator(image_size)

    def forward(self, neutral_face, au_activation):
        """
        FOMM baseline生成

        Args:
            neutral_face: (B, C, H, W)
            au_activation: (B, 17)

        Returns:
            generated_video: (B, C, T, H, W)
        """
        B = neutral_face.shape[0]
        T = 16  # 默认帧数

        # 检测关键点
        kp_features = self.kp_encoder(neutral_face)
        kp_features_flat = kp_features.flatten(1)
        keypoints = self.kp_head(kp_features_flat)  # (B, num_kp * 2)
        keypoints = keypoints.view(B, self.num_keypoints, 2)

        # 生成视频（简化：每帧使用相同运动）
        frames = []
        for t in range(T):
            # 简化的运动生成
            input_feat = torch.cat([keypoints.flatten(1), au_activation], dim=1)
            motion = self.motion_generator(input_feat)
            motion = motion.view(B, 2, self.image_size, self.image_size)

            # Warp
            frame = self.generator(neutral_face, motion)
            frames.append(frame)

        generated_video = torch.stack(frames, dim=2)

        return generated_video


# =============================================================================
# Demo
# =============================================================================

def demo_real_generation():
    """Demo真实图像生成"""
    print("\n" + "="*60)
    print("Censor-G Real Image Generation Demo")
    print("="*60)

    # 创建模型
    generator = CensorGGenerator(
        num_au=17,
        num_keypoints=68,
        num_frames=16,
        image_size=224
    )

    fomm_baseline = FOMMBaseline(num_keypoints=10, image_size=224)

    print(f"\n[Generator Parameters]: {sum(p.numel() for p in generator.parameters()):,}")

    # 创建输入
    B, C, H, W = 2, 3, 224, 224
    neutral_face = torch.randn(B, C, H, W) * 0.3 + 0.5  # 模拟中性脸

    # AU激活
    au_activation = torch.zeros(B, 17)
    au_activation[0, AU_INDEX['AU6']] = 0.7  # Happiness
    au_activation[0, AU_INDEX['AU12']] = 0.8
    au_activation[1, AU_INDEX['AU1']] = 0.6  # Surprise
    au_activation[1, AU_INDEX['AU2']] = 0.6
    au_activation[1, AU_INDEX['AU5']] = 0.7

    print("\n[Test 1] Censor-G Generation")

    with torch.no_grad():
        generated_video, motion_fields = generator(neutral_face, au_activation)

    print(f"  Generated video shape: {generated_video.shape}")

    # 分析运动场
    motion_magnitude = torch.sqrt(
        motion_fields[0][:, 0]**2 + motion_fields[0][:, 1]**2
    ).mean().item()
    print(f"  Motion magnitude: {motion_magnitude:.4f}")

    # 分析关键点位移
    print("\n[Test 2] Keypoint Displacement Analysis")

    # 获取中性脸关键点
    neutral_kp = generator.motion_estimator.keypoint_detector.get_neutral_keypoints()
    print(f"  Neutral keypoints shape: {neutral_kp.shape}")

    # 模拟AU12激活的关键点位移
    au12_idx = AU_INDEX['AU12']
    print(f"  AU12 affects keypoints: {AU_KEYPOINT_MAPPING['AU12']['keypoints']}")
    print(f"  AU12 direction: {AU_KEYPOINT_MAPPING['AU12']['direction']}")

    print("\n[Test 3] FOMM Baseline Comparison")

    with torch.no_grad():
        fomm_video = fomm_baseline(neutral_face, au_activation)

    print(f"  FOMM video shape: {fomm_video.shape}")

    # 计算生成差异
    diff = (generated_video - fomm_video).abs().mean().item()
    print(f"  Difference from FOMM: {diff:.4f}")

    print("\n[Test 4] Motion Field Visualization")

    # 运动场统计
    for t in [0, 4, 8, 12]:  # Onset, Apex, early Offset, late Offset
        motion = motion_fields[t]
        dx_mean = motion[:, 0].mean().item()
        dy_mean = motion[:, 1].mean().item()
        print(f"  Frame {t}: dx={dx_mean:.3f}, dy={dy_mean:.3f}")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)

    print("\n[Key Components]")
    print("  1. KeypointDetector: 68面部关键点")
    print("  2. MotionFieldEstimator: AU → 关键点位移 → 密集运动场")
    print("  3. ImageGenerator: 运动场warp生成")
    print("  4. FOMMBaseline: 对比方法")


if __name__ == '__main__':
    demo_real_generation()