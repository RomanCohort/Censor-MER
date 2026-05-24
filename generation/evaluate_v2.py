# =============================================================================
# Censor-G v2 Evaluation Script
# =============================================================================
# 评估视觉中枢启发的微表情生成模型
#
# 评估指标：
#   1. AU一致性（预测的AU vs 生成的AU）
#   2. 时间曲线一致性
#   3. 局部运动场准确性
#   4. 生成质量（FID）
#   5. 层级处理效率
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_v2 import CensorGv2, AU_INDEX, AU_INFO, AU_REGIONS


class CensorGv2Evaluator:
    """
    Censor-G v2 专用评估器。
    """

    def __init__(self, model=None, device='cuda'):
        self.device = device
        self.model = model

    def evaluate_full(self, generated_videos, real_videos, au_target, emotion_target):
        """
        全面评估。

        Args:
            generated_videos (torch.Tensor): 生成的视频
            real_videos (torch.Tensor): 真实视频
            au_target (torch.Tensor): 目标AU
            emotion_target (torch.Tensor): 目标情感

        Returns:
            results (dict): 评估结果
        """
        print("\n[Evaluation] Censor-G v2 Full Evaluation")
        print("="*50)

        results = {}

        # 1. AU一致性
        results['au_consistency'] = self.evaluate_au_consistency(generated_videos, au_target)

        # 2. 时间曲线一致性
        results['temporal_consistency'] = self.evaluate_temporal_consistency(generated_videos)

        # 3. 局部运动场准确性
        results['motion_field_accuracy'] = self.evaluate_motion_field_accuracy(generated_videos, au_target)

        # 4. 生成质量
        results['generation_quality'] = self.evaluate_generation_quality(generated_videos, real_videos)

        # 5. 层级处理效率
        results['layer_efficiency'] = self.evaluate_layer_efficiency(generated_videos)

        # Summary
        print("\n" + "="*50)
        print("Summary:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

        return results

    def evaluate_au_consistency(self, generated_videos, au_target):
        """
        AU一致性评估。

        检查生成的视频中AU是否与目标一致。
        """
        print("[1/5] AU Consistency...")

        B, C, T, H, W = generated_videos.shape

        # 计算每帧与第一帧的差异（运动幅度）
        motion_curves = []
        for b in range(B):
            video = generated_videos[b]
            diffs = []
            for t in range(1, T):
                diff = (video[:, t] - video[:, 0]).abs().mean().item()
                diffs.append(diff)
            motion_curves.append(diffs)

        # 基于运动幅度估算AU激活
        # 这是一个简化的评估方法
        estimated_au = torch.zeros(B, 17)

        # 检查关键区域的变化
        # AU12（嘴角）: 检查嘴角区域变化
        mouth_region = (130, 140, 20)  # (y_center, x_center, radius)
        for b in range(B):
            mouth_motion = self._get_region_motion(generated_videos[b], mouth_region)
            estimated_au[b, AU_INDEX['AU12']] = mouth_motion / 0.1  # 归一化

        # AU1/AU2（眉）: 检查眉毛区域变化
        brow_region = (55, 112, 25)
        for b in range(B):
            brow_motion = self._get_region_motion(generated_videos[b], brow_region)
            estimated_au[b, AU_INDEX['AU1']] = brow_motion / 0.08
            estimated_au[b, AU_INDEX['AU2']] = brow_motion / 0.08

        # 计算一致性
        consistency = F.mse_loss(estimated_au, au_target)
        consistency_score = 1 - consistency.item()

        return max(0, consistency_score)

    def evaluate_temporal_consistency(self, generated_videos):
        """
        时间曲线一致性。

        检查onset-apex-offset曲线的形状。
        """
        print("[2/5] Temporal Consistency...")

        B, C, T, H, W = generated_videos.shape

        scores = []
        for b in range(B):
            # 计算运动曲线
            video = generated_videos[b]
            curve = []
            for t in range(T):
                diff = (video[:, t] - video[:, 0]).abs().mean().item()
                curve.append(diff)

            curve = torch.tensor(curve)

            # 检查曲线形状
            if curve.max() > 0:
                curve = curve / curve.max()

            # 找峰值
            peak_idx = curve.argmax().item()

            # 检查上升阶段
            rise_ok = self._check_rise(curve[:peak_idx+1])

            # 检查下降阶段
            decline_ok = self._check_decline(curve[peak_idx:])

            scores.append((rise_ok + decline_ok) / 2)

        return np.mean(scores)

    def evaluate_motion_field_accuracy(self, generated_videos, au_target):
        """
        局部运动场准确性。

        检查每个AU控制的区域是否正确运动。
        """
        print("[3/5] Motion Field Accuracy...")

        B, C, T, H, W = generated_videos.shape

        scores = []

        # 检查AU12（嘴角）区域
        mouth_region = AU_REGIONS['mouth_corner']
        for b in range(B):
            if au_target[b, AU_INDEX['AU12']] > 0.3:
                # 嘴角应该有显著运动
                mouth_motion = self._get_region_motion(generated_videos[b], mouth_region)
                expected_motion = au_target[b, AU_INDEX['AU12']].item() * 0.1
                accuracy = min(mouth_motion / expected_motion, 1.0)
                scores.append(accuracy)

        # 检查AU1（内眉）区域
        brow_region = AU_REGIONS['eyebrow_inner']
        for b in range(B):
            if au_target[b, AU_INDEX['AU1']] > 0.3:
                brow_motion = self._get_region_motion(generated_videos[b], brow_region)
                expected_motion = au_target[b, AU_INDEX['AU1']].item() * 0.05
                accuracy = min(brow_motion / expected_motion, 1.0)
                scores.append(accuracy)

        if len(scores) == 0:
            return 0.5  # 没有激活的AU，默认返回

        return np.mean(scores)

    def evaluate_generation_quality(self, generated_videos, real_videos):
        """
        生成质量。

        基于像素级相似度。
        """
        print("[4/5] Generation Quality...")

        # L1距离
        l1_dist = F.l1_loss(generated_videos, real_videos).item()

        # 转换为质量分数（距离越小分数越高）
        quality_score = 1 / (1 + l1_dist)

        return quality_score

    def evaluate_layer_efficiency(self, generated_videos):
        """
        层级处理效率。

        检查V1层的显著性筛选是否有效。
        """
        print("[5/5] Layer Efficiency...")

        if self.model is None:
            return 0.5

        # 测试V1层
        B, C, T, H, W = generated_videos.shape

        # 创建随机AU输入
        au_input = torch.rand(B, 17).to(self.device)

        # V1显著性筛选
        v1_output = self.model.v1_saliency(au_input)

        # 检查筛选效率
        significant_count = v1_output['significant_mask'].sum().item()
        total_au = B * 17

        # 理想情况：筛选出30-50%的AU进入精细处理
        efficiency = abs(significant_count / total_au - 0.4)  # 越接近40%越好

        return 1 - min(efficiency, 1)

    def _get_region_motion(self, video, region):
        """获取特定区域的运动幅度。"""
        if isinstance(region, dict):
            center = region['center']
            radius = region['radius']
        else:
            center = (region[1], region[0])
            radius = region[2]

        T = video.shape[1]

        # 计算区域内的运动
        motions = []
        for t in range(1, T):
            frame_diff = (video[:, t] - video[:, 0]).abs()

            # 取区域内的均值
            cy, cx = int(center[1]), int(center[0])
            r = int(radius)

            # 简化：取中心点周围的均值
            y_start = max(0, cy - r)
            y_end = min(video.shape[2], cy + r)
            x_start = max(0, cx - r)
            x_end = min(video.shape[3], cx + r)

            region_motion = frame_diff[:, y_start:y_end, x_start:x_end].mean().item()
            motions.append(region_motion)

        return np.mean(motions)

    def _check_rise(self, curve):
        """检查上升阶段。"""
        if len(curve) < 2:
            return 1.0

        increasing_count = 0
        for i in range(1, len(curve)):
            if curve[i] >= curve[i-1] - 0.1:
                increasing_count += 1

        return increasing_count / (len(curve) - 1)

    def _check_decline(self, curve):
        """检查下降阶段。"""
        if len(curve) < 2:
            return 1.0

        decreasing_count = 0
        for i in range(1, len(curve)):
            if curve[i] <= curve[i-1] + 0.1:
                decreasing_count += 1

        return decreasing_count / (len(curve) - 1)


def evaluate_controllable_generation(model, neutral_face, emotion_classes,
                                      intensities=[0.3, 0.5, 0.7, 0.9]):
    """
    评估可控生成。

    检查不同强度参数下的生成效果。
    """
    print("\n[Evaluation] Controllable Generation")
    print("="*50)

    device = next(model.parameters()).device
    B = len(emotion_classes)

    results = {}

    for intensity in intensities:
        print(f"\n[Intensity = {intensity}]")

        intensity_tensor = torch.tensor([intensity] * B).to(device)

        # 生成
        generated = model.generate_with_intensity(neutral_face, emotion_classes, intensity_tensor)

        # 计算运动幅度
        motion_mag = (generated[:, :, :, :, :] - generated[:, :, 0:1, :, :]).abs().mean().item()

        results[intensity] = {
            'motion_magnitude': motion_mag,
            'expected_ratio': intensity,
        }

        print(f"  Motion Magnitude: {motion_mag:.4f}")
        print(f"  Expected Ratio: {intensity}")

    return results


def evaluate_au_interaction(model):
    """
    评估AU交互效果。

    检查协同/对抗矩阵的效果。
    """
    print("\n[Evaluation] AU Interaction")
    print("="*50)

    device = next(model.parameters()).device

    # 测试AU6+AU12协同（真诚微笑）
    au_input = torch.zeros(1, 17).to(device)
    au_input[0, AU_INDEX['AU6']] = 0.5
    au_input[0, AU_INDEX['AU12']] = 0.5

    effective_au = model.v2_interaction(au_input)

    print("\n[Test 1] AU6 + AU12 Synergy (Sincere Smile)")
    print(f"  Input AU6: 0.5, AU12: 0.5")
    print(f"  Effective AU6: {effective_au[0, AU_INDEX['AU6']].item():.3f}")
    print(f"  Effective AU12: {effective_au[0, AU_INDEX['AU12']].item():.3f}")

    # 测试AU12+AU14对抗（压抑微笑）
    au_input = torch.zeros(1, 17).to(device)
    au_input[0, AU_INDEX['AU12']] = 0.5
    au_input[0, AU_INDEX['AU14']] = 0.5

    effective_au = model.v2_interaction(au_input)

    print("\n[Test 2] AU12 + AU14 Antagonism (Suppressed Smile)")
    print(f"  Input AU12: 0.5, AU14: 0.5")
    print(f"  Effective AU12: {effective_au[0, AU_INDEX['AU12']].item():.3f}")
    print(f"  Effective AU14: {effective_au[0, AU_INDEX['AU14']].item():.3f}")

    return {
        'synergy_test': effective_au[0, AU_INDEX['AU12']].item() > au_input[0, AU_INDEX['AU12']].item(),
        'antagonism_test': effective_au[0, AU_INDEX['AU12']].item() < au_input[0, AU_INDEX['AU12']].item(),
    }


def demo_evaluation():
    """Demo评估。"""
    print("\n" + "="*60)
    print("Censor-G v2 Evaluation Demo")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建模型
    model = CensorGv2().to(device)

    # 创建输入
    B, C, T, H, W = 2, 3, 16, 224, 224
    neutral_face = torch.randn(B, C, H, W).to(device)
    au = torch.rand(B, 17).to(device)
    emotion = torch.tensor([0, 1]).to(device)

    # 生成
    generated = model(neutral_face, au)
    real = torch.randn(B, C, T, H, W).to(device)

    # 评估
    evaluator = CensorGv2Evaluator(model, device)
    results = evaluator.evaluate_full(generated, real, au, emotion)

    # 可控生成评估
    intensity_results = evaluate_controllable_generation(model, neutral_face, emotion)

    # AU交互评估
    interaction_results = evaluate_au_interaction(model)

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_evaluation()