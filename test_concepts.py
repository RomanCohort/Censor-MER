# =============================================================================
# Test: New Biomimetic Concepts
# =============================================================================
# Tests Dynamic Topology Networks & Meta-Plasticity Memory concepts
# without network dependencies.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datetime import datetime


# =============================================================================
# Part 1: Dynamic Topology Networks (DTN)
# =============================================================================

class TensionComputation(nn.Module):
    """计算特征图梯度（模拟细胞骨架张力）"""

    def __init__(self):
        super().__init__()

    def compute_tension(self, x):
        """
        x: (B, C, H, W)
        返回: (B, H, W) 张量场
        """
        # 计算梯度
        B, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        x_flat = x.view(B * C, 1, H, W)

        sobel_x_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=x.dtype, device=x.device)
        sobel_y_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=x.dtype, device=x.device)

        grad_x = F.conv2d(x_flat, sobel_x_kernel.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = F.conv2d(x_flat, sobel_y_kernel.unsqueeze(0).unsqueeze(0), padding=1)

        grad_x = grad_x.view(B, C, H, W)
        grad_y = grad_y.view(B, C, H, W)

        # 张量 = sqrt(grad_x^2 + grad_y^2)
        tension = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        tension = tension.mean(dim=1)  # (B, H, W)

        return tension


class MechanicalGate(nn.Module):
    """机械力敏感通道：阈值门控"""

    def __init__(self, threshold=0.5, gain=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.gain = nn.Parameter(torch.tensor(gain))

    def forward(self, tension):
        """
        tension: (B, H, W)
        返回: (B, H, W) 门控值 0 或 1
        """
        # sigmoid门控：alpha * tension - tau
        gate = torch.sigmoid(self.gain * tension - self.threshold)
        return gate


class DynamicTopologyLayer(nn.Module):
    """动态拓扑网络层：基于张量的门控"""

    def __init__(self, in_dim, hidden_dim=128, k=8):
        super().__init__()
        self.tension_gate = MechanicalGate()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        x: (B, N, D) 特征图
        返回: (B, N, D) 更新后的特征
        """
        B, N, D = x.shape

        # 简化测试：使用平均池化 + 张量门控
        # 1. 计算全局张量场（特征变化）
        x_centered = x - x.mean(dim=1, keepdim=True)
        tension = torch.norm(x_centered, dim=-1)  # (B, N)

        # 2. 门控应用
        gate = self.tension_gate(tension.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)
        out = x * gate * self.scale

        return out


def test_dtn():
    """测试动态拓扑网络"""
    print("\n" + "=" * 50)
    print(" Test: Dynamic Topology Networks (DTN)")
    print("=" * 50)

    # 模拟输入：特征图 (B=2, N=16, D=64)
    x = torch.randn(2, 16, 64)
    print(f"Input: {x.shape}")

    # 组件测试
    tension_comp = TensionComputation()
    # 输入视为 (B, C=1, H=4, W=4) 计算张力
    x_2d = x.mean(dim=-1).view(2, 1, 4, 4)  # (B, 1, 4, 4)
    tension = tension_comp.compute_tension(x_2d)
    print(f"Tension field: {tension.shape}")  # (B, 4, 4)
    print(f"  Max tension: {tension.max().item():.4f}")

    gate = MechanicalGate()
    gate_out = gate(tension)
    print(f"Gate output: {gate_out.shape}")
    print(f"  Open ratio: {(gate_out > 0.5).float().mean().item():.2%}")

    # 完整层测试
    dtn_layer = DynamicTopologyLayer(in_dim=64, hidden_dim=128)
    out = dtn_layer(x)
    print(f"Output: {out.shape}")

    print("OK DTN test passed!")


# =============================================================================
# Part 2: Meta-Plasticity Memory
# =============================================================================

class EmotionStimulusDetector(nn.Module):
    """情绪刺激检测器"""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, context_embeds):
        """
        context_embeds: (B, D)
        返回: (B, 1) 刺激分数
        """
        return self.classifier(context_embeds)


class MethylationRecord(nn.Module):
    """甲基化记录（可学习的记忆单元）"""

    def __init__(self, rank=8, target_dim=128):
        super().__init__()
        self.rank = rank
        # LoRA-like 更新矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, target_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(target_dim, rank) * 0.01)
        self.metadata = {
            'timestamp': None,
            'trigger_type': None,
            'intensity': 0.0,
        }

    def get_delta(self):
        """返回 LoRA 更新: B @ A"""
        return self.lora_B @ self.lora_A

    def consolidate(self, trigger_type, intensity):
        """固化记忆"""
        self.metadata['timestamp'] = datetime.now().isoformat()
        self.metadata['trigger_type'] = trigger_type
        self.metadata['intensity'] = intensity
        self.requires_grad = False  # 冻结


class MetaPlasticityMemory(nn.Module):
    """双轨制记忆系统"""

    def __init__(self, base_dim=128, rank=8, strong_threshold=0.8, weak_threshold=0.5):
        super().__init__()
        # 短期记忆：KV Cache (用tensor模拟)
        self.kv_cache = None

        # 长期记忆：LoRA权重
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold

        # 甲基化模块
        self.emotion_detector = EmotionStimulusDetector(hidden_dim=base_dim)
        self.methylation_slots = nn.ModuleList([
            MethylationRecord(rank=rank, target_dim=base_dim)
            for _ in range(4)  # 最多4个记忆槽
        ])
        self.slot_usage = [0] * 4

    def forward(self, x, context_embeds=None):
        """
        x: (B, D) 当前特征
        context_embeds: (B, D) 上下文特征
        返回: (B, D) 增强后的特征
        """
        B, D = x.shape

        # 1. 检测情绪刺激
        if context_embeds is not None:
            emotion_score = self.emotion_detector(context_embeds)

            # 2. 触发甲基化更新
            if (emotion_score > self.strong_threshold).any():
                intensity = emotion_score.max().item()
                self._trigger_methylation(
                    trigger_type="emotion_stimulus",
                    intensity=intensity
                )
        else:
            emotion_score = None

        # 3. 应用所有甲基化记忆
        modified = x
        for slot in self.methylation_slots:
            if slot.metadata['timestamp'] is not None:
                delta = slot.get_delta()  # (D, r) @ (r, D) = (D, D)
                modified = modified + delta.mean(dim=0, keepdim=True) * slot.metadata['intensity']

        return modified, emotion_score

    def _trigger_methylation(self, trigger_type, intensity):
        """触发甲基化更新"""
        # 找一个空闲槽位
        for i, slot in enumerate(self.methylation_slots):
            if slot.metadata['timestamp'] is None:
                slot.consolidate(trigger_type, intensity)
                self.slot_usage[i] += 1
                print(f"[Methylation] Slot {i} activated: {trigger_type}, intensity={intensity:.3f}")
                return

        # 如果都满了，替换最旧的
        oldest_idx = self.slot_usage.index(min(self.slot_usage))
        self.methylation_slots[oldest_idx] = MethylationRecord(
            rank=8, target_dim=128
        ).to(next(self.parameters()).device)
        self.methylation_slots[oldest_idx].consolidate(trigger_type, intensity)
        print(f"[Methylation] Slot {oldest_idx} replaced: {trigger_type}, intensity={intensity:.3f}")


def test_meta_plasticity():
    """测试元学习记忆"""
    print("\n" + "=" * 50)
    print(" Test: Meta-Plasticity Memory")
    print("=" * 50)

    # 模拟输入
    x = torch.randn(2, 128)  # 当前特征
    context = torch.randn(2, 128)  # 上下文

    print(f"Input features: {x.shape}")
    print(f"Context features: {context.shape}")

    # 检测情绪
    detector = EmotionStimulusDetector()
    emotion = detector(context)
    print(f"Emotion scores: {emotion.squeeze().tolist()}")

    # 双轨制记忆
    memory = MetaPlasticityMemory(base_dim=128, strong_threshold=0.7)
    out, emotion_out = memory(x, context)
    print(f"Output features: {out.shape}")

    # 打印甲基化记录
    records = [slot.metadata for slot in memory.methylation_slots]
    active = [r for r in records if r['timestamp'] is not None]
    print(f"Active methylation slots: {len(active)}/{len(records)}")

    print("OK Meta-Plasticity Memory test passed!")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(" Biomimetic Concepts Test: DTN + Meta-Plasticity")
    print("=" * 60)

    # 测试DTN
    test_dtn()

    # 测试元学习记忆
    test_meta_plasticity()

    print("\n" + "=" * 60)
    print(" All Tests Passed!")
    print("=" * 60)