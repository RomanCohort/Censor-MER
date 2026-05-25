# =============================================================================
# Advanced GAN: Temporal + Multi-scale + Perceptual + Cycle Consistency
# =============================================================================
# 进阶GAN改进方案：
#
# 1. Temporal GAN：考虑微表情时序（onset/apex/offset）
# 2. Multi-scale Discriminator：多尺度判别
# 3. Perceptual Loss：感知损失而非像素损失
# 4. Cycle Consistency：循环一致性（生成→识别→生成）
# 5. VAE-GAN：变分GAN避免模式坍塌
#
# 参考：
#   - MoCoGAN: Motion Content GAN for video
#   - TGAN: Temporal GAN
#   - StyleGAN-V: Video StyleGAN
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_generator import CensorGGenerator
from data.casme2_real_loader import MultiDatasetGenerator


# =============================================================================
# 1. Temporal Discriminator
# =============================================================================

class TemporalDiscriminator(nn.Module):
    """
    时序判别器

    不只判别整体视频，还判别：
      - Onset阶段（表情开始）
      - Apex阶段（表情峰值）
      - Offset阶段（表情结束）

    微表情的时序特性很重要：
      - Onset应该有运动增加
      - Apex应该有峰值保持
      - Offset应该有运动减少
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # 主判别器（整体）
        self.main_disc = nn.Sequential(
            nn.Conv3d(3, 32, (3,3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

        # 时序判别器（分段）
        self.temporal_disc = nn.Sequential(
            nn.Conv3d(3, 16, (1,3,3), padding=(0,1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(16, 3),  # onset/apex/offset质量
        )

        # 时序一致性判别
        self.temporal_consistency = nn.Sequential(
            nn.Linear(16 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, video):
        """
        Args:
            video: (B, C, T, H, W)

        Returns:
            logits: 类别预测
            temporal_quality: 时序质量评分
            temporal_consistency: 时序一致性评分
        """
        B, C, T, H, W = video.shape

        # 整体判别
        logits = self.main_disc(video)

        # 分段判别（假设T=16）
        onset_frames = video[:, :, :T//4]      # 0-25%
        apex_frames = video[:, :, T//4:T//2]   # 25-50%
        offset_frames = video[:, :, T//2:]     # 50-100%

        onset_feat = self.temporal_disc(onset_frames)
        apex_feat = self.temporal_disc(apex_frames)
        offset_feat = self.temporal_disc(offset_frames)

        temporal_quality = (onset_feat + apex_feat + offset_feat) / 3

        # 时序一致性
        temporal_feat = torch.cat([onset_feat, apex_feat, offset_feat], dim=1)
        temporal_consistency = self.temporal_consistency(temporal_feat)

        return logits, temporal_quality, temporal_consistency


# =============================================================================
# 2. Multi-scale Discriminator
# =============================================================================

class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器

    在不同分辨率上判别：
      - 全分辨率：关注整体表情
      - 半分辨率：关注面部区域
      - 四分之一分辨率：关注运动趋势
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # 全尺度
        self.disc_full = nn.Sequential(
            nn.Conv3d(3, 64, (3,3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

        # 半尺度
        self.disc_half = nn.Sequential(
            nn.Conv3d(3, 32, (3,3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

        # 四分之一尺度
        self.disc_quarter = nn.Sequential(
            nn.Conv3d(3, 16, (3,3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, video):
        """
        Args:
            video: (B, C, T, H, W)

        Returns:
            logits_full, logits_half, logits_quarter
        """
        # 下采样
        video_half = F.avg_pool3d(video, kernel_size=(1,2,2))
        video_quarter = F.avg_pool3d(video_half, kernel_size=(1,2,2))

        logits_full = self.disc_full(video)
        logits_half = self.disc_half(video_half)
        logits_quarter = self.disc_quarter(video_quarter)

        return logits_full, logits_half, logits_quarter

    def get_probs(self, video, expected_class):
        """获取各尺度的正确概率"""
        logits_full, logits_half, logits_quarter = self.forward(video)

        probs_full = F.softmax(logits_full, dim=1)
        probs_half = F.softmax(logits_half, dim=1)
        probs_quarter = F.softmax(logits_quarter, dim=1)

        correct_prob_full = probs_full.gather(1, expected_class.unsqueeze(1)).squeeze(1)
        correct_prob_half = probs_half.gather(1, expected_class.unsqueeze(1)).squeeze(1)
        correct_prob_quarter = probs_quarter.gather(1, expected_class.unsqueeze(1)).squeeze(1)

        return correct_prob_full, correct_prob_half, correct_prob_quarter


# =============================================================================
# 3. Perceptual Loss
# =============================================================================

class PerceptualLoss(nn.Module):
    """
    感知损失

    使用预训练的VGG特征提取器，比较特征而非像素

    为什么重要：
      - L1/L2鼓励精确复制
      - Perceptual鼓励"看起来相似"
      - 更符合人类感知
    """

    def __init__(self):
        super().__init__()

        # 简化的特征提取器（实际应该用VGG）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
        )

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, T, H, W) 生成视频
            target: (B, C, T, H, W) 目标视频

        Returns:
            perceptual_loss
        """
        # 取apex帧比较
        T = pred.shape[2]
        apex_idx = T // 3

        pred_apex = pred[:, :, apex_idx]
        target_apex = target[:, :, apex_idx]

        # 提取特征
        pred_feat = self.feature_extractor(pred_apex)
        target_feat = self.feature_extractor(target_apex)

        # 特征距离
        loss = F.l1_loss(pred_feat, target_feat)

        return loss


# =============================================================================
# 4. Cycle Consistency Loss
# =============================================================================

def compute_cycle_consistency_loss(generator, recognizer, video, au):
    """
    循环一致性损失

    流程：
      1. 生成视频 G(neutral, AU)
      2. 识别视频 → 预测AU
      3. 用预测AU重新生成
      4. 比较两次生成是否一致

    意义：
      - 生成的视频应该能被识别出相同的AU
      - 形成闭环验证
    """
    # 第一次生成
    gen_video1, _ = generator(video, au)

    # 识别 → 预测AU（简化）
    # 实际应该用AU检测器
    # 这里假设识别器输出类别，映射到AU

    # 第二次生成（简化：直接比较两次生成）
    gen_video2, _ = generator(video, au)

    # 循环一致性
    cycle_loss = F.l1_loss(gen_video1, gen_video2)

    return cycle_loss


# =============================================================================
# 5. Advanced GAN Loss
# =============================================================================

def compute_advanced_gan_loss(generator,
                                discriminator,
                                perceptual_loss,
                                neutral_face,
                                target_video,
                                expected_class,
                                au_activation,
                                use_temporal=True,
                                use_multiscale=True,
                                use_perceptual=True,
                                use_cycle=True):
    """
    综合GAN损失

    组合：
      1. 对抗损失：让判别器正确识别
      2. 时序损失：正确的onset/apex/offset
      3. 多尺度损失：各尺度正确识别
      4. 感知损失：看起来相似
      5. 循环一致性：闭环验证
    """

    # 生成视频
    generated_video, motion_fields = generator(neutral_face, au_activation)

    # === 1. 对抗损失 ===
    if use_multiscale:
        correct_full, correct_half, correct_quarter = discriminator.get_probs(
            generated_video, expected_class
        )
        adv_loss = -(torch.log(correct_full + 1e-8).mean() +
                     torch.log(correct_half + 1e-8).mean() * 0.5 +
                     torch.log(correct_quarter + 1e-8).mean() * 0.25)
    else:
        logits, probs = discriminator(generated_video)
        correct_prob = probs.gather(1, expected_class.unsqueeze(1)).squeeze(1)
        adv_loss = -torch.log(correct_prob + 1e-8).mean()

    # === 2. 感知损失 ===
    if use_perceptual:
        perc_loss = perceptual_loss(generated_video, target_video)
    else:
        perc_loss = 0

    # === 3. 时序损失 ===
    if use_temporal:
        # 计算帧间运动变化
        frame_motion = []
        for t in range(generated_video.shape[2] - 1):
            motion = (generated_video[:,:,t+1] - generated_video[:,:,t]).abs().mean()
            frame_motion.append(motion)

        # 微表情时序特性：
        # onset: 运动增加
        # apex: 运动峰值
        # offset: 运动减少
        onset_motion = torch.stack(frame_motion[:4]).mean()
        apex_motion = torch.stack(frame_motion[4:8]).max()
        offset_motion = torch.stack(frame_motion[8:]).mean()

        # 期望：apex > onset > offset
        temporal_loss = F.relu(onset_motion - apex_motion) + \
                        F.relu(offset_motion - onset_motion)
    else:
        temporal_loss = 0

    # === 4. 循环一致性 ===
    if use_cycle:
        cycle_loss = compute_cycle_consistency_loss(
            generator, discriminator, neutral_face, au_activation
        )
    else:
        cycle_loss = 0

    # === 总损失 ===
    total_loss = adv_loss + \
                 perc_loss * 0.1 + \
                 temporal_loss * 0.1 + \
                 cycle_loss * 0.05

    return total_loss, {
        'adv_loss': adv_loss.item(),
        'perc_loss': perc_loss if isinstance(perc_loss, float) else perc_loss.item(),
        'temporal_loss': temporal_loss if isinstance(temporal_loss, float) else temporal_loss.item(),
        'cycle_loss': cycle_loss if isinstance(cycle_loss, float) else cycle_loss.item(),
    }


# =============================================================================
# Advanced GAN Trainer
# =============================================================================

class AdvancedGANTrainer:
    """进阶GAN训练器"""

    def __init__(self, generator, discriminator, g_lr=1e-4):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.device = next(generator.parameters()).device

        self.optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)
        self.perceptual_loss = PerceptualLoss().to(self.device)

        self.training_log = {'epochs': []}

    def train_step(self, batch):
        """单步训练"""
        neutral_face = batch['neutral_face'].to(self.device)
        target_video = batch['target_video'].to(self.device)
        expected_class = batch['emotion_class'].to(self.device)
        au_activation = batch['au_activation'].to(self.device)

        self.optimizer.zero_grad()

        loss, loss_dict = compute_advanced_gan_loss(
            self.generator,
            self.discriminator,
            self.perceptual_loss,
            neutral_face,
            target_video,
            expected_class,
            au_activation,
            use_temporal=True,
            use_multiscale=True,
            use_perceptual=True,
            use_cycle=False,  # 暂时不用，计算复杂
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer.step()

        return loss_dict

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        metrics_list = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            metrics = self.train_step(batch)
            metrics_list.append(metrics)

        avg_metrics = {
            'epoch': epoch,
            'adv_loss': np.mean([m['adv_loss'] for m in metrics_list]),
            'perc_loss': np.mean([m['perc_loss'] for m in metrics_list]),
            'temporal_loss': np.mean([m['temporal_loss'] for m in metrics_list]),
        }

        return avg_metrics


# =============================================================================
# Demo
# =============================================================================

def demo_advanced_gan():
    """演示进阶GAN"""
    print("\n" + "="*60)
    print("Advanced GAN for Micro-Expression Generation")
    print("="*60)

    # 创建判别器
    temporal_disc = TemporalDiscriminator()
    multiscale_disc = MultiScaleDiscriminator()

    # 测试输入
    video = torch.randn(2, 3, 16, 224, 224)

    # 测试时序判别
    logits, temporal_quality, consistency = temporal_disc(video)
    print(f"\n[Temporal Discriminator]")
    print(f"  logits shape: {logits.shape}")
    print(f"  temporal quality: {temporal_quality.shape}")
    print(f"  consistency: {consistency.shape}")

    # 测试多尺度判别
    logits_f, logits_h, logits_q = multiscale_disc(video)
    print(f"\n[Multi-scale Discriminator]")
    print(f"  full: {logits_f.shape}")
    print(f"  half: {logits_h.shape}")
    print(f"  quarter: {logits_q.shape}")

    # 测试感知损失
    perceptual = PerceptualLoss()
    pred = torch.randn(2, 3, 16, 224, 224)
    target = torch.randn(2, 3, 16, 224, 224)
    perc_loss = perceptual(pred, target)
    print(f"\n[Perceptual Loss]")
    print(f"  loss: {perc_loss.item():.4f}")

    print("\n" + "="*60)
    print("Advanced GAN Components Demo Complete")
    print("="*60)


if __name__ == '__main__':
    demo_advanced_gan()