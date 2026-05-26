# =============================================================================
# RLHF Training with Recognition-based Reward
# =============================================================================
# 使用识别器作为奖励模型的RLHF训练
#
# 流程：
#   1. 生成器生成视频
#   2. 识别器评估生成质量
#   3. 计算奖励（识别正确→高奖励）
#   4. PPO优化生成器
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
from model.censor_g_snn import CensorGSNN
from model.backbones import FastSubcorticalPathway, SlowCorticalPathway
from model.fusion import TSFmicroFusion
from data.casme2_real_loader import MultiDatasetGenerator, CASME2_EMOTION_MAPPING


# =============================================================================
# Load Real Recognition Model (87% accuracy)
# =============================================================================

def load_recognition_model(checkpoint_path: str, num_classes: int = 5):
    """
    加载真实的识别模型

    Args:
        checkpoint_path: checkpoint路径
        num_classes: 类别数

    Returns:
        model: 识别模型
    """
    print(f"[RecognitionModel] Loading from {checkpoint_path}")

    # 创建模型结构（简化版，不需要完全匹配原模型）
    class CensorRecognizer(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()

            # 双通路
            self.fast_pathway = FastSubcorticalPathway()
            self.slow_pathway = SlowCorticalPathway()

            # 融合（使用默认config）
            from config.defaults import FUSION_CONFIG
            self.fusion = TSFmicroFusion(config=FUSION_CONFIG)

            # 简化分类头（代替复杂的MoE）
            from config.defaults import FUSION_CONFIG
            fused_dim = FUSION_CONFIG['fused_dim']
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, frames, flow=None):
            """
            Args:
                frames: (B, C, T, H, W) RGB帧
                flow: (B, 2, T, H, W) 光流（可选）

            Returns:
                logits: (B, num_classes)
            """
            # 如果没有光流，从帧计算
            if flow is None:
                # 简化：用帧差代替光流
                flow = frames[:, :, 1:] - frames[:, :, :-1]
                flow = F.pad(flow, (0, 0, 0, 0, 0, 1), mode='replicate')
                flow = flow[:, :2]  # 取前2通道

            # 双通路处理
            fast_feat = self.fast_pathway(flow)

            # Slow pathway返回(pooled, spatial_map)，取第一个
            slow_out = self.slow_pathway(frames)
            if isinstance(slow_out, tuple):
                slow_feat = slow_out[0]  # pooled features
            else:
                slow_feat = slow_out

            # 融合
            fused_feat = self.fusion(fast_feat, slow_feat)

            # 分类
            logits = self.classifier(fused_feat)

            return logits

    # 创建模型
    model = CensorRecognizer(num_classes=num_classes)

    # 加载权重
    try:
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

        # 根据checkpoint结构加载
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            # 尝试直接加载
            model.load_state_dict(ckpt)

        print(f"[RecognitionModel] Successfully loaded checkpoint")

        # 如果有准确率信息，打印
        if 'best_acc' in ckpt:
            print(f"  Best accuracy: {ckpt['best_acc']:.2%}")
        elif 'accuracy' in ckpt:
            print(f"  Accuracy: {ckpt['accuracy']:.2%}")

    except Exception as e:
        print(f"[RecognitionModel] Warning: Could not load full checkpoint: {e}")
        print(f"  Using partial weights or random initialization")

    return model


# =============================================================================
# Recognition-based Reward Model
# =============================================================================

class RecognitionRewardModel(nn.Module):
    """
    基于识别器的奖励模型

    用预训练的识别器评估生成质量
    """

    def __init__(self, recognizer_checkpoint: str = None, num_classes: int = 4):
        super().__init__()

        self.num_classes = num_classes

        # 加载真实识别器或使用简化版
        if recognizer_checkpoint and os.path.exists(recognizer_checkpoint):
            self.recognizer = load_recognition_model(recognizer_checkpoint, num_classes=5)
        else:
            print("[RecognitionRewardModel] Using simplified recognizer (no checkpoint)")
            self.recognizer = self._build_simple_recognizer()

    def _build_simple_recognizer(self):
        """构建简单的识别器"""
        return nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        识别视频的情感类别

        Args:
            video: (B, C, T, H, W) - 生成视频只有3通道

        Returns:
            logits: (B, num_classes)
        """
        # 如果是真实识别器，需要适配输入格式
        if hasattr(self.recognizer, 'slow_pathway'):
            # Slow pathway期望6通道（RGB + rPPG）
            # rPPG需要计算，简化：用零填充
            if video.shape[1] == 3:
                # 添加3个零通道作为rPPG
                rppg = torch.zeros_like(video)
                video_6ch = torch.cat([video, rppg], dim=1)  # (B, 6, T, H, W)
            else:
                video_6ch = video

            return self.recognizer(video_6ch)
        else:
            # 简化识别器
            return self.recognizer(video)

    def compute_reward(self,
                       generated_video: torch.Tensor,
                       expected_class: torch.Tensor) -> torch.Tensor:
        """
        计算奖励（改进版）

        改进：
          1. 多维度奖励：识别正确 + 置信度 + AU匹配
          2. 梯度可传递：不detach奖励
          3. 软奖励：鼓励接近正确而非完全正确

        Args:
            generated_video: (B, C, T, H, W) 生成的视频
            expected_class: (B,) 期望的类别

        Returns:
            reward: (B,) 奖励值
        """
        # 识别生成的视频
        logits = self.forward(generated_video)
        probs = F.softmax(logits, dim=1)

        # 预测类别
        predicted_class = probs.argmax(dim=1)

        # === 奖励1：正确类别概率（软奖励）===
        # 正确类别的概率越高 → 奖励越高
        correct_prob = probs.gather(1, expected_class.unsqueeze(1)).squeeze(1)

        # === 奖励2：预测正确额外奖励 ===
        correct_mask = (predicted_class == expected_class).float()

        # === 奖励3：置信度奖励 ===
        confidence = probs.max(dim=1)[0]

        # === 奖励4：运动幅度奖励 ===
        # PHASE 3 FIX: 分段运动奖励，鼓励适度运动而非静止
        MIN_MOTION = 0.02   # 约5像素（下限）
        MAX_MOTION = 0.08   # 约18像素（上限）
        TARGET_MOTION = 0.04  # 约10像素（目标）

        # 计算视频帧间差异（鼓励运动）
        frame_diff = generated_video[:, :, 1:] - generated_video[:, :, :-1]
        motion_magnitude = frame_diff.abs().mean(dim=[1, 2, 3, 4])

        # 分段奖励：
        # - 太小（<MIN）：低奖励（太弱看不见）
        # - 适中（MIN~MAX）：高奖励（可见但不过度）
        # - 太大（>MAX）：中等奖励（过度夸张）
        motion_reward = torch.where(
            motion_magnitude < MIN_MOTION,
            torch.exp(-(motion_magnitude - MIN_MOTION)**2 / 0.001),  # 太小：低奖励
            torch.where(
                motion_magnitude <= MAX_MOTION,
                torch.ones_like(motion_magnitude),  # 适中：高奖励
                torch.exp(-(motion_magnitude - MAX_MOTION)**2 / 0.002)  # 太大：中等奖励
            )
        )

        # === 综合奖励 ===
        reward = (
            correct_prob * 1.5 +            # 正确类别概率（主要）
            correct_mask * confidence * 1.0 +  # 正确识别额外奖励
            motion_reward * 1.0             # PHASE 3 FIX: 权重从0.5改为1.0
        )

        return reward


# =============================================================================
# PPO Training
# =============================================================================

class PPOTrainer:
    """PPO训练器"""

    def __init__(self,
                 generator: nn.Module,
                 reward_model: nn.Module,
                 lr: float = 1e-5,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01):
        """
        Args:
            generator: 生成器
            reward_model: 奖励模型（识别器）
            lr: 学习率
            clip_ratio: PPO裁剪比例
            entropy_coef: 熵奖励系数
        """
        self.generator = generator
        self.reward_model = reward_model
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

        # 设备
        self.device = next(generator.parameters()).device

        # 优化器
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

        # 价值函数（baseline）
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

        # 记录
        self.training_log = {
            'config': {
                'lr': lr,
                'clip_ratio': clip_ratio,
                'entropy_coef': entropy_coef,
            },
            'epochs': [],
        }

    def compute_log_prob(self, au_activation: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率（简化）"""
        # 在实际PPO中，这里需要计算策略的log prob
        # 简化：假设AU激活来自高斯分布
        mean = au_activation
        std = torch.ones_like(au_activation) * 0.1
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(au_activation).sum(dim=-1)
        return log_prob

    def train_step(self, batch: dict) -> dict:
        """
        单步训练

        Args:
            batch: 包含neutral_face, target_video, expected_class等

        Returns:
            metrics: 训练指标
        """
        neutral_face = batch['neutral_face'].to(self.device)
        target_video = batch['target_video'].to(self.device)
        expected_class = batch['emotion_class'].to(self.device)
        au_activation = batch['au_activation'].to(self.device)

        # === Step 1: 生成视频 ===
        generated_video, motion_fields = self.generator(neutral_face, au_activation)

        # === Step 2: 计算奖励 ===
        with torch.no_grad():
            reward = self.reward_model.compute_reward(generated_video, expected_class)

        # === Step 3: 计算价值（baseline）===
        # 简化：使用移动平均作为baseline
        # 实际应该用价值网络
        value = reward.mean()  # 简化

        # === Step 4: 计算优势 ===
        advantage = reward - value
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # === Step 5: 简化Policy Gradient更新 ===
        # 直接优化奖励，不使用完整PPO
        self.optimizer.zero_grad()

        # 重新生成（确保梯度追踪）
        generated_video, motion_fields = self.generator(neutral_face, au_activation)

        # 计算奖励（作为目标）
        reward = self.reward_model.compute_reward(generated_video, expected_class)

        # 简化损失：最大化奖励 = 最小化负奖励
        loss = -reward.mean()

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer.step()

        total_loss = loss.item()

        # 返回指标
        return {
            'loss': total_loss,
            'reward': reward.mean().item(),
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """训练一个epoch"""
        metrics_list = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            metrics = self.train_step(batch)
            metrics_list.append(metrics)

        # 汇总
        avg_metrics = {
            'epoch': epoch,
            'loss': np.mean([m['loss'] for m in metrics_list]),
            'reward': np.mean([m['reward'] for m in metrics_list]),
        }

        return avg_metrics


# =============================================================================
# Main Training Function
# =============================================================================

def train_rlhf(args):
    """RLHF训练主函数"""

    print("\n" + "="*60)
    print("RLHF Training with Recognition-based Reward")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # === 1. 创建数据集 ===
    print("\n[1] Creating dataset...")

    data_roots = {
        'CASME2': args.casme2_root,
        'SMIC': args.smic_root,
        'SAMM': args.samm_root,
    }

    # 过滤掉None的路径
    data_roots = {k: v for k, v in data_roots.items() if v and os.path.exists(v)}

    if not data_roots:
        print("[Error] No valid data roots provided")
        return

    dataset = MultiDatasetGenerator(
        data_roots=data_roots,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"  Dataset size: {len(dataset)}")

    # === 2. 创建模型 ===
    print("\n[2] Creating models...")

    generator = CensorGGenerator(
        num_au=17,
        num_keypoints=68,
        num_frames=args.num_frames,
        image_size=args.image_size,
    ).to(device)

    # 加载预训练权重
    if args.generator_checkpoint:
        ckpt = torch.load(args.generator_checkpoint, weights_only=False, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        print(f"  Loaded generator from {args.generator_checkpoint}")

    reward_model = RecognitionRewardModel(
        recognizer_checkpoint=args.recognizer_checkpoint,
    ).to(device)

    print(f"  Generator params: {sum(p.numel() for p in generator.parameters()):,}")

    # === 3. 创建训练器 ===
    print("\n[3] Creating PPO trainer...")

    trainer = PPOTrainer(
        generator=generator,
        reward_model=reward_model,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
    )

    # === 4. 训练循环 ===
    print("\n[4] Training...")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    best_reward = 0

    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch)

        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Loss: {metrics['loss']:.4f}")
        print(f"    Reward: {metrics['reward']:.4f}")

        trainer.training_log['epochs'].append(metrics)

        # 保存checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'rlhf_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'metrics': metrics,
            }, checkpoint_path)
            print(f"    Saved: {checkpoint_path}")

        # 保存最佳模型
        if metrics['reward'] > best_reward:
            best_reward = metrics['reward']
            best_path = os.path.join(args.save_dir, 'rlhf_best.pth')
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'metrics': metrics,
            }, best_path)
            print(f"    [Best] Saved: {best_path}")

    # === 5. 保存训练日志 ===
    log_path = os.path.join(args.log_dir, 'rlhf_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(trainer.training_log, f, indent=2)
    print(f"\n  Training log saved: {log_path}")

    # === 6. 保存最终模型 ===
    final_path = os.path.join(args.save_dir, 'rlhf_final.pth')
    torch.save({
        'epoch': args.epochs,
        'generator': generator.state_dict(),
    }, final_path)
    print(f"  Final model saved: {final_path}")

    print("\n" + "="*60)
    print("[Training Complete]")
    print("="*60)

    return generator, trainer.training_log


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='RLHF Training with Recognition-based Reward')

    # 数据参数
    parser.add_argument('--casme2_root', type=str, default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', type=str, default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', type=str, default='/root/data/SAMM')

    # 模型参数
    parser.add_argument('--generator_checkpoint', type=str,
                        default='./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth',
                        help='Pretrained generator checkpoint')
    parser.add_argument('--recognizer_checkpoint', type=str, default=None,
                        help='Pretrained recognizer checkpoint (optional)')

    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints/rlhf_generator')
    parser.add_argument('--log_dir', type=str, default='./logs/rlhf_generator')
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    train_rlhf(args)


if __name__ == '__main__':
    main()