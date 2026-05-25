# =============================================================================
# GAN Training: Generator vs Recognition Discriminator
# =============================================================================
# 把识别器作为判别器，形成GAN对抗训练
#
# 模式：
#   Generator: 生成微表情视频
#   Discriminator (= Recognizer): 判断生成视频的情感类别
#
# 训练：
#   G想让D正确识别 → 生成正确表情
#   D想正确分类 → 监督G
#
# 优势：
#   - 直接监督，比Policy Gradient更高效
#   - 判别器天然适合"识别是否正确"
#   - 无需复杂的reward计算
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
from data.casme2_real_loader import MultiDatasetGenerator, CASME2_EMOTION_MAPPING


# =============================================================================
# Simple Discriminator (for GAN training)
# =============================================================================

class SimpleDiscriminator(nn.Module):
    """简化判别器"""

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, video):
        feat = self.features(video)
        feat = feat.flatten(1)
        logits = self.classifier(feat)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def classify(self, video, expected_class):
        logits, probs = self.forward(video)
        correct_prob = probs.gather(1, expected_class.unsqueeze(1)).squeeze(1)
        predicted_class = probs.argmax(dim=1)
        return correct_prob, predicted_class, probs


# =============================================================================
# Discriminator = Recognition Model
# =============================================================================

def create_discriminator_from_recognizer(recognizer_checkpoint: str, num_classes: int = 5):
    """
    从识别器创建判别器

    判别器的作用：判断生成视频的情感类别是否正确
    """
    from generation.train_rlhf import load_recognition_model

    # 加载识别器
    recognizer = load_recognition_model(recognizer_checkpoint, num_classes=num_classes)

    # 包装为判别器
    class RecognitionDiscriminator(nn.Module):
        def __init__(self, recognizer, num_classes):
            super().__init__()
            self.recognizer = recognizer
            self.num_classes = num_classes

        def forward(self, video):
            """判断视频的情感类别"""
            logits = self.recognizer(video)
            probs = F.softmax(logits, dim=1)
            return logits, probs

        def classify(self, video, expected_class):
            """
            分类并返回正确概率

            Args:
                video: 生成的视频
                expected_class: 期望的类别

            Returns:
                correct_prob: 正确类别的概率
                predicted_class: 预测的类别
            """
            logits, probs = self.forward(video)

            # 正确类别的概率
            correct_prob = probs.gather(1, expected_class.unsqueeze(1)).squeeze(1)

            # 预测类别
            predicted_class = probs.argmax(dim=1)

            return correct_prob, predicted_class, probs

    return RecognitionDiscriminator(recognizer, num_classes)


# =============================================================================
# GAN Loss Functions
# =============================================================================

def compute_generator_loss(discriminator, generated_video, expected_class):
    """
    生成器损失

    目标：让判别器正确识别生成视频的情感类别

    Loss = -log(D正确识别的概率)
    """
    correct_prob, predicted_class, probs = discriminator.classify(generated_video, expected_class)

    # 生成器想让正确概率最大化 → 最小化负log概率
    g_loss = -torch.log(correct_prob + 1e-8).mean()

    # 辅助损失：鼓励高置信度
    confidence_loss = -probs.max(dim=1)[0].mean() * 0.1

    total_loss = g_loss + confidence_loss

    return total_loss, correct_prob.mean().item()


def compute_discriminator_loss(discriminator, real_video, expected_class, generated_video):
    """
    判别器损失

    目标：正确分类真实视频，同时监督生成器

    这里我们固定判别器（用预训练的识别器），只训练生成器
    """
    # 分类真实视频
    logits_real, probs_real = discriminator(real_video)

    # 分类损失（真实视频应该被正确分类）
    d_loss = F.cross_entropy(logits_real, expected_class)

    return d_loss


# =============================================================================
# GAN Trainer
# =============================================================================

class GANTrainer:
    """GAN训练器"""

    def __init__(self,
                 generator,
                 discriminator,
                 freeze_discriminator: bool = True,
                 g_lr: float = 1e-4,
                 d_lr: float = 1e-5):
        """
        Args:
            generator: 生成器
            discriminator: 判别器（识别器）
            freeze_discriminator: 是否冻结判别器
            g_lr: 生成器学习率
            d_lr: 判别器学习率
        """
        self.generator = generator
        self.discriminator = discriminator
        self.freeze_discriminator = freeze_discriminator

        # 设备
        self.device = next(generator.parameters()).device

        # 优化器
        self.g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)

        if not freeze_discriminator:
            self.d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
        else:
            self.d_optimizer = None

        # 记录
        self.training_log = {
            'config': {
                'g_lr': g_lr,
                'd_lr': d_lr,
                'freeze_discriminator': freeze_discriminator,
            },
            'epochs': [],
        }

    def train_step(self, batch, train_discriminator: bool = False):
        """
        单步训练

        Args:
            batch: 数据batch
            train_discriminator: 是否训练判别器
        """
        neutral_face = batch['neutral_face'].to(self.device)
        target_video = batch['target_video'].to(self.device)
        expected_class = batch['emotion_class'].to(self.device)
        au_activation = batch['au_activation'].to(self.device)

        # === Step 1: 训练生成器 ===
        self.g_optimizer.zero_grad()

        # 生成视频
        generated_video, motion_fields = self.generator(neutral_face, au_activation)

        # 计算生成器损失
        g_loss, correct_prob = compute_generator_loss(
            self.discriminator,
            generated_video,
            expected_class
        )

        # 反向传播
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.g_optimizer.step()

        # === Step 2: 训练判别器（可选）===
        d_loss = 0
        if train_discriminator and self.d_optimizer is not None:
            self.d_optimizer.zero_grad()

            d_loss = compute_discriminator_loss(
                self.discriminator,
                target_video,
                expected_class,
                generated_video.detach()
            )

            d_loss.backward()
            self.d_optimizer.step()
            d_loss = d_loss.item()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss,
            'correct_prob': correct_prob,
        }

    def train_epoch(self, dataloader, epoch, train_d_every: int = 5):
        """训练一个epoch"""
        metrics_list = []

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # 每train_d_every步训练一次判别器
            train_d = (i % train_d_every == 0) and not self.freeze_discriminator

            metrics = self.train_step(batch, train_discriminator=train_d)
            metrics_list.append(metrics)

        # 汇总
        avg_metrics = {
            'epoch': epoch,
            'g_loss': np.mean([m['g_loss'] for m in metrics_list]),
            'd_loss': np.mean([m['d_loss'] for m in metrics_list]),
            'correct_prob': np.mean([m['correct_prob'] for m in metrics_list]),
        }

        return avg_metrics


# =============================================================================
# Main Training Function
# =============================================================================

def train_gan(args):
    """GAN训练主函数"""

    print("\n" + "="*60)
    print("GAN Training: Generator vs Recognition Discriminator")
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

    # 使用简化判别器（避免复杂依赖）
    discriminator = SimpleDiscriminator(num_classes=5).to(device)

    print(f"  Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # === 3. 创建训练器 ===
    print("\n[3] Creating GAN trainer...")

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        freeze_discriminator=args.freeze_discriminator,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
    )

    # === 4. 训练循环 ===
    print("\n[4] Training...")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    best_correct_prob = 0

    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch)

        print(f"\n  Epoch {epoch} Summary:")
        print(f"    G Loss: {metrics['g_loss']:.4f}")
        print(f"    D Loss: {metrics['d_loss']:.4f}")
        print(f"    Correct Prob: {metrics['correct_prob']:.4f}")

        trainer.training_log['epochs'].append(metrics)

        # 保存checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'gan_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'metrics': metrics,
            }, checkpoint_path)
            print(f"    Saved: {checkpoint_path}")

        # 保存最佳模型
        if metrics['correct_prob'] > best_correct_prob:
            best_correct_prob = metrics['correct_prob']
            best_path = os.path.join(args.save_dir, 'gan_best.pth')
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'metrics': metrics,
            }, best_path)
            print(f"    [Best] Saved: {best_path} (correct_prob={best_correct_prob:.4f})")

    # === 5. 保存训练日志 ===
    log_path = os.path.join(args.log_dir, 'gan_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(trainer.training_log, f, indent=2)
    print(f"\n  Training log saved: {log_path}")

    # === 6. 保存最终模型 ===
    final_path = os.path.join(args.save_dir, 'gan_final.pth')
    torch.save({
        'epoch': args.epochs,
        'generator': generator.state_dict(),
    }, final_path)
    print(f"  Final model saved: {final_path}")

    print("\n" + "="*60)
    print("[GAN Training Complete]")
    print(f"Best Correct Prob: {best_correct_prob:.4f}")
    print("="*60)

    return generator, trainer.training_log


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GAN Training with Recognition Discriminator')

    # 数据参数
    parser.add_argument('--casme2_root', type=str, default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', type=str, default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', type=str, default='/root/data/SAMM')

    # 模型参数
    parser.add_argument('--generator_checkpoint', type=str,
                        default='./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth')
    parser.add_argument('--recognizer_checkpoint', type=str,
                        default='./checkpoints/cross_casme2/cross_src_casme2_best.pth')

    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)

    # GAN参数
    parser.add_argument('--freeze_discriminator', type=bool, default=True,
                        help='冻结判别器（只训练生成器）')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='生成器学习率')
    parser.add_argument('--d_lr', type=float, default=1e-5, help='判别器学习率')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints/gan_generator')
    parser.add_argument('--log_dir', type=str, default='./logs/gan_generator')
    parser.add_argument('--save_every', type=int, default=10)

    args = parser.parse_args()

    train_gan(args)


if __name__ == '__main__':
    main()