# =============================================================================
# Train Hybrid Model: Diffusion + Blendshape + GAN
# =============================================================================
# 使用三重数据集(CASME2 + SMIC + SAMM)训练混合模型
#
# 三阶段训练：
#   Stage 1: 扩散模型预训练
#   Stage 2: GAN对抗微调
#   Stage 3: 联合训练
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

from model.diffusion_blendshape import (
    MicroExpressionDiffusionUNet,
    MicroExpressionDiffusion,
    BlendshapeSystem,
)
from model.hybrid_generator import HybridMicroExpressionGenerator, HybridTrainer
from generation.train_gan import create_discriminator_from_recognizer
from data.casme2_real_loader import MultiDatasetGenerator


# =============================================================================
# Simplified Discriminator for Hybrid Training
# =============================================================================

class SimpleDiscriminator(nn.Module):
    """简化判别器（避免复杂依赖）"""

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
        feat = feat.flatten(1)  # 保持batch维度
        logits = self.classifier(feat)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def classify(self, video, expected_class):
        logits, probs = self.forward(video)
        correct_prob = probs.gather(1, expected_class.unsqueeze(1)).squeeze(1)
        predicted_class = probs.argmax(dim=1)
        return correct_prob, predicted_class, probs


# =============================================================================
# Simplified Diffusion Model
# =============================================================================

class SimplifiedDiffusion(nn.Module):
    """简化扩散模型"""

    def __init__(self, num_frames=16, image_size=64, num_blendshapes=52):
        super().__init__()

        self.num_frames = num_frames
        self.num_timesteps = 1000
        self.num_blendshapes = num_blendshapes

        # β schedule
        self.betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # UNet
        self.model = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 3, 3, padding=1),
        )

        # Blendshape条件
        self.cond_encoder = nn.Sequential(
            nn.Linear(num_blendshapes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

    def forward_diffusion(self, x0, t):
        """前向扩散"""
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].to(x0.device)
        alpha_bar = alpha_bar.view(-1, 1, 1, 1, 1)
        xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise
        return xt, noise

    def denoise_step(self, xt, t, blendshape):
        """去噪一步"""
        # 条件编码
        cond = self.cond_encoder(blendshape)
        cond = cond.view(-1, 256, 1, 1, 1)

        # 添加条件（简化）
        # 实际应该有更复杂的条件注入

        # 预测噪声
        noise_pred = self.model(xt)

        return noise_pred

    def generate(self, neutral_face, blendshape, num_steps=20):
        """生成视频"""
        B, C, T, H, W = neutral_face.shape
        device = neutral_face.device

        # 从噪声开始
        xt = torch.randn(B, C, T, H, W, device=device)

        # 去噪
        for i in range(num_steps):
            t = self.num_timesteps - i - 1
            t_tensor = torch.tensor([t] * B, device=device)

            # 预测噪声
            noise_pred = self.denoise_step(xt, t_tensor, blendshape)

            # 去噪公式
            alpha = self.alphas[t].to(device)
            alpha_bar = self.alpha_bars[t].to(device)

            if i < num_steps - 1:
                noise = torch.randn_like(xt)
            else:
                noise = 0

            xt = (xt - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred) / alpha.sqrt()
            xt = xt + self.betas[t].to(device).sqrt() * noise

        return xt


# =============================================================================
# Training Functions
# =============================================================================

def train_hybrid(args):
    """混合模型训练主函数"""

    print("\n" + "="*70)
    print("Hybrid Model Training: Diffusion + Blendshape + GAN")
    print("Dataset: CASME2 + SMIC + SAMM")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # === 1. 创建数据集 ===
    print("\n[1] Creating dataset...")

    data_roots = {}
    if args.casme2_root and os.path.exists(args.casme2_root):
        data_roots['CASME2'] = args.casme2_root
    if args.smic_root and os.path.exists(args.smic_root):
        data_roots['SMIC'] = args.smic_root
    if args.samm_root and os.path.exists(args.samm_root):
        data_roots['SAMM'] = args.samm_root

    if not data_roots:
        print("[Error] No valid data roots provided")
        print("  Please provide at least one dataset path")
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

    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Datasets: {list(data_roots.keys())}")

    # === 2. 创建模型 ===
    print("\n[2] Creating models...")

    # 扩散模型
    diffusion = SimplifiedDiffusion(
        num_frames=args.num_frames,
        image_size=args.image_size,
        num_blendshapes=52,
    ).to(device)

    # 判别器
    discriminator = SimpleDiscriminator(num_classes=5).to(device)

    # 混合模型
    hybrid_model = HybridMicroExpressionGenerator(
        diffusion_model=diffusion,
        discriminator=discriminator,
        num_blendshapes=52,
        num_frames=args.num_frames,
    ).to(device)

    print(f"  Diffusion params: {sum(p.numel() for p in diffusion.parameters()):,}")
    print(f"  Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # === 3. 创建训练器 ===
    print("\n[3] Creating trainer...")

    trainer = HybridTrainer(
        hybrid_model=hybrid_model,
        diffusion_lr=args.diffusion_lr,
        refiner_lr=args.refiner_lr,
        discriminator_lr=args.discriminator_lr,
    )

    # === 4. 三阶段训练 ===
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Stage 1: 扩散预训练
    print("\n" + "="*70)
    print("Stage 1: Diffusion Pretraining")
    print("="*70)

    for epoch in range(1, args.stage1_epochs + 1):
        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Stage1 Epoch {epoch}"):
            neutral_face = batch['neutral_face'].to(device)
            target_video = batch['target_video'].to(device)
            au_activation = batch['au_activation'].to(device)

            # AU → Blendshape
            blendshape = hybrid_model.blendshape_system.au_to_blendshape(au_activation).to(device).to(device)

            # 扩散训练步
            B = target_video.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (B,))  # CPU上

            # 前向扩散
            xt, noise = diffusion.forward_diffusion(target_video, t)

            # 预测噪声
            noise_pred = diffusion.denoise_step(xt, t, blendshape)

            # 损失
            loss = F.mse_loss(noise_pred, noise)

            # 反向传播
            trainer.diffusion_optimizer.zero_grad()
            loss.backward()
            trainer.diffusion_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch}, Diffusion Loss: {avg_loss:.4f}")

        trainer.training_log['stage1'].append({
            'epoch': epoch,
            'loss': avg_loss,
        })

        # 保存checkpoint
        if epoch % 5 == 0:
            save_path = os.path.join(args.save_dir, f'stage1_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'diffusion': diffusion.state_dict(),
            }, save_path)
            print(f"    Saved: {save_path}")

    # Stage 2: GAN微调
    print("\n" + "="*70)
    print("Stage 2: GAN Adversarial Finetuning")
    print("="*70)

    # 冻结扩散模型
    for param in diffusion.parameters():
        param.requires_grad = False

    for epoch in range(1, args.stage2_epochs + 1):
        total_g_loss = 0
        total_d_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Stage2 Epoch {epoch}"):
            neutral_face = batch['neutral_face'].to(device)
            target_video = batch['target_video'].to(device)
            expected_class = batch['emotion_class'].to(device)
            au_activation = batch['au_activation'].to(device)

            blendshape = hybrid_model.blendshape_system.au_to_blendshape(au_activation).to(device)

            # === 生成器训练 ===
            # 扩散生成（简化：用少量去噪步）
            # neutral_face是单帧，需要扩展为视频
            B, C, H, W = neutral_face.shape
            neutral_video = neutral_face.unsqueeze(2).expand(B, C, args.num_frames, H, W)

            with torch.no_grad():
                gen_video = diffusion.generate(neutral_video, blendshape, num_steps=5)

            # 精修
            refined = gen_video + hybrid_model.refiner(gen_video) * 0.1

            # 判别器评分
            correct_prob, _, _ = discriminator.classify(refined, expected_class)

            # G损失
            g_loss = -torch.log(correct_prob + 1e-8).mean()

            trainer.refiner_optimizer.zero_grad()
            g_loss.backward()
            trainer.refiner_optimizer.step()

            # === 判别器训练 ===
            # 真实视频分类
            logits_real, _ = discriminator(target_video)
            d_loss = F.cross_entropy(logits_real, expected_class)

            trainer.disc_optimizer.zero_grad()
            d_loss.backward()
            trainer.disc_optimizer.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            num_batches += 1

        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        print(f"  Epoch {epoch}, G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

        trainer.training_log['stage2'].append({
            'epoch': epoch,
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
        })

        if epoch % 5 == 0:
            save_path = os.path.join(args.save_dir, f'stage2_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'refiner': hybrid_model.refiner.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, save_path)
            print(f"    Saved: {save_path}")

    # 解冻扩散模型
    for param in diffusion.parameters():
        param.requires_grad = True

    # Stage 3: 联合训练
    print("\n" + "="*70)
    print("Stage 3: Joint Training")
    print("="*70)

    for epoch in range(1, args.stage3_epochs + 1):
        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Stage3 Epoch {epoch}"):
            neutral_face = batch['neutral_face'].to(device)
            target_video = batch['target_video'].to(device)
            expected_class = batch['emotion_class'].to(device)
            au_activation = batch['au_activation'].to(device)

            blendshape = hybrid_model.blendshape_system.au_to_blendshape(au_activation).to(device)

            # === 扩散损失 ===
            B = target_video.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (B,))
            xt, noise = diffusion.forward_diffusion(target_video, t)
            noise_pred = diffusion.denoise_step(xt, t, blendshape)
            diff_loss = F.mse_loss(noise_pred, noise)

            # === GAN损失 ===
            # neutral_face是单帧，需要扩展为视频
            B, C, H, W = neutral_face.shape
            neutral_video = neutral_face.unsqueeze(2).expand(B, C, args.num_frames, H, W)

            with torch.no_grad():
                gen_video = diffusion.generate(neutral_video, blendshape, num_steps=5)
            refined = gen_video + hybrid_model.refiner(gen_video) * 0.1
            correct_prob, _, _ = discriminator.classify(refined, expected_class)
            gan_loss = -torch.log(correct_prob + 1e-8).mean()

            # === 总损失 ===
            total = diff_loss + gan_loss * 0.5

            trainer.diffusion_optimizer.zero_grad()
            trainer.refiner_optimizer.zero_grad()
            total.backward()
            trainer.diffusion_optimizer.step()
            trainer.refiner_optimizer.step()

            total_loss += total.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch}, Joint Loss: {avg_loss:.4f}")

        trainer.training_log['stage3'].append({
            'epoch': epoch,
            'loss': avg_loss,
        })

        if epoch % 5 == 0:
            save_path = os.path.join(args.save_dir, f'stage3_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'diffusion': diffusion.state_dict(),
                'refiner': hybrid_model.refiner.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, save_path)
            print(f"    Saved: {save_path}")

    # === 5. 保存最终模型 ===
    print("\n" + "="*70)
    print("Saving Final Model")
    print("="*70)

    final_path = os.path.join(args.save_dir, 'hybrid_final.pth')
    torch.save({
        'diffusion': diffusion.state_dict(),
        'refiner': hybrid_model.refiner.state_dict(),
        'discriminator': discriminator.state_dict(),
        'blendshape_system': hybrid_model.blendshape_system,
    }, final_path)
    print(f"  Final model saved: {final_path}")

    # 保存训练日志
    log_path = os.path.join(args.log_dir, 'hybrid_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(trainer.training_log, f, indent=2)
    print(f"  Training log saved: {log_path}")

    print("\n" + "="*70)
    print("Hybrid Training Complete!")
    print("="*70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Model')

    # 数据
    parser.add_argument('--casme2_root', type=str, default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', type=str, default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', type=str, default='/root/data/SAMM')

    # 训练参数
    parser.add_argument('--stage1_epochs', type=int, default=10)
    parser.add_argument('--stage2_epochs', type=int, default=10)
    parser.add_argument('--stage3_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)

    # 学习率
    parser.add_argument('--diffusion_lr', type=float, default=1e-4)
    parser.add_argument('--refiner_lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-5)

    # 模型参数
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=64)

    # 保存
    parser.add_argument('--save_dir', type=str, default='./checkpoints/hybrid_model')
    parser.add_argument('--log_dir', type=str, default='./logs/hybrid_model')

    args = parser.parse_args()

    train_hybrid(args)


if __name__ == '__main__':
    main()