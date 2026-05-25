# =============================================================================
# Train Censor-G Generator on Real CASME2 Data
# =============================================================================
# 真实图像生成训练脚本
#
# 训练流程：
#   1. 加载真实CASME2数据（中性脸 + 微表情视频）
#   2. 训练Censor-G Generator
#   3. 对比FOMM baseline
#   4. 计算真实FID、SSIM等指标
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_generator import CensorGGenerator, FOMMBaseline, VideoDiscriminator
from data.casme2_real_loader import CASME2RealDataset, CASME2GeneratorDataset


def compute_fid(fake_features, real_features):
    """计算FID"""
    mu_fake = fake_features.mean(axis=0)
    mu_real = real_features.mean(axis=0)

    cov_fake = np.cov(fake_features, rowvar=False) + np.eye(fake_features.shape[1]) * 1e-6
    cov_real = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[1]) * 1e-6

    mean_diff = np.sum((mu_fake - mu_real) ** 2)

    cov_sqrt = np.sqrt(cov_fake * cov_real + 1e-8)
    cov_term = np.trace(cov_fake + cov_real - 2 * cov_sqrt)

    fid = mean_diff + cov_term

    return float(np.clip(fid, 0, 1000))


def compute_ssim(generated, target):
    """计算SSIM"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    gen_mean = generated.mean()
    target_mean = target.mean()

    gen_var = generated.var()
    target_var = target.var()

    cov = ((generated - gen_mean) * (target - target_mean)).mean()

    luminance = (2 * gen_mean * target_mean + C1) / (gen_mean ** 2 + target_mean ** 2 + C1)
    contrast = (2 * torch.sqrt(gen_var * target_var) + C2) / (gen_var + target_var + C2)
    structure = (cov + C2) / (torch.sqrt(gen_var * target_var) + C2)

    ssim = luminance * contrast * structure

    return ssim.item()


def compute_temporal_consistency(video):
    """计算时间一致性"""
    frame_diff = torch.abs(video[:, :, 1:] - video[:, :, :-1])
    consistency = 1 - frame_diff.mean().item()
    return consistency


def train_generator(args):
    """训练生成器"""

    print("\n" + "="*60)
    print("Censor-G Generator Training")
    print("="*60)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # 创建数据集
    print("\n[1] Creating dataset...")
    if args.data_root and os.path.exists(args.data_root):
        dataset = CASME2RealDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            num_frames=args.num_frames,
        )
    else:
        print(f"  [Warning] Data root not found: {args.data_root}")
        print(f"  Using simulated dataset with {args.num_samples} samples")
        dataset = CASME2GeneratorDataset(
            data_root=None,
            num_samples=args.num_samples,
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
    print(f"  Batch size: {args.batch_size}")

    # 创建模型
    print("\n[2] Creating models...")
    generator = CensorGGenerator(
        num_au=17,
        num_keypoints=68,
        num_frames=args.num_frames,
        image_size=args.image_size,
    ).to(device)

    fomm_baseline = FOMMBaseline(
        num_keypoints=10,
        image_size=args.image_size,
    ).to(device)

    discriminator = VideoDiscriminator(
        num_frames=args.num_frames,
        image_size=args.image_size,
    ).to(device)

    print(f"  Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  FOMM params: {sum(p.numel() for p in fomm_baseline.parameters()):,}")

    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))
    fomm_optimizer = optim.Adam(fomm_baseline.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 损失函数
    l1_loss = nn.L1Loss()

    # 训练循环
    print("\n[3] Training...")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    training_log = {
        'config': vars(args),
        'epochs': [],
    }

    best_fid = float('inf')

    for epoch in range(args.epochs):
        epoch_log = {
            'epoch': epoch + 1,
            'g_loss': [],
            'd_loss': [],
            'fomm_loss': [],
            'fid': [],
            'ssim': [],
            'temporal': [],
        }

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:
            neutral_face = batch['neutral_face'].to(device)
            target_video = batch['target_video'].to(device)
            au_activation = batch['au_activation'].to(device)

            # === Censor-G Generator ===
            g_optimizer.zero_grad()

            generated_video, motion_fields = generator(neutral_face, au_activation)

            # L1重建损失
            recon_loss = l1_loss(generated_video, target_video)

            # GAN损失（可选）
            if args.use_gan:
                fake_logits = discriminator(generated_video)
                gan_loss = F.binary_cross_entropy_with_logits(
                    fake_logits, torch.ones_like(fake_logits)
                )
                g_loss = recon_loss + args.gan_weight * gan_loss
            else:
                g_loss = recon_loss

            g_loss.backward()
            g_optimizer.step()

            # === Discriminator（可选）===
            if args.use_gan:
                d_optimizer.zero_grad()

                real_logits = discriminator(target_video)
                fake_logits = discriminator(generated_video.detach())

                d_real_loss = F.binary_cross_entropy_with_logits(
                    real_logits, torch.ones_like(real_logits)
                )
                d_fake_loss = F.binary_cross_entropy_with_logits(
                    fake_logits, torch.zeros_like(fake_logits)
                )
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                d_optimizer.step()

            # === FOMM Baseline ===
            fomm_optimizer.zero_grad()

            fomm_video = fomm_baseline(neutral_face, au_activation)
            fomm_loss = l1_loss(fomm_video, target_video)

            fomm_loss.backward()
            fomm_optimizer.step()

            # === 记录 ===
            epoch_log['g_loss'].append(g_loss.item())
            if args.use_gan:
                epoch_log['d_loss'].append(d_loss.item())
            epoch_log['fomm_loss'].append(fomm_loss.item())

            # 计算指标
            ssim = compute_ssim(generated_video.mean(dim=2), target_video.mean(dim=2))
            temporal = compute_temporal_consistency(generated_video)

            epoch_log['ssim'].append(ssim)
            epoch_log['temporal'].append(temporal)

            # 更新进度条
            progress_bar.set_postfix({
                'g_loss': f'{g_loss.item():.4f}',
                'ssim': f'{ssim:.4f}',
            })

        # Epoch统计
        epoch_stats = {
            'epoch': epoch + 1,
            'g_loss_mean': np.mean(epoch_log['g_loss']),
            'ssim_mean': np.mean(epoch_log['ssim']),
            'temporal_mean': np.mean(epoch_log['temporal']),
            'fomm_loss_mean': np.mean(epoch_log['fomm_loss']),
        }

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    G Loss: {epoch_stats['g_loss_mean']:.4f}")
        print(f"    SSIM: {epoch_stats['ssim_mean']:.4f}")
        print(f"    Temporal: {epoch_stats['temporal_mean']:.4f}")
        print(f"    FOMM Loss: {epoch_stats['fomm_loss_mean']:.4f}")

        training_log['epochs'].append(epoch_stats)

        # 保存checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch_stats['ssim_mean'] > best_fid:
            checkpoint_path = os.path.join(
                args.save_dir, f'censor_g_gen_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'fomm': fomm_baseline.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'fomm_optimizer': fomm_optimizer.state_dict(),
                'stats': epoch_stats,
            }, checkpoint_path)
            print(f"    Saved: {checkpoint_path}")

            if epoch_stats['ssim_mean'] > best_fid:
                best_fid = epoch_stats['ssim_mean']
                best_path = os.path.join(args.save_dir, 'censor_g_gen_best.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'generator': generator.state_dict(),
                    'fomm': fomm_baseline.state_dict(),
                    'stats': epoch_stats,
                }, best_path)
                print(f"    [Best] Saved: {best_path}")

    # 保存训练日志
    log_path = os.path.join(args.log_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\n  Training log saved: {log_path}")

    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'censor_g_gen_final.pth')
    torch.save({
        'epoch': args.epochs,
        'generator': generator.state_dict(),
        'fomm': fomm_baseline.state_dict(),
    }, final_path)
    print(f"  Final model saved: {final_path}")

    print("\n" + "="*60)
    print("[Training Complete]")
    print("="*60)

    return generator, fomm_baseline, training_log


def main():
    parser = argparse.ArgumentParser(description='Train Censor-G Generator')

    # 数据参数
    parser.add_argument('--data_root', type=str, default=None,
                        help='CASME2 data root directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of simulated samples (if no real data)')

    # 模型参数
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_gan', action='store_true')
    parser.add_argument('--gan_weight', type=float, default=0.1)

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints/censor_g_gen')
    parser.add_argument('--log_dir', type=str, default='./logs/censor_g_gen')
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    train_generator(args)


if __name__ == '__main__':
    main()