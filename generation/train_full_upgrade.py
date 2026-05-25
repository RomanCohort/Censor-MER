# =============================================================================
# 完整升级版训练
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.full_upgrade_system import (
    FullDiffusionUNet,
    TemporalConstraint,
    ImprovedBlendshapeSystem,
    MultiDimensionalEvaluation,
)
from generation.train_rlhf import load_recognition_model
from data.casme2_real_loader import MultiDatasetGenerator


# =============================================================================
# 完整升级版训练器
# =============================================================================

def train_full_upgrade(args):
    """完整升级版训练"""

    print("\n" + "="*70)
    print("Full Upgrade Training: Target 70%+ Recognition Rate")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # === 1. 数据 ===
    print("\n[1] Loading Dataset")

    data_roots = {}
    for name, path in [('CASME2', args.casme2_root), ('SMIC', args.smic_root), ('SAMM', args.samm_root)]:
        if path and os.path.exists(path):
            data_roots[name] = path

    dataset = MultiDatasetGenerator(
        data_roots=data_roots,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Dataset: {len(dataset)} samples")

    # === 2. 模型 ===
    print("\n[2] Creating Full Upgrade Model")

    # 完整扩散模型
    diffusion = FullDiffusionUNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(8, 4),
        time_embed_dim=512,
        blendshape_dim=52,
    ).to(device)

    # 87%识别器
    recognizer = load_recognition_model(args.recognizer_checkpoint, num_classes=5).to(device)

    # 改进Blendshape
    blendshape_system = ImprovedBlendshapeSystem()

    print(f"Diffusion params: {sum(p.numel() for p in diffusion.parameters()):,}")
    print(f"Using 87% recognizer")
    print(f"Using improved blendshape mapping")

    # === 3. 训练 ===
    print("\n[3] Training")

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)

    # β schedule
    betas = torch.linspace(0.0001, 0.02, 1000)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    os.makedirs(args.save_dir, exist_ok=True)

    best_acc = 0
    training_log = {'epochs': []}

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        correct_count = 0
        total_count = 0
        ssim_sum = 0
        motion_sum = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            neutral = batch['neutral_face'].to(device)
            target = batch['target_video'].to(device)
            emotion_class = batch['emotion_class'].to(device)
            au = batch['au_activation'].to(device)

            B = neutral.shape[0]

            # Blendshape
            bs = blendshape_system.au_to_blendshape(au).to(device)

            # 扩展为视频
            neutral_video = neutral.unsqueeze(2).expand(B, 3, args.num_frames, args.image_size, args.image_size)

            # === 扩散损失 ===
            t = torch.randint(0, 1000, (B,))
            alpha_bar = alpha_bars[t].to(device).view(-1, 1, 1, 1, 1)

            noise = torch.randn_like(target)
            xt = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise

            noise_pred = diffusion(xt, t.to(device), bs)
            diff_loss = F.mse_loss(noise_pred, noise)

            # === 生成 ===
            with torch.no_grad():
                # 从噪声生成
                gen_noise = torch.randn(B, 3, args.num_frames, args.image_size, args.image_size, device=device)
                gen = diffusion(gen_noise, torch.zeros(B, dtype=torch.long, device=device), bs)

                # 识别
                rppg = torch.zeros_like(gen)
                gen_6ch = torch.cat([gen, rppg], dim=1)
                logits = recognizer(gen_6ch)
                probs = F.softmax(logits, dim=1)
                pred_class = probs.argmax(1)

            # === 时序损失 ===
            temporal_loss = TemporalConstraint.temporal_loss(gen)
            smoothness_loss = TemporalConstraint.smoothness_loss(gen)

            # === 总损失 ===
            loss = diff_loss + temporal_loss * 0.1 + smoothness_loss * 0.05

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()

            # 统计
            total_loss += loss.item()
            correct_count += (pred_class == emotion_class).sum().item()
            total_count += B

            # SSIM
            ssim = MultiDimensionalEvaluation.compute_ssim(gen, target)
            ssim_sum += ssim.item()

            # 运动幅度
            motion = TemporalConstraint.compute_motion_profile(gen)
            motion_sum += motion.mean().item()

        # Epoch统计
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_count / total_count
        avg_ssim = ssim_sum / len(dataloader)
        avg_motion = motion_sum / len(dataloader)

        print(f"Epoch {epoch}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  Motion: {avg_motion:.6f}")

        training_log['epochs'].append({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'ssim': avg_ssim,
            'motion': avg_motion,
        })

        # 保存最佳
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'diffusion': diffusion.state_dict(),
                'epoch': epoch,
                'accuracy': accuracy,
            }, os.path.join(args.save_dir, 'full_upgrade_best.pth'))
            print(f"  [Best] Saved! Acc={accuracy:.2%}")

        # 定期保存
        if epoch % 10 == 0:
            torch.save({
                'diffusion': diffusion.state_dict(),
            }, os.path.join(args.save_dir, f'full_epoch_{epoch}.pth'))

    # === 保存最终模型 ===
    torch.save({
        'diffusion': diffusion.state_dict(),
        'training_log': training_log,
    }, os.path.join(args.save_dir, 'full_upgrade_final.pth'))

    import json
    with open(os.path.join(args.save_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "="*70)
    print(f"Training Complete!")
    print(f"Best Accuracy: {best_acc:.2%}")
    print("="*70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--recognizer_checkpoint', default='./checkpoints/cross_casme2/cross_src_casme2_best.pth')
    parser.add_argument('--casme2_root', default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', default='/root/data/SAMM')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=64)

    parser.add_argument('--pretrain_epochs', type=int, default=20)
    parser.add_argument('--finetune_epochs', type=int, default=30)

    parser.add_argument('--save_dir', default='./checkpoints/full_upgrade')

    args = parser.parse_args()
    train_full_upgrade(args)


if __name__ == '__main__':
    main()