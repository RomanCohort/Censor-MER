# =============================================================================
# 调整版Hybrid训练：加强判别器
# =============================================================================
# 诊断发现：判别器太弱（只有14%准确率）
#
# 调整方案：
#   1. 使用87%识别器作为判别器（而不是随机初始化）
#   2. 增加判别器训练轮数
#   3. 判别器预训练阶段
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.train_hybrid import SimplifiedDiffusion
from generation.train_rlhf import load_recognition_model
from model.diffusion_blendshape import BlendshapeSystem
from data.casme2_real_loader import MultiDatasetGenerator


# =============================================================================
# 使用87%识别器作为判别器
# =============================================================================

class StrongDiscriminator(nn.Module):
    """强判别器：使用87%识别器"""

    def __init__(self, recognizer_checkpoint):
        super().__init__()

        # 加载87%识别器
        self.recognizer = load_recognition_model(recognizer_checkpoint, num_classes=5)

    def forward(self, video):
        """判别"""
        # 简化：只用3通道，添加3个零通道
        if video.shape[1] == 3:
            rppg = torch.zeros_like(video)
            video = torch.cat([video, rppg], dim=1)

        logits = self.recognizer(video)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def classify(self, video, expected_class):
        logits, probs = self.forward(video)
        correct_prob = probs.gather(1, expected_class.unsqueeze(1)).squeeze(1)
        predicted_class = probs.argmax(dim=1)
        return correct_prob, predicted_class, probs


# =============================================================================
# 调整版训练
# =============================================================================

def train_hybrid_improved(args):
    """调整版Hybrid训练"""

    print("\n" + "="*70)
    print("Hybrid Training (Improved): Strong Discriminator")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. 数据 ===
    print("\n[1] Loading Dataset")

    data_roots = {'CASME2': args.casme2_root}
    if args.smic_root:
        data_roots['SMIC'] = args.smic_root
    if args.samm_root:
        data_roots['SAMM'] = args.samm_root

    dataset = MultiDatasetGenerator(
        data_roots=data_roots,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"  Dataset: {len(dataset)} samples")

    # === 2. 模型 ===
    print("\n[2] Creating Models")

    # 扩散模型（使用已训练好的）
    diffusion = SimplifiedDiffusion().to(device)
    if args.diffusion_checkpoint:
        ckpt = torch.load(args.diffusion_checkpoint, map_location=device)
        diffusion.load_state_dict(ckpt['diffusion'])
        print(f"  Loaded diffusion from {args.diffusion_checkpoint}")
    else:
        print("  Using new diffusion model")

    # 强判别器（87%识别器）
    discriminator = StrongDiscriminator(args.recognizer_checkpoint).to(device)
    print(f"  Using 87% recognizer as discriminator")

    blendshape_system = BlendshapeSystem()

    # === 3. 训练 ===
    print("\n[3] Training")

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)
    training_log = {'epochs': []}

    best_acc = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        correct_count = 0
        total_count = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            neutral = batch['neutral_face'].to(device)
            target = batch['target_video'].to(device)
            emotion_class = batch['emotion_class'].to(device)
            au = batch['au_activation'].to(device)

            # Blendshape
            bs = blendshape_system.au_to_blendshape(au).to(device)

            # 扩散
            B = target.shape[0]
            t = torch.randint(0, 1000, (B,))
            xt, noise = diffusion.forward_diffusion(target, t)
            noise_pred = diffusion.denoise_step(xt, t, bs)
            diff_loss = F.mse_loss(noise_pred, noise)

            # 生成 + 判别
            neutral_video = neutral.unsqueeze(2).expand(B, 3, args.num_frames, args.image_size, args.image_size)
            gen = diffusion.generate(neutral_video, bs, num_steps=10)
            correct_prob, pred_class, probs = discriminator.classify(gen, emotion_class)

            # GAN损失
            gan_loss = -torch.log(correct_prob + 1e-8).mean()

            # 总损失
            loss = diff_loss + gan_loss * 0.3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_count += (pred_class == emotion_class).sum().item()
            total_count += B

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_count / total_count

        print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2%}")

        training_log['epochs'].append({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
        })

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'diffusion': diffusion.state_dict(),
                'epoch': epoch,
                'accuracy': accuracy,
            }, os.path.join(args.save_dir, 'hybrid_improved_best.pth'))
            print(f"    [Best] Saved, Acc={accuracy:.2%}")

    # 保存最终模型
    torch.save({
        'diffusion': diffusion.state_dict(),
    }, os.path.join(args.save_dir, 'hybrid_improved_final.pth'))

    print("\n" + "="*70)
    print(f"Training Complete! Best Acc: {best_acc:.2%}")
    print("="*70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--casme2_root', default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', default='/root/data/SAMM')

    parser.add_argument('--diffusion_checkpoint',
                        default='./checkpoints/hybrid_model_v2/hybrid_final.pth')
    parser.add_argument('--recognizer_checkpoint',
                        default='./checkpoints/cross_casme2/cross_src_casme2_best.pth')

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--save_dir', default='./checkpoints/hybrid_improved')

    args = parser.parse_args()
    train_hybrid_improved(args)


if __name__ == '__main__':
    main()