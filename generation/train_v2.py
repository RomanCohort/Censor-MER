# =============================================================================
# Censor-G v2 Training Script
# =============================================================================
# 训练视觉中枢启发的微表情生成模型
#
# 训练策略：
#   1. 大部分参数基于神经科学手工设计（不需要训练）
#   2. 只训练运动场生成器和冲突解决网络
#   3. 训练时间：~3小时
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_v2 import CensorGv2, AU_INDEX, AU_INFO
from generation.generation_loss import MicroExpressionGenerationLoss, GANLoss, Discriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Train Censor-G v2')

    # Dataset
    parser.add_argument('--dataset', type=str, default='casme2')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/data/CASME2')

    # Model
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--threshold', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--trainable_only', action='store_true',
                        help='Only train trainable modules (field generators, conflict resolver)')

    # GAN
    parser.add_argument('--use_gan', action='store_true')

    # Saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints/censor_g_v2')
    parser.add_argument('--log_dir', type=str, default='./logs/censor_g_v2')

    return parser.parse_args()


def freeze_non_trainable(model):
    """
    冻结基于神经科学手工设计的参数。

    这些参数基于文献设计，不需要从数据学习：
      - V1 threshold
      - V2 协同/对抗矩阵的初始值
      - V3 时间参数的初始值
      - V4 区域参数
    """
    # V1: threshold是固定的
    # V2: 协同/对抗矩阵可以微调（可选）
    # V3: 时间参数可以微调（可选）
    # V4: 区域参数固定，field_generators需要训练
    # IT: conflict_resolver需要训练

    # 如果trainable_only=True，冻结大部分
    if hasattr(model, 'trainable_only') and model.trainable_only:
        # 只训练field_generators和conflict_resolver
        for name, param in model.named_parameters():
            if 'field_generators' not in name and 'conflict_resolver' not in name:
                param.requires_grad = False

    print("[Freeze] Trainable parameters:")
    trainable_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += param.numel()
            print(f"  {name}: {param.shape}")

    total_count = sum(p.numel() for p in model.parameters())
    print(f"[Freeze] Trainable: {trainable_count} / {total_count} ({100*trainable_count/total_count:.1f}%)")


def create_dummy_dataset(args):
    """创建dummy数据集用于测试。"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=200):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                'neutral_face': torch.randn(3, args.image_size, args.image_size),
                'au': torch.rand(17),
                'emotion': torch.randint(0, 4, (1,)).item(),
                'target_video': torch.randn(3, args.num_frames, args.image_size, args.image_size),
                'intensity': torch.rand(1).item() + 0.3,
            }

    return DummyDataset()


def train_v2(model, discriminator, dataloader, args, device):
    """
    训练Censor-G v2。

    训练流程：
      1. Phase 1: 训练运动场生成器（无GAN）
      2. Phase 2: 训练冲突解决网络
      3. Phase 3: GAN训练（可选）
    """
    print("\n[Training] Censor-G v2")
    print("="*50)

    # 冻结非训练参数
    freeze_non_trainable(model)

    # 只获取可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(trainable_params, lr=args.lr)

    if args.use_gan:
        d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5)
        gan_loss = GANLoss(gan_mode='standard')

    gen_loss_fn = MicroExpressionGenerationLoss()

    for epoch in range(args.epochs):
        g_loss_total = 0
        d_loss_total = 0

        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
            neutral_face = batch['neutral_face'].to(device)
            au = batch['au'].to(device)
            emotion = torch.tensor([batch['emotion']]).long().to(device) if isinstance(batch['emotion'], int) else batch['emotion'].to(device)
            target_video = batch['target_video'].to(device)
            intensity = batch['intensity'] if isinstance(batch['intensity'], float) else batch['intensity'].to(device)

            B = neutral_face.shape[0]

            # 调整intensity维度
            if isinstance(intensity, float):
                intensity = torch.tensor([intensity] * B).to(device)

            # =================================
            # Generator训练
            # =================================
            model.train()

            # 生成
            generated = model(neutral_face, au)

            # 计算损失
            losses, g_loss = gen_loss_fn(generated, target_video, au, au)

            if args.use_gan:
                discriminator.eval()
                d_fake = discriminator(generated)
                gan_g_loss = gan_loss(d_fake, True)
                g_loss += gan_g_loss

            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()

            g_loss_total += g_loss.item()

            # =================================
            # Discriminator训练
            # =================================
            if args.use_gan:
                model.eval()
                discriminator.train()

                with torch.no_grad():
                    generated = model(neutral_face, au)

                d_real = discriminator(target_video)
                d_loss_real = gan_loss(d_real, True)

                d_fake = discriminator(generated)
                d_loss_fake = gan_loss(d_fake, False)

                d_loss = (d_loss_real + d_loss_fake) / 2

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                d_loss_total += d_loss.item()

        # Epoch summary
        g_avg = g_loss_total / len(dataloader)
        d_avg = d_loss_total / len(dataloader) if args.use_gan else 0

        print(f"Epoch {epoch+1}: G Loss = {g_avg:.4f}, D Loss = {d_avg:.4f}")

        # Save
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'censor_g_v2_epoch_{epoch+1}.pth')
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'g_loss': g_avg,
            }, save_path)
            print(f"  Saved: {save_path}")

    print("[Training] Complete!")


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Setup] Device: {device}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create model
    print("[Model] Creating Censor-G v2...")
    model = CensorGv2(
        num_frames=args.num_frames,
        image_size=args.image_size,
        threshold=args.threshold,
    ).to(device)

    # Discriminator
    if args.use_gan:
        discriminator = Discriminator().to(device)
    else:
        discriminator = None

    # Dataset
    print("[Dataset] Loading...")
    dataset = create_dummy_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train
    train_v2(model, discriminator, dataloader, args, device)

    # Save final
    final_path = os.path.join(args.save_dir, 'censor_g_v2_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n[Complete] Saved: {final_path}")


if __name__ == '__main__':
    main()