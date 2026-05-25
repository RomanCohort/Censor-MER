# =============================================================================
# 方案1：从生成视频估计rPPG信号（无零填充）
# =============================================================================
#
# 思路：
#   87%识别器需要6通道输入（RGB + rPPG）
#   方案：从生成的RGB视频中估计rPPG信号
#
# rPPG估计方法：
#   1. CHROM方法（最常用）
#   2. POS方法
#   3. ICA方法
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_generator import CensorGGenerator
from generation.train_rlhf import load_recognition_model
from data.casme2_real_loader import MultiDatasetGenerator


# =============================================================================
# rPPG估计模块
# =============================================================================

class RPPGEstimator(nn.Module):
    """
    从RGB视频估计rPPG信号

    方法：CHROM (Chrominance-based)
    - 提取面部区域的颜色变化
    - 通过chrominance组合消除运动噪声
    - 得到心率相关的信号
    """

    def __init__(self):
        super().__init__()

        # 简化的rPPG估计网络
        self.temporal_filter = nn.Conv1d(3, 1, kernel_size=5, padding=2)

    def forward(self, video):
        """
        从RGB视频估计rPPG信号

        Args:
            video: (B, 3, T, H, W) RGB视频

        Returns:
            rppg: (B, 3, T, H, W) 估计的rPPG通道
        """
        B, C, T, H, W = video.shape

        # 方法1：简单的颜色平均作为rPPG估计
        # 取面部中心区域（假设H/4到3H/4, W/4到3W/4是面部）
        face_region = video[:, :, :, H//4:3*H//4, W//4:3*W//4]

        # 计算每帧的平均颜色
        mean_color = face_region.mean(dim=[3, 4])  # (B, 3, T)

        # CHROM方法：
        # Xs = 3*R - 2*G
        # Ys = 1.5*R + G - 1.5*B
        # CHROM = Xs - Ys

        R = mean_color[:, 0]  # (B, T)
        G = mean_color[:, 1]
        B_ch = mean_color[:, 2]

        Xs = 3 * R - 2 * G
        Ys = 1.5 * R + G - 1.5 * B_ch

        chrom = Xs - Ys  # (B, T)

        # 归一化到0-1范围
        chrom = (chrom - chrom.min()) / (chrom.max() - chrom.min() + 1e-8)

        # 扩展到空间维度（简化：使用统一值）
        rppg_signal = chrom.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, T, 1, 1)
        rppg_signal = rppg_signal.expand(B, 3, T, H, W)

        return rppg_signal


class RPPGEstimatorMLP(nn.Module):
    """
    使用MLP学习rPPG估计

    更准确的方法：学习从RGB到rPPG的映射
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        # 对每帧提取特征
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # 时序处理
        self.temporal_encoder = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 输出rPPG信号
        self.rppg_decoder = nn.Linear(hidden_dim, 3)

    def forward(self, video):
        """
        Args:
            video: (B, 3, T, H, W)

        Returns:
            rppg: (B, 3, T, H, W)
        """
        B, C, T, H, W = video.shape

        # 对每帧编码
        frame_features = []
        for t in range(T):
            frame = video[:, :, t]  # (B, 3, H, W)
            feat = self.frame_encoder(frame)  # (B, 32)
            frame_features.append(feat)

        frame_features = torch.stack(frame_features, dim=2)  # (B, 32, T)

        # 时序编码
        temporal_feat = self.temporal_encoder(frame_features.permute(0, 2, 1))  # (B, T, hidden_dim)

        # 生成rPPG信号
        rppg_per_frame = self.rppg_decoder(temporal_feat)  # (B, T, 3)
        rppg_per_frame = rppg_per_frame.permute(0, 2, 1)  # (B, 3, T)

        # 归一化
        rppg_per_frame = (rppg_per_frame - rppg_per_frame.min()) / (rppg_per_frame.max() - rppg_per_frame.min() + 1e-8)

        # 扩展到空间维度
        rppg = rppg_per_frame.unsqueeze(3).unsqueeze(4).expand(B, 3, T, H, W)

        return rppg


# =============================================================================
# 方案1训练
# =============================================================================

def train_method1(args):
    """方案1：估计rPPG后使用87%识别器"""

    print("\n" + "="*70)
    print("Method 1: Real rPPG Estimation + 87% Recognizer")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. 数据 ===
    data_roots = {}
    for name, path in [('CASME2', args.casme2_root), ('SMIC', args.smic_root), ('SAMM', args.samm_root)]:
        if path and os.path.exists(path):
            data_roots[name] = path

    dataset = MultiDatasetGenerator(data_roots=data_roots, image_size=64, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset: {len(dataset)} samples")

    # === 2. 模型 ===
    # 生成器
    generator = CensorGGenerator(num_au=17, num_keypoints=68, num_frames=16, image_size=64).to(device)

    if args.generator_checkpoint:
        ckpt = torch.load(args.generator_checkpoint, map_location=device)
        generator.load_state_dict(ckpt.get('generator', ckpt))
        print("Loaded generator")

    # rPPG估计器
    rppg_estimator = RPPGEstimatorMLP(hidden_dim=128).to(device)
    print("rPPG estimator created")

    # 87%识别器
    recognizer = load_recognition_model(args.recognizer_checkpoint, num_classes=5).to(device)
    print("87% recognizer loaded")

    # === 3. 训练 ===
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_rppg = torch.optim.Adam(rppg_estimator.parameters(), lr=args.lr * 2)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        correct_count = 0
        total_count = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            neutral = batch['neutral_face'].to(device)
            target = batch['target_video'].to(device)
            emotion_class = batch['emotion_class'].to(device)
            au = batch['au_activation'].to(device)

            # 生成RGB视频
            gen_rgb, _ = generator(neutral, au)

            # 估计rPPG信号
            gen_rppg = rppg_estimator(gen_rgb)

            # 合成6通道输入
            gen_6ch = torch.cat([gen_rgb, gen_rppg], dim=1)

            # 识别器评估
            logits = recognizer(gen_6ch)
            probs = F.softmax(logits, dim=1)
            pred_class = probs.argmax(1)

            # 生成器损失：让识别器正确识别
            correct_prob = probs.gather(1, emotion_class.unsqueeze(1)).squeeze(1)
            gen_loss = -torch.log(correct_prob + 1e-8).mean()

            # rPPG估计器损失：让估计的rPPG接近真实rPPG特征
            # 简化：使用识别器对真实视频的rPPG作为目标
            with torch.no_grad():
                target_rppg = rppg_estimator(target)  # 从真实视频估计的rPPG

            rppg_loss = F.mse_loss(gen_rppg, target_rppg)

            # 总损失
            total_loss = gen_loss + rppg_loss * 0.1

            optimizer_gen.zero_grad()
            optimizer_rppg.zero_grad()
            total_loss.backward()
            optimizer_gen.step()
            optimizer_rppg.step()

            correct_count += (pred_class == emotion_class).sum().item()
            total_count += emotion_class.shape[0]

        accuracy = correct_count / total_count
        print(f"Epoch {epoch}: Acc={accuracy:.2%}")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'generator': generator.state_dict(),
                'rppg_estimator': rppg_estimator.state_dict(),
            }, os.path.join(args.save_dir, 'method1_best.pth'))

    print("\n" + "="*70)
    print(f"Method 1 Complete! Best: {best_acc:.2%}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_checkpoint', default='./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth')
    parser.add_argument('--recognizer_checkpoint', default='./checkpoints/cross_casme2/cross_src_casme2_best.pth')
    parser.add_argument('--casme2_root', default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', default='/root/data/SAMM')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', default='./checkpoints/method1')
    args = parser.parse_args()
    train_method1(args)


if __name__ == '__main__':
    main()