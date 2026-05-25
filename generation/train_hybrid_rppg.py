# =============================================================================
# Hybrid + rPPG估计：扩散 + Blendshape + 87%识别器（无零填充）
# =============================================================================
#
# 这是最有潜力突破70%的方案：
#   1. 扩散模型：高质量生成
#   2. Blendshape：精确控制
#   3. rPPG估计：真实心率信号，不破坏87%识别器
#   4. 87%识别器：强监督
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.train_hybrid import SimplifiedDiffusion
from generation.train_rlhf import load_recognition_model
from model.diffusion_blendshape import BlendshapeSystem
from data.casme2_real_loader import MultiDatasetGenerator


# =============================================================================
# rPPG估计器
# =============================================================================

class RPPGEstimator(nn.Module):
    """
    从RGB视频估计rPPG信号

    CHROM方法：
      - 提取面部颜色变化
      - 组合RGB得到心率相关信号
    """

    def __init__(self):
        super().__init__()

        # 面部区域检测（简化版）
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(3, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 8, 8)),  # 保留时间维度
        )

        # 时序特征
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(16 * 64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # rPPG信号生成
        self.rppg_decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 输出3通道rPPG
        )

    def forward(self, video):
        """
        Args:
            video: (B, 3, T, H, W) RGB视频

        Returns:
            rppg: (B, 3, T, H, W) rPPG信号
        """
        B, C, T, H, W = video.shape

        # 方法1：CHROM chrominance方法（更准确）
        # 取面部中心区域
        face_region = video[:, :, :, H//4:3*H//4, W//4:3*W//4]

        # 计算每帧平均颜色
        mean_color = face_region.mean(dim=[3, 4])  # (B, 3, T)

        # CHROM计算
        R = mean_color[:, 0]
        G = mean_color[:, 1]
        B_ch = mean_color[:, 2]

        # Chrominance组合
        Xs = 3 * R - 2 * G
        Ys = 1.5 * R + G - 1.5 * B_ch

        # rPPG信号
        chrom = Xs - Ys  # (B, T)

        # 归一化到0-1
        chrom = (chrom - chrom.min(dim=1, keepdim=True)[0]) / \
                (chrom.max(dim=1, keepdim=True)[0] - chrom.min(dim=1, keepdim=True)[0] + 1e-8)

        # 扩展为3通道（使用不同的相位）
        rppg_ch1 = chrom  # 基础信号
        rppg_ch2 = torch.roll(chrom, shifts=2, dims=1)  # 相移
        rppg_ch3 = torch.roll(chrom, shifts=-2, dims=1)

        rppg = torch.stack([rppg_ch1, rppg_ch2, rppg_ch3], dim=1)  # (B, 3, T)

        # 扩展到空间维度
        rppg = rppg.unsqueeze(3).unsqueeze(4).expand(B, 3, T, H, W)

        return rppg


class ImprovedRPPGEstimator(nn.Module):
    """改进版rPPG估计器"""

    def __init__(self, hidden_dim=128):
        super().__init__()

        # 空间特征
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # 时序编码
        self.temporal_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # rPPG解码
        self.decoder = nn.Linear(hidden_dim, 3)

    def forward(self, video):
        B, C, T, H, W = video.shape

        # 对每帧编码
        frame_features = []
        for t in range(T):
            frame = video[:, :, t]  # (B, 3, H, W)
            feat = self.spatial_encoder(frame)  # (B, 64)
            frame_features.append(feat)

        frame_features = torch.stack(frame_features, dim=2)  # (B, 64, T)

        # 时序编码
        temporal_feat = self.temporal_encoder(frame_features.permute(0, 2, 1))  # (B, T, hidden)

        # 生成rPPG
        rppg_per_frame = self.decoder(temporal_feat)  # (B, T, 3)
        rppg = rppg_per_frame.permute(0, 2, 1)  # (B, 3, T)

        # 归一化
        rppg = torch.sigmoid(rppg)  # (B, 3, T)

        # 扩展到空间
        rppg = rppg.unsqueeze(3).unsqueeze(4).expand(B, 3, T, H, W)

        return rppg


# =============================================================================
# Hybrid + rPPG训练
# =============================================================================

def train_hybrid_rppg(args):
    """Hybrid + rPPG估计训练"""

    print("\n" + "="*70)
    print("Hybrid + rPPG Estimation: Diffusion + Blendshape + 87% Recognizer")
    print("Target: 50-70% (Most Promising Method)")
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

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} samples")

    # === 2. 模型 ===
    print("\n[2] Creating Models")

    # 扩散模型（继承已训练的）
    diffusion = SimplifiedDiffusion(
        num_frames=args.num_frames,
        image_size=args.image_size,
    ).to(device)

    if args.diffusion_checkpoint:
        ckpt = torch.load(args.diffusion_checkpoint, map_location=device)
        diffusion.load_state_dict(ckpt.get('diffusion', ckpt))
        print(f"Loaded diffusion from {args.diffusion_checkpoint}")

    # Blendshape系统
    blendshape_system = BlendshapeSystem()
    print("Blendshape system ready")

    # rPPG估计器
    rppg_estimator = ImprovedRPPGEstimator(hidden_dim=128).to(device)
    print("rPPG estimator created")

    # 87%识别器
    recognizer = load_recognition_model(args.recognizer_checkpoint, num_classes=5).to(device)
    print("87% recognizer loaded")

    print(f"Diffusion params: {sum(p.numel() for p in diffusion.parameters()):,}")
    print(f"rPPG estimator params: {sum(p.numel() for p in rppg_estimator.parameters()):,}")

    # === 3. 训练 ===
    print("\n[3] Training")

    optimizer_diff = torch.optim.Adam(diffusion.parameters(), lr=args.lr)
    optimizer_rppg = torch.optim.Adam(rppg_estimator.parameters(), lr=args.lr * 2)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0
    training_log = {'epochs': []}

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        correct_count = 0
        total_count = 0

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
            alpha_bar = diffusion.alpha_bars[t].to(device).view(-1, 1, 1, 1, 1)

            noise = torch.randn_like(target)
            xt = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise

            noise_pred = diffusion.denoise_step(xt, t.to(device), bs)
            diff_loss = F.mse_loss(noise_pred, noise)

            # === 生成 + rPPG估计 ===
            with torch.no_grad():
                gen_noise = torch.randn(B, 3, args.num_frames, args.image_size, args.image_size, device=device)
                gen = diffusion.generate(neutral_video, bs, num_steps=10)

            # 估计rPPG
            gen_rppg = rppg_estimator(gen)

            # 合成6通道
            gen_6ch = torch.cat([gen, gen_rppg], dim=1)

            # 识别器评估
            logits = recognizer(gen_6ch)
            probs = F.softmax(logits, dim=1)
            pred_class = probs.argmax(1)

            # 正确概率
            correct_prob = probs.gather(1, emotion_class.unsqueeze(1)).squeeze(1)

            # 生成器损失
            gen_loss = -torch.log(correct_prob + 1e-8).mean()

            # rPPG损失：让估计的rPPG接近真实视频的rPPG
            with torch.no_grad():
                target_rppg = rppg_estimator(target)

            rppg_loss = F.mse_loss(gen_rppg, target_rppg)

            # 总损失
            total = diff_loss + gen_loss * 0.5 + rppg_loss * 0.1

            optimizer_diff.zero_grad()
            optimizer_rppg.zero_grad()
            total.backward()
            optimizer_diff.step()
            optimizer_rppg.step()

            total_loss += total.item()
            correct_count += (pred_class == emotion_class).sum().item()
            total_count += B

        # Epoch统计
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_count / total_count

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2%}")

        training_log['epochs'].append({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
        })

        # 保存最佳
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'diffusion': diffusion.state_dict(),
                'rppg_estimator': rppg_estimator.state_dict(),
                'accuracy': accuracy,
            }, os.path.join(args.save_dir, 'hybrid_rppg_best.pth'))
            print(f"  [Best] Saved! Acc={accuracy:.2%}")

        if epoch % 10 == 0:
            torch.save({
                'diffusion': diffusion.state_dict(),
                'rppg_estimator': rppg_estimator.state_dict(),
            }, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))

    # === 保存最终 ===
    import json
    torch.save({
        'diffusion': diffusion.state_dict(),
        'rppg_estimator': rppg_estimator.state_dict(),
        'training_log': training_log,
    }, os.path.join(args.save_dir, 'hybrid_rppg_final.pth'))

    with open(os.path.join(args.save_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "="*70)
    print(f"Hybrid + rPPG Complete!")
    print(f"Best Accuracy: {best_acc:.2%}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--diffusion_checkpoint', default='./checkpoints/hybrid_model_v2/hybrid_final.pth')
    parser.add_argument('--recognizer_checkpoint', default='./checkpoints/cross_casme2/cross_src_casme2_best.pth')
    parser.add_argument('--casme2_root', default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', default='/root/data/SAMM')

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=64)

    parser.add_argument('--save_dir', default='./checkpoints/hybrid_rppg')

    args = parser.parse_args()
    train_hybrid_rppg(args)


if __name__ == '__main__':
    main()