# =============================================================================
# 改进版RLHF训练：使用87%识别器作为奖励模型
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

from model.censor_g_generator import CensorGGenerator
from generation.train_rlhf import load_recognition_model
from data.casme2_real_loader import MultiDatasetGenerator


# =============================================================================
# 强奖励模型（87%识别器）
# =============================================================================

class StrongRewardModel(nn.Module):
    """使用87%识别器作为奖励模型"""

    def __init__(self, recognizer_checkpoint):
        super().__init__()
        self.recognizer = load_recognition_model(recognizer_checkpoint, num_classes=5)

    def forward(self, video):
        if video.shape[1] == 3:
            rppg = torch.zeros_like(video)
            video = torch.cat([video, rppg], dim=1)
        logits = self.recognizer(video)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def compute_reward(self, video, expected_class):
        logits, probs = self.forward(video)
        correct_prob = probs.gather(1, expected_class.unsqueeze(1)).squeeze(1)
        confidence = probs.max(dim=1)[0]
        reward = correct_prob * 2.0 + confidence * 0.5
        return reward


# =============================================================================
# 改进版RLHF训练
# =============================================================================

def train_rlhf_improved(args):
    print("\n" + "="*60)
    print("Improved RLHF Training with 87% Reward Model")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 数据 ===
    data_roots = {}
    for name, path in [('CASME2', args.casme2_root), ('SMIC', args.smic_root), ('SAMM', args.samm_root)]:
        if path and os.path.exists(path):
            data_roots[name] = path

    dataset = MultiDatasetGenerator(data_roots=data_roots, image_size=64, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset: {len(dataset)} samples")

    # === 模型 ===
    generator = CensorGGenerator(num_au=17, num_keypoints=68, num_frames=16, image_size=64).to(device)

    if args.generator_checkpoint:
        ckpt = torch.load(args.generator_checkpoint, map_location=device)
        generator.load_state_dict(ckpt.get('generator', ckpt))
        print(f"Loaded generator")

    # 强奖励模型（87%识别器）
    reward_model = StrongRewardModel(args.recognizer_checkpoint).to(device)
    print(f"Using 87% recognizer as reward model")

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    # === 训练 ===
    best_reward = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        total_reward = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            neutral = batch['neutral_face'].to(device)
            emotion_class = batch['emotion_class'].to(device)
            au = batch['au_activation'].to(device)

            # 生成
            gen_video, _ = generator(neutral, au)

            # 计算奖励
            reward = reward_model.compute_reward(gen_video, emotion_class)

            # 损失 = 最小化负奖励
            loss = -reward.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reward += reward.mean().item()

        avg_loss = total_loss / len(dataloader)
        avg_reward = total_reward / len(dataloader)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({'generator': generator.state_dict()},
                       os.path.join(args.save_dir, 'rlhf_improved_best.pth'))

        if epoch % 10 == 0:
            torch.save({'generator': generator.state_dict()},
                       os.path.join(args.save_dir, f'rlhf_epoch_{epoch}.pth'))

    torch.save({'generator': generator.state_dict()},
               os.path.join(args.save_dir, 'rlhf_improved_final.pth'))

    print("\n" + "="*60)
    print(f"RLHF Improved Complete! Best Reward: {best_reward:.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_checkpoint', default='./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth')
    parser.add_argument('--recognizer_checkpoint', default='./checkpoints/cross_casme2/cross_src_casme2_best.pth')
    parser.add_argument('--casme2_root', default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', default='/root/data/SAMM')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', default='./checkpoints/rlhf_improved')
    args = parser.parse_args()
    train_rlhf_improved(args)


if __name__ == '__main__':
    main()