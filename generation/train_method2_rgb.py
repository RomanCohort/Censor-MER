# =============================================================================
# 方案2：使用3通道识别器（不使用零填充）
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
from data.casme2_real_loader import MultiDatasetGenerator, CASME2_EMOTION_MAPPING


# =============================================================================
# 3通道识别器
# =============================================================================

class RGBRecognizer(nn.Module):
    """3通道RGB识别器"""

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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, video):
        feat = self.features(video)
        logits = self.classifier(feat)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def classify(self, video, expected_class):
        logits, probs = self.forward(video)
        correct_prob = probs.gather(1, expected_class.unsqueeze(1)).squeeze(1)
        predicted_class = probs.argmax(dim=1)
        return correct_prob, predicted_class, probs


# =============================================================================
# 方案2训练
# =============================================================================

def train_method2(args):
    """方案2：使用3通道识别器"""

    print("\n" + "="*70)
    print("Method 2: 3-Channel RGB Recognizer (No Zero Padding)")
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

    # 3通道识别器（先预训练）
    recognizer = RGBRecognizer(num_classes=5).to(device)

    print("Training RGB recognizer on real data first...")

    # 预训练识别器
    recognizer_optimizer = torch.optim.Adam(recognizer.parameters(), lr=1e-3)

    for epoch in range(10):
        correct = 0
        total = 0

        for batch in dataloader:
            target = batch['target_video'].to(device)
            emotion_class = batch['emotion_class'].to(device)

            logits, probs = recognizer(target)
            loss = F.cross_entropy(logits, emotion_class)

            recognizer_optimizer.zero_grad()
            loss.backward()
            recognizer_optimizer.step()

            correct += (probs.argmax(1) == emotion_class).sum().item()
            total += emotion_class.shape[0]

        acc = correct / total
        print(f"  Recognizer Epoch {epoch}: {acc:.2%}")

    print(f"RGB Recognizer trained: ~{acc:.2%}")

    # === 3. 生成器训练 ===
    print("\nTraining generator...")

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        correct_count = 0
        total_count = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            neutral = batch['neutral_face'].to(device)
            emotion_class = batch['emotion_class'].to(device)
            au = batch['au_activation'].to(device)

            # 生成
            gen_video, _ = generator(neutral, au)

            # 识别器评估（直接3通道，无需零填充）
            correct_prob, pred_class, _ = recognizer.classify(gen_video, emotion_class)

            # 损失
            loss = -torch.log(correct_prob + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_count += (pred_class == emotion_class).sum().item()
            total_count += emotion_class.shape[0]

        accuracy = correct_count / total_count
        print(f"Epoch {epoch}: Acc={accuracy:.2%}")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({'generator': generator.state_dict()},
                       os.path.join(args.save_dir, 'method2_best.pth'))

    print("\n" + "="*70)
    print(f"Method 2 Complete! Best: {best_acc:.2%}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_checkpoint', default='./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth')
    parser.add_argument('--casme2_root', default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--smic_root', default='/root/SMIC_all_cropped')
    parser.add_argument('--samm_root', default='/root/data/SAMM')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', default='./checkpoints/method2')
    args = parser.parse_args()
    train_method2(args)


if __name__ == '__main__':
    main()