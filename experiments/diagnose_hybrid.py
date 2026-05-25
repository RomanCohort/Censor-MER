#!/usr/bin/env python3
"""
诊断Hybrid模型问题

检查：
  1. 数据分布（各情感类别比例）
  2. Blendshape映射（是否正确）
  3. 判别器能力（能否区分各情感）
  4. 训练日志（各阶段Loss变化）
"""

import torch
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.casme2_real_loader import MultiDatasetGenerator, CASME2_EMOTION_MAPPING
from model.diffusion_blendshape import BlendshapeSystem
from generation.train_hybrid import SimplifiedDiffusion, SimpleDiscriminator


def diagnose():
    print("\n" + "="*70)
    print("Hybrid Model Diagnosis")
    print("="*70)

    # === 1. 数据分布检查 ===
    print("\n[1] Dataset Distribution Check")

    dataset = MultiDatasetGenerator(
        data_roots={'CASME2': '/root/autodl-tmp/data/CASME2'},
        image_size=64, num_frames=16
    )

    emotion_count = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        emotion = sample['emotion_name']
        emotion_count[emotion] = emotion_count.get(emotion, 0) + 1

    print(f"  Total samples: {len(dataset)}")
    print(f"  Emotion distribution:")
    for emotion, count in sorted(emotion_count.items(), key=lambda x: -x[1]):
        pct = count / len(dataset) * 100
        print(f"    {emotion}: {count} ({pct:.1f}%)")

    # === 2. Blendshape映射检查 ===
    print("\n[2] Blendshape Mapping Check")

    bs_system = BlendshapeSystem()

    emotions = ['happiness', 'surprise', 'disgust', 'repression']

    for emotion in emotions:
        bs_low = bs_system.get_emotion_blendshape(emotion, 0.4)
        bs_high = bs_system.get_emotion_blendshape(emotion, 0.8)

        # 找到非零的blendshapes
        nonzero_low = [(i, v) for i, v in enumerate(bs_low.tolist()) if v > 0.1]
        nonzero_high = [(i, v) for i, v in enumerate(bs_high.tolist()) if v > 0.1]

        print(f"\n  {emotion}:")
        print(f"    Low intensity (0.4): {len(nonzero_low)} blendshapes")
        for idx, val in nonzero_low[:3]:
            print(f"      [{idx}] = {val:.2f}")

        print(f"    High intensity (0.8): {len(nonzero_high)} blendshapes")
        for idx, val in nonzero_high[:3]:
            print(f"      [{idx}] = {val:.2f}")

    # === 3. 判别器能力检查 ===
    print("\n[3] Discriminator Capability Check")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载判别器
    checkpoint = torch.load('./checkpoints/hybrid_model_v2/hybrid_final.pth', map_location=device)

    discriminator = SimpleDiscriminator(num_classes=5).to(device)
    discriminator.load_state_dict(checkpoint['discriminator'])

    # 测试判别器对真实视频的识别能力
    print(f"\n  Testing discriminator on real CASME2 videos:")

    emotion_correct = {}
    emotion_total = {}

    for i in range(50):
        sample = dataset[i]
        video = sample['target_video'].unsqueeze(0).to(device)
        emotion = sample['emotion_name']
        expected_class = CASME2_EMOTION_MAPPING.get(emotion, 0)

        with torch.no_grad():
            logits, probs = discriminator(video)
            predicted_class = probs.argmax(1).item()

        emotion_total[emotion] = emotion_total.get(emotion, 0) + 1
        if predicted_class == expected_class:
            emotion_correct[emotion] = emotion_correct.get(emotion, 0) + 1

    print(f"\n  Discriminator accuracy on real videos:")
    for emotion in emotion_total:
        correct = emotion_correct.get(emotion, 0)
        total = emotion_total[emotion]
        acc = correct / total * 100
        print(f"    {emotion}: {correct}/{total} = {acc:.1f}%")

    # === 4. 训练日志检查 ===
    print("\n[4] Training Log Check")

    log_path = './logs/hybrid_model_v2/hybrid_training_log.json'
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)

        print(f"\n  Stage 1 (Diffusion Pretraining):")
        if log.get('stage1'):
            losses = [e['loss'] for e in log['stage1']]
            print(f"    Start loss: {losses[0]:.4f}")
            print(f"    End loss: {losses[-1]:.4f}")
            print(f"    Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

        print(f"\n  Stage 2 (GAN Finetuning):")
        if log.get('stage2'):
            g_losses = [e['g_loss'] for e in log['stage2']]
            d_losses = [e['d_loss'] for e in log['stage2']]
            print(f"    G loss: {g_losses[0]:.4f} -> {g_losses[-1]:.4f}")
            print(f"    D loss: {d_losses[0]:.4f} -> {d_losses[-1]:.4f}")

        print(f"\n  Stage 3 (Joint Training):")
        if log.get('stage3'):
            losses = [e['loss'] for e in log['stage3']]
            print(f"    Start loss: {losses[0]:.4f}")
            print(f"    End loss: {losses[-1]:.4f}")

    # === 5. 诊断总结 ===
    print("\n" + "="*70)
    print("Diagnosis Summary")
    print("="*70)

    # 判别器能力
    disc_avg_acc = sum(emotion_correct.values()) / sum(emotion_total.values()) * 100
    print(f"\n  Discriminator average accuracy: {disc_avg_acc:.1f}%")
    print(f"    - If <50%: Discriminator too weak")
    print(f"    - If >80%: Discriminator OK, problem in generator")

    # 数据平衡
    max_count = max(emotion_count.values())
    min_count = min(emotion_count.values())
    imbalance_ratio = max_count / min_count
    print(f"\n  Data imbalance ratio: {imbalance_ratio:.1f}")
    print(f"    - If >3: Need data balancing")

    print("\n" + "="*70)


if __name__ == '__main__':
    diagnose()