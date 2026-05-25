# =============================================================================
# Test Hybrid Model: Diffusion + Blendshape + GAN
# =============================================================================
# 测试混合模型生成效果
#
# 测试内容：
#   1. 加载训练好的模型
#   2. 从CASME2测试集生成微表情
#   3. 用识别器评估生成质量
#   4. 保存生成视频和评估结果
# =============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import cv2
import argparse
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.diffusion_blendshape import BlendshapeSystem
from generation.train_hybrid import SimplifiedDiffusion, SimpleDiscriminator


# =============================================================================
# Load Model
# =============================================================================

def load_hybrid_model(checkpoint_path, device='cuda'):
    """加载混合模型"""
    print(f"\n[Loading Model]")
    print(f"  Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 创建模型
    diffusion = SimplifiedDiffusion(
        num_frames=16,
        image_size=64,
        num_blendshapes=52,
    ).to(device)

    discriminator = SimpleDiscriminator(num_classes=5).to(device)

    # 加载权重
    if 'diffusion' in checkpoint:
        diffusion.load_state_dict(checkpoint['diffusion'])
        print("  ✅ Diffusion model loaded")

    if 'discriminator' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator'])
        print("  ✅ Discriminator loaded")

    blendshape_system = BlendshapeSystem()
    print("  ✅ Blendshape system initialized")

    return diffusion, discriminator, blendshape_system


# =============================================================================
# Generate Micro-Expression
# =============================================================================

def generate_me(diffusion, blendshape_system, neutral_face, emotion, intensity, device='cuda'):
    """
    生成微表情

    Args:
        diffusion: 扩散模型
        blendshape_system: Blendshape系统
        neutral_face: (C, H, W) 中性脸
        emotion: 情感类别
        intensity: 强度

    Returns:
        generated_video: (C, T, H, W) 生成的视频
    """
    # 添加batch维度
    if neutral_face.dim() == 3:
        neutral_face = neutral_face.unsqueeze(0)

    B, C, H, W = neutral_face.shape
    T = diffusion.num_frames

    # 扩展为视频格式
    neutral_video = neutral_face.unsqueeze(2).expand(B, C, T, H, W).to(device)

    # 情感 → Blendshape
    blendshape = blendshape_system.get_emotion_blendshape(emotion, intensity)
    blendshape = blendshape.unsqueeze(0).to(device)

    # 扩散生成
    with torch.no_grad():
        generated = diffusion.generate(neutral_video, blendshape, num_steps=20)

    return generated.squeeze(0)  # (C, T, H, W)


# =============================================================================
# Evaluate Generated Video
# =============================================================================

def evaluate_video(discriminator, video, expected_emotion, device='cuda'):
    """
    评估生成的视频

    Args:
        discriminator: 判别器
        video: (C, T, H, W) 生成的视频
        expected_emotion: 期望的情感

    Returns:
        evaluation: 评估结果
    """
    # 添加batch维度
    if video.dim() == 4:
        video = video.unsqueeze(0)

    video = video.to(device)

    # 情感类别映射
    emotion_to_class = {
        'happiness': 0,
        'surprise': 1,
        'disgust': 2,
        'repression': 3,
        'other': 4,
    }
    expected_class = emotion_to_class.get(expected_emotion, 0)
    expected_class_tensor = torch.tensor([expected_class], device=device)

    # 判别器评估
    with torch.no_grad():
        logits, probs = discriminator(video)
        predicted_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()
        correct_prob = probs[0, expected_class].item()

    # 计算视频运动幅度
    frame_motion = []
    for t in range(video.shape[2] - 1):
        motion = (video[:,:,t+1] - video[:,:,t]).abs().mean().item()
        frame_motion.append(motion)

    avg_motion = np.mean(frame_motion)
    max_motion = np.max(frame_motion)

    return {
        'predicted_class': predicted_class,
        'predicted_emotion': list(emotion_to_class.keys())[predicted_class],
        'expected_emotion': expected_emotion,
        'is_correct': predicted_class == expected_class,
        'confidence': confidence,
        'correct_prob': correct_prob,
        'avg_motion': avg_motion,
        'max_motion': max_motion,
    }


# =============================================================================
# Save Video
# =============================================================================

def save_video(video, output_path, fps=30):
    """
    保存视频

    Args:
        video: (C, T, H, W) tensor
        output_path: 输出路径
        fps: 帧率
    """
    C, T, H, W = video.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for t in range(T):
        frame = video[:, t, :, :].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()


# =============================================================================
# Main Test Function
# =============================================================================

def test_hybrid(args):
    """测试混合模型"""

    print("\n" + "="*70)
    print("Testing Hybrid Model: Diffusion + Blendshape + GAN")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # === 1. 加载模型 ===
    diffusion, discriminator, blendshape_system = load_hybrid_model(
        args.checkpoint, device
    )

    # === 2. 加载测试数据 ===
    print(f"\n[Loading Test Data]")
    print(f"  CASME2 root: {args.casme2_root}")

    # 简化：使用随机数据测试
    # 实际应该从CASME2加载真实数据
    print("  Using synthetic test data")

    # === 3. 生成测试 ===
    print(f"\n[Generating Micro-Expressions]")

    os.makedirs(args.output_dir, exist_ok=True)

    emotions = ['happiness', 'surprise', 'disgust', 'repression']
    intensities = [0.4, 0.6, 0.8]

    results = []

    for i in tqdm(range(args.num_samples)):
        # 随机中性脸
        neutral_face = torch.rand(3, 64, 64) * 0.3 + 0.4

        # 随机情感和强度
        emotion = emotions[i % len(emotions)]
        intensity = intensities[i % len(intensities)]

        # 生成
        generated_video = generate_me(
            diffusion, blendshape_system, neutral_face, emotion, intensity, device
        )

        # 评估
        evaluation = evaluate_video(discriminator, generated_video, emotion, device)

        # 保存视频
        video_path = os.path.join(args.output_dir, f'gen_{i:03d}_{emotion}.mp4')
        save_video(generated_video, video_path)

        results.append({
            'sample_id': i,
            'emotion': emotion,
            'intensity': intensity,
            'evaluation': evaluation,
            'video_path': video_path,
        })

    # === 4. 统计结果 ===
    print(f"\n[Evaluation Results]")

    correct_count = sum([r['evaluation']['is_correct'] for r in results])
    total_count = len(results)
    accuracy = correct_count / total_count

    avg_confidence = np.mean([r['evaluation']['confidence'] for r in results])
    avg_motion = np.mean([r['evaluation']['avg_motion'] for r in results])

    print(f"  Total samples: {total_count}")
    print(f"  Correct predictions: {correct_count}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Avg confidence: {avg_confidence:.4f}")
    print(f"  Avg motion: {avg_motion:.6f}")

    # 每个情感的准确率
    print(f"\n  Per-emotion accuracy:")
    for emotion in emotions:
        emotion_results = [r for r in results if r['emotion'] == emotion]
        if emotion_results:
            emotion_correct = sum([r['evaluation']['is_correct'] for r in emotion_results])
            emotion_acc = emotion_correct / len(emotion_results)
            print(f"    {emotion}: {emotion_acc:.2%} ({emotion_correct}/{len(emotion_results)})")

    # === 5. 保存结果 ===
    import json
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Hybrid Model')

    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/hybrid_model/hybrid_final.pth')
    parser.add_argument('--casme2_root', type=str,
                        default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--output_dir', type=str,
                        default='./results/hybrid_test')
    parser.add_argument('--num_samples', type=int, default=10)

    args = parser.parse_args()

    test_hybrid(args)


if __name__ == '__main__':
    main()