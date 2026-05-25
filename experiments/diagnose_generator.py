"""
诊断脚本：检查生成器运动场问题
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_generator import CensorGGenerator, AU_KEYPOINT_MAPPING, AU_INDEX

def diagnose_generator(checkpoint_path):
    """诊断生成器"""
    print("="*60)
    print("Censor-G Generator Diagnosis")
    print("="*60)

    # 加载模型
    gen = CensorGGenerator()
    try:
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        gen.load_state_dict(ckpt['generator'])
        print(f"[OK] Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    gen.eval()

    # 1. 检查AU→关键点权重
    print("\n[1] AU → Keypoint Weight Analysis")
    weights = gen.motion_estimator.au_to_keypoint_weights
    print(f"  Weight shape: {weights.shape}")  # (17, 68, 2)

    # 统计每个AU的最大权重
    au_max_weights = []
    for au_idx in range(17):
        au_weight = weights[au_idx]
        max_w = au_weight.abs().max().item()
        mean_w = au_weight.abs().mean().item()
        au_max_weights.append((au_idx, max_w, mean_w))

    print("\n  AU with largest weights:")
    au_max_weights.sort(key=lambda x: x[1], reverse=True)
    for au_idx, max_w, mean_w in au_max_weights[:5]:
        au_name = list(AU_INDEX.keys())[list(AU_INDEX.values()).index(au_idx)]
        print(f"    {au_name}: max={max_w:.4f}, mean={mean_w:.4f}")

    # 2. 测试运动场
    print("\n[2] Motion Field Test")
    au = torch.zeros(1, 17)

    # 测试AU12 (smile)
    au[0, AU_INDEX['AU12']] = 1.0
    neutral = torch.randn(1, 3, 224, 224) * 0.1 + 0.5

    with torch.no_grad():
        video, motions = gen(neutral, au)

    motion = motions[0]
    print(f"  Motion field shape: {motion.shape}")  # (1, 2, 224, 224)

    # 统计运动
    dx = motion[0, 0].mean().item()
    dy = motion[0, 1].mean().item()
    mag = torch.sqrt(motion[0, 0]**2 + motion[0, 1]**2).mean().item()
    max_mag = torch.sqrt(motion[0, 0]**2 + motion[0, 1]**2).max().item()

    print(f"  dx mean: {dx:.6f} pixels")
    print(f"  dy mean: {dy:.6f} pixels")
    print(f"  Magnitude mean: {mag:.6f} pixels")
    print(f"  Magnitude max: {max_mag:.6f} pixels")

    # 3. 关键点位移检查
    print("\n[3] Keypoint Displacement Check")
    kp_detector = gen.motion_estimator.keypoint_detector
    neutral_kp = kp_detector.get_neutral_keypoints()
    print(f"  Neutral keypoints shape: {neutral_kp.shape}")  # (68, 2)

    # 计算AU12对应的关键点位移
    au12_idx = AU_INDEX['AU12']
    au12_weights = weights[au12_idx]  # (68, 2)

    # 找到受影响最大的关键点
    affected_kp = au12_weights.abs().sum(dim=1).topk(5)
    print(f"\n  AU12 affects keypoints (indices):")
    for kp_idx, effect in zip(affected_kp.indices.tolist(), affected_kp.values.tolist()):
        direction = au12_weights[kp_idx]
        print(f"    KP {kp_idx}: effect={effect:.4f}, dx={direction[0].item():.2f}, dy={direction[1].item():.2f}")

    # 4. 视频变化检查
    print("\n[4] Video Frame Change")
    # 计算相邻帧差异
    for t in range(min(3, video.shape[2])):
        frame_diff = (video[:,:,t+1] - video[:,:,t]).abs().mean().item()
        print(f"  Frame {t}→{t+1} diff: {frame_diff:.6f}")

    # 计算整体视频变化
    video_var = video.var().item()
    print(f"  Video variance: {video_var:.6f}")

    # 5. 问题诊断
    print("\n[5] Diagnosis Result")
    issues = []

    if max_mag < 1.0:
        issues.append("❌ Motion field too small (<1 pixel)")
        issues.append(f"   → AU weights need scaling (current max: {max_mag:.4f})")

    if video_var < 0.01:
        issues.append("❌ Video variance too low (<0.01)")
        issues.append("   → Generator producing near-constant frames")

    if weights.abs().max().item() < 0.5:
        issues.append("❌ AU→Keypoint weights too small")
        issues.append("   → Need to scale up initial weights")

    if len(issues) == 0:
        print("  ✅ No major issues detected")
    else:
        print("  Issues found:")
        for issue in issues:
            print(f"  {issue}")

    # 6. 建议修复
    print("\n[6] Recommended Fixes")
    print("  1. Scale AU→Keypoint weights by factor 10-50")
    print("  2. Add motion regularization loss")
    print("  3. Use perceptual loss instead of L1")
    print("  4. Check if neutral_face != target first frame")

    return {
        'motion_magnitude': mag,
        'max_motion': max_mag,
        'video_variance': video_var,
        'au_max_weight': weights.abs().max().item(),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/censor_g_gen_multi/censor_g_gen_final.pth')
    args = parser.parse_args()

    diagnose_generator(args.checkpoint)