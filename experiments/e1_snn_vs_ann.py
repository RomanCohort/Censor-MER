# =============================================================================
# SNN vs ANN Comparison Experiment
# =============================================================================
# 实验E1：对比SNN和ANN的生成效果
#
# 目标：
#   证明SNN机制的优势：
#     1. 发放率编码更符合神经科学
#     2. 突触连接更真实
#     3. 膜电位动力学更准确
#     4. 感受野结构更合理
#
# 评估指标：
#   - FID (生成质量)
#   - AU-Acc (AU一致性)
#   - Temporal-Consistency (时间曲线一致性)
#   - Neuro-Interpretability (神经科学可解释性)
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_snn import CensorGSNN, AU_INDEX, AU_INFO
from model.censor_g_v2 import CensorGv2
from generation.generation_loss import MicroExpressionGenerationLoss


def compute_fid(fake_features, real_features):
    """
    计算FID (Fréchet Inception Distance)

    FID = ||mu_f - mu_r||^2 + Tr(Cov_f + Cov_r - 2*sqrt(Cov_f*Cov_r))
    """
    # 均值
    mu_fake = fake_features.mean(axis=0)
    mu_real = real_features.mean(axis=0)

    # 协方差
    cov_fake = np.cov(fake_features, rowvar=False)
    cov_real = np.cov(real_features, rowvar=False)

    # 均值差异
    mean_diff = np.sum((mu_fake - mu_real) ** 2)

    # 协方差项
    cov_sqrt = np.sqrt(cov_fake * cov_real + 1e-8)
    cov_term = np.trace(cov_fake + cov_real - 2 * cov_sqrt)

    fid = mean_diff + cov_term

    return fid


def compute_au_accuracy(generated_au, target_au):
    """
    计算AU一致性

    AU-Acc = 1 - ||generated_au - target_au||
    """
    diff = np.abs(generated_au - target_au)
    au_acc = 1 - diff.mean()

    return au_acc


def compute_temporal_consistency(video):
    """
    计算时间一致性

    Temporal-Consistency = 1 - mean(frame_diff)
    """
    if len(video.shape) != 5:
        return 0.0

    # 帧差异
    frame_diff = np.abs(video[:, :, 1:] - video[:, :, :-1])
    consistency = 1 - frame_diff.mean()

    return consistency


def analyze_spike_interpretability(model, au_input, device):
    """
    分析SNN神经科学可解释性

    指标：
      1. 发放率分布（是否符合神经发放率编码）
      2. 脉冲时间模式（是否有周期性）
      3. AU显著性（是否符合快速通路概念）
      4. 突触效应（EPSP/IPSP是否有差异）
    """
    with torch.no_grad():
        # V1: 发放率
        firing_rate, spikes = model.v1_spiking(au_input)

        # 发放率分布
        rate_std = firing_rate.std().item()
        rate_range = firing_rate.max().item() - firing_rate.min().item()

        # V2: 突触效应
        effective_au = model.v2_circuit(firing_rate)
        synaptic_effect = (effective_au - firing_rate).abs().mean().item()

        # V3: 时间动力学差异（快肌 vs 慢肌）
        au_temporal = model.v3_temporal(effective_au)

        # 快肌AU（AU1, AU5）
        fast_au_idx = [AU_INDEX['AU1'], AU_INDEX['AU5']]
        fast_onset = au_temporal[0, fast_au_idx, :4].mean().item()

        # 慢肌AU（AU17）
        slow_au_idx = [AU_INDEX['AU17']]
        slow_onset = au_temporal[0, slow_au_idx, :4].mean().item()

        # 时间差异
        temporal_diff = fast_onset - slow_onset

    return {
        'firing_rate_std': rate_std,
        'firing_rate_range': rate_range,
        'synaptic_effect': synaptic_effect,
        'temporal_diff_fast_vs_slow': temporal_diff,
        'spike_count': spikes.sum().item(),
    }


def run_snn_vs_ann_experiment(args):
    """
    SNN vs ANN对比实验
    """
    print("\n" + "="*60)
    print("Experiment E1: SNN vs ANN Comparison")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    print("\n[Models] Loading...")
    snn_model = CensorGSNN(
        num_frames=args.num_frames,
        image_size=args.image_size,
    ).to(device)

    ann_model = CensorGv2(
        num_frames=args.num_frames,
        image_size=args.image_size,
    ).to(device)

    # 加载预训练权重（如果有）
    if args.snn_checkpoint:
        snn_model.load_state_dict(torch.load(args.snn_checkpoint)['model'])
        print(f"  SNN loaded: {args.snn_checkpoint}")

    if args.ann_checkpoint:
        ann_model.load_state_dict(torch.load(args.ann_checkpoint)['model'])
        print(f"  ANN loaded: {args.ann_checkpoint}")

    # 创建测试数据
    print("\n[Dataset] Creating test samples...")
    num_samples = args.num_samples

    # 测试AU配置
    test_configs = [
        {'name': 'happiness', 'au': {AU_INDEX['AU6']: 0.6, AU_INDEX['AU12']: 0.8}},
        {'name': 'surprise', 'au': {AU_INDEX['AU1']: 0.7, AU_INDEX['AU2']: 0.7, AU_INDEX['AU5']: 0.8}},
        {'name': 'disgust', 'au': {AU_INDEX['AU4']: 0.5, AU_INDEX['AU9']: 0.7}},
        {'name': 'repression', 'au': {AU_INDEX['AU14']: 0.6, AU_INDEX['AU17']: 0.4}},
    ]

    results = {
        'snn': {'fid': [], 'au_acc': [], 'temporal': [], 'interpretability': []},
        'ann': {'fid': [], 'au_acc': [], 'temporal': []},
    }

    # 运行对比
    print("\n[Evaluation] Running comparison...")
    gen_loss_fn = MicroExpressionGenerationLoss()

    for config in test_configs:
        print(f"\n  Config: {config['name']}")

        # 创建输入
        neutral_face = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
        au_input = torch.zeros(args.batch_size, 17).to(device)
        for au_idx, val in config['au'].items():
            au_input[:, au_idx] = val

        target_video = torch.randn(args.batch_size, 3, args.num_frames, args.image_size, args.image_size).to(device)

        # SNN生成
        snn_model.eval()
        with torch.no_grad():
            snn_generated, snn_spikes = snn_model(neutral_face, au_input, return_spikes=True)
            snn_losses, snn_loss = gen_loss_fn(snn_generated, target_video, au_input, au_input)

        # SNN可解释性分析
        snn_interp = analyze_spike_interpretability(snn_model, au_input, device)

        # ANN生成
        ann_model.eval()
        with torch.no_grad():
            ann_generated = ann_model(neutral_face, au_input)
            ann_losses, ann_loss = gen_loss_fn(ann_generated, target_video, au_input, au_input)

        # 计算指标
        snn_fid = compute_fid(
            snn_generated.flatten(2).mean(-1).cpu().numpy(),
            target_video.flatten(2).mean(-1).cpu().numpy()
        )
        ann_fid = compute_fid(
            ann_generated.flatten(2).mean(-1).cpu().numpy(),
            target_video.flatten(2).mean(-1).cpu().numpy()
        )

        # 记录结果
        results['snn']['fid'].append(snn_fid)
        results['snn']['au_acc'].append(1 - snn_losses['au'].item())
        results['snn']['temporal'].append(compute_temporal_consistency(snn_generated.cpu().numpy()))
        results['snn']['interpretability'].append(snn_interp)

        results['ann']['fid'].append(ann_fid)
        results['ann']['au_acc'].append(1 - ann_losses['au'].item())
        results['ann']['temporal'].append(compute_temporal_consistency(ann_generated.cpu().numpy()))

        print(f"    SNN FID: {snn_fid:.4f}, AU-Acc: {results['snn']['au_acc'][-1]:.4f}")
        print(f"    ANN FID: {ann_fid:.4f}, AU-Acc: {results['ann']['au_acc'][-1]:.4f}")
        print(f"    SNN Spike Rate Std: {snn_interp['firing_rate_std']:.4f}")
        print(f"    SNN Temporal Diff (fast-slow): {snn_interp['temporal_diff_fast_vs_slow']:.4f}")

    # 汇总统计
    print("\n" + "="*60)
    print("[Results] SNN vs ANN")
    print("="*60)

    print("\n  SNN Average:")
    print(f"    FID: {np.mean(results['snn']['fid']):.4f}")
    print(f"    AU-Acc: {np.mean(results['snn']['au_acc']):.4f}")
    print(f"    Temporal: {np.mean(results['snn']['temporal']):.4f}")

    print("\n  ANN Average:")
    print(f"    FID: {np.mean(results['ann']['fid']):.4f}")
    print(f"    AU-Acc: {np.mean(results['ann']['au_acc']):.4f}")
    print(f"    Temporal: {np.mean(results['ann']['temporal']):.4f}")

    # SNN特有指标
    print("\n  SNN Neuro-Interpretability:")
    avg_interp = {
        'firing_rate_std': np.mean([r['firing_rate_std'] for r in results['snn']['interpretability']]),
        'synaptic_effect': np.mean([r['synaptic_effect'] for r in results['snn']['interpretability']]),
        'temporal_diff': np.mean([r['temporal_diff_fast_vs_slow'] for r in results['snn']['interpretability']]),
        'spike_count': np.mean([r['spike_count'] for r in results['snn']['interpretability']]),
    }
    print(f"    Firing Rate Std: {avg_interp['firing_rate_std']:.4f}")
    print(f"    Synaptic Effect: {avg_interp['synaptic_effect']:.4f}")
    print(f"    Temporal Diff (Fast vs Slow): {avg_interp['temporal_diff']:.4f}")
    print(f"    Average Spike Count: {avg_interp['spike_count']:.1f}")

    # 保存结果
    save_path = os.path.join(args.save_dir, 'e1_snn_vs_ann_results.json')
    with open(save_path, 'w') as f:
        json.dump({
            'snn': {
                'fid_mean': np.mean(results['snn']['fid']),
                'au_acc_mean': np.mean(results['snn']['au_acc']),
                'temporal_mean': np.mean(results['snn']['temporal']),
            },
            'ann': {
                'fid_mean': np.mean(results['ann']['fid']),
                'au_acc_mean': np.mean(results['ann']['au_acc']),
                'temporal_mean': np.mean(results['ann']['temporal']),
            },
            'snn_interpretability': avg_interp,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    return results


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='E1: SNN vs ANN')

    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--snn_checkpoint', type=str, default=None)
    parser.add_argument('--ann_checkpoint', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./results/experiments')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 运行实验（AU_INDEX已在文件开头导入）
    results = run_snn_vs_ann_experiment(args)

    print("\n[E1] Complete!")


if __name__ == '__main__':
    main()