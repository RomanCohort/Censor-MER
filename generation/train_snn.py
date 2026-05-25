# =============================================================================
# Censor-G SNN Training Script
# =============================================================================
# 训练真正的脉冲神经网络微表情生成模型
#
# 神经科学机制：
#   1. V1层：LIF脉冲发放（可学习阈值）
#   2. V2层：突触权重可塑性（EPSP/IPSP）
#   3. V3层：膜时间参数微调
#   4. V4层：感受野权重
#   5. IT层：冲突解决网络
#
# 训练策略：
#   - SNN参数大部分基于神经科学文献设计（初始化）
#   - 只微调阈值、突触权重、整合权重
#   - 训练时间：~3小时
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
import time
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_snn import CensorGSNN, AU_INDEX, AU_INFO
from generation.generation_loss import MicroExpressionGenerationLoss, GANLoss, Discriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Train Censor-G SNN')

    # Dataset
    parser.add_argument('--dataset', type=str, default='casme2')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/data/CASME2')

    # SNN Model Parameters
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--spike_threshold', type=float, default=1.0,
                        help='Initial spike threshold for LIF neurons')
    parser.add_argument('--spike_time_steps', type=int, default=10,
                        help='Number of time steps for spike integration')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_mode', type=str, default='all',
                        choices=['all', 'snn_only', 'integration_only'],
                        help='Training mode: all=snn+integration, snn_only=thresholds+synapses, integration_only=IT')

    # GAN
    parser.add_argument('--use_gan', action='store_true')
    parser.add_argument('--gan_start_epoch', type=int, default=10,
                        help='Start GAN training after this epoch')

    # Saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints/censor_g_snn')
    parser.add_argument('--log_dir', type=str, default='./logs/censor_g_snn')
    parser.add_argument('--save_every', type=int, default=5)

    # Spike analysis
    parser.add_argument('--analyze_spikes', action='store_true',
                        help='Analyze spike patterns during training')

    return parser.parse_args()


def configure_trainable_params(model, train_mode='all'):
    """
    配置可训练参数

    神经科学依据：
      - 阈值可塑性：神经元阈值随经验调整
      - 突触可塑性：LTP/LTD改变突触权重
      - 膜参数相对稳定：基于肌肉生理学
    """

    # 默认所有参数可训练
    for param in model.parameters():
        param.requires_grad = True

    if train_mode == 'snn_only':
        # 只训练SNN核心参数：阈值、突触权重
        for name, param in model.named_parameters():
            if 'au_thresholds' in name or 'W_excitatory' in name or 'W_inhibitory' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif train_mode == 'integration_only':
        # 只训练整合层（IT）
        for name, param in model.named_parameters():
            if 'it_fusion' in name or 'integration_weights' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # 统计
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    print(f"\n[Trainable Params] Mode: {train_mode}")
    print(f"  Trainable: {trainable_count:,}")
    print(f"  Total: {total_count:,}")
    print(f"  Ratio: {100*trainable_count/total_count:.1f}%")

    # 打印可训练参数
    print("\n  Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name}: {param.shape} ({param.numel():,})")


def create_dummy_dataset(args, num_samples=200):
    """
    创建模拟数据集用于训练测试

    TODO: 替换为真实CASME2数据集
    """
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 模拟数据
            neutral_face = torch.randn(3, args.image_size, args.image_size)

            # AU激活（模拟情感）
            au = torch.zeros(17)
            emotion = torch.randint(0, 4, (1,)).item()

            # 情感 → AU映射
            if emotion == 0:  # Happiness
                au[AU_INDEX['AU6']] = 0.6 + torch.rand(1).item() * 0.2
                au[AU_INDEX['AU12']] = 0.7 + torch.rand(1).item() * 0.2
                au[AU_INDEX['AU25']] = 0.1 + torch.rand(1).item() * 0.1
            elif emotion == 1:  # Surprise
                au[AU_INDEX['AU1']] = 0.6 + torch.rand(1).item() * 0.2
                au[AU_INDEX['AU2']] = 0.6 + torch.rand(1).item() * 0.2
                au[AU_INDEX['AU5']] = 0.7 + torch.rand(1).item() * 0.2
            elif emotion == 2:  # Disgust
                au[AU_INDEX['AU4']] = 0.4 + torch.rand(1).item() * 0.2
                au[AU_INDEX['AU9']] = 0.6 + torch.rand(1).item() * 0.2
            elif emotion == 3:  # Repression
                au[AU_INDEX['AU14']] = 0.5 + torch.rand(1).item() * 0.2
                au[AU_INDEX['AU17']] = 0.3 + torch.rand(1).item() * 0.2

            # 目标视频（模拟微表情）
            target_video = torch.randn(3, args.num_frames, args.image_size, args.image_size)

            intensity = 0.5 + torch.rand(1).item() * 0.5

            return {
                'neutral_face': neutral_face,
                'au': au,
                'emotion': emotion,
                'target_video': target_video,
                'intensity': intensity,
            }

    return DummyDataset()


def analyze_spike_patterns(model, au_input, device):
    """
    分析脉冲模式

    神经科学分析：
      - 发放率分布
      - 脉冲时间模式
      - AU显著性统计
    """
    with torch.no_grad():
        firing_rate, spikes = model.v1_spiking(au_input)

    # 发放率统计
    mean_rate = firing_rate.mean().item()
    max_rate = firing_rate.max().item()

    # 脉冲时间分布
    spike_times = spikes.sum(dim=0).sum(dim=0)  # (T,)
    peak_time = spike_times.argmax().item()

    # 显著AU
    significant_mask = model.v1_spiking.get_saliency_mask(firing_rate)
    num_significant = significant_mask.sum().item()

    return {
        'mean_firing_rate': mean_rate,
        'max_firing_rate': max_rate,
        'peak_spike_time': peak_time,
        'num_significant_au': num_significant,
        'spike_history': spikes.cpu().numpy(),
    }


def train_snn(model, discriminator, dataloader, args, device):
    """
    训练Censor-G SNN

    训练流程：
      Phase 1 (0-10 epoch): 基础生成训练（无GAN）
      Phase 2 (10+ epoch): GAN对抗训练（可选）
    """
    print("\n" + "="*60)
    print("[Training] Censor-G SNN")
    print("="*60)

    # 配置可训练参数
    configure_trainable_params(model, args.train_mode)

    # 获取可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # GAN
    if args.use_gan:
        d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5)
        gan_loss = GANLoss(gan_mode='standard')

    gen_loss_fn = MicroExpressionGenerationLoss()

    # 训练日志
    log_history = {
        'epoch': [],
        'g_loss': [],
        'd_loss': [],
        'spike_stats': [],
    }

    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_start = time.time()

        g_loss_total = 0
        d_loss_total = 0
        spike_stats_epoch = []

        # Phase判断
        use_gan_phase = args.use_gan and epoch >= args.gan_start_epoch

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            neutral_face = batch['neutral_face'].to(device)
            au = batch['au'].to(device)
            emotion = batch['emotion']
            if isinstance(emotion, int):
                emotion = torch.tensor([emotion]).long().to(device)
            else:
                emotion = emotion.long().to(device)
            target_video = batch['target_video'].to(device)
            intensity = batch['intensity']
            if isinstance(intensity, float):
                intensity = torch.tensor([intensity] * neutral_face.shape[0]).to(device)
            else:
                intensity = intensity.to(device)

            B = neutral_face.shape[0]

            # =================================
            # Generator训练
            # =================================
            model.train()

            # SNN生成（返回脉冲历史）
            generated, spike_history = model(neutral_face, au, return_spikes=True)

            # 计算损失
            losses, g_loss = gen_loss_fn(generated, target_video, au, au)

            # 脉冲分析（可选）
            if args.analyze_spikes:
                stats = analyze_spike_patterns(model, au, device)
                spike_stats_epoch.append(stats)

            # GAN损失
            if use_gan_phase:
                discriminator.eval()
                d_fake = discriminator(generated)
                gan_g_loss = gan_loss(d_fake, True)
                g_loss = g_loss + 0.1 * gan_g_loss  # GAN权重

            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()

            g_loss_total += g_loss.item()

            # =================================
            # Discriminator训练
            # =================================
            if use_gan_phase:
                model.eval()
                discriminator.train()

                with torch.no_grad():
                    generated = model(neutral_face, au)

                d_real = discriminator(target_video)
                d_loss_real = gan_loss(d_real, True)

                d_fake = discriminator(generated.detach())
                d_loss_fake = gan_loss(d_fake, False)

                d_loss = (d_loss_real + d_loss_fake) / 2

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                d_loss_total += d_loss.item()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        g_avg = g_loss_total / len(dataloader)
        d_avg = d_loss_total / len(dataloader) if use_gan_phase else 0

        print(f"\nEpoch {epoch+1}:")
        print(f"  G Loss: {g_avg:.4f}")
        if use_gan_phase:
            print(f"  D Loss: {d_avg:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # SNN参数状态
        print(f"  Spike Threshold (avg): {model.v1_spiking.au_thresholds.mean().item():.3f}")
        print(f"  Excitatory Synapse (max): {model.v2_circuit.W_excitatory.max().item():.3f}")
        print(f"  Inhibitory Synapse (max): {model.v2_circuit.W_inhibitory.max().item():.3f}")

        # 记录日志
        log_history['epoch'].append(epoch + 1)
        log_history['g_loss'].append(g_avg)
        log_history['d_loss'].append(d_avg)

        if args.analyze_spikes and spike_stats_epoch:
            avg_spike_stats = {
                'mean_firing_rate': np.mean([s['mean_firing_rate'] for s in spike_stats_epoch]),
                'max_firing_rate': np.max([s['max_firing_rate'] for s in spike_stats_epoch]),
                'num_significant_au': np.mean([s['num_significant_au'] for s in spike_stats_epoch]),
            }
            log_history['spike_stats'].append(avg_spike_stats)
            print(f"  Spike Stats: rate={avg_spike_stats['mean_firing_rate']:.3f}, "
                  f"significant={avg_spike_stats['num_significant_au']:.1f}")

        # 保存最佳模型
        if g_avg < best_loss:
            best_loss = g_avg
            best_path = os.path.join(args.save_dir, 'censor_g_snn_best.pth')
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'g_loss': g_avg,
                'args': vars(args),
            }, best_path)
            print(f"  [Best] Saved: {best_path}")

        # 定期保存
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f'censor_g_snn_epoch_{epoch+1}.pth')
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'g_loss': g_avg,
                'args': vars(args),
            }, save_path)
            print(f"  Saved: {save_path}")

    # 保存训练日志
    log_path = os.path.join(args.log_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f"\n[Log] Saved: {log_path}")

    print("\n[Training] Complete!")
    print(f"  Best Loss: {best_loss:.4f}")

    return model, log_history


def run_validation(model, dataloader, args, device, num_samples=10):
    """
    验证生成效果

    评估指标：
      - Reconstruction loss
      - AU consistency
      - Spike pattern analysis
    """
    print("\n" + "="*60)
    print("[Validation] Evaluating SNN generation")
    print("="*60)

    model.eval()
    gen_loss_fn = MicroExpressionGenerationLoss()

    total_loss = 0
    spike_rates = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            neutral_face = batch['neutral_face'].to(device)
            au = batch['au'].to(device)
            target_video = batch['target_video'].to(device)

            generated, spikes = model(neutral_face, au, return_spikes=True)

            losses, loss = gen_loss_fn(generated, target_video, au, au)
            total_loss += loss.item()

            # 脉冲统计
            firing_rate = spikes.sum(dim=-1) / spikes.shape[-1]
            spike_rates.append(firing_rate.mean().item())

    avg_loss = total_loss / num_samples
    avg_spike_rate = np.mean(spike_rates)

    print(f"  Reconstruction Loss: {avg_loss:.4f}")
    print(f"  Average Spike Rate: {avg_spike_rate:.3f}")

    return avg_loss, avg_spike_rate


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Setup]")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Train Mode: {args.train_mode}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create SNN model
    print("\n[Model] Creating Censor-G SNN...")
    model = CensorGSNN(
        num_frames=args.num_frames,
        image_size=args.image_size,
        spike_threshold=args.spike_threshold,
        spike_time_steps=args.spike_time_steps,
    ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total_params:,}")

    # Discriminator
    if args.use_gan:
        discriminator = Discriminator().to(device)
        print("  GAN Enabled: Yes")
    else:
        discriminator = None
        print("  GAN Enabled: No")

    # Dataset
    print("\n[Dataset] Loading...")
    dataset = create_dummy_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  Samples: {len(dataset)}")
    print(f"  Batches: {len(dataloader)}")

    # Train
    model, log_history = train_snn(model, discriminator, dataloader, args, device)

    # Validation
    avg_loss, avg_spike = run_validation(model, dataloader, args, device)

    # Save final
    final_path = os.path.join(args.save_dir, 'censor_g_snn_final.pth')
    torch.save({
        'model': model.state_dict(),
        'log_history': log_history,
        'validation': {
            'loss': avg_loss,
            'spike_rate': avg_spike,
        },
        'args': vars(args),
    }, final_path)
    print(f"\n[Complete] Saved: {final_path}")

    # 打印训练总结
    print("\n" + "="*60)
    print("[Summary] SNN Training Results")
    print("="*60)
    print(f"  Best Loss: {min(log_history['g_loss']):.4f}")
    print(f"  Final Loss: {log_history['g_loss'][-1]:.4f}")
    print(f"  Validation Loss: {avg_loss:.4f}")
    print(f"  Validation Spike Rate: {avg_spike:.3f}")

    # SNN参数最终状态
    print("\n[SNN Parameters]")
    print(f"  Spike Thresholds: {model.v1_spiking.au_thresholds.detach().cpu().numpy()[:5]}")
    print(f"  Excitatory Max: {model.v2_circuit.W_excitatory.max().item():.3f}")
    print(f"  Inhibitory Max: {model.v2_circuit.W_inhibitory.max().item():.3f}")


if __name__ == '__main__':
    main()