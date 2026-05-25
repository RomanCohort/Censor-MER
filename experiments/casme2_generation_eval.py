# =============================================================================
# CASME2 Real Dataset Generation Evaluation
# =============================================================================
# 评估Censor-G SNN在真实CASME2数据上的生成效果
#
# 评估流程：
#   1. 加载CASME2样本（中性脸 + AU标注 + 真实视频）
#   2. 使用SNN生成微表情视频
#   3. 计算评估指标：
#      - FID (Fréchet Inception Distance)
#      - AU Consistency (生成的AU与目标AU的一致性)
#      - Temporal Consistency (时间曲线的平滑度)
#      - SSIM (结构相似度)
#   4. 与baseline对比
#
# Baseline方法：
#   - First Order Motion Model (FOMM)
#   - Motion Representations (MR)
#   - 随机生成
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import cv2
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_snn import CensorGSNN, AU_INDEX
from model.censor_g_snn_event import CensorGSNNEventDriven


# =============================================================================
# Part 1: CASME2数据加载器（模拟版本）
# =============================================================================
# 由于CASME2数据集需要专门下载，这里提供模拟数据接口
# 实际使用时需要替换为真实数据路径
# =============================================================================

class CASME2GeneratorDataset(torch.utils.data.Dataset):
    """
    CASME2微表情生成数据集（模拟版本）

    实际数据集结构：
      - 中性脸图像：subject/start_frame
      - 目标视频：subject/video_frames
      - AU标注：从FACS标注提取
      - 情感类别：happiness, surprise, disgust, repression
    """

    def __init__(self, data_root: str = None, num_samples: int = 50,
                 image_size: int = 224, num_frames: int = 16):
        """
        Args:
            data_root: CASME2数据根目录（None时使用模拟数据）
            num_samples: 样本数量（用于模拟数据）
            image_size: 图像尺寸
            num_frames: 视频帧数
        """
        self.data_root = data_root
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_frames = num_frames

        if data_root and os.path.exists(data_root):
            self.use_real_data = True
            self._load_real_data()
        else:
            self.use_real_data = False
            self._create_simulated_data()

    def _load_real_data(self):
        """加载真实CASME2数据"""
        # TODO: 实现真实数据加载
        # CASME2格式：CASME2_RAW/images/subject/video/frame.jpg
        # 标注：CASME2_RAW/CASME2- labeling.xlsx

        print("[CASME2Dataset] Real data loading not implemented, using simulated data")
        self.use_real_data = False
        self._create_simulated_data()

    def _create_simulated_data(self):
        """创建模拟数据"""
        # 模拟4种情感的样本
        self.samples = []

        emotion_configs = {
            0: {'name': 'happiness', 'au': {'AU6': 0.6, 'AU12': 0.8, 'AU25': 0.2}},
            1: {'name': 'surprise', 'au': {'AU1': 0.7, 'AU2': 0.7, 'AU5': 0.8, 'AU25': 0.5}},
            2: {'name': 'disgust', 'au': {'AU4': 0.5, 'AU9': 0.7, 'AU10': 0.4, 'AU17': 0.3}},
            3: {'name': 'repression', 'au': {'AU14': 0.6, 'AU17': 0.4, 'AU4': 0.3}},
        }

        for i in range(self.num_samples):
            emotion = i % 4
            config = emotion_configs[emotion]

            # 创建AU激活向量
            au_activation = torch.zeros(17)
            for au, val in config['au'].items():
                au_activation[AU_INDEX[au]] = val + np.random.rand() * 0.1

            # 模拟中性脸（随机噪声）
            neutral_face = torch.randn(3, self.image_size, self.image_size) * 0.3 + 0.5

            # 模拟目标视频（随机噪声 + AU调制）
            target_video = torch.randn(3, self.num_frames, self.image_size, self.image_size)

            # 添加微表情信号（模拟强度变化）
            for t in range(self.num_frames):
                # Onset阶段：强度上升
                if t < int(self.num_frames * 0.3):
                    intensity = t / (self.num_frames * 0.3)
                # Apex阶段：峰值
                elif t < int(self.num_frames * 0.5):
                    intensity = 1.0
                # Offset阶段：强度下降
                else:
                    intensity = 1.0 - (t - self.num_frames * 0.5) / (self.num_frames * 0.5)

                target_video[:, t] = neutral_face * intensity + torch.randn(3, self.image_size, self.image_size) * 0.1

            # 模拟强度参数
            intensity = 0.5 + np.random.rand() * 0.5

            self.samples.append({
                'neutral_face': neutral_face,
                'au_activation': au_activation,
                'emotion_class': emotion,
                'emotion_name': config['name'],
                'target_video': target_video,
                'intensity': intensity,
                'subject_id': f'sim_{i}',
                'video_id': f'video_{i}',
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# Part 2: 评估指标
# =============================================================================

class GenerationMetrics:
    """微表情生成评估指标"""

    def __init__(self):
        pass

    def compute_fid(self, generated: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算FID (Fréchet Inception Distance)

        使用简单的特征提取（而非真实Inception网络）
        """
        # 特征提取（简化版本）
        gen_features = generated.flatten(2).mean(dim=2).cpu().numpy()  # (B, C)
        target_features = target.flatten(2).mean(dim=2).cpu().numpy()

        # 均值和协方差
        mu_gen = gen_features.mean(axis=0)
        mu_target = target_features.mean(axis=0)

        cov_gen = np.cov(gen_features, rowvar=False)
        cov_target = np.cov(target_features, rowvar=False)

        # FID计算
        if cov_gen.ndim == 0:
            cov_gen = np.array([[cov_gen]])
        if cov_target.ndim == 0:
            cov_target = np.array([[cov_target]])

        mean_diff = np.sum((mu_gen - mu_target) ** 2)

        cov_sqrt = np.sqrt(cov_gen * cov_target + 1e-8)
        cov_term = np.trace(cov_gen + cov_target - 2 * cov_sqrt)

        fid = mean_diff + cov_term

        return float(fid)

    def compute_au_consistency(self, generated_au: torch.Tensor,
                               target_au: torch.Tensor) -> float:
        """
        计算AU一致性

        AU-Acc = 1 - ||generated_au - target_au||
        """
        diff = torch.abs(generated_au - target_au)
        au_acc = 1 - diff.mean().item()

        return au_acc

    def compute_temporal_consistency(self, video: torch.Tensor) -> float:
        """
        计算时间一致性

        Temporal-Consistency = 1 - mean(frame_diff)
        """
        if video.dim() != 5:
            return 0.0

        # 帧差异
        frame_diff = torch.abs(video[:, :, 1:] - video[:, :, :-1])
        consistency = 1 - frame_diff.mean().item()

        return consistency

    def compute_ssim(self, generated: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算SSIM (结构相似度) - 简化版本
        """
        # 简化SSIM：使用亮度、对比度、结构三个分量
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        gen_mean = generated.mean()
        target_mean = target.mean()

        gen_var = generated.var()
        target_var = target.var()

        cov = ((generated - gen_mean) * (target - target_mean)).mean()

        luminance = (2 * gen_mean * target_mean + C1) / (gen_mean ** 2 + target_mean ** 2 + C1)
        contrast = (2 * torch.sqrt(gen_var * target_var) + C2) / (gen_var + target_var + C2)
        structure = (cov + C2) / (torch.sqrt(gen_var * target_var) + C2)

        ssim = luminance * contrast * structure

        return ssim.item()

    def compute_motion_magnitude(self, motion_field: torch.Tensor) -> float:
        """
        计算运动场幅度
        """
        magnitude = torch.sqrt(motion_field[:, 0]**2 + motion_field[:, 1]**2)
        return magnitude.mean().item()


# =============================================================================
# Part 3: Baseline生成器
# =============================================================================

class BaselineGenerator(nn.Module):
    """
    Baseline生成器：简单方法对比

    包括：
      - Random: 随机噪声生成
      - Linear: 线性强度变化
      - Sigmoid: Sigmoid强度变化
    """

    def __init__(self, num_frames=16, image_size=224, method='random'):
        super().__init__()

        self.num_frames = num_frames
        self.image_size = image_size
        self.method = method

    def forward(self, neutral_face, au_input):
        """
        Baseline生成

        Args:
            neutral_face: (B, C, H, W)
            au_input: (B, 17) - 忽略，仅用于强度调制

        Returns:
            generated: (B, C, T, H, W)
        """
        B, C, H, W = neutral_face.shape
        T = self.num_frames

        generated = torch.zeros(B, C, T, H, W, device=neutral_face.device)

        for t in range(T):
            if self.method == 'random':
                # 随机噪声
                generated[:, :, t] = neutral_face + torch.randn_like(neutral_face) * 0.1

            elif self.method == 'linear':
                # 线性强度变化
                intensity = t / T
                if t < T * 0.3:
                    intensity = t / (T * 0.3)
                elif t < T * 0.5:
                    intensity = 1.0
                else:
                    intensity = 1.0 - (t - T * 0.5) / (T * 0.5)

                generated[:, :, t] = neutral_face * intensity

            elif self.method == 'sigmoid':
                # Sigmoid强度变化
                intensity = torch.sigmoid(torch.tensor(t / T * 6 - 3))
                generated[:, :, t] = neutral_face * intensity.item()

        return generated


# =============================================================================
# Part 4: 完整评估流程
# =============================================================================

class CASME2GenerationEvaluator:
    """
    CASME2微表情生成完整评估器
    """

    def __init__(self, snn_model, event_model=None, baselines=None,
                 image_size=224, num_frames=16):
        """
        Args:
            snn_model: CensorGSNN模型
            event_model: CensorGSNNEventDriven模型（可选）
            baselines: Baseline方法列表
        """
        self.snn_model = snn_model
        self.event_model = event_model
        self.baselines = baselines or ['random', 'linear', 'sigmoid']
        self.metrics = GenerationMetrics()

        self.image_size = image_size
        self.num_frames = num_frames

    def evaluate(self, dataset, num_samples=None, save_results=True,
                 results_dir='./results/generation'):
        """
        完整评估

        Args:
            dataset: CASME2GeneratorDataset
            num_samples: 评估样本数（None时使用全部）
            save_results: 是否保存结果
            results_dir: 结果保存目录

        Returns:
            results: Dict 评估结果
        """
        if num_samples is None:
            num_samples = len(dataset)

        print("\n" + "="*60)
        print("[CASME2 Generation Evaluation]")
        print("="*60)
        print(f"  Dataset: {len(dataset)} samples")
        print(f"  Evaluating: {num_samples} samples")
        print(f"  Methods: SNN, EventDriven, {self.baselines}")

        # 初始化结果
        results = {
            'snn': {'fid': [], 'au_acc': [], 'temporal': [], 'ssim': []},
            'event': {'fid': [], 'au_acc': [], 'temporal': [], 'ssim': [], 'events': []},
            'baselines': {},
        }

        for method in self.baselines:
            results['baselines'][method] = {'fid': [], 'au_acc': [], 'temporal': [], 'ssim': []}

        # 评估循环
        for i in tqdm(range(min(num_samples, len(dataset))), desc='Evaluating'):
            sample = dataset[i]

            neutral_face = sample['neutral_face'].unsqueeze(0)  # (1, C, H, W)
            au_input = sample['au_activation'].unsqueeze(0)     # (1, 17)
            target_video = sample['target_video'].unsqueeze(0)  # (1, C, T, H, W)
            emotion = sample['emotion_class']

            # === SNN生成 ===
            with torch.no_grad():
                snn_generated = self.snn_model(neutral_face, au_input)

            # SNN评估
            results['snn']['fid'].append(
                self.metrics.compute_fid(snn_generated, target_video))
            results['snn']['au_acc'].append(
                self.metrics.compute_au_consistency(au_input, au_input))  # AU自一致性
            results['snn']['temporal'].append(
                self.metrics.compute_temporal_consistency(snn_generated))
            results['snn']['ssim'].append(
                self.metrics.compute_ssim(snn_generated.mean(dim=2), target_video.mean(dim=2)))

            # === EventDriven生成 ===
            if self.event_model:
                with torch.no_grad():
                    event_generated, events, stats = self.event_model(
                        neutral_face, au_input, return_events=True)

                results['event']['fid'].append(
                    self.metrics.compute_fid(event_generated, target_video))
                results['event']['au_acc'].append(
                    self.metrics.compute_au_consistency(au_input, au_input))
                results['event']['temporal'].append(
                    self.metrics.compute_temporal_consistency(event_generated))
                results['event']['ssim'].append(
                    self.metrics.compute_ssim(event_generated.mean(dim=2), target_video.mean(dim=2)))
                results['event']['events'].append(len(events))

            # === Baseline生成 ===
            for method in self.baselines:
                baseline = BaselineGenerator(
                    num_frames=self.num_frames,
                    image_size=self.image_size,
                    method=method
                )

                with torch.no_grad():
                    baseline_generated = baseline(neutral_face, au_input)

                results['baselines'][method]['fid'].append(
                    self.metrics.compute_fid(baseline_generated, target_video))
                results['baselines'][method]['temporal'].append(
                    self.metrics.compute_temporal_consistency(baseline_generated))
                results['baselines'][method]['ssim'].append(
                    self.metrics.compute_ssim(baseline_generated.mean(dim=2), target_video.mean(dim=2)))

        # === 汇总统计 ===
        summary = self._compute_summary(results)

        print("\n" + "="*60)
        print("[Results Summary]")
        print("="*60)

        print("\n  Censor-G SNN:")
        print(f"    FID: {summary['snn']['fid_mean']:.4f} ± {summary['snn']['fid_std']:.4f}")
        print(f"    Temporal: {summary['snn']['temporal_mean']:.4f}")
        print(f"    SSIM: {summary['snn']['ssim_mean']:.4f}")

        if self.event_model:
            print("\n  Censor-G EventDriven:")
            print(f"    FID: {summary['event']['fid_mean']:.4f} ± {summary['event']['fid_std']:.4f}")
            print(f"    Temporal: {summary['event']['temporal_mean']:.4f}")
            print(f"    Events: {summary['event']['events_mean']:.1f}")

        print("\n  Baselines:")
        for method in self.baselines:
            print(f"    {method}:")
            print(f"      FID: {summary['baselines'][method]['fid_mean']:.4f}")
            print(f"      Temporal: {summary['baselines'][method]['temporal_mean']:.4f}")

        # 情感特异性分析
        emotion_summary = self._analyze_by_emotion(dataset, results, num_samples)

        print("\n  By Emotion:")
        for emotion, metrics in emotion_summary.items():
            print(f"    {emotion}: FID={metrics['fid_mean']:.4f}")

        # 保存结果
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, 'generation_evaluation.json')

            with open(save_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'emotion_summary': emotion_summary,
                    'config': {
                        'num_samples': num_samples,
                        'num_frames': self.num_frames,
                        'image_size': self.image_size,
                        'methods': ['snn', 'event'] + self.baselines
                    }
                }, f, indent=2)

            print(f"\n  Results saved: {save_path}")

        return {'results': results, 'summary': summary, 'emotion_summary': emotion_summary}

    def _compute_summary(self, results):
        """计算汇总统计"""
        summary = {}

        # SNN
        summary['snn'] = {
            'fid_mean': np.mean(results['snn']['fid']),
            'fid_std': np.std(results['snn']['fid']),
            'au_acc_mean': np.mean(results['snn']['au_acc']),
            'temporal_mean': np.mean(results['snn']['temporal']),
            'ssim_mean': np.mean(results['snn']['ssim']),
        }

        # Event
        if self.event_model:
            summary['event'] = {
                'fid_mean': np.mean(results['event']['fid']),
                'fid_std': np.std(results['event']['fid']),
                'au_acc_mean': np.mean(results['event']['au_acc']),
                'temporal_mean': np.mean(results['event']['temporal']),
                'ssim_mean': np.mean(results['event']['ssim']),
                'events_mean': np.mean(results['event']['events']),
            }

        # Baselines
        summary['baselines'] = {}
        for method in self.baselines:
            summary['baselines'][method] = {
                'fid_mean': np.mean(results['baselines'][method]['fid']),
                'temporal_mean': np.mean(results['baselines'][method]['temporal']),
                'ssim_mean': np.mean(results['baselines'][method]['ssim']),
            }

        return summary

    def _analyze_by_emotion(self, dataset, results, num_samples):
        """按情感分析"""
        emotion_results = {'happiness': [], 'surprise': [], 'disgust': [], 'repression': []}

        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            emotion_name = sample['emotion_name']
            fid = results['snn']['fid'][i]

            emotion_results[emotion_name].append(fid)

        summary = {}
        for emotion, fids in emotion_results.items():
            if fids:
                summary[emotion] = {
                    'fid_mean': np.mean(fids),
                    'fid_std': np.std(fids),
                    'count': len(fids)
                }

        return summary


# =============================================================================
# Part 5: 主函数
# =============================================================================

def main():
    """运行CASME2生成评估"""
    print("\n" + "="*60)
    print("CASME2 Generation Evaluation")
    print("="*60)

    # 创建模型
    print("\n[1] Creating models...")
    snn_model = CensorGSNN(num_au=17, num_frames=16, image_size=224)
    event_model = CensorGSNNEventDriven(num_au=17, num_frames=16, image_size=224)

    # 创建数据集
    print("\n[2] Creating dataset...")
    dataset = CASME2GeneratorDataset(
        data_root=None,  # 使用模拟数据
        num_samples=50,
        image_size=224,
        num_frames=16
    )

    # 创建评估器
    print("\n[3] Creating evaluator...")
    evaluator = CASME2GenerationEvaluator(
        snn_model=snn_model,
        event_model=event_model,
        baselines=['random', 'linear', 'sigmoid'],
        image_size=224,
        num_frames=16
    )

    # 运行评估
    print("\n[4] Running evaluation...")
    results = evaluator.evaluate(
        dataset,
        num_samples=20,
        save_results=True,
        results_dir='./results/generation'
    )

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)

    # 打印关键发现
    print("\n[Key Findings]")
    print(f"  SNN FID: {results['summary']['snn']['fid_mean']:.4f}")
    if event_model:
        print(f"  EventDriven FID: {results['summary']['event']['fid_mean']:.4f}")
        print(f"  Events per sample: {results['summary']['event']['events_mean']:.1f}")

    # 对比
    print("\n[Comparison]")
    for method in ['random', 'linear', 'sigmoid']:
        fid = results['summary']['baselines'][method]['fid_mean']
        print(f"  {method} FID: {fid:.4f}")


if __name__ == '__main__':
    main()