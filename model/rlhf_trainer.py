# =============================================================================
# RLHF for Micro-Expression Generation
# =============================================================================
# 人类反馈强化学习流程
#
# 流程：
#   1. 生成多个候选视频
#   2. 人类标注偏好（哪个更好）
#   3. 训练Reward Model
#   4. 用PPO优化Generator
#
# 奖励信号：
#   - 表情自然度 (1-5分)
#   - 运动流畅度 (1-5分)
#   - 与提示词匹配度 (1-5分)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


# =============================================================================
# Part 1: 人类反馈数据结构
# =============================================================================

@dataclass
class HumanFeedback:
    """人类反馈数据"""
    video_id: str
    prompt: str
    generated_video_path: str
    reference_video_path: Optional[str]  # 可选的真实视频

    # 评分 (1-5)
    naturalness: float  # 自然度
    smoothness: float   # 流畅度
    prompt_match: float # 与提示词匹配度
    overall: float      # 总体评分

    # 比较反馈（可选）
    comparison_winner: Optional[str] = None  # 哪个视频更好
    comparison_reason: Optional[str] = None  # 原因说明


class FeedbackDataset(Dataset):
    """人类反馈数据集"""

    def __init__(self, feedback_list: List[HumanFeedback]):
        self.feedback = feedback_list

    def __len__(self):
        return len(self.feedback)

    def __getitem__(self, idx):
        fb = self.feedback[idx]
        return {
            'video_id': fb.video_id,
            'prompt': fb.prompt,
            'naturalness': fb.naturalness,
            'smoothness': fb.smoothness,
            'prompt_match': fb.prompt_match,
            'overall': fb.overall,
        }


# =============================================================================
# Part 2: Reward Model
# =============================================================================

class RewardModel(nn.Module):
    """
    奖励模型

    输入：生成的视频 + 提示词
    输出：奖励分数（预测人类评分）
    """

    def __init__(self, num_frames=16, image_size=224, hidden_dim=256):
        super().__init__()

        self.num_frames = num_frames
        self.image_size = image_size

        # 视频编码器（3D CNN）
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # 文本编码器（简单embedding）
        self.prompt_embedding = nn.Embedding(1000, hidden_dim)  # 简化版

        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(128 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 多维度评分头（可选）
        self.multi_head = nn.Sequential(
            nn.Linear(128 + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # naturalness, smoothness, prompt_match
        )

    def forward(self, video, prompt_ids=None):
        """
        Args:
            video: (B, C, T, H, W) 生成的视频
            prompt_ids: (B,) 提示词ID（简化版）

        Returns:
            reward: (B, 1) 奖励分数
            multi_scores: (B, 3) 多维度评分
        """
        # 视频特征
        video_feat = self.video_encoder(video)
        video_feat = video_feat.view(video_feat.size(0), -1)  # (B, 128)

        # 提示词特征（简化：用随机embedding代替）
        if prompt_ids is None:
            prompt_ids = torch.zeros(video.size(0), dtype=torch.long, device=video.device)
        prompt_feat = self.prompt_embedding(prompt_ids)  # (B, hidden_dim)

        # 拼接
        combined = torch.cat([video_feat, prompt_feat], dim=1)

        # 预测奖励
        reward = self.reward_head(combined)
        multi_scores = self.multi_head(combined)

        return reward, multi_scores

    def predict_reward(self, video, prompt_ids=None):
        """预测奖励分数"""
        with torch.no_grad():
            reward, _ = self.forward(video, prompt_ids)
            return reward


# =============================================================================
# Part 3: PPO训练器
# =============================================================================

class PPOTrainer:
    """
    PPO训练器

    使用Proximal Policy Optimization优化Generator
    """

    def __init__(self,
                 generator,
                 reward_model,
                 lr=1e-5,
                 clip_ratio=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01):
        """
        Args:
            generator: 微表情生成器
            reward_model: 奖励模型
            lr: 学习率
            clip_ratio: PPO裁剪比例
            value_coef: 价值损失系数
            entropy_coef: 熵奖励系数
        """
        self.generator = generator
        self.reward_model = reward_model
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # 优化器
        self.gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        self.reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)

        # 价值函数（用于baseline）
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute_advantage(self, rewards, values):
        """计算优势函数"""
        advantages = rewards - values.detach()
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def train_step(self, batch, epoch=0):
        """
        单步训练

        Args:
            batch: 包含neutral_face, au_activation, prompt_ids等
            epoch: 当前epoch
        """
        neutral_face = batch['neutral_face']
        au_activation = batch['au_activation']
        prompt_ids = batch.get('prompt_ids', None)

        # === Step 1: 生成视频 ===
        generated_video, motion_fields = self.generator(neutral_face, au_activation)

        # === Step 2: 计算奖励 ===
        reward, multi_scores = self.reward_model(generated_video, prompt_ids)

        # === Step 3: 计算价值 ===
        # 用视频特征计算baseline
        video_feat = self.reward_model.video_encoder(generated_video)
        video_feat = video_feat.view(video_feat.size(0), -1)
        value = self.value_head(video_feat)

        # === Step 4: 计算优势 ===
        advantage = self.compute_advantage(reward, value)

        # === Step 5: PPO更新 ===
        # 这里简化了PPO的完整流程，实际需要：
        # - 计算old policy的log prob
        # - 计算new policy的log prob
        # - 计算ratio并裁剪

        # 简化版：直接最大化奖励
        loss = -advantage.mean()

        # 反向传播
        self.gen_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.gen_optimizer.step()

        return {
            'loss': loss.item(),
            'reward': reward.mean().item(),
            'advantage': advantage.mean().item(),
        }

    def train_reward_model(self, feedback_data: List[HumanFeedback], epochs=10):
        """
        训练奖励模型

        Args:
            feedback_data: 人类反馈数据
            epochs: 训练epoch数
        """
        print(f"\n[PPOTrainer] Training Reward Model with {len(feedback_data)} samples")

        dataset = FeedbackDataset(feedback_data)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0

            for batch in dataloader:
                # 简化：用评分作为目标
                target_reward = torch.tensor(batch['overall'], dtype=torch.float32)

                # 前向传播（这里需要实际的video tensor）
                # 简化版：直接用评分训练
                loss = F.mse_loss(
                    torch.randn(len(target_reward)),  # 占位符
                    target_reward
                )

                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# =============================================================================
# Part 4: 人类反馈收集接口
# =============================================================================

class FeedbackCollector:
    """
    人类反馈收集器

    提供简单的接口收集人类反馈：
      1. 生成候选视频
      2. 展示给用户
      3. 收集评分和比较
    """

    def __init__(self, output_dir='./human_feedback'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.feedback_list = []

    def collect_rating(self,
                       video_path: str,
                       prompt: str,
                       reference_path: str = None) -> Dict:
        """
        收集单个视频的评分

        Returns:
            feedback: 人类反馈数据
        """
        print(f"\n[FeedbackCollector] Please rate the video: {video_path}")
        print(f"  Prompt: {prompt}")

        # 这里应该是实际的UI界面
        # 简化版：模拟收集
        print("  Please enter ratings (1-5):")
        print("    - Naturalness: ", end="")
        # naturalness = float(input())  # 实际使用时取消注释

        # 模拟评分
        naturalness = 3.5
        smoothness = 4.0
        prompt_match = 4.5
        overall = (naturalness + smoothness + prompt_match) / 3

        feedback = HumanFeedback(
            video_id=os.path.basename(video_path),
            prompt=prompt,
            generated_video_path=video_path,
            reference_video_path=reference_path,
            naturalness=naturalness,
            smoothness=smoothness,
            prompt_match=prompt_match,
            overall=overall,
        )

        self.feedback_list.append(feedback)
        return vars(feedback)

    def collect_comparison(self,
                           video1_path: str,
                           video2_path: str,
                           prompt: str) -> Dict:
        """
        收集两个视频的比较反馈

        Returns:
            comparison: 哪个更好，为什么
        """
        print(f"\n[FeedbackCollector] Compare two videos")
        print(f"  Prompt: {prompt}")
        print(f"  Video 1: {video1_path}")
        print(f"  Video 2: {video2_path}")

        # 模拟比较
        winner = "video1"  # 或 "video2"
        reason = "More natural expression"

        return {
            'winner': winner,
            'reason': reason,
        }

    def save_feedback(self, filename='feedback.json'):
        """保存反馈数据"""
        output_path = os.path.join(self.output_dir, filename)

        data = [vars(fb) for fb in self.feedback_list]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[FeedbackCollector] Saved {len(data)} feedback to {output_path}")

    def load_feedback(self, filename='feedback.json'):
        """加载反馈数据"""
        input_path = os.path.join(self.output_dir, filename)

        with open(input_path, 'r') as f:
            data = json.load(f)

        self.feedback_list = [HumanFeedback(**d) for d in data]
        print(f"[FeedbackCollector] Loaded {len(self.feedback_list)} feedback")

        return self.feedback_list


# =============================================================================
# Part 5: 完整的RLHF流程
# =============================================================================

class RLHFTrainer:
    """
    完整的RLHF训练流程

    Step 1: 预训练Generator
    Step 2: 收集人类反馈
    Step 3: 训练Reward Model
    Step 4: 用PPO优化Generator
    Step 5: 迭代优化
    """

    def __init__(self,
                 generator,
                 checkpoint_path: str = None):
        """
        Args:
            generator: 微表情生成器
            checkpoint_path: 预训练权重路径
        """
        self.generator = generator
        self.reward_model = RewardModel()
        self.ppo_trainer = PPOTrainer(generator, self.reward_model)
        self.feedback_collector = FeedbackCollector()

        # 加载预训练权重
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
            self.generator.load_state_dict(ckpt['generator'])
            print(f"[RLHFTrainer] Loaded generator from {checkpoint_path}")

    def step1_generate_candidates(self,
                                  prompts: List[str],
                                  image_paths: List[str],
                                  num_candidates: int = 4):
        """
        Step 1: 为每个提示词生成多个候选视频

        Args:
            prompts: 提示词列表
            image_paths: 图像路径列表
            num_candidates: 每个提示词生成多少个候选
        """
        print(f"\n[RLHFTrainer] Step 1: Generating {num_candidates} candidates per prompt")

        candidates = []

        for prompt, image_path in zip(prompts, image_paths):
            prompt_candidates = []

            for i in range(num_candidates):
                # 使用不同的AU扰动生成多样化候选
                output_path = f"./candidates/{prompt}_{i}.mp4"

                # 简化：模拟生成
                prompt_candidates.append({
                    'path': output_path,
                    'prompt': prompt,
                    'image': image_path,
                })

            candidates.append(prompt_candidates)

        return candidates

    def step2_collect_feedback(self, candidates):
        """
        Step 2: 收集人类反馈

        实际应用中，这里应该：
          - 展示视频给用户
          - 用户打分或比较
          - 记录反馈
        """
        print(f"\n[RLHFTrainer] Step 2: Collecting human feedback")

        # 模拟收集反馈
        feedback = []

        for prompt_candidates in candidates:
            for candidate in prompt_candidates:
                fb = self.feedback_collector.collect_rating(
                    video_path=candidate['path'],
                    prompt=candidate['prompt']
                )
                feedback.append(fb)

        return feedback

    def step3_train_reward_model(self, feedback, epochs=10):
        """
        Step 3: 训练奖励模型
        """
        print(f"\n[RLHFTrainer] Step 3: Training reward model")

        self.ppo_trainer.train_reward_model(feedback, epochs=epochs)

    def step4_ppo_optimization(self, dataloader, epochs=5):
        """
        Step 4: 用PPO优化Generator
        """
        print(f"\n[RLHFTrainer] Step 4: PPO optimization")

        for epoch in range(epochs):
            total_reward = 0

            for batch in dataloader:
                metrics = self.ppo_trainer.train_step(batch, epoch)
                total_reward += metrics['reward']

            avg_reward = total_reward / len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs}, Avg Reward: {avg_reward:.4f}")

    def train(self,
              prompts: List[str],
              image_paths: List[str],
              num_iterations: int = 3):
        """
        完整的RLHF训练循环
        """
        print("\n" + "="*60)
        print("RLHF Training Pipeline")
        print("="*60)

        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration+1}/{num_iterations}")
            print(f"{'='*60}")

            # Step 1: 生成候选
            candidates = self.step1_generate_candidates(prompts, image_paths)

            # Step 2: 收集反馈
            feedback = self.step2_collect_feedback(candidates)

            # Step 3: 训练奖励模型
            self.step3_train_reward_model(feedback)

            # Step 4: PPO优化
            # 这里需要实际的dataloader
            # self.step4_ppo_optimization(dataloader)

            # 保存检查点
            torch.save({
                'generator': self.generator.state_dict(),
                'reward_model': self.reward_model.state_dict(),
                'iteration': iteration + 1,
            }, f'./checkpoints/rlhf_iteration_{iteration+1}.pth')

        print("\n" + "="*60)
        print("RLHF Training Complete!")
        print("="*60)


# =============================================================================
# Demo
# =============================================================================

def demo_rlhf():
    """演示RLHF流程"""
    print("\n" + "="*60)
    print("RLHF for Micro-Expression Generation Demo")
    print("="*60)

    # 创建模拟Generator
    from model.censor_g_generator import CensorGGenerator

    generator = CensorGGenerator()
    reward_model = RewardModel()

    # 模拟输入
    neutral_face = torch.randn(2, 3, 224, 224) * 0.1 + 0.5
    au_activation = torch.zeros(2, 17)
    au_activation[0, 8] = 0.8  # AU12 smile
    au_activation[1, 5] = 0.6  # AU6

    # 生成视频
    with torch.no_grad():
        video, motions = generator(neutral_face, au_activation)

    print(f"\n[Demo] Generated video shape: {video.shape}")

    # 预测奖励
    reward, multi_scores = reward_model(video)

    print(f"\n[Demo] Predicted reward: {reward[0].item():.4f}")
    print(f"  Multi-dimension scores:")
    print(f"    Naturalness: {multi_scores[0, 0].item():.4f}")
    print(f"    Smoothness: {multi_scores[0, 1].item():.4f}")
    print(f"    Prompt Match: {multi_scores[0, 2].item():.4f}")

    # 创建RLHF Trainer
    rlhf_trainer = RLHFTrainer(generator)

    # 模拟完整流程
    prompts = ["微笑", "惊讶"]
    image_paths = ["image1.jpg", "image2.jpg"]

    # 这里只是展示流程，实际需要真实数据和人类反馈
    # rlhf_trainer.train(prompts, image_paths)

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


# =============================================================================
# Part 6: 专家反馈 vs 普通用户反馈
# =============================================================================

class ExpertFeedbackCollector:
    """
    专家反馈收集器

    针对微表情领域的特殊需求：
      1. 专业评估人员（心理学背景）
      2. FACS编码专家
      3. 微表情识别训练有素的人员

    评估维度：
      - AU准确性：生成的AU是否正确
      - 时间特性：onset/apex/offset是否合理
      - 微表情特征：是否符合微表情定义
    """

    def __init__(self):
        self.expert_feedback = []

    def collect_expert_rating(self,
                               video_path: str,
                               prompt: str,
                               expected_au: Dict = None) -> Dict:
        """
        收集专家级评估

        Args:
            video_path: 视频路径
            prompt: 提示词
            expected_au: 期望的AU激活（用于验证）

        Returns:
            expert_scores: 专家评分
        """
        print(f"\n[ExpertFeedbackCollector] Expert Evaluation")
        print(f"  Video: {video_path}")
        print(f"  Prompt: {prompt}")

        # 专家评估维度
        scores = {
            # 1. 微表情特性评分 (1-5)
            'micro_expression_quality': 0.0,  # 是否符合微表情定义
            'duration_appropriate': 0.0,       # 持续时间是否合理
            'intensity_appropriate': 0.0,     # 强度是否适中

            # 2. AU准确性 (1-5)
            'au_accuracy': 0.0,               # AU激活是否正确
            'au_temporal': 0.0,               # AU时间变化是否合理

            # 3. 自然度 (1-5)
            'naturalness': 0.0,               # 表情是否自然
            'smoothness': 0.0,                # 运动是否流畅

            # 4. 与提示词匹配 (1-5)
            'prompt_match': 0.0,              # 是否符合提示词

            # 5. 总体评分
            'overall': 0.0,

            # 6. 专家评语
            'comments': '',
        }

        # 模拟专家评分（实际需要真实专家填写）
        # 这里用模拟数据
        scores['micro_expression_quality'] = 3.5
        scores['duration_appropriate'] = 4.0
        scores['intensity_appropriate'] = 3.8
        scores['au_accuracy'] = 3.2
        scores['au_temporal'] = 3.5
        scores['naturalness'] = 4.0
        scores['smoothness'] = 4.2
        scores['prompt_match'] = 3.8
        scores['overall'] = sum([
            scores['micro_expression_quality'],
            scores['au_accuracy'],
            scores['naturalness'],
            scores['prompt_match'],
        ]) / 4
        scores['comments'] = "Good micro-expression quality, AU activation could be more precise"

        return scores


class AutomaticEvaluator:
    """
    自动评估器（替代人类专家）

    使用预训练模型自动评估生成质量：
      1. 微表情识别器 → 判断生成的视频是否被正确识别
      2. AU检测器 → 检测生成的AU是否正确
      3. 时间分析器 → 分析onset/apex/offset
      4. FID/LPIPS → 图像质量评估
    """

    def __init__(self, recognizer_checkpoint: str = None):
        """
        Args:
            recognizer_checkpoint: 微表情识别器checkpoint
        """
        # 这里可以加载预训练的识别器
        # 用于自动评估生成质量
        self.recognizer = None

    def evaluate_micro_expression_quality(self, video: torch.Tensor) -> float:
        """
        评估微表情质量

        使用识别器判断生成视频的类别置信度
        """
        # 如果有预训练识别器
        if self.recognizer is not None:
            with torch.no_grad():
                logits = self.recognizer(video)
                confidence = F.softmax(logits, dim=1).max().item()
                return confidence

        # 否则返回模拟分数
        return 0.7

    def evaluate_au_accuracy(self,
                             generated_video: torch.Tensor,
                             expected_au: torch.Tensor) -> float:
        """
        评估AU准确性

        使用AU检测器检测生成视频的AU
        与期望的AU对比
        """
        # 模拟AU检测
        # 实际需要用预训练的AU检测器
        detected_au = torch.rand_like(expected_au) * 0.5 + expected_au * 0.5

        # 计算相关性
        correlation = F.cosine_similarity(
            detected_au.unsqueeze(0),
            expected_au.unsqueeze(0)
        ).item()

        return max(0, correlation)

    def evaluate_temporal_characteristics(self, video: torch.Tensor) -> Dict:
        """
        评估时间特性

        分析：
          - onset时长
          - apex时长
          - offset时长
          - 是否符合微表情定义
        """
        T = video.shape[2]

        # 计算帧间运动
        frame_motion = []
        for t in range(T - 1):
            motion = (video[:, :, t+1] - video[:, :, t]).abs().mean().item()
            frame_motion.append(motion)

        # 找到apex（运动最大的帧）
        apex_frame = np.argmax(frame_motion)
        max_motion = max(frame_motion)

        # 判断时间特性
        # 微表情：onset < 0.5秒，apex < 0.2秒，offset < 0.5秒
        onset_frames = apex_frame
        apex_frames = 1  # 简化
        offset_frames = T - apex_frame - 1

        # 微表情定义：总时长 < 0.5秒（假设30fps）
        total_duration = T / 30.0
        is_micro = total_duration < 0.5

        return {
            'onset_frames': onset_frames,
            'apex_frame': apex_frame,
            'offset_frames': offset_frames,
            'total_duration': total_duration,
            'is_micro_expression': is_micro,
            'max_motion': max_motion,
        }

    def comprehensive_evaluate(self,
                               generated_video: torch.Tensor,
                               expected_au: torch.Tensor = None,
                               prompt: str = None) -> Dict:
        """
        综合评估

        Returns:
            evaluation: 综合评估结果
        """
        # 1. 微表情质量
        me_quality = self.evaluate_micro_expression_quality(generated_video)

        # 2. AU准确性
        au_accuracy = 0.5
        if expected_au is not None:
            au_accuracy = self.evaluate_au_accuracy(generated_video, expected_au)

        # 3. 时间特性
        temporal = self.evaluate_temporal_characteristics(generated_video)

        # 4. 综合评分
        overall = (me_quality + au_accuracy) / 2

        return {
            'micro_expression_quality': me_quality,
            'au_accuracy': au_accuracy,
            'temporal': temporal,
            'overall': overall,
        }


# =============================================================================
# Part 7: 混合反馈策略
# =============================================================================

class HybridFeedbackStrategy:
    """
    混合反馈策略

    结合：
      1. 自动评估器（快速、低成本）
      2. 专家反馈（高质量、高成本）
      3. 普通用户反馈（多样性、中等成本）
    """

    def __init__(self):
        self.auto_evaluator = AutomaticEvaluator()
        self.expert_collector = ExpertFeedbackCollector()

    def get_feedback_weights(self, sample_type: str = 'random') -> Dict:
        """
        获取不同反馈源的权重

        Args:
            sample_type: 样本类型

        Returns:
            weights: 各反馈源的权重
        """
        if sample_type == 'random':
            # 随机抽样：主要用自动评估
            return {
                'automatic': 0.7,
                'expert': 0.1,
                'user': 0.2,
            }
        elif sample_type == 'difficult':
            # 困难样本：需要更多专家反馈
            return {
                'automatic': 0.3,
                'expert': 0.5,
                'user': 0.2,
            }
        elif sample_type == 'validation':
            # 验证样本：需要专家确认
            return {
                'automatic': 0.2,
                'expert': 0.7,
                'user': 0.1,
            }
        else:
            return {
                'automatic': 0.5,
                'expert': 0.3,
                'user': 0.2,
            }

    def collect_hybrid_feedback(self,
                                video: torch.Tensor,
                                expected_au: torch.Tensor = None,
                                sample_type: str = 'random') -> Dict:
        """
        收集混合反馈

        Args:
            video: 生成的视频
            expected_au: 期望的AU
            sample_type: 样本类型

        Returns:
            feedback: 综合反馈
        """
        weights = self.get_feedback_weights(sample_type)

        # 1. 自动评估
        auto_eval = self.auto_evaluator.comprehensive_evaluate(video, expected_au)

        # 2. 专家反馈（模拟）
        expert_scores = {
            'overall': 0.7,  # 模拟
        }

        # 3. 用户反馈（模拟）
        user_scores = {
            'overall': 0.75,  # 模拟
        }

        # 4. 加权综合
        overall = (
            weights['automatic'] * auto_eval['overall'] +
            weights['expert'] * expert_scores['overall'] +
            weights['user'] * user_scores['overall']
        )

        return {
            'automatic_evaluation': auto_eval,
            'expert_scores': expert_scores,
            'user_scores': user_scores,
            'weights': weights,
            'overall_reward': overall,
        }


# =============================================================================
# Part 8: Demo with Expert Feedback
# =============================================================================

def demo_expert_feedback():
    """演示专家反馈流程"""
    print("\n" + "="*60)
    print("Expert Feedback for Micro-Expression Generation")
    print("="*60)

    # 创建评估器
    auto_evaluator = AutomaticEvaluator()
    expert_collector = ExpertFeedbackCollector()
    hybrid_strategy = HybridFeedbackStrategy()

    # 模拟生成视频
    video = torch.randn(1, 3, 16, 224, 224) * 0.1 + 0.5
    expected_au = torch.zeros(17)
    expected_au[8] = 0.8  # AU12

    # 1. 自动评估
    print("\n[1] Automatic Evaluation")
    auto_eval = auto_evaluator.comprehensive_evaluate(video, expected_au)
    print(f"  Micro-expression quality: {auto_eval['micro_expression_quality']:.4f}")
    print(f"  AU accuracy: {auto_eval['au_accuracy']:.4f}")
    print(f"  Is micro-expression: {auto_eval['temporal']['is_micro_expression']}")

    # 2. 专家评估
    print("\n[2] Expert Evaluation")
    expert_scores = expert_collector.collect_expert_rating(
        video_path='demo_video.mp4',
        prompt='微笑',
        expected_au={'AU12': 0.8}
    )
    print(f"  AU accuracy: {expert_scores['au_accuracy']:.2f}")
    print(f"  Naturalness: {expert_scores['naturalness']:.2f}")
    print(f"  Overall: {expert_scores['overall']:.2f}")
    print(f"  Comments: {expert_scores['comments']}")

    # 3. 混合策略
    print("\n[3] Hybrid Feedback Strategy")
    hybrid_feedback = hybrid_strategy.collect_hybrid_feedback(
        video=video,
        expected_au=expected_au,
        sample_type='random'
    )
    print(f"  Weights: {hybrid_feedback['weights']}")
    print(f"  Overall reward: {hybrid_feedback['overall_reward']:.4f}")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    # 可以选择运行不同的demo
    demo_rlhf()
    # demo_expert_feedback()