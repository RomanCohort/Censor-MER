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


if __name__ == '__main__':
    demo_rlhf()