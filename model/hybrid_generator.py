# =============================================================================
# Hybrid Model: Diffusion + Blendshape + GAN Discriminator
# =============================================================================
# 三者结合：
#   1. 扩散模型：高质量生成
#   2. Blendshape：精确控制
#   3. GAN判别器：快速监督
#
# 训练流程：
#   Stage 1: 扩散模型预训练（高质量）
#   Stage 2: GAN对抗微调（快速优化）
#   Stage 3: 联合训练（最佳效果）
#
# 推理：
#   Blendshape条件 → 扩散去噪 → GAN精修 → 最终输出
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


# =============================================================================
# Hybrid Architecture
# =============================================================================

class HybridMicroExpressionGenerator(nn.Module):
    """
    混合微表情生成器

    组合：
      - 扩散模型：生成基础
      - Blendshape：精确控制
      - GAN判别器：质量监督
    """

    def __init__(self,
                 diffusion_model,
                 discriminator,
                 num_blendshapes=52,
                 num_frames=16):
        super().__init__()

        self.diffusion = diffusion_model
        self.discriminator = discriminator
        self.num_blendshapes = num_blendshapes
        self.num_frames = num_frames

        # Blendshape系统
        from model.diffusion_blendshape import BlendshapeSystem
        self.blendshape_system = BlendshapeSystem(num_blendshapes)

        # 精修网络（GAN输出 → 精修）
        self.refiner = nn.Sequential(
            nn.Conv3d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 3, 3, 1, 1),
        )

    def forward(self, neutral_face, blendshape, use_refiner=True):
        """
        生成流程：
          1. 扩散模型生成
          2. GAN判别器评估
          3. 精修网络优化

        Args:
            neutral_face: (B, C, T, H, W)
            blendshape: (B, 52)
            use_refiner: 是否使用精修

        Returns:
            generated_video: 生成的视频
            disc_score: 判别器评分
        """
        # 1. 扩散模型生成
        diff_output = self.diffusion.generate(neutral_face, blendshape, num_steps=20)

        # 2. GAN判别器评估
        with torch.no_grad():
            disc_logits, disc_probs = self.discriminator(diff_output)
            disc_score = disc_probs.max(dim=1)[0].mean()

        # 3. 精修网络（可选）
        if use_refiner:
            refined = diff_output + self.refiner(diff_output) * 0.1
        else:
            refined = diff_output

        return refined, disc_score

    def generate_from_au(self, neutral_face, au_activation):
        """从AU生成"""
        blendshape = self.blendshape_system.au_to_blendshape(au_activation)
        return self.forward(neutral_face, blendshape)

    def generate_from_emotion(self, neutral_face, emotion, intensity):
        """从情感生成"""
        blendshape = self.blendshape_system.get_emotion_blendshape(emotion, intensity)
        blendshape = blendshape.unsqueeze(0).expand(neutral_face.shape[0], -1)
        return self.forward(neutral_face, blendshape)


# =============================================================================
# Hybrid Training Strategy
# =============================================================================

class HybridTrainer:
    """
    混合训练策略

    Stage 1: 扩散模型预训练
      - 只训练扩散模型
      - 目标：学习高质量生成

    Stage 2: GAN对抗微调
      - 固定扩散模型
      - 训练精修网络
      - 目标：快速优化质量

    Stage 3: 联合训练
      - 同时优化所有组件
      - 目标：最佳效果
    """

    def __init__(self,
                 hybrid_model,
                 diffusion_lr=1e-4,
                 refiner_lr=1e-4,
                 discriminator_lr=1e-5):
        """
        Args:
            hybrid_model: 混合模型
            diffusion_lr: 扩散模型学习率
            refiner_lr: 精修网络学习率
            discriminator_lr: 判别器学习率
        """
        self.model = hybrid_model

        # 分组优化器
        self.diffusion_optimizer = torch.optim.Adam(
            hybrid_model.diffusion.model.parameters(),
            lr=diffusion_lr
        )

        self.refiner_optimizer = torch.optim.Adam(
            hybrid_model.refiner.parameters(),
            lr=refiner_lr
        )

        self.disc_optimizer = torch.optim.Adam(
            hybrid_model.discriminator.parameters(),
            lr=discriminator_lr
        )

        # 训练记录
        self.training_log = {
            'stage1': [],  # 扩散预训练
            'stage2': [],  # GAN微调
            'stage3': [],  # 联合训练
        }

    def stage1_diffusion_pretrain(self, dataloader, epochs=10):
        """
        Stage 1: 扩散模型预训练

        只训练扩散模型，学习高质量生成
        """
        print("\n" + "="*60)
        print("Stage 1: Diffusion Pretraining")
        print("="*60)

        for epoch in range(epochs):
            total_loss = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                neutral_face = batch['neutral_face']
                target_video = batch['target_video']
                au_activation = batch['au_activation']

                # AU → Blendshape
                blendshape = self.model.blendshape_system.au_to_blendshape(au_activation)

                # 扩散模型训练
                loss = self._train_diffusion_step(target_video, blendshape)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}, Diffusion Loss: {avg_loss:.4f}")

            self.training_log['stage1'].append({
                'epoch': epoch + 1,
                'loss': avg_loss,
            })

        print("Stage 1 Complete!\n")

    def stage2_gan_finetune(self, dataloader, epochs=10):
        """
        Stage 2: GAN对抗微调

        固定扩散模型，训练精修网络
        """
        print("\n" + "="*60)
        print("Stage 2: GAN Adversarial Finetuning")
        print("="*60)

        # 冻结扩散模型
        for param in self.model.diffusion.model.parameters():
            param.requires_grad = False

        for epoch in range(epochs):
            total_g_loss = 0
            total_d_loss = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                neutral_face = batch['neutral_face']
                target_video = batch['target_video']
                expected_class = batch['emotion_class']
                au_activation = batch['au_activation']

                # 训练精修网络（生成器）
                g_loss = self._train_refiner_step(
                    neutral_face, target_video, expected_class, au_activation
                )

                # 训练判别器
                d_loss = self._train_discriminator_step(
                    neutral_face, target_video, expected_class, au_activation
                )

                total_g_loss += g_loss
                total_d_loss += d_loss

            avg_g_loss = total_g_loss / len(dataloader)
            avg_d_loss = total_d_loss / len(dataloader)

            print(f"  Epoch {epoch+1}, G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

            self.training_log['stage2'].append({
                'epoch': epoch + 1,
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            })

        # 解冻扩散模型
        for param in self.model.diffusion.model.parameters():
            param.requires_grad = True

        print("Stage 2 Complete!\n")

    def stage3_joint_training(self, dataloader, epochs=10):
        """
        Stage 3: 联合训练

        同时优化所有组件
        """
        print("\n" + "="*60)
        print("Stage 3: Joint Training")
        print("="*60)

        for epoch in range(epochs):
            total_loss = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                neutral_face = batch['neutral_face']
                target_video = batch['target_video']
                expected_class = batch['emotion_class']
                au_activation = batch['au_activation']

                # 联合训练
                loss = self._train_joint_step(
                    neutral_face, target_video, expected_class, au_activation
                )
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}, Joint Loss: {avg_loss:.4f}")

            self.training_log['stage3'].append({
                'epoch': epoch + 1,
                'loss': avg_loss,
            })

        print("Stage 3 Complete!\n")

    def _train_diffusion_step(self, target_video, blendshape):
        """扩散模型训练步"""
        B = target_video.shape[0]
        diffusion = self.model.diffusion

        # 随机时间步
        t = torch.randint(0, diffusion.num_timesteps, (B,))

        # 前向扩散
        xt, noise = diffusion.forward_diffusion(target_video, t)

        # 预测噪声
        noise_pred = diffusion.model(xt, t, blendshape)

        # 损失
        loss = F.mse_loss(noise_pred, noise)

        # 反向传播
        self.diffusion_optimizer.zero_grad()
        loss.backward()
        self.diffusion_optimizer.step()

        return loss.item()

    def _train_refiner_step(self, neutral_face, target_video, expected_class, au_activation):
        """精修网络训练步"""
        blendshape = self.model.blendshape_system.au_to_blendshape(au_activation)

        # 扩散生成
        with torch.no_grad():
            diff_output = self.model.diffusion.generate(neutral_face, blendshape, num_steps=10)

        # 精修
        refined = diff_output + self.model.refiner(diff_output) * 0.1

        # 判别器评分
        correct_prob, _, _ = self.model.discriminator.classify(refined, expected_class)

        # 生成器损失：让判别器正确识别
        g_loss = -torch.log(correct_prob + 1e-8).mean()

        # 反向传播
        self.refiner_optimizer.zero_grad()
        g_loss.backward()
        self.refiner_optimizer.step()

        return g_loss.item()

    def _train_discriminator_step(self, neutral_face, target_video, expected_class, au_activation):
        """判别器训练步"""
        blendshape = self.model.blendshape_system.au_to_blendshape(au_activation)

        # 扩散生成
        with torch.no_grad():
            diff_output = self.model.diffusion.generate(neutral_face, blendshape, num_steps=10)
            refined = diff_output + self.model.refiner(diff_output) * 0.1

        # 判别器分类
        logits, probs = self.model.discriminator(refined.detach())

        # 判别器损失：正确分类
        d_loss = F.cross_entropy(logits, expected_class)

        # 反向传播
        self.disc_optimizer.zero_grad()
        d_loss.backward()
        self.disc_optimizer.step()

        return d_loss.item()

    def _train_joint_step(self, neutral_face, target_video, expected_class, au_activation):
        """联合训练步"""
        blendshape = self.model.blendshape_system.au_to_blendshape(au_activation)

        # === 扩散损失 ===
        B = target_video.shape[0]
        t = torch.randint(0, self.model.diffusion.num_timesteps, (B,))
        xt, noise = self.model.diffusion.forward_diffusion(target_video, t)
        noise_pred = self.model.diffusion.model(xt, t, blendshape)
        diff_loss = F.mse_loss(noise_pred, noise)

        # === 生成 + 精修 ===
        diff_output = self.model.diffusion.generate(neutral_face, blendshape, num_steps=10)
        refined = diff_output + self.model.refiner(diff_output) * 0.1

        # === GAN损失 ===
        correct_prob, _, _ = self.model.discriminator.classify(refined, expected_class)
        gan_loss = -torch.log(correct_prob + 1e-8).mean()

        # === 总损失 ===
        total_loss = diff_loss + gan_loss * 0.5

        # 反向传播
        self.diffusion_optimizer.zero_grad()
        self.refiner_optimizer.zero_grad()
        total_loss.backward()
        self.diffusion_optimizer.step()
        self.refiner_optimizer.step()

        return total_loss.item()

    def full_training(self, dataloader, stage1_epochs=10, stage2_epochs=10, stage3_epochs=10):
        """完整三阶段训练"""
        self.stage1_diffusion_pretrain(dataloader, epochs=stage1_epochs)
        self.stage2_gan_finetune(dataloader, epochs=stage2_epochs)
        self.stage3_joint_training(dataloader, epochs=stage3_epochs)

        print("\n" + "="*60)
        print("Full Training Complete!")
        print("="*60)


# =============================================================================
# Inference Pipeline
# =============================================================================

class HybridInference:
    """混合推理管道"""

    def __init__(self, hybrid_model):
        self.model = hybrid_model

    def generate(self,
                 neutral_face,
                 prompt=None,
                 emotion=None,
                 intensity=0.6,
                 au_activation=None):
        """
        灵活生成接口

        支持：
          - 自然语言提示词
          - 情感类别
          - AU激活
        """
        # 提示词 → 情感
        if prompt:
            emotion = self._parse_prompt(prompt)

        # 情感 → Blendshape
        if emotion:
            blendshape = self.model.blendshape_system.get_emotion_blendshape(
                emotion, intensity
            )
            blendshape = blendshape.unsqueeze(0).expand(neutral_face.shape[0], -1)

        # AU → Blendshape
        elif au_activation is not None:
            blendshape = self.model.blendshape_system.au_to_blendshape(au_activation)

        else:
            raise ValueError("需要提供prompt、emotion或au_activation")

        # 生成
        generated, disc_score = self.model(neutral_face, blendshape)

        return {
            'video': generated,
            'disc_score': disc_score,
            'emotion': emotion,
            'blendshape': blendshape,
        }

    def _parse_prompt(self, prompt):
        """解析提示词"""
        prompt_lower = prompt.lower()

        if '微笑' in prompt_lower or 'smile' in prompt_lower:
            return 'happiness'
        elif '惊讶' in prompt_lower or 'surprise' in prompt_lower:
            return 'surprise'
        elif '厌恶' in prompt_lower or 'disgust' in prompt_lower:
            return 'disgust'
        elif '压抑' in prompt_lower or 'repression' in prompt_lower:
            return 'repression'
        else:
            return 'happiness'  # 默认


# =============================================================================
# Demo
# =============================================================================

def demo_hybrid():
    """演示混合模型"""
    print("\n" + "="*70)
    print("Hybrid Model: Diffusion + Blendshape + GAN")
    print("="*70)

    print("""
    架构：
      ┌─────────────────────────────────────────────────────────┐
      │  输入: 中性脸 + Blendshape条件                           │
      │    ↓                                                    │
      │  扩散模型: 去噪生成 (20步)                               │
      │    ↓                                                    │
      │  GAN判别器: 评估质量                                     │
      │    ↓                                                    │
      │  精修网络: 快速优化                                      │
      │    ↓                                                    │
      │  输出: 高质量微表情视频                                  │
      └─────────────────────────────────────────────────────────┘

    训练策略：
      Stage 1: 扩散预训练 (学习高质量生成)
      Stage 2: GAN微调 (快速优化)
      Stage 3: 联合训练 (最佳效果)

    优势：
      ✅ 扩散: 高质量、避免模式坍塌
      ✅ Blendshape: 精确控制52维参数
      ✅ GAN: 快速监督、识别器验证
      ✅ 精修: 快速优化细节
    """)

    print("="*70)


if __name__ == '__main__':
    demo_hybrid()