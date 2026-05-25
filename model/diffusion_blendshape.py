# =============================================================================
# Diffusion + Blendshape: Precision Micro-Expression Generation
# =============================================================================
# 扩散模型 + Blendshape精确控制
#
# 优势：
#   1. 扩散模型：高质量生成、可控性强、避免模式坍塌
#   2. Blendshape：精确控制、3D几何准确、行业标准
#   3. 结合：精确控制 + 高质量生成
#
# 流程：
#   Blendshape参数 → 条件扩散 → 生成微表情视频
#
# 参考：
#   - ControlNet (2023): 条件控制扩散
#   - DreamBooth: 个性化生成
#   - FaceDiffusion: 面部扩散
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


# =============================================================================
# 1. Blendshape System
# =============================================================================

class BlendshapeSystem:
    """
    Blendshape系统

    Blendshape是3D面部动画的标准：
      - ARKit: 52个blendshapes（苹果标准）
      - FACS: 46个AU（Ekman标准）
      - Faceware: 自定义

    微表情Blendshape映射：
      - AU12 → mouthSmile
      - AU1+AU2 → browInnerUp
      - AU4 → browDown
      - AU5 → eyeWide
      - AU9+AU10 → noseSneeze + mouthUpperUp
    """

    # ARKit 52 Blendshapes（苹果标准）
    ARKIT_BLENDSHAPES = {
        # 眉毛
        'browInnerUp': 0,       # AU1+AU2 (惊讶眉毛内侧)
        'browDownLeft': 1,      # AU4 (愤怒眉毛)
        'browDownRight': 2,
        'browOuterUpLeft': 3,   # AU2 (惊讶眉毛外侧)
        'browOuterUpRight': 4,

        # 眼睛
        'eyeLookDownLeft': 5,
        'eyeLookDownRight': 6,
        'eyeLookInLeft': 7,
        'eyeLookInRight': 8,
        'eyeLookOutLeft': 9,
        'eyeLookOutRight': 10,
        'eyeLookUpLeft': 11,
        'eyeLookUpRight': 12,
        'eyeBlinkLeft': 13,     # AU43 (眨眼)
        'eyeBlinkRight': 14,
        'eyeSquintLeft': 15,    # AU7 (眯眼)
        'eyeSquintRight': 16,
        'eyeWideLeft': 17,      # AU5 (惊讶眼睛张大)
        'eyeWideRight': 18,

        # 脸颊
        'cheek puff': 19,       # AU13 (脸颊鼓起)
        'cheekSquintLeft': 20,  # AU6 (微笑脸颊)
        'cheekSquintRight': 21,

        # 鼻子
        'noseSneeze': 22,       # AU9 (鼻子皱起)

        # 嘴巴
        'mouthClose': 23,       # AU24 (嘴唇闭合)
        'mouthFunnel': 24,      # AU22 (嘴唇收紧)
        'mouthPucker': 25,      # AU18 (嘴唇嘟起)
        'mouthLeft': 26,        # AU12左侧
        'mouthRight': 27,       # AU12右侧
        'mouthSmileLeft': 28,   # AU12 (微笑嘴角)
        'mouthSmileRight': 29,
        'mouthFrownLeft': 30,   # AU15 (悲伤嘴角)
        'mouthFrownRight': 31,
        'mouthDimpleLeft': 32,  # AU14 (嘴角凹陷)
        'mouthDimpleRight': 33,
        'mouthStretchLeft': 34, # AU20 (嘴唇拉伸)
        'mouthStretchRight': 35,
        'mouthRollLower': 36,   # AU17 (下巴上扬)
        'mouthRollUpper': 37,
        'mouthShrugLower': 38,
        'mouthShrugUpper': 39,
        'mouthPressLeft': 40,   # AU24 (嘴唇压缩)
        'mouthPressRight': 41,
        'mouthUpperUpLeft': 42, # AU10 (上唇上扬)
        'mouthUpperUpRight': 43,
        'mouthLowerDownLeft': 44,
        'mouthLowerDownRight': 45,

        # 下颌
        'jawForward': 46,       # AU26 (下颌前伸)
        'jawLeft': 47,
        'jawRight': 48,
        'jawOpen': 49,          # AU26 (下颌下落)

        # 嘴唇
        'chinRaiserLower': 50,  # AU17 (下巴上扬)
        'chinRaiserUpper': 51,
    }

    # FACS AU → Blendshape映射
    AU_TO_BLENDSHAPE = {
        'AU1': ['browInnerUp'],
        'AU2': ['browOuterUpLeft', 'browOuterUpRight'],
        'AU4': ['browDownLeft', 'browDownRight'],
        'AU5': ['eyeWideLeft', 'eyeWideRight'],
        'AU6': ['cheekSquintLeft', 'cheekSquintRight'],
        'AU7': ['eyeSquintLeft', 'eyeSquintRight'],
        'AU9': ['noseSneeze'],
        'AU10': ['mouthUpperUpLeft', 'mouthUpperUpRight'],
        'AU12': ['mouthSmileLeft', 'mouthSmileRight'],
        'AU14': ['mouthDimpleLeft', 'mouthDimpleRight'],
        'AU15': ['mouthFrownLeft', 'mouthFrownRight'],
        'AU17': ['chinRaiserLower', 'chinRaiserUpper'],
        'AU20': ['mouthStretchLeft', 'mouthStretchRight'],
        'AU24': ['mouthClose', 'mouthPressLeft', 'mouthPressRight'],
        'AU25': ['jawOpen'],  # Lips part
        'AU26': ['jawOpen', 'jawForward'],
        'AU43': ['eyeBlinkLeft', 'eyeBlinkRight'],
    }

    def __init__(self, num_blendshapes=52):
        self.num_blendshapes = num_blendshapes

    def au_to_blendshape(self, au_activation: torch.Tensor) -> torch.Tensor:
        """
        AU → Blendshape转换

        Args:
            au_activation: (B, 17) AU激活

        Returns:
            blendshape: (B, 52) Blendshape参数
        """
        B = au_activation.shape[0]
        blendshape = torch.zeros(B, self.num_blendshapes)

        # AU索引
        AU_INDEX = {
            'AU1': 0, 'AU2': 1, 'AU4': 2, 'AU5': 3, 'AU6': 4, 'AU7': 5,
            'AU9': 6, 'AU10': 7, 'AU12': 8, 'AU14': 9, 'AU15': 10,
            'AU17': 11, 'AU20': 12, 'AU24': 13, 'AU25': 14, 'AU26': 15,
        }

        for au_name, au_idx in AU_INDEX.items():
            if au_idx < au_activation.shape[1]:
                au_value = au_activation[:, au_idx]

                # 映射到blendshape
                blendshape_names = self.AU_TO_BLENDSHAPE.get(au_name, [])
                for bs_name in blendshape_names:
                    if bs_name in self.ARKIT_BLENDSHAPES:
                        bs_idx = self.ARKIT_BLENDSHAPES[bs_name]
                        blendshape[:, bs_idx] = au_value

        return blendshape

    def get_emotion_blendshape(self, emotion: str, intensity: float) -> torch.Tensor:
        """
        情感 → Blendshape

        Args:
            emotion: 情感类别
            intensity: 强度 (0-1)

        Returns:
            blendshape: (52,) Blendshape参数
        """
        emotion_blendshapes = {
            'happiness': {
                'mouthSmileLeft': intensity * 0.8,
                'mouthSmileRight': intensity * 0.8,
                'cheekSquintLeft': intensity * 0.6,
                'cheekSquintRight': intensity * 0.6,
            },
            'surprise': {
                'browInnerUp': intensity * 0.7,
                'browOuterUpLeft': intensity * 0.5,
                'browOuterUpRight': intensity * 0.5,
                'eyeWideLeft': intensity * 0.6,
                'eyeWideRight': intensity * 0.6,
                'jawOpen': intensity * 0.4,
            },
            'disgust': {
                'noseSneeze': intensity * 0.6,
                'mouthUpperUpLeft': intensity * 0.4,
                'mouthUpperUpRight': intensity * 0.4,
                'browDownLeft': intensity * 0.3,
                'browDownRight': intensity * 0.3,
            },
            'repression': {
                'mouthDimpleLeft': intensity * 0.5,
                'mouthDimpleRight': intensity * 0.5,
                'chinRaiserLower': intensity * 0.4,
                'mouthPressLeft': intensity * 0.3,
                'mouthPressRight': intensity * 0.3,
            },
        }

        blendshape = torch.zeros(self.num_blendshapes)

        if emotion in emotion_blendshapes:
            for bs_name, value in emotion_blendshapes[emotion].items():
                if bs_name in self.ARKIT_BLENDSHAPES:
                    bs_idx = self.ARKIT_BLENDSHAPES[bs_name]
                    blendshape[bs_idx] = value

        return blendshape


# =============================================================================
# 2. Diffusion Model Components
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码（用于时间步）"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class BlendshapeConditionEncoder(nn.Module):
    """
    Blendshape条件编码器

    将Blendshape参数编码为扩散模型的条件
    """

    def __init__(self, num_blendshapes=52, cond_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_blendshapes, 128),
            nn.ReLU(),
            nn.Linear(128, cond_dim),
            nn.ReLU(),
        )

    def forward(self, blendshape):
        """
        Args:
            blendshape: (B, 52) Blendshape参数

        Returns:
            cond: (B, cond_dim) 条件编码
        """
        return self.encoder(blendshape)


class MicroExpressionDiffusionUNet(nn.Module):
    """
    微表情扩散模型UNet

    结构：
      - 编码器：提取特征
      - 条件注入：Blendshape + 时间步
      - 解码器：生成视频帧
      - 注意力：保持时间一致性
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_frames=16,
                 cond_dim=256,
                 num_blendshapes=52):
        super().__init__()

        self.num_frames = num_frames

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
        )

        # Blendshape条件编码
        self.cond_encoder = BlendshapeConditionEncoder(num_blendshapes, cond_dim)

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 条件注入层（类似ControlNet）
        self.cond_injection = nn.ModuleList([
            nn.Linear(cond_dim, 256),
            nn.Linear(cond_dim, 256),
            nn.Linear(cond_dim, 256),
        ])

        # 中间层（带注意力）
        self.middle = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 时序注意力（保持帧间一致性）
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True,
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, blendshape):
        """
        Args:
            x: (B, C, T, H, W) 噪声视频
            t: (B,) 时间步
            blendshape: (B, 52) Blendshape条件

        Returns:
            output: (B, C, T, H, W) 去噪预测
        """
        B, C, T, H, W = x.shape

        # 时间编码
        t_emb = self.time_mlp(t)

        # Blendshape条件编码
        cond_emb = self.cond_encoder(blendshape)

        # 合并条件
        cond = t_emb + cond_emb

        # 编码
        feat = self.encoder(x)

        # 条件注入
        for i, injection in enumerate(self.cond_injection):
            cond_add = injection(cond)
            feat = feat + cond_add.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 中间处理
        feat = self.middle(feat)

        # 时序注意力
        # reshape为(B*T, H*W, C)进行注意力
        feat_flat = feat.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        feat_flat = feat_flat.reshape(B, T, 256, H*W)
        feat_flat = feat_flat.permute(0, 3, 1, 2)  # (B, H*W, T, C)
        feat_flat = feat_flat.reshape(B*H*W, T, 256)

        feat_attn, _ = self.temporal_attention(feat_flat, feat_flat, feat_flat)
        feat_attn = feat_attn.reshape(B, H*W, T, 256)
        feat_attn = feat_attn.permute(0, 2, 3, 1)
        feat_attn = feat_attn.reshape(B, 256, T, H, W)

        feat = feat + feat_attn

        # 解码
        output = self.decoder(feat)

        return output


# =============================================================================
# 3. Diffusion Process
# =============================================================================

class MicroExpressionDiffusion:
    """
    微表情扩散模型

    前向过程：添加噪声
    反向过程：去噪生成

    条件：Blendshape参数精确控制表情
    """

    def __init__(self,
                 model,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02):
        """
        Args:
            model: UNet模型
            num_timesteps: 时间步数
            beta_start: β起始值
            beta_end: β结束值
        """
        self.model = model
        self.num_timesteps = num_timesteps

        # β schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t):
        """
        前向扩散：添加噪声

        Args:
            x0: (B, C, T, H, W) 原始视频
            t: (B,) 时间步

        Returns:
            xt: 加噪后的视频
            noise: 添加的噪声
        """
        noise = torch.randn_like(x0)

        alpha_bar = self.alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

        return xt, noise

    def reverse_diffusion(self, xt, blendshape, num_steps=50):
        """
        反向扩散：去噪生成

        Args:
            xt: (B, C, T, H, W) 噪声视频
            blendshape: (B, 52) Blendshape条件
            num_steps: 去噪步数

        Returns:
            x0: 生成的视频
        """
        B = xt.shape[0]

        for i in tqdm(range(num_steps), desc="Denoising"):
            t = self.num_timesteps - i - 1
            t_tensor = torch.tensor([t] * B, dtype=torch.long)

            # 预测噪声
            noise_pred = self.model(xt, t_tensor, blendshape)

            # 去噪
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            if i < num_steps - 1:
                noise = torch.randn_like(xt)
            else:
                noise = 0

            xt = (xt - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred) / alpha.sqrt()
            xt = xt + self.betas[t].sqrt() * noise

        return xt

    def generate(self, neutral_face, blendshape, num_steps=50):
        """
        从中性脸生成微表情

        Args:
            neutral_face: (B, C, T, H, W) 中性脸视频
            blendshape: (B, 52) Blendshape条件
            num_steps: 去噪步数

        Returns:
            generated_video: 生成的微表情视频
        """
        # 从纯噪声开始
        xt = torch.randn_like(neutral_face)

        # 条件去噪
        x0 = self.reverse_diffusion(xt, blendshape, num_steps)

        return x0


# =============================================================================
# 4. Training
# =============================================================================

class DiffusionTrainer:
    """扩散模型训练器"""

    def __init__(self, model, lr=1e-4):
        self.model = model
        self.diffusion = MicroExpressionDiffusion(model)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.blendshape_system = BlendshapeSystem()

    def train_step(self, neutral_face, target_video, au_activation):
        """
        单步训练

        Args:
            neutral_face: (B, C, T, H, W)
            target_video: (B, C, T, H, W)
            au_activation: (B, 17)
        """
        B = neutral_face.shape[0]

        # AU → Blendshape
        blendshape = self.blendshape_system.au_to_blendshape(au_activation)

        # 随机时间步
        t = torch.randint(0, self.diffusion.num_timesteps, (B,))

        # 前向扩散
        xt, noise = self.diffusion.forward_diffusion(target_video, t)

        # 预测噪声
        noise_pred = self.model(xt, t, blendshape)

        # 损失
        loss = F.mse_loss(noise_pred, noise)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def generate(self, neutral_face, emotion, intensity):
        """
        生成微表情

        Args:
            neutral_face: (B, C, T, H, W)
            emotion: 情感类别
            intensity: 强度
        """
        # 情感 → Blendshape
        blendshape = self.blendshape_system.get_emotion_blendshape(emotion, intensity)
        blendshape = blendshape.unsqueeze(0).expand(neutral_face.shape[0], -1)

        # 生成
        generated = self.diffusion.generate(neutral_face, blendshape)

        return generated


# =============================================================================
# 5. Demo
# =============================================================================

def demo_diffusion_blendshape():
    """演示扩散模型 + Blendshape"""
    print("\n" + "="*70)
    print("Diffusion + Blendshape: Precision Micro-Expression Generation")
    print("="*70)

    # 创建系统
    blendshape_system = BlendshapeSystem()

    # 测试AU → Blendshape转换
    au_activation = torch.zeros(1, 17)
    au_activation[0, 8] = 0.8  # AU12 (微笑)

    blendshape = blendshape_system.au_to_blendshape(au_activation)
    print(f"\n[AU → Blendshape]")
    print(f"  AU12 = 0.8 → mouthSmileLeft = {blendshape[0, 28].item():.2f}")
    print(f"                   mouthSmileRight = {blendshape[0, 29].item():.2f}")

    # 测试情感 → Blendshape
    surprise_blendshape = blendshape_system.get_emotion_blendshape('surprise', 0.6)
    print(f"\n[Emotion → Blendshape]")
    print(f"  Surprise (intensity=0.6):")
    print(f"    browInnerUp: {surprise_blendshape[0].item():.2f}")
    print(f"    eyeWideLeft: {surprise_blendshape[17].item():.2f}")
    print(f"    jawOpen: {surprise_blendshape[49].item():.2f}")

    # 创建模型
    print(f"\n[Model Architecture]")
    model = MicroExpressionDiffusionUNet(
        in_channels=3,
        out_channels=3,
        num_frames=16,
        cond_dim=256,
        num_blendshapes=52,
    )
    print(f"  UNet created with 52 blendshape conditions")

    # 测试扩散过程
    print(f"\n[Diffusion Process]")
    diffusion = MicroExpressionDiffusion(model)

    # 模拟输入
    neutral_face = torch.randn(1, 3, 16, 64, 64)
    target_video = torch.randn(1, 3, 16, 64, 64)

    # 前向扩散
    t = torch.tensor([500])
    xt, noise = diffusion.forward_diffusion(target_video, t)
    print(f"  Forward diffusion at t=500:")
    print(f"    Input: {target_video.mean().item():.4f}")
    print(f"    Noisy: {xt.mean().item():.4f}")

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)


if __name__ == '__main__':
    demo_diffusion_blendshape()