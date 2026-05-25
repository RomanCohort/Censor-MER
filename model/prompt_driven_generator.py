# =============================================================================
# Prompt-Driven Micro-Expression Generator
# =============================================================================
# 用户场景：输入基准图片 + 提示词，输出微表情动画
#
# 示例：
#   输入：user_photo.jpg + "微笑"
#   输出：微笑微表情动画视频
#
# 架构：
#   1. PromptParser: 提示词 → 情感分类 + 强度
#   2. EmotionToAU: 情感 → AU激活向量
#   3. Generator: AU → 运动场 → 视频生成
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import cv2
import os

# 导入基础生成器
from model.censor_g_generator import CensorGGenerator, AU_INDEX


# =============================================================================
# Part 1: 提示词解析器
# =============================================================================

class PromptParser:
    """
    提示词解析器

    将用户输入的提示词转换为：
      - 情感类别（happiness, surprise, disgust, repression）
      - 强度等级（0.0-1.0）
      - 可选的附加参数（持续时间、速度等）
    """

    # 情感关键词映射
    EMOTION_KEYWORDS = {
        'happiness': ['微笑', '开心', '高兴', '快乐', '笑', '喜悦', 'smile', 'happy', 'joy', 'laugh'],
        'surprise': ['惊讶', '吃惊', '惊喜', '震惊', '惊讶', 'surprise', 'shock', 'wow'],
        'disgust': ['厌恶', '恶心', '嫌弃', '反感', '皱眉', 'disgust', 'yuck'],
        'repression': ['压抑', '悲伤', '难过', '沮丧', '沉思', '忧郁', 'sad', 'depressed', 'sadness'],
        'fear': ['恐惧', '害怕', '惊恐', 'fear', 'scared'],
        'anger': ['愤怒', '生气', '恼怒', 'anger', 'angry'],
    }

    # 强度关键词映射
    INTENSITY_KEYWORDS = {
        'strong': ['强烈', '很大', '明显', '夸张', 'strong', 'intense', 'big'],
        'medium': ['中等', '适中', '一般', 'medium', 'moderate'],
        'weak': ['轻微', '小', '细微', '微弱', 'weak', 'subtle', 'small'],
        'micro': ['微表情', '微小', '细微', '极小', 'micro', 'tiny'],
    }

    # 速度关键词映射
    SPEED_KEYWORDS = {
        'fast': ['快速', '快', '突然', 'fast', 'quick', 'sudden'],
        'normal': ['正常', '一般', 'normal', 'regular'],
        'slow': ['慢速', '慢', '缓慢', 'slow', 'gradual'],
    }

    def parse(self, prompt: str) -> Dict:
        """
        解析提示词

        Args:
            prompt: 用户输入的提示词（如"微笑"、"轻微惊讶"）

        Returns:
            dict: {
                'emotion': str,       # 情感类别
                'intensity': float,   # 强度 (0.0-1.0)
                'speed': str,         # 速度类别
                'duration': float,    # 持续时间（可选）
            }
        """
        prompt_lower = prompt.lower()

        # 识别情感
        emotion = None
        for emo, keywords in self.EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower:
                    emotion = emo
                    break
            if emotion:
                break

        if emotion is None:
            emotion = 'happiness'  # 默认微笑

        # 识别强度
        intensity = 0.6  # 默认中等强度
        for level, keywords in self.INTENSITY_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower:
                    if level == 'strong':
                        intensity = 0.8
                    elif level == 'medium':
                        intensity = 0.6
                    elif level == 'weak':
                        intensity = 0.4
                    elif level == 'micro':
                        intensity = 0.25  # 微表情强度较低
                    break

        # 识别速度
        speed = 'normal'
        for spd, keywords in self.SPEED_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower:
                    speed = spd
                    break

        return {
            'emotion': emotion,
            'intensity': intensity,
            'speed': speed,
            'original_prompt': prompt,
        }

    def parse_batch(self, prompts: List[str]) -> List[Dict]:
        """批量解析提示词"""
        return [self.parse(p) for p in prompts]


# =============================================================================
# Part 2: 情感到AU转换器
# =============================================================================

class EmotionToAUConverter:
    """
    情感到AU激活转换器

    将情感类别转换为FACS AU激活向量
    """

    # 情感到AU的详细映射（基于FACS研究）
    EMOTION_AU_MAPPING = {
        'happiness': {
            'AU6': 0.7,   # Cheek raiser
            'AU12': 0.8,  # Lip corner puller
            'AU25': 0.3,  # Lips part
            'AU7': 0.3,   # Lid tightener
        },
        'surprise': {
            'AU1': 0.6,   # Inner brow raiser
            'AU2': 0.6,   # Outer brow raiser
            'AU5': 0.7,   # Upper lid raiser
            'AU25': 0.5,  # Lips part
            'AU26': 0.4,  # Jaw drop
        },
        'disgust': {
            'AU4': 0.5,   # Brow lowerer
            'AU9': 0.6,   # Nose wrinkler
            'AU10': 0.4,  # Upper lip raiser
            'AU17': 0.3,  # Chin raiser
        },
        'repression': {
            'AU14': 0.5,  # Dimpler
            'AU17': 0.4,  # Chin raiser
            'AU4': 0.3,   # Brow lowerer
        },
        'fear': {
            'AU1': 0.5,
            'AU2': 0.5,
            'AU4': 0.4,
            'AU5': 0.6,
            'AU20': 0.4,  # Lip stretcher
        },
        'anger': {
            'AU4': 0.6,
            'AU5': 0.4,
            'AU7': 0.5,
            'AU23': 0.4,  # Lip tightener
            'AU24': 0.3,  # Lip presser
        },
    }

    def convert(self, emotion: str, intensity: float = 0.6) -> torch.Tensor:
        """
        情感转换为AU激活向量

        Args:
            emotion: 情感类别
            intensity: 强度 (0.0-1.0)

        Returns:
            au_activation: (17,) AU激活向量
        """
        au = torch.zeros(17)

        # 获取该情感的AU映射
        au_mapping = self.EMOTION_AU_MAPPING.get(emotion, {})

        # 应用强度调制
        for au_name, base_value in au_mapping.items():
            au_idx = AU_INDEX.get(au_name, None)
            if au_idx is not None:
                # 强度调制：base_value * intensity
                au[au_idx] = base_value * intensity

        return au

    def convert_batch(self, emotions: List[str], intensities: List[float]) -> torch.Tensor:
        """批量转换"""
        au_batch = []
        for emotion, intensity in zip(emotions, intensities):
            au_batch.append(self.convert(emotion, intensity))
        return torch.stack(au_batch)


# =============================================================================
# Part 3: 完整的提示词驱动生成器
# =============================================================================

class PromptDrivenGenerator:
    """
    提示词驱动微表情生成器

    完整流程：
      1. 用户输入图片 + 提示词
      2. 解析提示词 → 情感 + 强度
      3. 情感 → AU激活
      4. AU → 运动场 → 视频生成
    """

    def __init__(self, checkpoint_path: str = None, image_size: int = 224, num_frames: int = 16):
        """
        Args:
            checkpoint_path: 生成器checkpoint路径（可选）
            image_size: 输出图像尺寸
            num_frames: 输出视频帧数
        """
        self.image_size = image_size
        self.num_frames = num_frames

        # 创建组件
        self.prompt_parser = PromptParser()
        self.emotion_converter = EmotionToAUConverter()
        self.generator = CensorGGenerator(
            num_au=17,
            num_keypoints=68,
            num_frames=num_frames,
            image_size=image_size,
        )

        # 加载预训练权重（如果有）
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
            self.generator.load_state_dict(ckpt['generator'])
            print(f"[PromptDrivenGenerator] Loaded checkpoint: {checkpoint_path}")

        self.generator.eval()

    def generate(self,
                 image_path: str,
                 prompt: str,
                 output_path: str = None,
                 return_video: bool = True) -> Dict:
        """
        生成微表情视频

        Args:
            image_path: 输入图像路径
            prompt: 提示词（如"微笑"、"惊讶"）
            output_path: 输出视频路径（可选）
            return_video: 是否返回视频张量

        Returns:
            dict: {
                'video': Tensor,       # 生成的视频
                'emotion': str,        # 情感类别
                'au': Tensor,          # AU激活
                'frames': int,         # 帧数
                'output_path': str,    # 输出路径（如果保存了）
            }
        """
        # 1. 加载图像
        image = self._load_image(image_path)

        # 2. 解析提示词
        parsed = self.prompt_parser.parse(prompt)
        emotion = parsed['emotion']
        intensity = parsed['intensity']

        print(f"[PromptDrivenGenerator] Prompt: '{prompt}'")
        print(f"  → Emotion: {emotion}, Intensity: {intensity:.2f}")

        # 3. 情感 → AU
        au_activation = self.emotion_converter.convert(emotion, intensity)

        # 打印激活的AU
        active_au = []
        for au_name, idx in AU_INDEX.items():
            if au_activation[idx] > 0.1:
                active_au.append(f"{au_name}={au_activation[idx].item():.2f}")
        print(f"  → Active AU: {', '.join(active_au)}")

        # 4. 生成视频
        with torch.no_grad():
            video, motions = self.generator(image, au_activation)

        # 5. 保存输出（可选）
        if output_path:
            self._save_video(video, output_path)
            print(f"  → Saved to: {output_path}")

        result = {
            'video': video if return_video else None,
            'emotion': emotion,
            'intensity': intensity,
            'au': au_activation,
            'frames': self.num_frames,
            'prompt': prompt,
            'output_path': output_path,
        }

        return result

    def generate_batch(self,
                       image_paths: List[str],
                       prompts: List[str],
                       output_dir: str = None) -> List[Dict]:
        """批量生成"""
        results = []
        for i, (img_path, prompt) in enumerate(zip(image_paths, prompts)):
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"output_{i}.mp4")
            result = self.generate(img_path, prompt, output_path)
            results.append(result)
        return results

    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载图像"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image = image.unsqueeze(0)  # (1, C, H, W)

        return image

    def _save_video(self, video: torch.Tensor, output_path: str):
        """保存视频"""
        # video: (1, C, T, H, W)
        video_np = video[0].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        video_np = (video_np * 255).astype(np.uint8)
        video_np = video_np[..., ::-1]  # RGB to BGR

        # 使用cv2保存
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30,
                                 (self.image_size, self.image_size))

        for frame in video_np:
            writer.write(frame)

        writer.release()


# =============================================================================
# Demo: 交互式生成
# =============================================================================

def demo_prompt_generator():
    """演示提示词驱动生成"""
    print("\n" + "="*60)
    print("Prompt-Driven Micro-Expression Generator Demo")
    print("="*60)

    # 创建生成器（使用模拟数据，因为没有真实checkpoint）
    generator = PromptDrivenGenerator(checkpoint_path=None)

    # 测试提示词解析
    test_prompts = [
        "微笑",
        "轻微惊讶",
        "强烈厌恶",
        "压抑悲伤",
        "smile",
        "big surprise",
    ]

    print("\n[1] Prompt Parsing Test")
    for prompt in test_prompts:
        parsed = generator.prompt_parser.parse(prompt)
        print(f"  '{prompt}' → emotion={parsed['emotion']}, intensity={parsed['intensity']:.2f}")

    # 测试情感→AU转换
    print("\n[2] Emotion → AU Conversion Test")
    emotions = ['happiness', 'surprise', 'disgust', 'repression']
    for emotion in emotions:
        au = generator.emotion_converter.convert(emotion, intensity=0.6)
        active_au = []
        for au_name, idx in AU_INDEX.items():
            if au[idx] > 0.1:
                active_au.append(f"{au_name}={au[idx].item():.2f}")
        print(f"  {emotion}: {', '.join(active_au)}")

    # 测试完整生成流程（模拟图像）
    print("\n[3] Full Generation Test (Simulated)")

    # 创建模拟图像
    sim_image = torch.randn(1, 3, 224, 224) * 0.1 + 0.5

    for prompt in ["微笑", "惊讶"]:
        parsed = generator.prompt_parser.parse(prompt)
        au = generator.emotion_converter.convert(parsed['emotion'], parsed['intensity'])

        with torch.no_grad():
            video, motions = generator.generator(sim_image, au)

        # 分析运动
        motion_mag = motions[0].abs().mean().item()
        print(f"  '{prompt}' → video shape {video.shape}, motion mag {motion_mag:.4f}")

    print("\n[4] Interactive Mode")
    print("  Usage:")
    print("    generator.generate('user_photo.jpg', '微笑', 'output.mp4')")
    print("    generator.generate('user_photo.jpg', '惊讶', 'output.mp4')")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_prompt_generator()