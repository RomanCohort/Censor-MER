# =============================================================================
# LLM-based Prompt Analyzer for Micro-Expression Generation
# =============================================================================
# 使用LLM智能理解用户提示词，生成精细的AU参数
#
# 功能：
#   1. 智能情感识别（支持复杂描述）
#   2. AU参数精细化（多个AU组合）
#   3. 时间曲线定制（onset/apex/offset时长）
#   4. 多情感混合（如"先惊讶后微笑"）
# =============================================================================

import torch
import json
import os
import requests
from typing import Dict, List, Optional, Tuple
import re

from model.censor_g_generator import AU_INDEX


# =============================================================================
# LLM Interface
# =============================================================================

class LLMInterface:
    """
    LLM接口

    支持多种LLM后端：
      - OpenAI API (GPT-4)
      - Anthropic API (Claude)
      - DeepSeek API (国产，性价比高)
      - 本地模型 (Ollama)
      - AutoDL内置模型
    """

    def __init__(self,
                 backend: str = 'deepseek',
                 api_key: str = None,
                 model: str = None,
                 base_url: str = None):
        """
        Args:
            backend: LLM后端 ('openai', 'anthropic', 'deepseek', 'ollama', 'autodl')
            api_key: API密钥
            model: 模型名称
            base_url: 自定义API地址
        """
        self.backend = backend
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY', '')
        self.model = model or self._get_default_model()
        self.base_url = base_url

    def _get_default_model(self) -> str:
        """获取默认模型"""
        if self.backend == 'openai':
            return 'gpt-4o-mini'
        elif self.backend == 'anthropic':
            return 'claude-3-haiku-20240307'
        elif self.backend == 'deepseek':
            return 'deepseek-chat'  # 或 'deepseek-coder'
        elif self.backend == 'ollama':
            return 'llama3'
        elif self.backend == 'autodl':
            return 'local-model'
        return 'deepseek-chat'

    def call(self, prompt: str, system_prompt: str = None) -> str:
        """
        调用LLM

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词

        Returns:
            response: LLM响应文本
        """
        if self.backend == 'openai':
            return self._call_openai(prompt, system_prompt)
        elif self.backend == 'anthropic':
            return self._call_anthropic(prompt, system_prompt)
        elif self.backend == 'deepseek':
            return self._call_deepseek(prompt, system_prompt)
        elif self.backend == 'ollama':
            return self._call_ollama(prompt, system_prompt)
        elif self.backend == 'autodl':
            return self._call_autodl(prompt, system_prompt)
        else:
            # 降级到规则解析
            return self._fallback_parse(prompt)

    def _call_deepseek(self, prompt: str, system_prompt: str) -> str:
        """
        调用DeepSeek API

        DeepSeek API兼容OpenAI格式，性价比高：
          - deepseek-chat: 通用对话模型
          - deepseek-coder: 代码专用模型
        价格: ~0.001元/千tokens（比GPT-4便宜100倍）
        """
        try:
            import openai

            # DeepSeek API地址
            base_url = self.base_url or "https://api.deepseek.com/v1"

            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=base_url,
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"[LLMInterface] DeepSeek call failed: {e}")
            return self._fallback_parse(prompt)

    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        """调用OpenAI API"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLMInterface] OpenAI call failed: {e}")
            return self._fallback_parse(prompt)

    def _call_anthropic(self, prompt: str, system_prompt: str) -> str:
        """调用Anthropic API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"[LLMInterface] Anthropic call failed: {e}")
            return self._fallback_parse(prompt)

    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        """调用Ollama本地模型"""
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": self.model,
                "prompt": f"{system_prompt or ''}\n\n{prompt}",
                "stream": False,
            }
            response = requests.post(url, json=data, timeout=30)
            return response.json().get('response', '')
        except Exception as e:
            print(f"[LLMInterface] Ollama call failed: {e}")
            return self._fallback_parse(prompt)

    def _call_autodl(self, prompt: str, system_prompt: str) -> str:
        """调用AutoDL内置模型"""
        # AutoDL可能有自己的模型API
        try:
            # 假设使用本地模型或API
            url = self.base_url or "http://localhost:8000/v1/chat/completions"
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 500,
                "temperature": 0.3,
            }
            response = requests.post(url, json=data, timeout=30)
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        except Exception as e:
            print(f"[LLMInterface] AutoDL call failed: {e}")
            return self._fallback_parse(prompt)

    def _fallback_parse(self, prompt: str) -> str:
        """降级解析（规则方法）"""
        # 简单规则匹配
        emotion_keywords = {
            'happiness': ['微笑', '开心', '高兴', '快乐', 'smile', 'happy'],
            'surprise': ['惊讶', '吃惊', 'surprise', 'shock'],
            'disgust': ['厌恶', '恶心', 'disgust'],
            'repression': ['压抑', '悲伤', 'sad'],
        }

        emotion = 'happiness'
        intensity = 0.6

        for emo, keywords in emotion_keywords.items():
            for kw in keywords:
                if kw in prompt.lower():
                    emotion = emo
                    break

        # 返回JSON格式
        return json.dumps({
            "emotion": emotion,
            "intensity": intensity,
            "au": self._emotion_to_au_simple(emotion, intensity),
        })

    def _emotion_to_au_simple(self, emotion: str, intensity: float) -> Dict:
        """简单情感→AU映射"""
        mapping = {
            'happiness': {'AU6': 0.7, 'AU12': 0.8, 'AU25': 0.3},
            'surprise': {'AU1': 0.6, 'AU2': 0.6, 'AU5': 0.7, 'AU25': 0.5},
            'disgust': {'AU4': 0.5, 'AU9': 0.6, 'AU10': 0.4, 'AU17': 0.3},
            'repression': {'AU14': 0.5, 'AU17': 0.4, 'AU4': 0.3},
        }
        base = mapping.get(emotion, {})
        return {k: v * intensity for k, v in base.items()}


# =============================================================================
# LLM-based Prompt Analyzer
# =============================================================================

class LLMPromptAnalyzer:
    """
    LLM驱动的提示词分析器

    使用LLM理解用户提示词，生成精确的：
      - AU激活参数
      - 时间曲线参数
      - 多情感混合序列
    """

    SYSTEM_PROMPT = """
你是一个微表情生成专家。用户会描述想要生成的面部表情，你需要将其转换为FACS AU编码参数。

FACS (Facial Action Coding System) AU编码：
- AU1: Inner Brow Raiser (眉毛内侧上扬) - 惊讶
- AU2: Outer Brow Raiser (眉毛外侧上扬) - 惊讶
- AU4: Brow Lowerer (眉毛下压) - 厌恶、愤怒
- AU5: Upper Lid Raiser (上眼睑上扬) - 惊讶
- AU6: Cheek Raiser (脸颊上扬) - 微笑
- AU7: Lid Tightener (眼睑收紧)
- AU9: Nose Wrinkler (鼻子皱起) - 厌恶
- AU10: Upper Lip Raiser (上唇上扬) - 厌恶
- AU12: Lip Corner Puller (嘴角上扬) - 微笑
- AU14: Dimpler (嘴角凹陷) - 压抑
- AU15: Lip Corner Depressor (嘴角下压) - 悲伤
- AU17: Chin Raiser (下巴上扬)
- AU20: Lip Stretcher (嘴唇拉伸)
- AU23: Lip Tightener (嘴唇收紧)
- AU24: Lip Pressor (嘴唇压缩)
- AU25: Lips Part (嘴唇分开)
- AU26: Jaw Drop (下颌下落) - 惊讶

请将用户描述转换为JSON格式，包含：
1. emotion: 主要情感类别 (happiness/surprise/disgust/repression/fear/anger)
2. intensity: 整体强度 (0.1-1.0)
3. au: AU激活字典，格式 {"AU编号": 强度值}
4. temporal: 时间参数 (可选)
   - onset_frames: onset阶段帧数
   - apex_frames: apex阶段帧数
   - offset_frames: offset阶段帧数
5. sequence: 如果有多个情感，按时间顺序列出

示例：
用户输入: "一个标准的微笑"
输出:
{
  "emotion": "happiness",
  "intensity": 0.6,
  "au": {"AU6": 0.42, "AU12": 0.48, "AU25": 0.18},
  "temporal": {"onset_frames": 4, "apex_frames": 3, "offset_frames": 8}
}

用户输入: "先惊讶然后转为微笑"
输出:
{
  "sequence": [
    {"emotion": "surprise", "intensity": 0.5, "au": {"AU1": 0.3, "AU2": 0.3, "AU5": 0.35}, "duration": 5},
    {"emotion": "happiness", "intensity": 0.6, "au": {"AU6": 0.42, "AU12": 0.48}, "duration": 10}
  ]
}

请只返回JSON，不要添加额外解释。
"""

    def __init__(self,
                 llm_backend: str = 'deepseek',
                 api_key: str = None,
                 model: str = None):
        """
        Args:
            llm_backend: LLM后端 (默认deepseek)
            api_key: API密钥
            model: 模型名称
        """
        self.llm = LLMInterface(
            backend=llm_backend,
            api_key=api_key,
            model=model,
        )

    def analyze(self, user_prompt: str) -> Dict:
        """
        分析用户提示词

        Args:
            user_prompt: 用户输入（如"微笑"、"惊讶转为微笑"）

        Returns:
            analysis: 解析后的参数字典
        """
        # 调用LLM
        llm_response = self.llm.call(
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
        )

        # 解析JSON响应
        try:
            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = json.loads(llm_response)
        except json.JSONDecodeError:
            # 降级处理
            print(f"[LLMPromptAnalyzer] JSON parse failed, using fallback")
            analysis = self._fallback_analysis(user_prompt)

        # 转换AU名称为索引
        analysis = self._normalize_analysis(analysis)

        return analysis

    def _fallback_analysis(self, prompt: str) -> Dict:
        """降级分析"""
        prompt_lower = prompt.lower()

        if any(kw in prompt_lower for kw in ['微笑', '开心', 'smile', 'happy']):
            return {
                'emotion': 'happiness',
                'intensity': 0.6,
                'au': {'AU6': 0.42, 'AU12': 0.48, 'AU25': 0.18},
            }
        elif any(kw in prompt_lower for kw in ['惊讶', 'surprise']):
            return {
                'emotion': 'surprise',
                'intensity': 0.6,
                'au': {'AU1': 0.36, 'AU2': 0.36, 'AU5': 0.42, 'AU25': 0.30},
            }
        elif any(kw in prompt_lower for kw in ['厌恶', 'disgust']):
            return {
                'emotion': 'disgust',
                'intensity': 0.6,
                'au': {'AU4': 0.30, 'AU9': 0.36, 'AU10': 0.24, 'AU17': 0.18},
            }
        else:
            return {
                'emotion': 'happiness',
                'intensity': 0.5,
                'au': {'AU12': 0.4},
            }

    def _normalize_analysis(self, analysis: Dict) -> Dict:
        """标准化分析结果"""
        # 转换AU名称为张量索引
        if 'au' in analysis:
            au_tensor = torch.zeros(17)
            for au_name, value in analysis['au'].items():
                au_idx = AU_INDEX.get(au_name, None)
                if au_idx is not None:
                    au_tensor[au_idx] = float(value)
            analysis['au_tensor'] = au_tensor

        # 处理序列
        if 'sequence' in analysis:
            for item in analysis['sequence']:
                if 'au' in item:
                    au_tensor = torch.zeros(17)
                    for au_name, value in item['au'].items():
                        au_idx = AU_INDEX.get(au_name, None)
                        if au_idx is not None:
                            au_tensor[au_idx] = float(value)
                    item['au_tensor'] = au_tensor

        return analysis

    def analyze_batch(self, prompts: List[str]) -> List[Dict]:
        """批量分析"""
        return [self.analyze(p) for p in prompts]


# =============================================================================
# LLM驱动的完整生成器
# =============================================================================

class LLMDrivenGenerator:
    """
    LLM驱动的微表情生成器

    完整流程：
      1. 用户输入图片 + 自然语言提示词
      2. LLM理解提示词 → AU参数 + 时间曲线
      3. AU → 运动场 → 视频生成
    """

    def __init__(self,
                 checkpoint_path: str = None,
                 llm_backend: str = 'deepseek',
                 api_key: str = None,
                 image_size: int = 224,
                 num_frames: int = 16):
        """
        Args:
            checkpoint_path: 生成器checkpoint
            llm_backend: LLM后端 ('deepseek', 'openai', 'anthropic', 'ollama', 'autodl')
            api_key: LLM API密钥
            image_size: 图像尺寸
            num_frames: 帧数
        """
        from model.prompt_driven_generator import PromptDrivenGenerator

        self.generator = PromptDrivenGenerator(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            num_frames=num_frames,
        )
        self.llm_analyzer = LLMPromptAnalyzer(
            llm_backend=llm_backend,
            api_key=api_key,
        )

    def generate(self,
                 image_path: str,
                 prompt: str,
                 output_path: str = None) -> Dict:
        """
        LLM驱动的生成

        Args:
            image_path: 用户图片
            prompt: 自然语言提示词
            output_path: 输出路径

        Returns:
            result: 生成结果
        """
        print(f"[LLMDrivenGenerator] Processing: '{prompt}'")

        # 1. LLM分析提示词
        analysis = self.llm_analyzer.analyze(prompt)
        print(f"  LLM Analysis: emotion={analysis.get('emotion')}, intensity={analysis.get('intensity')}")

        # 2. 获取AU参数
        au_tensor = analysis.get('au_tensor', torch.zeros(17))

        # 打印激活的AU
        active_au = []
        for au_name, idx in AU_INDEX.items():
            if au_tensor[idx] > 0.05:
                active_au.append(f"{au_name}={au_tensor[idx].item():.2f}")
        print(f"  Active AU: {', '.join(active_au) if active_au else 'none'}")

        # 3. 加载图像
        image = self.generator._load_image(image_path)

        # 4. 生成视频
        with torch.no_grad():
            video, motions = self.generator.generator(image, au_tensor)

        # 5. 保存
        if output_path:
            self.generator._save_video(video, output_path)
            print(f"  Saved: {output_path}")

        return {
            'video': video,
            'analysis': analysis,
            'au_tensor': au_tensor,
            'output_path': output_path,
        }


# =============================================================================
# Demo
# =============================================================================

def demo_llm_generator():
    """演示LLM驱动的生成"""
    print("\n" + "="*60)
    print("LLM-Driven Micro-Expression Generator Demo")
    print("="*60)

    # 创建分析器（使用降级模式，因为没有API key）
    analyzer = LLMPromptAnalyzer(llm_backend='deepseek', api_key='dummy')

    # 测试提示词
    test_prompts = [
        "微笑",
        "一个标准的惊讶表情",
        "轻微的厌恶感",
        "压抑的悲伤",
        "先惊讶然后转为微笑",
        "强烈的愤怒",
        "一个微表情：嘴角微微上扬的喜悦",
    ]

    print("\n[1] Prompt Analysis Test")
    for prompt in test_prompts:
        analysis = analyzer.analyze(prompt)
        print(f"\n  Prompt: '{prompt}'")
        print(f"    → Emotion: {analysis.get('emotion', 'N/A')}")
        print(f"    → Intensity: {analysis.get('intensity', 'N/A')}")

        au = analysis.get('au', {})
        active_au = [f"{k}:{v:.2f}" for k, v in au.items() if v > 0.1]
        print(f"    → AU: {', '.join(active_au) if active_au else 'none'}")

        if 'temporal' in analysis:
            print(f"    → Temporal: onset={analysis['temporal'].get('onset_frames')}, apex={analysis['temporal'].get('apex_frames')}")

        if 'sequence' in analysis:
            print(f"    → Sequence: {len(analysis['sequence'])} stages")

    print("\n[2] Complex Prompt Handling")
    # 测试复杂提示词（多情感混合）
    complex_prompt = "先表现出惊讶的表情，然后逐渐转为会心的微笑"
    analysis = analyzer.analyze(complex_prompt)
    print(f"\n  Prompt: '{complex_prompt}'")
    if 'sequence' in analysis:
        for i, stage in enumerate(analysis['sequence']):
            print(f"    Stage {i+1}: {stage.get('emotion')} (intensity={stage.get('intensity')})")

    print("\n[3] Usage Example")
    print("  # With DeepSeek API (推荐，性价比高):")
    print("  gen = LLMDrivenGenerator(")
    print("      checkpoint_path='model.pth',")
    print("      llm_backend='deepseek',")
    print("      api_key='your-deepseek-api-key'")
    print("  )")
    print("  gen.generate('user_photo.jpg', '微笑', 'output.mp4')")
    print("  gen.generate('user_photo.jpg', '先惊讶后微笑', 'output2.mp4')")
    print("")
    print("  # With OpenAI API:")
    print("  gen = LLMDrivenGenerator(")
    print("      checkpoint_path='model.pth',")
    print("      llm_backend='openai',")
    print("      api_key='your-openai-key'")
    print("  )")
    print("  gen.generate('user_photo.jpg', '微笑', 'output.mp4')")
    print("  gen.generate('user_photo.jpg', '先惊讶后微笑', 'output2.mp4')")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_llm_generator()