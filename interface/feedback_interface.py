# =============================================================================
# Web Interface for Micro-Expression Feedback Collection
# =============================================================================
# 使用Gradio搭建Web界面，收集人类反馈
#
# 功能：
#   1. 用户上传图片 + 输入提示词
#   2. 生成微表情视频
#   3. 用户评分（自然度、流畅度、匹配度）
#   4. 保存反馈数据用于RLHF
# =============================================================================

import gradio as gr
import torch
import numpy as np
import cv2
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 导入模型
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.censor_g_generator import CensorGGenerator
from model.prompt_driven_generator import PromptDrivenGenerator
from model.llm_prompt_analyzer import LLMPromptAnalyzer, LLMDrivenGenerator


# =============================================================================
# Feedback Storage
# =============================================================================

class FeedbackStorage:
    """反馈数据存储"""

    def __init__(self, storage_dir='./feedback_data'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.feedback_file = os.path.join(storage_dir, 'feedback.json')
        self.feedback_data = self._load_feedback()

    def _load_feedback(self) -> List[Dict]:
        """加载已有反馈"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []

    def save_feedback(self, feedback: Dict):
        """保存反馈"""
        feedback['timestamp'] = datetime.now().isoformat()
        feedback['id'] = len(self.feedback_data) + 1

        self.feedback_data.append(feedback)

        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)

        print(f"[FeedbackStorage] Saved feedback #{feedback['id']}")

        return feedback['id']

    def get_statistics(self) -> Dict:
        """获取反馈统计"""
        if not self.feedback_data:
            return {}

        # 计算平均分数
        avg_naturalness = np.mean([f['naturalness'] for f in self.feedback_data])
        avg_smoothness = np.mean([f['smoothness'] for f in self.feedback_data])
        avg_match = np.mean([f['prompt_match'] for f in self.feedback_data])
        avg_overall = np.mean([f['overall'] for f in self.feedback_data])

        return {
            'total_feedback': len(self.feedback_data),
            'avg_naturalness': avg_naturalness,
            'avg_smoothness': avg_smoothness,
            'avg_prompt_match': avg_match,
            'avg_overall': avg_overall,
        }


# =============================================================================
# Video Generator Wrapper
# =============================================================================

class VideoGeneratorWrapper:
    """视频生成器包装"""

    def __init__(self, checkpoint_path: str = None, use_llm: bool = True, api_key: str = None):
        """
        Args:
            checkpoint_path: 模型checkpoint路径
            use_llm: 是否使用LLM分析提示词
            api_key: DeepSeek API key
        """
        self.use_llm = use_llm

        if use_llm and api_key:
            self.generator = LLMDrivenGenerator(
                checkpoint_path=checkpoint_path,
                llm_backend='deepseek',
                api_key=api_key,
            )
            print("[VideoGeneratorWrapper] Using LLM-driven generator (DeepSeek)")
        else:
            self.generator = PromptDrivenGenerator(
                checkpoint_path=checkpoint_path,
            )
            print("[VideoGeneratorWrapper] Using prompt-driven generator (local)")

    def generate(self, image_path: str, prompt: str, output_path: str = None) -> Tuple[str, Dict]:
        """
        生成微表情视频

        Returns:
            video_path: 生成的视频路径
            info: 生成信息
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp4')

        try:
            result = self.generator.generate(
                image_path=image_path,
                prompt=prompt,
                output_path=output_path,
            )

            info = {
                'emotion': result.get('emotion', 'unknown'),
                'intensity': result.get('intensity', 0.6),
                'active_au': self._format_active_au(result.get('au', None)),
            }

            return output_path, info

        except Exception as e:
            print(f"[VideoGeneratorWrapper] Error: {e}")
            # 返回模拟视频
            return self._generate_dummy_video(output_path), {'error': str(e)}

    def _format_active_au(self, au_tensor) -> str:
        """格式化活跃的AU"""
        if au_tensor is None:
            return "N/A"

        from model.censor_g_generator import AU_INDEX

        active = []
        for au_name, idx in AU_INDEX.items():
            if au_tensor[idx] > 0.1:
                active.append(f"{au_name}:{au_tensor[idx].item():.2f}")

        return ", ".join(active) if active else "none"

    def _generate_dummy_video(self, output_path: str) -> str:
        """生成模拟视频（用于错误情况）"""
        # 创建简单动画
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (224, 224))

        for i in range(16):
            # 创建渐变帧
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:, :] = [i * 15, 100, 200]
            writer.write(frame)

        writer.release()
        return output_path


# =============================================================================
# Gradio Interface
# =============================================================================

def create_feedback_interface(checkpoint_path: str = None, api_key: str = None):
    """创建Gradio反馈收集界面"""

    # 初始化组件
    storage = FeedbackStorage()
    generator = VideoGeneratorWrapper(
        checkpoint_path=checkpoint_path,
        use_llm=False,  # 不使用LLM，直接本地生成
        api_key=api_key,
    )

    # CSS样式
    css = """
    .feedback-container {
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }
    .rating-slider {
        width: 100%;
    }
    .video-display {
        width: 400px;
        height: 300px;
    }
    .statistics-box {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .batch-test-box {
        background: #e8f4e8;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    """

    # === 主界面 ===
    with gr.Blocks(css=css, title="微表情生成反馈收集") as interface:

        gr.Markdown("""
        # 微表情生成反馈收集系统

        **使用流程：**
        1. 上传基准图片（中性脸）
        2. 输入提示词（如"微笑"、"惊讶"、"轻微厌恶"）
        3. 点击生成，观看生成的视频
        4. 对生成效果进行评分
        5. 提交反馈

        **提示词示例：**
        - 简单：微笑、惊讶、厌恶
        - 强度：轻微微笑、强烈惊讶
        - 复杂：先惊讶后微笑
        """)

        # === 第一部分：生成 ===
        with gr.Row():
            with gr.Column(scale=1):
                # 输入
                image_input = gr.Image(
                    label="上传基准图片",
                    type="filepath",
                    height=300,
                )

                prompt_input = gr.Textbox(
                    label="输入提示词",
                    placeholder="例如：微笑、惊讶、轻微厌恶...",
                    lines=2,
                )

                generate_btn = gr.Button(
                    "生成微表情视频",
                    variant="primary",
                    size="lg",
                )

                # 生成信息
                generation_info = gr.JSON(
                    label="生成信息",
                    visible=True,
                )

            with gr.Column(scale=1):
                # 输出
                video_output = gr.Video(
                    label="生成的微表情视频",
                    height=300,
                )

                # 参考：真实微表情（可选）
                with gr.Accordion("参考视频（可选）", open=False):
                    reference_video = gr.Video(
                        label="真实微表情参考",
                        height=200,
                    )

        # === 第二部分：评分 ===
        gr.Markdown("## 请对生成效果进行评分")

        with gr.Row():
            naturalness_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=0.5,
                value=3,
                label="自然度 (表情是否自然)",
                info="1=非常僵硬, 5=非常自然",
            )

            smoothness_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=0.5,
                value=3,
                label="流畅度 (运动是否流畅)",
                info="1=卡顿, 5=非常流畅",
            )

        with gr.Row():
            match_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=0.5,
                value=3,
                label="匹配度 (是否符合提示词)",
                info="1=完全不匹配, 5=完美匹配",
            )

            overall_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=0.5,
                value=3,
                label="总体评分",
                info="1=很差, 5=很好",
            )

        # 专家级评分（可选）
        with gr.Accordion("专家级评分（可选）", open=False):
            gr.Markdown("如果您是微表情/FACS专家，请填写以下评分：")

            with gr.Row():
                au_accuracy_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=0.5,
                    value=3,
                    label="AU准确性",
                    info="生成的AU激活是否正确",
                )

                micro_quality_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=0.5,
                    value=3,
                    label="微表情质量",
                    info="是否符合微表情定义（时长<500ms）",
                )

            expert_comments = gr.Textbox(
                label="专家评语",
                placeholder="请描述生成效果的问题或优点...",
                lines=3,
            )

        # 评论
        user_comments = gr.Textbox(
            label="用户评论（可选）",
            placeholder="请描述您对生成效果的感受...",
            lines=2,
        )

        # 提交按钮
        submit_btn = gr.Button(
            "提交反馈",
            variant="secondary",
            size="lg",
        )

        # 提交结果
        submit_result = gr.Textbox(
            label="提交结果",
            visible=True,
        )

        # === 第三部分：统计 ===
        gr.Markdown("## 反馈统计")

        statistics_display = gr.JSON(
            label="当前反馈统计",
            value=storage.get_statistics(),
        )

        refresh_stats_btn = gr.Button("刷新统计")
        download_feedback_btn = gr.Button("下载反馈数据")

        # === 事件处理 ===

        # 生成视频
        def generate_video(image_path, prompt):
            if image_path is None:
                return None, {"error": "请先上传图片"}

            if not prompt:
                return None, {"error": "请输入提示词"}

            video_path, info = generator.generate(image_path, prompt)
            return video_path, info

        generate_btn.click(
            fn=generate_video,
            inputs=[image_input, prompt_input],
            outputs=[video_output, generation_info],
        )

        # 提交反馈
        def submit_feedback(
            image_path,
            prompt,
            video_path,
            naturalness,
            smoothness,
            match,
            overall,
            au_accuracy,
            micro_quality,
            expert_comment,
            user_comment,
        ):
            if video_path is None:
                return "请先生成视频"

            feedback = {
                'image_path': image_path,
                'prompt': prompt,
                'video_path': video_path,
                'naturalness': naturalness,
                'smoothness': smoothness,
                'prompt_match': match,
                'overall': overall,
                'au_accuracy': au_accuracy if au_accuracy != 3 else None,
                'micro_quality': micro_quality if micro_quality != 3 else None,
                'expert_comments': expert_comment if expert_comment else None,
                'user_comments': user_comment if user_comment else None,
            }

            feedback_id = storage.save_feedback(feedback)

            return f"反馈 #{feedback_id} 已保存！感谢您的反馈。"

        submit_btn.click(
            fn=submit_feedback,
            inputs=[
                image_input,
                prompt_input,
                video_output,
                naturalness_slider,
                smoothness_slider,
                match_slider,
                overall_slider,
                au_accuracy_slider,
                micro_quality_slider,
                expert_comments,
                user_comments,
            ],
            outputs=[submit_result],
        )

        # 刷新统计
        def refresh_statistics():
            return storage.get_statistics()

        refresh_stats_btn.click(
            fn=refresh_statistics,
            outputs=[statistics_display],
        )

        # 下载反馈
        def download_feedback():
            return storage.feedback_file

        download_feedback_btn.click(
            fn=download_feedback,
            outputs=[gr.File()],
        )

    return interface


# =============================================================================
# Comparison Interface (for pairwise feedback)
# =============================================================================

def create_comparison_interface(checkpoint_path: str = None, api_key: str = None):
    """创建比较式反馈界面（用于选择哪个更好）"""

    storage = FeedbackStorage()
    generator = VideoGeneratorWrapper(
        checkpoint_path=checkpoint_path,
        use_llm=True,
        api_key=api_key,
    )

    with gr.Blocks(title="微表情生成比较") as interface:

        gr.Markdown("""
        # 微表情生成比较

        **使用流程：**
        1. 上传基准图片
        2. 输入提示词
        3. 生成两个候选视频
        4. 选择更好的一个
        5. 提交比较反馈
        """)

        # 输入
        with gr.Row():
            image_input = gr.Image(label="上传基准图片", type="filepath")
            prompt_input = gr.Textbox(label="输入提示词", placeholder="微笑...")

        generate_btn = gr.Button("生成两个候选视频")

        # 输出
        with gr.Row():
            video1_output = gr.Video(label="候选视频 1")
            video2_output = gr.Video(label="候选视频 2")

        # 比较
        gr.Markdown("## 请选择更好的视频")

        with gr.Row():
            choice_radio = gr.Radio(
                choices=["视频1更好", "视频2更好", "两者差不多", "两者都不好"],
                label="您的选择",
            )

            reason_text = gr.Textbox(
                label="选择原因（可选）",
                placeholder="为什么选择这个视频？",
                lines=2,
            )

        submit_btn = gr.Button("提交比较反馈")
        result_text = gr.Textbox(label="提交结果")

        # 事件处理
        def generate_two_videos(image_path, prompt):
            if image_path is None or not prompt:
                return None, None

            # 生成两个视频（使用不同的参数扰动）
            video1_path, info1 = generator.generate(image_path, prompt, tempfile.mktemp(suffix='.mp4'))

            # 略微修改prompt或参数
            video2_path, info2 = generator.generate(image_path, prompt, tempfile.mktemp(suffix='.mp4'))

            return video1_path, video2_path

        generate_btn.click(
            fn=generate_two_videos,
            inputs=[image_input, prompt_input],
            outputs=[video1_output, video2_output],
        )

        def submit_comparison(image_path, prompt, video1_path, video2_path, choice, reason):
            if not choice:
                return "请选择一个选项"

            feedback = {
                'type': 'comparison',
                'image_path': image_path,
                'prompt': prompt,
                'video1_path': video1_path,
                'video2_path': video2_path,
                'choice': choice,
                'reason': reason,
            }

            feedback_id = storage.save_feedback(feedback)
            return f"比较反馈 #{feedback_id} 已保存！"

        submit_btn.click(
            fn=submit_comparison,
            inputs=[image_input, prompt_input, video1_output, video2_output, choice_radio, reason_text],
            outputs=[result_text],
        )

    return interface


# =============================================================================
# Launch Function
# =============================================================================

def launch_interface(checkpoint_path: str = None,
                     api_key: str = None,
                     share: bool = True,
                     port: int = 7860):
    """
    启动Web界面

    Args:
        checkpoint_path: 模型checkpoint路径
        api_key: DeepSeek API key
        share: 是否创建公网链接
        port: 端口号
    """
    print("\n" + "="*60)
    print("Micro-Expression Feedback Collection Interface")
    print("="*60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"API Key: {'provided' if api_key else 'not provided'}")
    print(f"Share: {share}")
    print(f"Port: {port}")

    # 创建界面
    interface = create_feedback_interface(
        checkpoint_path=checkpoint_path,
        api_key=api_key,
    )

    # 启动
    interface.launch(
        share=share,
        server_port=port,
        server_name="0.0.0.0",
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launch feedback collection interface')
    parser.add_argument('--checkpoint', type=str, default=None, help='Generator checkpoint path')
    parser.add_argument('--api_key', type=str, default=None, help='DeepSeek API key')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--port', type=int, default=7860, help='Server port')

    args = parser.parse_args()

    launch_interface(
        checkpoint_path=args.checkpoint,
        api_key=args.api_key,
        share=args.share,
        port=args.port,
    )