# =============================================================================
# Streamlit Web Interface for Micro-Expression Feedback Collection
# =============================================================================
# 替代Gradio界面，使用Streamlit实现：
#   1. 单次评分 - 上传图片、生成视频、评分
#   2. 批量测试 - 多图片批量生成和评分
#   3. 提示词库 - 预定义模板选择
#   4. 历史记录 - 查看、编辑、删除反馈
#   5. 数据导出 - JSON/CSV/Excel导出
# =============================================================================

import streamlit as st
import torch
import numpy as np
import cv2
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd

# 设置页面
st.set_page_config(
    page_title="Censor - 微表情反馈收集",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入本地模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface.feedback_db import FeedbackDatabase, get_db
from model.censor_g_generator import CensorGGenerator, AU_INDEX
from model.prompt_driven_generator import PromptDrivenGenerator, EMOTION_KEYWORDS, INTENSITY_KEYWORDS

# =============================================================================
# Video Generator Wrapper (复用原有逻辑)
# =============================================================================

class VideoGeneratorWrapper:
    """视频生成器包装"""

    def __init__(self, checkpoint_path: str = None):
        self.generator = PromptDrivenGenerator(checkpoint_path=checkpoint_path)
        print("[VideoGeneratorWrapper] Using PromptDrivenGenerator")

    def generate(self, image_path: str, prompt: str, output_path: str = None) -> Tuple[str, Dict]:
        """生成微表情视频"""
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
            return self._generate_dummy_video(output_path), {'error': str(e)}

    def _format_active_au(self, au_tensor) -> str:
        """格式化活跃的AU"""
        if au_tensor is None:
            return "N/A"

        active = []
        for au_name, idx in AU_INDEX.items():
            if au_tensor[idx] > 0.1:
                active.append(f"{au_name}:{au_tensor[idx].item():.2f}")

        return ", ".join(active) if active else "none"

    def _generate_dummy_video(self, output_path: str) -> str:
        """生成模拟视频（错误情况下）"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (224, 224))

        for i in range(16):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:, :] = [i * 15, 100, 200]
            writer.write(frame)

        writer.release()
        return output_path


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """初始化session state"""
    if 'db' not in st.session_state:
        st.session_state.db = get_db()

    if 'generator' not in st.session_state:
        st.session_state.generator = VideoGeneratorWrapper()

    if 'current_video' not in st.session_state:
        st.session_state.current_video = None

    if 'current_info' not in st.session_state:
        st.session_state.current_info = {}

    if 'batch_tasks' not in st.session_state:
        st.session_state.batch_tasks = []

    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []


# =============================================================================
# Tab 1: Single Rating (单次评分)
# =============================================================================

def render_single_rating():
    """单次评分页面"""
    st.header("单次评分")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入")

        # 图片上传
        uploaded_image = st.file_uploader(
            "上传基准图片",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="上传一张中性脸作为基准"
        )

        # 保存上传的图片到临时文件
        image_path = None
        if uploaded_image:
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, uploaded_image.name)
            with open(image_path, 'wb') as f:
                f.write(uploaded_image.getbuffer())
            st.image(uploaded_image, caption="基准图片", width=300)

        # 提示词输入
        st.markdown("**提示词**")

        # 从提示词库选择
        templates = st.session_state.db.get_prompt_templates()
        template_names = [t['name'] for t in templates]

        selected_template = st.selectbox(
            "选择预设提示词",
            options=['自定义'] + template_names,
            index=0
        )

        if selected_template == '自定义':
            prompt = st.text_input(
                "输入自定义提示词",
                placeholder="例如：微笑、惊讶、轻微厌恶..."
            )
        else:
            # 找到对应的模板
            template = next((t for t in templates if t['name'] == selected_template), None)
            prompt = template['template'] if template else ''
            st.info(f"提示词: `{prompt}`")

        # 生成按钮
        generate_btn = st.button("生成微表情视频", type="primary")

    with col2:
        st.subheader("生成结果")

        if generate_btn:
            if image_path is None:
                st.error("请先上传图片")
            elif not prompt:
                st.error("请输入提示词")
            else:
                with st.spinner("正在生成视频..."):
                    video_path, info = st.session_state.generator.generate(image_path, prompt)

                    st.session_state.current_video = video_path
                    st.session_state.current_info = info
                    st.session_state.current_image_path = image_path
                    st.session_state.current_prompt = prompt

        # 显示视频
        if st.session_state.current_video:
            st.video(st.session_state.current_video)

            # 显示生成信息
            st.json(st.session_state.current_info)

    # 评分部分
    st.markdown("---")
    st.header("评分")

    if st.session_state.current_video:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            naturalness = st.slider(
                "自然度",
                min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                help="表情是否自然 (1=僵硬, 5=非常自然)"
            )

        with col2:
            smoothness = st.slider(
                "流畅度",
                min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                help="运动是否流畅 (1=卡顿, 5=非常流畅)"
            )

        with col3:
            prompt_match = st.slider(
                "匹配度",
                min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                help="是否符合提示词 (1=不匹配, 5=完美匹配)"
            )

        with col4:
            overall = st.slider(
                "总体评分",
                min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                help="总体评价"
            )

        # 专家级评分（可选）
        with st.expander("专家级评分（可选）"):
            au_accuracy = st.slider(
                "AU准确性",
                min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                help="生成的AU激活是否正确"
            )

            micro_quality = st.slider(
                "微表情质量",
                min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                help="是否符合微表情定义 (时长<500ms)"
            )

            expert_comments = st.text_area(
                "专家评语",
                placeholder="请描述生成效果的问题或优点..."
            )

        # 用户评论
        user_comments = st.text_area(
            "用户评论",
            placeholder="请描述您对生成效果的感受..."
        )

        # 提交按钮
        submit_btn = st.button("提交反馈", type="secondary")

        if submit_btn:
            feedback = {
                'image_path': st.session_state.current_image_path,
                'prompt': st.session_state.current_prompt,
                'video_path': st.session_state.current_video,
                'naturalness': naturalness,
                'smoothness': smoothness,
                'prompt_match': prompt_match,
                'overall': overall,
                'au_accuracy': au_accuracy if au_accuracy != 3.0 else None,
                'micro_quality': micro_quality if micro_quality != 3.0 else None,
                'expert_comments': expert_comments if expert_comments else None,
                'user_comments': user_comments if user_comments else None,
                'emotion': st.session_state.current_info.get('emotion'),
                'intensity': st.session_state.current_info.get('intensity'),
                'active_au': st.session_state.current_info.get('active_au'),
            }

            feedback_id = st.session_state.db.save_feedback(feedback)

            st.success(f"反馈 #{feedback_id} 已保存！")

            # 清空当前状态
            st.session_state.current_video = None
            st.session_state.current_info = {}

    else:
        st.info("请先生成视频再进行评分")


# =============================================================================
# Tab 2: Batch Testing (批量测试)
# =============================================================================

def render_batch_testing():
    """批量测试页面"""
    st.header("批量测试")

    # 说明
    st.markdown("""
    **批量测试流程：**
    1. 上传多张基准图片
    2. 选择或输入提示词
    3. 点击批量生成
    4. 快速评分每个视频
    5. 一键提交所有评分
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("批量输入")

        # 多图片上传
        batch_images = st.file_uploader(
            "上传多张基准图片",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="可一次上传多张图片"
        )

        if batch_images:
            st.info(f"已上传 {len(batch_images)} 张图片")

            # 显示上传的图片
            for i, img in enumerate(batch_images[:3]):  # 只显示前3张
                st.image(img, caption=f"图片 {i+1}", width=150)

        # 提示词选择
        templates = st.session_state.db.get_prompt_templates()
        template_names = [t['name'] for t in templates]

        batch_prompt_type = st.radio(
            "提示词模式",
            options=['统一提示词', '随机提示词', '按顺序分配'],
            horizontal=True
        )

        if batch_prompt_type == '统一提示词':
            selected_template = st.selectbox(
                "选择提示词",
                options=['自定义'] + template_names
            )

            if selected_template == '自定义':
                batch_prompt = st.text_input("输入提示词")
            else:
                template = next((t for t in templates if t['name'] == selected_template), None)
                batch_prompt = template['template'] if template else ''

        elif batch_prompt_type == '随机提示词':
            # 选择随机范围
            batch_category = st.multiselect(
                "选择情感类别",
                options=['basic', 'intensity', 'complex'],
                default=['basic']
            )
            batch_prompt = None  # 表示随机

        else:  # 按顺序分配
            batch_prompts_input = st.text_area(
                "输入提示词列表（每行一个）",
                placeholder="微笑\n惊讶\n厌恶..."
            )
            batch_prompt = batch_prompts_input.strip().split('\n') if batch_prompts_input else []

        # 批量生成按钮
        batch_generate_btn = st.button("批量生成视频", type="primary")

    with col2:
        st.subheader("批量结果")

        if batch_generate_btn:
            if not batch_images:
                st.error("请先上传图片")
            else:
                # 保存图片到临时目录
                temp_dir = tempfile.mkdtemp()
                image_paths = []
                for img in batch_images:
                    path = os.path.join(temp_dir, img.name)
                    with open(path, 'wb') as f:
                        f.write(img.getbuffer())
                    image_paths.append(path)

                # 获取提示词列表
                prompts = []
                if batch_prompt_type == '统一提示词':
                    prompts = [batch_prompt] * len(image_paths)
                elif batch_prompt_type == '随机提示词':
                    filtered_templates = [t for t in templates if t['category'] in batch_category]
                    import random
                    prompts = [random.choice(filtered_templates)['template'] for _ in image_paths]
                else:  # 按顺序分配
                    prompts = batch_prompt[:len(image_paths)]
                    # 如果提示词不够，用第一个补齐
                    while len(prompts) < len(image_paths):
                        prompts.append(prompts[0] if prompts else '微笑')

                # 批量生成
                progress_bar = st.progress(0)
                status_text = st.empty()

                batch_results = []
                for i, (img_path, prompt) in enumerate(zip(image_paths, prompts)):
                    status_text.text(f"正在生成 {i+1}/{len(image_paths)}: {prompt}")

                    video_path, info = st.session_state.generator.generate(img_path, prompt)

                    batch_results.append({
                        'image_path': img_path,
                        'prompt': prompt,
                        'video_path': video_path,
                        'info': info,
                        'naturalness': 3.0,
                        'smoothness': 3.0,
                        'prompt_match': 3.0,
                        'overall': 3.0,
                    })

                    progress_bar.progress((i + 1) / len(image_paths))

                status_text.text(f"完成！共生成 {len(batch_results)} 个视频")
                st.session_state.batch_results = batch_results

        # 显示批量结果列表
        if st.session_state.batch_results:
            st.markdown("---")
            st.subheader("快速评分")

            for i, result in enumerate(st.session_state.batch_results):
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 2])

                    with col1:
                        st.video(result['video_path'])
                        st.caption(f"提示词: {result['prompt']}")

                    with col2:
                        # 快速评分滑块
                        result['overall'] = st.slider(
                            f"总体评分 #{i+1}",
                            min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                            key=f"batch_overall_{i}"
                        )

                    with col3:
                        # 快捷按钮
                        quick_rating = st.selectbox(
                            f"快捷评分 #{i+1}",
                            options=['--', '很差 (1)', '一般 (2)', '还行 (3)', '不错 (4)', '很好 (5)'],
                            key=f"batch_quick_{i}"
                        )

                        if quick_rating != '--':
                            result['overall'] = float(quick_rating.split('(')[1].split(')')[0])

                    st.markdown("---")

            # 一键提交所有
            batch_submit_btn = st.button("一键提交所有评分", type="secondary")

            if batch_submit_btn:
                submitted_count = 0
                for result in st.session_state.batch_results:
                    feedback = {
                        'image_path': result['image_path'],
                        'prompt': result['prompt'],
                        'video_path': result['video_path'],
                        'naturalness': result['naturalness'],
                        'smoothness': result['smoothness'],
                        'prompt_match': result['prompt_match'],
                        'overall': result['overall'],
                        'emotion': result['info'].get('emotion'),
                        'intensity': result['info'].get('intensity'),
                        'active_au': result['info'].get('active_au'),
                    }

                    st.session_state.db.save_feedback(feedback)
                    submitted_count += 1

                st.success(f"已提交 {submitted_count} 条反馈！")
                st.session_state.batch_results = []


# =============================================================================
# Tab 3: Prompt Library (提示词库)
# =============================================================================

def render_prompt_library():
    """提示词库页面"""
    st.header("提示词库")

    # 获取模板
    templates = st.session_state.db.get_prompt_templates()

    # 分类筛选
    col1, col2 = st.columns([1, 3])

    with col1:
        category_filter = st.multiselect(
            "筛选类别",
            options=['basic', 'intensity', 'complex'],
            default=['basic', 'intensity', 'complex']
        )

        emotion_filter = st.multiselect(
            "筛选情感",
            options=['happiness', 'surprise', 'disgust', 'fear', 'anger', 'repression', 'contempt'],
            default=[]
        )

    # 过滤模板
    filtered_templates = templates
    if category_filter:
        filtered_templates = [t for t in filtered_templates if t['category'] in category_filter]
    if emotion_filter:
        filtered_templates = [t for t in filtered_templates if t['emotion'] in emotion_filter]

    with col2:
        st.subheader(f"共 {len(filtered_templates)} 个提示词模板")

        # 显示模板列表
        for template in filtered_templates:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**{template['name']}**")
                    st.code(template['template'])

                with col2:
                    st.caption(f"类别: {template['category']}")
                    st.caption(f"情感: {template['emotion']}")
                    st.caption(f"强度: {template['intensity']}")

                with col3:
                    st.metric("使用次数", template['usage_count'])

                    if template['is_custom']:
                        if st.button("删除", key=f"del_template_{template['id']}"):
                            if st.session_state.db.delete_template(template['id']):
                                st.success("已删除")
                                st.rerun()

                st.markdown("---")

    # 添加自定义模板
    st.subheader("添加自定义提示词")

    col1, col2, col3 = st.columns(3)

    with col1:
        custom_name = st.text_input("名称")
        custom_template = st.text_input("提示词内容")

    with col2:
        custom_category = st.selectbox("类别", options=['basic', 'intensity', 'complex'])
        custom_emotion = st.selectbox("情感", options=[
            'happiness', 'surprise', 'disgust', 'fear', 'anger', 'repression', 'contempt', 'other'
        ])

    with col3:
        custom_intensity = st.selectbox("强度", options=['weak', 'medium', 'strong', 'micro'])

    if st.button("添加模板"):
        if custom_name and custom_template:
            template_id = st.session_state.db.add_custom_template(
                name=custom_name,
                template=custom_template,
                category=custom_category,
                emotion=custom_emotion,
                intensity=custom_intensity
            )
            st.success(f"已添加模板 #{template_id}")
            st.rerun()
        else:
            st.error("请填写名称和提示词内容")


# =============================================================================
# Tab 4: History (历史记录)
# =============================================================================

def render_history():
    """历史记录页面"""
    st.header("历史记录")

    # 筛选条件
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        quality_filter = st.selectbox(
            "质量筛选",
            options=['全部', '高质量', '普通', '低质量'],
            index=0
        )

    with col2:
        min_overall = st.slider(
            "最小评分",
            min_value=0.0, max_value=5.0, value=0.0, step=0.5
        )

    with col3:
        search_prompt = st.text_input("搜索提示词")

    with col4:
        limit = st.number_input("显示数量", min_value=10, max_value=500, value=50)

    # 获取反馈列表
    quality_map = {
        '全部': None,
        '高质量': 'high',
        '普通': 'normal',
        '低质量': 'low'
    }

    feedbacks = st.session_state.db.get_all_feedbacks(
        limit=limit,
        quality_filter=quality_map.get(quality_filter),
        min_overall=min_overall if min_overall > 0 else None
    )

    # 搜索过滤
    if search_prompt:
        feedbacks = [f for f in feedbacks if search_prompt.lower() in f['prompt'].lower()]

    st.info(f"共找到 {len(feedbacks)} 条记录")

    # 统计信息
    stats = st.session_state.db.get_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总反馈数", stats.get('total_feedbacks', 0))

    with col2:
        st.metric("平均总体评分", stats.get('avg_overall', 0))

    with col3:
        st.metric("高质量数", stats.get('quality_distribution', {}).get('high', 0))

    with col4:
        st.metric("最近24h", stats.get('recent_24h', 0))

    # 显示反馈列表
    for fb in feedbacks:
        with st.expander(f"#{fb['id']} | {fb['prompt']} | 总评分: {fb['overall']}"):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"**时间**: {fb['timestamp']}")
                st.markdown(f"**提示词**: {fb['prompt']}")

                # 评分显示
                st.markdown("**评分**:")
                st.progress(fb['naturalness'] / 5.0, text=f"自然度: {fb['naturalness']}")
                st.progress(fb['smoothness'] / 5.0, text=f"流畅度: {fb['smoothness']}")
                st.progress(fb['prompt_match'] / 5.0, text=f"匹配度: {fb['prompt_match']}")
                st.progress(fb['overall'] / 5.0, text=f"总体: {fb['overall']}")

            with col2:
                if fb['video_path'] and os.path.exists(fb['video_path']):
                    st.video(fb['video_path'])
                else:
                    st.info("视频文件不存在")

                # 评论
                if fb['user_comments']:
                    st.markdown(f"**用户评论**: {fb['user_comments']}")
                if fb['expert_comments']:
                    st.markdown(f"**专家评语**: {fb['expert_comments']}")

            # 编辑/删除操作
            col1, col2 = st.columns(2)

            with col1:
                if st.button("编辑评分", key=f"edit_{fb['id']}"):
                    st.session_state.editing_id = fb['id']

            with col2:
                if st.button("删除", key=f"delete_{fb['id']}"):
                    st.session_state.db.delete_feedback(fb['id'])
                    st.success("已删除")
                    st.rerun()

            # 编辑模式
            if st.session_state.get('editing_id') == fb['id']:
                st.markdown("**编辑评分**")

                new_overall = st.slider(
                    "新总体评分",
                    min_value=1.0, max_value=5.0, value=float(fb['overall']), step=0.5,
                    key=f"new_overall_{fb['id']}"
                )

                new_comments = st.text_area(
                    "新评论",
                    value=fb['user_comments'] or '',
                    key=f"new_comments_{fb['id']}"
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("保存修改", key=f"save_{fb['id']}"):
                        st.session_state.db.update_feedback(
                            fb['id'],
                            {'overall': new_overall, 'user_comments': new_comments}
                        )
                        st.success("已保存")
                        st.session_state.editing_id = None
                        st.rerun()

                with col2:
                    if st.button("取消", key=f"cancel_{fb['id']}"):
                        st.session_state.editing_id = None
                        st.rerun()


# =============================================================================
# Tab 5: Data Export (数据导出)
# =============================================================================

def render_data_export():
    """数据导出页面"""
    st.header("数据导出")

    # 导出设置
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("导出设置")

        export_format = st.radio(
            "导出格式",
            options=['JSON', 'CSV', 'Excel'],
            horizontal=True
        )

        # 筛选条件
        export_quality = st.selectbox(
            "质量筛选",
            options=['全部', '高质量 (>=4)', '普通', '低质量 (<2)'],
            index=0
        )

        export_min_overall = st.slider(
            "最小评分",
            min_value=0.0, max_value=5.0, value=0.0, step=0.5
        )

        st.markdown("---")

        # 数据清洗
        st.subheader("数据清洗")

        low_quality_count = len(st.session_state.db.get_low_quality_feedbacks())
        st.metric("低质量反馈数", low_quality_count)

        if st.button("删除所有低质量反馈 (<2分)", type="secondary"):
            deleted = st.session_state.db.batch_delete_low_quality(threshold=2.0)
            st.success(f"已删除 {deleted} 条低质量反馈")
            st.rerun()

        st.markdown("---")

        # 导入数据
        st.subheader("导入数据")

        import_file = st.file_uploader(
            "导入JSON文件",
            type=['json'],
            help="导入原有的 feedback.json 文件"
        )

        if import_file:
            temp_path = tempfile.mktemp(suffix='.json')
            with open(temp_path, 'wb') as f:
                f.write(import_file.getbuffer())

            if st.button("导入"):
                count = st.session_state.db.import_from_json(temp_path)
                st.success(f"已导入 {count} 条记录")
                st.rerun()

    with col2:
        st.subheader("预览数据")

        # 获取数据预览
        quality_map = {
            '全部': None,
            '高质量 (>=4)': 'high',
            '普通': 'normal',
            '低质量 (<2)': 'low'
        }

        preview_data = st.session_state.db.get_all_feedbacks(
            limit=100,
            quality_filter=quality_map.get(export_quality),
            min_overall=export_min_overall if export_min_overall > 0 else None
        )

        if preview_data:
            # 转换为DataFrame显示
            df = pd.DataFrame(preview_data)

            # 选择显示列
            display_columns = st.multiselect(
                "显示列",
                options=df.columns.tolist(),
                default=['id', 'timestamp', 'prompt', 'overall', 'naturalness', 'smoothness', 'prompt_match']
            )

            st.dataframe(df[display_columns], use_container_width=True)

            # 导出按钮
            st.markdown("---")

            if st.button(f"导出 {export_format}", type="primary"):
                quality_val = quality_map.get(export_quality)
                min_overall_val = export_min_overall if export_min_overall > 0 else None

                if export_format == 'JSON':
                    output_path = st.session_state.db.export_to_json(
                        quality_filter=quality_val,
                        min_overall=min_overall_val
                    )
                elif export_format == 'CSV':
                    output_path = st.session_state.db.export_to_csv(
                        quality_filter=quality_val,
                        min_overall=min_overall_val
                    )
                else:  # Excel
                    output_path = st.session_state.db.export_to_excel(
                        quality_filter=quality_val,
                        min_overall=min_overall_val
                    )

                st.success(f"已导出到: {output_path}")

                # 提供下载链接
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label=f"下载 {export_format} 文件",
                        data=f,
                        file_name=os.path.basename(output_path),
                        mime='application/octet-stream'
                    )

        else:
            st.info("没有数据可导出")

        # 统计图表
        st.markdown("---")
        st.subheader("评分分布")

        stats = st.session_state.db.get_statistics()

        # 评分分布柱状图
        if preview_data:
            import plotly.express as px

            # Overall评分分布
            overall_values = [f['overall'] for f in preview_data]

            fig = px.histogram(
                x=overall_values,
                title="总体评分分布",
                labels={'x': '评分', 'y': '数量'},
                nbins=10,
                range_x=[0, 5]
            )

            st.plotly_chart(fig, use_container_width=True)

            # 情感分布
            emotion_dist = stats.get('emotion_distribution', {})
            if emotion_dist:
                fig2 = px.pie(
                    values=list(emotion_dist.values()),
                    names=list(emotion_dist.keys()),
                    title="情感类型分布"
                )
                st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """侧边栏"""
    with st.sidebar:
        st.header("系统状态")

        # 统计信息
        stats = st.session_state.db.get_statistics()

        st.metric("总反馈数", stats.get('total_feedbacks', 0))
        st.metric("平均评分", stats.get('avg_overall', 0))

        st.markdown("---")

        # 快捷操作
        st.subheader("快捷操作")

        if st.button("刷新数据"):
            st.rerun()

        # 清空session state
        if st.button("清空当前状态"):
            st.session_state.current_video = None
            st.session_state.current_info = {}
            st.session_state.batch_results = []
            st.success("已清空")

        st.markdown("---")

        # 帮助
        st.subheader("帮助")

        with st.expander("评分标准"):
            st.markdown("""
            **自然度**: 表情是否自然真实
            - 1分: 僵硬、机械感
            - 2分: 略显僵硬
            - 3分: 一般
            - 4分: 较自然
            - 5分: 非常自然

            **流畅度**: 运动过程是否流畅
            - 1分: 明显卡顿
            - 5分: 完全流畅

            **匹配度**: 是否符合提示词描述
            - 1分: 完全不匹配
            - 5分: 完美匹配

            **总体评分**: 综合评价
            """)


# =============================================================================
# Main
# =============================================================================

def main():
    # 初始化
    init_session_state()

    # 标题
    st.title("Censor - 微表情反馈收集系统")
    st.markdown("**用于收集微表情视频生成的人类评分数据**")
    st.markdown("---")

    # 侧边栏
    render_sidebar()

    # 主界面 Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "单次评分",
        "批量测试",
        "提示词库",
        "历史记录",
        "数据导出"
    ])

    with tab1:
        render_single_rating()

    with tab2:
        render_batch_testing()

    with tab3:
        render_prompt_library()

    with tab4:
        render_history()

    with tab5:
        render_data_export()

    # 页脚
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:gray;'>"
        f"Censor 微表情反馈收集系统 | {datetime.now().strftime('%Y-%m-%d')}"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()