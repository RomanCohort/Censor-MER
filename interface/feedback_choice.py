# =============================================================================
# 简洁选择式反馈界面 - "哪个更像具有XX微表情"
# =============================================================================
# 极简设计：
#   1. 显示两个候选视频
#   2. 显示目标微表情
#   3. 用户点击选择哪个更像
#   4. 自动进入下一题
# =============================================================================

import streamlit as st
import os
import json
import tempfile
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 设置页面 - 简洁风格
st.set_page_config(
    page_title="微表情选择",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 导入本地模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface.feedback_db import FeedbackDatabase, get_db

# =============================================================================
# 自定义CSS - 极简风格
# =============================================================================

STYLES = """
<style>
    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 主容器 */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }

    /* 标题 */
    .question-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #1f77b4;
    }

    /* 视频容器 */
    .video-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1rem 0;
    }

    /* 选择按钮 */
    .choice-btn {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem;
        border-radius: 12px;
        background: #f0f2f6;
        cursor: pointer;
        transition: all 0.2s;
        border: 3px solid transparent;
    }

    .choice-btn:hover {
        background: #e8f4e8;
        border-color: #28a745;
    }

    .choice-label {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }

    /* 统计信息 */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .stat-item {
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #666;
    }

    /* 进度条 */
    .progress-text {
        text-align: center;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
"""


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """初始化session state"""
    if 'db' not in st.session_state:
        st.session_state.db = get_db()

    if 'current_pair' not in st.session_state:
        st.session_state.current_pair = None

    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = None

    if 'total_choices' not in st.session_state:
        st.session_state.total_choices = 0

    if 'correct_choices' not in st.session_state:
        st.session_state.correct_choices = 0

    if 'history' not in st.session_state:
        st.session_state.history = []


# =============================================================================
# 模拟数据 - 实际使用时替换为真实生成
# =============================================================================

# 微表情列表
MICRO_EXPRESSIONS = [
    "微笑",
    "惊讶",
    "厌恶",
    "恐惧",
    "愤怒",
    "悲伤",
    " contempt",  # 轻蔑
    "轻微微笑",
    "轻微惊讶",
    "轻微厌恶",
]

# 模拟视频路径（实际使用时替换）
DUMMY_VIDEOS = {
    'left': None,
    'right': None,
}


def get_random_emotion():
    """随机选择一个微表情"""
    return random.choice(MICRO_EXPRESSIONS)


def generate_comparison_pair(emotion: str) -> Tuple[Dict, Dict]:
    """
    生成一对比较视频

    实际使用时，这里应该：
    1. 从数据库中获取真实视频
    2. 或者调用生成器生成两个候选
    3. 其中一个可能是真实微表情视频（ground truth）

    Returns:
        (video_a, video_b): 两个视频的信息
    """
    # 模拟数据
    video_a = {
        'id': random.randint(1000, 9999),
        'path': None,  # 实际路径
        'label': 'A',
        'metadata': {
            'emotion': emotion,
            'source': 'generated_v1',
        }
    }

    video_b = {
        'id': random.randint(1000, 9999),
        'path': None,  # 实际路径
        'label': 'B',
        'metadata': {
            'emotion': emotion,
            'source': 'generated_v2',
        }
    }

    return video_a, video_b


# =============================================================================
# 主界面
# =============================================================================

def render_main():
    """渲染主界面"""

    # CSS样式
    st.markdown(STYLES, unsafe_allow_html=True)

    # 统计条
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("已选择", st.session_state.total_choices)
    with col2:
        pass
    with col3:
        pass

    # 进度提示
    st.markdown(f"<p class='progress-text'>第 {st.session_state.total_choices + 1} 题</p>", unsafe_allow_html=True)

    # 如果没有当前题目，生成新的
    if st.session_state.current_pair is None:
        st.session_state.current_emotion = get_random_emotion()
        st.session_state.current_pair = generate_comparison_pair(st.session_state.current_emotion)

    video_a, video_b = st.session_state.current_pair
    emotion = st.session_state.current_emotion

    # 问题标题
    st.markdown(
        f"<h2 class='question-title'>哪个更像具有「{emotion}」微表情？</h2>",
        unsafe_allow_html=True
    )

    # 视频展示区
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

        # 视频占位（实际使用时替换为真实视频）
        st.empty()  # 视频占位

        # 如果有真实视频
        if video_a.get('path'):
            st.video(video_a['path'])
        else:
            # 模拟显示
            st.info("🎬 视频 A")
            st.markdown("*[生成的微表情视频]*")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

        if video_b.get('path'):
            st.video(video_b['path'])
        else:
            st.info("🎬 视频 B")
            st.markdown("*[生成的微表情视频]*")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 选择按钮
    col1, col2, col3 = st.columns([1, 2, 2])

    with col2:
        if st.button("⬅️ 选择 A", type="primary", use_container_width=True):
            handle_choice('A', video_a, video_b, emotion)

    with col3:
        if st.button("选择 B ➡️", type="primary", use_container_width=True):
            handle_choice('B', video_a, video_b, emotion)

    # 底部选项
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("两个都不像"):
            handle_choice('neither', video_a, video_b, emotion)

    with col2:
        if st.button("两个都很像"):
            handle_choice('both', video_a, video_b, emotion)

    with col3:
        if st.button("无法判断"):
            handle_choice('uncertain', video_a, video_b, emotion)


def handle_choice(choice: str, video_a: Dict, video_b: Dict, emotion: str):
    """处理用户选择"""

    # 记录选择
    record = {
        'timestamp': datetime.now().isoformat(),
        'emotion': emotion,
        'video_a_id': video_a.get('id'),
        'video_b_id': video_b.get('id'),
        'choice': choice,
        'video_a_source': video_a.get('metadata', {}).get('source'),
        'video_b_source': video_b.get('metadata', {}).get('source'),
    }

    # 保存到数据库
    feedback = {
        'prompt': f"选择式比较: {emotion}",
        'naturalness': 0,  # 不适用
        'smoothness': 0,
        'prompt_match': 0,
        'overall': 0,
        'is_comparison': 1,
        'comparison_winner': choice,
        'emotion': emotion,
        'user_comments': json.dumps({
            'video_a_id': video_a.get('id'),
            'video_b_id': video_b.get('id'),
            'choice': choice,
        }),
    }

    feedback_id = st.session_state.db.save_feedback(feedback)

    # 更新统计
    st.session_state.total_choices += 1
    st.session_state.history.append(record)

    # 清空当前题目，准备下一题
    st.session_state.current_pair = None
    st.session_state.current_emotion = None

    # 显示反馈
    st.toast(f"已选择: {choice}", icon="✅")

    # 刷新页面
    st.rerun()


# =============================================================================
# 管理页面（可选）
# =============================================================================

def render_admin():
    """管理/统计页面"""
    st.header("📊 选择统计")

    # 统计
    stats = st.session_state.db.get_statistics()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总选择次数", st.session_state.total_choices)
    with col2:
        st.metric("数据库总记录", stats.get('total_feedbacks', 0))
    with col3:
        st.metric("比较记录", len(st.session_state.history))

    # 最近选择
    if st.session_state.history:
        st.subheader("最近选择")
        import pandas as pd
        df = pd.DataFrame(st.session_state.history[-10:])
        st.dataframe(df, use_container_width=True)

    # 清除按钮
    if st.button("重置统计"):
        st.session_state.total_choices = 0
        st.session_state.history = []
        st.rerun()


# =============================================================================
# 主入口
# =============================================================================

def main():
    # 初始化
    init_session_state()

    # 简单导航
    if st.query_params.get('page') == 'admin':
        render_admin()
    else:
        render_main()

        # 底部链接
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📊 查看统计"):
                st.query_params['page'] = 'admin'
                st.rerun()


if __name__ == '__main__':
    main()
