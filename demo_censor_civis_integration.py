"""
Censor CVEmotionBridge → Civis Lucri-Faber 完整集成演示

方案1实现: 使用现有 FER/DeepFace backend，无需下载 MMEW 数据集

流程:
    Camera/Video → CVEmotionBridge → Civis Agent → Hormone System

运行方式:
    python demo_censor_civis_integration.py

作者: Claude + YAN
"""

import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

# ============================================================================
# 路径配置
# ============================================================================

CENSOR_PATH = Path("D:/censor")
CIVIS_PATH = Path("D:/civis_lucri_faber")

sys.path.insert(0, str(CENSOR_PATH))
sys.path.insert(0, str(CIVIS_PATH))

# ============================================================================
# 导入
# ============================================================================

print("=" * 70)
print(" Censor → Civis 集成演示")
print("=" * 70)

# Censor CV Emotion Bridge
from model.cv_emotion_bridge import (
    CVEmotionBridge,
    create_bridge_for_civis,
    EmotionDetectionResult,
)

print("[1/4] Censor CVEmotionBridge 导入成功")

# Civis Core
try:
    from core.event_bus import EventBus
    from core.events import BRAIN_UPDATE, EMOTION_PROCESS
    from core.advanced_emotion_integration import IntegratedAdvancedEmotionSystem
    from core.hormone_system import HormoneSystem
    from core.hpa_axis import HPAAxis
    CIVIS_AVAILABLE = True
    print("[2/4] Civis 核心模块导入成功")
except ImportError as e:
    print(f"[2/4] Civis 模块导入失败: {e}")
    CIVIS_AVAILABLE = False

# ============================================================================
# 简化版 Civis Agent Wrapper
# ============================================================================

class SimpleCivisEmotionAgent:
    """
    简化版 Civis 情绪 Agent

    只包含核心情绪相关模块:
    - EventBus: 事件总线
    - AdvancedEmotionSystem: 高级情绪系统
    - HormoneSystem: 激素系统
    - HPAAxis: HPA轴

    用于演示 Censor CV 情绪检测如何影响 Civis 内部状态。
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

        # 事件总线
        self.bus = EventBus()
        print("    [ EventBus 初始化 ]")

        # 高级情绪系统
        if CIVIS_AVAILABLE:
            self.advanced_emotion = IntegratedAdvancedEmotionSystem(
                input_dim=64,
                hidden_dim=64,
                event_bus=self.bus,
            )
            print("    [ AdvancedEmotionSystem 初始化 ]")

            # 激素系统
            self.hormones = HormoneSystem(event_bus=self.bus)
            print("    [ HormoneSystem 初始化 ]")

            # HPA轴
            self.hpa = HPAAxis(event_bus=self.bus)
            print("    [ HPAAxis 初始化 ]")
        else:
            self.advanced_emotion = None
            self.hormones = None
            self.hpa = None
            print("    [ Civis 模块不可用，使用模拟模式 ]")

        # 内部状态
        self.internal_state = torch.randn(64)
        self.current_hour = 12.0

        # 历史记录
        self.history = []

        print(f"    [ SimpleCivisEmotionAgent 初始化完成，device={device} ]")

    def process_cv_emotion(self, cv_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 CVEmotionBridge 的检测结果

        将 CV 情绪转换为 Civis 内部状态变化。

        Args:
            cv_result: CVEmotionBridge.get_civis_compatible_output() 返回的字典

        Returns:
            Civis 内部状态变化摘要
        """
        timestamp = time.time()

        # 提取 CV 检测结果
        emotion_tensor = cv_result.get('emotion_tensor', torch.zeros(7))
        dominant_emotion = cv_result.get('dominant_emotion', 'neutral')
        confidence = cv_result.get('confidence', 0.5)
        valence = cv_result.get('valence', 0.0)
        arousal = cv_result.get('arousal', 0.0)
        criticality = cv_result.get('criticality', 0.0)
        heart_rate = cv_result.get('rppg_heart_rate')

        # 构建 user_emotion tensor (扩展到 64 维)
        user_emotion = torch.zeros(64)
        user_emotion[:7] = emotion_tensor
        user_emotion[7:14] = emotion_tensor * 0.5
        user_emotion[21] = arousal
        user_emotion[22] = valence
        user_emotion[23] = criticality
        if heart_rate:
            user_emotion[24] = heart_rate / 150.0

        # 调用 Civis 高级情绪系统
        civis_output = {}

        if self.advanced_emotion is not None:
            try:
                # 发布 EMOTION_PROCESS 事件
                self.bus.publish(EMOTION_PROCESS, {
                    'user_emotion': user_emotion,
                    'user_proximity': 0.7,  # 模拟近距离交互
                    'state': self.internal_state,
                    'hour': self.current_hour,
                    'external_cortisol': arousal * 0.3,  # 高唤醒 -> 皮质醇
                    'external_oxytocin': max(0, valence) * 0.2,  # 正情绪 -> 催产素
                })

                # 处理
                result = self.advanced_emotion.process(
                    state=self.internal_state,
                    user_emotion=user_emotion,
                    user_proximity=0.7,
                    external_cortisol=arousal * 0.3,
                    external_oxytocin=max(0, valence) * 0.2,
                )

                civis_output['emotion_result'] = result

                # 获取激素状态
                if self.hormones:
                    hormone_summary = self.hormones.get_summary()
                    civis_output['hormones'] = hormone_summary

                    # 根据 CV 结果更新激素
                    # 高唤醒 (愤怒/恐惧) -> 皮质醇上升
                    if arousal > 0.6 and valence < 0:
                        # 模拟压力反应
                        civis_output['hormone_effect'] = 'cortisol_increase'
                    # 正情绪 (快乐) -> 催产素上升
                    elif valence > 0.3:
                        civis_output['hormone_effect'] = 'oxytocin_increase'

                # 发布 BRAIN_UPDATE 触发激素系统
                self.bus.publish(BRAIN_UPDATE, {
                    'internal_state': {
                        'emotion_criticality': criticality,
                        'alignment_score': 1.0 - abs(valence),
                        'social_engagement': confidence,
                    }
                })

            except Exception as e:
                civis_output['error'] = str(e)
        else:
            # 模拟模式
            civis_output = {
                'emotion_result': {
                    'mood_valence': valence,
                    'mood_arousal': arousal,
                    'dominant_emotion': dominant_emotion,
                },
                'hormones': {
                    'cortisol': arousal * 0.3 + 0.3,
                    'oxytocin': max(0, valence) * 0.2 + 0.3,
                },
                'simulated': True,
            }

        # 记录历史
        self.history.append({
            'timestamp': timestamp,
            'cv_input': {
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'valence': valence,
                'arousal': arousal,
            },
            'civis_output': civis_output,
        })

        return {
            'cv_input': cv_result,
            'civis_state': civis_output,
            'agent_summary': self._build_summary(cv_result, civis_output),
        }

    def _build_summary(self, cv_result: Dict, civis_output: Dict) -> str:
        """构建人类可读的摘要"""
        cv_emo = cv_result.get('dominant_emotion', 'neutral')
        cv_conf = cv_result.get('confidence', 0.0)
        cv_val = cv_result.get('valence', 0.0)
        cv_aro = cv_result.get('arousal', 0.0)

        civis_emo = civis_output.get('emotion_result', {})
        hormones = civis_output.get('hormones', {})

        # 提取心境状态
        if isinstance(civis_emo, dict):
            mood_val = civis_emo.get('mood_valence', cv_val)
            mood_aro = civis_emo.get('mood_arousal', cv_aro)
        else:
            mood_val = cv_val
            mood_aro = cv_aro

        # 激素
        cortisol = hormones.get('cortisol', 0.3) if isinstance(hormones, dict) else 0.3
        oxytocin = hormones.get('oxytocin', 0.3) if isinstance(hormones, dict) else 0.3

        summary = (
            f"CV检测: {cv_emo} (置信度={cv_conf:.2f}, 效价={cv_val:.2f}, 唤醒={cv_aro:.2f})\n"
            f"Civis状态: 心境效价={mood_val:.2f}, 心境唤醒={mood_aro:.2f}\n"
            f"激素水平: 皮质醇={cortisol:.2f}, 催产素={oxytocin:.2f}"
        )

        return summary

    def get_history(self, last_n: int = 10) -> list:
        """获取最近 N 条历史记录"""
        return self.history[-last_n:]


# ============================================================================
# 完整集成演示
# ============================================================================

class CensorCivisIntegrationDemo:
    """
    完整集成演示

    流程:
        1. CVEmotionBridge 初始化 (FER/DeepFace backend)
        2. SimpleCivisEmotionAgent 初始化
        3. 处理视频帧/摄像头
        4. 展示情绪 → Civis → 激素 的完整链条
    """

    def __init__(self, backend: str = 'fer', device: str = 'cpu'):
        print("\n" + "=" * 70)
        print(" 初始化集成系统")
        print("=" * 70)

        # CV Emotion Bridge
        print("\n[步骤 1] 初始化 CV Emotion Bridge...")
        self.cv_bridge = create_bridge_for_civis(device=device)
        print(f"    Backend: {self.cv_bridge.backend_name}")
        print(f"    Available: {self.cv_bridge.backend.available}")

        # Civis Agent
        print("\n[步骤 2] 初始化 Civis 情绪 Agent...")
        self.civis_agent = SimpleCivisEmotionAgent(device=device)

        print("\n" + "=" * 70)
        print(" 系统初始化完成!")
        print("=" * 70)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧: CV检测 → Civis处理

        Args:
            frame: BGR 图像 (cv2 格式)

        Returns:
            完整的处理结果
        """
        # 1. CV 情绪检测
        cv_result = self.cv_bridge.detect(frame)
        cv_output = self.cv_bridge.get_civis_compatible_output()

        # 2. Civis 处理
        civis_result = self.civis_agent.process_cv_emotion(cv_output)

        return civis_result

    def process_video_file(self, video_path: str, fps: int = 15) -> list:
        """
        处理视频文件

        Args:
            video_path: 视频文件路径
            fps: 处理帧率

        Returns:
            所有帧的处理结果列表
        """
        cap = cv2.VideoCapture(video_path)
        results = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (30 // fps) == 0:
                result = self.process_frame(frame)
                results.append(result)
                print(f"Frame {frame_idx}: {result['agent_summary'][:50]}...")

            frame_idx += 1

        cap.release()
        return results

    def run_camera(self, camera_id: int = 0, display: bool = True):
        """
        运行实时摄像头演示

        Args:
            camera_id: 摄像头 ID
            display: 是否显示可视化窗口
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {camera_id}")
            return

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        print("\n" + "=" * 70)
        print(" 实时摄像头演示")
        print("=" * 70)
        print("按 'q' 退出, 按 's' 保存快照")
        print("=" * 70)

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理
            result = self.process_frame(frame)

            if display:
                frame = self._draw_result(frame, result)
                cv2.imshow('Censor → Civis Integration', frame)

            frame_count += 1

            # 每 30 帧打印一次状态
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"\n[Frame {frame_count}] FPS: {fps:.1f}")
                print(result['agent_summary'])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('snapshot_integration.png', frame)
                print("快照已保存: snapshot_integration.png")

        cap.release()
        cv2.destroyAllWindows()

        # 打印最终统计
        self._print_final_stats()

    def _draw_result(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """在帧上绘制结果"""
        cv_input = result.get('cv_input', {})
        civis_state = result.get('civis_state', {})

        # CV 检测结果 (左上)
        cv_emo = cv_input.get('dominant_emotion', 'neutral')
        cv_conf = cv_input.get('confidence', 0.0)
        cv_text = f"CV: {cv_emo} ({cv_conf:.2f})"
        cv2.putText(frame, cv_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Civis 状态 (右上)
        hormones = civis_state.get('hormones', {})
        if isinstance(hormones, dict):
            cortisol = hormones.get('cortisol', 0.3)
            oxytocin = hormones.get('oxytocin', 0.3)
            civis_text = f"Civis: Cor={cortisol:.2f} Oxy={oxytocin:.2f}"
            cv2.putText(frame, civis_text, (frame.shape[1] - 180, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # 效价/唤醒 (左下)
        valence = cv_input.get('valence', 0.0)
        arousal = cv_input.get('arousal', 0.0)
        va_text = f"V={valence:.2f} A={arousal:.2f}"
        cv2.putText(frame, va_text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
        stats = self.cv_bridge.get_stats()
        fps_text = f"FPS: {stats['fps']:.1f}"
        cv2.putText(frame, fps_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def _print_final_stats(self):
        """打印最终统计"""
        print("\n" + "=" * 70)
        print(" 演示结束统计")
        print("=" * 70)

        cv_stats = self.cv_bridge.get_stats()
        print(f"\nCV Bridge:")
        print(f"  处理帧数: {cv_stats['frames_processed']}")
        print(f"  平均 FPS: {cv_stats['fps']:.1f}")
        print(f"  Backend: {cv_stats['backend']}")

        history = self.civis_agent.get_history(last_n=5)
        print(f"\nCivis Agent (最近5条):")
        for i, h in enumerate(history):
            cv = h['cv_input']
            print(f"  [{i}] {cv['dominant_emotion']} "
                  f"(V={cv['valence']:.2f}, A={cv['arousal']:.2f})")

        print("=" * 70)


# ============================================================================
# 主程序
# ============================================================================

def demo_synthetic_frames():
    """使用合成帧演示"""
    print("\n" + "=" * 70)
    print(" Demo 1: 合成帧测试")
    print("=" * 70)

    demo = CensorCivisIntegrationDemo(backend='fer')

    # 创建测试帧
    print("\n创建合成测试帧...")
    frames = []
    for i in range(20):
        frame = np.random.randint(100, 200, (240, 320, 3), dtype=np.uint8)
        # 添加人脸模拟
        if i < 10:
            # "Happy" - 亮色调
            cv2.ellipse(frame, (160, 120), (80, 100), 0, 0, 360, (255, 220, 180), -1)
            cv2.ellipse(frame, (160, 160), (40, 15), 0, 0, 180, (200, 150, 150), -1)  # 微笑
        else:
            # "Sad" - 暗色调
            cv2.ellipse(frame, (160, 120), (80, 100), 0, 0, 360, (150, 130, 160), -1)
            cv2.ellipse(frame, (160, 170), (30, 10), 0, 180, 360, (100, 100, 100), -1)  # 下垂嘴
        frames.append(frame)

    print(f"创建了 {len(frames)} 帧")

    # 处理
    print("\n处理帧...")
    for i, frame in enumerate(frames):
        result = demo.process_frame(frame)

        if i % 5 == 0:
            print(f"\n[Frame {i}]")
            print(result['agent_summary'])

    print("\n" + "-" * 70)
    print(" Demo 1 完成!")
    print("-" * 70)


def demo_quick_test():
    """快速功能测试"""
    print("\n" + "=" * 70)
    print(" Demo 2: 快速功能测试")
    print("=" * 70)

    demo = CensorCivisIntegrationDemo(backend='fer')

    # 单帧测试
    frame = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    cv2.ellipse(frame, (112, 112), (70, 90), 0, 0, 360, (220, 200, 180), -1)

    print("\n处理单帧...")
    result = demo.process_frame(frame)

    print("\n" + "-" * 40)
    print(" 处理结果:")
    print("-" * 40)

    # CV 输入
    cv_in = result['cv_input']
    print(f"\n[CV 检测结果]")
    print(f"  主要情绪: {cv_in['dominant_emotion']}")
    print(f"  置信度: {cv_in['confidence']:.3f}")
    print(f"  效价: {cv_in['valence']:.3f}")
    print(f"  唤醒度: {cv_in['arousal']:.3f}")

    # Civis 输出
    civis = result['civis_state']
    print(f"\n[Civis 状态]")
    if 'hormones' in civis:
        hormones = civis['hormones']
        if isinstance(hormones, dict):
            print(f"  皮质醇: {hormones.get('cortisol', 0.3):.3f}")
            print(f"  催产素: {hormones.get('oxytocin', 0.3):.3f}")

    print(f"\n[完整摘要]")
    print(result['agent_summary'])

    print("\n" + "-" * 70)
    print(" Demo 2 完成!")
    print("-" * 70)


def main():
    """主入口"""
    print("=" * 70)
    print(" Censor CVEmotionBridge → Civis Lucri-Faber 集成演示")
    print("=" * 70)
    print("\n方案 1: 使用现有 FER/DeepFace backend")
    print("无需下载 MMEW 数据集")
    print("=" * 70)

    # 运行演示
    demo_synthetic_frames()
    demo_quick_test()

    # 最终汇总
    print("\n" + "=" * 70)
    print(" 集成演示完成!")
    print("=" * 70)

    print("\n架构图:")
    print("""
    ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
    │  Camera/     │     │ CVEmotionBridge │     │ Civis Agent   │
    │  Video Frame │ --> │ (FER/DeepFace)  │ --> │               │
    └──────────────┘     └─────────────────┘     └───────┬───────┘
                                                      │
                              ┌─────────────────────────┼─────────────────────┐
                              │                         │                     │
                              ▼                         ▼                     ▼
                       ┌─────────────┐          ┌─────────────┐       ┌─────────────┐
                       │ Advanced    │          │  Hormone    │       │    HPA      │
                       │ Emotion     │          │  System     │       │    Axis     │
                       │ System      │          │             │       │             │
                       └─────────────┘          └─────────────┘       └─────────────┘
                              │                         │
                              ▼                         ▼
                       mood_valence             cortisol ↑↓
                       mood_arousal             oxytocin ↑↓
    """)

    print("\n下一步:")
    print("  1. 运行实时摄像头: demo.run_camera()")
    print("  2. 处理视频文件: demo.process_video_file('video.mp4')")
    print("  3. 集成到完整 Civis agent")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()