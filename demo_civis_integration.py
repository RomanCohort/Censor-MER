"""
CV Emotion Bridge - Civis Lucri-Faber 集成示例

展示如何将 Censor 的 CV 情绪检测输出连接到 Civis 的仿生脑系统。

核心流程:
    Camera/Video -> CVEmotionBridge -> Civis AdvancedEmotionSystem -> Agent

运行方式:
    python demo_civis_integration.py
"""

import cv2
import numpy as np
import torch
import sys
import time
from pathlib import Path

# Censor 路径
sys.path.insert(0, str(Path(__file__).parent))

# Civis 路径 (假设在 D:/civis_lucri_faber)
CIVIS_PATH = Path(__file__).parent.parent / 'civis_lucri_faber'
if CIVIS_PATH.exists():
    sys.path.insert(0, str(CIVIS_PATH))
    CIVIS_AVAILABLE = True
else:
    CIVIS_AVAILABLE = False
    print(f"[WARN] Civis path not found: {CIVIS_PATH}")

from model.cv_emotion_bridge import (
    CVEmotionBridge,
    create_bridge_for_civis,
    EmotionDetectionResult,
)


class CivisIntegrationDemo:
    """
    Censor -> Civis 集成演示

    展示三种集成方式:
    1. 简单模式: 直接传递 emotion_tensor
    2. 增强模式: 传递多模态信号 (情绪 + 心率 + 眼动)
    3. 完整模式: 直接注入 Agent 事件流
    """

    def __init__(self, use_civis: bool = True, device: str = 'cpu'):
        self.use_civis = use_civis
        self.device = device

        # CV Emotion Bridge
        self.bridge = create_bridge_for_civis(device=device)
        print(f"[INFO] CV Bridge initialized with backend: {self.bridge.backend_name}")

        # Civis Advanced Emotion System (如果可用)
        if use_civis and CIVIS_AVAILABLE:
            try:
                from core.advanced_emotion_integration import IntegratedAdvancedEmotionSystem
                self.advanced_emotion = IntegratedAdvancedEmotionSystem(
                    input_dim=64,
                    hidden_dim=64,
                )
                print("[INFO] Civis Advanced Emotion System initialized")
                self.civis_ready = True
            except ImportError as e:
                print(f"[WARN] Could not import Civis emotion system: {e}")
                self.civis_ready = False
        else:
            self.civis_ready = False
            self.advanced_emotion = None

        # Agent State (模拟)
        self.internal_state = torch.randn(64)

        # 历史
        self._history = []

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        处理单帧: CV检测 -> Civis处理

        Returns:
            dict with both CV and Civis outputs
        """
        # 1. CV 情绪检测
        cv_result = self.bridge.detect(frame)

        # 2. 获取 Civis 兼容格式
        civis_input = self.bridge.get_civis_compatible_output()

        # 3. 如果 Civis 可用, 调用高级情绪系统
        if self.civis_ready and self.advanced_emotion is not None:
            civis_output = self._process_with_civis(civis_input)
        else:
            civis_output = civis_input

        # 4. 记录历史
        self._history.append({
            'cv_result': cv_result,
            'civis_output': civis_output,
            'timestamp': time.time(),
        })

        return {
            'cv': cv_result,
            'civis': civis_output,
        }

    def _process_with_civis(self, cv_input: dict) -> dict:
        """
        将 CV 输出传入 Civis 高级情绪系统

        Civis 接口:
            IntegratedAdvancedEmotionSystem.process(
                state: Tensor,
                user_emotion: Tensor,  # 来自 CV
                user_proximity: float,
                external_cortisol: float,
                ...
            )
        """
        # 构建 user_emotion tensor (扩展到 Civis 期望的维度)
        emotion_7dim = cv_input['emotion_tensor']  # [7]

        # Civis 期望 user_emotion 是 64 维, 我们用重复 + 零填充
        # 实际应用中应该有更好的编码方式
        user_emotion_64dim = torch.zeros(64)
        user_emotion_64dim[:7] = emotion_7dim
        user_emotion_64dim[7:14] = emotion_7dim * 0.5  # 辅助特征
        user_emotion_64dim[14:21] = emotion_7dim * 0.25

        # 添加多模态信号
        if cv_input['rppg_heart_rate'] is not None:
            user_emotion_64dim[21] = cv_input['rppg_heart_rate'] / 150.0  # 归一化心率

        if cv_input['gaze_direction'] is not None:
            user_emotion_64dim[22] = cv_input['gaze_direction'][0]
            user_emotion_64dim[23] = cv_input['gaze_direction'][1]

        user_emotion_64dim[24] = cv_input['arousal']
        user_emotion_64dim[25] = cv_input['valence']
        user_emotion_64dim[26] = cv_input['criticality']
        user_emotion_64dim[27] = cv_input['velocity']

        try:
            result = self.advanced_emotion.process(
                state=self.internal_state,
                user_emotion=user_emotion_64dim,
                user_proximity=0.7,  # 模拟近距离交互
                external_cortisol=cv_input['arousal'] * 0.3,  # 高唤醒 -> 高皮质醇
                external_oxytocin=max(0, cv_input['valence']) * 0.2,  # 正情绪 -> 催产素
            )

            return {
                'advanced_state': result.get('advanced_state'),
                'mood_state': result.get('mood_state'),
                'emotion_velocity': result.get('emotion_velocity'),
                'criticality': result.get('criticality'),
                # 保留原始 CV 输入
                'cv_input': cv_input,
            }
        except Exception as e:
            print(f"[WARN] Civis processing error: {e}")
            return {'error': str(e), 'cv_input': cv_input}

    def run_camera(self, camera_id: int = 0):
        """运行实时摄像头演示"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_id}")
            return

        print("\n[INFO] Starting real-time integration demo")
        print("Press 'q' to quit")

        # 设置较低分辨率以提高速度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            output = self.process_frame(frame)
            cv_result = output['cv']
            civis_output = output['civis']

            # 绘制 CV 结果
            frame = self._draw_cv_results(frame, cv_result)

            # 绘制 Civis 结果 (如果可用)
            if self.civis_ready:
                frame = self._draw_civis_results(frame, civis_output)

            # 绘制 FPS
            fps = frame_count / (time.time() - start_time) if frame_count > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frame_count += 1

            cv2.imshow('Censor -> Civis Integration', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # 打印汇总
        self._print_summary()

    def _draw_cv_results(self, frame: np.ndarray, result: EmotionDetectionResult) -> np.ndarray:
        """绘制 CV 检测结果"""
        # 主要情绪
        text = f"CV: {result.dominant_emotion} ({result.confidence:.2f})"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 情绪概率条
        emotions = ['Ang', 'Dis', 'Fear', 'Hap', 'Sad', 'Sur', 'Neu']
        for i, (emo, val) in enumerate(zip(emotions, result.emotion_tensor)):
            bar_w = int(val.item() * 50)
            y = 50 + i * 15
            cv2.rectangle(frame, (10, y), (10 + bar_w, y + 12),
                          (100 + i * 20, 200 - i * 10, 150), -1)
            cv2.putText(frame, emo, (65, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _draw_civis_results(self, frame: np.ndarray, civis_output: dict) -> np.ndarray:
        """绘制 Civis 系统输出"""
        x_offset = frame.shape[1] - 150

        if 'advanced_state' in civis_output and civis_output['advanced_state']:
            state = civis_output['advanced_state']

            # 心境状态
            if hasattr(state, 'mood_valence'):
                val = state.mood_valence
                ar = state.mood_arousal
                cv2.putText(frame, f"Val: {val:.2f}", (x_offset, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                cv2.putText(frame, f"Aro: {ar:.2f}", (x_offset, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # 皮质醇/催产素 (如果有)
        if 'cv_input' in civis_output:
            cv_input = civis_output['cv_input']
            if 'external_cortisol' in cv_input:
                cv2.putText(frame, f"Cor: {cv_input['external_cortisol']:.2f}",
                            (x_offset, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 100), 1)

        return frame

    def _print_summary(self):
        """打印运行汇总"""
        print("\n" + "=" * 50)
        print(" Integration Demo Summary")
        print("=" * 50)

        stats = self.bridge.get_stats()
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Average FPS: {stats['fps']:.1f}")
        print(f"  Backend used: {stats['backend']}")

        if self._history:
            # 最后几帧的情绪变化
            last_emotions = [h['cv_result'].dominant_emotion for h in self._history[-10:]]
            print(f"  Recent emotions: {last_emotions}")

        print("=" * 50)


def demo_simple():
    """简单集成演示"""
    print("\n" + "=" * 60)
    print(" Demo 1: Simple CV -> Civis Integration")
    print("=" * 60)

    demo = CivisIntegrationDemo(use_civis=CIVIS_AVAILABLE)

    # 创建测试帧
    print("\n[1] Creating test frames...")
    frames = []
    for i in range(20):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # 模拟不同情绪的视觉特征
        if i < 10:
            # "Happy" frames
            cv2.ellipse(frame, (112, 112), (60, 80), 0, 0, 360, (255, 220, 180), -1)
        else:
            # "Sad" frames
            cv2.ellipse(frame, (112, 112), (60, 80), 0, 0, 360, (120, 100, 150), -1)
        frames.append(frame)

    print(f"    Created {len(frames)} test frames")

    # 处理
    print("\n[2] Processing frames through CV -> Civis pipeline...")
    results = []
    for i, frame in enumerate(frames):
        output = demo.process_frame(frame)
        results.append(output)

        cv = output['cv']
        civis = output['civis']

        print(f"    Frame {i}: CV={cv.dominant_emotion} ({cv.confidence:.2f})")

        if 'mood_valence' in civis:
            print(f"             Civis mood: val={civis['mood_valence']:.2f}, aro={civis['mood_arousal']:.2f}")

    print("\n[PASS] Simple demo completed")


def demo_sequence():
    """序列处理演示"""
    print("\n" + "=" * 60)
    print(" Demo 2: Sequence Processing (Temporal Dynamics)")
    print("=" * 60)

    demo = CivisIntegrationDemo(use_civis=CIVIS_AVAILABLE)

    # 创建视频序列
    print("\n[1] Creating test video sequence...")
    frames = []
    for i in range(30):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # 模拟情绪渐变: neutral -> happy -> sad -> neutral
        phase = i / 30
        if phase < 0.25:
            brightness = 150  # neutral
        elif phase < 0.5:
            brightness = 200 + int(50 * (phase - 0.25) * 4)  # happy increasing
        elif phase < 0.75:
            brightness = 200 - int(100 * (phase - 0.5) * 4)  # happy -> sad
        else:
            brightness = 100 + int(50 * (phase - 0.75) * 4)  # sad -> neutral

        cv2.ellipse(frame, (112, 112), (60, 80), 0, 0, 360, (brightness, brightness-20, brightness-40), -1)
        frames.append(frame)

    print(f"    Created {len(frames)} frames with emotion transition")

    # 使用序列处理
    print("\n[2] Processing as sequence...")
    try:
        cv_results = demo.bridge.detect_sequence(frames)
        print(f"    Processed {len(cv_results)} frames in batch")

        # 分析情绪变化
        emotions = [r.dominant_emotion for r in cv_results]
        print(f"    Emotion sequence: {emotions}")

    except Exception as e:
        print(f"    [WARN] Sequence processing error: {e}")

    print("\n[PASS] Sequence demo completed")


def main():
    print("=" * 60)
    print(" Censor -> Civis Lucri-Faber Integration Demo")
    print("=" * 60)

    print(f"\nCensor path: {Path(__file__).parent}")
    print(f"Civis path: {CIVIS_PATH} (available: {CIVIS_AVAILABLE})")

    # 运行演示
    demo_simple()
    demo_sequence()

    # 可选: 实时摄像头
    print("\n" + "=" * 60)
    print(" Optional: Real-time Camera Integration")
    print("=" * 60)
    print("\nWould you like to run real-time camera integration?")
    print("This requires a camera and may be slow without GPU acceleration.")
    print("\nTo run camera demo manually:")
    print("    demo = CivisIntegrationDemo()")
    print("    demo.run_camera()")

    print("\n" + "=" * 60)
    print(" Demo Complete!")
    print("=" * 60)

    print("\nIntegration Architecture:")
    print("""
    +-------------+     +----------------+     +------------------+
    |   Camera    | --> | CVEmotionBridge| --> | Civis Advanced   |
    |   Frame     |     | (Censor MER)   |     | Emotion System   |
    +-------------+     +----------------+     +------------------+
                               |                        |
                               v                        v
                        +-------------+          +-------------+
                        |  Civis      |          |   Hormone   |
                        |  Agent      | <-- --> |   System    |
                        +-------------+          +-------------+

    Key Signal Flow:
    - CV detects: emotion, heart rate (rPPG), gaze
    - Civis receives: user_emotion tensor, arousal, valence
    - Hormones respond: cortisol (stress), oxytocin (positive)
    """)


if __name__ == '__main__':
    main()