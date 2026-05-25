# =============================================================================
# Improved Temporal Dynamics Model for Censor-G SNN
# =============================================================================
# 改进点：
#   1. 引入简化Hodgkin-Huxley神经元模型
#   2. 添加不同情感的典型时长参数（基于微表情研究文献）
#   3. 实现肌肉疲劳效应（重复收缩后强度衰减）
#   4. 添加微表情强度的时间调制（亚阈值微表情）
#
# 神经科学依据：
#   - Hodgkin-Huxley模型：描述离子通道动力学
#   - 微表情持续时间：1/25-1/5秒（Yan et al., 2013）
#   - 肌肉疲劳：ATP耗尽导致收缩力下降
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional

# AU常量定义
AU_INFO = {
    'AU1': {'name': 'Inner Brow Raiser', 'muscle_type': 'fast', 'region': 'eyebrow_inner',
            'tau_m': 5.0, 'max_intensity': 0.8, 'fatigue_rate': 0.02},
    'AU2': {'name': 'Outer Brow Raiser', 'muscle_type': 'fast', 'region': 'eyebrow_outer',
            'tau_m': 5.0, 'max_intensity': 0.8, 'fatigue_rate': 0.02},
    'AU4': {'name': 'Brow Lowerer', 'muscle_type': 'fast', 'region': 'eyebrow',
            'tau_m': 6.0, 'max_intensity': 0.7, 'fatigue_rate': 0.03},
    'AU5': {'name': 'Upper Lid Raiser', 'muscle_type': 'fast', 'region': 'eye_upper',
            'tau_m': 4.0, 'max_intensity': 0.9, 'fatigue_rate': 0.01},
    'AU6': {'name': 'Cheek Raiser', 'muscle_type': 'mixed', 'region': 'cheek',
            'tau_m': 10.0, 'max_intensity': 0.7, 'fatigue_rate': 0.04},
    'AU7': {'name': 'Lid Tightener', 'muscle_type': 'fast', 'region': 'eye',
            'tau_m': 4.5, 'max_intensity': 0.6, 'fatigue_rate': 0.02},
    'AU9': {'name': 'Nose Wrinkler', 'muscle_type': 'fast', 'region': 'nose',
            'tau_m': 5.5, 'max_intensity': 0.7, 'fatigue_rate': 0.03},
    'AU10': {'name': 'Upper Lip Raiser', 'muscle_type': 'mixed', 'region': 'lip_upper',
             'tau_m': 12.0, 'max_intensity': 0.6, 'fatigue_rate': 0.05},
    'AU12': {'name': 'Lip Corner Puller', 'muscle_type': 'mixed', 'region': 'mouth_corner',
             'tau_m': 15.0, 'max_intensity': 0.85, 'fatigue_rate': 0.06},
    'AU14': {'name': 'Dimpler', 'muscle_type': 'slow', 'region': 'mouth_corner',
             'tau_m': 25.0, 'max_intensity': 0.5, 'fatigue_rate': 0.08},
    'AU15': {'name': 'Lip Corner Depressor', 'muscle_type': 'slow', 'region': 'mouth_corner',
             'tau_m': 28.0, 'max_intensity': 0.6, 'fatigue_rate': 0.07},
    'AU17': {'name': 'Chin Raiser', 'muscle_type': 'slow', 'region': 'chin',
             'tau_m': 30.0, 'max_intensity': 0.5, 'fatigue_rate': 0.09},
    'AU20': {'name': 'Lip Stretcher', 'muscle_type': 'mixed', 'region': 'mouth',
             'tau_m': 18.0, 'max_intensity': 0.6, 'fatigue_rate': 0.05},
    'AU23': {'name': 'Lip Tightener', 'muscle_type': 'slow', 'region': 'mouth',
             'tau_m': 22.0, 'max_intensity': 0.5, 'fatigue_rate': 0.08},
    'AU24': {'name': 'Lip Pressor', 'muscle_type': 'slow', 'region': 'mouth',
             'tau_m': 24.0, 'max_intensity': 0.4, 'fatigue_rate': 0.09},
    'AU25': {'name': 'Lips Part', 'muscle_type': 'mixed', 'region': 'mouth',
             'tau_m': 20.0, 'max_intensity': 0.7, 'fatigue_rate': 0.04},
    'AU26': {'name': 'Jaw Drop', 'muscle_type': 'slow', 'region': 'jaw',
             'tau_m': 35.0, 'max_intensity': 0.6, 'fatigue_rate': 0.10},
}

AU_INDEX = {au: i for i, au in enumerate(sorted(AU_INFO.keys()))}


# =============================================================================
# Part 1: 情感时长参数（基于微表情研究）
# =============================================================================
# 参考文献：Yan et al. (2013) CASME II数据库分析
#   - Happiness: 平均时长约8-10帧 (25fps)
#   - Surprise: 平均时长约6-8帧 (较快)
#   - Disgust: 平均时长约10-12帧
#   - Repression/Others: 平均时长约12-16帧
# =============================================================================

EMOTION_DURATION_PARAMS = {
    0: {  # Happiness
        'name': 'happiness',
        'typical_frames': 10,
        'onset_frames': 3,    # 快速onset
        'apex_frames': 2,
        'offset_frames': 5,
        'intensity_peak': 0.85,
        'velocity_profile': 'smooth',  # 平滑上升
    },
    1: {  # Surprise
        'name': 'surprise',
        'typical_frames': 6,
        'onset_frames': 2,    # 非常快速onset
        'apex_frames': 1,
        'offset_frames': 3,
        'intensity_peak': 0.9,
        'velocity_profile': 'sharp',  # 急剧上升
    },
    2: {  # Disgust
        'name': 'disgust',
        'typical_frames': 12,
        'onset_frames': 4,
        'apex_frames': 3,
        'offset_frames': 5,
        'intensity_peak': 0.7,
        'velocity_profile': 'gradual',  # 逐渐上升
    },
    3: {  # Repression
        'name': 'repression',
        'typical_frames': 16,
        'onset_frames': 5,    # 较慢onset
        'apex_frames': 4,
        'offset_frames': 7,
        'intensity_peak': 0.6,
        'velocity_profile': 'subtle',  # 微妙上升
    },
}


# =============================================================================
# Part 2: 简化Hodgkin-Huxley神经元模型
# =============================================================================
# 原始HH模型描述离子通道动力学：
#   C * dV/dt = I - gNa*m^3*h*(V-ENa) - gK*n^4*(V-EK) - gL*(V-EL)
#
# 简化版本（Izhikevich 2003）：
#   dV/dt = 0.04*V^2 + 5*V + 140 - u + I
#   du/dt = a*(b*V - u)
#   当 V > 30: V = c, u = u + d
#
# 参数映射：
#   a: 恢复时间尺度（快肌小，慢肌大）
#   b: V对u的敏感度
#   c: 重置后膜电位
#   d: 重置后u增量
# =============================================================================

class SimplifiedHHNeuron(nn.Module):
    """
    简化Hodgkin-Huxley神经元模型

    Izhikevich模型参数映射到肌肉纤维类型：
      - 快肌：a=0.02, b=0.2, c=-65, d=8 (快速发放)
      - 混合肌：a=0.02, b=0.2, c=-65, d=6
      - 慢肌：a=0.01, b=0.2, c=-65, d=2 (缓慢发放)
    """

    def __init__(self, muscle_type='mixed'):
        super().__init__()

        # 根据肌肉类型设置参数
        if muscle_type == 'fast':
            self.a = 0.02   # 快恢复
            self.b = 0.2
            self.c = -65   # 重置电位
            self.d = 8     # 大跳跃
        elif muscle_type == 'slow':
            self.a = 0.01   # 慢恢复
            self.b = 0.2
            self.c = -65
            self.d = 2     # 小跳跃
        else:  # mixed
            self.a = 0.02
            self.b = 0.2
            self.c = -65
            self.d = 6

        # 状态变量（膜电位V，恢复变量u）
        self.register_buffer('V', torch.tensor(-70.0))  # 静息电位
        self.register_buffer('u', torch.tensor(self.b * -70.0))

    def step(self, I_input):
        """
        单步模拟

        Args:
            I_input (torch.Tensor): 输入电流

        Returns:
            spike (torch.Tensor): 是否发放
            V (torch.Tensor): 当前膜电位
        """
        # HH动力学
        V_new = 0.04 * self.V ** 2 + 5 * self.V + 140 - self.u + I_input
        u_new = self.a * (self.b * self.V - self.u)

        # 发放判断
        spike = (self.V >= 30).float()

        # 重置
        if spike > 0:
            self.V = torch.tensor(self.c)
            self.u = self.u + self.d
        else:
            self.V = V_new
            self.u = u_new

        return spike, self.V

    def reset(self):
        """重置到静息状态"""
        self.V = torch.tensor(-70.0)
        self.u = torch.tensor(self.b * -70.0)


# =============================================================================
# Part 3: 改进的V3层：完整时间动力学
# =============================================================================

class V3ImprovedTemporalDynamics(nn.Module):
    """
    改进的V3层：完整时间动力学模型

    特性：
      1. 简化HH神经元模型（不同肌肉类型）
      2. 情感特异性时长参数
      3. 肌肉疲劳效应
      4. 亚阈值微表情强度调制
    """

    def __init__(self, num_au=17, num_frames=16, frame_rate=25):
        super().__init__()

        self.num_au = num_au
        self.num_frames = num_frames
        self.frame_rate = frame_rate  # fps

        # AU膜时间参数
        tau_m_values = []
        max_intensity_values = []
        fatigue_rates = []
        muscle_types = []

        for au in sorted(AU_INFO.keys()):
            info = AU_INFO[au]
            tau_m_values.append(info['tau_m'])
            max_intensity_values.append(info['max_intensity'])
            fatigue_rates.append(info['fatigue_rate'])
            muscle_types.append(info['muscle_type'])

        self.register_buffer('tau_m', torch.tensor(tau_m_values))
        self.register_buffer('max_intensity', torch.tensor(max_intensity_values))
        self.register_buffer('fatigue_rate', torch.tensor(fatigue_rates))

        # 简化HH神经元（每个AU一个）
        self.hh_neurons = nn.ModuleList([
            SimplifiedHHNeuron(muscle_type=mt)
            for mt in muscle_types
        ])

        # 可学习的相位比例
        self.onset_ratio = nn.Parameter(torch.tensor(0.3))
        self.apex_ratio = nn.Parameter(torch.tensor(0.2))

        # 强度调制参数
        self.intensity_modulation = nn.Parameter(torch.tensor(1.0))

    def forward(self, au_activation, num_frames=None, emotion_class=None, intensity=None):
        """
        改进的时间动力学生成

        Args:
            au_activation (torch.Tensor): AU激活强度，shape (B, 17)
            num_frames (int): 输出帧数
            emotion_class (torch.Tensor): 情感类别索引，shape (B,)
            intensity (torch.Tensor): 微表情强度，shape (B,)

        Returns:
            au_temporal (torch.Tensor): AU时间场，shape (B, 17, T)
            temporal_info (Dict): 时间动力学信息
        """
        T = num_frames or self.num_frames
        B = au_activation.shape[0]

        # 获取情感时长参数
        if emotion_class is not None:
            duration_params = self._get_emotion_duration(emotion_class, T)
        else:
            duration_params = self._get_default_duration(T)

        # 初始化疲劳状态
        fatigue_state = torch.zeros(B, self.num_au)

        # 生成时间曲线
        temporal_curves = []

        for au_idx in range(self.num_au):
            tau = self.tau_m[au_idx]
            max_int = self.max_intensity[au_idx]
            fatigue_r = self.fatigue_rate[au_idx]

            # 为每个样本生成曲线
            curves_batch = []

            for b in range(B):
                onset_frames = duration_params[b]['onset_frames']
                apex_frames = duration_params[b]['apex_frames']
                offset_frames = duration_params[b]['offset_frames']
                velocity_profile = duration_params[b]['velocity_profile']
                intensity_peak = duration_params[b]['intensity_peak']

                # 应用AU强度
                au_int = au_activation[b, au_idx] * intensity_peak * self.intensity_modulation

                # 生成曲线（包含疲劳）
                curve = self._generate_curve_with_fatigue(
                    T, tau, au_int, fatigue_r,
                    onset_frames, apex_frames, offset_frames,
                    velocity_profile, fatigue_state[b, au_idx]
                )

                # 更新疲劳状态
                fatigue_state[b, au_idx] = fatigue_state[b, au_idx] + fatigue_r * curve.sum()

                curves_batch.append(curve)

            temporal_curves.append(torch.stack(curves_batch))

        # 组合
        temporal_curves = torch.stack(temporal_curves, dim=1)  # (B, 17, T)

        temporal_info = {
            'duration_params': duration_params,
            'fatigue_state': fatigue_state,
            'velocity_profiles': [duration_params[i]['velocity_profile'] for i in range(B)]
        }

        return temporal_curves, temporal_info

    def _get_emotion_duration(self, emotion_class, T):
        """获取情感特异性时长参数"""
        B = emotion_class.shape[0]
        params = []

        for b in range(B):
            emotion = emotion_class[b].item()
            emotion_params = EMOTION_DURATION_PARAMS.get(emotion, EMOTION_DURATION_PARAMS[0])

            # 根据实际帧数调整
            scale = T / emotion_params['typical_frames']

            params.append({
                'onset_frames': int(emotion_params['onset_frames'] * scale),
                'apex_frames': int(emotion_params['apex_frames'] * scale),
                'offset_frames': int(emotion_params['offset_frames'] * scale),
                'velocity_profile': emotion_params['velocity_profile'],
                'intensity_peak': emotion_params['intensity_peak'],
                'emotion': emotion_params['name']
            })

        return params

    def _get_default_duration(self, T):
        """获取默认时长参数"""
        return [{
            'onset_frames': int(T * 0.3),
            'apex_frames': int(T * 0.2),
            'offset_frames': int(T * 0.5),
            'velocity_profile': 'smooth',
            'intensity_peak': 0.8,
            'emotion': 'default'
        }]

    def _generate_curve_with_fatigue(self, T, tau_m, intensity, fatigue_rate,
                                       onset_frames, apex_frames, offset_frames,
                                       velocity_profile, initial_fatigue):
        """
        生成包含疲劳效应的时间曲线

        肌肉疲劳模型：
          - 每次收缩消耗ATP
          - 疲劳累积导致强度衰减
          - 指数衰减：intensity * exp(-fatigue * t)
        """
        curve = torch.zeros(T)

        # 计算边界
        onset_end = onset_frames
        apex_end = onset_end + apex_frames

        # 疲劳衰减因子
        fatigue_factor = math.exp(-initial_fatigue * 0.1)

        # === Onset阶段 ===
        for t in range(onset_end):
            progress = t / onset_end

            # 根据速度曲线调整
            if velocity_profile == 'sharp':
                # 急剧上升（惊讶）
                v = progress ** 0.3
            elif velocity_profile == 'gradual':
                # 逐渐上升（厌恶）
                v = progress ** 1.5
            elif velocity_profile == 'subtle':
                # 微妙上升（压抑）
                v = progress ** 2.0
            else:
                # 平滑上升（快乐）
                v = progress ** 0.7

            # 膜电位积分
            curve[t] = intensity * (1 - math.exp(-v * t / tau_m)) * fatigue_factor

        # === Apex阶段 ===
        for t in range(onset_end, apex_end):
            # Apex期间考虑疲劳衰减
            t_in_apex = t - onset_end
            fatigue_decay = math.exp(-fatigue_rate * t_in_apex)
            curve[t] = intensity * fatigue_factor * fatigue_decay

        # === Offset阶段 ===
        tau_decay = tau_m * 1.5
        for t in range(apex_end, T):
            dt = t - apex_end

            # 自然衰减 + 疲劳累积
            natural_decay = math.exp(-dt / tau_decay)
            fatigue_decay = math.exp(-fatigue_rate * (apex_frames + dt))
            curve[t] = intensity * natural_decay * fatigue_factor * fatigue_decay

        return curve

    def _generate_curve_with_hh(self, T, neuron, intensity):
        """
        使用简化HH模型生成时间曲线

        通过连续刺激观察发放模式
        """
        neuron.reset()
        curve = torch.zeros(T)

        for t in range(T):
            # 输入电流随时间变化
            I_input = intensity * (1 - math.exp(-t / 10))
            spike, V = neuron.step(torch.tensor(I_input))

            # 发放率编码强度
            curve[t] = spike

        # 平滑发放序列
        curve = F.avg_pool1d(curve.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1).squeeze()

        return curve


# =============================================================================
# Part 4: 亚阈值微表情强度调制
# =============================================================================
# 亚阈值微表情：强度低于正常微表情阈值（<0.5）
# 特点：
#   - 更短时长
#   - 更低强度
#   - 更快onset/offset
# =============================================================================

class SubthresholdIntensityModulator(nn.Module):
    """
    亚阈值微表情强度调制器

    根据强度参数调整时间动力学：
      - 正常强度 (0.5-1.0): 标准时间曲线
      - 亚阈值 (0.1-0.5): 缩短时长，降低峰值
      - 极微弱 (<0.1): 极短时长，极低强度
    """

    def __init__(self):
        super().__init__()

        # 强度阈值
        self.normal_threshold = 0.5
        self.subthreshold_boundary = 0.1

    def forward(self, intensity, temporal_curve):
        """
        强度调制

        Args:
            intensity (torch.Tensor): 强度值，shape (B,) 或 scalar
            temporal_curve (torch.Tensor): 原始时间曲线，shape (B, 17, T)

        Returns:
            modulated_curve (torch.Tensor): 调制后的时间曲线
            modulation_info (Dict): 调制信息
        """
        if isinstance(intensity, float):
            intensity = torch.tensor([intensity])

        intensity_val = intensity.mean().item()

        if intensity_val < self.subthreshold_boundary:
            # 极微弱：极短时长，极低强度
            duration_factor = 0.3
            intensity_factor = 0.1
            modulation_type = 'extremely_subtle'

        elif intensity_val < self.normal_threshold:
            # 亚阈值：缩短时长，降低峰值
            duration_factor = 0.5 + intensity_val
            intensity_factor = intensity_val
            modulation_type = 'subthreshold'

        else:
            # 正常强度
            duration_factor = 1.0
            intensity_factor = 1.0
            modulation_type = 'normal'

        # 应用调制
        T = temporal_curve.shape[2]
        effective_frames = int(T * duration_factor)

        # 截断或填充
        if effective_frames < T:
            # 截断
            modulated_curve = temporal_curve[:, :, :effective_frames]
            # 填充零
            padding = torch.zeros_like(temporal_curve[:, :, effective_frames:])
            modulated_curve = torch.cat([modulated_curve, padding], dim=2)
        else:
            modulated_curve = temporal_curve

        # 强度缩放
        modulated_curve = modulated_curve * intensity_factor

        modulation_info = {
            'intensity': intensity_val,
            'modulation_type': modulation_type,
            'duration_factor': duration_factor,
            'intensity_factor': intensity_factor,
            'effective_frames': effective_frames
        }

        return modulated_curve, modulation_info


# =============================================================================
# Part 5: 集成改进的V3层到完整系统
# =============================================================================

def create_improved_v3(num_au=17, num_frames=16, frame_rate=25):
    """工厂函数：创建改进的V3层"""
    return V3ImprovedTemporalDynamics(
        num_au=num_au,
        num_frames=num_frames,
        frame_rate=frame_rate
    )


# =============================================================================
# Demo
# =============================================================================

def demo_improved_temporal():
    """Demo改进的时间动力学"""
    print("\n" + "="*60)
    print("Improved V3 Temporal Dynamics Demo")
    print("="*60)

    # 创建模型
    model = V3ImprovedTemporalDynamics(num_au=17, num_frames=16, frame_rate=25)

    # 测试不同情感的时间曲线
    print("\n[Test 1] Emotion-Specific Duration")

    emotions = ['happiness', 'surprise', 'disgust', 'repression']
    emotion_indices = [0, 1, 2, 3]

    for emotion, idx in zip(emotions, emotion_indices):
        emotion_class = torch.tensor([idx])
        au_activation = torch.ones(1, 17) * 0.5

        au_temporal, info = model(au_activation, emotion_class=emotion_class)

        params = info['duration_params'][0]
        print(f"\n  {emotion}:")
        print(f"    Onset: {params['onset_frames']} frames")
        print(f"    Apex: {params['apex_frames']} frames")
        print(f"    Offset: {params['offset_frames']} frames")
        print(f"    Profile: {params['velocity_profile']}")
        print(f"    Peak Intensity: {params['intensity_peak']}")

        # 打印AU12的时间曲线样本
        au12_curve = au_temporal[0, AU_INDEX['AU12'], :]
        print(f"    AU12 curve (first 8): {au12_curve[:8].tolist()}")

    print("\n[Test 2] Fatigue Effect")
    # 多次生成，观察疲劳累积
    au_activation = torch.ones(1, 17) * 0.8

    for i in range(3):
        au_temporal, info = model(au_activation)
        fatigue = info['fatigue_state'][0, AU_INDEX['AU12']].item()
        print(f"  Generation {i+1}: AU12 fatigue = {fatigue:.4f}")

    print("\n[Test 3] Subthreshold Intensity")
    modulator = SubthresholdIntensityModulator()

    intensities = [0.05, 0.3, 0.7, 1.0]

    for intensity in intensities:
        au_temporal = torch.randn(1, 17, 16) * 0.5 + 0.5
        modulated, info = modulator(intensity, au_temporal)

        print(f"\n  Intensity {intensity}:")
        print(f"    Type: {info['modulation_type']}")
        print(f"    Duration factor: {info['duration_factor']}")
        print(f"    Intensity factor: {info['intensity_factor']}")
        print(f"    Effective frames: {info['effective_frames']}")

    print("\n[Test 4] HH Neuron Model")
    # 测试不同肌肉类型的HH神经元
    neuron_fast = SimplifiedHHNeuron('fast')
    neuron_slow = SimplifiedHHNeuron('slow')

    I_input = torch.tensor(10.0)  # 输入电流

    print("\n  Fast muscle neuron:")
    for t in range(10):
        spike, V = neuron_fast.step(I_input * (t / 10))
        print(f"    t={t}: spike={spike.item():.1f}, V={V.item():.2f}")

    print("\n  Slow muscle neuron:")
    neuron_slow.reset()
    for t in range(10):
        spike, V = neuron_slow.step(I_input * (t / 10))
        print(f"    t={t}: spike={spike.item():.1f}, V={V.item():.2f}")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_improved_temporal()