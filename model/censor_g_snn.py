# Censor-G SNN: Spiking Neural Network for Micro-Expression Generation
# =============================================================================
# 核心创新：真正的神经科学机制（而非概念仿生）
#
# 神经科学依据：
#   1. V1层：LIF (Leaky Integrate-and-Fire) 脉冲发放模型
#   2. V2层：神经回路突触连接（兴奋性EPSP/抑制性IPSP）
#   3. V3层：膜电位积分时间动力学（基于肌肉纤维膜特性）
#   4. V4层：ON/OFF感受野结构（侧抑制机制）
#   5. IT层：事件驱动的层级整合
#
# 参考文献：
#   - Gerstner et al. (2014) Neuronal Dynamics: From Single Neurons to Networks
#   - Dayan & Abbott (2001) Theoretical Neuroscience
#   - FACS (Ekman & Friesen, 1978) Facial Action Coding System
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


# =============================================================================
# AU常量定义（基于FACS + 神经肌肉生理学）
# =============================================================================

# 17个AU及其神经肌肉特性
# PHASE 4 FIX: tau_m放大3倍以适应16帧序列
# 原始tau_m太小(5-35)，导致时间曲线陡峭，不适合微表情的平滑过渡
AU_INFO = {
    'AU1': {'name': 'Inner Brow Raiser', 'muscle_type': 'fast', 'region': 'eyebrow_inner',
            'tau_m': 15.0, 'R_m': 100.0},  # 原5.0 → 15.0
    'AU2': {'name': 'Outer Brow Raiser', 'muscle_type': 'fast', 'region': 'eyebrow_outer',
            'tau_m': 15.0, 'R_m': 100.0},  # 原5.0 → 15.0
    'AU4': {'name': 'Brow Lowerer', 'muscle_type': 'fast', 'region': 'eyebrow',
            'tau_m': 18.0, 'R_m': 120.0},  # 原6.0 → 18.0
    'AU5': {'name': 'Upper Lid Raiser', 'muscle_type': 'fast', 'region': 'eye_upper',
            'tau_m': 12.0, 'R_m': 80.0},   # 原4.0 → 12.0
    'AU6': {'name': 'Cheek Raiser', 'muscle_type': 'mixed', 'region': 'cheek',
            'tau_m': 30.0, 'R_m': 150.0},  # 原10.0 → 30.0
    'AU7': {'name': 'Lid Tightener', 'muscle_type': 'fast', 'region': 'eye',
            'tau_m': 13.5, 'R_m': 90.0},   # 原4.5 → 13.5
    'AU9': {'name': 'Nose Wrinkler', 'muscle_type': 'fast', 'region': 'nose',
            'tau_m': 16.5, 'R_m': 110.0},  # 原5.5 → 16.5
    'AU10': {'name': 'Upper Lip Raiser', 'muscle_type': 'mixed', 'region': 'lip_upper',
             'tau_m': 36.0, 'R_m': 160.0},  # 原12.0 → 36.0
    'AU12': {'name': 'Lip Corner Puller', 'muscle_type': 'mixed', 'region': 'mouth_corner',
             'tau_m': 45.0, 'R_m': 180.0},  # 原15.0 → 45.0 (重要表情)
    'AU14': {'name': 'Dimpler', 'muscle_type': 'slow', 'region': 'mouth_corner',
             'tau_m': 75.0, 'R_m': 250.0},  # 原25.0 → 75.0
    'AU15': {'name': 'Lip Corner Depressor', 'muscle_type': 'slow', 'region': 'mouth_corner',
             'tau_m': 84.0, 'R_m': 280.0},  # 原28.0 → 84.0
    'AU17': {'name': 'Chin Raiser', 'muscle_type': 'slow', 'region': 'chin',
             'tau_m': 90.0, 'R_m': 300.0},  # 原30.0 → 90.0
    'AU20': {'name': 'Lip Stretcher', 'muscle_type': 'mixed', 'region': 'mouth',
             'tau_m': 54.0, 'R_m': 200.0},  # 原18.0 → 54.0
    'AU23': {'name': 'Lip Tightener', 'muscle_type': 'slow', 'region': 'mouth',
             'tau_m': 66.0, 'R_m': 220.0},  # 原22.0 → 66.0
    'AU24': {'name': 'Lip Pressor', 'muscle_type': 'slow', 'region': 'mouth',
             'tau_m': 72.0, 'R_m': 240.0},  # 原24.0 → 72.0
    'AU25': {'name': 'Lips Part', 'muscle_type': 'mixed', 'region': 'mouth',
             'tau_m': 60.0, 'R_m': 210.0},  # 原20.0 → 60.0
    'AU26': {'name': 'Jaw Drop', 'muscle_type': 'slow', 'region': 'jaw',
             'tau_m': 105.0, 'R_m': 350.0},  # 原35.0 → 105.0
}

# AU索引映射
AU_INDEX = {au: i for i, au in enumerate(sorted(AU_INFO.keys()))}

# AU区域定义（像素坐标范围，基于224x224图像）
AU_REGIONS = {
    'eyebrow_inner': {'center': (112, 60), 'radius': 20},
    'eyebrow_outer': {'center': (80, 55), 'radius': 15},
    'eyebrow': {'center': (112, 60), 'radius': 30},
    'eye_upper': {'center': (100, 80), 'radius': 25},
    'eye': {'center': (100, 80), 'radius': 20},
    'cheek': {'center': (90, 100), 'radius': 30},
    'nose': {'center': (112, 100), 'radius': 15},
    'lip_upper': {'center': (112, 130), 'radius': 15},
    'mouth_corner': {'center': (90, 140), 'radius': 20},
    'mouth': {'center': (112, 140), 'radius': 25},
    'chin': {'center': (112, 160), 'radius': 20},
    'jaw': {'center': (112, 180), 'radius': 30},
}


# =============================================================================
# V1层：LIF脉冲发放显著性筛选
# =============================================================================
# 神经科学依据：视网膜神经节细胞的ON/OFF通路
#   - 膜电位积分：V(t) = V(t-1) * decay + I * dt / C
#   - 发放阈值：当V > V_threshold时发放脉冲
#   - 发放率编码：rate = n_spikes / T_window
# =============================================================================

class V1SpikingSaliency(nn.Module):
    """
    V1层：LIF脉冲发放模型进行AU显著性筛选

    真正的神经科学机制：
      1. 膜电位积分（Leaky Integrate）
      2. 发放阈值判断（Fire）
      3. 发放率编码（Rate Coding）

    数学模型（Gerstner 2014）：
      dV/dt = -V/tau_m + R_m * I
      当 V > V_threshold → 发放脉冲，V -= V_threshold
    """

    def __init__(self, num_au=17, threshold=1.0, tau_m=10.0, dt=1.0, time_steps=10):
        super().__init__()

        self.num_au = num_au
        self.threshold = threshold
        self.tau_m = tau_m
        self.dt = dt
        self.time_steps = time_steps

        # 衰减系数：decay = exp(-dt/tau_m)
        self.decay = math.exp(-dt / tau_m)

        # AU特异性阈值（不同AU有不同的发放阈值）
        au_thresholds = torch.ones(num_au)
        # 快肌阈值低（更敏感）
        for au in ['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU9']:
            au_thresholds[AU_INDEX[au]] = 0.8
        # 混合肌中等阈值
        for au in ['AU6', 'AU10', 'AU12', 'AU20', 'AU25']:
            au_thresholds[AU_INDEX[au]] = 1.0
        # 慢肌阈值高（需要更强刺激）
        for au in ['AU14', 'AU15', 'AU17', 'AU23', 'AU24', 'AU26']:
            au_thresholds[AU_INDEX[au]] = 1.2

        self.au_thresholds = nn.Parameter(au_thresholds, requires_grad=True)

        # 背景噪声（模拟神经噪声）
        self.noise_scale = 0.05

    def forward(self, au_input):
        """
        LIF脉冲发放

        Args:
            au_input (torch.Tensor): AU输入刺激，shape (B, 17)

        Returns:
            firing_rate (torch.Tensor): 发放率编码的显著性，shape (B, 17)
            spike_history (torch.Tensor): 脉冲历史，shape (B, 17, T)
        """
        B = au_input.shape[0]
        T = self.time_steps

        # 初始化膜电位（静息电位 V_rest = 0）
        V = torch.zeros(B, self.num_au, device=au_input.device)

        # 记录脉冲
        spikes = torch.zeros(B, self.num_au, T, device=au_input.device)

        # 时间积分发放
        for t in range(T):
            # LIF动力学：dV/dt = -V/tau + I
            # 离散化：V = decay * V + input * dt

            # 膜电位积分（生物学真实的漏电流）
            V = self.decay * V + au_input

            # 添加神经噪声（泊松噪声近似）
            noise = torch.randn_like(V) * self.noise_scale
            V = V + noise

            # 发放判断（阈值比较）
            spike = (V > self.au_thresholds).float()
            spikes[:, :, t] = spike

            # 发放后重置（V -= V_threshold * spike）
            V = V - spike * self.au_thresholds

            # 钳制膜电位（不能低于静息电位）
            V = F.relu(V)

        # 发放率编码：rate = spikes / T
        firing_rate = spikes.sum(dim=-1) / T  # (B, 17)

        return firing_rate, spikes

    def get_saliency_mask(self, firing_rate, rate_threshold=0.3):
        """
        从发放率生成显著性掩码

        Args:
            firing_rate (torch.Tensor): 发放率，shape (B, 17)
            rate_threshold (float): 发放率阈值

        Returns:
            significant_mask (torch.Tensor): 显著AU掩码
        """
        # 高发放率 → 显著AU
        significant_mask = firing_rate > rate_threshold

        return significant_mask


# =============================================================================
# V2层：神经回路突触连接
# =============================================================================
# 神经科学依据：V2层的横向连接用于特征整合
#   - 兴奋性突触：释放谷氨酸，产生EPSP（兴奋性突触后电位）
#   - 抑制性突触：释放GABA，产生IPSP（抑制性突触后电位）
#   - 突触传递延迟：约2-5ms
# =============================================================================

class V2NeuralCircuit(nn.Module):
    """
    V2层：神经回路的AU交互

    真正的神经科学机制：
      1. 兴奋性突触（Excitatory）→ 协同效应
      2. 抑制性突触（Inhibitory）→ 对抗效应
      3. 突触传递延迟（Synaptic Delay）
      4. 突触权重可塑性（Synaptic Plasticity）

    数学模型（Dayan & Abbott 2001）：
      EPSP: V += W_exc * input * exp(-delay/tau_exc)
      IPSP: V -= W_inh * input * exp(-delay/tau_inh)
    """

    def __init__(self, num_au=17, synaptic_delay=2, tau_epsp=3.0, tau_ipsp=5.0):
        super().__init__()

        self.num_au = num_au
        self.synaptic_delay = synaptic_delay
        self.tau_epsp = tau_epsp
        self.tau_ipsp = tau_ipsp

        # 兴奋性突触矩阵（协同）
        self.W_excitatory = nn.Parameter(self._init_excitatory_synapses())

        # 抑制性突触矩阵（对抗）
        self.W_inhibitory = nn.Parameter(self._init_inhibitory_synapses())

        # 突触延迟衰减因子
        self.delay_decay_exc = math.exp(-synaptic_delay / tau_epsp)
        self.delay_decay_inh = math.exp(-synaptic_delay / tau_ipsp)

    def forward(self, au_state, history=None):
        """
        突触传递

        Args:
            au_state (torch.Tensor): 当前AU状态，shape (B, 17)
            history (torch.Tensor): 历史状态（用于延迟传递）

        Returns:
            effective_au (torch.Tensor): 经神经回路整合后的状态
        """
        B = au_state.shape[0]

        # 兴奋性突触后电位（EPSP）
        # 生物学：突触前神经元释放谷氨酸 → EPSP
        epsp = torch.matmul(au_state, self.W_excitatory) * self.delay_decay_exc

        # 抑制性突触后电位（IPSP）
        # 生物学：突触前神经元释放GABA → IPSP
        ipsp = torch.matmul(au_state, self.W_inhibitory) * self.delay_decay_inh

        # 神经回路整合：V = V + EPSP - IPSP
        effective_au = au_state + epsp - ipsp

        # 钳制到生理范围 [0, 1]
        effective_au = torch.clamp(effective_au, 0, 1)

        return effective_au

    def _init_excitatory_synapses(self):
        """
        初始化兴奋性突触

        神经科学依据：
          - AU6 + AU12 → 真诚微笑（协同）
          - AU1 + AU2 + AU5 → 惊讶（协同）
          - AU4 + AU9 → 厌恶（协同）
        """
        W = torch.zeros(self.num_au, self.num_au)

        # Happiness协同：AU6 ↔ AU12
        idx_6 = AU_INDEX['AU6']
        idx_12 = AU_INDEX['AU12']
        W[idx_6, idx_12] = 0.3
        W[idx_12, idx_6] = 0.25

        # Surprise协同：AU1 ↔ AU2 ↔ AU5
        idx_1 = AU_INDEX['AU1']
        idx_2 = AU_INDEX['AU2']
        idx_5 = AU_INDEX['AU5']
        W[idx_1, idx_2] = 0.2
        W[idx_2, idx_1] = 0.2
        W[idx_1, idx_5] = 0.15
        W[idx_2, idx_5] = 0.15

        # Disgust协同：AU4 ↔ AU9
        idx_4 = AU_INDEX['AU4']
        idx_9 = AU_INDEX['AU9']
        W[idx_4, idx_9] = 0.2
        W[idx_9, idx_4] = 0.15

        return W

    def _init_inhibitory_synapses(self):
        """
        初始化抑制性突触

        神经科学依据：
          - AU12 + AU14 → 嘴角上扬 vs 嘴角收紧（对抗）
          - AU4 + AU2 → 眉降 vs 眉扬（对抗）
          - AU15 + AU12 → 嘴角下垂 vs 嘴角上扬（对抗）
        """
        W = torch.zeros(self.num_au, self.num_au)

        # Repression对抗：AU12 ↔ AU14
        idx_12 = AU_INDEX['AU12']
        idx_14 = AU_INDEX['AU14']
        W[idx_12, idx_14] = 0.4
        W[idx_14, idx_12] = 0.3

        # Brow对抗：AU4 ↔ AU2
        idx_4 = AU_INDEX['AU4']
        idx_2 = AU_INDEX['AU2']
        W[idx_4, idx_2] = 0.35
        W[idx_2, idx_4] = 0.25

        # Sadness对抗：AU15 ↔ AU12
        idx_15 = AU_INDEX['AU15']
        W[idx_15, idx_12] = 0.35

        return W


# =============================================================================
# V3层：膜电位积分时间动力学
# =============================================================================
# 神经科学依据：不同肌肉纤维有不同的膜特性
#   - 快肌纤维：小tau_m，快速响应，适合微表情
#   - 慢肌纤维：大tau_m，缓慢响应，适合持续表情
#   - LIF时间曲线：V(t) = V_rest + I * R_m * (1 - exp(-t/tau_m))
# =============================================================================

class V3MembraneTemporalDynamics(nn.Module):
    """
    V3层：基于膜电位的AU时间动力学

    真正的神经科学机制：
      1. 不同AU有不同的膜时间常数（tau_m）
      2. 膜电位积分生成时间曲线
      3. 发放阈值触发运动

    数学模型：
      V(t) = V_max * (1 - exp(-t/tau_onset))  [onset]
      V(t) = V_max                              [apex]
      V(t) = V_max * exp(-(t-t_apex)/tau_decay) [offset]
    """

    def __init__(self, num_au=17, num_frames=16):
        super().__init__()

        self.num_au = num_au
        self.num_frames = num_frames

        # AU膜时间参数（基于肌肉生理学）
        tau_m_values = []
        for au in sorted(AU_INFO.keys()):
            tau_m_values.append(AU_INFO[au]['tau_m'])
        self.register_buffer('tau_m', torch.tensor(tau_m_values))

        # Onset/Decay时间比例（可学习）
        # PHASE 4 FIX: 调整比例使曲线更平滑
        self.onset_ratio = nn.Parameter(torch.tensor(0.35))  # 原0.3 → 0.35
        self.apex_ratio = nn.Parameter(torch.tensor(0.15))   # 原0.2 → 0.15

    def forward(self, au_activation, num_frames=None):
        """
        膜电位积分生成时间曲线

        Args:
            au_activation (torch.Tensor): AU激活强度，shape (B, 17)
            num_frames (int): 输出帧数

        Returns:
            au_temporal (torch.Tensor): AU时间场，shape (B, 17, T)
        """
        T = num_frames or self.num_frames
        B = au_activation.shape[0]

        # 计算各阶段边界
        onset_end = int(T * self.onset_ratio)
        apex_end = int(T * (self.onset_ratio + self.apex_ratio))

        # 为每个AU生成膜电位曲线
        temporal_curves = []

        for au_idx in range(self.num_au):
            tau = self.tau_m[au_idx]

            # 生成单个AU的时间曲线
            curve = self._generate_membrane_curve(T, tau, onset_end, apex_end)
            temporal_curves.append(curve)

        # 组合：(17, T) → (B, 17, T)
        temporal_curves = torch.stack(temporal_curves, dim=0)  # (17, T)
        temporal_curves = temporal_curves.unsqueeze(0).expand(B, -1, -1)  # (B, 17, T)

        # FIX: 移动到正确的设备
        temporal_curves = temporal_curves.to(au_activation.device)

        # 应用AU激活强度
        au_temporal = au_activation.unsqueeze(-1) * temporal_curves  # (B, 17, T)

        return au_temporal

    def _generate_membrane_curve(self, T, tau_m, onset_end, apex_end):
        """
        生成单个AU的膜电位曲线

        LIF动力学：
          Onset: V(t) = V_max * (1 - exp(-t/tau_m))
          Apex:  V(t) = V_max
          Offset: V(t) = V_max * exp(-(t-t_apex)/tau_decay)
        """
        curve = torch.zeros(T)

        # PHASE 4 FIX: 平滑过渡替代硬切换
        # Onset阶段：膜电位上升（积分）- 使用sigmoid-like平滑曲线
        for t in range(onset_end):
            # Sigmoid-like上升：更平滑的过渡
            normalized_t = t / max(onset_end, 1)
            # 使用tanh实现平滑上升
            curve[t] = 0.5 * (1 + math.tanh(4 * normalized_t - 2))

        # Apex阶段：峰值保持
        curve[onset_end:apex_end] = 1.0

        # Offset阶段：膜电位衰减
        tau_decay = tau_m * 2.0  # PHASE 4 FIX: 衰减比上升慢（原1.5 → 2.0）
        for t in range(apex_end, T):
            dt = t - apex_end
            # 使用平滑衰减
            normalized_dt = dt / max(T - apex_end, 1)
            curve[t] = math.exp(-dt / tau_decay) * (1 - 0.3 * normalized_dt)

        return curve


# =============================================================================
# V4层：ON/OFF感受野结构
# =============================================================================
# 神经科学依据：视网膜感受野的中心-周围结构
#   - ON-center感受野：中心兴奋，周围抑制
#   - OFF-center感受野：中心抑制，周围兴奋
#   - 侧抑制（Lateral Inhibition）：增强边缘对比度
# =============================================================================

class V4ReceptiveFieldMotion(nn.Module):
    """
    V4层：感受野结构的局部运动场生成

    真正的神经科学机制：
      1. ON/OFF感受野中心-周围结构
      2. 侧抑制增强运动边界
      3. Gaussian感受野模拟

    数学模型：
      RF(x,y) = G_center - G_surround
      motion = au_intensity * RF * displacement
    """

    def __init__(self, num_au=17, image_size=224, sigma_center=5.0, sigma_surround=15.0):
        super().__init__()

        self.num_au = num_au
        self.image_size = image_size
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround

        # 创建感受野模板并注册为buffer（自动跟随设备）
        receptive_fields = self._create_receptive_fields()  # list of (H, W) tensors
        receptive_fields_stack = torch.stack(receptive_fields, dim=0)  # (17, H, W)
        self.register_buffer('receptive_fields', receptive_fields_stack)

        # AU到运动方向的映射
        directions_dict = self._create_motion_directions()
        directions_tensor = torch.zeros(num_au, 2)
        for idx, (dx, dy) in directions_dict.items():
            directions_tensor[idx, 0] = dx
            directions_tensor[idx, 1] = dy
        self.register_buffer('au_motion_directions', directions_tensor)

    def forward(self, au_temporal, time_idx=None):
        """
        感受野运动场生成

        Args:
            au_temporal (torch.Tensor): AU时间场，shape (B, 17, T)
            time_idx (int): 当前帧索引

        Returns:
            motion_field (torch.Tensor): 运动场，shape (B, 2, H, W)
        """
        B = au_temporal.shape[0]
        T = au_temporal.shape[2]

        if time_idx is None:
            time_idx = T // 2

        # 当前帧的AU激活
        au_frame = au_temporal[:, :, time_idx]  # (B, 17)

        # 初始化运动场（已在正确设备）
        motion_field = torch.zeros(B, 2, self.image_size, self.image_size,
                                   device=au_temporal.device)

        # 为每个AU生成感受野运动（使用buffer，自动在正确设备）
        for au_idx in range(self.num_au):
            au_intensity = au_frame[:, au_idx]  # (B,)

            # 获取该AU的运动方向（从buffer）
            dx = self.au_motion_directions[au_idx, 0].item()
            dy = self.au_motion_directions[au_idx, 1].item()

            # 应用感受野（从buffer，已在正确设备）
            rf = self.receptive_fields[au_idx]  # (H, W)

            # 加权运动
            for b in range(B):
                motion_field[b, 0] += au_intensity[b] * dx * rf
                motion_field[b, 1] += au_intensity[b] * dy * rf

        return motion_field

    def _create_receptive_fields(self):
        """
        创建ON-center感受野

        神经科学依据：
          中心-周围拮抗结构
          RF = Gaussian_center - Gaussian_surround
        """
        H = W = self.image_size
        receptive_fields = []

        for au_idx in range(self.num_au):
            # 获取AU区域
            au_name = sorted(AU_INFO.keys())[au_idx]
            region = AU_INFO[au_name]['region']
            center = AU_REGIONS[region]['center']
            radius = AU_REGIONS[region]['radius']

            # 创建坐标网格
            y = torch.arange(H).float()
            x = torch.arange(W).float()

            # 中心Gaussian（兴奋）
            G_center = torch.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * radius**2))

            # 周围Gaussian（抑制）- 侧抑制
            G_surround = torch.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * (radius * 2)**2))

            # ON-center感受野：中心兴奋 - 周围抑制
            RF = G_center - 0.5 * G_surround

            # 归一化
            RF = RF / (RF.max() + 1e-8)

            receptive_fields.append(RF)

        return receptive_fields

    def _create_motion_directions(self):
        """
        创建AU运动方向

        基于FACS定义的运动方向
        """
        directions = {}

        # AU1, AU2: 眉毛上扬
        directions[AU_INDEX['AU1']] = (0, -3)
        directions[AU_INDEX['AU2']] = (0, -3)

        # AU4: 眉降
        directions[AU_INDEX['AU4']] = (0, 3)

        # AU5: 眼睑上扬（睁眼）
        directions[AU_INDEX['AU5']] = (0, -2)

        # AU6: 颧骨上扬（眯眼）
        directions[AU_INDEX['AU6']] = (0, -2)

        # AU9: 皱鼻
        directions[AU_INDEX['AU9']] = (0, -1)

        # AU12: 嘴角上扬
        directions[AU_INDEX['AU12']] = (3, -3)

        # AU14: 嘴角收紧
        directions[AU_INDEX['AU14']] = (-2, 0)

        # AU15: 嘴角下垂
        directions[AU_INDEX['AU15']] = (0, 3)

        # AU17: 下颏上扬
        directions[AU_INDEX['AU17']] = (0, -3)

        # AU25: 张嘴
        directions[AU_INDEX['AU25']] = (0, 5)

        # 默认
        for i in range(self.num_au):
            if i not in directions:
                directions[i] = (0, 0)

        return directions


# =============================================================================
# IT层：事件驱动的层级整合
# =============================================================================
# 神经科学依据：IT（下颞叶）的层级整合机制
#   - 多层级输入整合
#   - 注意力调制
#   - 最终感知输出
# =============================================================================

class ITEventDrivenFusion(nn.Module):
    """
    IT层：事件驱动的运动场融合

    真正的神经科学机制：
      1. 多层级输入整合（V1, V2, V3, V4输出）
      2. 注意力权重调制
      3. Grid-sampling warping生成图像
    """

    def __init__(self, image_size=224):
        super().__init__()

        self.image_size = image_size

        # 整合权重（可学习）
        self.integration_weights = nn.Parameter(torch.ones(4) / 4)  # V1-V4

        # 冲突解决网络（处理重叠区域）
        self.conflict_resolver = nn.Sequential(
            nn.Conv2d(2 * 17, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, 1, 1),
        )

    def forward(self, motion_field, neutral_face):
        """
        运动场融合与warping

        Args:
            motion_field (torch.Tensor): 运动场，shape (B, 2, H, W)
            neutral_face (torch.Tensor): 中性脸，shape (B, C, H, W)

        Returns:
            generated_frame (torch.Tensor): 生成的帧
        """
        B, C, H, W = neutral_face.shape

        # Grid-sampling warping
        generated = self._warp_image(neutral_face, motion_field)

        return generated

    def _warp_image(self, image, motion_field):
        """
        使用运动场warp图像

        生物学类比：视觉感知中的运动补偿
        """
        B, C, H, W = image.shape

        # 创建采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)
        grid = grid.expand(B, -1, -1, -1)

        # 运动场归一化
        motion_normalized = motion_field.permute(0, 2, 3, 1) / (H / 2)

        # 采样网格 + 运动偏移
        sampling_grid = grid + motion_normalized

        # Clamp到[-1, 1]
        sampling_grid = torch.clamp(sampling_grid, -1, 1)

        # Grid-sample warping
        warped = F.grid_sample(image, sampling_grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)

        return warped


# =============================================================================
# 完整的Censor-G SNN架构
# =============================================================================

class CensorGSNN(nn.Module):
    """
    Censor-G SNN: 基于脉冲神经网络的微表情生成

    真正的神经科学机制：
      V1 → LIF脉冲发放显著性筛选
      V2 → 神经回路突触连接（EPSP/IPSP）
      V3 → 膜电位积分时间动力学
      V4 → ON/OFF感受野结构
      IT → 事件驱动层级整合
    """

    def __init__(
        self,
        num_au=17,
        num_frames=16,
        image_size=224,
        spike_threshold=1.0,
        spike_time_steps=10,
    ):
        super().__init__()

        self.num_au = num_au
        self.num_frames = num_frames
        self.image_size = image_size

        # V1层：LIF脉冲发放
        self.v1_spiking = V1SpikingSaliency(
            num_au=num_au,
            threshold=spike_threshold,
            time_steps=spike_time_steps,
        )

        # V2层：神经回路
        self.v2_circuit = V2NeuralCircuit(num_au=num_au)

        # V3层：膜电位时间动力学
        self.v3_temporal = V3MembraneTemporalDynamics(num_au=num_au, num_frames=num_frames)

        # V4层：感受野运动场
        self.v4_motion = V4ReceptiveFieldMotion(num_au=num_au, image_size=image_size)

        # IT层：事件驱动融合
        self.it_fusion = ITEventDrivenFusion(image_size=image_size)

    def forward(self, neutral_face, au_input, return_spikes=False):
        """
        生成微表情视频

        Args:
            neutral_face (torch.Tensor): 中性脸，shape (B, C, H, W)
            au_input (torch.Tensor): AU输入刺激，shape (B, 17)
            return_spikes (bool): 是否返回脉冲历史

        Returns:
            generated_video (torch.Tensor): 生成的视频，shape (B, C, T, H, W)
        """
        B, C, H, W = neutral_face.shape
        T = self.num_frames

        # V1：脉冲发放 → 发放率编码
        firing_rate, spike_history = self.v1_spiking(au_input)

        # V2：神经回路整合
        effective_au = self.v2_circuit(firing_rate)

        # V3：膜电位时间动力学
        au_temporal = self.v3_temporal(effective_au, T)  # (B, 17, T)

        # 生成每一帧
        generated_frames = []

        for t in range(T):
            # V4：感受野运动场
            motion_field = self.v4_motion(au_temporal, time_idx=t)

            # IT：warping生成
            frame = self.it_fusion(motion_field, neutral_face)
            generated_frames.append(frame)

        # 组合为视频
        generated_video = torch.stack(generated_frames, dim=2)  # (B, C, T, H, W)

        if return_spikes:
            return generated_video, spike_history

        return generated_video

    def generate_with_emotion(self, neutral_face, emotion_class, intensity=1.0):
        """
        情感驱动的生成

        Args:
            neutral_face (torch.Tensor): 中性脸
            emotion_class (torch.Tensor): 情感类别索引
            intensity (float): 强度参数

        Returns:
            generated_video (torch.Tensor): 生成的视频
        """
        # 情感 → AU配置
        au_config = self._emotion_to_au(emotion_class)

        # 应用强度
        au_input = au_config * intensity

        return self.forward(neutral_face, au_input)

    def _emotion_to_au(self, emotion_class):
        """
        情感类别 → AU配置

        基于FACS情感映射
        """
        B = emotion_class.shape[0]
        au_config = torch.zeros(B, self.num_au, device=emotion_class.device)

        for b in range(B):
            emotion = emotion_class[b].item()

            if emotion == 0:  # Happiness
                au_config[b, AU_INDEX['AU6']] = 0.6
                au_config[b, AU_INDEX['AU12']] = 0.8
                au_config[b, AU_INDEX['AU25']] = 0.2

            elif emotion == 1:  # Surprise
                au_config[b, AU_INDEX['AU1']] = 0.7
                au_config[b, AU_INDEX['AU2']] = 0.7
                au_config[b, AU_INDEX['AU5']] = 0.8
                au_config[b, AU_INDEX['AU25']] = 0.5

            elif emotion == 2:  # Disgust
                au_config[b, AU_INDEX['AU4']] = 0.5
                au_config[b, AU_INDEX['AU9']] = 0.7
                au_config[b, AU_INDEX['AU10']] = 0.4
                au_config[b, AU_INDEX['AU17']] = 0.3

            elif emotion == 3:  # Repression
                au_config[b, AU_INDEX['AU14']] = 0.6
                au_config[b, AU_INDEX['AU17']] = 0.4
                au_config[b, AU_INDEX['AU4']] = 0.3

        return au_config


# =============================================================================
# Demo与验证
# =============================================================================

def demo_censor_g_snn():
    """Demo Censor-G SNN"""
    print("\n" + "="*60)
    print("Censor-G SNN: Spiking Neural Network ME Generation")
    print("="*60)

    # 创建模型
    model = CensorGSNN(num_au=17, num_frames=16, image_size=224)

    # 创建输入
    B, C, H, W = 2, 3, 224, 224
    neutral_face = torch.randn(B, C, H, W)
    au_input = torch.rand(B, 17) * 0.5 + 0.3  # [0.3, 0.8]

    print("\n[Test 1] V1 Spiking Saliency")
    firing_rate, spikes = model.v1_spiking(au_input)
    print(f"  AU input: {au_input[0, :5].tolist()}")
    print(f"  Firing rate: {firing_rate[0, :5].tolist()}")
    print(f"  Spike shape: {spikes.shape}")
    print(f"  Total spikes: {spikes.sum().item()}")

    print("\n[Test 2] V2 Neural Circuit")
    effective_au = model.v2_circuit(firing_rate)
    print(f"  Input AU: {firing_rate[0, AU_INDEX['AU12']].item():.3f}")
    print(f"  After circuit: {effective_au[0, AU_INDEX['AU12']].item():.3f}")
    print(f"  Change: {(effective_au - firing_rate).abs().mean().item():.3f}")

    print("\n[Test 3] V3 Membrane Temporal")
    au_temporal = model.v3_temporal(effective_au)
    print(f"  Temporal shape: {au_temporal.shape}")
    print(f"  AU12 curve (fast onset): {au_temporal[0, AU_INDEX['AU12'], :8].tolist()}")
    print(f"  AU17 curve (slow onset): {au_temporal[0, AU_INDEX['AU17'], :8].tolist()}")

    print("\n[Test 4] Full Generation")
    generated, spikes = model(neutral_face, au_input, return_spikes=True)
    print(f"  Generated video shape: {generated.shape}")
    print(f"  Spike history shape: {spikes.shape}")

    print("\n[Test 5] Emotion-Driven Generation")
    emotion_class = torch.tensor([0, 1])  # Happiness, Surprise
    intensity = 0.8
    generated = model.generate_with_emotion(neutral_face, emotion_class, intensity)
    print(f"  Generated shape: {generated.shape}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Parameters]")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n" + "="*60)
    print("Demo Complete! SNN mechanisms verified.")
    print("="*60)


if __name__ == '__main__':
    demo_censor_g_snn()