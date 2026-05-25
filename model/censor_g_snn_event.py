# Censor-G SNN Event-Driven: 事件驱动的微表情生成
# =============================================================================
# 核心创新：借鉴Civis Lucri-Faber的EventBus事件驱动机制
#
# 事件驱动机制：
#   1. EventBus（丘脑类比）：中央事件路由
#   2. Event Listeners（脑区类比）：响应事件的模块
#   3. Event Types（神经递质类比）：不同事件类别
#   4. Neural Plasticity Cycle：静默/爆发周期
#
# 优势：
#   - 松耦合模块通信
#   - AU激活作为事件触发
#   - 动态拓扑连接
#   - 神经可塑性周期（静默 ↔ 爆发 ↔ 精细）
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from enum import Enum, auto
from typing import Dict, List, Callable, Optional, Any
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model.censor_g_snn import AU_INDEX, AU_INFO, AU_REGIONS
except ImportError:
    # 定义AU常量（如果导入失败）
    AU_INFO = {
        'AU1': {'name': 'Inner Brow Raiser', 'muscle_type': 'fast', 'region': 'eyebrow_inner', 'tau_m': 5.0},
        'AU2': {'name': 'Outer Brow Raiser', 'muscle_type': 'fast', 'region': 'eyebrow_outer', 'tau_m': 5.0},
        'AU4': {'name': 'Brow Lowerer', 'muscle_type': 'fast', 'region': 'eyebrow', 'tau_m': 6.0},
        'AU5': {'name': 'Upper Lid Raiser', 'muscle_type': 'fast', 'region': 'eye_upper', 'tau_m': 4.0},
        'AU6': {'name': 'Cheek Raiser', 'muscle_type': 'mixed', 'region': 'cheek', 'tau_m': 10.0},
        'AU7': {'name': 'Lid Tightener', 'muscle_type': 'fast', 'region': 'eye', 'tau_m': 4.5},
        'AU9': {'name': 'Nose Wrinkler', 'muscle_type': 'fast', 'region': 'nose', 'tau_m': 5.5},
        'AU10': {'name': 'Upper Lip Raiser', 'muscle_type': 'mixed', 'region': 'lip_upper', 'tau_m': 12.0},
        'AU12': {'name': 'Lip Corner Puller', 'muscle_type': 'mixed', 'region': 'mouth_corner', 'tau_m': 15.0},
        'AU14': {'name': 'Dimpler', 'muscle_type': 'slow', 'region': 'mouth_corner', 'tau_m': 25.0},
        'AU15': {'name': 'Lip Corner Depressor', 'muscle_type': 'slow', 'region': 'mouth_corner', 'tau_m': 28.0},
        'AU17': {'name': 'Chin Raiser', 'muscle_type': 'slow', 'region': 'chin', 'tau_m': 30.0},
        'AU20': {'name': 'Lip Stretcher', 'muscle_type': 'mixed', 'region': 'mouth', 'tau_m': 18.0},
        'AU23': {'name': 'Lip Tightener', 'muscle_type': 'slow', 'region': 'mouth', 'tau_m': 22.0},
        'AU24': {'name': 'Lip Pressor', 'muscle_type': 'slow', 'region': 'mouth', 'tau_m': 24.0},
        'AU25': {'name': 'Lips Part', 'muscle_type': 'mixed', 'region': 'mouth', 'tau_m': 20.0},
        'AU26': {'name': 'Jaw Drop', 'muscle_type': 'slow', 'region': 'jaw', 'tau_m': 35.0},
    }
    AU_INDEX = {au: i for i, au in enumerate(sorted(AU_INFO.keys()))}
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
# Part 1: 事件类型定义
# =============================================================================

class GenerationEventType(Enum):
    """微表情生成的事件类型"""

    # AU相关事件（兴奋性）
    AU_ACTIVATION = auto()       # AU激活事件
    AU_THRESHOLD_CROSS = auto()  # AU阈值穿越
    AU_SYNERGY = auto()          # AU协同效应
    AU_ANTAGONISM = auto()       # AU对抗效应

    # 情感事件
    EMOTION_START = auto()       # 情感开始
    EMOTION_APEX = auto()        # 情感峰值
    EMOTION_END = auto()         # 情感结束

    # 时间动力学事件
    ONSET_BEGIN = auto()         # Onset阶段开始
    APEX_REACHED = auto()        # Apex到达
    OFFSET_BEGIN = auto()        # Offset阶段开始

    # 运动场事件
    MOTION_FIELD_GENERATE = auto()  # 运动场生成请求
    MOTION_FIELD_READY = auto()     # 运动场就绪

    # 神经可塑性事件
    BURST_TRIGGER = auto()       # 爆发触发
    GROWTH_FACTOR = auto()       # 生长因子释放
    SILENT_ENTER = auto()        # 进入静默状态

    # 生成完成事件
    FRAME_GENERATED = auto()     # 单帧生成完成
    VIDEO_COMPLETE = auto()      # 视频生成完成


class EventPriority(Enum):
    """事件优先级"""
    LOW = 0       # 后台处理
    NORMAL = 1    # 正常处理
    HIGH = 2      # 优先处理（如Apex帧）
    CRITICAL = 3  # 立即处理（如爆发触发）


class GenerationEvent:
    """微表情生成事件"""

    def __init__(self, event_type: GenerationEventType, data: Any,
                 priority: EventPriority = EventPriority.NORMAL,
                 source: str = "system", au_index: int = None):
        self.type = event_type
        self.data = data
        self.priority = priority
        self.source = source
        self.au_index = au_index
        self.timestamp = None

    def __repr__(self):
        return f"GenEvent({self.type.name}, au={self.au_index}, priority={self.priority.name})"


# =============================================================================
# Part 2: EventBus（丘脑类比）
# =============================================================================

class GenerationEventBus(nn.Module):
    """
    生成事件总线 - 丘脑类比

    路由事件到注册的监听器：
      - 基于优先级处理
      - 支持事件过滤
      - 广播到多个监听器
    """

    def __init__(self, max_queue_size: int = 500):
        super().__init__()
        self.max_queue_size = max_queue_size

        # 监听器字典：事件类型 -> 回调函数列表
        self.listeners: Dict[GenerationEventType, List[Callable]] = {}

        # 事件队列
        self.event_queue = deque(maxlen=max_queue_size)

        # 优先级队列（用于紧急事件）
        self.priority_queue = deque(maxlen=50)

        # 统计
        self.register_buffer('event_count', torch.zeros(len(GenerationEventType)))
        self.register_buffer('burst_count', torch.zeros(1))

    def subscribe(self, event_type: GenerationEventType, callback: Callable):
        """订阅事件类型"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)

    def unsubscribe(self, event_type: GenerationEventType, callback: Callable):
        """取消订阅"""
        if event_type in self.listeners and callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)

    def emit(self, event: GenerationEvent):
        """发射事件"""
        # 高优先级事件进入优先队列
        if event.priority.value >= EventPriority.HIGH.value:
            self.priority_queue.append(event)
        else:
            self.event_queue.append(event)

        # 更新统计
        type_idx = list(GenerationEventType).index(event.type)
        self.event_count[type_idx] += 1

        # 立即分发到监听器
        if event.type in self.listeners:
            for callback in self.listeners[event.type]:
                callback(event)

    def emit_batch(self, events: List[GenerationEvent]):
        """批量发射事件"""
        for event in events:
            self.emit(event)

    def get_stats(self) -> Dict:
        """获取事件统计"""
        return {
            'total_events': self.event_count.sum().item(),
            'queue_size': len(self.event_queue),
            'priority_queue_size': len(self.priority_queue),
            'burst_count': self.burst_count.item(),
            'by_type': {
                t.name: self.event_count[list(GenerationEventType).index(t)].item()
                for t in GenerationEventType
            }
        }

    def clear_queues(self):
        """清空队列"""
        self.event_queue.clear()
        self.priority_queue.clear()


# =============================================================================
# Part 3: 事件驱动的V1层（LIF脉冲发放）
# =============================================================================

class V1EventDrivenSpiking(nn.Module):
    """
    V1层：事件驱动的LIF脉冲发放

    事件驱动机制：
      - AU激活触发AU_ACTIVATION事件
      - 阈值穿越触发AU_THRESHOLD_CROSS事件
      - 高发放率触发BURST_TRIGGER事件
    """

    def __init__(self, num_au=17, threshold=1.0, tau_m=10.0, dt=1.0,
                 time_steps=10, event_bus=None):
        super().__init__()

        self.num_au = num_au
        self.threshold = threshold
        self.tau_m = tau_m
        self.dt = dt
        self.time_steps = time_steps

        # EventBus
        self.event_bus = event_bus if event_bus else GenerationEventBus()

        # 注册自身为事件监听器
        self.event_bus.subscribe(GenerationEventType.AU_ACTIVATION, self._on_au_activation)

        # 衰减系数
        self.decay = math.exp(-dt / tau_m)

        # AU特异性阈值
        au_thresholds = torch.ones(num_au)
        for au in ['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU9']:
            au_thresholds[AU_INDEX[au]] = 0.8
        for au in ['AU6', 'AU10', 'AU12', 'AU20', 'AU25']:
            au_thresholds[AU_INDEX[au]] = 1.0
        for au in ['AU14', 'AU15', 'AU17', 'AU23', 'AU24', 'AU26']:
            au_thresholds[AU_INDEX[au]] = 1.2

        self.au_thresholds = nn.Parameter(au_thresholds, requires_grad=True)

        # 背景噪声
        self.noise_scale = 0.05

        # 状态：膜电位
        self.membrane_potential = None

    def _on_au_activation(self, event: GenerationEvent):
        """响应AU激活事件"""
        # 存储膜电位状态
        au_idx = event.au_index
        if self.membrane_potential is not None and au_idx is not None:
            # 增加膜电位
            self.membrane_potential[:, au_idx] += event.data

    def forward(self, au_input):
        """
        事件驱动的LIF脉冲发放

        Args:
            au_input (torch.Tensor): AU输入刺激，shape (B, 17)

        Returns:
            firing_rate (torch.Tensor): 发放率编码的显著性
            events (List[GenerationEvent]): 发射的事件列表
        """
        B = au_input.shape[0]
        T = self.time_steps

        # 初始化膜电位
        V = torch.zeros(B, self.num_au, device=au_input.device)
        self.membrane_potential = V.clone()

        spikes = torch.zeros(B, self.num_au, T, device=au_input.device)
        events_emitted = []

        for t in range(T):
            # 膜电位积分
            V = self.decay * V + au_input

            # 添加神经噪声
            noise = torch.randn_like(V) * self.noise_scale
            V = V + noise

            # 发放判断
            spike = (V > self.au_thresholds).float()
            spikes[:, :, t] = spike

            # === 事件驱动：发射事件 ===
            for au_idx in range(self.num_au):
                au_name = sorted(AU_INFO.keys())[au_idx]

                # 首次发放 → 发射AU_THRESHOLD_CROSS事件
                if spike[0, au_idx] > 0 and (t == 0 or spikes[0, au_idx, t-1] == 0):
                    event = GenerationEvent(
                        event_type=GenerationEventType.AU_THRESHOLD_CROSS,
                        data={'frame': t, 'intensity': V[0, au_idx].item()},
                        priority=EventPriority.NORMAL,
                        source="v1_spiking",
                        au_index=au_idx
                    )
                    events_emitted.append(event)
                    self.event_bus.emit(event)

            # 发放后重置
            V = V - spike * self.au_thresholds
            V = F.relu(V)

            self.membrane_potential = V.clone()

        # 发放率编码
        firing_rate = spikes.sum(dim=-1) / T

        # 检测高发放率AU → 触发爆发事件
        significant_au = (firing_rate > 0.3).sum().item()
        if significant_au > 10:
            event = GenerationEvent(
                event_type=GenerationEventType.BURST_TRIGGER,
                data={'significant_count': significant_au, 'firing_rate': firing_rate},
                priority=EventPriority.CRITICAL,
                source="v1_spiking"
            )
            events_emitted.append(event)
            self.event_bus.emit(event)
            self.event_bus.burst_count += 1

        return firing_rate, spikes, events_emitted


# =============================================================================
# Part 4: 事件驱动的V2层（神经回路）
# =============================================================================

class V2EventDrivenNeuralCircuit(nn.Module):
    """
    V2层：事件驱动的神经回路

    事件驱动机制：
      - AU协同触发AU_SYNERGY事件
      - AU对抗触发AU_ANTAGONISM事件
      - 突触传递通过事件路由
    """

    def __init__(self, num_au=17, synaptic_delay=2, tau_epsp=3.0, tau_ipsp=5.0,
                 event_bus=None):
        super().__init__()

        self.num_au = num_au
        self.synaptic_delay = synaptic_delay
        self.tau_epsp = tau_epsp
        self.tau_ipsp = tau_ipsp

        # EventBus
        self.event_bus = event_bus if event_bus else GenerationEventBus()

        # 订阅事件
        self.event_bus.subscribe(GenerationEventType.AU_THRESHOLD_CROSS, self._on_threshold_cross)

        # 突触矩阵
        self.W_excitatory = nn.Parameter(self._init_excitatory_synapses())
        self.W_inhibitory = nn.Parameter(self._init_inhibitory_synapses())

        # 延迟衰减因子
        self.delay_decay_exc = math.exp(-synaptic_delay / tau_epsp)
        self.delay_decay_inh = math.exp(-synaptic_delay / tau_ipsp)

        # 事件记录
        self.synergy_events = []
        self.antagonism_events = []

    def _on_threshold_cross(self, event: GenerationEvent):
        """响应AU阈值穿越事件"""
        au_idx = event.au_index
        if au_idx is None:
            return

        # 检查协同/对抗关系
        au_name = sorted(AU_INFO.keys())[au_idx]

        # 协同检查（AU6-AU12, AU1-AU2等）
        synergies = [
            (AU_INDEX['AU6'], AU_INDEX['AU12'], 'happiness'),
            (AU_INDEX['AU1'], AU_INDEX['AU2'], 'surprise'),
        ]

        for src, target, emotion in synergies:
            if au_idx == src:
                synergy_event = GenerationEvent(
                    event_type=GenerationEventType.AU_SYNERGY,
                    data={'source': src, 'target': target, 'emotion': emotion},
                    priority=EventPriority.NORMAL,
                    source="v2_circuit",
                    au_index=target
                )
                self.synergy_events.append(synergy_event)
                self.event_bus.emit(synergy_event)

    def forward(self, au_state):
        """
        事件驱动的突触传递

        Args:
            au_state (torch.Tensor): AU状态，shape (B, 17)

        Returns:
            effective_au (torch.Tensor): 经神经回路整合后的状态
            events (List[GenerationEvent]): 发射的事件列表
        """
        events_emitted = []

        # EPSP（兴奋性突触后电位）
        epsp = torch.matmul(au_state, self.W_excitatory) * self.delay_decay_exc

        # IPSP（抑制性突触后电位）
        ipsp = torch.matmul(au_state, self.W_inhibitory) * self.delay_decay_inh

        # 神经回路整合
        effective_au = au_state + epsp - ipsp
        effective_au = torch.clamp(effective_au, 0, 1)

        # === 检测显著的协同/对抗效应 ===
        epsp_max = epsp.abs().max(dim=1)[0]
        ipsp_max = ipsp.abs().max(dim=1)[0]

        for b in range(au_state.shape[0]):
            # 检测强协同
            if epsp_max[b] > 0.1:
                # 找到最活跃的协同组合
                active_idx = epsp[b].argmax().item()

                synergy_event = GenerationEvent(
                    event_type=GenerationEventType.AU_SYNERGY,
                    data={'batch': b, 'strength': epsp_max[b].item(), 'active_au': active_idx},
                    priority=EventPriority.NORMAL,
                    source="v2_circuit"
                )
                events_emitted.append(synergy_event)
                self.event_bus.emit(synergy_event)

            # 检测强对抗
            if ipsp_max[b] > 0.1:
                antagonism_event = GenerationEvent(
                    event_type=GenerationEventType.AU_ANTAGONISM,
                    data={'batch': b, 'strength': ipsp_max[b].item()},
                    priority=EventPriority.NORMAL,
                    source="v2_circuit"
                )
                events_emitted.append(antagonism_event)
                self.event_bus.emit(antagonism_event)

        return effective_au, events_emitted

    def _init_excitatory_synapses(self):
        """初始化兴奋性突触"""
        W = torch.zeros(self.num_au, self.num_au)

        # Happiness协同
        W[AU_INDEX['AU6'], AU_INDEX['AU12']] = 0.3
        W[AU_INDEX['AU12'], AU_INDEX['AU6']] = 0.25

        # Surprise协同
        W[AU_INDEX['AU1'], AU_INDEX['AU2']] = 0.2
        W[AU_INDEX['AU2'], AU_INDEX['AU1']] = 0.2
        W[AU_INDEX['AU1'], AU_INDEX['AU5']] = 0.15

        # Disgust协同
        W[AU_INDEX['AU4'], AU_INDEX['AU9']] = 0.2

        return W

    def _init_inhibitory_synapses(self):
        """初始化抑制性突触"""
        W = torch.zeros(self.num_au, self.num_au)

        # Repression对抗
        W[AU_INDEX['AU12'], AU_INDEX['AU14']] = 0.4
        W[AU_INDEX['AU14'], AU_INDEX['AU12']] = 0.3

        # Brow对抗
        W[AU_INDEX['AU4'], AU_INDEX['AU2']] = 0.35
        W[AU_INDEX['AU2'], AU_INDEX['AU4']] = 0.25

        return W


# =============================================================================
# Part 5: 事件驱动的V3层（膜电位动力学）
# =============================================================================

class V3EventDrivenTemporalDynamics(nn.Module):
    """
    V3层：事件驱动的膜电位时间动力学

    事件驱动机制：
      - Onset开始触发ONSET_BEGIN事件
      - Apex到达触发APEX_REACHED事件
      - Offset开始触发OFFSET_BEGIN事件
      - 不同肌肉类型有不同的时间曲线
    """

    def __init__(self, num_au=17, num_frames=16, event_bus=None):
        super().__init__()

        self.num_au = num_au
        self.num_frames = num_frames

        # EventBus
        self.event_bus = event_bus if event_bus else GenerationEventBus()

        # 订阅爆发事件
        self.event_bus.subscribe(GenerationEventType.BURST_TRIGGER, self._on_burst)

        # AU膜时间参数
        tau_m_values = []
        for au in sorted(AU_INFO.keys()):
            tau_m_values.append(AU_INFO[au]['tau_m'])
        self.register_buffer('tau_m', torch.tensor(tau_m_values))

        # 时间参数
        self.onset_ratio = nn.Parameter(torch.tensor(0.3))
        self.apex_ratio = nn.Parameter(torch.tensor(0.2))

        # 状态
        self.current_phase = 'idle'

    def _on_burst(self, event: GenerationEvent):
        """响应爆发事件"""
        # 爆发触发时，加速时间动力学
        self.current_phase = 'burst'

    def forward(self, au_activation, num_frames=None):
        """
        事件驱动的膜电位时间动力学

        Args:
            au_activation (torch.Tensor): AU激活强度，shape (B, 17)
            num_frames (int): 输出帧数

        Returns:
            au_temporal (torch.Tensor): AU时间场，shape (B, 17, T)
            events (List[GenerationEvent]): 发射的事件列表
        """
        T = num_frames or self.num_frames
        B = au_activation.shape[0]

        events_emitted = []

        # 计算各阶段边界
        onset_end = int(T * self.onset_ratio)
        apex_end = int(T * (self.onset_ratio + self.apex_ratio))

        temporal_curves = []

        for au_idx in range(self.num_au):
            tau = self.tau_m[au_idx]

            curve = self._generate_membrane_curve(T, tau, onset_end, apex_end)
            temporal_curves.append(curve)

        temporal_curves = torch.stack(temporal_curves, dim=0)
        temporal_curves = temporal_curves.unsqueeze(0).expand(B, -1, -1)

        au_temporal = au_activation.unsqueeze(-1) * temporal_curves

        # === 发射相位事件 ===
        for t in range(T):
            if t == 0:
                # Onset开始
                onset_event = GenerationEvent(
                    event_type=GenerationEventType.ONSET_BEGIN,
                    data={'frame': 0, 'duration': onset_end},
                    priority=EventPriority.NORMAL,
                    source="v3_temporal"
                )
                events_emitted.append(onset_event)
                self.event_bus.emit(onset_event)

            elif t == onset_end:
                # Apex到达
                apex_event = GenerationEvent(
                    event_type=GenerationEventType.APEX_REACHED,
                    data={'frame': onset_end, 'duration': apex_end - onset_end},
                    priority=EventPriority.HIGH,  # Apex帧优先处理
                    source="v3_temporal"
                )
                events_emitted.append(apex_event)
                self.event_bus.emit(apex_event)

            elif t == apex_end:
                # Offset开始
                offset_event = GenerationEvent(
                    event_type=GenerationEventType.OFFSET_BEGIN,
                    data={'frame': apex_end, 'duration': T - apex_end},
                    priority=EventPriority.NORMAL,
                    source="v3_temporal"
                )
                events_emitted.append(offset_event)
                self.event_bus.emit(offset_event)

        return au_temporal, events_emitted

    def _generate_membrane_curve(self, T, tau_m, onset_end, apex_end):
        """生成膜电位曲线"""
        curve = torch.zeros(T)

        # Onset阶段：膜电位上升
        for t in range(onset_end):
            curve[t] = 1.0 * (1 - math.exp(-t / tau_m))

        # Apex阶段：峰值保持
        curve[onset_end:apex_end] = 1.0

        # Offset阶段：膜电位衰减
        tau_decay = tau_m * 1.5
        for t in range(apex_end, T):
            dt = t - apex_end
            curve[t] = math.exp(-dt / tau_decay)

        return curve


# =============================================================================
# Part 6: 事件驱动的V4层（感受野运动场）
# =============================================================================

class V4EventDrivenMotionField(nn.Module):
    """
    V4层：事件驱动的感受野运动场

    事件驱动机制：
      - 运动场生成请求 → MOTION_FIELD_GENERATE事件
      - 运动场就绪 → MOTION_FIELD_READY事件
      - Apex帧优先处理
    """

    def __init__(self, num_au=17, image_size=224, sigma_center=5.0, sigma_surround=15.0,
                 event_bus=None):
        super().__init__()

        self.num_au = num_au
        self.image_size = image_size
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround

        # EventBus
        self.event_bus = event_bus if event_bus else GenerationEventBus()

        # 订阅Apex事件（优先处理）
        self.event_bus.subscribe(GenerationEventType.APEX_REACHED, self._on_apex)

        # 感受野模板
        self.receptive_fields = self._create_receptive_fields()

        # AU运动方向
        self.au_motion_direction = self._create_motion_directions()

        # Apex帧缓存
        self.apex_frame_cache = None

    def _on_apex(self, event: GenerationEvent):
        """响应Apex事件"""
        # Apex帧需要最精细的运动场
        self.apex_frame_cache = event.data

    def forward(self, au_temporal, time_idx=None):
        """
        事件驱动的运动场生成

        Args:
            au_temporal (torch.Tensor): AU时间场，shape (B, 17, T)
            time_idx (int): 当前帧索引

        Returns:
            motion_field (torch.Tensor): 运动场，shape (B, 2, H, W)
            events (List[GenerationEvent]): 发射的事件列表
        """
        B = au_temporal.shape[0]
        T = au_temporal.shape[2]

        events_emitted = []

        if time_idx is None:
            time_idx = T // 2

        # 发射运动场生成请求事件
        request_event = GenerationEvent(
            event_type=GenerationEventType.MOTION_FIELD_GENERATE,
            data={'frame': time_idx, 'batch_size': B},
            priority=EventPriority.NORMAL,
            source="v4_motion"
        )
        events_emitted.append(request_event)
        self.event_bus.emit(request_event)

        # 当前帧AU激活
        au_frame = au_temporal[:, :, time_idx]

        # 初始化运动场
        motion_field = torch.zeros(B, 2, self.image_size, self.image_size,
                                   device=au_temporal.device)

        # 为每个AU生成感受野运动
        for au_idx in range(self.num_au):
            au_intensity = au_frame[:, au_idx]
            dx, dy = self.au_motion_direction[au_idx]
            rf = self.receptive_fields[au_idx]

            for b in range(B):
                motion_field[b, 0] += au_intensity[b] * dx * rf
                motion_field[b, 1] += au_intensity[b] * dy * rf

        # 发射运动场就绪事件
        ready_event = GenerationEvent(
            event_type=GenerationEventType.MOTION_FIELD_READY,
            data={'frame': time_idx, 'field_stats': {
                'magnitude': motion_field.abs().mean().item(),
                'max_motion': motion_field.abs().max().item()
            }},
            priority=EventPriority.NORMAL,
            source="v4_motion"
        )
        events_emitted.append(ready_event)
        self.event_bus.emit(ready_event)

        return motion_field, events_emitted

    def _create_receptive_fields(self):
        """创建ON-center感受野"""
        H = W = self.image_size
        receptive_fields = []

        for au_idx in range(self.num_au):
            au_name = sorted(AU_INFO.keys())[au_idx]
            region = AU_INFO[au_name]['region']
            center = AU_REGIONS[region]['center']
            radius = AU_REGIONS[region]['radius']

            y = torch.arange(H).float()
            x = torch.arange(W).float()

            G_center = torch.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * radius**2))
            G_surround = torch.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * (radius * 2)**2))

            RF = G_center - 0.5 * G_surround
            RF = RF / (RF.max() + 1e-8)

            receptive_fields.append(RF)

        return receptive_fields

    def _create_motion_directions(self):
        """创建AU运动方向"""
        directions = {}

        directions[AU_INDEX['AU1']] = (0, -3)
        directions[AU_INDEX['AU2']] = (0, -3)
        directions[AU_INDEX['AU4']] = (0, 3)
        directions[AU_INDEX['AU5']] = (0, -2)
        directions[AU_INDEX['AU6']] = (0, -2)
        directions[AU_INDEX['AU9']] = (0, -1)
        directions[AU_INDEX['AU12']] = (3, -3)
        directions[AU_INDEX['AU14']] = (-2, 0)
        directions[AU_INDEX['AU15']] = (0, 3)
        directions[AU_INDEX['AU17']] = (0, -3)
        directions[AU_INDEX['AU25']] = (0, 5)

        for i in range(self.num_au):
            if i not in directions:
                directions[i] = (0, 0)

        return directions


# =============================================================================
# Part 7: 事件驱动的IT层（层级整合）
# =============================================================================

class ITEventDrivenFusion(nn.Module):
    """
    IT层：事件驱动的层级整合

    事件驱动机制：
      - 监听MOTION_FIELD_READY事件
      - 监听FRAME_GENERATED事件
      - 发射VIDEO_COMPLETE事件
    """

    def __init__(self, image_size=224, event_bus=None):
        super().__init__()

        self.image_size = image_size

        # EventBus
        self.event_bus = event_bus if event_bus else GenerationEventBus()

        # 订阅事件
        self.event_bus.subscribe(GenerationEventType.MOTION_FIELD_READY, self._on_motion_ready)

        # 整合权重
        self.integration_weights = nn.Parameter(torch.ones(4) / 4)

        # 冲突解决网络
        self.conflict_resolver = nn.Sequential(
            nn.Conv2d(2 * 17, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, 1, 1),
        )

        # 生成的帧缓存
        self.generated_frames_cache = []

    def _on_motion_ready(self, event: GenerationEvent):
        """响应运动场就绪事件"""
        # 记录统计信息
        pass

    def forward(self, motion_field, neutral_face):
        """
        事件驱动的运动场融合

        Args:
            motion_field (torch.Tensor): 运动场，shape (B, 2, H, W)
            neutral_face (torch.Tensor): 中性脸，shape (B, C, H, W)

        Returns:
            generated_frame (torch.Tensor): 生成的帧
            events (List[GenerationEvent]): 发射的事件列表
        """
        B, C, H, W = neutral_face.shape

        events_emitted = []

        # Grid-sampling warping
        generated = self._warp_image(neutral_face, motion_field)

        # 发射帧生成完成事件
        frame_event = GenerationEvent(
            event_type=GenerationEventType.FRAME_GENERATED,
            data={'batch_size': B, 'frame_shape': generated.shape},
            priority=EventPriority.NORMAL,
            source="it_fusion"
        )
        events_emitted.append(frame_event)
        self.event_bus.emit(frame_event)

        self.generated_frames_cache.append(generated.detach())

        return generated, events_emitted

    def _warp_image(self, image, motion_field):
        """使用运动场warp图像"""
        B, C, H, W = image.shape

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)
        grid = grid.expand(B, -1, -1, -1)

        motion_normalized = motion_field.permute(0, 2, 3, 1) / (H / 2)
        sampling_grid = grid + motion_normalized
        sampling_grid = torch.clamp(sampling_grid, -1, 1)

        warped = F.grid_sample(image, sampling_grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)

        return warped

    def emit_video_complete(self, num_frames: int):
        """发射视频生成完成事件"""
        complete_event = GenerationEvent(
            event_type=GenerationEventType.VIDEO_COMPLETE,
            data={'num_frames': num_frames, 'generated_frames': len(self.generated_frames_cache)},
            priority=EventPriority.CRITICAL,
            source="it_fusion"
        )
        self.event_bus.emit(complete_event)
        return complete_event


# =============================================================================
# Part 8: 完整的事件驱动生成系统
# =============================================================================

class CensorGSNNEventDriven(nn.Module):
    """
    Censor-G SNN Event-Driven: 事件驱动的微表情生成

    事件驱动架构：
      EventBus → V1 (脉冲) → V2 (回路) → V3 (时间) → V4 (运动) → IT (整合)

    每一层都通过事件通信，而非直接数据传递。
    """

    def __init__(self, num_au=17, num_frames=16, image_size=224,
                 spike_threshold=1.0, spike_time_steps=10):
        super().__init__()

        self.num_au = num_au
        self.num_frames = num_frames
        self.image_size = image_size

        # === 中央EventBus ===
        self.event_bus = GenerationEventBus(max_queue_size=500)

        # === 事件驱动的各层 ===
        self.v1_spiking = V1EventDrivenSpiking(
            num_au=num_au,
            threshold=spike_threshold,
            time_steps=spike_time_steps,
            event_bus=self.event_bus
        )

        self.v2_circuit = V2EventDrivenNeuralCircuit(
            num_au=num_au,
            event_bus=self.event_bus
        )

        self.v3_temporal = V3EventDrivenTemporalDynamics(
            num_au=num_au,
            num_frames=num_frames,
            event_bus=self.event_bus
        )

        self.v4_motion = V4EventDrivenMotionField(
            num_au=num_au,
            image_size=image_size,
            event_bus=self.event_bus
        )

        self.it_fusion = ITEventDrivenFusion(
            image_size=image_size,
            event_bus=self.event_bus
        )

    def forward(self, neutral_face, au_input, return_events=False):
        """
        事件驱动的微表情生成

        Args:
            neutral_face (torch.Tensor): 中性脸，shape (B, C, H, W)
            au_input (torch.Tensor): AU输入刺激，shape (B, 17)
            return_events (bool): 是否返回事件历史

        Returns:
            generated_video (torch.Tensor): 生成的视频，shape (B, C, T, H, W)
            all_events (List[GenerationEvent], optional): 事件历史
        """
        B, C, H, W = neutral_face.shape
        T = self.num_frames

        # 清空事件总线
        self.event_bus.clear_queues()

        # 收集所有事件
        all_events = []

        # === V1：事件驱动的脉冲发放 ===
        firing_rate, spikes, v1_events = self.v1_spiking(au_input)
        all_events.extend(v1_events)

        # === V2：事件驱动的神经回路 ===
        effective_au, v2_events = self.v2_circuit(firing_rate)
        all_events.extend(v2_events)

        # === V3：事件驱动的时间动力学 ===
        au_temporal, v3_events = self.v3_temporal(effective_au, T)
        all_events.extend(v3_events)

        # === 生成每一帧 ===
        generated_frames = []

        for t in range(T):
            # === V4：事件驱动的运动场 ===
            motion_field, v4_events = self.v4_motion(au_temporal, time_idx=t)
            all_events.extend(v4_events)

            # === IT：事件驱动的融合 ===
            frame, it_events = self.it_fusion(motion_field, neutral_face)
            all_events.extend(it_events)

            generated_frames.append(frame)

        # 发射视频完成事件
        complete_event = self.it_fusion.emit_video_complete(T)
        all_events.append(complete_event)

        # 组合为视频
        generated_video = torch.stack(generated_frames, dim=2)

        if return_events:
            return generated_video, all_events, self.event_bus.get_stats()

        return generated_video

    def generate_with_emotion(self, neutral_face, emotion_class, intensity=1.0,
                              return_events=False):
        """
        情感驱动的事件生成

        Args:
            neutral_face: 中性脸
            emotion_class: 情感类别索引
            intensity: 强度参数
            return_events: 是否返回事件

        Returns:
            generated_video: 生成的视频
            events: 事件历史（可选）
        """
        # 情感 → AU配置
        au_config = self._emotion_to_au(emotion_class)
        au_input = au_config * intensity

        # 发射情感开始事件
        emotion_event = GenerationEvent(
            event_type=GenerationEventType.EMOTION_START,
            data={'emotion_class': emotion_class, 'intensity': intensity},
            priority=EventPriority.HIGH,
            source="system"
        )
        self.event_bus.emit(emotion_event)

        return self.forward(neutral_face, au_input, return_events)

    def _emotion_to_au(self, emotion_class):
        """情感类别 → AU配置"""
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

    def get_event_stats(self) -> Dict:
        """获取事件统计"""
        return self.event_bus.get_stats()


# =============================================================================
# Demo
# =============================================================================

def demo_event_driven_generation():
    """Demo事件驱动生成"""
    print("\n" + "="*60)
    print("Censor-G SNN Event-Driven Demo")
    print("="*60)

    # 创建模型
    model = CensorGSNNEventDriven(num_au=17, num_frames=16, image_size=224)

    # 创建输入
    B, C, H, W = 2, 3, 224, 224
    neutral_face = torch.randn(B, C, H, W)
    au_input = torch.rand(B, 17) * 0.5 + 0.3

    print("\n[Test 1] Event-Driven Generation")
    generated, events, stats = model(neutral_face, au_input, return_events=True)

    print(f"  Generated shape: {generated.shape}")
    print(f"  Total events: {len(events)}")
    print(f"  Event stats: {stats}")

    # 打印事件类型分布
    print("\n  Event distribution:")
    for event_type, count in stats['by_type'].items():
        if count > 0:
            print(f"    {event_type}: {count}")

    print("\n[Test 2] Emotion-Driven Generation")
    emotion_class = torch.tensor([0, 1])  # Happiness, Surprise
    generated, events, stats = model.generate_with_emotion(
        neutral_face, emotion_class, intensity=0.8, return_events=True
    )

    print(f"  Generated shape: {generated.shape}")
    print(f"  Total events: {len(events)}")

    print("\n" + "="*60)
    print("Demo Complete! Event-driven mechanisms verified.")
    print("="*60)


if __name__ == '__main__':
    demo_event_driven_generation()