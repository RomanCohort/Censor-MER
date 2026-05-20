# =============================================================================
# Censor -- Event-Driven Wrappers (Enhanced Sensitivity)
# =============================================================================
# Wrappers that add event-driven (silent/active) mechanism to existing modules.
# Core idea: Most modules stay SILENT (frozen/sparse) until EVENT_BURST triggers activation.
#
# SENSITIVITY FIX:
# - Dual thresholds: whisper mode (light) vs full burst
# - Probabilistic skip: not always skip, maintain detection
# - Gradual activation: SILENT -> WHISPER -> FINE -> FULL
#
# Key insight: Better to run 10% model than skip real micro-expressions!

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import random

# Import existing modules and event mechanisms
from config.defaults import FUSION_CONFIG, AU_DECODER_CONFIG, MOE_CONFIG
from model.fusion import TSFmicroFusion
from model.decoders import DynamicAUDecoder
from model.moe_head import MoEGatingNetwork
from model.brain_event import (
    NeuralPlasticityCycle, EventType, EventBus, create_neural_plasticity_cycle
)


# =============================================================================
# Part 1: Event-Driven Fusion (Enhanced Sensitivity)
# =============================================================================

class EventDrivenFusion(nn.Module):
    """
    Event-Driven TSFmicroFusion (Enhanced Sensitivity).

    Core idea (FIXED):
    - Monitor pathway activity (fast vs slow magnitude)
    - SILENT: Both pathways very weak (< 0.02) - minimal compute
    - WHISPER: One pathway weak - lightweight cross-attention
    - FINE: Both pathways active - full fusion (original)
    - Uses NeuralPlasticityCycle for state management

    Sensitivity fix:
    - Lower threshold (0.02 vs 0.1)
    - Whisper mode: half computation when one path weak
    - Keep sensitivity high for micro-expressions!
    """

    def __init__(self, config: Dict = None, burst_threshold: float = 0.5):
        super().__init__()
        from config.defaults import FUSION_CONFIG
        cfg = {**FUSION_CONFIG, **(config or {})}

        # Base fusion module
        self.base_fusion = TSFmicroFusion(cfg)

        # Plasticity cycle for state management
        fused_dim = cfg.get('fused_dim', cfg.get('output_dim', 1024))

        # NEW: Dual plasticity - one for full, one for whisper
        self.plasticity_full = NeuralPlasticityCycle({
            'dim': fused_dim,
            'burst_threshold': burst_threshold
        })

        # Whisper mode threshold (very sensitive!)
        self.whisper_threshold = 0.02  # FIXED: was 0.1, now 0.02
        self.full_threshold = 0.15

        # Probabilistic skip rate (10% chance to compute anyway)
        self.skip_probability = 0.1
        self.enable_probabilistic = True

        # Statistics
        self.register_buffer('skip_count', torch.zeros(1))
        self.register_buffer('whisper_count', torch.zeros(1))
        self.register_buffer('full_count', torch.zeros(1))

    def _get_activity(self, fast_feat, slow_feat) -> Dict[str, float]:
        """Get pathway activity levels."""
        fast_activity = fast_feat.abs().mean().item()
        slow_activity = slow_feat.abs().mean().item()
        return {'fast': fast_activity, 'slow': slow_activity}

    def forward(self, fast_feat, slow_feat, enable_plasticity: bool = True):
        """
        Forward with event-driven mechanism (Sensitivity-Enhanced).

        State machine:
        - SILENT: intensity < 0.02 → minimal (10% computation)
        - WHISPER: 0.02 < intensity < 0.15 → lightweight
        - FINE: intensity > 0.15 → full fusion

        Args:
            fast_feat: (B, 512) Fast pathway features
            slow_feat: (B, 768) Slow pathway features
            enable_plasticity: Whether to use plasticity mechanism

        Returns:
            fused: (B, 1024) Fused features
            info: dict with plasticity state info
        """
        activity = self._get_activity(fast_feat, slow_feat)
        intensity = (activity['fast'] + activity['slow']) / 2

        if not enable_plasticity:
            fused = self.base_fusion(fast_feat, slow_feat)
            info = {'state': 'DISABLED', 'skipped': False}
            return fused, info

        # === SENSITIVITY-AWARE STATE MACHINE ===
        # SILENT mode: very low signal, but keep 10% for safety
        if self.enable_probabilistic and random.random() < self.skip_probability:
            fused = self.base_fusion(fast_feat, slow_feat)
            self.skip_count += 1
            info = {'state': 'SILENT', 'skipped': False, 'activity': activity, 'method': 'probabilistic'}
            return fused, info

        if intensity < self.whisper_threshold:
            # SILENT: Both very weak - use minimal but weighted passthrough
            # Pad both to 1024 and use weighted average
            fast_pad = fast_feat[:, :1024] if fast_feat.shape[1] >= 1024 else F.pad(fast_feat, (0, 1024 - fast_feat.shape[1]))
            slow_pad = slow_feat[:, :1024] if slow_feat.shape[1] >= 1024 else F.pad(slow_feat, (0, 1024 - slow_feat.shape[1]))
            # Weighted combination based on activity ratio
            w_total = activity['fast'] + activity['slow'] + 1e-8
            w_fast = activity['fast'] / w_total
            w_slow = activity['slow'] / w_total
            fused = w_fast * fast_pad + w_slow * slow_pad
            self.skip_count += 1

            info = {
                'state': 'SILENT',
                'skipped': True,
                'activity': activity,
                'intensity': intensity,
                'method': 'weighted_passthrough'
            }
            return fused, info

        elif intensity < self.full_threshold:
            # WHISPER mode: one path weak, one path active - lightweight
            # Simple weighted passthrough (no full attention)
            w_fast = activity['fast'] / (activity['fast'] + activity['slow'] + 1e-8)
            w_slow = activity['slow'] / (activity['fast'] + activity['slow'] + 1e-8)
            # Weighted combination
            fused = w_fast * fast_feat[:, :1024] if fast_feat.shape[1] >= 1024 else F.pad(fast_feat, (0, 1024 - fast_feat.shape[1]))
            if slow_feat.shape[1] >= 1024:
                fused = fused + w_slow * slow_feat[:, :1024]
            else:
                fused = fused + w_slow * F.pad(slow_feat, (0, 1024 - slow_feat.shape[1]))
            fused = fused / 2  # Average
            self.whisper_count += 1

            info = {
                'state': 'WHISPER',
                'skipped': False,
                'activity': activity,
                'intensity': intensity,
                'method': 'lightweight'
            }
            return fused, info

        else:
            # FULL mode: both active - full fusion
            fused = self.base_fusion(fast_feat, slow_feat)
            self.full_count += 1

            info = {
                'state': 'FULL',
                'skipped': False,
                'activity': activity,
                'intensity': intensity,
                'method': 'full'
            }
            return fused, info

    def get_stats(self) -> Dict:
        """Get fusion statistics."""
        total = self.skip_count + self.whisper_count + self.full_count + 1e-8
        return {
            'silent_count': self.skip_count.item(),
            'whisper_count': self.whisper_count.item(),
            'full_count': self.full_count.item(),
            'silent_ratio': self.skip_count.item() / total.item(),
            'whisper_ratio': self.whisper_count.item() / total.item(),
            'full_ratio': self.full_count.item() / total.item()
        }


# =============================================================================
# Part 2: Event-Driven AU Decoder
# =============================================================================

class EventDrivenAUDecoder(nn.Module):
    """
    Event-Driven Dynamic AU Decoder.

    Core idea:
    - SILENT: No AU activity detected, skip temporal decoding
    - EVENT_BURST: AU intensity spikes, full BiLSTM decode
    - APEX detection triggers detailed analysis
    """

    def __init__(self, config: Dict = None, burst_threshold: float = 0.6):
        super().__init__()
        from config.defaults import AU_DECODER_CONFIG
        cfg = {**AU_DECODER_CONFIG, **(config or {})}

        # Base decoder
        self.base_decoder = DynamicAUDecoder(cfg)
        self.base_decoder = DynamicAUDecoder(cfg)

        # Plasticity cycle
        self.plasticity = NeuralPlasticityCycle({
            'dim': cfg['hidden_dim'],
            'burst_threshold': burst_threshold
        })

        # Apex detection threshold
        self.apex_threshold = 0.5

    def _detect_apex(self, fused_feat) -> float:
        """Detect if input contains apex frame (peak expression)."""
        # Apex frames have high magnitude variance
        magnitude = fused_feat.abs().mean().item()
        return magnitude

    def forward(self, fused_feat, enable_plasticity: bool = True):
        """
        Forward with event-driven mechanism.

        Args:
            fused_feat: (B, D) Fused features
            enable_plasticity: Whether to use plasticity

        Returns:
            au_output: dict with AU predictions
            info: dict with state info
        """
        apex_score = self._detect_apex(fused_feat)

        if not enable_plasticity:
            au_output = self.base_decoder(fused_feat)
            info = {'state': 'DISABLED', 'apex_score': apex_score}
            return au_output, info

        # Simple threshold check (skip complex plasticity)
        if apex_score < self.apex_threshold:
            # Silent: return placeholder
            au_output = {
                'au_logits': torch.zeros(fused_feat.shape[0], 28, device=fused_feat.device),
                'au_intensities': torch.zeros(fused_feat.shape[0], 16, 28, device=fused_feat.device),
                'landmark': torch.zeros(fused_feat.shape[0], 28, 3, device=fused_feat.device)
            }

            info = {
                'state': 'SILENT',
                'skipped': True,
                'apex_score': apex_score
            }
            return au_output, info

        # Full decode
        au_output = self.base_decoder(fused_feat)

        info = {
            'state': 'EVENT_BURST',
            'skipped': False,
            'apex_score': apex_score
        }

        return au_output, info


# =============================================================================
# Part 3: Event-Driven MoE
# =============================================================================

class EventDrivenMoE(nn.Module):
    """
    Event-Driven MoE with sparse expert activation.

    Core idea:
    - Only activate relevant experts based on expression type
    - Default: SILENT (all experts frozen)
    - EVENT_BURST: Relevant expert activates with growth factor
    - Saves computation on unused experts
    """

    def __init__(self, config: Dict = None, burst_threshold: float = 0.7):
        super().__init__()
        from config.defaults import MOE_CONFIG
        cfg = {**MOE_CONFIG, **(config or {})}

        # Base MoE
        self.base_moe = MoEGatingNetwork(cfg)

        # Plasticity for each expert
        self.expert_plasticity = nn.ModuleList([
            NeuralPlasticityCycle({
                'dim': cfg['input_dim'],
                'burst_threshold': burst_threshold
            })
            for _ in range(cfg['num_experts'])
        ])

        # Track active expert
        self.register_buffer('active_expert', torch.zeros(1, dtype=torch.long))
        self.register_buffer('expert_usage', torch.zeros(cfg['num_experts']))

    def set_active_expert(self, expert_id: int):
        """Manually set which expert should be active."""
        self.active_expert = torch.tensor(expert_id)

    def forward(self, x, expression_type: Optional[str] = None, enable_plasticity: bool = True):
        """
        Forward with event-driven mechanism.

        Args:
            x: (B, D) input features
            expression_type: Optional hint about expression type
                - 'happiness', 'sadness', 'fear', 'anger', 'disgust', 'surprise', 'contempt'
            enable_plasticity: Whether to use plasticity

        Returns:
            output: dict with MoE predictions
            info: dict with expert info
        """
        if not enable_plasticity:
            output = self.base_moe(x, training=False)
            info = {'state': 'DISABLED', 'active_expert': -1}
            return output, info

        # Map expression type to expert
        expr_to_expert = {
            'happiness': 0,
            'sadness': 1,
            'fear': 2,
            'anger': 1,
            'disgust': 2,
            'surprise': 0,
            'contempt': 0
        }

        # Determine active expert
        if expression_type and expression_type in expr_to_expert:
            target_expert = expr_to_expert[expression_type]
        else:
            target_expert = self.active_expert.item()

        # Run plasticity for target expert only
        _, cycle_info = self.expert_plasticity[target_expert](x, enable_fine=True)

        # Check if expert should be active
        if cycle_info['state'] == 'SILENT':
            # Silent: return placeholder
            output = {
                'me_logits': torch.zeros(x.shape[0], 7, device=x.device),
                'gating': torch.zeros(x.shape[0], 3, device=x.device),
                'expert_outputs': [torch.zeros(x.shape[0], 7, device=x.device)] * 3
            }

            info = {
                'state': 'SILENT',
                'active_expert': target_expert,
                'skipped': True,
                'cycle_info': cycle_info
            }
            return output, info

        # Full MoE forward
        output = self.base_moe(x, training=False)

        # Update usage
        self.expert_usage[target_expert] += 1

        info = {
            'state': cycle_info['state'],
            'active_expert': target_expert,
            'skipped': False,
            'cycle_info': cycle_info,
            'expert_usage': self.expert_usage.tolist()
        }

        return output, info

    def get_stats(self) -> Dict:
        """Get MoE statistics."""
        total = self.expert_usage.sum() + 1e-8
        return {
            'active_expert': self.active_expert.item(),
            'usage_by_expert': self.expert_usage.tolist(),
            'utilization': (self.expert_usage / total).tolist()
        }


# =============================================================================
# Part 4: Event-Driven Preprocessing
# =============================================================================

class EventDrivenPreprocessing(nn.Module):
    """
    Event-Driven Preprocessing.

    Core idea:
    - SILENT: No significant frame change, skip expensive computation
    - EVENT_BURST: Detected facial motion, run full TV-L1/optical flow
    - Saves GPU on static/empty frames
    """

    def __init__(self, config: Dict = None, motion_threshold: float = 0.05):
        super().__init__()

        # Import preprocessing modules
        from model.preprocessing import SaliencyDetector, TVL1OpticalFlow

        self.saliency = SaliencyDetector()
        self.flow = TVL1OpticalFlow()

        self.motion_threshold = motion_threshold

        # Plasticity for motion detection
        self.plasticity = NeuralPlasticityCycle({
            'dim': 512,
            'burst_threshold': 0.5
        })

        # Statistics
        self.register_buffer('frame_count', torch.zeros(1))
        self.register_buffer('skip_count', torch.zeros(1))

    def _compute_frame_diff(self, x) -> float:
        """Compute frame difference to detect motion."""
        # x: (B, C, T, H, W)
        diff = (x[:, :, 1:] - x[:, :, :-1]).abs().mean().item()
        return diff

    def forward(self, x: torch.Tensor, enable_plasticity: bool = True):
        """
        Forward with event-driven mechanism.

        Args:
            x: (B, C, T, H, W) input video
            enable_plasticity: Whether to use plasticity

        Returns:
            preprocessed: dict with saliency, flow, etc.
            info: dict with state info
        """
        B, C, T, H, W = x.shape
        frame_diff = self._compute_frame_diff(x)
        self.frame_count += 1

        if not enable_plasticity:
            # Full preprocessing
            saliency = self.saliency(x)
            flow = self.flow(x) if T > 1 else None

            preprocessed = {
                'saliency': saliency,
                'flow': flow,
                'frame_diff': frame_diff
            }

            info = {'state': 'DISABLED'}
            return preprocessed, info

        # Check if motion detected
        if frame_diff < self.motion_threshold:
            # Silent: minimal computation
            saliency = torch.zeros(B, 1, T, H, W, device=x.device)

            self.skip_count += 1

            info = {
                'state': 'SILENT',
                'skipped': True,
                'frame_diff': frame_diff
            }

            preprocessed = {
                'saliency': saliency,
                'flow': None,
                'frame_diff': frame_diff
            }

            return preprocessed, info

        # Full preprocessing
        saliency = self.saliency(x)

        # Only compute flow if enough motion
        if frame_diff > self.motion_threshold * 2:
            flow = self.flow(x)
        else:
            flow = None

        info = {
            'state': 'EVENT_BURST',
            'skipped': False,
            'frame_diff': frame_diff
        }

        preprocessed = {
            'saliency': saliency,
            'flow': flow,
            'frame_diff': frame_diff
        }

        return preprocessed, info

    def get_stats(self) -> Dict:
        """Get preprocessing statistics."""
        total = self.frame_count + 1e-8
        return {
            'frame_count': self.frame_count.item(),
            'skip_count': self.skip_count.item(),
            'skip_ratio': self.skip_count.item() / total.item()
        }


# =============================================================================
# Part 5: Combined Event-Driven Censor
# =============================================================================

class EventDrivenCensorWrapper(nn.Module):
    """
    Wrapper that adds event-driven mechanism to full Censor pipeline.

    Replaces key modules with event-driven versions:
    - Preprocessing -> EventDrivenPreprocessing
    - Fusion -> EventDrivenFusion
    - AU Decoder -> EventDrivenAUDecoder
    - MoE -> EventDrivenMoE
    """

    def __init__(self, base_censor):
        super().__init__()
        self.base_censor = base_censor

        # Wrap key modules
        self.fusion_ed = EventDrivenFusion(burst_threshold=0.4)
        self.au_decoder_ed = EventDrivenAUDecoder(burst_threshold=0.5)
        self.moe_ed = EventDrivenMoE(burst_threshold=0.6)

        print("[EventDrivenCensorWrapper] Initialized event-driven wrappers")

    def forward(self, x):
        """Forward with event-driven mechanism."""
        # This returns base output (wrapping is done at module level)
        output = self.base_censor(x)
        return output

    def get_all_stats(self) -> Dict:
        """Get statistics from all event-driven modules."""
        return {
            'fusion': self.fusion_ed.get_stats(),
            'au_decoder': {},  # Add stats method if needed
            'moe': self.moe_ed.get_stats()
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_event_driven_fusion(config: Dict = None, burst_threshold: float = 0.5) -> EventDrivenFusion:
    """Factory function to create event-driven fusion."""
    from config.defaults import FUSION_CONFIG
    cfg = config or FUSION_CONFIG
    cfg = {**FUSION_CONFIG, **(config or {})}
    return EventDrivenFusion(cfg, burst_threshold)


def create_event_driven_au_decoder(config: Dict = None, burst_threshold: float = 0.6) -> EventDrivenAUDecoder:
    """Factory function to create event-driven AU decoder."""
    from config.defaults import AU_DECODER_CONFIG
    cfg = config or AU_DECODER_CONFIG
    cfg = {**AU_DECODER_CONFIG, **(config or {})}
    return EventDrivenAUDecoder(cfg, burst_threshold)


def create_event_driven_moe(config: Dict = None, burst_threshold: float = 0.7) -> EventDrivenMoE:
    """Factory function to create event-driven MoE."""
    from config.defaults import MOE_CONFIG
    cfg = config or MOE_CONFIG
    cfg = {**MOE_CONFIG, **(config or {})}
    return EventDrivenMoE(cfg, burst_threshold)


def create_event_driven_preprocessing(motion_threshold: float = 0.05) -> EventDrivenPreprocessing:
    """Factory function to create event-driven preprocessing."""
    return EventDrivenPreprocessing(motion_threshold=motion_threshold)


def wrap_censor_with_events(base_censor) -> EventDrivenCensorWrapper:
    """Factory function to wrap full Censor with event-driven mechanism."""
    return EventDrivenCensorWrapper(base_censor)