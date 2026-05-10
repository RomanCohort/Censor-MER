# =============================================================================
# Censor -- Biomimetic Enhancement: DTN + Meta-Plasticity
# =============================================================================
# Integrated modules from the biomimetic concepts:
#   1. Dynamic Topology Networks (DTN) -> Enhanced FFA with tension gating
#   2. Meta-Plasticity Memory -> Long-term memory with LoRA consolidation
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datetime import datetime


# =============================================================================
# Part 1: DTN-Enhanced FFA (Tension-Gated Feature Fusion)
# =============================================================================

class TensionComputation(nn.Module):
    """
    Computes tension field from feature differences.
    Simulates cytoskeleton tension in biological cells.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Learnable tension projection
        self.tension_proj = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: (B, D) feature vector
        返回: (B, 1) tension score
        """
        # Center features around mean (simulate deviation from baseline)
        x_centered = x - x.mean(dim=0, keepdim=True)
        tension = self.tension_proj(x_centered ** 2)  # squared deviation
        return tension


class MechanicalGate(nn.Module):
    """
    Threshold gating inspired by mechanosensitive channels.
    When tension exceeds threshold, gate opens (feature passes).
    """

    def __init__(self, threshold=0.5, gain=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.gain = nn.Parameter(torch.tensor(gain))

    def forward(self, tension):
        """
        tension: (B, ...) tension values
        返回: (B, ...) gate values in (0, 1)
        """
        return torch.sigmoid(self.gain * tension - self.threshold)


class DTNEnhancedFFA(nn.Module):
    """
    DTN-enhanced Feature Fusion Attention.

    Integrates:
    1. Original SE-style gating (from FFA)
    2. New: Tension-based mechanical gating from DTN

    The tension is computed from feature deviation, and the mechanical
    gate acts as a "second opinion" - only high-tension features get
    amplified.
    """

    def __init__(self, config=None):
        super().__init__()
        # Import from original FFA
        from config.defaults import FFA_CONFIG
        cfg = config or FFA_CONFIG

        fast_dim = cfg['fast_dim']
        slow_dim = cfg['slow_dim']
        joint_dim = fast_dim + slow_dim
        reduction = cfg['reduction_ratio']
        squeeze_dim = max(joint_dim // reduction, 16)

        # === Original FFA components ===
        self.squeeze = nn.Linear(joint_dim, squeeze_dim)
        self.relu = nn.ReLU(inplace=True)
        self.excitation = nn.Linear(squeeze_dim, joint_dim)
        self.sigmoid = nn.Sigmoid()

        # === New: DTN Tension components ===
        self.tension_fast = TensionComputation(fast_dim)
        self.tension_slow = TensionComputation(slow_dim)
        self.mech_gate_fast = MechanicalGate(threshold=0.3)
        self.mech_gate_slow = MechanicalGate(threshold=0.3)

        # === Learnable fusion weight for SE + Mechanical ===
        self.dtn_weight = nn.Parameter(torch.tensor(0.5))

        # Initialization
        nn.init.kaiming_normal_(self.squeeze.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.squeeze.bias, 0)
        nn.init.xavier_uniform_(self.excitation.weight)
        nn.init.constant_(self.excitation.bias, 0)

    def forward(self, fast_feat, slow_feat):
        """
        Args:
            fast_feat (torch.Tensor): Fast pathway output, shape (B, 512)
            slow_feat (torch.Tensor): Slow pathway output, shape (B, 768)
        Returns:
            fast_gated (torch.Tensor): DTN-enhanced fast features
            slow_gated (torch.Tensor): DTN-enhanced slow features
        """
        print(f"[DTN-FFA] Inputs: fast={fast_feat.shape}, slow={slow_feat.shape}")

        B = fast_feat.shape[0]

        # === 1. Original SE Gating ===
        joint = torch.cat([fast_feat, slow_feat], dim=1)  # (B, 1280)
        s = self.squeeze(joint)
        s = self.relu(s)
        se_gate = self.excitation(s)  # (B, 1280)
        se_gate = self.sigmoid(se_gate)

        fast_dim = fast_feat.shape[1]
        se_gate_fast = se_gate[:, :fast_dim]
        se_gate_slow = se_gate[:, fast_dim:]

        # === 2. New: DTN Tension Gating ===
        # Compute tension from feature deviation
        tension_fast = self.tension_fast(fast_feat)  # (B, 1)
        tension_slow = self.tension_slow(slow_feat)  # (B, 1)

        # Mechanical gate (threshold-based)
        mech_gate_fast = self.mech_gate_fast(tension_fast)  # (B, 1)
        mech_gate_slow = self.mech_gate_slow(tension_slow)  # (B, 1)

        # Expand to feature dimension
        mech_gate_fast = mech_gate_fast.expand(-1, fast_dim)
        mech_gate_slow = mech_gate_slow.expand(-1, slow_feat.shape[1])

        print(f"[DTN-FFA] Tension: fast={tension_fast.mean().item():.4f}, slow={tension_slow.mean().item():.4f}")
        print(f"[DTN-FFA] Mech gate: fast={mech_gate_fast.mean().item():.4f}, slow={mech_gate_slow.mean().item():.4f}")

        # === 3. Fusion: SE + Mechanical ===
        # Weighted combination: dtn_weight * mechanical + (1-dtn_weight) * se
        dtn_w = self.dtn_weight.sigmoid()  # (1,)

        fast_gated = (
            (1 - dtn_w) * (fast_feat * se_gate_fast) +
            dtn_w * (fast_feat * mech_gate_fast)
        )
        slow_gated = (
            (1 - dtn_w) * (slow_feat * se_gate_slow) +
            dtn_w * (slow_feat * mech_gate_slow)
        )

        print(f"[DTN-FFA] Outputs: fast_gated={fast_gated.shape}, slow_gated={slow_gated.shape}")
        return fast_gated, slow_gated


# =============================================================================
# Part 2: Meta-Plasticity Memory (Long-term Weight Consolidation)
# =============================================================================

class EmotionStimulusDetector(nn.Module):
    """
    Detects "emotion stimulus" from context features.
    Triggers methylation when intensity exceeds threshold.

    High feature magnitude = high emotion stimulus (energetic expression).
    """

    def __init__(self, input_dim=1024):
        super().__init__()
        # Simple magnitude-based detector
        # High variance/energy in features = strong emotion
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, context_feat):
        """
        context_feat: (B, D)
        返回: (B, 1) stimulus score in (0, 1)
        """
        # Magnitude-based scoring
        magnitude = torch.norm(context_feat, dim=1, keepdim=True)  # (B, 1)
        # Scale to (0, 1) range - higher energy = higher stimulus
        score = torch.sigmoid(self.scale * (magnitude - 5.0))
        return score


class MethylationSlot(nn.Module):
    """
    Single methylation slot (LoRA-like weight consolidation).
    Stores permanent weight updates triggered by significant events.
    """

    def __init__(self, rank=8, target_dim=1024):
        super().__init__()
        self.rank = rank
        self.target_dim = target_dim
        # LoRA decomposition: W = B @ A
        self.lora_A = nn.Parameter(torch.randn(rank, target_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(target_dim, rank))
        # Metadata (non-learnable)
        self.register_buffer('timestamp', torch.zeros(1, dtype=torch.long))
        self.register_buffer('intensity', torch.zeros(1))

    def get_delta(self):
        """Return LoRA update: B @ A"""
        return torch.mm(self.lora_B, self.lora_A)

    def consolidate(self, timestamp, intensity):
        """Mark this slot as consolidated"""
        # Force evaluation mode and freeze
        self.timestamp.fill_(timestamp)
        self.intensity.fill_(intensity)
        # No requires_grad change needed - we keep it trainable for flexibility
        print(f"[MethylationSlot] Consolidated at step {timestamp}, intensity={intensity:.3f}")


class MetaPlasticityMemory(nn.Module):
    """
    Meta-Plasticity Memory: Dual-track memory system.

    Track 1 (Short-term): KV Cache-style attention (existing in transformer)
    Track 2 (Long-term): LoRA weight consolidation (this module)

    When emotion stimulus > strong_threshold, trigger methylation -
    consolidate the current adaptation into permanent weights.
    """

    def __init__(self, input_dim=1024, num_slots=4, rank=8,
                 strong_threshold=0.8, weak_threshold=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.rank = rank

        # Emotion detector
        self.emotion_detector = EmotionStimulusDetector(input_dim)

        # Methylation slots
        self.slots = nn.ModuleList([
            MethylationSlot(rank=rank, target_dim=input_dim)
            for _ in range(num_slots)
        ])
        self.slot_usage = [0] * num_slots

        # Statistics
        self.register_buffer('total_consolidations', torch.zeros(1))

        print(f"[MetaPlasticityMemory] Initialized with {num_slots} slots, rank={rank}")

    def forward(self, fused_feat, context_feat=None):
        """
        Args:
            fused_feat: (B, D) current features
            context_feat: (B, D) context for emotion detection
        Returns:
            enhanced_feat: (B, D) features with long-term memory applied
            emotion_score: (B, 1) emotion stimulus scores
        """
        B = fused_feat.shape[0]

        # Default: identity (no modification)
        enhanced = fused_feat.clone()
        emotion_score = None

        if context_feat is not None:
            # 1. Detect emotion stimulus
            emotion_score = self.emotion_detector(context_feat)

            # 2. Check for strong stimulus -> trigger methylation
            strong_mask = (emotion_score > self.strong_threshold).squeeze()
            if strong_mask.any():
                # Get max intensity in batch
                max_intensity = emotion_score.max().item()
                step = self.total_consolidations.item() + 1

                # Trigger consolidation on first strong sample
                self._trigger_methylation(step, max_intensity)
                self.total_consolidations += 1

        # 3. Apply all consolidated methylations
        for slot in self.slots:
            if slot.timestamp.item() > 0:  # Has been consolidated
                delta = slot.get_delta()  # (D, r) @ (r, D) = (D, D)
                # Apply with intensity weighting
                weight = slot.intensity.item()
                enhanced = enhanced + torch.mm(delta, enhanced.T).T * weight * 0.1

        return enhanced, emotion_score

    def _trigger_methylation(self, step, intensity):
        """Trigger methylation in an available slot"""
        # Find empty slot or oldest
        for i, slot in enumerate(self.slots):
            if slot.timestamp.item() == 0:
                slot.consolidate(step, intensity)
                self.slot_usage[i] += 1
                print(f"[MetaPlasticity] Slot {i} activated: step={step}, intensity={intensity:.3f}")
                return

        # All slots full - replace oldest
        oldest_idx = self.slot_usage.index(min(self.slot_usage))
        self.slots[oldest_idx] = MethylationSlot(
            rank=self.rank, target_dim=self.input_dim
        ).to(next(self.parameters()).device)
        self.slots[oldest_idx].consolidate(step, intensity)
        self.slot_usage[oldest_idx] += 1
        print(f"[MetaPlasticity] Slot {oldest_idx} replaced: step={step}, intensity={intensity:.3f}")


# =============================================================================
# Integration: Wrap original Censor modules
# =============================================================================

class CensorWithBiomimeticEnhancement(nn.Module):
    """
    Wrapper that adds DTN + Meta-Plasticity to Censor pipeline.
    Insert after fusion stage.
    """

    def __init__(self, base_censor):
        super().__init__()
        self.base_censor = base_censor

        # Add DTN-enhanced attention (replace FFA)
        from config.defaults import FFA_CONFIG
        self.dtn_ffa = DTNEnhancedFFA(FFA_CONFIG)

        # Add Meta-Plasticity Memory (after fusion)
        from config.defaults import FUSION_CONFIG
        fusion_dim = FUSION_CONFIG['output_dim']
        self.meta_memory = MetaPlasticityMemory(
            input_dim=fusion_dim,
            num_slots=4,
            rank=8,
            strong_threshold=0.8
        )

        print("[Censor+DTN] Initialized with biomimetic enhancements")

    def forward(self, x):
        """Forward with DTN + Meta-Plasticity"""
        # Note: This is a simplified wrapper.
        # For full integration, modify main.py to use these modules.
        return self.base_censor(x)