# =============================================================================
# Censor -- Human-Like Attention Event-Driven Mechanism
# =============================================================================
# Inspired by human attention dynamics:
#   1. Default Mode: "daydreaming" - baseline monitoring (10% attention)
#   2. Salience Detection: "orienting response" - detect change (30% attention)
#   3. Expression Emergence: "focused attention" - full analysis (100%)
#
# Key: NEVER fully silent! Always keep baseline awareness.
# But can rapidly gear-shift when expression emerges.
#
# State transitions:
#   AMBIENT → (salience detected) → ORIENTING → (expression confirmed) → FOCUSED
#     ↓                                              ↑
#     ←←←←←←← (decay back) ←←←←←←←←←←←←←←←←←
#
# The "attention bottleneck" ensures sensitivity while saving compute.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from config.defaults import FUSION_CONFIG, AU_DECODER_CONFIG, MOE_CONFIG


# =============================================================================
# Part 1: Human-Like Attention State Machine
# =============================================================================

class AttentionState:
    """Three-tier attention states like human brain."""

    # Default baseline (daydreaming) - keep 10% monitoring
    AMBIENT = "ambient"      # 10% attention, fast check

    # Salience detected - orienting response
    ORIENTING = "orienting"  # 30% attention, deeper check

    # Expression confirmed - focused attention
    FOCUSED = "focused"     # 100% attention, full analysis


class SalienceDetector(nn.Module):
    """
    Detects salience changes (bottom-up attention).

    Like human visual saliency: detects edges, motion, novelty.
    Output: salience score [0, 1]
    """

    def __init__(self, input_dim=512):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """Detect salience."""
        return self.detector(features)


class ExpressionConfidence(nn.Module):
    """
    Confirms expression presence (top-down attention).

    Checks if features match known expression patterns.
    Output: confidence [0, 1]
    """

    def __init__(self, input_dim=512):
        super().__init__()
        # Simple threshold-based confidence
        self.threshold = nn.Parameter(torch.tensor(0.3))

    def forward(self, features):
        """Check expression confidence."""
        magnitude = features.abs().mean(dim=-1, keepdim=True)
        confidence = torch.sigmoid((magnitude - self.threshold) * 2)
        return confidence


class HumanAttentionController(nn.Module):
    """
    Human-like attention controller.

    Key features:
    1. ALWAYS baseline monitoring (never fully silent)
    2. Salience-driven bottom-up attention
    3. Expression-driven top-down confirmation
    4. Rapid gear-shift on expression emergence
    5. Gradual decay back to ambient
    """

    def __init__(self, input_dim=1024):
        super().__init__()
        self.input_dim = input_dim

        # Components
        self.salience_detector = SalienceDetector(input_dim)
        self.expression_confidence = ExpressionConfidence(input_dim)

        # Attention levels for each state
        self.attention_levels = {
            AttentionState.AMBIENT: 0.1,    # 10% - baseline
            AttentionState.ORIENTING: 0.3,   # 30% - orienting
            AttentionState.FOCUSED: 1.0,       # 100% - focused
        }

        # Thresholds
        self.salience_threshold = 0.15   # Trigger orienting
        self.confidence_threshold = 0.4   # Trigger focused

        # Decay rate (return to ambient)
        self.decay_rate = 0.95

        # State tracking
        self.register_buffer('current_state_emb', torch.tensor(0.1))
        self.register_buffer('orient_count', torch.zeros(1))
        self.register_buffer('focus_count', torch.zeros(1))
        self.register_buffer('total_passes', torch.zeros(1))

    def forward(self, features: torch.Tensor) -> Tuple[str, Dict]:
        """
        Compute attention state.

        Args:
            features: (B, D) input features

        Returns:
            state: attention state name
            info: dict with attention details
        """
        B = features.shape[0]
        self.total_passes += 1

        # === Step 1: Always run baseline (AMBIENT mode) ===
        # This is the key: never fully silent!

        # === Step 2: Bottom-up attention (salience) ===
        salience = self.salience_detector(features).mean().item()

        # === Step 3: Top-down attention (expression confidence) ===
        confidence = self.expression_confidence(features).mean().item()

        # === State Machine ===
        if salience > self.salience_threshold and confidence > self.confidence_threshold:
            # Expression confirmed: FOCUSED
            state = AttentionState.FOCUSED
            self.focus_count += 1
            attention_level = self.attention_levels[AttentionState.FOCUSED]

        elif salience > self.salience_threshold:
            # Salience detected but not confirmed: ORIENTING
            state = AttentionState.ORIENTING
            self.orient_count += 1
            attention_level = self.attention_levels[AttentionState.ORIENTING]

        else:
            # Default: AMBIENT (always!)
            state = AttentionState.AMBIENT
            attention_level = self.attention_levels[AttentionState.AMBIENT]

        # Update state embedding for next frame
        self.current_state_emb = self.current_state_emb * self.decay_rate + attention_level * (1 - self.decay_rate)

        info = {
            'state': state,
            'salience': salience,
            'confidence': confidence,
            'attention_level': attention_level,
            'state_emb': self.current_state_emb.item()
        }

        return state, info


# =============================================================================
# Part 2: Event-Driven Fusion (Human Attention)
# =============================================================================

class EventDrivenFusionHuman(nn.Module):
    """
    Fusion with human-like attention.

    Unlike previous version that could skip entirely,
    this ALWAYS produces output but at different fidelity:
    - AMBIENT: quick weighted pass (10% compute, keeps monitoring)
    - ORIENTING: partial attention (30% compute)
    - FOCUSED: full fusion (100% compute)
    """

    def __init__(self, config: Dict = None):
        super().__init__()
        from config.defaults import FUSION_CONFIG
        cfg = {**FUSION_CONFIG, **(config or {})}

        # Base fusion (full computation)
        from model.fusion import TSFmicroFusion
        self.base_fusion = TSFmicroFusion(cfg)

        # Attention controller - use COMBINED dim (512 + 768 = 1280)
        combined_dim = cfg.get('fast_dim', 512) + cfg.get('slow_dim', 768)
        self.attention = HumanAttentionController(combined_dim)

        # Stats
        self.register_buffer('state_counts', torch.zeros(3))

    def forward(self, fast_feat: torch.Tensor, slow_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward with human attention.

        Args:
            fast_feat: (B, 512)
            slow_feat: (B, 768)

        Returns:
            fused: (B, 1024)
            info: dict with attention state
        """
        # Combine for attention检测
        combined = torch.cat([fast_feat, slow_feat], dim=1)

        # Get attention state
        state, attn_info = self.attention(combined)

        # === Compute based on attention level ===
        if state == AttentionState.AMBIENT:
            # Quick weighted pass (keep baseline awareness!)
            # Don't zero - use lightweight weighted average
            fast_pad = fast_feat[:, :1024] if fast_feat.shape[1] >= 1024 else F.pad(fast_feat, (0, 1024 - fast_feat.shape[1]))
            slow_pad = slow_feat[:, :1024] if slow_feat.shape[1] >= 1024 else F.pad(slow_feat, (0, 1024 - slow_feat.shape[1]))

            # Activity-weighted combination
            fast_act = fast_feat.abs().mean() + 1e-8
            slow_act = slow_feat.abs().mean() + 1e-8
            w_fast = fast_act / (fast_act + slow_act)
            w_slow = slow_act / (fast_act + slow_act)

            fused = w_fast * fast_pad + w_slow * slow_pad

            idx = 0
            method = 'ambient_weighted'

        elif state == AttentionState.ORIENTING:
            # Partial attention - single direction
            fast_pad = fast_feat[:, :1024] if fast_feat.shape[1] >= 1024 else F.pad(fast_feat, (0, 1024 - fast_feat.shape[1]))
            slow_pad = slow_feat[:, :1024] if slow_feat.shape[1] >= 1024 else F.pad(slow_feat, (0, 1024 - slow_feat.shape[1]))

            # Just use average (lighter than full attention)
            fused = (fast_pad + slow_pad) / 2

            idx = 1
            method = 'orienting_average'

        else:  # FOCUSED
            # Full attention
            fused = self.base_fusion(fast_feat, slow_feat)

            idx = 2
            method = 'focused_full'

        self.state_counts[idx] += 1

        info = {
            'state': state,
            'method': method,
            'salience': attn_info['salience'],
            'confidence': attn_info['confidence'],
            'attention_level': attn_info['attention_level']
        }

        return fused, info

    def get_stats(self) -> Dict:
        """Get state statistics."""
        total = self.state_counts.sum() + 1e-8
        return {
            'ambient': self.state_counts[0].item(),
            'orienting': self.state_counts[1].item(),
            'focused': self.state_counts[2].item(),
            'ambient_pct': self.state_counts[0].item() / total.item() * 100,
            'orienting_pct': self.state_counts[1].item() / total.item() * 100,
            'focused_pct': self.state_counts[2].item() / total.item() * 100,
        }


# =============================================================================
# Part 3: Event-Driven AU Decoder (Human Attention)
# =============================================================================

class EventDrivenAUDecoderHuman(nn.Module):
    """
    AU Decoder with human-like attention.

    Always produces output but at varying fidelity.
    """

    def __init__(self, config: Dict = None):
        super().__init__()
        from config.defaults import AU_DECODER_CONFIG
        cfg = {**AU_DECODER_CONFIG, **(config or {})}

        from model.decoders import DynamicAUDecoder
        self.base_decoder = DynamicAUDecoder(cfg)

        # Simple attention - use magnitude as salience
        self.apex_threshold = 0.3

    def forward(self, fused_feat: torch.Tensor) -> Tuple[Tuple, Dict]:
        """Forward with attention."""
        magnitude = fused_feat.abs().mean()

        # Always run full decoder
        au_intensities, opd = self.base_decoder(fused_feat)

        # Determine state based on magnitude
        if magnitude < self.apex_threshold * 0.5:
            state = AttentionState.AMBIENT
        elif magnitude < self.apex_threshold:
            state = AttentionState.ORIENTING
        else:
            state = AttentionState.FOCUSED

        info = {
            'state': state,
            'magnitude': magnitude.item(),
            'apex_threshold': self.apex_threshold
        }

        # Return as tuple
        au_output = (au_intensities, opd)
        return au_output, info


# =============================================================================
# Part 4: Event-Driven MoE (Human Attention)
# =============================================================================

class EventDrivenMoEHuman(nn.Module):
    """
    MoE with human-like attention.

    Always runs but with different expert activation.
    """

    def __init__(self, config: Dict = None):
        super().__init__()
        from config.defaults import MOE_CONFIG
        cfg = {**MOE_CONFIG, **(config or {})}

        from model.moe_head import MoEGatingNetwork
        self.base_moe = MoEGatingNetwork(cfg)

        # Expression type mapping
        self.expr_to_expert = {
            'happiness': 0, 'surprise': 0, 'contempt': 0,
            'sadness': 1, 'fear': 1, 'anger': 1,
            'disgust': 2,
        }

        self.register_buffer('active_expert', torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor, expression_type: Optional[str] = None) -> Tuple[Dict, Dict]:
        """Forward with attention."""
        magnitude = x.abs().mean()

        # Determine target expert
        if expression_type and expression_type in self.expr_to_expert:
            target_expert = self.expr_to_expert[expression_type]
        else:
            target_expert = self.active_expert.item()

        # Check attention level
        if magnitude < 0.1:
            state = AttentionState.AMBIENT
        elif magnitude < 0.3:
            state = AttentionState.ORIENTING
        else:
            state = AttentionState.FOCUSED

        # Always run (but can use different fidelity)
        output = self.base_moe(x, training=False)

        info = {
            'state': state,
            'active_expert': target_expert,
            'magnitude': magnitude.item()
        }

        return output, info


# =============================================================================
# Factory Functions
# =============================================================================

def create_human_attention_fusion(config: Dict = None) -> EventDrivenFusionHuman:
    """Create human-attention fusion."""
    return EventDrivenFusionHuman(config)


def create_human_attention_au_decoder(config: Dict = None) -> EventDrivenAUDecoderHuman:
    """Create human-attention AU decoder."""
    return EventDrivenAUDecoderHuman(config)


def create_human_attention_moe(config: Dict = None) -> EventDrivenMoEHuman:
    """Create human-attention MoE."""
    return EventDrivenMoEHuman(config)