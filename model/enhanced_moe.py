# =============================================================================
# Censor -- Enhanced MoE with Biological Gating
# =============================================================================
# Enhanced version of MoE that adds:
#   1. Membrane potential accumulation
#   2. Emotional state modulation
#   3. Bio-style gating
#
# Can work in three modes:
#   - "standard": Original MoE behavior
#   - "bio": Full BioMoE behavior
#   - "hybrid": Bio-gating + original experts
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.defaults import MOE_CONFIG


# =============================================================================
# Import original MoE
# =============================================================================

class EnhancedMoE(nn.Module):
    """
    Enhanced MoE that supports biological gating mechanisms.

    Modes:
        - "standard": Original MoE (Stateless, input-only routing)
        - "bio": Full BioMoE (membrane potential + emotional state)
        - "hybrid": Original experts + bio gating

    Usage:
        # Use standard (original behavior)
        moe = EnhancedMoE(mode="standard")

        # Use full bio
        moe = EnhancedMoE(mode="bio")

        # Hybrid (recommended)
        moe = EnhancedMoE(mode="hybrid", enable_membrane=True)
    """

    def __init__(self, config=None, mode="hybrid", enable_membrane=True,
                 enable_emotion=True, decay_rate=0.95):
        super().__init__()
        cfg = config or MOE_CONFIG

        self.mode = mode
        self.enable_membrane = enable_membrane
        self.enable_emotion = enable_emotion

        # === Original MoE components ===
        input_dim = cfg['input_dim']
        output_dim = cfg['num_classes']  # 7 ME categories
        num_experts = cfg['num_experts']
        expert_hidden = cfg['hidden_dim']  # 512
        self.top_k = cfg['top_k']
        self.noise_type = cfg.get('noise_type', 'correlated')
        self.noise_std = cfg.get('noise_std', 0.1)

        # Gating network (original)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
        self.gate_noise = nn.Parameter(torch.zeros(num_experts), requires_grad=True)

        # Experts (original)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, expert_hidden // 2),
                nn.GELU(),
                nn.Linear(expert_hidden // 2, output_dim)
            )
            for _ in range(num_experts)
        ])

        # === Biological enhancement components ===
        if mode in ["bio", "hybrid"]:
            # Import from biomoe
            from model.biomoe import (
                MembranePotential, EmotionalState, BioGate
            )

            # Membrane potentials (one per expert + one global)
            self.membrane_global = MembranePotential(input_dim, decay_rate=decay_rate)

            # Emotional state
            self.emotional_state = EmotionalState(input_dim) if enable_emotion else None

            # Bio gate (adds membrane + emotion to original gating)
            self.bio_gate = BioGate(input_dim, num_experts)

        # === State tracking ===
        self.register_buffer('total_calls', torch.zeros(1))
        self.register_buffer('avg_emotion', torch.zeros(1))

        # Initialize
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        print(f"[EnhancedMoE] Mode: {mode}, membrane: {enable_membrane}, emotion: {enable_emotion}")

    def forward(self, x, feedback=None, update_state=True):
        """
        Args:
            x: (B, D) input features
            feedback: optional feedback about prediction quality:
                - 1.0: user confirmed correct
                - 0.0: prediction was wrong
                - -1.0: user explicitly corrected
            update_state: whether to update membrane/emotion
        Returns:
            output: (B, output_dim) predictions
            gate_weights: (B, K) routing weights
            aux_loss: load balancing loss
            info: dict with membrane/emotion info
        """
        B = x.shape[0]

        if self.mode == "standard":
            return self._forward_standard(x)

        elif self.mode == "bio":
            return self._forward_bio(x, update_state)

        elif self.mode == "hybrid":
            return self._forward_hybrid(x, update_state)

    def _forward_standard(self, x):
        """Original MoE behavior"""
        # Gating
        gate_logits = self.gate(x)

        # Add noise for load balancing
        if self.training and self.noise_type == "correlated":
            noise = torch.randn_like(gate_logits) * self.gate_noise.unsqueeze(0)
            gate_logits = gate_logits + noise

        # Top-k selection
        top_k_vals, top_k_idx = torch.topk(gate_logits, self.top_k, dim=1)
        gate_weights = F.softmax(top_k_vals, dim=1)

        # Mask to top-k
        gate_mask = torch.full_like(gate_logits, float('-inf'))
        gate_mask.scatter_(1, top_k_idx, 0)
        gate_probs = F.softmax(gate_mask, dim=1)

        # Expert computation
        outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(x)
            outputs.append(out)
        expert_stack = torch.stack(outputs, dim=1)

        # Weighted combination
        output = (gate_probs.unsqueeze(-1) * expert_stack).sum(dim=1)

        # Load balancing - count primary expert selections
        usage = torch.zeros(self.top_k, device=x.device)
        for idx in top_k_idx[:, 0]:
            usage[idx.item()] += 1
        usage = usage / usage.sum() * self.top_k  # Normalize

        aux_loss = (usage - 1.0 / self.top_k).pow(2).sum() * 0.01

        info = {'mode': 'standard', 'expert_usage': usage.tolist()}

        return output, gate_weights, aux_loss, info

    def _forward_bio(self, x, update_state):
        """Full BioMoE behavior"""
        # Import directly to avoid circular reference
        from model.biomoe import BioMoE

        # Use a simpler inline version
        return self._forward_hybrid(x, update_state)

    def _forward_hybrid(self, x, update_state):
        """Hybrid: Original experts + bio gating WITH FEEDBACK"""
        B = x.shape[0]

        # === KEY: Get membrane potential (feedback passed externally via apply_feedback) ===
        membrane_info = {}
        emotional_state = 0.0

        if self.enable_membrane:
            membrane_pot, activation = self.membrane_global(x, feedback=None, update=update_state)
            membrane_info['activation'] = activation.item()

            if self.enable_emotion and self.emotional_state is not None:
                stats = self.membrane_global.get_state()
                emotional_state = self.emotional_state(None, feedback_stats=stats)
                emotional_state_val = emotional_state.item() if isinstance(emotional_state, torch.Tensor) else emotional_state
                membrane_info['emotional_state'] = emotional_state_val
                self.avg_emotion = 0.9 * self.avg_emotion + 0.1 * emotional_state_val

        # === Standard gating ===
        gate_logits = self.gate(x)  # (B, K)

        # === Bio modulation ===
        if self.enable_membrane and self.enable_emotion:
            # Add membrane bias
            membrane_bias = self.bio_gate.membrane_bias  # (K,)
            gate_logits = gate_logits + membrane_bias.unsqueeze(0) * 0.1

            # Add emotional modulation
            if emotional_state != 0.0:
                emotion_gain = self.bio_gate.emotion_gain  # (K,)
                gate_logits = gate_logits + emotion_gain.unsqueeze(0) * emotional_state * 0.1

        # Top-k selection
        top_k_vals, top_k_idx = torch.topk(gate_logits, self.top_k, dim=1)
        gate_weights = F.softmax(top_k_vals, dim=1)

        # Mask to top-k
        gate_mask = torch.full_like(gate_logits, float('-inf'))
        gate_mask.scatter_(1, top_k_idx, 0)
        gate_probs = F.softmax(gate_mask, dim=1)

        # === Expert computation (with membrane modulation) ===
        outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(x)

            # Modulate by membrane activation if available
            if self.enable_membrane and 'activation' in membrane_info:
                mem_act = membrane_info['activation']
                out = out * (1 + mem_act * 0.1)  # Slight modulation

            outputs.append(out)
        expert_stack = torch.stack(outputs, dim=1)

        # Weighted combination
        output = (gate_probs.unsqueeze(-1) * expert_stack).sum(dim=1)

        # Load balancing - simplified
        # Just measure how evenly experts are used
        expert_idx = top_k_idx[:, 0]
        usage = torch.tensor([(expert_idx == i).float().sum() for i in range(3)])
        usage = usage / (usage.sum() + 1e-8)
        aux_loss = (usage.mean() - 0.33).pow(2) * 0.01  # Simpler loss

        # Update stats
        self.total_calls += 1

        info = {
            'mode': self.mode,
            'membrane_activation': membrane_info.get('activation', None),
            'emotional_state': emotional_state if isinstance(emotional_state, float) else emotional_state.item(),
            'expert_usage': usage.tolist()
        }

        return output, gate_weights, aux_loss, info

    def get_state(self):
        """Get current membrane/emotion state with feedback stats"""
        state = {
            'total_calls': self.total_calls.item(),
            'avg_emotion': self.avg_emotion.item()
        }
        if self.enable_membrane:
            mem_state = self.membrane_global.get_state()
            state['membrane_potential'] = mem_state.get('potential', 0)
            state['positive_count'] = mem_state.get('positive_count', 0)
            state['negative_count'] = mem_state.get('negative_count', 0)
            state['accuracy'] = mem_state.get('accuracy', 0)
        return state

    def apply_feedback(self, feedback):
        """Apply feedback externally (called by user UI)"""
        if self.enable_membrane:
            self.membrane_global.apply_feedback(feedback)

    def reset_state(self):
        """Reset membrane state"""
        if self.enable_membrane:
            self.membrane_global.reset()
        self.total_calls.fill_(0)
        self.avg_emotion.fill_(0)
        print("[EnhancedMoE] State reset")


# =============================================================================
# Wrapper for Censor integration
# =============================================================================

class EnhancedMoEWrapper(nn.Module):
    """
    Drop-in replacement for Censor's MoE.

    Usage:
        from model.moe_head import MoEGatingNetwork
        enhanced = EnhancedMoEWrapper(MoEGatingNetwork())
    """

    def __init__(self, original_moe=None, mode="hybrid",
                 enable_membrane=True, enable_emotion=True):
        super().__init__()

        # Create enhanced version
        self.enhanced_moe = EnhancedMoE(
            mode=mode,
            enable_membrane=enable_membrane,
            enable_emotion=enable_emotion
        )

        print(f"[EnhancedMoEWrapper] Initialized in '{mode}' mode")

    def forward(self, x):
        output, gates, aux_loss, info = self.enhanced_moe(x)
        return output, gates, aux_loss