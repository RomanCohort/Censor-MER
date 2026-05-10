# =============================================================================
# Censor -- Biological Gating Mechanism (BioMoE)
# =============================================================================
# Inspired by:
#   1. Neuronal membrane potential (膜电位) - cumulative state
#   2. Emotional state affecting cognition - mood-dependent routing
#   3. Homeostatic regulation - maintaining balance
#
# Key innovation: Gating depends on both input AND historical membrane potential.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque


# =============================================================================
# Part 1: Membrane Potential (膜电位)
# =============================================================================
# Biological analogy: Neurons maintain a membrane potential that changes
# with each input. High potential = "excited" state, low = "depressed".
# This potential affects how the neuron responds to future inputs.

class MembranePotential(nn.Module):
    """
    Membrane potential that accumulates over time, UPDATED with FEEDBACK.

    Key change: Now depends on actual effect (prediction correctness), not just input intensity.
    Feedback values:
        - 1.0: prediction was correct (positive reinforcement)
        - 0.0: prediction was wrong (negative reinforcement)
        - -1.0: user corrected the error (strong negative)
        - None: no feedback (maintain current state)
    """

    def __init__(self, dim, decay_rate=0.95, initial_potential=0.0):
        super().__init__()
        self.dim = dim
        self.decay_rate = decay_rate

        # Membrane potential (stateful - persists across calls)
        self.register_buffer('potential',
            torch.full((1, dim), initial_potential))

        # Statistics
        self.register_buffer('call_count', torch.zeros(1))
        self.register_buffer('avg_activation', torch.zeros(1))
        self.register_buffer('positive_count', torch.zeros(1))
        self.register_buffer('negative_count', torch.zeros(1))

    def forward(self, input_feat, feedback=None, update=True):
        """
        Args:
            input_feat: (B, D) current input
            feedback: optional feedback from user:
                - 1.0 (correct): increase positive potential
                - 0.0 (wrong): increase negative potential
                - -1.0 (corrected): strong negative
                - None: maintain state (ignore input)
            update: bool, whether to update potential
        Returns:
            potential: (B, D) current membrane potential
            activation: (B, 1) how "excited" the membrane is
        """
        current_potential = self.potential

        # === KEY CHANGE: Feedback-based update (not intensity-based) ===
        if feedback is not None:
            # Feedback directly modulates membrane potential
            positive_delta = F.relu(feedback) * (1 - self.decay_rate)  # reward: +delta
            negative_delta = F.relu(-feedback) * (1 - self.decay_rate) * 0.5  # penalty: -0.5*delta

            # Update buffer (scalar addition broadcasts)
            self.potential = self.potential + positive_delta - negative_delta
            self.potential = self.potential.clamp(-1, 1)  # Bound

            # Track stats
            self.positive_count += max(0, feedback)
            self.negative_count += max(0, -feedback)

            # Activation = how positive the recent feedback is
            activation = (self.positive_count / (self.call_count + 1)).clamp(0, 1)
            self.avg_activation = activation

            if update:
                self.call_count += 1

        else:
            # No feedback: decay toward baseline
            if update:
                self.potential = self.potential * self.decay_rate
                self.call_count += 1

            # Activation based on accumulated positive feedback ratio
            total = self.positive_count + self.negative_count + 1e-8
            activation = (self.positive_count / total).unsqueeze(0)

        # Activation level (0 = negative/bad history, 1 = positive/good history)
        activation_level = torch.sigmoid(activation * 4 - 2)  # Shift to (-1, 1) range better

        return current_potential, activation_level

    def apply_feedback(self, feedback):
        """External method to apply feedback directly"""
        # feedback in [-1, 1], convert to tensor
        if not isinstance(feedback, torch.Tensor):
            feedback = torch.tensor(feedback)
        # Call forward with no input but with feedback
        self.forward(None, feedback=feedback, update=True)

    def get_state(self):
        """Get current feedback state"""
        return {
            'potential': self.potential.mean().item(),
            'positive_count': self.positive_count.item(),
            'negative_count': self.negative_count.item(),
            'accuracy': self.positive_count.item() / max(1, self.call_count.item())
        }

    def reset(self):
        """Reset to baseline"""
        self.potential.fill_(0.0)
        self.call_count.fill_(0)
        self.positive_count.fill_(0)
        self.negative_count.fill_(0)


class EmotionalState(nn.Module):
    """
    Emotional state derived from membrane potential (NOW FEEDBACK-DRIVEN).

    Key change: Derives from actual prediction feedback, not input magnitude.
    - Good history (high positive_count) → positive mood
    - Bad history (high negative_count) → negative mood

    Output:
        mood: (B, 1) emotional bias in (-1, 1)
        - positive = confident, eager to routing to expert 1
        - negative = conservative, spread routing evenly
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Track feedback history as buffers
        self.register_buffer('positive_count', torch.zeros(1))
        self.register_buffer('negative_count', torch.zeros(1))

    def forward(self, membrane_potential, feedback_stats=None):
        """
        Args:
            membrane_potential: (B, D) - not used anymore, kept for compatibility
            feedback_stats: dict with positive_count, negative_count
        Returns:
            mood: (1, 1) emotional bias in (-1, 1)
        """
        # Now derive mood from FEEDBACK history, not input
        if feedback_stats is not None:
            pos = feedback_stats.get('positive_count', 0)
            neg = feedback_stats.get('negative_count', 0)
        else:
            pos = self.positive_count.item()
            neg = self.negative_count.item()

        total = pos + neg + 1e-8
        # Positive ratio determines mood
        # If mostly correct → positive mood (confident)
        # If mostly wrong → negative mood (suppressed)
        ratio = pos / total

        # Map to (-1, 1): if ratio > 0.5 → positive, else → negative
        mood_val = (ratio - 0.5) * 2  # (0 -> -1, 0.5 -> 0, 1 -> 1)
        mood = torch.tensor([[mood_val]])

        return mood


# =============================================================================
# Part 2: BioGating (Biological Gating Mechanism)
# =============================================================================

class BioGate(nn.Module):
    """
    Biological gating: Gate depends on BOTH input AND membrane potential.

    Unlike standard attention where gate = f(input),
    BioGate computes: gate = f(input, membrane_potential, emotional_state)

    This models:
        - "In a bad mood, everything looks negative"
        - "After success, I'm more confident"
    """

    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts

        # Standard input-based routing
        self.input_gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
        )

        # Membrane potential modifier
        # How much the membrane affects gating
        self.membrane_bias = nn.Parameter(torch.zeros(num_experts))

        # Emotional modifier
        # How mood amplifies/suppresses each expert
        self.emotion_gain = nn.Parameter(torch.ones(num_experts))

    def forward(self, input_feat, membrane_potential, emotional_state):
        """
        Args:
            input_feat: (B, D) current input
            membrane_potential: (1, D) or (B, D) accumulated state
            emotional_state: (1, 1) mood bias in (-1, 1)
        Returns:
            gate_logits: (B, K) routing logits
        """
        B = input_feat.shape[0]

        # 1. Input-based routing
        input_logits = self.input_gate(input_feat)  # (B, K)

        # 2. Membrane bias (history-dependent adjustment)
        # Aggregate membrane potential to expert space
        membrane_effect = self.membrane_bias.unsqueeze(0)  # (K,)

        # 3. Emotional modulation
        # Positive mood = amplify certain experts
        # Negative mood = suppress certain experts
        emotion_mod = self.emotion_gain * emotional_state  # (K,)

        # Combine: base + membrane + emotion
        # Broadcast membrane_effect and emotion_mod to batch
        membrane_effect = membrane_effect.expand(B, -1)
        emotion_mod = emotion_mod.expand(B, -1)

        gate_logits = input_logits + membrane_effect + emotion_mod

        return gate_logits


# =============================================================================
# Part 3: BioMoE (Biological MoE)
# =============================================================================

class BioMoE(nn.Module):
    """
    Biological MoE: Mixture of Experts with membrane potential.

    Key differences from standard MoE:
        1. Each expert has its own membrane potential
        2. Gating considers both input AND emotional state
        3. Expert selection is "mood-contingent"

    Structure:
        - N experts (MLPs)
        - N membrane potentials (one per expert)
        - 1 emotional state tracker
        - 1 bio gate
    """

    def __init__(self, input_dim, output_dim, num_experts=3, expert_hidden=2048,
                 k=2, decay_rate=0.95):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, output_dim)
            )
            for _ in range(num_experts)
        ])

        # Membrane potentials (one per expert)
        self.membrane_potentials = nn.ModuleList([
            MembranePotential(input_dim, decay_rate=decay_rate)
            for _ in range(num_experts)
        ])

        # Shared emotional state
        self.emotional_state = EmotionalState(input_dim)

        # Biological gating
        self.bio_gate = BioGate(input_dim, num_experts)

        # Load balancing (standard MoE technique)
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.load_weight = 0.01

    def forward(self, x, update_membrane=True):
        """
        Args:
            x: (B, D) input features
            update_membrane: bool, whether to update membrane potential
        Returns:
            output: (B, output_dim) weighted expert sum
            gate_weights: (B, K) routing weights
            auxiliary_loss: scalar for load balancing
            membrane_info: dict with emotional state info
        """
        B = x.shape[0]

        # === Step 1: Get membrane potentials for each expert ===
        membrane_feats = []
        activations = []
        for i, membrane in enumerate(self.membrane_potentials):
            mem_feat, activation = membrane(x, update=update_membrane)
            membrane_feats.append(mem_feat)
            activations.append(activation)

        # === Step 2: Get emotional state ===
        # Use average membrane potential
        avg_membrane = torch.stack(membrane_feats).mean(dim=0)  # (1, D)
        emotional_state = self.emotional_state(avg_membrane)  # (1, 1)

        # === Step 3: Bio gating ===
        gate_logits = self.bio_gate(x, avg_membrane, emotional_state)  # (B, K)

        # Top-k selection
        top_k_vals, top_k_idx = torch.topk(gate_logits, self.k, dim=1)
        gate_weights = F.softmax(top_k_vals, dim=1)

        # Mask to top-k
        gate_logits_masked = torch.full_like(gate_logits, float('-inf'))
        gate_logits_masked.scatter_(1, top_k_idx, 0)
        gate_probs = F.softmax(gate_logits_masked, dim=1)

        # === Update usage statistics ===
        # Count which experts were selected in top-k
        for idx in top_k_idx[:, 0]:  # Primary expert
            self.expert_usage[idx.item()] += 0.01
        self.expert_usage = self.expert_usage * 0.99  # Decay

        # === Step 4: Expert computation ===
        # Gather top-k experts
        expert_outputs = []
        for i in range(self.num_experts):
            # Apply membrane activation to expert input
            mem_activation = activations[i]  # (1, 1)
            expert_out = self.experts[i](x)  # (B, output_dim)
            # Modulate by membrane activation
            expert_out = expert_out * mem_activation
            expert_outputs.append(expert_out)

        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, K, output_dim)

        # === Step 5: Weighted combination ===
        # Apply all experts, weight by full gate distribution
        gate_probs_exp = gate_probs.unsqueeze(-1)  # (B, K, 1)
        output = (gate_probs_exp * expert_stack).sum(dim=1)  # (B, output_dim)

        # === Auxiliary loss (load balancing) ===
        aux_loss = self.load_weight * (self.expert_usage - 1.0 / self.num_experts).pow(2).sum()

        membrane_info = {
            'emotional_state': emotional_state.item(),
            'membrane_activations': [a.item() for a in activations],
            'expert_usage': self.expert_usage.tolist()
        }

        return output, gate_weights, aux_loss, membrane_info


# =============================================================================
# Part 4: Comparison with Standard MoE
# =============================================================================

class StandardMoEComparison:
    """
    Compare standard MoE vs BioMoE behavior.
    """

    @staticmethod
    def analyze_difference():
        """Show key differences"""
        print("\n" + "=" * 60)
        print(" Standard MoE vs BioMoE")
        print("=" * 60)
        print("""
| Aspect          | Standard MoE      | BioMoE                      |
|----------------|-------------------|----------------------------|
| Routing        | gate = f(input)    | gate = f(input, membrane)   |
| Memory         | None              | Membrane potential          |
| Emotional     | None              | Mood bias affects routing |
| Persistence   | Stateless         | Stateful (accumulates)      |
| Adaptation    | None              | Homeostatic regulation     |
| Efficiency    | High (all experts)| Controlled by threshold    |
        """)
        return


# =============================================================================
# Integration Interface
# =============================================================================

class BioMoEWrapper(nn.Module):
    """
    Wrapper to replace standard MoE in Censor.
    """

    def __init__(self, base_moe):
        super().__init__()
        # Wrap the original MoE with BioMoE
        from config.defaults import MOE_CONFIG

        self.bio_moe = BioMoE(
            input_dim=MOE_CONFIG['input_dim'],
            output_dim=MOE_CONFIG['output_dim'],
            num_experts=MOE_CONFIG['num_experts'],
            expert_hidden=MOE_CONFIG['expert_hidden'],
            k=MOE_CONFIG['top_k'],
            decay_rate=0.95
        )

        print("[BioMoEWrapper] Initialized biological MoE")

    def forward(self, x):
        output, gates, aux_loss, info = self.bio_moe(x)
        return output, gates, aux_loss, info