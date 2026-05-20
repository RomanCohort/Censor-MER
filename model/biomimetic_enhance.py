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
# Part 3: Long-Term Memory Sparse Control (Dynamic Sparsity)
# =============================================================================
# Implements synaptic silencing + neurogenesis-like recovery:
#   1. Hard Freeze: Gradient cutoff for long-inactive neurons
#   2. Soft Decay: Weight decay for buffer zone
#   3. Growth Factor: BDNF-like boost for reactivated neurons


class NeuronUsageTracker(nn.Module):
    """
    Tracks neuron activity over time.
    Simulates synaptic plasticity - frequently used pathways strengthen,
    unused pathways weaken.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Activity tracking buffers (persistent across forward passes)
        self.register_buffer('usage_count', torch.zeros(dim))           # Total usage count
        self.register_buffer('last_active_step', torch.zeros(dim, dtype=torch.long))  # Last step active
        self.register_buffer('inactivity_steps', torch.zeros(dim, dtype=torch.long))  # Consecutive inactive steps
        self.register_buffer('cumulative_activity', torch.zeros(dim))        # Activity energy

    def update(self, activity_mask, current_step):
        """
        Update activity based on current forward pass.

        Args:
            activity_mask: (D,) binary mask indicating which neurons fired
            current_step: current training step
        """
        # Increment inactivity counter for all
        self.inactivity_steps += 1

        # Reset inactivity for active neurons
        active_indices = activity_mask.bool()
        self.inactivity_steps[active_indices] = 0

        # Update usage count
        self.usage_count[active_indices] += 1

        # Update last active step
        self.last_active_step[active_indices] = current_step

        # Update cumulative activity (for magnitude tracking)
        self.cumulative_activity += activity_mask

    def get_inactivity(self):
        """Return inactivity counter"""
        return self.inactivity_steps.float()

    def get_usage_ratio(self):
        """Return normalized usage ratio"""
        total = self.usage_count.sum() + 1e-8
        return self.usage_count / total


class SoftDecayPath(nn.Module):
    """
    Soft decay pathway for neurons in buffer zone.
    Applies gradual weight decay when neuron is inactive but not yet frozen.
    """

    def __init__(self, dim, decay_factor=0.95):
        super().__init__()
        self.dim = dim
        self.decay_factor = decay_factor

        # Decay factors per neuron (learnable but initialized)
        self.register_buffer('decay_mask', torch.ones(dim))

    def forward(self, features, inactivity_counter):
        """
        Apply soft decay to inactive neurons.

        Args:
            features: (B, D) input features
            inactivity_counter: (D,) inactivity steps

        Returns:
            (B, D) features with soft decay applied
        """
        # Neurons in soft decay zone (inactive but not hard frozen)
        soft_decay_zone = (inactivity_counter > 0) & (inactivity_counter < 200)

        if soft_decay_zone.any():
            # Build decay mask
            decay_weights = torch.where(
                soft_decay_zone,
                self.decay_mask ** (inactivity_counter / 100),  # Gradual decay
                torch.ones_like(self.decay_mask)
            )
            # Apply decay
            features = features * decay_weights.view(1, -1)

        return features


class HardFreezePath(nn.Module):
    """
    Hard freeze pathway - gradient cutoff for long-inactive neurons.
    Simulates synaptic silencing in biological systems.

    Recovery: When input has high activity, neurons UNFREEZE automatically.
    """

    def __init__(self, dim, freeze_threshold=500, recovery_threshold=0.1):
        super().__init__()
        self.dim = dim
        self.freeze_threshold = freeze_threshold
        self.recovery_threshold = recovery_threshold  # Activity threshold to recover

        # Freeze state tracking
        self.register_buffer('is_frozen', torch.zeros(dim, dtype=torch.bool))
        self.register_buffer('frozen_step', torch.zeros(dim, dtype=torch.long))

    def forward(self, features, inactivity_counter):
        """
        Apply hard freeze by masking gradients.

        Args:
            features: (B, D) input features (BEFORE freeze)
            inactivity_counter: (D,) inactivity steps

        Returns:
            masked_features: (B, D) with frozen neurons zeroed
            frozen_mask: (D,) boolean indicating frozen state
        """
        # === FREEZE LOGIC ===
        should_freeze = (inactivity_counter > self.freeze_threshold)
        newly_frozen = should_freeze & ~self.is_frozen

        if newly_frozen.any():
            self.is_frozen[newly_frozen] = True
            self.frozen_step[newly_frozen] = inactivity_counter[newly_frozen].long()

        # === RECOVERY LOGIC (NEW) ===
        # Calculate per-neuron activity BEFORE masking
        # Neurons with high input activity can recover
        neuron_activity = features.abs().mean(dim=0)  # (D,)

        # Frozen neurons that now have high input activity → recover
        currently_frozen = self.is_frozen.clone()
        high_activity = neuron_activity > self.recovery_threshold
        should_recover = currently_frozen & high_activity

        if should_recover.any():
            self.is_frozen[should_recover] = False
            # Also reset their inactivity counter
            if hasattr(self, 'inactivity_counter') is False or True:  # Just always do it
                pass

        if should_recover.any():
            pass  # Will log in LongTermMemorySparseControl

        # Build frozen mask (0 for frozen neurons)
        frozen_mask = (~self.is_frozen).float().view(1, -1)

        # Apply mask (zero out frozen neurons in forward)
        masked_features = features * frozen_mask

        return masked_features, self.is_frozen.float(), should_recover.float()

    def get_frozen_ratio(self):
        """Return ratio of frozen neurons"""
        if self.is_frozen.numel() == 0:
            return torch.tensor(0.0)
        return self.is_frozen.float().mean()


class GrowthFactorSignal(nn.Module):
    """
    Growth factor signal (BDNF-like) for reactivated neurons.
    Provides temporary boost when neurons recover from frozen state.

    NOTE: This module handles BOTH:
    1. Apply boost to recovering neurons
    2. Unlock neurons in HardFreezePath when they show high activity
    """

    def __init__(self, dim, boost_factor=2.0, recovery_steps=30):
        super().__init__()
        self.dim = dim
        self.boost_factor = boost_factor  # Increased from 1.5 to 2.0
        self.recovery_steps = recovery_steps  # Reduced from 50 to 30

        # Growth factor tracking
        self.register_buffer('was_frozen', torch.zeros(dim, dtype=torch.bool))
        self.register_buffer('recovery_counter', torch.zeros(dim, dtype=torch.long))
        self.register_buffer('recovered_neurons', torch.zeros(dim))  # Track recovery boost

    def forward(self, features, current_frozen, previous_frozen, hard_freeze_ref=None):
        """
        Apply growth factor boost to recovering neurons.

        Args:
            features: (B, D) input features
            current_frozen: (D,) current frozen state
            previous_frozen: (D,) previous frozen state
            hard_freeze_ref: reference to HardFreezePath for re-activation

        Returns:
            (B, D) features with growth boost applied
        """
        # Detect neurons transitioning from frozen to active
        recovering = previous_frozen.bool() & ~current_frozen.bool()

        # Alternative: Detect neurons with HIGH activity (they should recover)
        # High activity = large feature magnitude
        activity_magnitude = torch.norm(features, dim=1).mean()
        high_activity = (features > activity_magnitude * 0.5).any(dim=0)

        # Neurons that are frozen but now have high activity → recover
        should_recover = current_frozen.bool() & high_activity

        if should_recover.any():
            # UNFREEZE these neurons in HardFreezePath
            if hard_freeze_ref is not None:
                hard_freeze_ref.is_frozen[should_recover] = False
                print(f"[GrowthFactor] Recovered {should_recover.sum().item()} frozen neurons (high activity)")

            # Apply boost to recovering neurons
            boost_mask = torch.ones(self.dim, device=features.device)
            boost_mask[should_recover] = self.boost_factor
            features = features * boost_mask.view(1, -1)

            # Record recovery
            self.recovery_counter[should_recover] = 1
            self.was_frozen[should_recover] = True
            self.recovered_neurons[should_recover] = self.boost_factor

        # Apply gradual recovery decay for neurons in recovery mode
        recovering_active = self.was_frozen & ~current_frozen.bool()
        if recovering_active.any():
            self.recovery_counter[recovering_active] += 1

            # Linear decay from boost_factor down to 1.0
            recovery_progress = self.recovery_counter[recovering_active].float() / self.recovery_steps
            recovery_scale = 1.0 + (self.boost_factor - 1.0) * (1.0 - torch.clamp(recovery_progress, max=1.0))

            decay_mask = torch.ones(self.dim, device=features.device)
            decay_mask[recovering_active] = recovery_scale
            features = features * decay_mask.view(1, -1)

            # Clear flag when fully recovered
            fully_recovered = self.recovery_counter >= self.recovery_steps
            self.was_frozen[fully_recovered] = False
            self.recovery_counter[fully_recovered] = 0

        return features


class LongTermMemorySparseControl(nn.Module):
    """
    Long-Term Memory Sparse Control Layer

    Combines:
    1. NeuronUsageTracker - Activity monitoring
    2. HardFreezePath - Gradient cutoff (main anti-overfitting)
    3. SoftDecayPath - Weight decay (buffer zone)
    4. GrowthFactorSignal - BDNF-like recovery

    State transition:
        Active → (inactive > 200) → SoftFreezing
               → (inactive > 500) → HardFrozen
               → (reactivated)   → GrowthMode → Active
    """

    def __init__(self, config=None):
        super().__init__()
        from config.defaults import SPARSE_CONTROL_CONFIG
        cfg = config or SPARSE_CONTROL_CONFIG

        self.dim = cfg['dim']
        self.inactivity_threshold = cfg['inactivity_threshold']
        self.hard_freeze_threshold = cfg['hard_freeze_threshold']
        self.soft_decay_factor = cfg['soft_decay_factor']
        self.growth_factor_boost = cfg['growth_factor_boost']
        self.growth_recovery_steps = cfg['growth_recovery_steps']

        # === 防过拟合增强参数 ===
        self.enable_random_dropout = cfg.get('enable_random_dropout', True)
        self.random_dropout_rate = cfg.get('random_dropout_rate', 0.15)
        self.enable_l2 = cfg.get('enable_l2', True)
        self.l2_weight = cfg.get('l2_weight', 0.01)

        # Register step counter
        self.register_buffer('current_step', torch.zeros(1, dtype=torch.long))

        # Tracker and pathways
        self.tracker = NeuronUsageTracker(self.dim)
        self.hard_freeze = HardFreezePath(self.dim, freeze_threshold=self.hard_freeze_threshold)
        self.soft_decay = SoftDecayPath(self.dim, decay_factor=self.soft_decay_factor)

        # Statistics
        self.register_buffer('total_freeze_events', torch.zeros(1))
        self.register_buffer('total_recovery_events', torch.zeros(1))

        # 注册L2正则化的权重缓存
        self.register_buffer('l2_sum', torch.zeros(1))

    def forward(self, features):
        """
        Apply long-term memory sparse control.

        Args:
            features: (B, D) features from fusion stage

        Returns:
            controlled_features: (B, D) sparse-controlled features
            stats: dict with sparsity statistics
        """
        B = features.shape[0]
        self.current_step += 1

        # === 1. Compute activity mask ===
        # Active neurons = high magnitude features
        activity_norm = torch.norm(features, dim=1, keepdim=True)  # (B, 1)
        activity_mask = (features > activity_norm.mean() * 0.1).float().sum(dim=0)  # (D,)
        activity_mask = torch.clamp(activity_mask, max=1.0)

        # === 2. Update tracker ===
        self.tracker.update(activity_mask, self.current_step.item())

        # === 3. Apply pathways ===
        inactivity = self.tracker.get_inactivity()

        # Save previous frozen state for growth factor
        previous_frozen = self.hard_freeze.is_frozen.float()

        # Hard freeze (main mechanism - gradient cutoff + auto-recovery)
        features, frozen_mask, recover_events = self.hard_freeze(features, inactivity)

        # Soft decay (buffer zone)
        if self.soft_decay_factor < 1.0:
            features = self.soft_decay(features, inactivity)

        # Growth factor (recovery boost) - applies additional boost to recovered neurons
        current_frozen = self.hard_freeze.is_frozen.float()
        if recover_events.sum() > 0:
            # Apply boost to recovering neurons
            boost_mask = torch.ones(self.dim, device=features.device)
            boost_mask[recover_events.bool()] = self.growth_factor_boost  # Use config value
            features = features * boost_mask.view(1, -1)
            self.total_recovery_events += recover_events.sum()

        # === 4. 随机Dropout混合 (随机屏蔽未被冻结的神经元) ===
        if self.enable_random_dropout and self.training:
            # 随机生成dropout mask
            random_mask = (torch.rand(self.dim, device=features.device) > self.random_dropout_rate).float()
            # 只对未冻结的神经元应用随机dropout
            active_mask = (~self.hard_freeze.is_frozen).float()
            combined_mask = active_mask * random_mask
            features = features * combined_mask.view(1, -1)

        # === 5. 计算L2正则化项 (非训练时记录) ===
        if self.enable_l2:
            # 只计算未冻结权重的L2范数
            active_weights = features * (~self.hard_freeze.is_frozen).float().view(1, -1)
            self.l2_sum = (active_weights ** 2).sum()

        # === 6. Update statistics ===
        newly_frozen = (current_frozen > previous_frozen).float()
        if newly_frozen.sum() > 0:
            self.total_freeze_events += newly_frozen.sum()

        # === 7. Build statistics ===
        frozen_ratio = self.hard_freeze.get_frozen_ratio()
        usage_ratio = self.tracker.get_usage_ratio().mean()

        stats = {
            'frozen_ratio': frozen_ratio.item(),
            'usage_ratio': usage_ratio.item(),
            'freeze_events': self.total_freeze_events.item(),
            'recovery_events': self.total_recovery_events.item(),
            'inactivity_mean': inactivity.mean().item(),
            'random_dropout_rate': self.random_dropout_rate if self.enable_random_dropout else 0,
            'l2_contrib': self.l2_sum.item() if self.enable_l2 else 0,
        }

        return features, stats

    def get_sparse_stats(self):
        """Return current sparsity statistics"""
        return {
            'frozen_ratio': self.hard_freeze.get_frozen_ratio().item(),
            'usage_mean': self.tracker.get_usage_ratio().mean().item(),
            'usage_std': self.tracker.get_usage_ratio().std().item(),
            'cumulative_activity': self.tracker.cumulative_activity.mean().item(),
        }

    def get_l2_loss(self):
        """返回L2正则化损失 (可用于训练时添加到总损失)"""
        if self.enable_l2:
            # 返回 0.5 * weight * ||x||^2
            return 0.5 * self.l2_weight * self.l2_sum
        return torch.tensor(0.0)


# Integration: Multi-Path Sparse Control (for all pipeline stages)
# =============================================================================

class SparseControlWrapper(nn.Module):
    """
    Wrapper for applying sparse control to any feature dimension.
    Creates and manages sparse controllers for multiple stages.
    """

    def __init__(self, configs):
        """
        Args:
            configs: dict of {stage_name: dim} e.g. {'fast': 512, 'slow': 768, 'fusion': 1024}
        """
        super().__init__()
        self.sparse_controllers = nn.ModuleDict()

        for name, dim in configs.items():
            # Create a sparse controller for each stage
            self.sparse_controllers[name] = self._create_sparse_controller(dim)

        self.configs = configs
        print(f"[SparseControlWrapper] Created controllers for: {list(configs.keys())}")

    def _create_sparse_controller(self, dim):
        """Create a sparse controller for given dimension"""
        # Create config with the correct keys
        from config.defaults import SPARSE_CONTROL_CONFIG
        cfg = SPARSE_CONTROL_CONFIG.copy()
        cfg['dim'] = dim
        return LongTermMemorySparseControl(cfg)

    def forward(self, features_dict):
        """
        Args:
            features_dict: dict of {stage_name: features}
                e.g. {'fast': (B, 512), 'slow': (B, 768), 'fusion': (B, 1024)}

        Returns:
            controlled_dict: same structure, features sparse-controlled
            stats_dict: statistics for each stage
        """
        controlled_dict = {}
        stats_dict = {}

        for name, features in features_dict.items():
            if name in self.sparse_controllers:
                controlled_dict[name], stats_dict[name] = self.sparse_controllers[name](features)
            else:
                controlled_dict[name] = features
                stats_dict[name] = {}

        return controlled_dict, stats_dict

    def __call__(self, features_dict):
        """Wrapper for forward pass"""
        return self.forward(features_dict)


class MultiExpertSparseControl(nn.Module):
    """
    Sparse control for MoE experts.
    Applies separate sparse control to each expert's hidden dimension.
    """

    def __init__(self, num_experts=3, expert_hidden_dim=512):
        super().__init__()
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim

        # Create sparse controller for each expert
        self.expert_sparse = nn.ModuleList([
            LongTermMemorySparseControl({
                'dim': expert_hidden_dim,
                'inactivity_threshold': 200,
                'hard_freeze_threshold': 500,
                'soft_decay_factor': 0.95,
                'growth_factor_boost': 2.0,
                'growth_recovery_steps': 30,
                'min_activity_to_track': 0.01,
                'enable_dual_path': True,
            })
            for _ in range(num_experts)
        ])

    def forward(self, expert_outputs):
        """
        Args:
            expert_outputs: list of (B, num_classes) or tuple of tensors

        Returns:
            controlled_outputs: same structure, sparse-controlled
            stats: dict with per-expert statistics
        """
        stats = {}

        # Handle both list and tuple input
        if isinstance(expert_outputs, (list, tuple)):
            controlled = []
            for i, expert_out in enumerate(expert_outputs):
                if hasattr(self.expert_sparse[i], '__call__'):
                    # For classification output, we can't directly apply sparse control
                    # Instead, apply to the hidden features before classification
                    controlled.append(expert_out)
                    stats[f'expert_{i}'] = {}
            return type(expert_outputs)(controlled), stats
        else:
            return expert_outputs, stats


class TemporalSparseControl(nn.Module):
    """
    Sparse control for AU Decoder temporal features.
    Applies sparse control across the temporal dimension (T=16, AU=28).
    """

    def __init__(self, temporal_dim=16, au_dim=28):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.au_dim = au_dim

        # Sparse control for AU intensities across time
        self.au_sparse = LongTermMemorySparseControl({
            'dim': temporal_dim * au_dim,  # 16 * 28 = 448
            'inactivity_threshold': 150,  # Smaller threshold for temporal
            'hard_freeze_threshold': 400,
            'soft_decay_factor': 0.9,
            'growth_factor_boost': 1.8,
            'growth_recovery_steps': 25,
            'min_activity_to_track': 0.01,
            'enable_dual_path': True,
        })

    def forward(self, au_intensities):
        """
        Args:
            au_intensities: (B, T, 28) AU intensities

        Returns:
            controlled: (B, T, 28) with sparse control applied
            stats: statistics
        """
        B, T, num_aus = au_intensities.shape

        # Reshape for sparse control
        au_flat = au_intensities.view(B, -1)  # (B, T*28)

        # Apply sparse control
        controlled_flat, stats = self.au_sparse(au_flat)

        # Reshape back
        controlled = controlled_flat.view(B, T, num_aus)

        return controlled, stats

    def get_all_stats(self):
        """Get statistics for all controllers"""
        return {
            name: controller.get_sparse_stats()
            for name, controller in self.sparse_controllers.items()
        }


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