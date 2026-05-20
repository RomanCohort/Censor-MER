# =============================================================================
# Censor -- Hierarchical Dynamic MoE (HieDyMoE)
# =============================================================================
# Combines:
#   B. Hierarchical: Coarse (group) → Fine (category)
#   C. Dynamic: Input-conditional expert selection
#
# Architecture:
#   Level-1: 3 coarse groups (Positive / Negative / Surprise)
#   Level-2: 2-4 fine experts per group
#   Dynamic Router: Input-conditional sub-expert selection
#
# Biological analogy:
#   - Coarse: Limbic system (emotion category)
#   - Fine: Cortical regions (specific expression patterns)
#   - Dynamic: Attention (what to pay attention to)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.defaults import MOE_CONFIG


# =============================================================================
# Expert Groups Definition
# =============================================================================
# Level-1: Coarse groups (3)
# Level-2: Fine experts per group (2-4)
# =============================================================================

# Coarse group definitions
COARSE_GROUPS = {
    0: {'name': 'Positive', 'categories': [0, 6]},       # Happiness, Contempt
    1: {'name': 'Negative', 'categories': [1, 2, 3, 4, 5]},  # Sadness, Fear, Anger, Disgust
    2: {'name': 'Surprise', 'categories': [2]},              # Surprise (alone)
}

# Fine expert definitions per group
FINE_EXPERTS = {
    0: [  # Positive group
        {'id': 'pos_strong', 'name': 'Happiness (Strong)', 'category': 0, 'intensity': 'high'},
        {'id': 'pos_weak', 'name': 'Happiness (Weak)', 'category': 0, 'intensity': 'low'},
        {'id': 'contempt', 'name': 'Contempt', 'category': 6, 'intensity': 'any'},
    ],
    1: [  # Negative group
        {'id': 'sadness', 'name': 'Sadness', 'category': 1, 'intensity': 'any'},
        {'id': 'fear', 'name': 'Fear', 'category': 2, 'intensity': 'any'},
        {'id': 'anger', 'name': 'Anger', 'category': 4, 'intensity': 'any'},
        {'id': 'disgust', 'name': 'Disgust', 'category': 5, 'intensity': 'any'},
    ],
    2: [  # Surprise group
        {'id': 'surprise_strong', 'name': 'Surprise (Strong)', 'category': 2, 'intensity': 'high'},
        {'id': 'surprise_weak', 'name': 'Surprise (Weak)', 'category': 2, 'intensity': 'low'},
    ],
}

NUM_FINE_EXPERTS = {g: len(FINE_EXPERTS[g]) for g in FINE_EXPERTS}


# =============================================================================
# Input Condition Encoder
# =============================================================================
# Encodes input features to determine routing conditions:
#   -光照condition: Bright / Normal / Dark
#   -遮挡condition: Clear / Partial / Occluded
#   -运动magnitude: Subtle / Moderate / Strong
# =============================================================================

class InputConditionEncoder(nn.Module):
    """
    Encodes input features to determine dynamic routing conditions.

    Extracts:
      - illumination: brightness level
      - occlusion: face visibility
      - motion_magnitude: expression intensity
    """

    def __init__(self, input_dim=1024, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Condition extraction layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # 3 conditions
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features, shape (B, input_dim)
        Returns:
            conditions (torch.Tensor): Softmax probabilities, shape (B, 3)
                - [:, 0]: illumination (0=bright, 1=normal, 2=dark)
                - [:, 1]: occlusion (0=clear, 1=partial, 2=occluded)
                - [:, 2]: motion (0=subtle, 1=moderate, 2=strong)
        """
        logits = self.encoder(x)  # (B, 3)
        return F.softmax(logits, dim=-1)


# =============================================================================
# Hierarchical Dynamic MoE
# =============================================================================

class HierarchicalDynamicMoE(nn.Module):
    """
    Hierarchical Dynamic MoE (HieDyMoE).

    Two-level architecture with dynamic routing:
      Level-1: 3 coarse experts (emotion groups)
      Level-2: Fine experts per group (2-4)
      Dynamic: Input-conditional sub-expert selection

    Architecture:
        Input (B, 1024)
          → Condition Encoder → routing_conditions
          → Level-1 Gate (coarse groups)
          → Level-2 Dynamic Router (fine experts)
          → Expert computation
          → Weighted output

    Advantages over flat MoE:
      - Hierarchical allows coarse→fine categorization
      - Dynamic routing adapts to input conditions
      - More specialized experts (8 total vs 3)
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or MOE_CONFIG

        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.num_classes = cfg['num_classes']
        self.top_k = cfg['top_k']

        self.num_coarse_groups = 3  # Positive, Negative, Surprise
        self.load_balancing_lambda = cfg['load_balancing_lambda']
        self.use_dynamic_routing = cfg.get('use_dynamic_routing', True)

        # =====================================================================
        # Level-1: Coarse Gate
        # =====================================================================
        self.coarse_gate = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_coarse_groups)
        )

        # =====================================================================
        # Condition Encoder (for dynamic routing)
        # =====================================================================
        if self.use_dynamic_routing:
            self.condition_encoder = InputConditionEncoder(
                self.input_dim,
                hidden_dim=cfg.get('condition_hidden_dim', 64)
            )

        # =====================================================================
        # Level-2: Fine Experts per Group
        # =====================================================================
        # Total: 3 + 4 + 2 = 9 experts
        self.num_fine_experts = sum(NUM_FINE_EXPERTS.values())

        self.fine_experts = nn.ModuleDict()
        for group_id, experts in FINE_EXPERTS.items():
            group_experts = nn.ModuleList()
            for _ in experts:
                group_experts.append(nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim, self.num_classes)
                ))
            self.fine_experts[f'group_{group_id}'] = group_experts

        # =====================================================================
        # Level-2 Dynamic Router per group
        # =====================================================================
        if self.use_dynamic_routing:
            self.dynamic_routers = nn.ModuleDict()
            for group_id, num_experts in NUM_FINE_EXPERTS.items():
                self.dynamic_routers[f'group_{group_id}'] = nn.Sequential(
                    nn.Linear(self.input_dim + 3, 32),  # +3 for condition
                    nn.ReLU(inplace=True),
                    nn.Linear(32, num_experts)
                )

        # Expert group mapping: which group each fine expert belongs to
        self._build_expert_group_map()

        # Weight initialization
        self._init_weights()

    def _build_expert_group_map(self):
        """Build mapping from fine expert index to coarse group."""
        self.expert_to_group = []
        self.group_to_expert_offset = []
        offset = 0
        for group_id in range(self.num_coarse_groups):
            self.group_to_expert_offset.append(offset)
            num = NUM_FINE_EXPERTS[group_id]
            self.expert_to_group.extend([group_id] * num)
            offset += num
        self.expert_to_group = torch.tensor(self.expert_to_group)

    def _init_weights(self):
        """Initialize weights."""
        # Coarse gate
        for layer in self.coarse_gate:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # Dynamic routers
        if self.use_dynamic_routing:
            for router in self.dynamic_routers.values():
                for layer in router:
                    if hasattr(layer, 'weight'):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0)

    def _get_coarse_weights(self, x):
        """Level-1 coarse gating."""
        coarse_logits = self.coarse_gate(x)  # (B, 3)
        return F.softmax(coarse_logits, dim=-1)

    def _get_fine_weights(self, x, conditions, group_id):
        """Level-2 fine gating with dynamic routing."""
        if not self.use_dynamic_routing:
            # Uniform routing within group
            num = NUM_FINE_EXPERTS[group_id]
            return torch.ones(x.size(0), num, device=x.device) / num

        # Concatenate input and conditions
        x_cond = torch.cat([x, conditions], dim=-1)  # (B, 1024 + 3)

        router = self.dynamic_routers[f'group_{group_id}']
        fine_logits = router(x_cond)  # (B, num_experts_in_group)

        # Top-k within group
        num_experts = NUM_FINE_EXPERTS[group_id]
        if self.top_k < num_experts:
            top_k_vals, top_k_idx = torch.topk(fine_logits, self.top_k, dim=1)
            mask = torch.zeros_like(fine_logits).scatter_(1, top_k_idx, 1.0)
            fine_logits = fine_logits.masked_fill(mask == 0, float('-inf'))

        return F.softmax(fine_logits, dim=-1)

    def forward(self, x, return_hierarchy=False):
        """
        Args:
            x (torch.Tensor): Input features, shape (B, 1024)
            return_hierarchy (bool): Return detailed hierarchy info
        Returns:
            output (torch.Tensor): Class logits, shape (B, 7)
            hierarchy_info (dict, optional): Detailed routing info
        """
        print(f"[HierarchicalDynamicMoE] Input: {x.shape}")
        B = x.shape[0]

        # =====================================================================
        # Level-1: Coarse Gating
        # =====================================================================
        coarse_weights = self._get_coarse_weights(x)  # (B, 3)
        print(f"[HierarchicalDynamicMoE] Coarse weights: {coarse_weights[0].detach().cpu().numpy()[:3]}")

        # Determine primary group (argmax)
        primary_group = coarse_weights.argmax(dim=1)  # (B,)

        # =====================================================================
        # Dynamic Conditions (for Level-2 routing)
        # =====================================================================
        if self.use_dynamic_routing:
            conditions = self.condition_encoder(x)  # (B, 3)
            print(f"[HierarchicalDynamicMoE] Conditions: {conditions[0].detach().cpu().numpy()}")
        else:
            conditions = None

        # =====================================================================
        # Level-2: Fine Expert Computation (per group)
        # =====================================================================
        # Compute fine outputs for each group, weight by coarse weight
        output = torch.zeros(B, self.num_classes, device=x.device)

        for group_id in range(self.num_coarse_groups):
            group_mask = (primary_group == group_id)
            if group_mask.sum() == 0:
                continue

            group_indices = group_mask.nonzero(as_tuple=True)[0]
            x_group = x[group_indices]  # (G, 1024)

            # Fine weights for this group
            if self.use_dynamic_routing and conditions is not None:
                conditions_group = conditions[group_indices]
                fine_weights = self._get_fine_weights(x_group, conditions_group, group_id)
            else:
                num = NUM_FINE_EXPERTS[group_id]
                fine_weights = torch.ones(x_group.size(0), num, device=x.device) / num

            # Fine expert outputs
            fine_experts = self.fine_experts[f'group_{group_id}']
            expert_outputs = []
            for expert in fine_experts:
                expert_outputs.append(expert(x_group))
            expert_stack = torch.stack(expert_outputs, dim=1)  # (G, num_fine, 7)

            # Weighted sum within group
            group_output = (fine_weights.unsqueeze(-1) * expert_stack).sum(dim=1)

            # Weight by coarse gate
            coarse_weight = coarse_weights[group_indices, group_id].view(-1, 1)
            output[group_indices] = group_output * coarse_weight

        # =====================================================================
        # Load Balancing Auxiliary Loss
        # =====================================================================
        # For simplicity, use coarse routing frequency
        routing_assignments = coarse_weights.argmax(dim=1)  # (B,)
        f_i = torch.zeros(self.num_coarse_groups, device=x.device)
        for g in range(self.num_coarse_groups):
            f_i[g] = (routing_assignments == g).float().mean()
        target = torch.ones_like(f_i) / self.num_coarse_groups
        aux_loss = self.load_balancing_lambda * F.mse_loss(f_i, target)

        print(f"[HierarchicalDynamicMoE] Output: {output.shape}")
        print(f"[HierarchicalDynamicMoE] Expert usage: {f_i.detach().cpu().numpy()}")

        if return_hierarchy:
            hierarchy_info = {
                'coarse_weights': coarse_weights,
                'primary_group': primary_group,
                'conditions': conditions if self.use_dynamic_routing else None,
                'aux_loss': aux_loss,
            }
            return output, hierarchy_info, aux_loss

        return output, None, aux_loss


# =============================================================================
# HierarchicalDynamicMoE with Load Balancing Loss
# =============================================================================

class HierarchicalDynamicMoEWithLoss(HierarchicalDynamicMoE):
    """
    Enhanced version with proper load balancing for both levels.

    Adds:
      - Fine-level load balancing loss
      - Expert utilization tracking
      - Temperature scaling for gating
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, return_hierarchy=False):
        """Forward with enhanced load balancing."""
        print(f"[HierarchicalDynamicMoEWithLoss] Input: {x.shape}")
        B = x.shape[0]

        # Temperature-scaled coarse gating
        coarse_logits = self.coarse_gate(x) / self.temperature.clamp(min=0.1)
        coarse_weights = F.softmax(coarse_logits, dim=-1)
        primary_group = coarse_weights.argmax(dim=1)

        # Conditions
        if self.use_dynamic_routing:
            conditions = self.condition_encoder(x)
        else:
            conditions = None

        # Fine computation
        output = torch.zeros(B, self.num_classes, device=x.device)

        for group_id in range(self.num_coarse_groups):
            group_mask = (primary_group == group_id)
            if group_mask.sum() == 0:
                continue

            group_indices = group_mask.nonzero(as_tuple=True)[0]
            x_group = x[group_indices]

            fine_weights = self._get_fine_weights(
                x_group,
                conditions[group_indices] if conditions is not None else None,
                group_id
            )

            fine_experts = self.fine_experts[f'group_{group_id}']
            expert_outputs = [expert(x_group) for expert in fine_experts]
            expert_stack = torch.stack(expert_outputs, dim=1)

            group_output = (fine_weights.unsqueeze(-1) * expert_stack).sum(dim=1)
            coarse_weight = coarse_weights[group_indices, group_id].view(-1, 1)
            output[group_indices] = group_output * coarse_weight

        # Combined load balancing loss
        # Coarse level
        routing_assignments = coarse_weights.argmax(dim=1)
        f_coarse = torch.zeros(self.num_coarse_groups, device=x.device)
        for g in range(self.num_coarse_groups):
            f_coarse[g] = (routing_assignments == g).float().mean()
        target_coarse = torch.ones_like(f_coarse) / self.num_coarse_groups
        loss_coarse = F.mse_loss(f_coarse, target_coarse)

        # Fine level per group
        loss_fine = 0
        if conditions is not None:
            for group_id in range(self.num_coarse_groups):
                # Get fine routing frequencies
                fine_logits = self.dynamic_routers[f'group_{group_id}'](
                    torch.cat([x, conditions], dim=-1)
                )
                f_fine = F.softmax(fine_logits, dim=0).mean(dim=0)
                target_fine = torch.ones_like(f_fine) / NUM_FINE_EXPERTS[group_id]
                loss_fine = loss_fine + F.mse_loss(f_fine, target_fine)

        aux_loss = self.load_balancing_lambda * (loss_coarse + 0.5 * loss_fine)

        if return_hierarchy:
            return output, {'coarse_weights': coarse_weights}, aux_loss
        return output, None, aux_loss


# =============================================================================
# Lightweight Variant (for mobile/embedded)
# =============================================================================

class HierarchicalDynamicMoELite(nn.Module):
    """
    Lightweight variant with shared experts and no dynamic routing.

    Trade-offs:
      - Fewer parameters
      - No dynamic routing
      - Still maintains hierarchical structure
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or MOE_CONFIG

        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.num_classes = cfg['num_classes']
        self.num_coarse_groups = 3

        # Single coarse gate
        self.coarse_gate = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_coarse_groups)
        )

        # Shared experts (one per group, simpler)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.num_classes)
            )
            for _ in range(self.num_coarse_groups)
        ])

    def forward(self, x):
        """Forward pass."""
        coarse_weights = F.softmax(self.coarse_gate(x), dim=-1)

        expert_outputs = [expert(x) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)

        output = (coarse_weights.unsqueeze(-1) * expert_stack).sum(dim=1)

        return output, coarse_weights, torch.tensor(0.0)


def create_hierarchical_dynamic_moe(config=None, lite=False):
    """
    Factory function to create HieDyMoE.

    Args:
        config: MoE config dict
        lite: Use lite variant
    """
    if lite:
        return HierarchicalDynamicMoELite(config)
    return HierarchicalDynamicMoE(config)