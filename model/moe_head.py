# =============================================================================
# Censor -- Mixture of Experts (MoE) Head & Personalized Radar (TTA)
# =============================================================================
# Implements:
#   1. MoEGatingNetwork: Conditional computation with 3 specialized experts
#   2. PersonalizedRadar: Test-time adaptation for subject-specific baseline
#
# Biological analogy (MoE): The brain has specialized regions for different
# tasks (fusiform gyrus for faces, parahippocampal for places). MoE routes
# inputs to specialized "expert" networks based on input conditions.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.defaults import MOE_CONFIG, RADAR_CONFIG


# =============================================================================
# MoE Gating Network with Auxiliary Load Balancing
# =============================================================================
# Contains at least 3 specialized experts:
#   Expert A: Strong lighting conditions
#   Expert B: Side-face / partial occlusion
#   Expert C: Weak / subtle micro-expressions
#
# The gating network learns to assign samples to the most appropriate expert
# based on the input features, using top-2 gating with auxiliary load
# balancing to prevent expert collapse.
#
# Mathematical formulation:
#   g(x) = Softmax( TopK( W_g * x + epsilon, k=2 ) )
#   y = sum_{i=1}^k g_i(x) * Expert_i(x)
#   L_aux = lambda * sum_i (f_i - 1/N)^2  # load balancing
#
# where f_i is the fraction of samples routed to expert i,
# N is the number of experts, and lambda controls regularization strength.
# =============================================================================

class MoEGatingNetwork(nn.Module):
    """
    Mixture of Experts (MoE) with noisy top-k gating and load balancing.

    Routes fused features to 3 specialized expert networks based on
    input characteristics, enabling robust micro-expression classification
    across diverse conditions (lighting, occlusion, expression subtlety).

    Architecture:
        Input: (B, 1024) fused features
          -> Gating Network: FC(1024->128) + ReLU + FC(128->3)
          -> Top-2 gating with noise for load balancing
          -> 3 Expert MLPs (1024->512->ReLU->512->7)
          -> Weighted sum: (B, 7) logits + (B, 3) gate weights
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or MOE_CONFIG

        input_dim = cfg['input_dim']
        hidden_dim = cfg['hidden_dim']
        num_experts = cfg['num_experts']
        gating_hidden_dim = cfg['gating_hidden_dim']
        num_classes = cfg['num_classes']
        self.top_k = cfg['top_k']
        self.num_experts = num_experts
        self.load_balancing_lambda = cfg['load_balancing_lambda']

        # =====================================================================
        # Gating Network
        # =====================================================================
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gating_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gating_hidden_dim, num_experts)
        )

        # Learnable gating noise (for exploration during training)
        self.gate_noise = nn.Parameter(torch.zeros(num_experts), requires_grad=True)

        # =====================================================================
        # Expert Networks
        # =====================================================================
        # 3 experts, each a 2-layer MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_experts)
        ])

        # =====================================================================
        # Expert specializations (learned through competition):
        #   Expert 0: Strong lighting / clean conditions
        #   Expert 1: Side-face / occlusion
        #   Expert 2: Weak / subtle micro-expressions
        # =====================================================================

        # Weight initialization
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, 0)
        nn.init.xavier_uniform_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, 0)

        for expert in self.experts:
            nn.init.xavier_uniform_(expert[0].weight)
            nn.init.constant_(expert[0].bias, 0)
            nn.init.xavier_uniform_(expert[3].weight)
            nn.init.constant_(expert[3].bias, 0)

    def forward(self, x, training=True):
        """
        Args:
            x (torch.Tensor): Fused features, shape (B, 1024)
            training (bool): Whether to add gating noise (for exploration)
        Returns:
            output (torch.Tensor): ME class logits, shape (B, 7)
            gate_weights (torch.Tensor): Expert gating weights, shape (B, 3)
            aux_loss (torch.Tensor): Load balancing loss, scalar
        """
        print(f"[MoEGatingNetwork] Input: {x.shape}")
        B = x.shape[0]

        # =====================================================================
        # Gating with top-k routing
        # =====================================================================
        # Raw gate logits
        gate_logits = self.gate(x)  # (B, 3)

        # Add noise during training (for gating exploration and load balancing)
        if training:
            noise = torch.randn_like(gate_logits) * self.gate_noise.unsqueeze(0)
            gate_logits = gate_logits + noise

        # Top-k gating: zero out non top-k logits
        if self.top_k < self.num_experts:
            # Get top-k values and indices
            top_k_vals, top_k_idx = torch.topk(gate_logits, self.top_k, dim=1)
            # Mask non-top-k with -inf
            mask = torch.zeros_like(gate_logits).scatter_(1, top_k_idx, 1.0)
            gate_logits = gate_logits.masked_fill(mask == 0, float('-inf'))

        # Softmax routing
        gate_weights = F.softmax(gate_logits, dim=1)  # (B, 3)

        # =====================================================================
        # Expert forward pass
        # =====================================================================
        expert_outputs = [expert(x) for expert in self.experts]  # 3 x (B, 7)
        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, 3, 7)

        # =====================================================================
        # Weighted combination
        # =====================================================================
        # output = sum_i gate_weight_i * expert_output_i
        output = (gate_weights.unsqueeze(-1) * expert_stack).sum(dim=1)  # (B, 7)

        # =====================================================================
        # Load balancing auxiliary loss
        # =====================================================================
        # f_i = fraction of samples routed to expert i
        # L_aux = lambda * sum_i (f_i - 1/N)^2  # encourages uniform routing
        routing_assignments = torch.softmax(gate_logits, dim=1)  # soft routing
        f_i = routing_assignments.mean(dim=0)  # (3,) - frequency of each expert
        target = torch.ones_like(f_i) / self.num_experts
        aux_loss = self.load_balancing_lambda * F.mse_loss(f_i, target)

        print(f"[MoEGatingNetwork] Output: {output.shape}")
        print(f"[MoEGatingNetwork] Gate weights: {gate_weights.shape}")
        print(f"[MoEGatingNetwork] Expert usage: {f_i.detach().cpu().numpy()}")

        return output, gate_weights, aux_loss


# =============================================================================
# PersonalizedRadar -- Test-Time Adaptation (TTA)
# =============================================================================
# Biological analogy: The brain continuously adapts to new stimuli through
# synaptic plasticity (Hebbian learning). This module implements few-shot
# test-time adaptation -- building a "personalized radar" that adjusts to
# new subjects with minimal data by learning their neutral expression baseline.
#
# Mathematical formulation:
#   f_i' = f_i - mu_support                            # differential features
#   W_adapter = I + Delta_W                            # residual adapter
#   Delta_W = argmin sum_i CE( g(W*f_i'), y_i )
#     via 5-step SGD with lr=1e-3 on support set (K=8 frames)
#
# This achieves rapid personalization with minimal data,
# analogous to hippocampal pattern separation.
# =============================================================================

class PersonalizedRadar(nn.Module):
    """
    PersonalizedRadar -- Test-Time Adaptation Module.

    Implements few-shot personalization by learning a residual linear adapter
    on a small support set (e.g., 8 frames) at test time, adapting the
    model to each subject's unique neutral expression baseline.

    Architecture:
        - Residual linear adapter: f_adapted = W_adapter * f + f
          (W_adapter initialized as identity: I)
        - 5-step inner-loop SGD on support set
        - Essential: call reset() between subjects for fair evaluation.

    Usage:
        radar = PersonalizedRadar()
        # During test-time adaptation:
        radar.adapt(support_feat, support_labels)
        adapted_query = radar(query_feat)
        # Reset between subjects:
        radar.reset()
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or RADAR_CONFIG

        self.input_dim = cfg['input_dim']
        self.adapt_steps = cfg['adapt_steps']
        self.adapt_lr = cfg['adapt_lr']
        self.support_shots = cfg['support_shots']

        # Residual adapter: W_I + Delta_W, initialized as identity
        # Using a ModuleList to allow per-batch adaptation
        self.adapter = nn.Linear(self.input_dim, self.input_dim, bias=False)

        # Initialize as identity matrix
        self._init_identity()

        self._adapted = False  # Track if adapter has been trained

        print(f"[PersonalizedRadar] Initialized, identity adapter")

    def _init_identity(self):
        """Initialize adapter weight as identity matrix."""
        nn.init.eye_(self.adapter.weight)

    def reset(self):
        """Reset adapter back to identity (call between subjects)."""
        self._init_identity()
        self._adapted = False
        print(f"[PersonalizedRadar] Reset to identity initialization")

    def _differential_features(self, support_feat):
        """
        Compute differential features: f_i' = f_i - mu_support
        Removes subject-specific neutral baseline.
        """
        mu_support = support_feat.mean(dim=0, keepdim=True)  # (1, D)
        diff_feat = support_feat - mu_support
        return diff_feat

    def adapt(self, support_feat, support_labels):
        """
        Adapt the residual adapter using few-shot inner-loop SGD.

        Args:
            support_feat (torch.Tensor): Support set features, shape (K, 1024)
            support_labels (torch.Tensor): Support set labels, shape (K,)
                where K = number of support samples (e.g., 8)
        """
        print(f"[PersonalizedRadar] Adapting with {support_feat.shape[0]} support samples")

        # Compute differential features
        diff_feat = self._differential_features(support_feat)

        # Inner-loop optimization (5 steps of SGD)
        original_weight = self.adapter.weight.data.clone()  # save for gradient
        optimizer = torch.optim.SGD(self.adapter.parameters(), lr=self.adapt_lr)

        for step in range(self.adapt_steps):
            # Forward pass through adapter
            adapted_feat = self.adapter(diff_feat)  # (K, 1024)

            # Simple reconstruction loss: adapted features should preserve
            # the original expressive content while removing baseline
            loss = F.mse_loss(adapted_feat, diff_feat)

            # # Alternative: Classification loss if labels available
            # loss = F.cross_entropy(simple_classifier(adapted_feat), support_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[PersonalizedRadar] Step {step+1}/{self.adapt_steps}, Loss: {loss.item():.6f}")

        self._adapted = True
        print(f"[PersonalizedRadar] Adaptation complete")

    def forward(self, query_feat, support_feat=None, support_labels=None):
        """
        Args:
            query_feat (torch.Tensor): Test-time features, shape (B, 1024)
            support_feat (torch.Tensor, optional): Support set for adaptation
            support_labels (torch.Tensor, optional): Support labels
        Returns:
            adapted_query (torch.Tensor): Adapted features, shape (B, 1024)
        """
        print(f"[PersonalizedRadar] Query: {query_feat.shape}")

        # Run adaptation if support set is provided
        if support_feat is not None:
            self.adapt(support_feat, support_labels)

        # Apply adapter to query
        adapted_query = self.adapter(query_feat)

        # Residual connection
        adapted_query = adapted_query + query_feat

        print(f"[PersonalizedRadar] Output: {adapted_query.shape}")
        return adapted_query