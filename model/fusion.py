# =============================================================================
# Censor -- TSFmicro: Temporal-Spatial Feature Micro-Fusion
# =============================================================================
# Implements bidirectional cross-attention between Fast and Slow pathways.
# Biological analogy: Cross-talk between ventral ("what") and dorsal ("where")
# pathways in the brain, mediated by reciprocal connections in area STP
# (superior temporal polysensory area).
#
# Mathematical formulation:
#   f_fast' = CrossAttn(Q=W_q*f_fast, K=W_k*f_slow, V=W_v*f_slow)
#   f_slow' = CrossAttn(Q=W_q'*f_slow, K=W_k'*f_fast, V=W_v'*f_fast)
#   g = sigma( W_g * [f_fast'; f_slow'] )
#   f_fused = g * f_fast' + (1-g) * f_slow'
# =============================================================================
#
# This fusion resolves the contradiction between local muscle contraction
# (captured by Fast pathway's high-temporal-resolution flow) and global
# dynamic changes (captured by Slow pathway's high-semantic features).

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.defaults import FUSION_CONFIG


class TSFmicroFusion(nn.Module):
    """
    TSFmicroFusion -- Temporal-Spatial Feature micro-Fusion via Cross-Attention.

    Performs bidirectional cross-attention between Fast and Slow pathway
    features in a high-dimensional (1024-D) space, with a learnable fusion
    gate to balance both pathways' contributions.

    Architecture:
        Input: fast (B, 512), slow (B, 768)
          -> Fast queries Slow (bidirectional cross-attention)
          -> Slow queries Fast (bidirectional cross-attention)
          -> Fusion gate: weighted combination
          -> Output: (B, 1024)
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or FUSION_CONFIG

        fast_dim = cfg['fast_dim']
        slow_dim = cfg['slow_dim']
        fused_dim = cfg['fused_dim']
        num_heads = cfg.get('num_heads', 8)

        # =====================================================================
        # Cross-Attention: Fast -> Slow
        # Fast queries, Slow provides keys and values
        # This allows Fast to "ask" Slow for relevant semantic context
        # =====================================================================
        self.fast_q = nn.Linear(fast_dim, fused_dim)
        self.slow_k = nn.Linear(slow_dim, fused_dim)
        self.slow_v = nn.Linear(slow_dim, fused_dim)

        # =====================================================================
        # Cross-Attention: Slow -> Fast
        # Slow queries, Fast provides keys and values
        # This allows Slow to "ask" Fast for relevant motion cues
        # =====================================================================
        self.slow_q = nn.Linear(slow_dim, fused_dim)
        self.fast_k = nn.Linear(fast_dim, fused_dim)
        self.fast_v = nn.Linear(fast_dim, fused_dim)

        # =====================================================================
        # Fusion Gate
        # =====================================================================
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid()
        )

        # =====================================================================
        # Output projection
        # =====================================================================
        self.output_proj = nn.Linear(fused_dim, fused_dim)
        self.output_norm = nn.LayerNorm(fused_dim)

        # Weight initialization
        for module in [self.fast_q, self.fast_k, self.fast_v, self.slow_q, self.slow_k, self.slow_v]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.fusion_gate[0].weight)
        nn.init.constant_(self.fusion_gate[0].bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

        self.scale = math.sqrt(fused_dim)

    def _cross_attention(self, q_feat, q_proj, kv_feat, k_proj, v_proj):
        """
        Perform single-direction cross-attention.

        Args:
            q_feat (torch.Tensor): Query features, (B, D_q)
            q_proj (nn.Linear): Query projection to fused dim
            kv_feat (torch.Tensor): Key/Value features, (B, D_kv)
            k_proj (nn.Linear): Key projection
            v_proj (nn.Linear): Value projection
        Returns:
            attended (torch.Tensor): Cross-attended features, (B, fused_dim)
        """
        Q = q_proj(q_feat).unsqueeze(1)  # (B, 1, fused_dim)
        K = k_proj(kv_feat).unsqueeze(1)  # (B, 1, fused_dim)
        V = v_proj(kv_feat).unsqueeze(1)  # (B, 1, fused_dim)

        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale  # (B, 1, 1)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attended = attn_weights @ V  # (B, 1, fused_dim)
        return attended.squeeze(1)  # (B, fused_dim)

    def forward(self, fast_feat, slow_feat):
        """
        Args:
            fast_feat (torch.Tensor): Fast pathway features, shape (B, 512)
            slow_feat (torch.Tensor): Slow pathway features, shape (B, 768)
        Returns:
            fused (torch.Tensor): Fused features, shape (B, 1024)
        """
        print(f"[TSFmicroFusion] Inputs: fast={fast_feat.shape}, slow={slow_feat.shape}")

        # =====================================================================
        # Bidirectional Cross-Attention
        # =====================================================================
        # Fast queries Slow (Fast asks Slow for semantic context)
        fast_attended = self._cross_attention(
            fast_feat, self.fast_q, slow_feat, self.slow_k, self.slow_v
        )  # (B, 1024)

        # Slow queries Fast (Slow asks Fast for motion context)
        slow_attended = self._cross_attention(
            slow_feat, self.slow_q, fast_feat, self.fast_k, self.fast_v
        )  # (B, 1024)

        # =====================================================================
        # Fusion Gate
        # =====================================================================
        # Learnable gate balances Fast and Slow contributions
        gate_input = torch.cat([fast_attended, slow_attended], dim=1)  # (B, 2048)
        gate = self.fusion_gate(gate_input)  # (B, 1024)

        # Weighted combination
        fused = gate * fast_attended + (1 - gate) * slow_attended  # (B, 1024)

        # Output projection
        fused = self.output_proj(fused)
        fused = self.output_norm(fused)

        print(f"[TSFmicroFusion] Output: {fused.shape}")
        return fused