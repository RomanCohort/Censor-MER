"""
Censor -- Biomimetic Dual-Pathway Micro-Expression Recognition System
=====================================================================
Main entry point. Instantiates the full Censor model and runs a forward pass
with dummy input to verify tensor shapes and signal flow.

Usage:
    python main.py

Expected output:
    Each module prints its input/output tensor shapes.
    Final output:
        - me_logits: (2, 7) micro-expression logits
        - au_intensities: (2, 16, 28) AU intensities
        - au_opd: (2, 28, 3) onset-peak-decay landmarks
        - apex_scores: (2, T/16) apex frame scores
        - expert_gates: (2, 3) MoE gating weights
        - template_reports: list of 2 structured clinical reports
"""

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config.defaults import (
    INPUT_CONFIG,
    FAST_PATHWAY_CONFIG,
    SLOW_PATHWAY_CONFIG,
    AMYGDALA_CONFIG,
    FFA_CONFIG,
    CASA_CONFIG,
    FUSION_CONFIG,
    AU_DECODER_CONFIG,
    MOE_CONFIG,
    RADAR_CONFIG,
    SPARSE_CONTROL_CONFIG,
)

from model.preprocessing import (
    SaliencyDetector, rPPGExtractor, TVL1OpticalFlow,
    AdaptiveOpticalFlow, SaliencyDetectorE2E
)
from model.backbones import FastSubcorticalPathway, SlowCorticalPathway
from model.attention import Amygdala, FFA, CASANet, AmygdalaWithPrior, CASANetLearnable
from model.fusion import TSFmicroFusion
from model.decoders import DynamicAUDecoder
from model.moe_head import (
    MoEGatingNetwork, PersonalizedRadar,
    PersonalizedRadarEnhanced
)
from model.biomoe import BioMoE  # Add Standard MoE comparison
from model.llm_report import EmotionReporter
from model.biomimetic_enhance import LongTermMemorySparseControl, SparseControlWrapper, TemporalSparseControl


# =============================================================================
# Censor -- Main Model Orchestrator
# =============================================================================

class Censor(nn.Module):
    """
    Censor: Biomimetic Dual-Pathway Micro-Expression Recognition System.

    The complete pipeline:
        Input (B, 3, T, H, W) RGB video
          -> Preprocessing: Saliency, rPPG, TV-L1 Optical Flow
          -> Dual Pathways: Fast (3D ResNet18) + Slow (3D Swin-Transformer)
          -> Attention Modulation: Amygdala + FFA + CASANet
          -> TSFmicroFusion: Bidirectional cross-attention fusion
          -> DynamicAUDecoder: BiLSTM + 28 AUs + OPD landmarks
          -> MoE Head: 3 experts + test-time personalization
          -> EmotionReporter: Structured clinical reports
    """

    def __init__(self, fast_preprocess=False, diff_mode=False, verbose=True,
                 enable_sparse_control=False, return_features=False,
                 pretrained_backbone=False, single_path=None,
                 no_moe=False, no_amyg=False, no_ffa=False,
                 no_casa=False, no_rppg=False):
        super().__init__()

        self.fast_preprocess = fast_preprocess
        self.diff_mode = diff_mode
        self.verbose = verbose
        self.enable_sparse_control = enable_sparse_control
        self.return_features = return_features
        self.pretrained_backbone = pretrained_backbone

        # Single-path mode for ablation study
        self.single_path = single_path  # None, 'fast', or 'slow'
        if single_path:
            print(f"[Censor] ABLATION: Single-path mode ({single_path})")

        # No amygdala gating for ablation study
        self.no_amyg = no_amyg
        if no_amyg:
            print("[Censor] ABLATION: No amygdala attention gating")

        # No FFA for ablation study
        self.no_ffa = no_ffa
        if no_ffa:
            print("[Censor] ABLATION: No FFA fusion")

        # No CASANet for ablation study
        self.no_casa = no_casa
        if no_casa:
            print("[Censor] ABLATION: No CASANet")

        # No rPPG for ablation study
        self.no_rppg = no_rppg
        if no_rppg:
            print("[Censor] ABLATION: No rPPG signal")

        # =====================================================================
        # Stage 1: Biomimetic Preprocessing
        # =====================================================================
        print("[Censor] Initializing Preprocessing...")
        self.saliency = SaliencyDetector()
        if not self.no_rppg:
            self.rppg = rPPGExtractor()
        else:
            self.rppg = None
        if not fast_preprocess:
            self.flow = TVL1OpticalFlow()
        else:
            self.flow = None
            print("[Censor] Fast preprocess mode: using frame difference instead of TV-L1 optical flow")

        # =====================================================================
        # Stage 2: Dual-Pathway Backbones
        # =====================================================================
        print("[Censor] Initializing Dual-Pathway Backbones...")
        self.fast_pathway = FastSubcorticalPathway(FAST_PATHWAY_CONFIG, pretrained=pretrained_backbone)
        self.slow_pathway = SlowCorticalPathway(SLOW_PATHWAY_CONFIG, pretrained=pretrained_backbone)

        # =====================================================================
        # Stage 3: Fusiform-Amygdala Attention Circuit
        # =====================================================================
        print("[Censor] Initializing Attention Modulation...")
        if not self.no_amyg:
            self.amygdala = Amygdala(AMYGDALA_CONFIG)
        else:
            self.amygdala = None
        if not self.no_ffa:
            self.ffa = FFA(FFA_CONFIG)
        else:
            self.ffa = None
        if not self.no_casa:
            self.casa = CASANet(CASA_CONFIG)
        else:
            self.casa = None

        # =====================================================================
        # Stage 4: Spatio-Temporal Fusion
        # =====================================================================
        print("[Censor] Initializing TSFmicroFusion...")
        self.fusion = TSFmicroFusion(FUSION_CONFIG)

        # Single-path projection layers (project to fused_dim=1024)
        if single_path == 'fast':
            self.single_path_proj = nn.Linear(512, 1024)
        elif single_path == 'slow':
            self.single_path_proj = nn.Linear(768, 1024)

        # =====================================================================
        # Stage 4.5: Long-Term Memory Sparse Control (Multi-Stage)
        # =====================================================================
        if enable_sparse_control:
            print("[Censor] Initializing Sparse Control Wrapper...")
            self.sparse_control = SparseControlWrapper({
                'fast_path': 512,          # FastPath output
                'slow_path': 768,          # SlowPath output
                'fusion': 1024,           # Fusion output
                'moe_coarse': 3,          # MoE coarse experts (groups)
                'moe_fine': 9,            # MoE fine experts (total)
            })
        else:
            self.sparse_control = None

        # =====================================================================
        # Stage 5: Dynamic AU Decoder
        # =====================================================================
        print("[Censor] Initializing AU Decoder...")
        self.au_decoder = DynamicAUDecoder(AU_DECODER_CONFIG)

        # =====================================================================
        # Stage 6: MoE Head & Personalized Radar (or Simple Linear for ablation)
        # =====================================================================
        self.no_moe = no_moe
        if no_moe:
            print("[Censor] ABLATION: No MoE — using simple linear head")
            self.simple_head = nn.Linear(1024, MOE_CONFIG['num_classes'])
        else:
            print("[Censor] Initializing MoE Head...")
            self.moe = MoEGatingNetwork(MOE_CONFIG)
        self.radar = PersonalizedRadar(RADAR_CONFIG)

        # =====================================================================
        # Stage 7: Emotion Reporter
        # =====================================================================
        print("[Censor] Initializing Emotion Reporter...")
        self.reporter = EmotionReporter()

        print("[Censor] Model initialized successfully!\n")

    def forward(self, x):
        """
        Full forward pass of the Censor model.

        Args:
            x (torch.Tensor): Raw RGB video, shape (B, C=3, T=16, H=224, W=224)

        Returns:
            dict with keys:
                - 'me_logits': (B, 7) micro-expression logits
                - 'au_intensities': (B, T, 28) AU intensities per frame
                - 'au_opd': (B, 28, 3) onset-peak-decay landmarks
                - 'apex_scores': (B, T_apa) apex frame scores
                - 'expert_gates': (B, 3) MoE gating weights
                - 'adapted_feat': (B, 1024) personalized features
                - 'template_report': list[str] structured reports
                - 'llm_report': list[str] free-text reports (placeholder)
        """
        # Training mode: suppress verbose prints
        if self.verbose:
            if self.verbose: print(f"\n{'='*60}")
            if self.verbose: print(f" Censor Forward Pass")
            if self.verbose: print(f"{'='*60}")
            if self.verbose: print(f"Input video: {x.shape}")

        B, C, T, H, W = x.shape

        # =====================================================================
        # Stage 1: Preprocessing
        # =====================================================================
        if self.verbose:
            if self.verbose: print(f"\n--- Stage 1: Preprocessing ---")

        # 1a) Saliency detection (fovea simulation)
        # Output: (B, 1, T, H, W) spatial prior map
        saliency_map = self.saliency(x)
        # Apply saliency modulation: weighted input
        x_salient = x * (1.0 + 0.3 * saliency_map.expand(-1, C, -1, -1, -1))

        # 1b) Lateral inhibition: onset-apex difference (replaces rPPG in diff_mode)
        # In biology, retinal ganglion cells use lateral inhibition to enhance
        # edges and suppress uniform regions. The onset-apex difference encodes
        # the micro-expression change directly -- far more informative than rPPG
        # for small datasets where rPPG estimation is unreliable.
        if self.no_rppg:
            # No rPPG: use zeros (rPPG signal disabled)
            rppg_heatmap = torch.zeros_like(x_salient)
            if self.verbose: print(f"[Censor] No rPPG: using zeros")
        elif self.diff_mode:
            # Onset frame (first frame = neutral baseline)
            onset_frame = x_salient[:, :, 0:1, :, :]  # (B, C, 1, H, W)
            # Apex frame (middle frame = peak expression)
            apex_idx = T // 2
            apex_frame = x_salient[:, :, apex_idx:apex_idx+1, :, :]  # (B, C, 1, H, W)
            # Lateral inhibition: diff = apex - onset (per-pixel change)
            onset_apex_diff = apex_frame - onset_frame  # (B, C, 1, H, W)
            # Replicate across all T frames (same spatial diff, broadcast temporally)
            onset_apex_diff = onset_apex_diff.expand(-1, -1, T, -1, -1)  # (B, C, T, H, W)
            # Use diff as the "change signal" instead of rPPG
            rppg_heatmap = onset_apex_diff
            if self.verbose: print(f"[Censor] Diff mode: onset-apex diff shape: {rppg_heatmap.shape}")
        else:
            # Original: rPPG blood-flow heatmap
            rppg_heatmap = self.rppg(x_salient)

        # 1c) Optical flow: TV-L1 (slow) or frame difference (fast)
        if self.fast_preprocess:
            # Fast mode: frame difference as motion proxy (GPU, ~ms)
            # Compute temporal difference between consecutive frames
            diff = x_salient[:, :, 1:, :, :] - x_salient[:, :, :-1, :, :]
            # Split into x-like and y-like motion channels
            flow_x = diff.mean(dim=1, keepdim=True)  # (B, 1, T-1, H, W)
            flow_y = diff.std(dim=1, keepdim=True)    # (B, 1, T-1, H, W)
            # Pad last frame
            flow_x_pad = flow_x[:, :, -1:, :, :]
            flow_y_pad = flow_y[:, :, -1:, :, :]
            flow_x = torch.cat([flow_x, flow_x_pad], dim=2)  # (B, 1, T, H, W)
            flow_y = torch.cat([flow_y, flow_y_pad], dim=2)  # (B, 1, T, H, W)
            flow_stack = torch.cat([flow_x, flow_y], dim=1)  # (B, 2, T, H, W)
        else:
            # Full mode: TV-L1 optical flow (CPU, ~seconds per batch)
            flow_maps = []
            for t in range(T - 1):
                flow_t = self.flow(
                    x_salient[:, :, t],
                    x_salient[:, :, t + 1]
                )
                flow_maps.append(flow_t)
            flow_stack = torch.stack(flow_maps, dim=2)
            flow_pad = flow_stack[:, :, -1:, :, :]
            flow_stack = torch.cat([flow_stack, flow_pad], dim=2)
        if self.verbose: print(f"[Censor] Flow stack: {flow_stack.shape}")

        # =====================================================================
        # Stage 2: Dual-Pathway Backbones
        # =====================================================================
        if self.verbose: print(f"\n--- Stage 2: Dual-Pathway Backbones ---")

        # Fast Pathway (subcortical): optical flow input
        # Input: (B, 2, T, H, W) -> Output: (B, 512)
        fast_feat = self.fast_pathway(flow_stack)

        # Slow Pathway (cortical): RGB + rPPG concatenated
        # Input: (B, 6, T, H, W) -> Output: (B, 768) pooled + (B, 768, T/16, H/32, W/32) spatial
        rgb_rppg = torch.cat([x_salient, rppg_heatmap], dim=1)  # (B, 6, T, H, W)
        slow_feat, slow_spatial = self.slow_pathway(rgb_rppg)

        # =====================================================================
        # ABLATION: Single-path shortcut
        # =====================================================================
        if self.single_path:
            if self.single_path == 'fast':
                # Only FastPath: project 512 → 1024, skip fusion
                fused_feat = self.single_path_proj(fast_feat)
            else:
                # Only SlowPath: project 768 → 1024, skip fusion
                fused_feat = self.single_path_proj(slow_feat)
            # Skip to Stage 5
            apex_scores = torch.zeros(B, 1, device=x.device)
            au_intensities, au_opd = self.au_decoder(fused_feat)
            if self.no_moe:
                me_logits = self.simple_head(fused_feat)
                expert_gates = torch.zeros(B, 1, device=x.device)
                moe_aux_loss = torch.tensor(0.0, device=x.device)
            else:
                me_logits, expert_gates, moe_aux_loss = self.moe(fused_feat)
            adapted_feat = self.radar(fused_feat)
            template_reports, llm_reports = {}, {}
            return {
                'me_logits': me_logits,
                'au_intensities': au_intensities,
                'au_opd': au_opd,
                'apex_scores': apex_scores,
                'expert_gates': expert_gates,
                'moe_aux_loss': moe_aux_loss,
                'adapted_feat': adapted_feat,
                'template_report': template_reports,
                'llm_report': llm_reports,
            }

        # =====================================================================
        # Stage 2.5: Sparse Control for Pathways
        # =====================================================================
        if self.enable_sparse_control and self.sparse_control is not None:
            if self.verbose: print(f"\n--- Stage 2.5: Sparse Control (Pathways) ---")
            pathway_feats = {'fast_path': fast_feat, 'slow_path': slow_feat}
            pathway_feats, pathway_stats = self.sparse_control(pathway_feats)
            fast_feat = pathway_feats['fast_path']
            slow_feat = pathway_feats['slow_path']
            for name, stats in pathway_stats.items():
                if stats:
                    if self.verbose: print(f"[Sparse-{name}] frozen={stats.get('frozen_ratio', 0):.3f}, usage={stats.get('usage_ratio', 0):.3f}")
        else:
            pathway_stats = {}

        # =====================================================================
        # Stage 3: Attention Modulation
        # =====================================================================
        if self.verbose: print(f"\n--- Stage 3: Attention Modulation ---")

        # Amygdala: generates Attention Prior Map from fast features
        # Input: (B, 512) -> Output: (B, 1, 14, 14)
        if self.amygdala is not None:
            apm = self.amygdala(fast_feat)
        else:
            # No amygdala gating: use uniform attention map
            apm = torch.ones(B, 1, 14, 14, device=x.device)

        # FFA: mutual channel recalibration between pathways
        # Inputs: (B, 512), (B, 768) -> Outputs: (B, 512), (B, 768)
        if self.ffa is not None:
            fast_gated, slow_gated = self.ffa(fast_feat, slow_feat)
        else:
            # No FFA: use original features directly
            fast_gated, slow_gated = fast_feat, slow_feat

        # CASANet: 3D contextual attention on Slow pathway spatial map
        # Input: (B, 768, T_s, H_s, W_s) -> Output: (B, 768, T_s, H_s, W_s) + (B, T_s) apex scores
        if slow_spatial is not None and self.casa is not None:
            casa_feat, apex_scores = self.casa(slow_spatial)
            # Pool spatial map from CASANet for fusion
            casa_pooled = casa_feat.mean(dim=[-1, -2, -3])  # (B, 768)
            # Gate: blend original slow_gated with CASA-attended features
            casa_gate = torch.sigmoid(casa_pooled.mean(dim=1, keepdim=True))  # (B, 1)
            slow_for_fusion = casa_gate * casa_pooled + (1 - casa_gate) * slow_gated
        else:
            # Fallback if spatial map is None or CASANet disabled
            casa_feat = None
            apex_scores = torch.zeros(B, 1, device=x.device)
            slow_for_fusion = slow_gated

        # =====================================================================
        # Stage 4: TSFmicroFusion
        # =====================================================================
        if self.verbose: print(f"\n--- Stage 4: TSFmicroFusion ---")

        fused_feat = self.fusion(fast_gated, slow_for_fusion)  # (B, 1024)

        # =====================================================================
        # Stage 4.5: Sparse Control for Fusion
        # =====================================================================
        if self.enable_sparse_control and self.sparse_control is not None:
            if self.verbose: print(f"\n--- Stage 4.5: Sparse Control ---")
            # Apply sparse control to fusion output
            fusion_feats, fusion_stats = self.sparse_control({'fusion': fused_feat})
            fused_feat = fusion_feats['fusion']
            fusion_stat = fusion_stats.get('fusion', {})
            if fusion_stat:
                if self.verbose: print(f"[Sparse-fusion] frozen={fusion_stat.get('frozen_ratio', 0):.3f}, "
                      f"usage={fusion_stat.get('usage_ratio', 0):.3f}")
        else:
            fusion_stats = {}

        # Collect all sparse stats
        all_sparse_stats = {**pathway_stats, **fusion_stats}

        # =====================================================================
        # Stage 5: Dynamic AU Decoder
        # =====================================================================
        if self.verbose: print(f"\n--- Stage 5: AU Decoder ---")

        au_intensities, au_opd = self.au_decoder(fused_feat)  # (B, 16, 28), (B, 28, 3)

        # =====================================================================
        # Stage 6: MoE Head or Simple Linear (ablation) & Personalized Radar
        # =====================================================================
        if self.verbose: print(f"\n--- Stage 6: Classification Head ---")

        if self.no_moe:
            me_logits = self.simple_head(fused_feat)
            expert_gates = torch.zeros(B, 1, device=x.device)
            moe_aux_loss = torch.tensor(0.0, device=x.device)
        else:
            me_logits, expert_gates, moe_aux_loss = self.moe(fused_feat)

        # Personalized Radar (test-time adaptation, skip in debug forward)
        adapted_feat = self.radar(fused_feat)  # (B, 1024) - identity pass in debug

        # =====================================================================
        # Stage 7: Emotion Reporter
        # =====================================================================
        if self.verbose: print(f"\n--- Stage 7: Emotion Reporter ---")

        # Skip reporter during training (no LLM needed)
        if self.verbose:
            template_reports, llm_reports = self.reporter(fused_feat, au_intensities, me_logits)
        else:
            template_reports, llm_reports = {}, {}

        # =====================================================================
        # Final Summary
        # =====================================================================
        if self.verbose: print(f"\n{'='*60}")
        if self.verbose: print(f" Final Output Summary")
        if self.verbose: print(f"{'='*60}")
        if self.verbose: print(f"  ME Logits:       {me_logits.shape}")
        if self.verbose: print(f"  AU Intensities:  {au_intensities.shape}")
        if self.verbose: print(f"  AU OPD:          {au_opd.shape}")
        if self.verbose: print(f"  Apex Scores:     {apex_scores.shape}")
        if self.verbose: print(f"  Expert Gates:    {expert_gates.shape}")
        if self.verbose: print(f"  MoE Aux Loss:    {moe_aux_loss.item():.6f}")
        if self.verbose: print(f"  Adapted Feat:    {adapted_feat.shape}")
        if self.verbose: print(f"  Reports:         {len(template_reports)} templates")
        if self.verbose: print(f"{'='*60}\n")

        return {
            'me_logits': me_logits,
            'au_intensities': au_intensities,
            'au_opd': au_opd,
            'apex_scores': apex_scores,
            'expert_gates': expert_gates,
            'moe_aux_loss': moe_aux_loss,
            'adapted_feat': adapted_feat,
            'template_report': template_reports,
            'llm_report': llm_reports,
            'sparse_stats': all_sparse_stats,
        }

    def extract_fast_features(self, x):
        """Extract fast pathway features only (for cached feature extraction)."""
        B, C, T, H, W = x.shape

        # Saliency
        saliency_map = self.saliency(x)
        x_salient = x * (1.0 + 0.3 * saliency_map.expand(-1, C, -1, -1, -1))

        # Frame difference as motion proxy
        diff = x_salient[:, :, 1:, :, :] - x_salient[:, :, :-1, :, :]
        flow_x = diff.mean(dim=1, keepdim=True)
        flow_y = diff.std(dim=1, keepdim=True)
        flow_x_pad = flow_x[:, :, -1:, :, :]
        flow_y_pad = flow_y[:, :, -1:, :, :]
        flow_x = torch.cat([flow_x, flow_x_pad], dim=2)
        flow_y = torch.cat([flow_y, flow_y_pad], dim=2)
        flow_stack = torch.cat([flow_x, flow_y], dim=1)

        # Fast pathway
        fast_feat = self.fast_pathway(flow_stack)
        return fast_feat

    def extract_slow_features(self, x):
        """Extract slow pathway features only (for cached feature extraction)."""
        B, C, T, H, W = x.shape

        # Saliency
        saliency_map = self.saliency(x)
        x_salient = x * (1.0 + 0.3 * saliency_map.expand(-1, C, -1, -1, -1))

        # rPPG heatmap
        if self.no_rppg:
            rppg_heatmap = torch.zeros_like(x_salient)
        elif self.diff_mode:
            onset_frame = x_salient[:, :, 0:1, :, :]
            apex_idx = T // 2
            apex_frame = x_salient[:, :, apex_idx:apex_idx+1, :, :]
            onset_apex_diff = apex_frame - onset_frame
            rppg_heatmap = onset_apex_diff.expand(-1, -1, T, -1, -1)
        else:
            rppg_heatmap = self.rppg(x_salient)

        # Concatenate RGB + rPPG
        rgb_rppg = torch.cat([x_salient, rppg_heatmap], dim=1)

        # Slow pathway
        slow_feat, slow_spatial = self.slow_pathway(rgb_rppg)
        return slow_feat

    def extract_rgb_features(self, x):
        """Extract RGB-only slow pathway features (without rPPG, for ablation)."""
        B, C, T, H, W = x.shape

        # Saliency
        saliency_map = self.saliency(x)
        x_salient = x * (1.0 + 0.3 * saliency_map.expand(-1, C, -1, -1, -1))

        # Zero rPPG
        rppg_heatmap = torch.zeros_like(x_salient)

        # Concatenate RGB + zero rPPG
        rgb_rppg = torch.cat([x_salient, rppg_heatmap], dim=1)

        # Slow pathway
        slow_feat, _ = self.slow_pathway(rgb_rppg)
        return slow_feat

    def extract_rppg_features(self, x):
        """Extract rPPG-only features (for ablation)."""
        B, C, T, H, W = x.shape

        # Saliency
        saliency_map = self.saliency(x)
        x_salient = x * (1.0 + 0.3 * saliency_map.expand(-1, C, -1, -1, -1))

        # rPPG heatmap
        if self.no_rppg:
            rppg_heatmap = torch.zeros_like(x_salient)
        elif self.diff_mode:
            onset_frame = x_salient[:, :, 0:1, :, :]
            apex_idx = T // 2
            apex_frame = x_salient[:, :, apex_idx:apex_idx+1, :, :]
            onset_apex_diff = apex_frame - onset_frame
            rppg_heatmap = onset_apex_diff.expand(-1, -1, T, -1, -1)
        else:
            rppg_heatmap = self.rppg(x_salient)

        # Zero RGB, only rPPG
        zero_rgb = torch.zeros_like(x_salient)
        rppg_only = torch.cat([zero_rgb, rppg_heatmap], dim=1)

        # Slow pathway
        slow_feat, _ = self.slow_pathway(rppg_only)
        return slow_feat


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    if self.verbose: print("=" * 60)
    if self.verbose: print(" Censor -- Biomimetic Dual-Pathway MER System")
    if self.verbose: print("=" * 60)

    # Build model
    model = Censor()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if self.verbose: print(f"Total parameters:     {total_params:>12,}")
    if self.verbose: print(f"Trainable parameters: {trainable_params:>12,}")
    if self.verbose: print()

    # Create dummy input: (B=2, C=3, T=16, H=224, W=224)
    dummy_input = torch.randn(2, 3, 16, 224, 224)
    if self.verbose: print(f"\nDummy input shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    if self.verbose: print("\n" + "=" * 60)
    if self.verbose: print(" Forward Pass Completed Successfully!")
    if self.verbose: print("=" * 60)

    # Print sample report
    if self.verbose: print("\nSample template report (Subject 0):")
    if self.verbose: print("-" * 40)
    if self.verbose: print(outputs['template_report'][0])

    # Verify all expected keys present
    expected_keys = [
        'me_logits', 'au_intensities', 'au_opd', 'apex_scores',
        'expert_gates', 'moe_aux_loss', 'adapted_feat',
        'template_report', 'llm_report'
    ]
    missing_keys = [k for k in expected_keys if k not in outputs]
    if missing_keys:
        if self.verbose: print(f"\nWARNING: Missing output keys: {missing_keys}")
    else:
        if self.verbose: print("\nAll expected output keys present. Verification complete.")

    if self.verbose: print("\nDone.")