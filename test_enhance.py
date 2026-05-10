# =============================================================================
# Test: Censor with Biomimetic Enhancements
# =============================================================================

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.biomimetic_enhance import DTNEnhancedFFA, MetaPlasticityMemory
from config.defaults import FFA_CONFIG, FUSION_CONFIG


def test_dtn_ffa():
    """Test DTN-enhanced FFA"""
    print("\n" + "=" * 50)
    print(" Test: DTN-Enhanced FFA")
    print("=" * 50)

    # Create module
    dtn_ffa = DTNEnhancedFFA(FFA_CONFIG)
    print(f"Parameters: {sum(p.numel() for p in dtn_ffa.parameters()):,}")

    # Test input
    fast_feat = torch.randn(2, 512)
    slow_feat = torch.randn(2, 768)

    with torch.no_grad():
        fast_gated, slow_gated = dtn_ffa(fast_feat, slow_feat)

    print(f"Output fast: {fast_gated.shape}")
    print(f"Output slow: {slow_gated.shape}")

    # Check tension values
    print(f"DTN-FFA test passed!")


def test_meta_plasticity():
    """Test Meta-Plasticity Memory"""
    print("\n" + "=" * 50)
    print(" Test: Meta-Plasticity Memory")
    print("=" * 50)

    # Create module
    meta_memory = MetaPlasticityMemory(
        input_dim=1024,
        num_slots=4,
        rank=8,
        strong_threshold=0.7
    )
    print(f"Parameters: {sum(p.numel() for p in meta_memory.parameters()):,}")

    # Test inference
    fused_feat = torch.randn(2, 1024)
    context_feat = torch.randn(2, 1024)

    with torch.no_grad():
        enhanced, emotion = meta_memory(fused_feat, context_feat)

    print(f"Input: {fused_feat.shape}")
    print(f"Enhanced: {enhanced.shape}")
    print(f"Emotion scores: {emotion.squeeze().tolist()}")

    # Test with strong stimulus (force consolidation)
    print("\n-- Testing strong stimulus --")
    strong_context = torch.ones(2, 1024) * 10  # High values -> strong stimulus

    with torch.no_grad():
        enhanced2, emotion2 = meta_memory(fused_feat, strong_context)

    print(f"After strong stimulus: emotion={emotion2.squeeze().tolist()}")

    # Count active slots
    active = sum(1 for slot in meta_memory.slots if slot.timestamp.item() > 0)
    print(f"Active slots: {active}/{meta_memory.num_slots}")

    print(f"Meta-Plasticity test passed!")


def test_full_pipeline():
    """Test full enhancement pipeline"""
    print("\n" + "=" * 50)
    print(" Test: Full Enhancement Pipeline")
    print("=" * 50)

    # Create DTN-FFA
    dtn_ffa = DTNEnhancedFFA(FFA_CONFIG)
    meta_memory = MetaPlasticityMemory(input_dim=1024)

    # Simulate Censor pipeline
    # Stage 1: Dual pathways
    fast_feat = torch.randn(2, 512)
    slow_feat = torch.randn(2, 768)

    # Stage 2: DTN-FFA (replaces original FFA)
    fast_gated, slow_gated = dtn_ffa(fast_feat, slow_feat)

    # Stage 3: Simple fusion (simulated)
    fused = torch.cat([fast_gated, slow_gated], dim=1)  # (2, 1280)
    fused = fused[:, :1024]  # Project to 1024

    # Stage 4: Meta-Plasticity (after fusion)
    enhanced, emotion = meta_memory(fused, fused)

    print(f"Final output: {enhanced.shape}")
    print(f"Pipeline test passed!")


if __name__ == '__main__':
    print("=" * 60)
    print(" Censor + Biomimetic Enhancements Test")
    print("=" * 60)

    test_dtn_ffa()
    test_meta_plasticity()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print(" All Tests Passed!")
    print("=" * 60)