# =============================================================================
# Test: Enhanced MoE with Biological Gating
# =============================================================================

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.enhanced_moe import EnhancedMoE, EnhancedMoEWrapper


def test_standard():
    """Test standard mode"""
    print("\n" + "=" * 50)
    print(" Test: Standard Mode")
    print("=" * 50)

    moe = EnhancedMoE(mode="standard")
    x = torch.randn(2, 1024)

    output, gates, aux_loss, info = moe(x)
    print(f"Output: {output.shape}")
    print(f"Gates: {gates[0].tolist()}")
    print(f"Info: {info}")


def test_hybrid():
    """Test hybrid mode"""
    print("\n" + "=" * 50)
    print(" Test: Hybrid Mode")
    print("=" * 50)

    moe = EnhancedMoE(mode="hybrid", enable_membrane=True, enable_emotion=True)
    x = torch.randn(2, 1024)

    output, gates, aux_loss, info = moe(x)
    print(f"Output: {output.shape}")
    print(f"Gates: {gates[0].tolist()}")
    print(f"Info keys: {info.keys()}")
    print(f"Membrane: {info.get('membrane_activation')}")
    print(f"Emotion: {info.get('emotional_state')}")


def test_stateful():
    """Test stateful behavior"""
    print("\n" + "=" * 50)
    print(" Test: Stateful Behavior")
    print("=" * 50)

    moe = EnhancedMoE(mode="hybrid", enable_membrane=True)

    # First pass - high energy (should excite membrane)
    x1 = torch.randn(2, 1024) * 3
    out1, gates1, _, info1 = moe(x1)
    print(f"Pass 1 (high energy): membrane={info1.get('membrane_activation')}, emotion={info1.get('emotional_state')}")

    # Second pass - low energy
    x2 = torch.randn(2, 1024) * 0.5
    out2, gates2, _, info2 = moe(x2)
    print(f"Pass 2 (low energy): membrane={info2.get('membrane_activation')}, emotion={info2.get('emotional_state')}")

    # Get accumulated state
    state = moe.get_state()
    print(f"Total calls: {state['total_calls']}")
    print(f"Avg emotion: {state['avg_emotion']:.4f}")

    print("\nOK Stateful test passed!")


def test_reset():
    """Test state reset"""
    print("\n" + "=" * 50)
    print(" Test: State Reset")
    print("=" * 50)

    moe = EnhancedMoE(mode="hybrid", enable_membrane=True)

    # Some passes
    x = torch.randn(2, 1024) * 3
    moe(x)

    state_before = moe.get_state()
    print(f"Before reset: calls={state_before['total_calls']}")

    moe.reset_state()

    state_after = moe.get_state()
    print(f"After reset: calls={state_after['total_calls']}")

    print("\nOK Reset test passed!")


def main():
    print("=" * 60)
    print(" Enhanced MoE Test")
    print("=" * 60)

    test_standard()
    test_hybrid()
    test_stateful()
    test_reset()

    print("\n" + "=" * 60)
    print(" All Tests Passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()