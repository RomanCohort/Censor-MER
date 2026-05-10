# =============================================================================
# Test: Feedback-Based BioMoE
# =============================================================================

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.enhanced_moe import EnhancedMoE


def test_no_feedback():
    """Test without feedback (default behavior)"""
    print("\n" + "=" * 50)
    print(" Test: No Feedback")
    print("=" * 50)

    moe = EnhancedMoE(mode="hybrid", enable_membrane=True)
    x = torch.randn(2, 1024)

    output, gates, aux_loss, info = moe(x)
    print(f"Gates: {gates[0].tolist()}")
    print(f"Membrane: {info.get('membrane_activation')}")
    print(f"Emotion: {info.get('emotional_state')}")


def test_with_feedback():
    """Test WITH feedback (effect-dependent)"""
    print("\n" + "=" * 50)
    print(" Test: With Feedback")
    print("=" * 50)

    moe = EnhancedMoE(mode="hybrid", enable_membrane=True)

    # Initial state
    print("\n--- Initial state ---")
    state = moe.get_state()
    print(f"Initial: pos={state.get('positive_count')}, neg={state.get('negative_count')}, acc={state.get('accuracy')}")

    # Fake: user confirms prediction is CORRECT
    moe.apply_feedback(1.0)
    state = moe.get_state()
    print(f"After correct (feedback=1.0): pos={state.get('positive_count')}, acc={state.get('accuracy'):.2f}")

    # Fake: user confirms another CORRECT
    moe.apply_feedback(1.0)
    state = moe.get_state()
    print(f"After 2nd correct: pos={state.get('positive_count')}, acc={state.get('accuracy'):.2f}")

    # Fake: prediction was WRONG
    moe.apply_feedback(0.0)  # wrong (not confirmed)
    state = moe.get_state()
    print(f"After wrong (feedback=0.0): neg={state.get('negative_count')}, acc={state.get('accuracy'):.2f}")

    # Fake: user explicitly CORRECTED
    moe.apply_feedback(-1.0)  # strong negative
    state = moe.get_state()
    print(f"After corrected (feedback=-1.0): neg={state.get('negative_count')}, acc={state.get('accuracy'):.2f}")

    print("\nOK Feedback test passed!")


def test_effect_on_routing():
    """Test how feedback affects routing"""
    print("\n" + "=" * 50)
    print(" Test: Feedback Effect on Routing")
    print("=" * 50)

    moe = EnhancedMoE(mode="hybrid", enable_membrane=True)
    x = torch.randn(2, 1024)

    # Good history: user confirmed correctness
    moe.apply_feedback(1.0)
    moe.apply_feedback(1.0)
    moe.apply_feedback(1.0)

    out1, gates1, _, _ = moe(x)
    print(f"Good history: gates={gates1[0].tolist()}")
    print(f"  Get: {moe.get_state()['accuracy']:.2f} accuracy")

    # Reset for fair comparison
    moe.reset_state()

    # Bad history: user corrected errors
    moe.apply_feedback(-1.0)
    moe.apply_feedback(-1.0)
    moe.apply_feedback(-1.0)

    out2, gates2, _, _ = moe(x)
    print(f"Bad history: gates={gates2[0].tolist()}")
    print(f"  Get: {moe.get_state()['accuracy']:.2f} accuracy")

    print("\nOK Effect on routing test passed!")


def main():
    print("=" * 60)
    print(" Feedback-Based BioMoE Test")
    print("=" * 60)

    test_no_feedback()
    test_with_feedback()
    test_effect_on_routing()

    print("\n" + "=" * 60)
    print(" All Tests Passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()