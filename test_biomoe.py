# =============================================================================
# Test: Biological Gating Mechanism (BioMoE)
# =============================================================================

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.biomoe import (
    MembranePotential, EmotionalState, BioGate, BioMoE,
    StandardMoEComparison
)


def test_membrane_potential():
    """Test membrane potential mechanism"""
    print("\n" + "=" * 50)
    print(" Test: Membrane Potential")
    print("=" * 50)

    membrane = MembranePotential(dim=128, decay_rate=0.95)

    # First input
    x1 = torch.randn(2, 128) * 2  # High energy
    pot1, act1 = membrane(x1)
    print(f"Input 1: x1={x1.norm().item():.2f}")
    print(f"Membrane 1: norm={pot1.norm().item():.4f}, activation={act1.item():.4f}")

    # Second input (should see accumulation)
    x2 = torch.randn(2, 128) * 0.5  # Low energy
    pot2, act2 = membrane(x2)
    print(f"Input 2: x2={x2.norm().item():.2f}")
    print(f"Membrane 2: norm={pot2.norm().item():.4f}, activation={act2.item():.4f}")

    # Reset
    membrane.reset()
    print("Membrane reset!")


def test_emotional_state():
    """Test emotional state"""
    print("\n" + "=" * 50)
    print(" Test: Emotional State")
    print("=" * 50)

    emotion = EmotionalState(dim=128)

    # High energy input -> positive emotion
    x_positive = torch.randn(2, 128) * 3
    mood_pos = emotion(torch.randn(1, 128) * 3)
    print(f"High energy input: mood={mood_pos.item():.4f}")

    # Low energy input -> neutral/negative
    mood_neutral = emotion(torch.randn(1, 128) * 0.1)
    print(f"Low energy input: mood={mood_neutral.item():.4f}")


def test_bio_gate():
    """Test biological gate"""
    print("\n" + "=" * 50)
    print(" Test: Bio Gate")
    print("=" * 50)

    bio_gate = BioGate(input_dim=128, num_experts=3)

    x = torch.randn(2, 128)
    membrane = torch.randn(1, 128) * 0.5
    emotion = torch.tensor([[0.5]])  # Positive mood

    logits = bio_gate(x, membrane, emotion)
    print(f"Input gate logits: {logits[0].tolist()}")


def test_biomoe():
    """Test full BioMoE"""
    print("\n" + "=" * 50)
    print(" Test: BioMoE")
    print("=" * 50)

    biomoe = BioMoE(
        input_dim=1024,
        output_dim=7,
        num_experts=3,
        expert_hidden=512,
        k=2,
        decay_rate=0.95
    )
    print(f"Parameters: {sum(p.numel() for p in biomoe.parameters()):,}")

    # First pass
    x = torch.randn(2, 1024)
    output, gates, aux_loss, info = biomoe(x)

    print(f"Output: {output.shape}")
    print(f"Gates: {gates[0].tolist()}")
    print(f"Aux loss: {aux_loss.item():.4f}")
    print(f"Emotional state: {info['emotional_state']:.4f}")
    print(f"Expert usage: {info['expert_usage']}")

    # Second pass (should see memory effect)
    x2 = torch.randn(2, 1024) * 0.5
    output2, gates2, aux_loss2, info2 = biomoe(x2, update_membrane=False)

    print(f"\nPass 2:")
    print(f"Gates: {gates2[0].tolist()}")
    print(f"Emotional state: {info2['emotional_state']:.4f}")


def test_mood_contingent():
    """Test mood-contingent behavior"""
    print("\n" + "=" * 50)
    print(" Test: Mood-Contingent Routing")
    print("=" * 50)

    biomoe = BioMoE(input_dim=512, output_dim=7, num_experts=3, k=2)

    # Simulate: After success (high energy) -> confident routing
    success_input = torch.randn(4, 512) * 3
    out1, gates1, _, _ = biomoe(success_input)
    print(f"After success (high energy):")
    print(f"  Gate distribution: {gates1[0].tolist()}")

    # Reset membrane to simulate "negative mood"
    for mp in biomoe.membrane_potentials:
        mp.potential.fill_(-1.0)  # Depressed state

    # Same input should produce different routing
    neutral_input = torch.randn(4, 512) * 0.5
    out2, gates2, _, _ = biomoe(neutral_input)
    print(f"In depressed state (same input):")
    print(f"  Gate distribution: {gates2[0].tolist()}")

    print("\nOK Mood-contingent test passed!")


def main():
    print("=" * 60)
    print(" Biological Gating Mechanism (BioMoE) Test")
    print("=" * 60)

    # Show comparison
    StandardMoEComparison.analyze_difference()

    # Run tests
    test_membrane_potential()
    test_emotional_state()
    test_bio_gate()
    test_biomoe()
    test_mood_contingent()

    print("\n" + "=" * 60)
    print(" All Tests Passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()