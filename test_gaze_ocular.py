# =============================================================================
# Censor -- Test Gaze Attention and Ocular Filter Modules
# =============================================================================
# Tests the newly implemented:
#   1. GazeDrivenAttention - gaze-based AU region attention
#   2. OcularMotionFilter - blink/saccade interference filtering
# =============================================================================

import torch
import torch.nn as nn
import numpy as np

# Import new modules
from model.gaze_attention import (
    GazeEstimator,
    AURegionAttention,
    GazeDrivenAttention,
    GazeEmotionCorrelation,
    AU_REGIONS,
    EMOTION_GAZE_PATTERNS,
    create_gaze_attention,
)

from model.ocular_filter import (
    BlinkDetector,
    SaccadeDetector,
    SmoothPursuitDetector,
    OcularMotionFilter,
    CleanSignalExtractor,
    BLINK_CONFIG,
    create_ocular_filter,
    create_clean_signal_extractor,
)


def test_gaze_estimator():
    """Test GazeEstimator module."""
    print("\n" + "="*60)
    print("[1] Testing GazeEstimator")
    print("="*60)

    model = GazeEstimator()

    # Test with encoded eye features
    eye_features = torch.randn(2, 64)
    gaze, confidence = model(eye_features)

    print(f"  Input: eye_features shape = {eye_features.shape}")
    print(f"  Output: gaze shape = {gaze.shape}, range = [{gaze.min():.3f}, {gaze.max():.3f}]")
    print(f"  Output: confidence shape = {confidence.shape}, range = [{confidence.min():.3f}, {confidence.max():.3f}]")

    # Test with raw eye region
    eye_region = torch.randn(2, 3, 32, 32)
    gaze, confidence = model(eye_region)

    print(f"\n  Input: eye_region shape = {eye_region.shape}")
    print(f"  Output: gaze shape = {gaze.shape}")
    print(f"  Output: confidence shape = {confidence.shape}")

    assert gaze.shape == (2, 2), "Gaze shape mismatch"
    assert confidence.shape == (2, 1), "Confidence shape mismatch"
    assert (gaze >= -1).all() and (gaze <= 1).all(), "Gaze values out of range"
    assert (confidence >= 0).all() and (confidence <= 1).all(), "Confidence out of range"

    print("\n  [PASSED] GazeEstimator test passed!")
    return True


def test_au_region_attention():
    """Test AURegionAttention module."""
    print("\n" + "="*60)
    print("[2] Testing AURegionAttention")
    print("="*60)

    model = AURegionAttention()

    # Test with gaze input
    gaze = torch.tensor([[0.3, -0.5], [-0.2, 0.1]])  # Looking right-down, left-up
    spatial_attention, region_weights = model(gaze)

    print(f"  Input: gaze shape = {gaze.shape}")
    print(f"  Output: spatial_attention shape = {spatial_attention.shape}")
    print(f"  Output: region_weights shape = {region_weights.shape}")
    print(f"\n  Region weights for sample 0: {region_weights[0].tolist()}")
    print(f"  (brows, eyes, nose, mouth)")

    # Test with emotion hint
    emotion_hint = torch.zeros(2, 11)
    emotion_hint[0, 0] = 1.0  # Happiness (Duchenne)
    emotion_hint[1, 6] = 1.0  # Disgust

    spatial_attention, region_weights = model(gaze, emotion_hint)

    print(f"\n  With emotion hint (Duchenne happiness vs Disgust):")
    print(f"  Region weights sample 0 (Duchenne): {region_weights[0].tolist()}")
    print(f"  Expected: higher eyes weight (Duchenne focuses on eyes)")
    print(f"  Region weights sample 1 (Disgust): {region_weights[1].tolist()}")
    print(f"  Expected: higher nose/mouth weight (Disgust focuses on nose)")

    assert spatial_attention.shape == (2, 1, 224, 224), "Spatial attention shape mismatch"
    assert region_weights.shape == (2, 4), "Region weights shape mismatch"

    print("\n  [PASSED] AURegionAttention test passed!")
    return True


def test_gaze_driven_attention():
    """Test complete GazeDrivenAttention pipeline."""
    print("\n" + "="*60)
    print("[3] Testing GazeDrivenAttention (Complete Pipeline)")
    print("="*60)

    model = GazeDrivenAttention()

    # Test with face features only (2D features)
    face_features_2d = torch.randn(2, 512)
    modulated, spatial_attention = model(face_features_2d)

    print(f"  Input: face_features (2D) shape = {face_features_2d.shape}")
    print(f"  Output: modulated features shape = {modulated.shape}")

    # Test with 4D face features
    face_features_4d = torch.randn(2, 64, 14, 14)
    modulated, spatial_attention = model(face_features_4d)

    print(f"  Input: face_features (4D) shape = {face_features_4d.shape}")
    print(f"  Output: modulated features shape = {modulated.shape}")
    print(f"  Output: spatial_attention shape = {spatial_attention.shape}")

    assert modulated.shape == face_features_4d.shape, "Modulated features shape mismatch"
    assert spatial_attention.shape[2:] == face_features_4d.shape[2:], "Spatial attention size mismatch"

    print("\n  [PASSED] GazeDrivenAttention test passed!")
    return True


def test_gaze_emotion_correlation():
    """Test GazeEmotionCorrelation for Duchenne/deception detection."""
    print("\n" + "="*60)
    print("[4] Testing GazeEmotionCorrelation")
    print("="*60)

    model = GazeEmotionCorrelation()

    # Test genuine vs fake smile pattern
    gaze = torch.tensor([[0.0, 0.0], [0.5, 0.3]])  # Center (genuine), aversion (fake)
    region_weights = torch.tensor([
        [0.3, 0.4, 0.1, 0.2],  # Balanced (genuine smile)
        [0.1, 0.2, 0.1, 0.6],  # Mouth-focused (fake smile)
    ])

    duchenne_score, deception_score, gaze_deviation = model(gaze, region_weights)

    print(f"  Sample 0 (center gaze, balanced weights):")
    print(f"    Duchenne score: {duchenne_score[0].item():.3f} (expect higher)")
    print(f"    Deception score: {deception_score[0].item():.3f} (expect lower)")
    print(f"    Gaze deviation: {gaze_deviation[0].item():.3f}")

    print(f"\n  Sample 1 (averted gaze, mouth-focused):")
    print(f"    Duchenne score: {duchenne_score[1].item():.3f} (expect lower)")
    print(f"    Deception score: {deception_score[1].item():.3f} (expect higher)")
    print(f"    Gaze deviation: {gaze_deviation[1].item():.3f}")

    assert duchenne_score.shape == (2, 1), "Duchenne score shape mismatch"
    assert deception_score.shape == (2, 1), "Deception score shape mismatch"

    print("\n  [PASSED] GazeEmotionCorrelation test passed!")
    return True


def test_blink_detector():
    """Test BlinkDetector module."""
    print("\n" + "="*60)
    print("[5] Testing BlinkDetector")
    print("="*60)

    model = BlinkDetector()

    # Create synthetic AU45 sequence with blink pattern
    T = 200  # 200 frames (1 second at 200fps)
    au45_sequence = torch.zeros(2, T)

    # Add blink pattern: quick rise -> plateau -> quick fall
    # Typical blink: ~60 frames (300ms)
    blink_start = 30
    blink_duration = 60

    # Create proper triangular pattern (rise then fall)
    rise_vals = torch.linspace(0.2, 0.8, blink_duration // 2)
    fall_vals = torch.linspace(0.8, 0.2, blink_duration // 2)
    blink_pattern = torch.cat([rise_vals, fall_vals])  # 60 elements total

    au45_sequence[0, blink_start:blink_start+blink_duration] = blink_pattern

    # Add another blink
    blink_start2 = 150
    au45_sequence[0, blink_start2:blink_start2+40] = torch.linspace(0.2, 0.8, 40)

    # Sample 1: No blinks (low random noise)
    au45_sequence[1] = torch.randn(T) * 0.1 + 0.1

    blink_mask, stats = model(au45_sequence, return_stats=True)

    print(f"  Input: au45_sequence shape = {au45_sequence.shape}")
    print(f"  Output: blink_mask shape = {blink_mask.shape}")
    print(f"\n  Blink statistics:")
    print(f"    Number of blinks: {stats['num_blinks']}")
    print(f"    Average duration: {stats['avg_duration']:.1f} frames")
    print(f"    Blink rate: {stats['blink_rate']:.1f} blinks/min")
    print(f"    Peak intensity: {stats['peak_intensity']:.3f}")

    print(f"\n  Blink mask active frames (sample 0): {blink_mask[0].sum().item()}")

    assert blink_mask.shape == (2, T), "Blink mask shape mismatch"
    assert stats['num_blinks'] >= 0, "Invalid blink count"

    print("\n  [PASSED] BlinkDetector test passed!")
    return True


def test_saccade_detector():
    """Test SaccadeDetector module."""
    print("\n" + "="*60)
    print("[6] Testing SaccadeDetector")
    print("="*60)

    model = SaccadeDetector()

    # Create synthetic gaze sequence with saccade pattern
    T = 200
    gaze_sequence = torch.zeros(2, T, 2)

    # Sample 0: Add saccade (sharp position change)
    # Normal: smooth drift, then saccade at frame 50 (sharp jump)
    gaze_sequence[0, :50, 0] = torch.linspace(-0.2, -0.15, 50)  # Slow drift
    gaze_sequence[0, 50:55, 0] = torch.linspace(-0.15, 0.3, 5)  # Saccade (sharp jump)
    gaze_sequence[0, 55:100, 0] = torch.linspace(0.3, 0.28, 45)  # Slow drift
    gaze_sequence[0, 100:, 0] = torch.linspace(0.28, 0.25, 100)

    # Y component
    gaze_sequence[0, :, 1] = torch.linspace(0.1, -0.1, T)

    # Sample 1: No saccades (smooth only)
    gaze_sequence[1, :, 0] = torch.linspace(0, 0.1, T)
    gaze_sequence[1, :, 1] = torch.linspace(0, -0.05, T)

    saccade_mask, velocity = model(gaze_sequence, return_velocity=True)

    print(f"  Input: gaze_sequence shape = {gaze_sequence.shape}")
    print(f"  Output: saccade_mask shape = {saccade_mask.shape}")
    print(f"  Output: velocity shape = {velocity.shape}")

    print(f"\n  Velocity profile (sample 0):")
    print(f"    Max velocity: {velocity[0].max().item():.3f}")
    print(f"    Velocity at saccade (frame 50-55): {velocity[0, 50:55].tolist()}")

    print(f"\n  Saccade detected frames (sample 0): {saccade_mask[0].sum().item()}")
    print(f"  Saccade detected frames (sample 1): {saccade_mask[1].sum().item()}")

    assert saccade_mask.shape == (2, T), "Saccade mask shape mismatch"
    assert velocity.shape == (2, T), "Velocity shape mismatch"

    print("\n  [PASSED] SaccadeDetector test passed!")
    return True


def test_ocular_motion_filter():
    """Test complete OcularMotionFilter pipeline."""
    print("\n" + "="*60)
    print("[7] Testing OcularMotionFilter (Complete Pipeline)")
    print("="*60)

    model = OcularMotionFilter()

    # Create synthetic inputs
    B, T, num_aus = 2, 200, 28
    H, W = 224, 224

    # AU sequence (28 AUs, AU45 is not in this range since AU indices are 1-28)
    au_sequence = torch.randn(B, T, num_aus) * 0.3 + 0.3

    # Optical flow
    optical_flow = torch.randn(B, 2, T, H, W) * 0.1

    # Gaze sequence
    gaze_sequence = torch.zeros(B, T, 2)
    gaze_sequence[0, 50:55, 0] = torch.linspace(-0.1, 0.4, 5)  # Saccade

    outputs = model(
        au_sequence,
        optical_flow,
        gaze_sequence,
        return_masks=True,
        return_stats=True
    )

    print(f"  Input: au_sequence shape = {au_sequence.shape}")
    print(f"  Input: optical_flow shape = {optical_flow.shape}")
    print(f"  Input: gaze_sequence shape = {gaze_sequence.shape}")

    print(f"\n  Output: clean_au shape = {outputs['clean_au'].shape}")
    print(f"  Output: clean_flow shape = {outputs['clean_flow'].shape}")

    print(f"\n  Detected masks:")
    print(f"    Blink frames (sample 0): {outputs['masks']['blink'][0].sum().item()}")
    print(f"    Saccade frames (sample 0): {outputs['masks']['saccade'][0].sum().item()}")
    print(f"    Total interference frames: {outputs['masks']['interference'][0].sum().item()}")

    print(f"\n  Statistics:")
    print(f"    Blink count: {outputs['stats']['blink']['num_blinks']}")
    print(f"    Saccade count: {outputs['stats']['saccade']['num_saccades']}")
    print(f"    Filter ratio: {outputs['stats']['filter_ratio']:.3f}")

    assert outputs['clean_au'].shape == au_sequence.shape, "Clean AU shape mismatch"
    assert outputs['clean_flow'].shape == optical_flow.shape, "Clean flow shape mismatch"

    print("\n  [PASSED] OcularMotionFilter test passed!")
    return True


def test_clean_signal_extractor():
    """Test CleanSignalExtractor with quality scoring."""
    print("\n" + "="*60)
    print("[8] Testing CleanSignalExtractor")
    print("="*60)

    model = CleanSignalExtractor()

    # Create inputs
    B, T, num_aus = 2, 200, 28
    H, W = 112, 112

    au_sequence = torch.randn(B, T, num_aus) * 0.2 + 0.4
    optical_flow = torch.randn(B, 2, T, H, W) * 0.05

    outputs = model(au_sequence, optical_flow, return_quality=True)

    print(f"  Input: au_sequence shape = {au_sequence.shape}")
    print(f"  Input: optical_flow shape = {optical_flow.shape}")

    print(f"\n  Output: clean AU shape = {outputs['au'].shape}")
    print(f"  Output: clean flow shape = {outputs['flow'].shape}")
    print(f"  Output: baseline shape = {outputs['baseline'].shape}")
    print(f"  Output: quality score = {outputs['quality'].tolist()}")

    assert outputs['au'].shape == au_sequence.shape
    assert outputs['flow'].shape == optical_flow.shape
    assert outputs['baseline'].shape == (B, num_aus)
    assert outputs['quality'].shape == (B, 1)

    print("\n  [PASSED] CleanSignalExtractor test passed!")
    return True


def test_integration():
    """Test integration of both modules."""
    print("\n" + "="*60)
    print("[9] Testing Integration: GazeAttention + OcularFilter")
    print("="*60)

    gaze_attention = GazeDrivenAttention()
    ocular_filter = OcularMotionFilter()

    # Simulate video processing pipeline
    B, T = 2, 200
    H, W = 224, 224
    num_aus = 28

    # Face features (from backbone) - 2D features
    face_features = torch.randn(B, 512)

    # Eye region
    eye_region = torch.randn(B, 3, 32, 32)

    # Emotion hint (from previous frame classification)
    emotion_hint = torch.zeros(B, 11)
    emotion_hint[0, 0] = 0.6  # Some happiness hint

    # Step 1: Apply gaze-driven attention
    modulated_features, spatial_attention, gaze, confidence = gaze_attention(
        face_features,
        eye_region=eye_region,
        emotion_hint=emotion_hint,
        return_gaze=True
    )

    print(f"  [Step 1] Gaze-driven attention applied")
    print(f"    Gaze estimate: {gaze.tolist()}")
    print(f"    Confidence: {confidence.tolist()}")

    # Step 2: Simulate AU sequence output (from decoder)
    au_sequence = torch.randn(B, T, num_aus) * 0.3

    # Step 3: Simulate optical flow
    optical_flow = torch.randn(B, 2, T, H, W) * 0.1

    # Step 4: Apply ocular filter
    clean_outputs = ocular_filter(
        au_sequence,
        optical_flow,
        gaze_sequence=torch.randn(B, T, 2),  # Simulated gaze sequence
        return_masks=True,
        return_stats=True
    )

    print(f"\n  [Step 2] Ocular motion filtering applied")
    print(f"    Interference frames filtered: {clean_outputs['masks']['interference'][0].sum().item()}")
    print(f"    Filter ratio: {clean_outputs['stats']['filter_ratio']:.3f}")

    # Step 5: Use clean signals for downstream
    clean_au = clean_outputs['clean_au']
    clean_flow = clean_outputs['clean_flow']

    print(f"\n  Final clean signals ready for ME classification")
    print(f"    Clean AU shape: {clean_au.shape}")
    print(f"    Clean flow shape: {clean_flow.shape}")

    print("\n  [PASSED] Integration test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Censor: Gaze Attention & Ocular Filter Tests")
    print("="*60)

    tests = [
        test_gaze_estimator,
        test_au_region_attention,
        test_gaze_driven_attention,
        test_gaze_emotion_correlation,
        test_blink_detector,
        test_saccade_detector,
        test_ocular_motion_filter,
        test_clean_signal_extractor,
        test_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n  [FAILED] Test failed: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed}/{len(tests)} passed, {failed} failed")
    print("="*60)

    # Print AU_REGIONS and EMOTION_GAZE_PATTERNS reference
    print("\n" + "-"*60)
    print("AU_REGIONS Reference:")
    print("-"*60)
    for region, info in AU_REGIONS.items():
        print(f"  {region}: {info['description']}")
        print(f"    AUs: {info['aus']}")

    print("\n" + "-"*60)
    print("EMOTION_GAZE_PATTERNS Reference:")
    print("-"*60)
    for emotion, weights in EMOTION_GAZE_PATTERNS.items():
        print(f"  {emotion}: {weights}")

    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)