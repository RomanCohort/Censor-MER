# =============================================================================
# Micro-Expression Generation Evaluation
# =============================================================================
# Purpose: Evaluate generated micro-expression videos.
#
# Metrics:
#   1. FID (Fréchet Inception Distance) - Visual quality
#   2. AU Consistency - AU activation match
#   3. Temporal Consistency - Onset-apex-offset curve match
#   4. Recognition Feedback - Generated samples' recognition accuracy
#   5. Keypoint Error - Motion field accuracy
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# FID Calculator
# =============================================================================

class FIDCalculator:
    """
    Calculate Fréchet Inception Distance for generated videos.

    FID measures visual quality by comparing feature distributions.
    Lower FID = better quality.
    """

    def __init__(self, device='cuda'):
        self.device = device
        # Feature extractor (simplified - use InceptionV3 in production)
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self):
        """Build feature extractor for FID."""
        # Simplified feature extractor
        # In production, use pretrained InceptionV3
        return nn.Sequential(
            nn.Conv3d(3, 32, 3, (1, 2, 2), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, (1, 2, 2), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, (1, 2, 2), 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        ).to(self.device)

    def extract_features(self, videos):
        """
        Extract features from videos.

        Args:
            videos (torch.Tensor): Videos, shape (B, C, T, H, W)

        Returns:
            features (torch.Tensor): Features, shape (B, D)
        """
        return self.feature_extractor(videos)

    def calculate_fid(self, real_features, fake_features):
        """
        Calculate FID between real and fake features.

        Args:
            real_features (torch.Tensor): Real video features
            fake_features (torch.Tensor): Generated video features

        Returns:
            fid (float): FID score
        """
        # Convert to numpy
        real = real_features.cpu().numpy()
        fake = fake_features.cpu().numpy()

        # Calculate statistics
        mu_real = np.mean(real, axis=0)
        sigma_real = np.cov(real, rowvar=False)

        mu_fake = np.mean(fake, axis=0)
        sigma_fake = np.cov(fake, rowvar=False)

        # FID formula: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        diff = mu_real - mu_fake

        # Simplified: use trace approximation
        covmean = 2.0 * sigma_fake  # Simplified approximation

        fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - covmean)

        return float(fid)

    def evaluate(self, real_videos, generated_videos):
        """
        Evaluate FID between real and generated videos.

        Args:
            real_videos (torch.Tensor): Real videos
            generated_videos (torch.Tensor): Generated videos

        Returns:
            fid (float): FID score
        """
        real_features = self.extract_features(real_videos)
        fake_features = self.extract_features(generated_videos)

        fid = self.calculate_fid(real_features, fake_features)
        return fid


# =============================================================================
# AU Consistency Calculator
# =============================================================================

class AUConsistencyCalculator:
    """
    Calculate AU consistency between generated and real videos.

    Measures how well AU activations match in generated videos.
    """

    def __init__(self, au_predictor=None, device='cuda'):
        self.device = device
        self.au_predictor = au_predictor

    def calculate_au_consistency(self, generated_videos, target_au):
        """
        Calculate AU consistency.

        Args:
            generated_videos (torch.Tensor): Generated videos
            target_au (torch.Tensor): Target AU activations

        Returns:
            consistency (float): AU consistency score (0-1)
        """
        if self.au_predictor is None:
            # Use simple heuristic: compare motion patterns
            return self._simple_au_check(generated_videos, target_au)

        # Predict AU from generated videos
        B, C, T, H, W = generated_videos.shape

        # Average frame features
        generated_au_pred = []
        for b in range(B):
            video = generated_videos[b:b+1]  # (1, C, T, H, W)
            # Extract features (placeholder)
            features = torch.zeros(1, 1024).to(self.device)
            emotion = torch.zeros(1).long().to(self.device)  # Placeholder
            au_pred = self.au_predictor(features, emotion)
            generated_au_pred.append(au_pred)

        generated_au = torch.cat(generated_au_pred, dim=0)  # (B, 17)

        # Calculate consistency (correlation or MSE)
        mse = F.mse_loss(generated_au, target_au)
        consistency = 1.0 - mse.item()  # Simplified mapping

        return max(0, consistency)

    def _simple_au_check(self, generated_videos, target_au):
        """Simple AU check without predictor."""
        # Check if motion patterns match AU expectations
        # This is a placeholder
        return 0.5  # Default score


# =============================================================================
# Temporal Consistency Calculator
# =============================================================================

class TemporalConsistencyCalculator:
    """
    Calculate temporal consistency of generated videos.

    Measures onset-apex-offset curve smoothness and correctness.
    """

    def calculate_temporal_consistency(self, generated_videos, expected_duration_ratio=0.5):
        """
        Calculate temporal consistency.

        Args:
            generated_videos (torch.Tensor): Generated videos
            expected_duration_ratio (float): Expected apex duration ratio

        Returns:
            consistency (float): Temporal consistency score
        """
        B, C, T, H, W = generated_videos.shape

        # Compute frame-to-frame motion magnitude
        motion_curve = self._compute_motion_curve(generated_videos)

        # Check curve shape
        # Expected: gradual rise -> peak -> gradual decline
        consistency_scores = []

        for b in range(B):
            curve = motion_curve[b]

            # Find peak (apex)
            peak_idx = curve.argmax().item()

            # Check rise phase (0 to peak)
            rise_consistency = self._check_rise_phase(curve[:peak_idx+1])

            # Check decline phase (peak to end)
            decline_consistency = self._check_decline_phase(curve[peak_idx:])

            total_consistency = (rise_consistency + decline_consistency) / 2
            consistency_scores.append(total_consistency)

        return np.mean(consistency_scores)

    def _compute_motion_curve(self, videos):
        """Compute motion magnitude curve."""
        B, C, T, H, W = videos.shape

        curves = []
        for b in range(B):
            video = videos[b]  # (C, T, H, W)

            # Compute frame differences
            diffs = []
            for t in range(1, T):
                diff = (video[:, t] - video[:, 0]).abs().mean()
                diffs.append(diff.item())

            # Normalize
            curve = torch.tensor(diffs)
            if curve.max() > 0:
                curve = curve / curve.max()

            curves.append(curve)

        return torch.stack(curves)

    def _check_rise_phase(self, curve):
        """Check if rise phase is smooth."""
        if len(curve) < 2:
            return 1.0

        # Should be generally increasing
        increasing_count = 0
        for i in range(1, len(curve)):
            if curve[i] >= curve[i-1] - 0.1:  # Allow small dips
                increasing_count += 1

        return increasing_count / (len(curve) - 1)

    def _check_decline_phase(self, curve):
        """Check if decline phase is smooth."""
        if len(curve) < 2:
            return 1.0

        # Should be generally decreasing
        decreasing_count = 0
        for i in range(1, len(curve)):
            if curve[i] <= curve[i-1] + 0.1:  # Allow small rises
                decreasing_count += 1

        return decreasing_count / (len(curve) - 1)


# =============================================================================
# Recognition Feedback Calculator
# =============================================================================

class RecognitionFeedbackCalculator:
    """
    Evaluate generated samples using recognition model.

    Measures if generated videos are correctly recognized.
    """

    def __init__(self, recognition_model=None, device='cuda'):
        self.device = device
        self.recognition_model = recognition_model

    def calculate_recognition_accuracy(self, generated_videos, target_emotion):
        """
        Calculate recognition accuracy on generated videos.

        Args:
            generated_videos (torch.Tensor): Generated videos
            target_emotion (torch.Tensor): Target emotion class

        Returns:
            accuracy (float): Recognition accuracy
        """
        if self.recognition_model is None:
            return self._placeholder_accuracy(target_emotion)

        B = generated_videos.shape[0]

        correct = 0
        for b in range(B):
            video = generated_videos[b:b+1].to(self.device)

            # Recognize
            pred_emotion = self._recognize(video)

            if pred_emotion == target_emotion[b].item():
                correct += 1

        accuracy = correct / B
        return accuracy

    def _recognize(self, video):
        """Recognize emotion from video."""
        # Placeholder - actual uses Censor
        return torch.randint(0, 4, (1,)).item()

    def _placeholder_accuracy(self, target_emotion):
        """Placeholder accuracy."""
        # Random accuracy ~25% for 4 classes
        return 0.25


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

class GenerationEvaluator:
    """
    Comprehensive evaluator for micro-expression generation.
    """

    def __init__(self, au_predictor=None, recognition_model=None, device='cuda'):
        self.device = device

        self.fid_calculator = FIDCalculator(device)
        self.au_calculator = AUConsistencyCalculator(au_predictor, device)
        self.temporal_calculator = TemporalConsistencyCalculator()
        self.recognition_calculator = RecognitionFeedbackCalculator(recognition_model, device)

    def evaluate(self, generated_videos, real_videos, target_au, target_emotion):
        """
        Comprehensive evaluation.

        Args:
            generated_videos (torch.Tensor): Generated videos, shape (B, C, T, H, W)
            real_videos (torch.Tensor): Real videos (for FID comparison)
            target_au (torch.Tensor): Target AU activations
            target_emotion (torch.Tensor): Target emotion classes

        Returns:
            results (dict): Evaluation metrics
        """
        print("\n[Evaluating Micro-Expression Generation]")
        print("=" * 50)

        results = {}

        # 1. FID
        print("[1/4] Computing FID...")
        fid = self.fid_calculator.evaluate(real_videos, generated_videos)
        results['FID'] = fid
        print(f"  FID: {fid:.4f}")

        # 2. AU Consistency
        print("[2/4] Computing AU Consistency...")
        au_consistency = self.au_calculator.calculate_au_consistency(
            generated_videos, target_au
        )
        results['AU_Consistency'] = au_consistency
        print(f"  AU Consistency: {au_consistency:.4f}")

        # 3. Temporal Consistency
        print("[3/4] Computing Temporal Consistency...")
        temporal_consistency = self.temporal_calculator.calculate_temporal_consistency(
            generated_videos
        )
        results['Temporal_Consistency'] = temporal_consistency
        print(f"  Temporal Consistency: {temporal_consistency:.4f}")

        # 4. Recognition Feedback
        print("[4/4] Computing Recognition Accuracy...")
        recognition_acc = self.recognition_calculator.calculate_recognition_accuracy(
            generated_videos, target_emotion
        )
        results['Recognition_Accuracy'] = recognition_acc
        print(f"  Recognition Accuracy: {recognition_acc:.4f}")

        # Summary
        print("\n" + "=" * 50)
        print("Summary:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

        return results

    def evaluate_controllable_generation(self, model, neutral_face, emotion_classes,
                                         intensities=[0.3, 0.5, 0.8, 1.0]):
        """
        Evaluate controllable generation across different intensities.

        Args:
            model: Generation model
            neutral_face (torch.Tensor): Neutral face
            emotion_classes (torch.Tensor): Emotion classes to generate
            intensities (list): Intensity values to test

        Returns:
            intensity_results (dict): Results for each intensity
        """
        print("\n[Evaluating Controllable Generation]")
        print("=" * 50)

        intensity_results = {}

        for intensity in intensities:
            print(f"\n[Intensity = {intensity}]")

            # Generate videos with this intensity
            intensity_tensor = torch.tensor([intensity] * len(emotion_classes)).to(self.device)

            generated = model.generate(
                neutral_face, emotion_classes,
                intensity=intensity_tensor.unsqueeze(-1)
            )

            # Evaluate temporal curve shape
            motion_curve = self.temporal_calculator._compute_motion_curve(generated)

            # Peak magnitude should scale with intensity
            peak_magnitude = motion_curve.max().item()

            intensity_results[intensity] = {
                'peak_magnitude': peak_magnitude,
                'expected_peak': intensity,
                'scale_accuracy': min(peak_magnitude / intensity, 1.0) if intensity > 0 else 1.0
            }

            print(f"  Peak Magnitude: {peak_magnitude:.4f}")
            print(f"  Scale Accuracy: {intensity_results[intensity]['scale_accuracy']:.4f}")

        return intensity_results


# =============================================================================
# Demo Evaluation
# =============================================================================

def demo_evaluation():
    """Demo the evaluation pipeline."""
    print("\n" + "="*60)
    print("Generation Evaluation Demo")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create evaluator
    evaluator = GenerationEvaluator(device=device)

    # Create dummy data
    B, C, T, H, W = 4, 3, 16, 224, 224
    generated_videos = torch.randn(B, C, T, H, W)
    real_videos = torch.randn(B, C, T, H, W)
    target_au = torch.rand(B, 17)
    target_emotion = torch.randint(0, 4, (B,))

    # Evaluate
    results = evaluator.evaluate(
        generated_videos, real_videos, target_au, target_emotion
    )

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_evaluation()