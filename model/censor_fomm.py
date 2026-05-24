# =============================================================================
# Censor-FOMM -- Unified Recognition + Generation System
# =============================================================================
# Purpose: Combine Censor (recognition) with FOMM (generation) for complete
#          micro-expression processing pipeline.
#
# Pipeline:
#   1. Recognition: Input video → Emotion + AU + Apex
#   2. Generation: Neutral face + Emotion + AU → Generated ME video
#
# Applications:
#   - VTuber emotional interaction
#   - Micro-expression synthesis
#   - Emotion feedback loop
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.au_predictor import AUPredictor, AUPredictorWithEmotion
from model.au_controller import AUController, TemporalModulation
from generation.fomm_adapter import FOMMAdapter, load_pretrained_fomm


class CensorFOMM(nn.Module):
    """
    Censor-FOMM Unified System.

    Combines recognition and generation for micro-expression processing.
    """

    def __init__(self, censor_model=None, fomm_checkpoint=None,
                 use_au_predictor=True, device='cuda'):
        super().__init__()

        self.device = device
        self.use_au_predictor = use_au_predictor

        # =====================================
        # Recognition Module (Censor)
        # =====================================
        # If censor_model is provided, use it
        # Otherwise, create a placeholder for feature extraction
        if censor_model is not None:
            self.censor = censor_model
        else:
            # Placeholder: simple feature extractor for demo
            self.feature_extractor = nn.Sequential(
                nn.Conv3d(3, 64, 3, (1, 2, 2), 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 128, 3, (1, 2, 2), 1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(128, 1024),
            )

        # AU Predictor (from recognition features)
        if use_au_predictor:
            self.au_predictor = AUPredictorWithEmotion(
                input_dim=1024,
                num_au=17,
                num_emotions=4
            )
        else:
            self.au_predictor = None

        # =====================================
        # Generation Module (FOMM)
        # =====================================
        # Load pretrained FOMM
        if fomm_checkpoint is not None:
            motion_extractor, generator = load_pretrained_fomm(
                fomm_checkpoint, device=device
            )
        else:
            # Use simplified FOMM for demo
            from generation.fomm_adapter import MotionExtractor, Generator
            motion_extractor = MotionExtractor()
            generator = Generator()
            motion_extractor = motion_extractor.to(device)
            generator = generator.to(device)

        # AU Controller (for generation)
        self.au_controller = AUController(num_au=17, num_keypoints=10)

        # Temporal Modulation (for generation)
        self.temporal_modulation = TemporalModulation()

        # FOMM Adapter (integrates AU into FOMM)
        self.fomm_adapter = FOMMAdapter(
            motion_extractor, generator,
            num_au=17, num_keypoints=10
        )

    def recognize(self, video):
        """
        Recognize emotion and AU from input video.

        Args:
            video (torch.Tensor): Input video, shape (B, C, T, H, W)

        Returns:
            emotion_class (torch.Tensor): Emotion predictions
            au_activation (torch.Tensor): AU activations
            apex_time (torch.Tensor): Apex frame indices
        """
        B, C, T, H, W = video.shape

        # Extract features
        if hasattr(self, 'censor') and self.censor is not None:
            # Use Censor for feature extraction
            # Get fused features from Censor's fusion layer
            features = self.censor(video, return_features=True)
        else:
            # Use placeholder feature extractor
            features = self.feature_extractor(video)  # (B, 1024)

        # Predict emotion (placeholder - actual implementation uses MoE)
        emotion_class = torch.randint(0, 4, (B,)).to(self.device)  # Random for demo

        # Predict AU activations
        if self.au_predictor is not None:
            au_activation = self.au_predictor(features, emotion_class)  # (B, 17)
        else:
            au_activation = torch.zeros(B, 17).to(self.device)

        # Detect apex (placeholder - actual uses CASANet)
        apex_time = torch.tensor([T // 2] * B).to(self.device)

        return emotion_class, au_activation, apex_time

    def generate(self, neutral_face, emotion_class, au_activation=None,
                 intensity=None, num_frames=16):
        """
        Generate micro-expression video.

        Args:
            neutral_face (torch.Tensor): Neutral face, shape (B, C, H, W)
            emotion_class (torch.Tensor): Emotion indices, shape (B,)
            au_activation (torch.Tensor, optional): AU activations, shape (B, 17)
            intensity (torch.Tensor, optional): Intensity values, shape (B, 1)
            num_frames (int): Number of frames

        Returns:
            generated_video (torch.Tensor): Generated ME video, shape (B, C, T, H, W)
        """
        B = neutral_face.shape[0]

        # If no AU provided, use emotion-based default AU
        if au_activation is None:
            # Get default AU from emotion (using au_predictor's prior)
            if self.au_predictor is not None:
                dummy_features = torch.zeros(B, 1024).to(self.device)
                au_activation = self.au_predictor(dummy_features, emotion_class)
            else:
                au_activation = self._get_default_au(emotion_class)

        # Generate temporal curve
        temporal_curve = self.temporal_modulation.generate_batch(
            batch_size=B,
            num_frames=num_frames,
            emotion_class=emotion_class,
            intensity=intensity
        )  # (B, T)

        # Generate video using FOMM adapter
        generated_video = self.fomm_adapter.generate_with_temporal_curve(
            neutral_face, au_activation, temporal_curve[0], num_frames
        )

        return generated_video

    def _get_default_au(self, emotion_class):
        """Get default AU configuration for each emotion."""
        B = emotion_class.shape[0]

        # Default AU configurations
        default_au = torch.zeros(B, 17).to(self.device)

        for b in range(B):
            emotion = emotion_class[b].item()

            if emotion == 0:  # Happiness
                default_au[b, 4] = 0.6  # AU6
                default_au[b, 8] = 0.8  # AU12
                default_au[b, 15] = 0.2  # AU25

            elif emotion == 1:  # Surprise
                default_au[b, 0] = 0.7  # AU1
                default_au[b, 1] = 0.7  # AU2
                default_au[b, 3] = 0.8  # AU5
                default_au[b, 15] = 0.5  # AU25

            elif emotion == 2:  # Disgust
                default_au[b, 2] = 0.5  # AU4
                default_au[b, 6] = 0.7  # AU9
                default_au[b, 7] = 0.4  # AU10
                default_au[b, 11] = 0.3  # AU17

            elif emotion == 3:  # Repression
                default_au[b, 9] = 0.6  # AU14
                default_au[b, 11] = 0.4  # AU17
                default_au[b, 2] = 0.3  # AU4

        return default_au

    def forward(self, video, neutral_face=None, mode='both'):
        """
        Full pipeline: recognize + generate.

        Args:
            video (torch.Tensor): Input video for recognition
            neutral_face (torch.Tensor, optional): Neutral face for generation
            mode (str): 'recognize', 'generate', or 'both'

        Returns:
            dict: Results depending on mode
        """
        results = {}

        if mode == 'recognize' or mode == 'both':
            emotion, au, apex = self.recognize(video)
            results['emotion'] = emotion
            results['au_activation'] = au
            results['apex_time'] = apex

        if mode == 'generate' or mode == 'both':
            if neutral_face is None:
                # Use first frame of input video as neutral face
                neutral_face = video[:, :, 0, :, :]

            if mode == 'both':
                # Use recognized emotion and AU
                emotion = results['emotion']
                au = results['au_activation']

            num_frames = video.shape[2]
            generated = self.generate(neutral_face, emotion, au, num_frames=num_frames)
            results['generated_video'] = generated

        return results


# =============================================================================
# Demo and Test Functions
# =============================================================================

def demo_censor_fomm():
    """Demo the Censor-FOMM system."""
    print("\n" + "="*60)
    print("Censor-FOMM Demo")
    print("="*60)

    # Create model
    model = CensorFOMM(fomm_checkpoint=None, device='cpu')

    # Create dummy inputs
    B, C, T, H, W = 2, 3, 16, 224, 224
    video = torch.randn(B, C, T, H, W)
    neutral_face = torch.randn(B, C, H, W)

    # Test recognition
    print("\n[Recognition Test]")
    emotion, au, apex = model.recognize(video)
    print(f"  Emotion: {emotion}")
    print(f"  AU activation shape: {au.shape}")
    print(f"  Apex time: {apex}")

    # Test generation
    print("\n[Generation Test]")
    emotion_class = torch.tensor([0, 1])  # Happiness, Surprise
    generated = model.generate(neutral_face, emotion_class, num_frames=16)
    print(f"  Generated video shape: {generated.shape}")

    # Test full pipeline
    print("\n[Full Pipeline Test]")
    results = model.forward(video, neutral_face, mode='both')
    print(f"  Results keys: {results.keys()}")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_censor_fomm()