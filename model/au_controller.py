# =============================================================================
# AU Controller -- Convert AU activations to FOMM keypoints
# =============================================================================
# Purpose: Given 17 AU activations, generate keypoint displacements for FOMM.
#
# FOMM (First Order Motion Model) uses 10 keypoints for face motion:
#   kp0-1: Eyebrow regions (left/right)
#   kp2-3: Eye regions (left/right)
#   kp4: Nose
#   kp5-6: Lip corners (left/right)
#   kp7: Chin/Jaw
#   kp8-9: Face outline
#
# AU → Keypoint mapping:
#   Each AU corresponds to specific keypoint movements.
#   For example, AU12 (Lip Corner Puller) → kp5, kp6 move outward and up.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class AUController(nn.Module):
    """
    AU Controller module.

    Converts 17 AU activations to 10 FOMM keypoint displacements.

    Architecture:
        Input: au_activation (B, 17), emotion_class (B,), intensity (B, 1)
          -> AU → Keypoint mapping: Linear(17 → 20) (10 keypoints × 2 directions)
          -> Emotion conditioning: Embedding(4 → 10)
          -> Intensity modulation: scale displacement
          -> Output: keypoint_displacement (B, 10, 2)
    """

    def __init__(self, num_au=17, num_keypoints=10, num_emotions=4):
        super().__init__()

        self.num_au = num_au
        self.num_keypoints = num_keypoints
        self.num_emotions = num_emotions

        # AU → Keypoint displacement mapping
        # 17 AU → 10 keypoints × 2 (x, y displacement)
        self.au_to_kp = nn.Linear(num_au, num_keypoints * 2)

        # Emotion conditioning (optional refinement)
        self.emotion_embed = nn.Embedding(num_emotions, num_keypoints)

        # Intensity scaling (learnable base intensity)
        self.base_intensity = nn.Parameter(torch.tensor(0.5))

        # AU → Keypoint prior mapping (hand-designed)
        # This provides a strong initialization for the mapping
        self._init_au_kp_prior()

    def _init_au_kp_prior(self):
        """
        Initialize AU → Keypoint mapping with hand-designed prior.

        Keypoint indices:
          0-1: eyebrows (left/right)
          2-3: eyes (left/right)
          4: nose
          5-6: mouth corners (left/right)
          7: chin
          8-9: face outline

        AU → Keypoint effects:
          AU1 (Inner Brow Raiser): kp0, kp1 move up
          AU2 (Outer Brow Raiser): kp0, kp1 move up (outer part)
          AU4 (Brow Lowerer): kp0, kp1 move down and inward
          AU5 (Upper Lid Raiser): kp2, kp3 move up (eyes open)
          AU6 (Cheek Raiser): kp2, kp3 squeeze (squint)
          AU9 (Nose Wrinkler): kp4 moves up
          AU12 (Lip Corner Puller): kp5, kp6 move up and outward
          AU14 (Dimpler): kp5, kp6 move inward
          AU17 (Chin Raiser): kp7 moves up
        """
        with torch.no_grad():
            # Create prior mapping matrix (17 AU → 20 outputs)
            prior = torch.zeros(17, 20)

            # AU1 (index 0): Inner Brow Raiser
            # kp0_y, kp1_y: eyebrows move up
            prior[0, 0 * 2 + 1] = 0.3   # kp0 y-displacement
            prior[0, 1 * 2 + 1] = 0.3   # kp1 y-displacement

            # AU2 (index 1): Outer Brow Raiser
            prior[1, 0 * 2 + 1] = 0.2   # kp0 y-displacement (outer)
            prior[1, 1 * 2 + 1] = 0.2   # kp1 y-displacement (outer)

            # AU4 (index 2): Brow Lowerer
            prior[2, 0 * 2 + 1] = -0.2  # kp0 y-displacement (down)
            prior[2, 1 * 2 + 1] = -0.2  # kp1 y-displacement (down)

            # AU5 (index 3): Upper Lid Raiser (eyes open)
            prior[3, 2 * 2 + 1] = -0.3  # kp2 y-displacement (lid up = eye bigger)
            prior[3, 3 * 2 + 1] = -0.3  # kp3 y-displacement

            # AU6 (index 4): Cheek Raiser (squint/smile)
            prior[4, 2 * 2 + 1] = 0.2   # kp2 y-displacement (squeeze)
            prior[4, 3 * 2 + 1] = 0.2   # kp3 y-displacement

            # AU9 (index 6): Nose Wrinkler
            prior[6, 4 * 2 + 1] = -0.1  # kp4 y-displacement (nose up)

            # AU12 (index 8): Lip Corner Puller (smile)
            prior[8, 5 * 2 + 0] = 0.2   # kp5 x-displacement (outward)
            prior[8, 5 * 2 + 1] = 0.3   # kp5 y-displacement (up)
            prior[8, 6 * 2 + 0] = -0.2  # kp6 x-displacement (outward, negative because right)
            prior[8, 6 * 2 + 1] = 0.3   # kp6 y-displacement (up)

            # AU14 (index 9): Dimpler (mouth corners tighten)
            prior[9, 5 * 2 + 0] = -0.1  # kp5 x-displacement (inward)
            prior[9, 6 * 2 + 0] = 0.1   # kp6 x-displacement (inward)

            # AU17 (index 11): Chin Raiser Lower
            prior[11, 7 * 2 + 1] = -0.2  # kp7 y-displacement (chin up)

            # AU25 (index 15): Lips Part (mouth open)
            prior[15, 7 * 2 + 1] = 0.2   # kp7 y-displacement (chin down)

            # Set the prior as initial weights
            self.au_to_kp.weight.copy_(prior)

    def forward(self, au_activation, emotion_class=None, intensity=None):
        """
        Args:
            au_activation (torch.Tensor): AU activations, shape (B, 17)
            emotion_class (torch.Tensor, optional): Emotion class indices, shape (B,)
            intensity (torch.Tensor, optional): Intensity parameter, shape (B, 1)

        Returns:
            keypoint_displacement (torch.Tensor): Keypoint displacements,
                                                  shape (B, 10, 2)
        """
        B = au_activation.shape[0]

        # AU → Keypoint displacement
        kp_displacement = self.au_to_kp(au_activation)  # (B, 20)
        kp_displacement = kp_displacement.view(B, self.num_keypoints, 2)  # (B, 10, 2)

        # Emotion conditioning (optional)
        if emotion_class is not None:
            emotion_cond = self.emotion_embed(emotion_class)  # (B, 10)
            emotion_cond = emotion_cond.view(B, self.num_keypoints, 1)  # (B, 10, 1)
            kp_displacement = kp_displacement + emotion_cond * 0.1

        # Intensity modulation
        if intensity is not None:
            # Scale displacement by intensity
            intensity_scale = intensity.view(B, 1, 1)  # (B, 1, 1)
            kp_displacement = kp_displacement * intensity_scale
        else:
            # Use base intensity
            kp_displacement = kp_displacement * self.base_intensity

        return kp_displacement


class TemporalModulation(nn.Module):
    """
    Temporal modulation for micro-expression dynamics.

    Generates onset-apex-offset curve to modulate keypoint displacement
    across frames.

    Micro-expression temporal characteristics:
      - Onset: 0-30% of duration, slow rise
      - Apex: 30-50% of duration, peak hold
      - Offset: 50-100% of duration, slow decline
      - Total duration: typically 1/25 to 1/5 second
    """

    def __init__(self, onset_ratio=0.3, apex_ratio=0.2, offset_ratio=0.5):
        super().__init__()

        self.onset_ratio = onset_ratio
        self.apex_ratio = apex_ratio
        self.offset_ratio = offset_ratio

        # Duration encoder (maps emotion to typical duration)
        self.duration_encoder = nn.Embedding(4, 1)  # 4 emotions → duration factor

    def forward(self, num_frames, emotion_class=None, intensity=1.0):
        """
        Generate temporal modulation curve.

        Args:
            num_frames (int): Number of frames in the generated video
            emotion_class (torch.Tensor, optional): Emotion indices, shape (B,)
            intensity (float or torch.Tensor): Peak intensity, range [0, 1]

        Returns:
            modulation_curve (torch.Tensor): Modulation coefficients for each frame,
                                             shape (B, num_frames) or (num_frames,)
        """
        T = num_frames

        # Create base curve
        curve = torch.zeros(T)

        # Onset phase (0 to onset_ratio * T)
        onset_end = int(T * self.onset_ratio)
        for t in range(onset_end):
            # Smooth rise (square root for gradual start)
            progress = t / onset_end
            curve[t] = intensity * (progress ** 0.5)

        # Apex phase (onset_end to onset_end + apex_ratio * T)
        apex_end = int(T * (self.onset_ratio + self.apex_ratio))
        curve[onset_end:apex_end] = intensity

        # Offset phase (apex_end to T)
        for t in range(apex_end, T):
            progress = (t - apex_end) / (T - apex_end)
            # Smooth decline (power > 1 for gradual end)
            curve[t] = intensity * (1 - progress ** 0.7)

        return curve

    def generate_batch(self, batch_size, num_frames, emotion_class, intensity=None):
        """
        Generate batch of temporal curves.

        Args:
            batch_size (int): Number of samples
            num_frames (int): Number of frames
            emotion_class (torch.Tensor): Emotion indices, shape (B,)
            intensity (torch.Tensor, optional): Intensity values, shape (B,)

        Returns:
            curves (torch.Tensor): Batch of curves, shape (B, num_frames)
        """
        B = batch_size
        T = num_frames

        curves = torch.zeros(B, T)

        for b in range(B):
            # Get duration factor from emotion
            duration_factor = self.duration_encoder(emotion_class[b]).squeeze()

            # Adjust curve shape based on emotion
            # Surprise is faster, happiness is more gradual
            if emotion_class[b] == 1:  # Surprise
                # Faster onset
                onset_ratio = 0.2
                apex_ratio = 0.3
            elif emotion_class[b] == 0:  # Happiness
                # More gradual
                onset_ratio = 0.35
                apex_ratio = 0.15
            else:
                onset_ratio = self.onset_ratio
                apex_ratio = self.apex_ratio

            # Generate curve
            int_val = intensity[b] if intensity is not None else 1.0
            curves[b] = self._generate_single(T, onset_ratio, apex_ratio, int_val)

        return curves

    def _generate_single(self, T, onset_ratio, apex_ratio, intensity):
        """Generate single temporal curve."""
        curve = torch.zeros(T)

        onset_end = int(T * onset_ratio)
        apex_end = int(T * (onset_ratio + apex_ratio))

        for t in range(onset_end):
            curve[t] = intensity * (t / onset_end) ** 0.5

        curve[onset_end:apex_end] = intensity

        for t in range(apex_end, T):
            progress = (t - apex_end) / (T - apex_end)
            curve[t] = intensity * (1 - progress ** 0.7)

        return curve


# =============================================================================
# AU-Keypoint Reference
# =============================================================================

# FOMM Keypoint meaning
FOMM_KEYPOINT_NAMES = [
    'left_eyebrow',
    'right_eyebrow',
    'left_eye',
    'right_eye',
    'nose',
    'left_mouth_corner',
    'right_mouth_corner',
    'chin',
    'left_face_outline',
    'right_face_outline',
]

# AU → Keypoint summary
AU_KEYPOINT_EFFECTS = {
    'AU1': {'keypoints': [0, 1], 'direction': 'up', 'region': 'eyebrow_inner'},
    'AU2': {'keypoints': [0, 1], 'direction': 'up', 'region': 'eyebrow_outer'},
    'AU4': {'keypoints': [0, 1], 'direction': 'down', 'region': 'eyebrow'},
    'AU5': {'keypoints': [2, 3], 'direction': 'open', 'region': 'eye'},
    'AU6': {'keypoints': [2, 3], 'direction': 'squeeze', 'region': 'eye'},
    'AU9': {'keypoints': [4], 'direction': 'up', 'region': 'nose'},
    'AU12': {'keypoints': [5, 6], 'direction': 'up_out', 'region': 'mouth'},
    'AU14': {'keypoints': [5, 6], 'direction': 'in', 'region': 'mouth'},
    'AU17': {'keypoints': [7], 'direction': 'up', 'region': 'chin'},
    'AU25': {'keypoints': [7], 'direction': 'down', 'region': 'chin'},
}