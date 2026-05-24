# =============================================================================
# FOMM Loader and Adapter -- Load pretrained FOMM and adapt for ME generation
# =============================================================================
# First Order Motion Model (FOMM) reference:
#   Paper: "First Order Motion Model for Image Animation"
#   Authors: Aliaksandr Siarohin et al., NeurIPS 2019
#   GitHub: https://github.com/AliaksandrSiarohin/first-order-model
#
# FOMM consists of two modules:
#   1. Motion Extractor: Detects keypoints and generates motion field
#   2. Generator: Generates animated images from source + motion
#
# Our adaptation:
#   - Keep Motion Extractor frozen (already learned face motion)
#   - Fine-tune Generator for micro-expression subtlety
#   - Inject AU conditions for fine-grained control
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# =============================================================================
# FOMM Architecture (simplified for integration)
# =============================================================================

class MotionExtractor(nn.Module):
    """
    FOMM Motion Extractor (simplified).

    Detects keypoints from driving video and generates motion field.

    Note: This is a simplified version for integration.
    Full FOMM implementation should be loaded from pretrained weights.
    """

    def __init__(self, num_keypoints=10, num_channels=3):
        super().__init__()

        self.num_keypoints = num_keypoints

        # Keypoint detector (simplified CNN)
        self.kp_detector = nn.Sequential(
            nn.Conv2d(num_channels, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Keypoint prediction heads
        self.kp_head = nn.Linear(256, num_keypoints * 2)  # x, y for each kp
        self.jacobian_head = nn.Linear(256, num_keypoints * 4)  # 2x2 Jacobian

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image or video frame, shape (B, C, H, W)

        Returns:
            keypoints (dict): Keypoint positions and Jacobians
        """
        B = x.shape[0]

        # Detect keypoints
        features = self.kp_detector(x).squeeze(-1).squeeze(-1)  # (B, 256)

        # Keypoint positions
        kp_positions = self.kp_head(features).view(B, self.num_keypoints, 2)  # (B, 10, 2)
        kp_positions = F.softmax(kp_positions.view(B, -1), dim=1).view(B, self.num_keypoints, 2)

        # Jacobians (local affine transformations)
        jacobians = self.jacobian_head(features).view(B, self.num_keypoints, 2, 2)  # (B, 10, 2, 2)

        return {
            'keypoints': kp_positions,
            'jacobians': jacobians
        }


class DenseMotionNetwork(nn.Module):
    """
    FOMM Dense Motion Network.

    Generates dense motion field from keypoints.
    """

    def __init__(self, num_keypoints=10, num_channels=3, scale_factor=0.25):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.scale_factor = scale_factor

        # Motion estimation network
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(num_channels + num_keypoints * 2, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Deformation field generator
        self.deformation_gen = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1),  # x, y deformation
        )

        # Occlusion map generator
        self.occlusion_gen = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, source_kp, driving_kp, source_image):
        """
        Generate dense motion field from keypoint differences.

        Args:
            source_kp (dict): Source keypoints
            driving_kp (dict): Driving keypoints (with AU modulation)
            source_image (torch.Tensor): Source image, shape (B, C, H, W)

        Returns:
            motion_field (torch.Tensor): Dense motion field
            occlusion_map (torch.Tensor): Occlusion map
        """
        B, C, H, W = source_image.shape

        # Compute keypoint differences
        kp_diff = driving_kp['keypoints'] - source_kp['keypoints']  # (B, 10, 2)

        # Create sparse motion representation
        # This is simplified - full FOMM uses more sophisticated sparse motion

        # Generate dense motion
        # Resize for processing
        H_s, W_s = int(H * self.scale_factor), int(W * self.scale_factor)
        source_small = F.interpolate(source_image, size=(H_s, W_s), mode='bilinear')

        # Create motion input
        motion_input = self._create_motion_input(source_small, kp_diff)

        # Encode motion
        motion_features = self.motion_encoder(motion_input)

        # Generate deformation field
        deformation = self.deformation_gen(motion_features)  # (B, 2, H_s, W_s)

        # Upsample to full resolution
        motion_field = F.interpolate(deformation, size=(H, W), mode='bilinear')

        # Generate occlusion map
        occlusion = self.occlusion_gen(motion_features)  # (B, 1, H_s, W_s)
        occlusion_map = F.interpolate(occlusion, size=(H, W), mode='bilinear')

        return motion_field, occlusion_map

    def _create_motion_input(self, source, kp_diff):
        """Create motion input from source and keypoint differences."""
        B, C, H, W = source.shape

        # Spread keypoint differences across spatial dimensions
        kp_diff_expanded = kp_diff.view(B, self.num_keypoints * 2, 1, 1)
        kp_diff_expanded = kp_diff_expanded.expand(-1, -1, H, W)

        # Concat with source
        motion_input = torch.cat([source, kp_diff_expanded], dim=1)

        return motion_input


class Generator(nn.Module):
    """
    FOMM Generator.

    Generates animated image from source + motion field.
    """

    def __init__(self, num_channels=3):
        super().__init__()

        # Warping network (use motion field to warp source)
        self.warping = nn.Identity()  # Simplified - actual warping uses motion_field

        # Inpainting network (fill in occluded regions)
        self.inpainting = nn.Sequential(
            nn.Conv2d(num_channels * 2, 64, 7, 1, 3),  # source + warped
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_channels, 7, 1, 3),
        )

    def forward(self, source_image, motion_field, occlusion_map):
        """
        Generate animated image.

        Args:
            source_image (torch.Tensor): Source image, shape (B, C, H, W)
            motion_field (torch.Tensor): Motion field, shape (B, 2, H, W)
            occlusion_map (torch.Tensor): Occlusion map, shape (B, 1, H, W)

        Returns:
            generated_image (torch.Tensor): Generated image
        """
        B, C, H, W = source_image.shape

        # Warp source using motion field
        # This is simplified - actual implementation uses grid_sample
        warped = self._warp_image(source_image, motion_field)

        # Combine warped and occlusion
        occlusion_expanded = occlusion_map.expand(-1, C, -1, -1)
        combined = warped * (1 - occlusion_expanded) + source_image * occlusion_expanded

        # Inpainting
        inpainting_input = torch.cat([source_image, combined], dim=1)
        generated = self.inpainting(inpainting_input)

        # Final blend
        generated_image = generated * occlusion_expanded + combined * (1 - occlusion_expanded)

        return generated_image

    def _warp_image(self, image, motion_field):
        """Warp image using motion field."""
        B, C, H, W = image.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W)
        )
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0).expand(B, -1, -1, -1)
        grid = grid.to(image.device)

        # Add motion field
        motion_field_normalized = motion_field.permute(0, 2, 3, 1)  # (B, H, W, 2)
        motion_field_normalized = motion_field_normalized / (H / 2)  # Normalize

        sampling_grid = grid + motion_field_normalized

        # Warp using grid_sample
        warped = F.grid_sample(image, sampling_grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)

        return warped


# =============================================================================
# FOMM Loader
# =============================================================================

def load_pretrained_fomm(checkpoint_path, device='cuda'):
    """
    Load pretrained FOMM model.

    Args:
        checkpoint_path (str): Path to FOMM checkpoint
        device (str): Device to load model

    Returns:
        motion_extractor (nn.Module): Motion extractor (frozen)
        generator (nn.Module): Generator (for fine-tuning)
    """
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] FOMM checkpoint not found: {checkpoint_path}")
        print("[Info] Using simplified FOMM architecture instead")
        motion_extractor = MotionExtractor()
        generator = Generator()
        return motion_extractor.to(device), generator.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create models
    motion_extractor = MotionExtractor()
    generator = Generator()

    # Load weights (adapt to actual FOMM checkpoint structure)
    if 'motion_extractor' in checkpoint:
        motion_extractor.load_state_dict(checkpoint['motion_extractor'])
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])

    # Freeze motion extractor
    for param in motion_extractor.parameters():
        param.requires_grad = False
    motion_extractor.eval()

    # Generator is trainable for fine-tuning
    generator.train()

    print(f"[FOMM] Loaded pretrained model from {checkpoint_path}")
    print(f"[FOMM] Motion extractor frozen, Generator trainable")

    return motion_extractor.to(device), generator.to(device)


# =============================================================================
# FOMM Adapter for Micro-Expression
# =============================================================================

class FOMMAdapter(nn.Module):
    """
    FOMM Adapter for Micro-Expression Generation.

    Adapts pretrained FOMM for micro-expression by:
      1. Injecting AU conditions into motion estimation
      2. Modulating motion magnitude for ME subtlety
      3. Adding temporal consistency
    """

    def __init__(self, pretrained_motion_extractor, pretrained_generator,
                 num_au=17, num_keypoints=10):
        super().__init__()

        self.num_au = num_au
        self.num_keypoints = num_keypoints

        # Use pretrained modules
        self.motion_extractor = pretrained_motion_extractor
        self.generator = pretrained_generator

        # Freeze motion extractor
        for param in self.motion_extractor.parameters():
            param.requires_grad = False

        # AU conditioning layer (inject AU into motion)
        self.au_condition = nn.Linear(num_au, num_keypoints * 2)

        # Motion magnitude scaling (ME has smaller motion)
        self.magnitude_scale = nn.Parameter(torch.tensor(0.3))  # ME: ~30% of normal motion

        # Dense motion network (new, for AU-driven motion)
        self.dense_motion = DenseMotionNetwork(num_keypoints)

    def forward(self, neutral_face, au_activation, emotion_class=None,
                intensity=None, num_frames=16):
        """
        Generate micro-expression video from neutral face and AU activation.

        Args:
            neutral_face (torch.Tensor): Neutral face image, shape (B, C, H, W)
            au_activation (torch.Tensor): AU activations, shape (B, 17)
            emotion_class (torch.Tensor, optional): Emotion indices
            intensity (torch.Tensor, optional): Intensity values
            num_frames (int): Number of frames to generate

        Returns:
            generated_video (torch.Tensor): Generated video, shape (B, C, T, H, W)
        """
        B, C, H, W = neutral_face.shape
        T = num_frames

        # Get source keypoints from neutral face
        with torch.no_grad():
            source_kp = self.motion_extractor(neutral_face)

        # Generate AU-driven keypoint displacement
        au_displacement = self.au_condition(au_activation)  # (B, 20)
        au_displacement = au_displacement.view(B, self.num_keypoints, 2)  # (B, 10, 2)

        # Scale for micro-expression subtlety
        au_displacement = au_displacement * self.magnitude_scale

        # Apply intensity modulation
        if intensity is not None:
            au_displacement = au_displacement * intensity.view(B, 1, 1)

        # Generate frames
        generated_frames = []

        # Temporal modulation (simplified - on/off for now)
        for t in range(T):
            # Compute frame-specific displacement
            # Onset: gradual increase, Apex: peak, Offset: gradual decrease
            temporal_factor = self._get_temporal_factor(t, T)

            # Apply temporal modulation to displacement
            frame_displacement = au_displacement * temporal_factor

            # Create driving keypoints
            driving_kp = {
                'keypoints': source_kp['keypoints'] + frame_displacement,
                'jacobians': source_kp['jacobians']  # Keep original Jacobians
            }

            # Generate dense motion
            motion_field, occlusion_map = self.dense_motion(
                source_kp, driving_kp, neutral_face
            )

            # Generate frame
            frame = self.generator(neutral_face, motion_field, occlusion_map)
            generated_frames.append(frame)

        # Stack frames into video
        generated_video = torch.stack(generated_frames, dim=2)  # (B, C, T, H, W)

        return generated_video

    def _get_temporal_factor(self, t, T):
        """
        Get temporal modulation factor for frame t.

        Micro-expression dynamics: onset-apex-offset
        """
        # Onset: 0-30% of frames, gradual rise
        if t < T * 0.3:
            return (t / (T * 0.3)) ** 0.5

        # Apex: 30-50% of frames, peak
        elif t < T * 0.5:
            return 1.0

        # Offset: 50-100% of frames, gradual decline
        else:
            progress = (t - T * 0.5) / (T * 0.5)
            return 1 - progress ** 0.7

    def generate_with_temporal_curve(self, neutral_face, au_activation,
                                     temporal_curve, num_frames=16):
        """
        Generate video with explicit temporal curve.

        Args:
            neutral_face (torch.Tensor): Neutral face
            au_activation (torch.Tensor): AU activations
            temporal_curve (torch.Tensor): Temporal modulation, shape (T,)
            num_frames (int): Number of frames

        Returns:
            generated_video (torch.Tensor): Generated video
        """
        B, C, H, W = neutral_face.shape
        T = num_frames

        # Get source keypoints
        with torch.no_grad():
            source_kp = self.motion_extractor(neutral_face)

        # Generate AU displacement
        au_displacement = self.au_condition(au_activation)
        au_displacement = au_displacement.view(B, self.num_keypoints, 2)
        au_displacement = au_displacement * self.magnitude_scale

        # Generate frames with temporal curve
        generated_frames = []

        for t in range(T):
            temporal_factor = temporal_curve[t]

            frame_displacement = au_displacement * temporal_factor

            driving_kp = {
                'keypoints': source_kp['keypoints'] + frame_displacement,
                'jacobians': source_kp['jacobians']
            }

            motion_field, occlusion_map = self.dense_motion(
                source_kp, driving_kp, neutral_face
            )

            frame = self.generator(neutral_face, motion_field, occlusion_map)
            generated_frames.append(frame)

        generated_video = torch.stack(generated_frames, dim=2)

        return generated_video