# =============================================================================
# Generation Loss Functions
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroExpressionGenerationLoss(nn.Module):
    """
    Loss function for micro-expression generation.

    Components:
      1. Reconstruction loss (pixel-level)
      2. Perceptual loss (feature-level)
      3. AU consistency loss
      4. Temporal consistency loss
      5. Motion magnitude constraint (ME subtlety)
    """

    def __init__(self, perceptual_weight=0.5, au_weight=0.3,
                 temporal_weight=0.2, magnitude_limit=0.3):
        super().__init__()

        self.perceptual_weight = perceptual_weight
        self.au_weight = au_weight
        self.temporal_weight = temporal_weight
        self.magnitude_limit = magnitude_limit

        # Perceptual loss (use VGG features if available)
        self.perceptual_net = None  # Can load VGG later

    def forward(self, generated, real, au_pred=None, au_target=None):
        """
        Compute total generation loss.

        Args:
            generated (torch.Tensor): Generated video, shape (B, C, T, H, W)
            real (torch.Tensor): Real video, shape (B, C, T, H, W)
            au_pred (torch.Tensor, optional): Predicted AU, shape (B, 17)
            au_target (torch.Tensor, optional): Target AU, shape (B, 17)

        Returns:
            losses (dict): Dictionary of individual losses
            total_loss (torch.Tensor): Total weighted loss
        """
        losses = {}

        # 1. Reconstruction loss (L1)
        losses['reconstruction'] = F.l1_loss(generated, real)

        # 2. Perceptual loss (feature-level)
        losses['perceptual'] = self._perceptual_loss(generated, real)

        # 3. AU consistency loss
        if au_pred is not None and au_target is not None:
            losses['au'] = F.mse_loss(au_pred, au_target)
        else:
            losses['au'] = torch.tensor(0.0)

        # 4. Temporal consistency loss (frame smoothness)
        losses['temporal'] = self._temporal_consistency_loss(generated)

        # 5. Motion magnitude constraint
        losses['magnitude'] = self._magnitude_constraint(generated, real)

        # Total loss
        total_loss = (
            losses['reconstruction'] +
            self.perceptual_weight * losses['perceptual'] +
            self.au_weight * losses['au'] +
            self.temporal_weight * losses['temporal'] +
            losses['magnitude']
        )

        return losses, total_loss

    def _perceptual_loss(self, generated, real):
        """
        Compute perceptual loss using feature similarity.

        Uses VGG features if available, otherwise uses simple CNN features.
        """
        if self.perceptual_net is None:
            # Simple feature extractor
            # Compare mean and std of features
            gen_features = self._extract_simple_features(generated)
            real_features = self._extract_simple_features(real)
            return F.mse_loss(gen_features, real_features)
        else:
            # Use VGG features
            return self._vgg_perceptual_loss(generated, real)

    def _extract_simple_features(self, video):
        """Extract simple features from video."""
        B, C, T, H, W = video.shape

        # Compute frame-level features
        frames = video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # Mean and std of each frame
        features = []
        for t in range(T):
            frame = frames[:, t]  # (B, C, H, W)
            frame_mean = frame.mean(dim=[2, 3])  # (B, C)
            frame_std = frame.std(dim=[2, 3])  # (B, C)
            features.append(torch.cat([frame_mean, frame_std], dim=1))  # (B, 2C)

        features = torch.stack(features, dim=1)  # (B, T, 2C)
        return features.mean(dim=1)  # (B, 2C)

    def _vgg_perceptual_loss(self, generated, real):
        """Compute VGG perceptual loss (placeholder)."""
        # This would use pretrained VGG features
        # For now, return 0
        return torch.tensor(0.0)

    def _temporal_consistency_loss(self, video):
        """
        Compute temporal consistency loss.

        Ensures smooth transitions between frames.
        """
        B, C, T, H, W = video.shape

        if T < 2:
            return torch.tensor(0.0)

        # Compute frame differences
        frame_diff = video[:, :, 1:, :, :] - video[:, :, :-1, :, :]

        # Penalize large jumps
        consistency_loss = frame_diff.abs().mean()

        return consistency_loss

    def _magnitude_constraint(self, generated, real):
        """
        Constrain motion magnitude for micro-expression subtlety.

        Micro-expressions have small motion magnitude.
        """
        # Compute motion magnitude (difference from neutral frame)
        neutral_frame = real[:, :, 0:1, :, :]  # First frame

        gen_motion = generated - neutral_frame
        real_motion = real - neutral_frame

        # Compute magnitude
        gen_magnitude = gen_motion.abs().mean()
        real_magnitude = real_motion.abs().mean()

        # Constraint: generated should not exceed real by much
        # and should stay within ME subtlety bounds
        magnitude_penalty = torch.relu(gen_magnitude - self.magnitude_limit)

        # Also penalize being too different from real
        magnitude_diff = (gen_magnitude - real_magnitude).abs()

        return magnitude_penalty + 0.1 * magnitude_diff


class GANLoss(nn.Module):
    """
    GAN loss for adversarial training.

    Supports standard GAN, LSGAN, and WGAN.
    """

    def __init__(self, gan_mode='standard', target_real_label=1.0,
                 target_fake_label=0.0):
        super().__init__()

        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'standard':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None  # WGAN uses Wasserstein distance
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def forward(self, prediction, target_is_real):
        """
        Compute GAN loss.

        Args:
            prediction (torch.Tensor): Discriminator output
            target_is_real (bool): Whether target is real or fake

        Returns:
            loss (torch.Tensor): GAN loss
        """
        if self.gan_mode == 'wgangp':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()

        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(prediction)
        return self.loss(prediction, target_tensor)


class Discriminator(nn.Module):
    """
    Discriminator for GAN training.

    Distinguishes real micro-expression videos from generated ones.
    """

    def __init__(self, num_channels=3, use_sigmoid=True):
        super().__init__()

        # 3D CNN for video discrimination
        self.conv_layers = nn.Sequential(
            nn.Conv3d(num_channels, 64, 3, (1, 2, 2), 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 3, (1, 2, 2), 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 3, (1, 2, 2), 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 3, (1, 2, 2), 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

        self.fc = nn.Linear(512, 1)

        self.use_sigmoid = use_sigmoid

    def forward(self, video):
        """
        Discriminate video.

        Args:
            video (torch.Tensor): Video to discriminate, shape (B, C, T, H, W)

        Returns:
            score (torch.Tensor): Real/fake score, shape (B, 1)
        """
        features = self.conv_layers(video).squeeze(-1).squeeze(-1).squeeze(-1)
        score = self.fc(features)

        if self.use_sigmoid:
            score = torch.sigmoid(score)

        return score