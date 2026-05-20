# =============================================================================
# Censor -- Gaze-Driven AU Attention Module
# =============================================================================
# Implements biomimetic gaze-based attention for AU region prioritization:
#   1. GazeEstimator: Estimate gaze direction from eye features
#   2. AURegionAttention: Map gaze to AU region weights
#   3. GazeDrivenAttention: Complete gaze-to-AU attention pipeline
#
# Biological basis:
#   - Eye movements correlate with emotional states
#   - Fear → rapid saccade to threat, dilated pupils
#   - Disgust → avoidance gaze shift
#   - Shame/Deception → gaze aversion (eye contact avoidance)
#   - Genuine smile (Duchenne) → eye contact focus
#   - Fake smile (Non-Duchenne) → mouth focus, eye avoidance
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.defaults import GAZE_ATTENTION_CONFIG


# =============================================================================
# AU Region Definitions (FACS Standard)
# =============================================================================
# AU regions are defined in normalized face coordinates (y from -1 to 1)
# where y=-1 is top of face, y=1 is bottom, x=-1 is left, x=1 is right

AU_REGIONS = {
    'brows': {
        'y_range': (-0.5, -0.35),
        'x_range': (-0.5, 0.5),
        'aus': [1, 2, 4],  # Inner Brow Raiser, Outer Brow Raiser, Brow Lowerer
        'description': '眉毛区域 - AU1/2/4'
    },
    'eyes': {
        'y_range': (-0.35, -0.15),
        'x_range': (-0.6, 0.6),
        'aus': [5, 6, 7, 43, 45],  # Upper Lid Raiser, Cheek Raiser, Lid Tightener, Eye Closure, Blink
        'description': '眼睛区域 - AU5/6/7/43/45'
    },
    'nose': {
        'y_range': (-0.15, 0.1),
        'x_range': (-0.2, 0.2),
        'aus': [9],  # Nose Wrinkler
        'description': '鼻子区域 - AU9'
    },
    'mouth': {
        'y_range': (0.15, 0.45),
        'x_range': (-0.4, 0.4),
        'aus': [10, 12, 14, 15, 17, 20, 23, 24, 25, 26, 27, 28],
        'description': '嘴巴区域 - AU10/12/14/15/17/20/23-28'
    },
}

# Emotion-specific gaze patterns
EMOTION_GAZE_PATTERNS = {
    'happiness_duchenne': {'eyes': 0.5, 'mouth': 0.3, 'brows': 0.2},  # 真笑看眼睛
    'happiness_non_duchenne': {'eyes': 0.2, 'mouth': 0.6, 'brows': 0.2},  # 假笑看嘴巴
    'surprise': {'eyes': 0.4, 'brows': 0.4, 'mouth': 0.2},  # 惊讶看眉眼
    'fear': {'eyes': 0.5, 'brows': 0.3, 'mouth': 0.2},  # 恐惧看眼睛(威胁)
    'disgust': {'nose': 0.4, 'mouth': 0.3, 'eyes': 0.3},  # 厌恶看鼻嘴
    'anger': {'eyes': 0.5, 'brows': 0.3, 'mouth': 0.2},  # 愤怒盯视
    'sadness': {'eyes': 0.3, 'mouth': 0.2, 'brows': 0.5},  # 悲伤低眉
    'contempt': {'eyes': 0.3, 'mouth': 0.5, 'nose': 0.2},  # 蔑视单侧嘴角
    'neutral': {'eyes': 0.25, 'mouth': 0.25, 'brows': 0.25, 'nose': 0.25},  # 中性均匀
}


class GazeEstimator(nn.Module):
    """
    Estimate gaze direction from eye region features.

    Uses eye landmark positions and iris detection to compute:
      - Horizontal gaze (dx): left/right looking
      - Vertical gaze (dy): up/down looking
      - Gaze confidence: how reliable the estimate is

    Architecture:
        Eye features (B, C, H, W) or (B, D)
          -> FC layers -> gaze direction (B, 2) + confidence (B, 1)
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or GAZE_ATTENTION_CONFIG

        self.input_dim = cfg.get('gaze_input_dim', 64)
        hidden_dim = cfg.get('gaze_hidden_dim', 32)

        # Eye region encoder (for raw 4D input)
        self.eye_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Gaze estimation network
        self.gaze_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Gaze direction output (dx, dy) in [-1, 1] normalized
        self.gaze_direction = nn.Linear(hidden_dim // 2, 2)
        self.tanh = nn.Tanh()

        # Gaze confidence output (0-1)
        self.gaze_confidence = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.gaze_net[0].weight)
        nn.init.xavier_uniform_(self.gaze_net[2].weight)
        nn.init.xavier_uniform_(self.gaze_direction.weight)

    def forward(self, eye_features):
        """
        Args:
            eye_features (torch.Tensor): Eye region features
                - Shape (B, D) if already encoded
                - Shape (B, C, H, W) if raw eye region (will be encoded)
        Returns:
            gaze (torch.Tensor): Gaze direction (B, 2), dx and dy in [-1, 1]
            confidence (torch.Tensor): Gaze confidence (B, 1) in [0, 1]
        """
        # Handle raw eye region input (4D tensor)
        if eye_features.dim() == 4:
            # (B, C, H, W) -> encode -> (B, input_dim)
            encoded = self.eye_encoder(eye_features)
            eye_features = encoded.squeeze(-1).squeeze(-1)  # (B, 64)
        elif eye_features.dim() == 2 and eye_features.shape[1] != self.input_dim:
            # Already 2D but wrong dimension - pad or truncate
            if eye_features.shape[1] < self.input_dim:
                # Pad with zeros
                padding = torch.zeros(eye_features.shape[0], self.input_dim - eye_features.shape[1], device=eye_features.device)
                eye_features = torch.cat([eye_features, padding], dim=1)
            else:
                # Truncate
                eye_features = eye_features[:, :self.input_dim]

        # Encode features
        encoded = self.gaze_net(eye_features)

        # Gaze direction (normalized to [-1, 1])
        gaze = self.tanh(self.gaze_direction(encoded))

        # Confidence
        confidence = self.gaze_confidence(encoded)

        return gaze, confidence


class AURegionAttention(nn.Module):
    """
    Map gaze direction to AU region attention weights.

    Creates spatial attention maps based on where the person is looking,
    which correlates with the active AU regions during micro-expressions.

    Architecture:
        Gaze (B, 2) + optional emotion hint (B, E)
          -> MLP -> region weights (B, num_regions)
          -> spatial attention maps (B, 1, H, W)
    """

    def __init__(self, config=None, num_regions=4):
        super().__init__()
        cfg = config or GAZE_ATTENTION_CONFIG

        self.num_regions = num_regions
        self.spatial_size = cfg.get('au_attention_size', 224)

        # Gaze to region weight mapping
        gaze_dim = 2
        emotion_dim = cfg.get('emotion_hint_dim', 11)  # ME categories

        self.region_weight_net = nn.Sequential(
            nn.Linear(gaze_dim + emotion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_regions),
            nn.Softmax(dim=-1)
        )

        # Region center positions (normalized coordinates)
        self.register_buffer('region_centers', self._compute_region_centers())

        # Spatial spread for each region (sigma for Gaussian)
        self.region_sigma = cfg.get('region_sigma', 0.15)

        # Learnable emotion-gaze modulation
        self.emotion_gaze_gate = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # (emotion_weight, gaze_weight)
            nn.Softmax(dim=-1)
        )

    def _compute_region_centers(self):
        """Compute center positions for each AU region."""
        centers = []
        region_names = ['brows', 'eyes', 'nose', 'mouth']

        for name in region_names:
            region = AU_REGIONS[name]
            y_center = (region['y_range'][0] + region['y_range'][1]) / 2
            x_center = (region['x_range'][0] + region['x_range'][1]) / 2
            centers.append([x_center, y_center])

        return torch.tensor(centers, dtype=torch.float32)

    def _create_region_spatial_map(self, center, sigma, H, W, device):
        """
        Create a Gaussian spatial attention map for a region.

        Args:
            center (tuple): (x, y) normalized center position
            sigma (float): Spread of the Gaussian
            H, W (int): Spatial dimensions
        Returns:
            map (torch.Tensor): (1, H, W) attention map
        """
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Gaussian centered at region center
        x_c, y_c = center
        spatial_map = torch.exp(
            -((X - x_c) ** 2 + (Y - y_c) ** 2) / (2 * sigma ** 2)
        )

        return spatial_map.unsqueeze(0)  # (1, H, W)

    def forward(self, gaze, emotion_hint=None, return_weights=True):
        """
        Args:
            gaze (torch.Tensor): Gaze direction (B, 2)
            emotion_hint (torch.Tensor, optional): Emotion probabilities (B, E)
            return_weights (bool): Whether to return region weights
        Returns:
            spatial_attention (torch.Tensor): (B, 1, H, W) spatial attention map
            region_weights (torch.Tensor): (B, num_regions) optional
        """
        B = gaze.shape[0]
        H = W = self.spatial_size
        device = gaze.device

        # Default emotion hint (neutral)
        if emotion_hint is None:
            emotion_hint = torch.zeros(B, 11, device=device)
            emotion_hint[:, 9] = 1.0  # Neutral-ish (sadness as default neutral)

        # Compute emotion-gaze modulation weights
        emo_gate = self.emotion_gaze_gate(emotion_hint)  # (B, 2)

        # Modulated gaze (emotion influence on gaze attention)
        modulated_gaze = gaze * emo_gate[:, 1:2] + torch.zeros_like(gaze) * emo_gate[:, 0:1]

        # Compute region weights from gaze + emotion
        combined = torch.cat([modulated_gaze, emotion_hint], dim=-1)
        region_weights = self.region_weight_net(combined)  # (B, num_regions)

        # Create spatial attention maps
        spatial_attention = torch.zeros(B, 1, H, W, device=device)

        for i, center in enumerate(self.region_centers):
            region_map = self._create_region_spatial_map(
                center.tolist(), self.region_sigma, H, W, device
            )
            # Weight each region map by computed weights
            spatial_attention = spatial_attention + region_weights[:, i:i+1, None, None] * region_map

        # Normalize spatial attention
        spatial_attention = spatial_attention / (spatial_attention.max() + 1e-8)

        if return_weights:
            return spatial_attention, region_weights
        return spatial_attention


class GazeDrivenAttention(nn.Module):
    """
    Complete gaze-driven AU attention pipeline.

    Integrates:
      1. Gaze estimation from eye features
      2. AU region attention mapping
      3. Feature modulation based on gaze

    This module enables the model to focus on AU regions
    that are likely active based on where the person is looking,
    following the biological principle that gaze and emotion are linked.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or GAZE_ATTENTION_CONFIG

        self.gaze_estimator = GazeEstimator(cfg)
        self.au_attention = AURegionAttention(cfg)

        # Eye feature encoder (if input is raw eye region)
        self.eye_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Feature modulation gate
        self.modulation_gate = nn.Sequential(
            nn.Linear(2 + 4, 16),  # gaze + region_weights
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # History tracking for temporal consistency
        self.gaze_history_length = cfg.get('gaze_history_length', 5)
        self.register_buffer('gaze_history', torch.zeros(1, self.gaze_history_length, 2))
        self.register_buffer('history_ptr', torch.tensor(0))

        # Output config
        self.output_attention_strength = cfg.get('output_attention_strength', 0.3)

    def _update_history(self, gaze):
        """Update gaze history buffer for temporal smoothing."""
        B = gaze.shape[0]

        # Expand buffer if needed
        if self.gaze_history.shape[0] < B:
            self.gaze_history = self.gaze_history.expand(B, -1, -1).clone()

        # Update circular buffer (clone to avoid in-place modification)
        ptr = self.history_ptr.item()
        new_history = self.gaze_history.clone()
        new_history[:, ptr] = gaze
        self.gaze_history = new_history
        self.history_ptr = torch.tensor((ptr + 1) % self.gaze_history_length)

    def _get_smoothed_gaze(self):
        """Get temporally smoothed gaze from history."""
        smoothed = self.gaze_history.mean(dim=1)  # Average over history
        return smoothed

    def forward(self, face_features, eye_region=None, emotion_hint=None,
                temporal_smooth=True, return_gaze=False):
        """
        Args:
            face_features (torch.Tensor): Face features (B, C, H, W) or (B, D)
            eye_region (torch.Tensor, optional): Raw eye region (B, 3, H_eye, W_eye)
            emotion_hint (torch.Tensor, optional): Emotion hint (B, E)
            temporal_smooth (bool): Apply temporal smoothing to gaze
            return_gaze (bool): Return gaze estimate
        Returns:
            modulated_features (torch.Tensor): Attention-modulated features
            spatial_attention (torch.Tensor): (B, 1, H, W) attention map
            gaze (torch.Tensor, optional): (B, 2) gaze estimate
        """
        B = face_features.shape[0]
        device = face_features.device

        # Estimate gaze from eye region or face features
        if eye_region is not None:
            eye_feat = self.eye_encoder(eye_region).squeeze(-1).squeeze(-1)
            gaze, confidence = self.gaze_estimator(eye_feat)
        else:
            # Use pooled face features as proxy
            if face_features.dim() == 4:
                face_pooled = F.adaptive_avg_pool2d(face_features, 1).squeeze(-1).squeeze(-1)
            else:
                face_pooled = face_features
            gaze, confidence = self.gaze_estimator(face_pooled[:, :64])

        # Temporal smoothing
        if temporal_smooth:
            self._update_history(gaze)
            gaze = self._get_smoothed_gaze()

        # Compute AU region attention
        spatial_attention, region_weights = self.au_attention(gaze, emotion_hint)

        # Modulate features based on spatial attention
        if face_features.dim() == 4:
            # (B, C, H, W) features
            # Resize attention to match feature spatial size
            H_feat, W_feat = face_features.shape[2:4]
            if H_feat != spatial_attention.shape[2] or W_feat != spatial_attention.shape[3]:
                spatial_attention = F.interpolate(
                    spatial_attention, size=(H_feat, W_feat), mode='bilinear', align_corners=False
                )

            # Compute modulation strength (expand to match spatial dimensions)
            gate_input = torch.cat([gaze, region_weights], dim=-1)
            modulation_strength = self.modulation_gate(gate_input)  # (B, 1)
            modulation_strength = modulation_strength.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

            # Apply attention modulation
            modulated = face_features * (1 + self.output_attention_strength * spatial_attention * modulation_strength)
        else:
            # (B, D) features - apply channel-wise modulation based on region weights
            # This requires a mapping from regions to channels (simplified)
            region_channel_weights = self._region_to_channel_weights(region_weights, face_features.shape[1])
            modulated = face_features * region_channel_weights

        if return_gaze:
            return modulated, spatial_attention, gaze, confidence

        return modulated, spatial_attention

    def _region_to_channel_weights(self, region_weights, num_channels):
        """
        Map region weights to channel weights (simplified heuristic).

        Args:
            region_weights (torch.Tensor): (B, 4) weights for brows/eyes/nose/mouth
            num_channels (int): Number of feature channels
        Returns:
            channel_weights (torch.Tensor): (B, num_channels)
        """
        B = region_weights.shape[0]

        # Simple mapping: distribute channels to regions
        # This can be replaced with learned mapping
        channel_per_region = num_channels // 4
        remainder = num_channels - 4 * channel_per_region

        channel_weights = torch.zeros(B, num_channels, device=region_weights.device)

        for i in range(4):
            start = i * channel_per_region
            end = start + channel_per_region
            channel_weights[:, start:end] = region_weights[:, i:i+1]

        # Handle remainder channels
        if remainder > 0:
            channel_weights[:, -remainder:] = region_weights[:, 0:1]

        return channel_weights


class GazeEmotionCorrelation(nn.Module):
    """
    Analyze correlation between gaze patterns and emotion states.

    Used for:
      1. Detecting genuine vs fake expressions (Duchenne marker)
      2. Identifying deception indicators (gaze aversion)
      3. Enhancing emotion classification with gaze cues
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or GAZE_ATTENTION_CONFIG

        # Duchenne marker detector (真笑 vs 假笑)
        self.duchenne_detector = nn.Sequential(
            nn.Linear(4, 16),  # region_weights input
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Deception indicator (隐瞒/欺骗的注视回避)
        self.deception_detector = nn.Sequential(
            nn.Linear(2, 8),  # gaze deviation
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Gaze deviation threshold (for deception detection)
        self.aversion_threshold = cfg.get('gaze_aversion_threshold', 0.5)

    def forward(self, gaze, region_weights):
        """
        Args:
            gaze (torch.Tensor): (B, 2) gaze direction
            region_weights (torch.Tensor): (B, 4) AU region weights
        Returns:
            duchenne_score (torch.Tensor): (B, 1) probability of genuine smile
            deception_score (torch.Tensor): (B, 1) probability of deception
            gaze_deviation (torch.Tensor): (B, 1) gaze deviation from center
        """
        # Duchenne marker: high eye + mouth weight = genuine
        # eyes=region_weights[:,1], mouth=region_weights[:,3]
        duchenne_score = self.duchenne_detector(region_weights)

        # Gaze deviation from center (aversion indicator)
        gaze_deviation = torch.norm(gaze, dim=-1, keepdim=True)

        # Deception: gaze aversion (deviation > threshold)
        deception_score = self.deception_detector(gaze)
        deception_score = deception_score * (gaze_deviation > self.aversion_threshold).float()

        return duchenne_score, deception_score, gaze_deviation


# =============================================================================
# Factory Functions
# =============================================================================

def create_gaze_attention(config=None):
    """Factory function to create GazeDrivenAttention module."""
    return GazeDrivenAttention(config or GAZE_ATTENTION_CONFIG)


def create_gaze_estimator(config=None):
    """Factory function to create GazeEstimator module."""
    return GazeEstimator(config or GAZE_ATTENTION_CONFIG)


def create_au_region_attention(config=None):
    """Factory function to create AURegionAttention module."""
    return AURegionAttention(config or GAZE_ATTENTION_CONFIG)