# =============================================================================
# AU Predictor -- Predict Action Unit activations from fused features
# =============================================================================
# Purpose: Given the fused features from Censor's dual-pathway encoder,
#          predict the activation intensity of 17 facial Action Units.
#
# Action Units (FACS - Facial Action Coding System):
#   AU1: Inner Brow Raiser
#   AU2: Outer Brow Raiser
#   AU4: Brow Lowerer
#   AU5: Upper Lid Raiser
#   AU6: Cheek Raiser ( Orbicularis Oculi )
#   AU7: Lower Lid Depressor
#   AU9: Nose Wrinkler
#   AU10: Upper Lip Raiser
#   AU12: Lip Corner Puller
#   AU14: Dimpler
#   AU15: Lip Corner Depressor
#   AU17: Chin Raiser Lower
#   AU20: Lip Stretch
#   AU23: Lip Tightener
#   AU24: Lip Pressor
#   AU25: Lips Part
#   AU26: Jaw Drop
#
# Reference: Ekman & Friesen, Facial Action Coding System (FACS), 1978
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class AUPredictor(nn.Module):
    """
    Action Unit Predictor module.

    Predicts 17 AU activation intensities (0-1) from fused features.

    Architecture:
        Input: fused_features (B, 1024) from Censor's fusion layer
          -> FC(1024 -> 256) -> ReLU -> Dropout(0.1)
          -> FC(256 -> 17) -> Sigmoid
          -> Output: au_activation (B, 17)
    """

    def __init__(self, input_dim=1024, num_au=17, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.num_au = num_au

        # Feature compression
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # AU prediction
        self.fc2 = nn.Linear(hidden_dim, num_au)
        self.sigmoid = nn.Sigmoid()  # AU activation range [0, 1]

        # Weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, fused_features):
        """
        Args:
            fused_features (torch.Tensor): Fused features from Censor,
                                           shape (B, 1024)

        Returns:
            au_activation (torch.Tensor): AU activation intensities,
                                          shape (B, 17), range [0, 1]
        """
        # Feature compression
        h = self.fc1(fused_features)  # (B, 256)
        h = self.relu(h)
        h = self.dropout(h)

        # AU prediction
        au_activation = self.fc2(h)  # (B, 17)
        au_activation = self.sigmoid(au_activation)

        return au_activation


class AUPredictorWithEmotion(nn.Module):
    """
    AU Predictor with emotion class conditioning.

    Predicts AU activations conditioned on emotion class,
    which helps generate AU configurations specific to each emotion.

    Architecture:
        Input: fused_features (B, 1024), emotion_class (B,)
          -> Emotion embedding: emotion -> (B, 64)
          -> Concat: features + emotion_embed -> (B, 1088)
          -> FC(1088 -> 256) -> ReLU -> Dropout
          -> FC(256 -> 17) -> Sigmoid
          -> Output: au_activation (B, 17)
    """

    def __init__(self, input_dim=1024, num_au=17, num_emotions=4,
                 hidden_dim=256, emotion_dim=64, dropout=0.1):
        super().__init__()

        self.num_au = num_au
        self.num_emotions = num_emotions

        # Emotion embedding
        self.emotion_embed = nn.Embedding(num_emotions, emotion_dim)

        # Feature + emotion fusion
        self.fc1 = nn.Linear(input_dim + emotion_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # AU prediction
        self.fc2 = nn.Linear(hidden_dim, num_au)
        self.sigmoid = nn.Sigmoid()

        # AU-emotion prior (predefined AU configurations for each emotion)
        # This helps the model learn emotion-specific AU patterns
        self.au_emotion_prior = self._create_au_emotion_prior()

        # Weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def _create_au_emotion_prior(self):
        """
        Create predefined AU configurations for each emotion.

        Based on Ekman's basic emotions and ME research.

        Returns:
            prior (torch.Tensor): (num_emotions, num_au)
        """
        # AU indices:
        # 0: AU1, 1: AU2, 2: AU4, 3: AU5, 4: AU6, 5: AU7
        # 6: AU9, 7: AU10, 8: AU12, 9: AU14, 10: AU15, 11: AU17
        # 12: AU20, 13: AU23, 14: AU24, 15: AU25, 16: AU26

        prior = torch.zeros(self.num_emotions, self.num_au)

        # Happiness: AU6 + AU12
        prior[0, 4] = 0.6   # AU6: Cheek Raiser
        prior[0, 8] = 0.8   # AU12: Lip Corner Puller
        prior[0, 15] = 0.2  # AU25: Lips Part (smile)

        # Surprise: AU1 + AU2 + AU5 + AU25
        prior[1, 0] = 0.7   # AU1: Inner Brow Raiser
        prior[1, 1] = 0.7   # AU2: Outer Brow Raiser
        prior[1, 3] = 0.8   # AU5: Upper Lid Raiser
        prior[1, 15] = 0.5  # AU25: Lips Part

        # Disgust: AU4 + AU9 + AU10 + AU17
        prior[2, 2] = 0.5   # AU4: Brow Lowerer
        prior[2, 6] = 0.7   # AU9: Nose Wrinkler
        prior[2, 7] = 0.4   # AU10: Upper Lip Raiser
        prior[2, 11] = 0.3  # AU17: Chin Raiser

        # Repression/Contempt: AU14 + AU17 + AU4
        prior[3, 9] = 0.6   # AU14: Dimpler
        prior[3, 11] = 0.4  # AU17: Chin Raiser
        prior[3, 2] = 0.3   # AU4: Brow Lowerer (mild)

        return prior

    def forward(self, fused_features, emotion_class):
        """
        Args:
            fused_features (torch.Tensor): Fused features, shape (B, 1024)
            emotion_class (torch.Tensor): Emotion class indices, shape (B,)

        Returns:
            au_activation (torch.Tensor): AU activations, shape (B, 17)
        """
        B = fused_features.shape[0]

        # Emotion embedding
        emotion_embed = self.emotion_embed(emotion_class)  # (B, emotion_dim)

        # Concat features with emotion embedding
        combined = torch.cat([fused_features, emotion_embed], dim=1)  # (B, 1088)

        # Feature compression
        h = self.fc1(combined)  # (B, 256)
        h = self.relu(h)
        h = self.dropout(h)

        # AU prediction
        au_activation = self.fc2(h)  # (B, 17)
        au_activation = self.sigmoid(au_activation)

        # Add emotion prior as a soft constraint
        # This helps enforce emotion-specific AU patterns
        emotion_prior = self.au_emotion_prior[emotion_class]  # (B, 17)
        au_activation = au_activation + 0.2 * emotion_prior  # Soft prior
        au_activation = torch.clamp(au_activation, 0, 1)  # Keep in [0, 1]

        return au_activation


# =============================================================================
# AU Names and Indices
# =============================================================================

AU_NAMES = [
    'AU1',   # 0: Inner Brow Raiser
    'AU2',   # 1: Outer Brow Raiser
    'AU4',   # 2: Brow Lowerer
    'AU5',   # 3: Upper Lid Raiser
    'AU6',   # 4: Cheek Raiser ( Orbicularis Oculi )
    'AU7',   # 5: Lower Lid Depressor
    'AU9',   # 6: Nose Wrinkler
    'AU10',  # 7: Upper Lip Raiser
    'AU12',  # 8: Lip Corner Puller
    'AU14',  # 9: Dimpler
    'AU15',  # 10: Lip Corner Depressor
    'AU17',  # 11: Chin Raiser Lower
    'AU20',  # 12: Lip Stretch
    'AU23',  # 13: Lip Tightener
    'AU24',  # 14: Lip Pressor
    'AU25',  # 15: Lips Part
    'AU26',  # 16: Jaw Drop
]

# Emotion to AU mapping (for reference)
EMOTION_AU_MAPPING = {
    'happiness': ['AU6', 'AU12', 'AU25'],
    'surprise': ['AU1', 'AU2', 'AU5', 'AU25'],
    'disgust': ['AU4', 'AU9', 'AU10', 'AU17'],
    'repression': ['AU14', 'AU17', 'AU4'],
}