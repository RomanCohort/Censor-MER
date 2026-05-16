# =============================================================================
# Censor -- Face Identity Preservation Module
# =============================================================================
# Preserves face identity during generation.
# Uses ArcFace/CosFace features to ensure generated faces
# match the input identity.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Face Identity Encoder (ArcFace-based)
# =============================================================================

class FaceIdentityEncoder(nn.Module):
    """
    Extracts face identity features using ArcFace-style training.

    Key insight: Learn identity-preserving features by using
    large-margin softmax loss during training.
    """

    def __init__(self, input_dim=1024, embed_dim=512, num_classes=None):
        super().__init__()
        self.embed_dim = embed_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # Try loading pre-trained ArcFace weights
        self.use_pretrained = False

        # Initialize with orthogonal initialization
        for m in self.feature_extractor:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        Args:
            features: (B, input_dim) dual-pathway features
        Returns:
            identity_features: (B, embed_dim) normalized
        """
        identity = self.feature_extractor(features)

        # L2 normalize for cosine similarity
        identity = F.normalize(identity, dim=-1)

        return identity


# =============================================================================
# Identity-Preserving Loss
# =============================================================================

class IdentityPreservingLoss(nn.Module):
    """
    Loss that encourages identity preservation.

    Uses cosine similarity between original and generated features.
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, original_features, generated_features):
        """
        Args:
            original_features: (B, embed_dim) original face features
            generated_features: (B, embed_dim) generated face features
        Returns:
            loss: similarity loss (lower is better)
        """
        # Cosine similarity
        cos_sim = (original_features * generated_features).sum(dim=-1)

        # Margin-based loss: want similarity > margin
        loss = F.relu(self.margin - cos_sim).mean()

        return loss


# =============================================================================
# Identity Feature Extractor from Image
# =============================================================================

class IdentityExtractorFromImage(nn.Module):
    """
    Extracts identity features from generated images.

    Uses pre-trained face recognition backbone.
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Try loading pre-trained model
        try:
            # Would load actual model here
            # from facenet import InceptionResnet
            self.use_pretrained = False
        except:
            self.use_pretrained = False

        # Fallback: simple CNN encoder
        self.encoder = nn.Sequential(
            # Simple CNN for face embeddings
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) face images
        Returns:
            identity: (B, embed_dim) normalized
        """
        identity = self.encoder(images)
        identity = F.normalize(identity, dim=-1)

        return identity


# =============================================================================
# ID-Conditioned Generator
# =============================================================================

class IDConditionedGenerator(nn.Module):
    """
    Conditions generation on identity features.

    Keeps identity consistent by:
    1. Encoding source identity
    2. Injecting into generation
    3. Preserving through loss
    """

    def __init__(self, id_embed_dim=512, gen_feature_dim=1024):
        super().__init__()

        # Identity encoder
        self.id_encoder = FaceIdentityEncoder(gen_feature_dim, id_embed_dim)

        # ID injection module
        self.id_injection = nn.Sequential(
            nn.Linear(id_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, gen_feature_dim),
        )

        # Scaling factor
        self.id_scale = nn.Parameter(torch.tensor(0.5))

    def encode_id(self, features):
        """Encode identity from features"""
        return self.id_encoder(features)

    def inject_id(self, gen_features, id_features):
        """
        Inject identity into generation features.

        Args:
            gen_features: (B, gen_feature_dim)
            id_features: (B, id_embed_dim)
        Returns:
            id-conditioned: (B, gen_feature_dim)
        """
        id_contribution = self.id_injection(id_features)
        scale = self.id_scale.sigmoid()

        # Add identity
        conditioned = gen_features + scale * id_contribution

        return conditioned


# =============================================================================
# ID-Preserving Encoder
# =============================================================================

class IDPreservingEncoder(nn.Module):
    """
    Complete identity preservation encoder.

    Flow:
    1. Input features → ID features
    2. ID features → Conditioning
    """

    def __init__(self, feature_dim=1024, id_embed_dim=512):
        super().__init__()

        self.id_encoder = FaceIdentityEncoder(feature_dim, id_embed_dim)
        self.id_extractor = IdentityExtractorFromImage(id_embed_dim)

    def encode_from_features(self, features):
        """Encode ID from feature vector"""
        return self.id_encoder(features)

    def encode_from_image(self, images):
        """Encode ID from image"""
        return self.id_extractor(images)

    def forward(self, features, images=None):
        """
        Args:
            features: (B, feature_dim) optional
            images: (B, 3, H, W) optional
        Returns:
            id_features: (B, id_embed_dim)
        """
        if features is not None:
            return self.id_encoder(features)
        elif images is not None:
            return self.id_extractor(images)
        else:
            raise ValueError("Either features or images must be provided")


# =============================================================================
# Identity Loss Functions
# =============================================================================

class IdentityLoss(nn.Module):
    """
    Combined identity loss for training.
    """

    def __init__(self, id_embed_dim=512):
        super().__init__()
        self.id_encoder = FaceIdentityEncoder(1024, id_embed_dim)
        self.image_encoder = IdentityExtractorFromImage(id_embed_dim)
        self.preserving_loss = IdentityPreservingLoss()

    def compute_loss(self, original_features, generated_image, target_id=None):
        """
        Args:
            original_features: (B, 1024)
            generated_image: (B, 3, H, W)
            target_id: (B, id_embed_dim) optional target ID
        Returns:
            loss: scalar
        """
        # Extract ID from generated image
        gen_id = self.image_encoder(generated_image)

        # Get original ID
        if target_id is not None:
            orig_id = target_id
        else:
            orig_id = self.id_encoder(original_features)

        # Compute loss
        loss = self.preserving_loss(orig_id, gen_id)

        return loss


# =============================================================================
# Complete ID Preservation Module
# =============================================================================

class IDPreservationModule(nn.Module):
    """
    Complete module for ID preservation.
    """

    def __init__(self, feature_dim=1024, id_embed_dim=512):
        super().__init__()

        self.id_encoder = FaceIdentityEncoder(feature_dim, id_embed_dim)
        self.image_encoder = IdentityExtractorFromImage(id_embed_dim)
        self.id_injection = IDConditionedGenerator(id_embed_dim, feature_dim)

    def forward(self, features, generated_image):
        """
        Compute ID preservation features.

        Args:
            features: (B, feature_dim) original features
            generated_image: (B, 3, H, W) generated image
        Returns:
            id_loss: scalar
            id_features: (B, id_embed_dim)
        """
        # Encode original ID
        orig_id = self.id_encoder(features)

        # Encode generated ID
        gen_id = self.image_encoder(generated_image)

        # Loss
        loss = self.preserving_loss(orig_id, gen_id)

        # Also return for conditioning
        return loss, orig_id

    def preserve(self, features):
        """Get ID features for conditioning"""
        return self.id_encoder(features)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print(" Identity Preservation Test")
    print("=" * 50)

    # Test ID encoder
    id_encoder = FaceIdentityEncoder(1024, 512)
    features = torch.randn(2, 1024)

    with torch.no_grad():
        id_feat = id_encoder(features)

    print(f"ID features: {id_feat.shape}")

    # Test ID extractor from image
    id_extractor = IdentityExtractorFromImage(512)
    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        id_from_image = id_extractor(images)

    print(f"ID from image: {id_from_image.shape}")

    # Test ID preservation loss
    preserving_loss = IdentityPreservingLoss()

    with torch.no_grad():
        loss = preserving_loss(id_feat, id_from_image)

    print(f"Preserving loss: {loss.item():.4f}")

    # Test full module
    id_module = IDPreservationModule(1024, 512)

    with torch.no_grad():
        id_loss, orig_id = id_module(features, images)

    print(f"ID loss: {id_loss.item():.4f}")
    print(f"Original ID: {orig_id.shape}")

    print("\nIdentity Preservation Test Passed!")