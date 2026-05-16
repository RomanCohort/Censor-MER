# =============================================================================
# Censor -- Text-Guided Face Generation Module
# =============================================================================
# Uses CLIP to condition face generation on text descriptions.
# Can describe: emotion, age, gender, lighting, accessories, etc.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# CLIP Text Encoder (simplified - use pre-trained)
# =============================================================================

class CLIPTextEncoder(nn.Module):
    """
    CLIP-based text encoder for face descriptions.

    Supports:
    - Emotion: "happy", "sad", "angry", "surprised"
    - Age: "young", "old", "child"
    - Gender: "male", "female"
    - Lighting: "dark", "bright", "backlit"
    - Accessories: "glasses", "hat"
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Try to load pre-trained CLIP
        try:
            import clip
            self.clip_model, _ = clip.load("ViT-B/32")
            self.clip_text_encoder = self.clip_model.transformer
            self.clip_dim = 512
            self.use_pretrained = True
            print("[CLIPTextEncoder] Loaded pre-trained CLIP")
        except ImportError:
            # Fallback: simple embedding
            self.use_pretrained = False
            self.clip_dim = embed_dim
            print("[CLIPTextEncoder] Using fallback encoder")

            # Simple vocabulary
            self.vocab = {
                'happy': 0, 'sad': 1, 'angry': 2, 'surprised': 3,
                'fear': 4, 'disgust': 5, 'neutral': 6,
                'young': 7, 'old': 8,
                'male': 9, 'female': 10,
                'dark': 11, 'bright': 12, 'backlit': 13,
                'glasses': 14, 'hat': 15,
            }
            self.vocab_size = len(self.vocab)

            self.embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Project to feature space
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def encode_text(self, text):
        """
        Encode text descriptions.

        Args:
            text: List of strings, e.g., ["happy young male", "sad old female"]
        Returns:
            embeddings: (B, embed_dim)
        """
        if self.use_pretrained:
            # Use CLIP
            import clip
            text_tokens = clip.tokenize(text).to(next(self.parameters()).device)
            with torch.no_grad():
                embeddings = self.clip_model.encode_text(text_tokens)
        else:
            # Simple tokenization
            embeddings = []
            for t in text:
                tokens = t.lower().split()
                ids = []
                for token in tokens:
                    ids.append(self.vocab.get(token, self.vocab_size - 1))
                ids = torch.tensor(ids, device=next(self.parameters()).device)
                emb = self.embedding(ids).mean(dim=0)
                embeddings.append(emb)

            embeddings = torch.stack(embeddings)

        # Project to target dimension
        embeddings = self.projection(embeddings)

        return embeddings

    def forward(self, text):
        """
        Args:
            text: List of strings ["description", ...]
        Returns:
            text_features: (B, embed_dim)
        """
        return self.encode_text(text)


# =============================================================================
# Text-to-Feature Adapter
# =============================================================================

class TextToFeatureAdapter(nn.Module):
    """
    Adapts text features to conditioning features.

    Bridges CLIP text space → generation feature space.
    """

    def __init__(self, text_dim=512, target_dim=1024):
        super().__init__()

        # Cross-attention for text conditioning
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4),
            nn.GELU(),
            nn.Linear(text_dim * 4, text_dim)
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(text_dim)
        self.norm2 = nn.LayerNorm(text_dim)

        # Final projection
        self.projection = nn.Linear(text_dim, target_dim)

    def forward(self, target_features, text_features):
        """
        Args:
            target_features: (B, T, target_dim) features to attend to
            text_features: (B, text_dim) text embeddings
        Returns:
            adapted: (B, target_dim)
        """
        # Expand text for attention
        query = text_features.unsqueeze(1)  # (B, 1, text_dim)
        key_value = target_features  # (B, T, target_dim)

        # Cross attention
        attn_out, _ = self.cross_attention(query, key_value, key_value)
        out = self.norm1(query + attn_out)

        # FFN
        out = self.norm2(out + self.ffn(out))

        # Project
        out = self.projection(out.squeeze(1))

        return out


# =============================================================================
# Emotional Text Conditioning
# =============================================================================

class EmotionalTextConditioning(nn.Module):
    """
    Maps text descriptions of emotion to emotion embeddings.

    Uses Plutchik's wheel of emotions as supervision.
    """

    def __init__(self, embed_dim=512):
        super().__init__()

        # Emotion categories (Plutchik's 8 primary)
        self.emotion_names = [
            'joy', 'sadness', 'anger', 'fear',
            'disgust', 'surprise', 'trust', 'anticipation'
        ]

        # Learnable emotion embeddings
        self.emotion_embeddings = nn.Embedding(len(self.emotion_names), embed_dim)

        # Intensity estimator (per emotion)
        self.intensity_estimator = nn.Sequential(
            nn.Linear(embed_dim, len(self.emotion_names)),
            nn.Sigmoid()
        )

    def forward(self, text_features):
        """
        Args:
            text_features: (B, embed_dim) CLIP text features
        Returns:
            emotion_logits: (B, 8) emotion intensities
            emotion_embedding: (B, embed_dim)
        """
        # Map to emotion space
        intensity = self.intensity_estimator(text_features)

        # Get dominant emotion embedding
        dominant = intensity.argmax(dim=-1)
        emotion_emb = self.emotion_embeddings(dominant)

        return intensity, emotion_emb


# =============================================================================
# Complete Text Guidance Pipeline
# =============================================================================

class TextGuidancePipeline(nn.Module):
    """
    Complete text guidance for face generation.

    Flow:
    1. Text description → CLIP features
    2. CLIP features → Emotion conditioning
    3. Emotion + other features → generation conditioning
    """

    def __init__(self, text_embed_dim=512, target_dim=1024):
        super().__init__()

        self.text_encoder = CLIPTextEncoder(text_embed_dim)
        self.text_adapter = TextToFeatureAdapter(text_embed_dim, target_dim)
        self.emotion_conditioning = EmotionalTextConditioning(text_embed_dim)

    def forward(self, target_features, text_descriptions):
        """
        Args:
            target_features: (B, target_dim) dual-pathway features
            text_descriptions: List[str] e.g., ["happy young male"]
        Returns:
            dict with text_features, emotion_intensity, conditioned_feature
        """
        # 1. Text encoding
        text_features = self.text_encoder(text_descriptions)  # (B, embed_dim)

        # 2. Emotion conditioning
        emotion_intensity, emotion_emb = self.emotion_conditioning(text_features)

        # 3. Cross-attention conditioning
        # Target: expand features for attention
        target_expanded = target_features.unsqueeze(1)  # (B, 1, target_dim)
        conditioned = self.text_adapter(target_expanded, text_features)

        return {
            'text_features': text_features,        # (B, embed_dim)
            'emotion_intensity': emotion_intensity,  # (B, 8)
            'emotion_embedding': emotion_emb,       # (B, embed_dim)
            'conditioned_feature': conditioned,     # (B, target_dim)
        }


# =============================================================================
# Attribute Selector (for predefined descriptions)
# =============================================================================

class AttributeSelector(nn.Module):
    """
    Selects from predefined attribute options.
    Useful for structured generation.
    """

    def __init__(self, embed_dim=512):
        super().__init__()

        # Predefined attributes
        self.attributes = {
            'emotion': ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprised', 'disgusted'],
            'age': ['child', 'young', 'middle', 'old'],
            'gender': ['male', 'female'],
            'lighting': ['natural', 'studio', 'backlit', 'soft', 'dramatic'],
        }

        # Embeddings per attribute
        self.attr_embeddings = nn.ModuleDict()
        for attr_name, attr_values in self.attributes.items():
            self.attr_embeddings[attr_name] = nn.Embedding(len(attr_values), embed_dim)

        # Selector heads
        self.selector_heads = nn.ModuleDict()
        for attr_name, attr_values in self.attributes.items():
            self.selector_heads[attr_name] = nn.Linear(embed_dim, len(attr_values))

    def select(self, text_features):
        """
        Select attributes from features.

        Args:
            text_features: (B, embed_dim)
        Returns:
            selected: dict of selected attribute indices
        """
        selected = {}
        for attr_name in self.attributes:
            logits = self.selector_heads[attr_name](text_features)
            selected[attr_name] = logits.argmax(dim=-1)

        return selected

    def get_attribute_embeddings(self, selected):
        """
        Get embeddings for selected attributes.

        Args:
            selected: dict of selected indices
        Returns:
            embeddings: (B, embed_dim)
        """
        embeddings = []
        for attr_name, attr_idx in selected.items():
            # Handle both tensor and int
            if isinstance(attr_idx, torch.Tensor):
                attr_idx = attr_idx.item() if attr_idx.dim() == 0 else attr_idx[0]
            emb = self.attr_embeddings[attr_name](torch.tensor(attr_idx, device=next(self.parameters()).device))
            embeddings.append(emb)

        return torch.stack(embeddings).mean(dim=0, keepdim=True)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print(" Text-Guided Generation Test")
    print("=" * 50)

    # Test encoder
    encoder = CLIPTextEncoder(512)

    texts = ["happy young male", "sad old female"]
    with torch.no_grad():
        text_features = encoder.encode_text(texts)

    print(f"Text features: {text_features.shape}")

    # Test pipeline
    pipeline = TextGuidancePipeline(512, 1024)
    target_features = torch.randn(2, 1024)

    with torch.no_grad():
        result = pipeline(target_features, texts)

    print(f"Text features: {result['text_features'].shape}")
    print(f"Emotion intensity: {result['emotion_intensity'].shape}")
    print(f"Conditioned feature: {result['conditioned_feature'].shape}")

    # Test attribute selector
    selector = AttributeSelector(512)
    selected = selector.select(text_features)
    print(f"Selected: {selected}")

    print("\nText-Guided Generation Test Passed!")