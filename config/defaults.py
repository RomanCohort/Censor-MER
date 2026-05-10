# Censor -- Configuration Defaults
# =============================================================================
# Central hyperparameter dictionary. All magic numbers are defined here.
# No hardcoded values should appear in model files.

INPUT_CONFIG = {
    'batch_size': 2,
    'channels': 3,       # RGB
    'temporal': 16,      # T frames
    'height': 224,
    'width': 224,
}

# =============================================================================
# Preprocessing Config
# =============================================================================
PREPROCESS_CONFIG = {
    # SaliencyDetector
    'pyramid_levels': 4,
    'gaussian_sigma': 1.5,
    'center_bias_strength': 1.0,
    # rPPGExtractor
    'rppg_window_size': 5,
    'rppg_bandpass_low': 0.5,   # Hz
    'rppg_bandpass_high': 4.0,  # Hz
    # TV-L1 Optical Flow
    'tvl1_tau': 0.25,
    'tvl1_lambda': 0.15,
    'tvl1_theta': 0.3,
}

# =============================================================================
# Fast Pathway Config (Subcortical - 3D ResNet18)
# =============================================================================
FAST_PATHWAY_CONFIG = {
    'input_channels': 2,    # optical flow: x + y
    'stem_channels': 64,
    'stem_kernel': (3, 7, 7),
    'stem_stride': (1, 2, 2),
    'stem_padding': (1, 3, 3),
    'layer_channels': [64, 128, 256],  # 3 stages
    'layer_strides': [(1, 1, 1), (2, 2, 2), (2, 2, 2)],
    'output_dim': 512,
}

# =============================================================================
# Slow Pathway Config (Cortical - 3D Swin-Transformer)
# =============================================================================
SLOW_PATHWAY_CONFIG = {
    'input_channels': 6,   # RGB (3) + rPPG (3)
    'embed_dim': 96,
    'patch_size': (2, 4, 4),      # temporal, height, width
    'patch_stride': (2, 4, 4),
    # Stage configs: (num_blocks, embed_dim, merge_stride)
    # Stages 1-3 have spatial downsampling (stride 2). Stage 4 projects channels only (stride 1).
    # H/32 = 224/32 = 7, W/32 = 224/32 = 7, T/16 = 16/16 = 1
    'stages': [
        {'depth': 2, 'dim': 96,  'merge_stride': (2, 2, 2)},
        {'depth': 2, 'dim': 192, 'merge_stride': (2, 2, 2)},
        {'depth': 6, 'dim': 384, 'merge_stride': (2, 2, 2)},
        {'depth': 2, 'dim': 768, 'merge_stride': (1, 1, 1)},  # channel projection only
    ],
    'output_dim': 768,
}

# =============================================================================
# Attention Module Config
# =============================================================================
AMYGDALA_CONFIG = {
    'input_dim': 512,        # fast pathway output dim
    'hidden_dim': 256,
    'output_spatial': [14, 14],  # spatial resolution of attention prior map
}

FFA_CONFIG = {
    'fast_dim': 512,
    'slow_dim': 768,
    'reduction_ratio': 16,
}

CASA_CONFIG = {
    'embed_dim': 768,         # slow pathway output
    'num_heads': 8,
    'ffn_dim': 768 * 4,
    'pyramid_size': [1, 7, 7],  # T/16=1, H/32=7, W/32=7 at deepest stage
}

# =============================================================================
# Fusion Config
# =============================================================================
FUSION_CONFIG = {
    'fast_dim': 512,
    'slow_dim': 768,
    'fused_dim': 1024,
    'num_heads': 8,
}

# =============================================================================
# AU Decoder Config
# =============================================================================
AU_DECODER_CONFIG = {
    'input_dim': 1024,        # fused feature dim
    'hidden_dim': 512,
    'num_layers': 2,
    'dropout': 0.3,
    'num_aus': 28,            # FACS Action Units
    'temporal_steps': 16,      # same as input T
    'threshold': 0.3,         # AU activation threshold
}

# =============================================================================
# MoE Config
# =============================================================================
MOE_CONFIG = {
    'input_dim': 1024,
    'hidden_dim': 512,
    'num_experts': 3,
    'gating_hidden_dim': 128,
    'num_classes': 7,         # ME categories
    'top_k': 2,
    'load_balancing_lambda': 0.01,  # weight for auxiliary loss
}

# =============================================================================
# PersonalizedRadar Config (TTA)
# =============================================================================
RADAR_CONFIG = {
    'input_dim': 1024,
    'adapt_steps': 5,
    'adapt_lr': 1e-3,
    'support_shots': 8,
}

# =============================================================================
# LLM Report Config
# =============================================================================
LLM_CONFIG = {
    'text_embed_dim': 256,
    'max_report_len': 128,
    # AU mapping table (FACS standard)
    'au_names': {
        1: "Inner Brow Raiser", 2: "Outer Brow Raiser", 4: "Brow Lowerer",
        5: "Upper Lid Raiser", 6: "Cheek Raiser", 7: "Lid Tightener",
        9: "Nose Wrinkler", 10: "Upper Lip Raiser", 11: "Nasolabial Deepener",
        12: "Lip Corner Puller", 14: "Dimpler", 15: "Lip Corner Depressor",
        17: "Chin Raiser", 20: "Lip Stretcher", 23: "Lip Tightener",
        24: "Lip Pressor", 25: "Lips Part", 26: "Jaw Drop",
        27: "Mouth Stretch", 28: "Lip Suck",
    },
    'me_categories': [
        "Happiness", "Sadness", "Surprise", "Fear",
        "Anger", "Disgust", "Contempt"
    ],
}

# =============================================================================
# Data Config
# =============================================================================
DATA_CONFIG = {
    'T': 16,                   # number of temporal frames
    'H': 224,                  # spatial height
    'W': 224,                  # spatial width
    'num_workers': 2,          # data loading workers
    'augment': True,           # enable data augmentation
    'compute_flow_on_the_fly': True,  # compute optical flow during loading
    'normalize_mean': [0.485, 0.456, 0.406],  # ImageNet mean
    'normalize_std': [0.229, 0.224, 0.225],   # ImageNet std
    'max_frames': 300,         # max frames to load from video
    'face_crop_ratio': 0.8,    # center crop ratio
    'temporal_jitter': True,   # random T-frame sampling
}
DATA_ROOT = './data'  # default root directory for datasets

# =============================================================================
# Weight Initialization Config
# =============================================================================
INIT_CONFIG = {
    'conv': 'kaiming_normal_',
    'linear': 'xavier_uniform_',
}