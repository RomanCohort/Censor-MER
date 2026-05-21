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
    # SaliencyDetectorE2E (new)
    'sigma_ratio': 0.15,  # Relative sigma for resolution-independence
    # rPPGExtractor
    'rppg_window_size': 5,
    'rppg_bandpass_low': 0.5,   # Hz
    'rppg_bandpass_high': 4.0,  # Hz
    # TV-L1 Optical Flow
    'tvl1_tau': 0.25,
    'tvl1_lambda': 0.15,
    'tvl1_theta': 0.3,
    # AdaptiveOpticalFlow (new)
    'fast_threshold': 0.1,  # Motion threshold for fast vs fine mode
    'use_tvl1': True,     # Enable TV-L1 in fine mode
    # AU Attention Config
    'au_attention_size': 224,
    'au_mask_threshold': 0.1,
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
    'num_classes': 8,          # CASME2 ME categories (8 classes: 0-7)
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
    # DeepSeek API configuration
    'deepseek_api_key': 'sk-9ae76c2d548c4d64a1a8220e0931b448',
    'deepseek_model': 'deepseek-chat',
    'deepseek_base_url': 'https://api.deepseek.com/v1',
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
    # Extended ME categories (11 classes based on MER datasets)
    # Combines: CASME II, SAMM, SMIC, MMEW annotations
    'me_categories': [
        "Happiness (Duchenne)",      # 0: 真笑 AU6+AU12 (眼睛皱纹)
        "Happiness (Non-Duchenne)", # 1: 假笑 AU12 only
        "Surprise (Strong)",        # 2: 强烈惊讶
        "Surprise (Weak)",         # 3: 轻微惊讶
        "Fear",                  # 4: 恐惧
        "Disgust (Strong)",      # 5: 强烈厌恶
        "Disgust (Weak)",        # 6: 轻微厌恶
        "Anger (Strong)",        # 7: 强烈愤怒
        "Anger (Weak)",         # 8: 轻微愤怒
        "Sadness",               # 9: 悲伤
        "Contempt",              # 10: 蔑视 (单侧嘴角)
    ],
    # Mapping from original 7-class to new 11-class
    'me_mapping_7to11': {
        0: [0, 1],        # Happiness → Duchenne / Non-Duchenne
        2: [2, 3],        # Surprise → Strong / Weak
        3: [4],          # Fear
        5: [5, 6],      # Disgust → Strong / Weak
        4: [7, 8],      # Anger → Strong / Weak
        1: [9],          # Sadness
        6: [10],         # Contempt
    },
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
# Visual Perception Config
# =============================================================================
VISUAL_PERCEPTION_CONFIG = {
    # PupilController
    'pupil_hidden_dim': 64,
    'pupil_base_gain': 0.8,
    'pupil_modulation_range': 0.4,

    # RetinalContrastNorm
    'retinal_kernel': 9,
    'retinal_alpha': 0.5,
    'retinal_beta': 0.0,

    # MachBandEnhancer
    'mach_band_strength': 0.3,
    'mach_band_sigma': 2.0,

    # CenterSurroundReceptiveField
    'center_sigma': 1.5,
    'surround_sigma': 3.0,

    # Options
    'enable_retinal': True,
    'enable_mach': True,
    'receptive_weight': 0.1,
}

# =============================================================================
# Weight Initialization Config
# =============================================================================
INIT_CONFIG = {
    'conv': 'kaiming_normal_',
    'linear': 'xavier_uniform_',
}

# =============================================================================
# Long-Term Memory Sparse Control Config
# =============================================================================
SPARSE_CONTROL_CONFIG = {
    'dim': 1024,                    # 神经元数量 (对应 fusion 输出)
    'inactivity_threshold': 100,     # 软冻结阈值 (步) - 更激进
    'hard_freeze_threshold': 200,     # 硬冻结阈值 (步) - 更激进
    'soft_decay_factor': 0.9,         # 软衰减系数 - 更激进
    'growth_factor_boost': 2.0,     # 恢复时增益 (2x)
    'growth_recovery_steps': 30,          # 恢复所需步数
    'min_activity_to_track': 0.01,   # 活跃度阈值
    'enable_dual_path': True,         # 启用双路径
    # 新增: 防过拟合增强
    'enable_random_dropout': True,   # 随机Dropout混合
    'random_dropout_rate': 0.15,     # 随机Dropout比例
    'enable_l2': True,              # L2正则化
    'l2_weight': 0.01,             # L2权重系数
}

# =============================================================================
# Gaze-Driven AU Attention Config
# =============================================================================
GAZE_ATTENTION_CONFIG = {
    # GazeEstimator
    'gaze_input_dim': 64,           # Eye feature input dimension
    'gaze_hidden_dim': 32,          # Hidden layer dimension

    # AURegionAttention
    'au_attention_size': 224,       # Spatial attention map size
    'emotion_hint_dim': 11,         # ME categories for emotion hint
    'region_sigma': 0.15,           # Gaussian spread for AU regions

    # GazeDrivenAttention
    'gaze_history_length': 5,       # Temporal smoothing history length
    'output_attention_strength': 0.3,  # Feature modulation strength

    # GazeEmotionCorrelation
    'gaze_aversion_threshold': 0.5, # Threshold for deception detection
}

# =============================================================================
# Ocular Motion Filter Config
# =============================================================================
OCULAR_FILTER_CONFIG = {
    # BlinkDetector
    'blink_au45_threshold': 0.5,    # AU45 activation threshold
    'blink_min_duration': 3,        # Min blink frames (~15ms at 200fps)
    'blink_max_duration': 80,       # Max blink frames (~400ms at 200fps)

    # SaccadeDetector
    'saccade_velocity_threshold': 0.5,  # Velocity threshold (normalized)
    'saccade_max_duration': 20,     # Max saccade frames (~100ms)
    'saccade_accel_threshold': 1.0, # Acceleration threshold
    'velocity_smooth_kernel': 3,    # Velocity smoothing kernel size

    # SmoothPursuitDetector
    'pursuit_velocity_min': 0.1,    # Min pursuit velocity
    'pursuit_velocity_max': 0.3,    # Max pursuit velocity
    'pursuit_min_duration': 10,     # Min pursuit frames

    # OcularMotionFilter
    'combination_mode': 'union',    # Mask combination: 'union', 'blink_only', 'weighted'
    'filter_strength': 0.8,         # Suppression strength (0-1)

    # CleanSignalExtractor
    'signal_smooth_window': 3,      # Temporal smoothing window
    'baseline_frames': 3,           # Frames for baseline computation
}