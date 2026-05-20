"""
Censor Model Components

Core modules for biomimetic micro-expression recognition.
"""

# CV Emotion Bridge (新模块)
from model.cv_emotion_bridge import (
    CVEmotionBridge,
    EmotionDetectionResult,
    TemporalEmotionState,
    FERBackend,
    DeepFaceBackend,
    CensorBackend,
    create_cv_bridge,
    create_bridge_for_civis,
    quick_detect,
)

# Preprocessing
from model.preprocessing import (
    SaliencyDetector,
    rPPGExtractor,
    TVL1OpticalFlow,
    AdaptiveOpticalFlow,
    SaliencyDetectorE2E,
)

# Backbones
from model.backbones import (
    FastSubcorticalPathway,
    SlowCorticalPathway,
)

# Attention
from model.attention import (
    Amygdala,
    FFA,
    CASANet,
    AmygdalaWithPrior,
    CASANetLearnable,
)

# AU Attention
from model.au_attention import (
    AULandmarkAttention,
    AUMaskedAttention,
)

# Gaze-Driven AU Attention (新模块)
from model.gaze_attention import (
    GazeEstimator,
    AURegionAttention,
    GazeDrivenAttention,
    GazeEmotionCorrelation,
    AU_REGIONS,
    EMOTION_GAZE_PATTERNS,
    create_gaze_attention,
    create_gaze_estimator,
    create_au_region_attention,
)

# Ocular Motion Filter (新模块)
from model.ocular_filter import (
    BlinkDetector,
    SaccadeDetector,
    SmoothPursuitDetector,
    OcularMotionFilter,
    CleanSignalExtractor,
    BLINK_CONFIG,
    create_ocular_filter,
    create_clean_signal_extractor,
    create_blink_detector,
    create_saccade_detector,
)

# Fusion
from model.fusion import TSFmicroFusion

# Decoders
from model.decoders import DynamicAUDecoder

# MoE
from model.moe_head import (
    MoEGatingNetwork,
    PersonalizedRadar,
    PersonalizedRadarEnhanced,
)
from model.biomoe import BioMoE

# Biomimetic Enhancement
from model.biomimetic_enhance import (
    LongTermMemorySparseControl,
    SparseControlWrapper,
    TemporalSparseControl,
)

# Reporting
from model.llm_report import EmotionReporter


__all__ = [
    # CV Emotion Bridge
    'CVEmotionBridge',
    'EmotionDetectionResult',
    'TemporalEmotionState',
    'FERBackend',
    'DeepFaceBackend',
    'CensorBackend',
    'create_cv_bridge',
    'create_bridge_for_civis',
    'quick_detect',

    # Preprocessing
    'SaliencyDetector',
    'rPPGExtractor',
    'TVL1OpticalFlow',
    'AdaptiveOpticalFlow',
    'SaliencyDetectorE2E',

    # Backbones
    'FastSubcorticalPathway',
    'SlowCorticalPathway',

    # Attention
    'Amygdala',
    'FFA',
    'CASANet',
    'AmygdalaWithPrior',
    'CASANetLearnable',

    # AU Attention
    'AULandmarkAttention',
    'AUMaskedAttention',

    # Gaze-Driven AU Attention
    'GazeEstimator',
    'AURegionAttention',
    'GazeDrivenAttention',
    'GazeEmotionCorrelation',
    'AU_REGIONS',
    'EMOTION_GAZE_PATTERNS',
    'create_gaze_attention',
    'create_gaze_estimator',
    'create_au_region_attention',

    # Ocular Motion Filter
    'BlinkDetector',
    'SaccadeDetector',
    'SmoothPursuitDetector',
    'OcularMotionFilter',
    'CleanSignalExtractor',
    'BLINK_CONFIG',
    'create_ocular_filter',
    'create_clean_signal_extractor',
    'create_blink_detector',
    'create_saccade_detector',

    # Fusion
    'TSFmicroFusion',

    # Decoders
    'DynamicAUDecoder',

    # MoE
    'MoEGatingNetwork',
    'PersonalizedRadar',
    'PersonalizedRadarEnhanced',
    'BioMoE',

    # Biomimetic Enhancement
    'LongTermMemorySparseControl',
    'SparseControlWrapper',
    'TemporalSparseControl',

    # Reporting
    'EmotionReporter',
]