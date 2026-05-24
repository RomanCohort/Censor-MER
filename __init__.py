# Censor -- Biomimetic Dual-Pathway Micro-Expression Recognition System
# =============================================================================
# Copyright (c) 2026 Censor Project
# All rights reserved.

"""
Censor: A brain-inspired micro-expression recognition system.
Bio-inspired dual-pathway architecture with foveal saliency, rPPG blood-flow
enhancement, fusiform-amygdala attention modulation, and reference face template.
"""

from .preprocessing import SaliencyDetector, rPPGExtractor, TVL1OpticalFlow
from .au_attention import AULandmarkAttention, AUMaskedAttention, create_au_attention_map
from .backbones import FastSubcorticalPathway, SlowCorticalPathway
from .attention import Amygdala, FFA, CASANet
from .fusion import TSFmicroFusion
from .decoders import DynamicAUDecoder
from .moe_head import MoEGatingNetwork, PersonalizedRadar
from .llm_report import EmotionReporter

__all__ = [
    'SaliencyDetector',
    'rPPGExtractor',
    'TVL1OpticalFlow',
    'AULandmarkAttention',
    'AUMaskedAttention',
    'create_au_attention_map',
    'FastSubcorticalPathway',
    'SlowCorticalPathway',
    'Amygdala',
    'FFA',
    'CASANet',
    'TSFmicroFusion',
    'DynamicAUDecoder',
    'MoEGatingNetwork',
    'PersonalizedRadar',
    'EmotionReporter',
]