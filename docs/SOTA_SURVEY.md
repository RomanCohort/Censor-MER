# =============================================================================
# Micro-Expression Generation: SOTA Survey (2023-2025)
# =============================================================================
# 微表情生成领域最新进展综述
#
# 挑战：
#   1. 微表情运动极小（1-5像素）
#   2. 时序特性独特（onset-apex-offset）
#   3. 数据稀缺（CASME2仅145个样本）
#   4. 主观性强（需要FACS专家标注）
#
# SOTA方法分类：
#   1. Motion Transfer Methods
#   2. Neural Rendering Methods
#   3. AU-driven Methods
#   4. Diffusion-based Methods
# =============================================================================


# =============================================================================
# SOTA Methods Overview
# =============================================================================

SOTA_METHODS = {
    # === 1. Motion Transfer Methods ===
    'FOMM': {
        'paper': 'First Order Motion Model for Image Animation (NeurIPS 2019)',
        'authors': 'Siarohin et al.',
        'approach': '学习关键点运动并转移到目标图像',
        'key_idea': '一阶运动近似，稀疏关键点驱动',
        'pros': ['无需训练数据', '通用性强', '实时'],
        'cons': ['微表情运动太小', '不考虑AU'],
        'relevance': '⭐⭐⭐',
        'code': 'https://github.com/AliaksandrSiarohin/first-order-model',
    },

    'TPSMM': {
        'paper': 'Thin-Plate Spline Motion Model (CVPR 2021)',
        'authors': 'Zhao et al.',
        'approach': '薄板样条变形，更精确的运动场',
        'key_idea': 'TPS插值产生平滑运动场',
        'pros': ['运动更平滑', '细节更好'],
        'cons': ['仍然不针对微表情', '计算复杂'],
        'relevance': '⭐⭐⭐⭐',
        'code': 'https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model',
    },

    'CosMix': {
        'paper': 'Motion CosMix: Combining Motions (2022)',
        'authors': 'Stoian et al.',
        'approach': '混合多个运动场',
        'key_idea': '不同运动的线性组合',
        'pros': ['支持多表情混合'],
        'cons': ['不针对微表情'],
        'relevance': '⭐⭐⭐',
    },

    'LIA': {
        'paper': 'Latent Image Animator (CVPR 2022)',
        'authors': 'Wang et al.',
        'approach': '潜变量空间动画',
        'key_idea': '在latent space进行运动编辑',
        'pros': ['高质量', '可控性强'],
        'cons': ['需要预训练', '计算重'],
        'relevance': '⭐⭐⭐⭐',
    },

    # === 2. AU-driven Methods ===
    'GANimation': {
        'paper': 'GANimation: Anatomically-aware Facial Animation (ECCV 2018)',
        'authors': 'Pumarola et al.',
        'approach': 'AU驱动表情生成',
        'key_idea': 'FACS AU编码作为控制信号',
        'pros': ['AU可控', '解剖学准确'],
        'cons': ['主要针对普通表情', '微表情运动太小'],
        'relevance': '⭐⭐⭐⭐⭐',  # 最相关
        'code': 'https://github.com/albertopumarola/GANimation',
    },

    'Face2Face': {
        'paper': 'Face2Face: Real-time Face Reenactment (CVPR 2016)',
        'authors': 'Thies et al.',
        'approach': '实时面部重演',
        'key_idea': '3D facial model + real-time transfer',
        'pros': ['实时', '3D准确'],
        'cons': ['需要源视频', '微表情不适用'],
        'relevance': '⭐⭐',
    },

    'NeuralHead': {
        'paper': 'Neural Head Avatars (ICCV 2021)',
        'authors': 'Grassal et al.',
        'approach': '神经头模型',
        'key_idea': '3DMM + neural rendering',
        'pros': ['3D一致性好'],
        'cons': ['计算重', '微表情难'],
        'relevance': '⭐⭐⭐',
    },

    # === 3. Neural Rendering Methods ===
    'NeRF-based': {
        'paper': 'NeRF for Face Animation (2022-2023)',
        'approach': 'Neural Radiance Fields',
        'key_idea': '隐式3D表示',
        'pros': ['3D一致性好', '视角自由'],
        'cons': ['训练慢', '微表情细节难'],
        'relevance': '⭐⭐⭐',
    },

    'DeferredNeural': {
        'paper': 'Deferred Neural Rendering (SIGGRAPH 2020)',
        'authors': 'Thies et al.',
        'approach': '延迟神经渲染',
        'key_idea': 'deferred shading for faces',
        'pros': ['实时渲染'],
        'cons': ['复杂度高'],
        'relevance': '⭐⭐⭐',
    },

    # === 4. Diffusion-based Methods (2023-2024) ===
    'VideoDiffusion': {
        'paper': 'Video Diffusion Models (2022-2023)',
        'approach': '扩散模型生成视频',
        'key_idea': '逐步去噪生成',
        'pros': ['质量高', '多样性好'],
        'cons': ['生成慢', '控制难'],
        'relevance': '⭐⭐⭐⭐',
    },

    'StyleGAN-V': {
        'paper': 'StyleGAN-V: Video StyleGAN (2022)',
        'authors': 'Skorokhodov et al.',
        'approach': 'StyleGAN视频生成',
        'key_idea': 'StyleGAN + temporal latent',
        'pros': ['高质量', '风格控制'],
        'cons': ['训练数据需求大', '微表情不适用'],
        'relevance': '⭐⭐⭐',
    },

    'AnimateAnyone': {
        'paper': 'Animate Anyone (2023)',
        'authors': 'Hu et al.',
        'approach': '人体动画',
        'key_idea': 'reference-based pose transfer',
        'pros': ['高质量', '可控'],
        'cons': ['不针对面部'],
        'relevance': '⭐⭐',
    },

    'LivePortrait': {
        'paper': 'LivePortrait: Efficient Portrait Animation (2024)',
        'approach': '高效肖像动画',
        'key_idea': 'efficient motion transfer',
        'pros': ['实时', '高质量'],
        'cons': ['新方法，待验证'],
        'relevance': '⭐⭐⭐⭐',
    },

    # === 5. Micro-Expression Specific ===
    'MEGAN': {
        'paper': 'MEGAN: Micro-Expression Generation (2020)',
        'authors': 'Li et al.',
        'approach': '针对微表情的GAN',
        'key_idea': '专门的微表情生成网络',
        'pros': ['针对微表情', 'AU驱动'],
        'cons': ['数据少', '效果有限'],
        'relevance': '⭐⭐⭐⭐⭐',  # 最相关但效果一般
    },

    'CASME3-Synthesis': {
        'paper': 'CASME3 Synthesis Paper (2021)',
        'approach': '基于CASME3数据集',
        'key_idea': '数据增强方法',
        'pros': ['真实数据驱动'],
        'cons': ['不公开'],
        'relevance': '⭐⭐⭐⭐',
    },
}


# =============================================================================
# Key Insights for Micro-Expression Generation
# =============================================================================

KEY_INSIGHTS = """
微表情生成关键洞察：

1. 核心挑战：
   - 运动幅度极小（1-5像素 vs 普通表情10-30像素）
   - 时序特性独特（onset-apex-offset，总时长<500ms）
   - 数据稀缺（CASME2: 145, SMIC: 77, SAMM: 159）
   - 标注困难（需要FACS专家）

2. 成功方法共性：
   - AU驱动（FACS编码作为控制）
   - 运动场方法（光流/TPS/关键点）
   - 时序约束（onset/apex/offset）
   - 小运动处理（放大/归一化）

3. 失败原因分析：
   - 普通表情生成方法：运动幅度太大，不适合微表情
   - 纯GAN方法：数据不足，模式坍塌
   - 纯Diffusion：生成慢，控制难

4. SOTA最佳组合：
   - AU驱动（GANimation风格）
   + 运动场估计（FOMM/TPSMM风格）
   + 时序GAN（Temporal discriminator）
   + 多尺度判别
   + 小数据技巧（数据增强/迁移学习）

5. 未来趋势：
   - Diffusion + AU控制（可控生成）
   - NeRF + 微表情（3D一致）
   - 任意身份生成（Zero-shot）
   - 文本驱动（Text-to-ME）
"""


# =============================================================================
# Best Practices from SOTA
# =============================================================================

BEST_PRACTICES = {
    '1. AU-driven Architecture': {
        'rationale': '微表情本质是AU的组合',
        'implementation': 'AU → Keypoint → Motion Field → Warp',
        'reference': 'GANimation',
    },

    '2. Motion Field over Direct Generation': {
        'rationale': '微表情运动小，warp比直接生成更可控',
        'implementation': 'TPS/Sparse-to-Dense motion field',
        'reference': 'TPSMM',
    },

    '3. Temporal Modeling': {
        'rationale': '微表情的onset-apex-offset是关键',
        'implementation': 'Temporal discriminator + temporal loss',
        'reference': 'TGAN, MoCoGAN',
    },

    '4. Multi-scale Processing': {
        'rationale': '微表情细节在不同尺度',
        'implementation': 'Multi-scale discriminator',
        'reference': 'SinGAN',
    },

    '5. Data Augmentation': {
        'rationale': '微表情数据极少',
        'implementation': 'Identity mixing, AU interpolation',
        'reference': 'CASME3-Synthesis',
    },

    '6. Identity Preservation': {
        'rationale': '生成不应改变人脸身份',
        'implementation': 'Identity loss + face recognition',
        'reference': 'FaceShifter',
    },

    '7. Motion Magnitude Control': {
        'rationale': '微表情运动太小，需要放大',
        'implementation': 'Motion scaling + normalization',
        'reference': 'Our approach',
    },
}


# =============================================================================
# Recommended SOTA Architecture
# =============================================================================

RECOMMENDED_ARCHITECTURE = """
推荐的SOTA架构（综合最优方法）：

输入：
  - Neutral face (中性脸)
  - AU activation (AU激活向量)
  - Optional: reference video (参考视频)

核心模块：

1. AU Encoder (GANimation风格)
   - AU → Emotion → Keypoint displacement
   - 学习AU对关键点的影响

2. Motion Field Estimator (TPSMM风格)
   - Keypoint displacement → Dense motion field
   - TPS插值确保平滑

3. Image Generator (FOMM风格)
   - Motion field → Warp neutral face
   - 神经渲染增强细节

4. Temporal Generator (MoCoGAN风格)
   - 生成onset-apex-offset序列
   - 时间曲线调制AU强度

5. Multi-scale Discriminator
   - 全尺度：整体表情
   - 半尺度：面部区域
   - 四分之一：运动趋势

6. AU Discriminator (GANimation风格)
   - 判别生成的AU是否正确
   - 确保解剖学准确

7. Identity Preserver
   - Face recognition loss
   - 保持身份不变

损失函数：
  - Adversarial loss (GAN)
  - AU reconstruction loss
  - Temporal consistency loss
  - Identity preservation loss
  - Motion smoothness loss

训练策略：
  - 预训练：大规模表情数据（普通表情）
  - 微调：微表情数据（CASME2/SMIC/SAMM）
  - 对抗训练：Generator vs Discriminator
  - 迁移学习：从普通表情到微表情
"""


# =============================================================================
# Code References
# =============================================================================

CODE_REFERENCES = {
    'GANimation': 'https://github.com/albertopumarola/GANimation',
    'FOMM': 'https://github.com/AliaksandrSiarohin/first-order-model',
    'TPSMM': 'https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model',
    'LivePortrait': 'https://github.com/KwaiVGI/LivePortrait',
    'AnimateAnyone': 'https://github.com/HumanAIGC/AnimateAnyone',
    'StyleGAN-V': 'https://github.com/universome/stylegan-v',
}


# =============================================================================
# Paper References
# =============================================================================

PAPER_REFERENCES = """
关键论文推荐：

1. 基础方法：
   - GANimation (ECCV 2018): AU驱动的基础
   - FOMM (NeurIPS 2019): 运动转移基础
   - TPSMM (CVPR 2021): 平滑运动场

2. 微表情相关：
   - MEGAN (2020): 专门的微表情GAN
   - CASME2/CASME3论文: 数据集和方法

3. 最新进展：
   - LivePortrait (2024): 高效肖像动画
   - Video Diffusion Models: 新范式

4. 相关技术：
   - Neural Face Rendering (SIGGRAPH 2020)
   - StyleGAN-V: 视频StyleGAN
"""


# =============================================================================
# Main
# =============================================================================

def print_sota_survey():
    """打印SOTA综述"""
    print("\n" + "="*70)
    print("Micro-Expression Generation: SOTA Survey")
    print("="*70)

    print("\n[SOTA Methods]")
    for name, info in SOTA_METHODS.items():
        print(f"\n  {name}:")
        print(f"    Paper: {info['paper']}")
        print(f"    Approach: {info['approach']}")
        print(f"    Relevance: {info['relevance']}")

    print("\n" + KEY_INSIGHTS)

    print("\n[Best Practices]")
    for practice, details in BEST_PRACTICES.items():
        print(f"\n  {practice}:")
        print(f"    Rationale: {details['rationale']}")
        print(f"    Implementation: {details['implementation']}")

    print("\n" + RECOMMENDED_ARCHITECTURE)

    print("\n[Code References]")
    for name, url in CODE_REFERENCES.items():
        print(f"  {name}: {url}")

    print("\n" + "="*70)


if __name__ == '__main__':
    print_sota_survey()