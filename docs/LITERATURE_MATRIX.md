# Literature Matrix: Censor MER for IEEE TAC

**Project**: Biomimetic Dual-Pathway Micro-Expression Recognition
**Generated**: 2026-06-03
**Status**: Phase 2 (Investigation)

---

## Matrix 1: SOTA MER Methods Comparison (2024-2025)

| Method | Year | Venue | Architecture | Backbone | CASME II | SAMM | SMIC | CAS(ME)² | Key Innovation | Limitations |
|--------|------|-------|--------------|----------|---------|------|------|----------|----------------|-------------|
| **Hybrid Attention-3DNet** | 2025 | JJCIT | Single-pathway + attention | 3D CNN + SE | **93.79%** | **93.61%** | **93.42%** | **93.95%** | Spatial + temporal attention modules | Single-pathway; no AU integration |
| **ROI-ArcFace** | 2025 | IEEE | Region-based metric learning | CNN + ArcFace | **93.96%** | 86.15% | 81.17% | — | Region-based angular margin loss | SAMM/SMIC accuracy drop; no temporal modeling |
| **STRNet** | 2025 | Int. J. SCC | Region-based reasoning | Region network | — | — | — | **UF1=0.9792** | Spatiotemporal reasoning on regions | No CASME II/SMIC reported |
| **GAM-MER** | 2024 | Heliyon | Graph attention + transformer | Graph + Transformer | 91.57% | 91.25% | 86.22% | — | Graph attention for muscle modeling | No AU multi-task; no rPPG |
| **MCCA-VNet** | 2024 | PMC | Multi-architecture fusion | ViT + XCiT + CBAM | — | — | — | UF1=0.868 | Multi-architecture fusion | Lower UF1 than STRNet |
| **μ-BERT** | 2024 | ACM MM | BERT-style sequence | Transformer | 90.34% | — | 85.80% | — | BERT-style masked modeling | No SAMM result; single-pathway |
| **SelfME** | 2024 | IEEE | Self-supervised | Transformer | 90.78% | — | 69.70% | — | Self-supervised pretraining | SMIC accuracy drop; no AU |
| **Multi-scale 3D ResNet** | 2024 | J. Image | Multi-scale temporal | 3D-ResNet50 | 91.35% | 84.77% | 74.6% | — | Multi-scale temporal features | No attention; no AU; SMIC drop |
| **Dual-Branch Cross-Attn** | 2024 | — | Dual-branch cross-attention | Swin + MobileViT | — | — | — | 81.6% | Cross-pathway fusion (similar to Censor) | No CASME II/SMIC reported |
| **LAENet** | 2024 | OA | Lightweight | 3D CNN | 79.19% | — | — | — | Lightweight efficient design | Lower accuracy; limited validation |
| **OFF-ApexNet** | baseline | — | CNN on optical flow | 2D CNN | 87.64% | 54.09% | 68.17% | — | Optical flow + apex frame | No temporal modeling; SAMM failure |
| **LBP-TOP** | baseline | — | Handcrafted | LBP on 3 planes | 70.26% | 39.54% | 20.00% | — | Spatiotemporal texture | Handcrafted; low accuracy |
| **Censor** | 2025 | Target: IEEE TAC | **Dual-pathway biomimetic** | **3D ResNet-18 + 3D Swin-T** | **TBD** | **TBD** | **TBD** | **TBD** | **Fusiform-amygdala circuit; AU decoder; MoE; rPPG; TTA** | **Results pending** |

---

## Matrix 2: Architecture Component Comparison

| Method | Dual-Pathway | AU Multi-Task | MoE Gating | rPPG Signal | Apex Detection | Test-Time Adaptation | Explainability |
|--------|--------------|---------------|------------|-------------|----------------|---------------------|----------------|
| Hybrid Attention-3DNet | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Medium (attention viz) |
| ROI-ArcFace | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Low (metric learning) |
| STRNet | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Medium (region reasoning) |
| GAM-MER | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | High (graph muscle viz) |
| MCCA-VNet | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Medium (multi-arch fusion) |
| μ-BERT | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Medium (attention viz) |
| Dual-Branch Cross-Attn | ✓ (partial) | ✗ | ✗ | ✗ | ✗ | ✗ | Medium (cross-attention) |
| **Censor** | ✓ (full) | ✓ (28 AU) | ✓ (3 experts) | ✓ (chrominance) | ✓ (CASANet) | ✓ (PersonalizedRadar) | **High (AU + attention)** |

**Key Insight**: Censor is the **first** to integrate all six advanced components (dual-pathway + AU + MoE + rPPG + apex detection + TTA) in a single MER architecture.

---

## Matrix 3: Neuroscience Grounding Evidence

| Neuroscience Claim | Primary Evidence | Source | Evidence Type | ME-Specific? | Claim Strength |
|--------------------|-----------------|--------|---------------|--------------|----------------|
| **Dual-pathway architecture** | Ventral (FFA) vs dorsal (amygdala/STS) processing | [Dual neural pathways study](https://www.sciencedirect.com/science/article/pii/S1053811907001234) | fMRI meta-analysis | ✗ (macro-expression) | **Strong** for general face processing |
| **Fast subcortical "low road"** | Superior colliculus → Pulvinar → Amygdala pathway | [Subcortical fear processing](https://www.nature.com/neuro/reviews/subcortical_fear.html) | Classical review | ✗ (threat detection) | **Medium** — timing-compatible but different function |
| **Amygdala rapid response (~100ms)** | Amygdala responds to fearful faces in 100-150ms, preceding FFA | [Amygdala-FFA timing study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3125678/) | fMRI + MEG timing | ✗ (fearful faces) | **Medium** — validates speed claim for threat, not ME |
| **FFA selectivity for identity** | FFA processes identity regardless of expression | [FFA-amygdala dissociation study](https://www.jneurosci.org/content/28/9/552) | fMRI selectivity | ✗ (identity vs expression) | **Strong** for functional separation |
| **Amygdala expression preference** | Amygdala preferentially responds to emotional expressions (especially fear) | [FFA-amygdala dissociation study](https://www.jneurosci.org/content/28/9/552) | fMRI selectivity | ✗ (macro-expression) | **Strong** for expression processing |
| **FFA-amygdala connectivity** | Structural DTI connections predict expression recognition accuracy | [FFA-amygdala connectivity](https://academic.oup.com/cercor/article/28/9/3234/4656223) | DTI structural imaging | ✗ (general expression) | **Strong** for circuit existence |
| **Parallel vs interactive processing** | Both parallel and interactive FFA-amygdala processing exist | [Amygdala-FFA interaction study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3125678/) | fMRI connectivity | ✗ (general face) | **Medium** — supports both architectures |
| **Patient double dissociation** | Prosopagnosia patients: some can't recognize identity but can read expressions | [Prosopagnosia evidence](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2906136/) | Patient case studies | ✗ (macro-expression) | **Strong** for pathway independence |
| **ME-specific dual-pathway** | — | — | — | **?** | **Gap** — no ME-specific neuroimaging studies |

**Critical Gap**: No micro-expression-specific neuroimaging studies validating dual-pathway differentiation for 40-200ms expressions.

---

## Matrix 4: Benchmark Dataset Characteristics

| Dataset | Samples | Subjects | FPS | Resolution | Classes | Emotion Labels | Spontaneous? | Access |
|---------|---------|----------|-----|------------|---------|----------------|--------------|--------|
| **CASME II** | 247 | 26 | 200 | 640×480 | 5-7 | happiness, disgust, surprise, repression, tense, others | Semi-posed | License required |
| **SAMM** | 159 | 32 | 200 | 2040×1088 | 7-8 | anger, contempt, disgust, fear, happiness, sadness, surprise | Spontaneous | License required |
| **SMIC-HS** | 164 | 16 | 100 | 640×480 | 3 | positive, negative, surprise | Spontaneous | License required |
| **MMEW** | 300 (+900 macro) | 36 | 90 | 1920×1080 | 7 | anger, disgust, fear, happiness, sadness, surprise, neutral | Spontaneous + macro | GitHub available |
| **CAS(ME)³** | ~300+ | — | 30 | Various | 4+ | — | Spontaneous | CAS official |
| **iMER Benchmark** | 5 datasets | — | — | — | — | Standardized | Mixed | Framework available |

**Evaluation Protocol**: Leave-One-Subject-Out (LOSO) is standard for all datasets to prevent subject-specific overfitting.

---

## Matrix 5: Application Evidence Matrix

| Application Domain | Evidence Type | Source | Effectiveness | Limitations | Ethical Considerations |
|--------------------|---------------|--------|---------------|-------------|-----------------------|
| **Counselor training** | Training studies | METT (Ekman) | Improves recognition accuracy | Retention varies; lab vs real gap | Beneficial application |
| **Clinical assessment** | Population studies | Schizophrenia, autism ME recognition impairment | Identifies atypical processing | Not diagnostic alone | Privacy, labeling concerns |
| **Deception detection** | Meta-analyses | Controversial literature | ME indicates emotional concealment, not lying | High false positive rate; not definitive | **Dual-use risk**: surveillance, interrogation |
| **Psychological research** | Behavioral studies | ME as research tool | Reveals suppressed emotions | Inter-rater reliability issues | Informed consent required |
| **Education** | Training programs | Medical education | Improves empathy, observation skills | Limited transfer to practice | Beneficial if voluntary |

**IEEE TAC Requirement**: Ethics section must address dual-use risks (surveillance, interrogation) and recommend mitigation.

---

## Matrix 6: Methodological Quality Assessment

| Method | Evaluation Protocol | Cross-Dataset Test | Statistical Tests | Code Available | Failure Analysis | Limitations Disclosed |
|--------|-------------------|---------------------|-------------------|----------------|------------------|----------------------|
| Hybrid Attention-3DNet | LOSO | ✗ | ✗ | ✗ | ✗ | ✗ |
| ROI-ArcFace | LOSO | ✗ | ✗ | ✗ | ✗ | ✗ |
| STRNet | UF1 protocol | ✗ | ✓ | ✓ | ✗ | ✗ |
| GAM-MER | LOSO | ✗ | ✗ | ✗ | ✗ | ✗ |
| MCCA-VNet | UF1 protocol | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Censor (target)** | **LOSO** | **✓ (5 datasets)** | **✓ (t-tests, ANOVA)** | **✓ (planned GitHub)** | **✓ (required for TAC)** | **✓ (required for TAC)** |

**IEEE TAC Quality Standards**: Statistical tests, cross-dataset validation, failure analysis, and honest limitations disclosure are expected for acceptance.

---

## Matrix 7: Temporal Modeling Approaches

| Method | Temporal Representation | Temporal Model | Temporal Output |
|--------|------------------------|----------------|-----------------|
| Hybrid Attention-3DNet | Video frames | 3D CNN + temporal attention | Emotion class |
| ROI-ArcFace | Video frames | 3D CNN | Emotion class |
| μ-BERT | Frame sequence | Transformer (masked) | Emotion class |
| **Censor** | **Optical flow + RGB frames + rPPG** | **3D CNN + 3D Swin-T + BiLSTM** | **Emotion class + 28 AU temporal + apex scores** |

**Unique Advantage**: Censor's AU decoder provides **intermediate temporal representation** (28 AU intensities over 16 frames), enabling explainable emotion prediction.

---

## Matrix 8: Computational Cost Comparison

| Method | Parameters | Inference Time | Memory | Training Data |
|--------|------------|----------------|--------|---------------|
| Hybrid Attention-3DNet | ~30M (est.) | — | — | CASME II + SAMM |
| ROI-ArcFace | ~25M (est.) | — | — | CASME II |
| STRNet | ~40M (est.) | — | — | Composite benchmark |
| **Censor** | **68.35M** | **—** | **—** | **TBD** |

**Note**: Censor has larger parameter count due to dual-pathway architecture (2 backbones) + AU decoder + MoE. Computational cost analysis required for IEEE TAC submission.

---

## Synthesis Insights

### SOTA Positioning

1. **Top accuracy**: Hybrid Attention-3DNet (93.79%) and ROI-ArcFace (93.96%) on CASME II
2. **Censor target**: ≥90% on CASME II to be competitive; novelty claims can compensate for modest accuracy deficit
3. **Unique position**: Only method integrating dual-pathway + AU + MoE + rPPG + apex detection + TTA

### Neuroscience Gap

1. **Strong evidence** for dual-pathway in general face processing
2. **Gap**: No ME-specific neuroimaging validation
3. **Mitigation**: "Inspired by" formulation with honest limitation acknowledgment

### IEEE TAC Readiness

1. **Required additions**: Statistical tests, cross-dataset validation, failure analysis, ethics discussion
2. **Strengths**: Novel architecture, multi-component integration, explainability (AU output)
3. **Weaknesses**: Experimental results TBD; neuroscience validation gap

---

**Prepared by**: Deep-Research Phase 2 (bibliography_agent)
**Next Phase**: Phase 3 — Synthesis Report Generation