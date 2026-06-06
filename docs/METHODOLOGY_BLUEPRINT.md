# Methodology Blueprint: Censor MER for IEEE TAC

**Project**: Biomimetic Dual-Pathway Micro-Expression Recognition System
**Research Question**: See RESEARCH_QUESTION_BRIEF.md
**Target Venue**: IEEE Transactions on Affective Computing
**Generated**: 2026-06-03

---

## 1. Research Paradigm

**Type**: Applied Computational Research with Neuroscience-Inspired Design

**Characteristics**:
- **Primary contribution**: Novel architecture design (biomimetic dual-pathway)
- **Validation strategy**: Computational benchmarking + planned human evaluation
- **Theoretical grounding**: Neuroscience-inspired, not neuroscience-validated
- **Application focus**: Affective computing (emotion recognition, clinical scenarios)

**IEEE TAC Alignment**:
- Falls under "Computational Models of Affective Processes"
- Requires: quantitative benchmarks, qualitative analysis, application demonstration
- Notifies: honest limitation disclosure on neuroscience validation

---

## 2. Method Selection

### 2.1 Core Research Methods

| Method | Purpose | IEEE TAC Evidence Tier |
|--------|---------|------------------------|
| **Architecture Design** | Novel dual-pathway instantiation | Tier 1 (primary contribution) |
| **Benchmark Evaluation** | Accuracy validation on CASME II/SAMM/SMIC/MMEW | Tier 1 (required) |
| **SOTA Comparison** | Positioning against Hybrid Attention-3DNet, ROI-ArcFace, STRNet | Tier 1 (required) |
| **Component Analysis** | AU decoder, apex detection, MoE routing contribution | Tier 2 (explainability) |
| **Cross-Dataset Validation** | Generalization across datasets | Tier 2 (robustness) |
| **Failure Analysis** | Limitation disclosure, error cases | Tier 2 (honest assessment) |
| **Human Evaluation** | Planned student experiments (July 2026) | Tier 3 (application validation) |
| **Neuroscience Literature Review** | Grounding for biomimetic claims | Tier 3 (theoretical context) |

### 2.2 Evaluation Protocol

**Quantitative Metrics**:
- **Primary**: Accuracy (7-class), F1-score, Unweighted F1 (UF1)
- **Secondary**: AU detection accuracy (28-class multi-label), AU F1
- **Structural**: SSIM (for optical flow quality), temporal coherence
- **Computational**: Parameters (68.35M), inference time, memory footprint

**Qualitative Analysis**:
- Attention visualization (amygdala gate, FFA, CASANet)
- AU activation heatmaps
- Apex frame detection demonstration
- Failure case categorization

**Statistical Tests**:
- t-tests for pairwise method comparison
- ANOVA for multi-method comparison
- Confidence intervals (95%) for all accuracy claims
- Cross-dataset significance tests

---

## 3. Data Strategy

### 3.1 Primary Benchmark Datasets

| Dataset | Samples | Subjects | FPS | Classes | Protocol |
|---------|---------|----------|-----|---------|----------|
| **CASME II** | 247 | 26 | 200 | 5-7 | Leave-one-subject-out (LOSO) |
| **SAMM** | 159 | 32 | 200 | 7-8 | LOSO |
| **SMIC-HS** | 164 | 16 | 100 | 3 | LOSO |
| **MMEW** | 300 | 36 | 90 | 7 | LOSO + macro-expression separation |
| **CAS(ME)³** | ~300+ | — | 30 | 4+ | Spontaneous protocol |

**LOSO Protocol Rationale**:
- IEEE TAC standard for MER evaluation
- Prevents subject-specific overfitting
- Enables cross-subject generalization claims
- Compatible with MEGC challenge protocols

### 3.2 Data Preprocessing

| Step | Module | Output | Validation Required |
|------|--------|--------|---------------------|
| Face Detection | External (MTCNN/RetinaFace) | 224×224 crop | Detection rate >95% |
| Alignment | External | Landmark-normalized | 68-point landmarks |
| Temporal Segmentation | Onset-Apex-Offset labels | 16-frame clips | Ground truth provided |
| Optical Flow | TVL1OpticalFlow | 2-channel flow | Visual inspection sample |
| rPPG Extraction | rPPGExtractor | 3-channel signal | SNR analysis |
| Saliency Map | SaliencyDetector | 1-channel map | Center bias verification |

### 3.3 Data Limitations

**Critical Gaps**:
- Most datasets require **signed license agreements** — accessibility barrier for reproducibility
- Sample sizes small (247 max) — statistical power limited
- Class imbalance (some emotions underrepresented)
- Cultural bias (primarily Chinese/European subjects)

**IEEE TAC Disclosure Required**: Dataset limitations section in Methods

---

## 4. Analytical Framework

### 4.1 Architecture Contribution Analysis

**Deconstruction Strategy**:

| Component | Contribution Claim | Ablation Test |
|-----------|-------------------|---------------|
| **Dual-Pathway** | Fast/slow pathway synergy | Single-pathway baseline |
| **Amygdala Gate** | Attention prior modulation | No-attention baseline |
| **FFA** | Cross-pathway gating | Concat baseline |
| **CASANet** | Apex frame detection | Random frame baseline |
| **TSFmicroFusion** | Bidirectional fusion | Unidirectional baseline |
| **AU Decoder** | Temporal AU modeling | Static AU baseline |
| **MoE Head** | Expert specialization | Single MLP baseline |
| **PersonalizedRadar** | Test-time adaptation | No-TTA baseline |

**Ablation Protocol**:
- Remove each component individually
- Measure accuracy drop
- Statistical significance test
- Contribution ranking

### 4.2 Neuroscience Grounding Analysis

**Literature Review Framework**:

| Neuroscience Claim | Source Evidence | Extrapolation Gap | Paper Statement |
|--------------------|-----------------|-------------------|-----------------|
| Dual-pathway architecture | Patient studies, fMRI (FFA vs amygdala) | Macro-expression, not ME-specific | "Inspired by" formulation |
| Fast subcortical route (~100ms) | MEG amygdala timing studies | Threat detection, not emotion discrimination | Timing-compatible claim |
| Fusiform-amygdala circuit | DTI connectivity studies | General face processing | Structural analogy claim |
| Amygdala attention modulation | fMRI attention studies | Macro-expression attention | Functional analogy claim |

**Honest Disclosure Protocol**:
- Each biomimetic claim accompanied by:
  - (a) Neuroscience source citation
  - (b) Evidence type (fMRI, MEG, patient study, DTI)
  - (c) Limitation (extrapolation from macro-expression)
  - (d) Claim strength ("inspired by" vs "validated by")

### 4.3 SOTA Positioning Analysis

**Comparison Framework**:

| Method | Year | Backbone | CASME II | SAMM | SMIC | Architecture Type |
|--------|------|----------|---------|------|------|-------------------|
| Hybrid Attention-3DNet | 2025 | 3D CNN + SE | 93.79% | 93.61% | 93.42% | Single-pathway + attention |
| ROI-ArcFace | 2025 | CNN + ROI | 93.96% | 86.15% | 81.17% | Region-based metric learning |
| STRNet | 2025 | Region-based | — | — | — | Spatiotemporal reasoning |
| GAM-MER | 2024 | Graph + Transformer | 91.57% | 91.25% | 86.22% | Graph attention |
| MCCA-VNet | 2024 | ViT + XCiT + CBAM | — | — | — | Multi-architecture fusion |
| μ-BERT | 2024 | BERT-style | 90.34% | — | 85.80% | Sequence modeling |
| Censor | 2025 | **Dual-pathway** + MoE | **TBD** | **TBD** | **TBD** | Biomimetic dual-pathway |

**Positioning Strategy**:
- If accuracy ≥90%: Competitive with SOTA; novelty claim (biomimetic architecture) + explainability advantage
- If accuracy 80-90%: Novelty-focused; architecture contribution + AU analysis + TTA practical value
- If accuracy <80%: Honest limitation statement; focus on architectural innovation rather than performance claims

**IEEE TAC Fairness Requirement**:
- Same evaluation protocol (LOSO) for all methods
- Same dataset splits (public standard splits)
- Same preprocessing (no method-specific augmentation)
- Statistical significance tests for all comparisons

---

## 5. Validity Criteria

### 5.1 Internal Validity

| Threat | Mitigation |
|--------|------------|
| **Overfitting** | LOSO protocol, cross-dataset validation, early stopping |
| **Data leakage** | Strict subject separation, temporal boundary enforcement |
| **Implementation bugs** | Modular testing, unit tests for each component |
| **Hyperparameter tuning** | Fixed hyperparameters across all experiments, grid search on validation set only |
| **Random seed variance** | Report mean ± std across multiple seeds (5 runs) |

### 5.2 External Validity

| Threat | Mitigation |
|--------|------------|
| **Dataset specificity** | Multi-dataset validation (5 benchmarks) |
| **Cultural bias** | Acknowledge dataset limitations; cross-cultural study as future work |
| **Protocol specificity** | Follow MEGC standard protocols; compare with published SOTA protocols |
| **Model specificity** | Ablation tests demonstrate component contributions |

### 5.3 Construct Validity

| Threat | Mitigation |
|--------|------------|
| **Accuracy ≠ Recognition Quality** | Add AU detection, apex detection, temporal coherence metrics |
| **Single metric bias** | Multi-metric evaluation (accuracy, F1, UF1) |
| **Machine ≠ Human perception** | Planned human evaluation study (July 2026) |

### 5.4 Ecological Validity

| Threat | Mitigation |
|--------|------------|
| **Lab-controlled datasets** | Acknowledge spontaneous/posed distinction; CAS(ME)³ for spontaneous ME |
| **Static evaluation** | Video-level evaluation, not frame-level |
| **No real-world test** | Limitation section: "Evaluation on benchmark datasets; real-world deployment requires further validation" |

---

## 6. IRB and Ethics Protocol

### 6.1 Human Evaluation Plan (July 2026)

**Study Design** (per PUBLICATION_PLAN_TAC.md):
- Participants: Students (n TBD)
- Task: ME recognition on generated/sampled clips
- Data collection: 2000+ feedback samples
- Metrics: Recognition accuracy, naturalness rating, training utility rating

**IRB Requirements**:
- Informed consent forms
- Data anonymization protocol
- Participant compensation disclosure
- Dual-use risk assessment (deception detection applications)

**IEEE TAC Requirement**: IRB approval documented in Methods section

### 6.2 AI Disclosure Statement

**IEEE TAC Requirement**: Explicit statement of AI tool usage in paper preparation.

**Template**:
> "This manuscript was prepared with the assistance of [tool names] for [specific tasks: literature search, grammar checking, formatting]. All scientific content, experimental design, and conclusions were authored by the human researchers. AI-generated text was not used for scientific claims or results interpretation."

### 6.3 Dual-Use Consideration

**Risk**: ME recognition technology can be used for:
- Positive: Clinical training, psychological assessment, communication enhancement
- Negative: Covert surveillance, interrogation enhancement, privacy violation

**IEEE TAC Requirement**: Ethics section discussing dual-use risks and mitigation recommendations.

---

## 7. Timeline and Milestones

| Phase | Timeline | Deliverable | IEEE TAC Readiness |
|-------|----------|-------------|-------------------|
| **Phase 1** | Now-June 2026 | RQ Brief, Methodology Blueprint (this document) | Design complete |
| **Phase 2** | July 2026 | Literature Matrix, SOTA Recognition Survey | Context complete |
| **Phase 3** | August 2026 | Benchmark experiments (CASME II/SAMM/SMIC) | Results TBD → Resolved |
| **Phase 4** | July 2026 | Human evaluation study | Application validation |
| **Phase 5** | September 2026 | Paper draft revision, figures, tables | Manuscript complete |
| **Phase 6** | October 2026 | Final integrity check, submission | Submit to IEEE TAC |

**Critical Path**: Experimental results (Phase 3) → Paper revision (Phase 5) → Submission (Phase 6)

---

## 8. Expected Deliverables

### 8.1 Paper Structure (IEEE TAC Format)

| Section | Content | Status |
|---------|---------|--------|
| **Abstract** | Problem, method, results, application (200 words) | Draft exists, results TBD |
| **Introduction** | ME importance, data scarcity, biomimetic motivation (1000 words) | Draft complete |
| **Related Work** | MER methods, Transformers, AU detection, MoE, biomimetic computing (1500 words) | Draft complete, SOTA survey needs replacement |
| **Method** | 11 modules detailed, equations, diagrams (3000 words) | Draft complete |
| **Experiments** | Datasets, settings, quantitative results, qualitative analysis, ablation (2500 words) | **TBD** — requires experimental runs |
| **Discussion** | Neuroscience grounding analysis, limitations, applications, ethics (1000 words) | Partial — needs neuroscience literature review |
| **Conclusion** | Summary, future work (500 words) | Draft complete |
| **References** | 46+ citations | Draft complete, needs MER recognition additions |

### 8.2 Supplementary Materials

| Material | Content | Status |
|----------|---------|--------|
| **Code** | GitHub repository with trained models | Planned |
| **Video Demo** | ME recognition examples, attention visualization | Planned |
| **Dataset Access** | License information, preprocessing scripts | Planned |
| **Human Evaluation Data** | Student feedback dataset (after July 2026) | Planned |

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Experimental results below SOTA** | Medium | High | Focus on novelty + explainability; honest limitation statement |
| **IRB approval delayed** | Low | Medium | Early submission; backup plan for human eval |
| **Neuroscience reviewer objection** | Medium | Medium | Honest disclosure; "inspired by" formulation |
| **Dataset license barrier** | Low | Low | Public dataset information; code for reproducibility |
| **SOTA method comparison unfairness** | Medium | High | Strict protocol matching; statistical tests |

---

## 10. Success Criteria

### 10.1 Publication Acceptance

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Novelty** | Clear architectural contribution | Reviewer assessment |
| **Validation** | Benchmark results + human evaluation | Tables II-VI complete |
| **Honest Disclosure** | Neuroscience gap acknowledged | Discussion section |
| **Fair Comparison** | Same protocols for SOTA | Methods section |
| **Ethics Compliance** | IRB + AI disclosure + dual-use discussion | Ethics section |

### 10.2 Scientific Contribution

| Contribution | Target | Evidence |
|--------------|--------|----------|
| **Architecture** | First explicit dual-pathway MER | Related Work positioning |
| **Explainability** | AU + attention visualization | Qualitative section |
| **Application** | Clinical training potential | Human evaluation + Discussion |
| **Reproducibility** | Code + models + dataset info | Supplementary materials |

---

## 11. IRON RULES Compliance

1. **All claims must have citations** — Neuroscience claims cite literature; SOTA metrics cite papers; dataset stats cite original papers
2. **Evidence hierarchy** — Benchmark results (cohort-level) + neuroscience literature (case-series) clearly distinguished
3. **Contradictions disclosed** — TBD results acknowledged; neuroscience extrapolation limitation explicit
4. **AI disclosure** — IEEE TAC template included in Methods

---

**Prepared by**: Deep-Research Phase 1 (research_architect_agent)
**Reviewed by**: ARS Orchestrator
**Status**: Phase 1 Complete → Proceed to Phase 2 (Bibliography)