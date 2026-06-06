# Research Question Brief: Censor MER for IEEE TAC

**Project**: Biomimetic Dual-Pathway Micro-Expression Recognition System (Censor)
**Target Venue**: IEEE Transactions on Affective Computing (TAC)
**Impact Factor**: 8.5+ | **Acceptance Rate**: ~20-25%
**Generated**: 2026-06-03

---

## 1. Primary Research Question

> **How can a biomimetic dual-pathway architecture emulating the fusiform-amygdala circuit improve micro-expression recognition accuracy and explainability while demonstrating practical value in affective computing applications?**

---

## 2. FINER Criteria Scoring

| Criterion | Score (1-5) | Justification |
|-----------|-------------|---------------|
| **Feasible** | 4 | Architecture (68.35M params, 11 modules) implemented in PyTorch; benchmark datasets (CASME II, SAMM, SMIC, MMEW) accessible via license agreements; training pipeline complete. **GAP**: Experimental results TBD in paper draft — need validation runs. |
| **Interesting** | 5 | First explicit instantiation of dual-pathway fusiform-amygdala architecture for MER; MoE gating + AU temporal modeling + rPPG physiology integration novel combination; addresses explainability gap in deep MER systems. IEEE TAC readership interested in neuroscience-grounded affective computing. |
| **Novel** | 4 | Biomimetic architectural design claims novelty, but **critical gap**: neuroscience validation literature focuses on macro-expression, not micro-expression specific pathways. Needs explicit acknowledgment that dual-pathway analogy is inspired by macro-expression neuroscience, with limited ME-specific validation. SOTA comparison (Hybrid Attention-3DNet 93.79%, STRNet UF1=0.9792) provides clear positioning target. |
| **Ethical** | 3 | **Concern**: Human evaluation planned (student experiments July 2026 per PUBLICATION_PLAN_TAC.md) but IRB approval status unclear. Data collection (2000+ feedback samples) requires ethical oversight. Deception detection applications raise dual-use concerns. IEEE TAC requires explicit AI disclosure and ethical statement. |
| **Relevant** | 5 | Directly aligns with IEEE TAC scope: affective computing, emotion recognition, computational models of emotional processes. Clinical applications (counselor training, psychological assessment) match TAC's applied affective computing focus. |

**Total FINER Score**: 21/25 (84%)

---

## 3. Scope Boundaries

### 3.1 In-Scope

1. **Architectural Innovation**
   - Dual-pathway design (fast subcortical: 3D ResNet-18 on optical flow; slow cortical: 3D Swin Transformer on RGB+rPPG)
   - Fusiform-amygdala attention circuit (AmygdalaGate, FFA, CASANet apex detection)
   - TSFmicroFusion bidirectional cross-attention (1024-D)
   - Dynamic AU Decoder (BiLSTM, 28 AUs)
   - MoE gating (3 experts, noisy top-2, load-balancing)
   - PersonalizedRadar test-time adaptation

2. **Benchmark Evaluation**
   - CASME II (247 samples, 26 subjects, 200 fps)
   - SAMM (159 samples, 32 subjects, 200 fps)
   - SMIC-HS (164 samples, 16 subjects, 100 fps)
   - MMEW (300 samples, 36 subjects, 90 fps)
   - CAS(ME)³ (~300+ samples, 30 fps)

3. **Affective Computing Applications**
   - Emotion recognition accuracy benchmarking
   - AU detection as intermediate representation
   - rPPG physiological signal integration
   - Template-based emotion reporting

4. **SOTA Comparison (2024-2025)**
   - Hybrid Attention-3DNet (JJCIT 2025): 93.79% CASME II
   - ROI-ArcFace (IEEE 2025): 93.96% CASME II
   - STRNet (Int. J. SCC 2025): UF1=0.9792
   - GAM-MER (Heliyon 2024): 91.57% CASME II
   - MCCA-VNet (PMC 2024): UF1=0.868

### 3.2 Out-of-Scope

1. **Micro-Expression Generation** — Separate project (diffusion+Blendshape+GAN per PUBLICATION_PLAN_TAC.md)
2. **Real-time Deployment Optimization** — Focus is on architectural novelty, not inference speed
3. **Cross-cultural ME Analysis** — Datasets are primarily Chinese/European; cultural generalization deferred
4. **Macro-Expression Recognition** — Censor optimized for 40-200ms ME duration specifically
5. **Neuroimaging Validation** — No fMRI/EEG studies planned to validate biomimetic claims with human subjects

### 3.3 Limitations Acknowledgment (Critical)

**Neuroscience Evidence Gap**:
- The dual-pathway fusiform-amygdala circuit neuroscience literature (FFA-amygdala connectivity, subcortical "low road" pathway) primarily addresses **macro-expression** and general face processing, not micro-expression specifically.
- Neuroimaging evidence shows amygdala responds to fearful faces within 100-150ms (preceding FFA), but ME-specific timing (40-200ms total duration) may not allow full pathway differentiation.
- **Honest claim**: Censor's architecture is "inspired by" dual-pathway neuroscience rather than "validated by" ME-specific neuroscience evidence. This distinction must be explicit in IEEE TAC submission to avoid overstating biomimetic grounding.

**Experimental Results TBD**:
- Paper draft (paper_ieee_en.md) Tables II-VI show "TBD" for Censor accuracy on all datasets.
- IEEE TAC requires complete experimental validation. Options:
  - (a) Run full benchmark experiments before submission
  - (b) Honest limitation statement: "Experimental validation pending; this paper presents architectural design and theoretical framework"
  - (c) Submit as "method paper" with preliminary synthetic data results (not recommended for TAC standards)

---

## 4. Sub-Questions

### SQ1: Neuroscience Grounding Validation

> **What is the empirical neuroscience evidence supporting dual-pathway architecture for micro-expression processing, and what limitations exist in extrapolating from macro-expression face processing literature?**

**Evidence Status**:
- Strong evidence for dual-pathway in **general face processing**: FFA for identity, amygdala/STS for expression (patient studies, fMRI, MEG)
- **Subcortical "low road"**: Superior colliculus → Pulvinar → Amygdala responds to fearful faces in ~100ms
- **Critical gap**: No ME-specific neuroimaging studies demonstrating pathway differentiation for 40-200ms expressions
- **Claim adjustment**: "Inspired by" vs "Validated by" distinction essential

**Sources**:
- [Dual neural pathways for face processing](https://www.sciencedirect.com/science/article/pii/S1053811907001234)
- [Amygdala and FFA: Parallel or interactive](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3125678/)
- [Dissociable roles of FFA and amygdala](https://www.jneurosci.org/content/28/9/552)

### SQ2: Affective Computing Benchmark Positioning

> **What performance threshold must Censor achieve to position competitively against 2024-2025 SOTA methods on IEEE TAC's evidentiary standards?**

**Threshold Analysis**:
- **CASME II target**: ≥90% accuracy (below Hybrid Attention-3DNet 93.79%, ROI-ArcFace 93.96%)
- **Competitive positioning**: Novelty claims (biomimetic architecture) can compensate for modest accuracy deficit IF:
  - (a) Explainability advantage demonstrated (AU analysis, attention visualization)
  - (b) Multi-task capability validated (AU detection + ME classification + rPPG)
  - (c) Test-time adaptation (PersonalizedRadar) shows practical value

**IEEE TAC Evidentiary Standards**:
- Must include: quantitative accuracy, qualitative visualization, statistical significance tests, cross-dataset generalization, failure case analysis
- SOTA comparison must be **fair** (same protocols, same splits) — direct table comparison required

### SQ3: Clinical Application Evidence

> **What empirical evidence supports micro-expression recognition utility in clinical/psychological scenarios, and what validation path exists for Censor's application claims?**

**Current Evidence**:
- **Counselor training**: METT (Micro Expression Training Tool) shows improved recognition in clinicians (Ekman 2002 onwards)
- **Clinical populations**: ME recognition impairment documented in schizophrenia, autism spectrum conditions
- **Deception detection**: Controversial — micro-expressions indicate emotional concealment, not definitive lying; false positive rates problematic

**Validation Path for Censor**:
1. **Phase 1**: Benchmark accuracy validation (machine evaluation)
2. **Phase 2**: Human evaluation study (student experiments July 2026 per PUBLICATION_PLAN_TAC.md)
3. **Phase 3**: Applied scenario pilot (counselor training simulation, psychological assessment tool)

**Ethical Requirement**: IRB approval for human evaluation data collection must be documented in IEEE TAC submission.

---

## 5. Honest Gap Assessment

### 5.1 Experimental Results TBD

**Current State**: Paper draft Tables II-VI show "TBD" for Censor results.

**IEEE TAC Requirement**: Complete experimental validation with:
- Accuracy metrics (accuracy, F1, UF1) on all benchmark datasets
- Cross-dataset generalization experiments
- Statistical significance tests (t-tests, ANOVA)
- Failure case analysis
- Computational cost analysis (params, inference time, memory)

**Resolution Options**:
- **Option A** (recommended): Run full benchmark experiments before submission (timeline: September per PUBLICATION_PLAN_TAC.md)
- **Option B**: Submit as "method proposal paper" — lower acceptance probability at TAC
- **Option C**: Preliminary results on synthetic data with honest limitation statement — likely reviewer objection

### 5.2 SOTA Survey Covers Generation, Not Recognition

**Current State**: D:\censor\docs\SOTA_SURVEY.md lists ME generation methods (GANimation, FOMM, TPSMM, LivePortrait).

**IEEE TAC Requirement**: SOTA comparison must address MER recognition methods, not generation.

**Resolution Required**:
- Replace SOTA_SURVEY.md with MER recognition survey covering:
  - Handcrafted methods (LBP-TOP, MDMO)
  - Deep learning methods (3D CNNs, Transformers)
  - Recent SOTA 2024-2025 (Hybrid Attention-3DNet, ROI-ArcFace, STRNet, GAM-MER, MCCA-VNet)
- Include MEGC (Micro-Expression Grand Challenge) results for context

### 5.3 Neuroscience Validation Gap

**Claim in Architecture**: "Simulates fusiform-amygdala circuit in human visual pathway"

**Evidence Reality**:
- Neuroscience literature validates dual-pathway for **general face processing** and **macro-expression**
- **No ME-specific** neuroimaging studies validating pathway differentiation for 40-200ms expressions
- Amygdala "low road" timing (~100ms) compatible with ME duration, but evidence is for threat detection, not subtle emotion discrimination

**IEEE TAC Requirement**: Honest acknowledgment of extrapolation from macro-expression neuroscience.

**Resolution Required**: Explicit statement in paper:
> "Censor's dual-pathway architecture is inspired by the fusiform-amygdala circuit established for general face processing and macro-expression perception. Direct neuroimaging validation for micro-expression-specific pathway differentiation remains an open research question. Our contribution is the architectural instantiation of this neuroscience-inspired design, evaluated through computational benchmarks rather than neural validation."

---

## 6. IEEE TAC Submission Readiness Assessment

| Component | Status | Action Required |
|-----------|--------|-----------------|
| Architecture Documentation | Complete | README.md + TECHNICAL_EN.md comprehensive |
| Mathematical Formulation | Complete | Equations in paper draft + README |
| SOTA Comparison | **Incomplete** | Replace generation survey with recognition survey |
| Experimental Results | **TBD** | Run benchmark experiments (timeline: Sept 2026) |
| Neuroscience Validation | **Gap** | Add honest acknowledgment of extrapolation |
| Clinical Applications | Planned | Student experiments July 2026; IRB approval needed |
| AI Disclosure | TBD | IEEE TAC requires explicit AI usage statement |
| Code Availability | Planned | GitHub release before submission |

**Timeline (per PUBLICATION_PLAN_TAC.md)**:
- Phase 1: Model training/validation (now)
- Phase 2: Human evaluation (July 2026)
- Phase 3: Iteration (August 2026)
- Phase 4: Paper writing (September 2026)
- Phase 5: Submission (October 2026)

---

## 7. IRON RULES Compliance

1. **All claims must have citations** — Neuroscience claims cite fMRI/MEG/patient studies; SOTA metrics cite specific papers
2. **Evidence hierarchy**: Meta-analyses > RCTs > Cohort > Case reports > Expert opinion — SOTA accuracy benchmarks are cohort-level (dataset studies); neuroscience is case-series + neuroimaging
3. **Contradictions disclosed**: Paper TBD results vs submission requirement acknowledged; neuroscience extrapolation limitation explicit
4. **AI disclosure**: Required in IEEE TAC submission

---

## 8. Next Steps

1. **Phase 2 (Investigation)**: Bibliography agent to compile MER recognition SOTA literature matrix
2. **Phase 3 (Analysis)**: Synthesis agent to produce gap analysis and positioning strategy
3. **Phase 4 (Composition)**: Report compiler to generate IEEE TAC-ready synthesis report
4. **Phase 5 (Review)**: Devil's Advocate checkpoint on neuroscience validation claims
5. **Phase 6 (Revision)**: Incorporate feedback and finalize

---

**Prepared by**: Deep-Research Phase 1 (research_question_agent)
**Reviewed by**: ARS Orchestrator
**Status**: Phase 1 Complete → Proceed to Phase 2