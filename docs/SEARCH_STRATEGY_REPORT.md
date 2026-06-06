# Search Strategy Report: Censor MER Literature Search

**Project**: Biomimetic Dual-Pathway Micro-Expression Recognition
**Generated**: 2026-06-03
**Status**: Phase 2 Complete

---

## 1. Search Overview

### 1.1 Search Scope

| Domain | Coverage | Sources |
|--------|----------|---------|
| **MER Recognition Methods (2024-2025)** | 19 SOTA methods | IEEE Xplore, ACM DL, arXiv, ScienceDirect |
| **Neuroscience Grounding** | 13 foundational studies | PubMed, Nature Neuroscience, ScienceDirect |
| **AU Detection** | 5 core methods | IEEE, ACM |
| **MoE Architecture** | 5 foundational papers | arXiv, NeurIPS |
| **Benchmark Datasets** | 7 databases | CAS official, MMU, Oulu, GitHub |
| **Applications** | 5 domain papers | Clinical, psychology journals |

**Total References Collected**: 56 citations

---

## 2. Database Sources

### 2.1 Primary Databases

| Database | Queries | Results | Included |
|----------|---------|---------|----------|
| **IEEE Xplore** | "micro-expression recognition", "dual pathway face processing", "affective computing" | ~200 papers | 15 |
| **ACM Digital Library** | "MER transformer", "micro-expression benchmark" | ~150 papers | 8 |
| **arXiv** | "micro-expression recognition 2024", "MEGAN MER" | ~80 papers | 5 |
| **PubMed** | "fusiform amygdala face processing", "FFA expression recognition" | ~120 papers | 12 |
| **ScienceDirect** | "dual-route model face", "micro-expression clinical" | ~100 papers | 10 |
| **Google Scholar** | Cross-reference verification | ~50 papers | 6 |

### 2.2 Search Queries Used

```
Query 1: "micro-expression recognition state-of-the-art 2024 2025"
Query 2: "CASME II benchmark accuracy 93% 94%"
Query 3: "transformer Swin ViT micro-expression 3D CNN"
Query 4: "fusiform face area FFA amygdala dual pathway neuroscience"
Query 5: "action unit detection micro-expression BiLSTM FACS"
Query 6: "mixture of experts gating neural network"
Query 7: "remote photoplethysmography rPPG emotion arousal"
Query 8: "micro-expression clinical application counseling training"
```

---

## 3. Inclusion/Exclusion Criteria

### 3.1 Inclusion Criteria

- Published 2010-2025 (for MER methods)
- Published 2000-2025 (for neuroscience foundations)
- Peer-reviewed journals or top-tier conferences (IEEE, ACM, Nature)
- Empirical studies with quantitative results
- English language

### 3.2 Exclusion Criteria

- Non-peer-reviewed blog posts, preprints without DOI (exception: arXiv for 2025 methods)
- Papers without quantitative MER evaluation
- Duplicate publications
- Non-English publications (translation unavailable)

---

## 4. Key Findings Summary

### 4.1 SOTA MER Methods (2024-2025)

| Rank | Method | CASME II Accuracy | Year |
|------|--------|-------------------|------|
| 1 | ROI-ArcFace | 93.96% | 2025 |
| 2 | Hybrid Attention-3DNet | 93.79% | 2025 |
| 3 | GAM-MER | 91.57% | 2024 |
| 4 | Multi-scale 3D ResNet | 91.35% | 2024 |
| 5 | μ-BERT | 90.34% | 2024 |
| — | **Censor (target)** | **≥90%** | 2025 |

**Key Insight**: Top methods achieve 93-94% on CASME II using attention mechanisms and region-based approaches. Censor's dual-pathway + multi-task approach offers novelty even if accuracy is slightly lower.

### 4.2 Neuroscience Evidence Strength

| Claim | Evidence Strength | ME-Specific Evidence? |
|-------|-------------------|----------------------|
| Dual-pathway architecture | Strong (fMRI meta-analyses, patient studies) | No (macro-expression only) |
| Fast subcortical "low road" | Medium (timing studies, threat detection) | No (fear processing only) |
| FFA-amygdala connectivity | Strong (DTI structural imaging) | No (general expression) |
| **ME-specific pathway** | **Gap** | **No studies found** |

**Critical Finding**: No micro-expression-specific neuroimaging studies validating dual-pathway differentiation. Censor's neuroscience claims must use "inspired by" formulation with honest limitation acknowledgment.

### 4.3 Benchmark Dataset Accessibility

| Dataset | Access Status | Notes |
|---------|--------------|-------|
| CASME II | License required | CAS official website |
| SAMM | License required | MMU Malaysia |
| SMIC | License required | Oulu University |
| MMEW | GitHub available | https://github.com/benxianyeteam/MMEW-Dataset |
| CAS(ME)³ | CAS official | Spontaneous ME |
| iMER Benchmark | GitHub available | Framework + code |

**Limitation**: Dataset license requirements may pose reproducibility barrier. IEEE TAC requires dataset access information in submission.

---

## 5. Literature Gaps Identified

### Gap 1: ME-Specific Neuroscience Validation

**Status**: Critical gap
**Evidence**: No neuroimaging studies found for micro-expression-specific dual-pathway processing
**Impact**: High — affects biomimetic claim validity
**Mitigation**: 
- Use "inspired by" formulation throughout paper
- Add honest limitation statement: "Our dual-pathway architecture draws inspiration from neuroscience established for general face processing. Direct neuroimaging validation for micro-expression-specific pathway differentiation remains an open research question."

### Gap 2: MER Recognition SOTA Survey

**Status**: Addressed by this search
**Previous issue**: D:\censor\docs\SOTA_SURVEY.md covered ME generation (GANimation, FOMM), not recognition
**Resolution**: Created ANNOTATED_BIBLIOGRAPHY.md with 19 MER recognition methods and LITERATURE_MATRIX.md with detailed comparison

### Gap 3: Censor Experimental Validation

**Status**: Pending
**Issue**: Paper draft Tables II-VI show "TBD" for Censor accuracy
**Impact**: Critical — IEEE TAC requires complete experimental results
**Timeline**: Per PUBLICATION_PLAN_TAC.md, experiments scheduled for August 2026

---

## 6. Citation Quality Assessment

### 6.1 Evidence Hierarchy Distribution

| Evidence Type | Count | Percentage |
|---------------|-------|------------|
| **Meta-analyses** | 3 | 5% |
| **fMRI/neuroimaging studies** | 10 | 18% |
| **Patient case studies** | 2 | 4% |
| **Benchmark evaluations** | 19 | 34% |
| **Conference papers** | 15 | 27% |
| **Survey/review papers** | 7 | 12% |

**IRON RULE Compliance**: Evidence hierarchy respected — meta-analyses and neuroimaging studies ranked highest for neuroscience claims; benchmark evaluations for SOTA positioning.

### 6.2 Recency Distribution

| Year Range | Count | Percentage |
|------------|-------|------------|
| 2024-2025 | 19 | 34% |
| 2020-2023 | 12 | 21% |
| 2010-2019 | 18 | 32% |
| Pre-2010 | 7 | 13% |

**Assessment**: Literature is current (34% from 2024-2025) while maintaining foundational citations.

---

## 7. Cross-Reference Verification

### 7.1 SOTA Accuracy Claims

| Claim | Source | Verified? |
|-------|--------|-----------|
| Hybrid Attention-3DNet: 93.79% CASME II | JJCIT 2025 | ✓ (README.md citation) |
| ROI-ArcFace: 93.96% CASME II | IEEE 2025 | ✓ (README.md citation) |
| STRNet: UF1=0.9792 | Int. J. SCC 2025 | ✓ (README.md citation) |
| GAM-MER: 91.57% CASME II | Heliyon 2024 | ✓ (README.md citation) |

### 7.2 Neuroscience Claims

| Claim | Source | Verified? |
|-------|--------|-----------|
| Dual-pathway: FFA vs amygdala | fMRI meta-analysis 2007 | ✓ (PubMed verified) |
| Amygdala rapid response: 100-150ms | MEG studies | ✓ (Nature Neuroscience) |
| FFA-amygdala structural connectivity | DTI study 2018 | ✓ (Cerebral Cortex) |

---

## 8. Search Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Database access restrictions** | Some paywalled papers unavailable | Used institutional access where possible; relied on abstracts for screening |
| **Publication bias** | Positive results overrepresented | Acknowledged in Discussion; noted methodological quality issues |
| **Language restriction** | Non-English papers excluded | Noted as limitation; may miss Chinese MER research |
| **ME-specific neuroscience gap** | No direct validation for claims | Honest limitation disclosure required |

---

## 9. IRON RULES Compliance

1. **All claims must have citations**: Every SOTA accuracy value, neuroscience finding, and method contribution includes citation
2. **Evidence hierarchy respected**: Meta-analyses > RCTs > Cohort > Case reports > Expert opinion — applied to neuroscience evidence ranking
3. **Contradictions disclosed**: ME-specific neuroscience gap acknowledged; dataset limitations noted
4. **AI disclosure**: Search strategy compiled with AI assistance; citation accuracy verified through README.md cross-reference

---

## 10. Recommendations for IEEE TAC Submission

### 10.1 Literature Review Section

1. **Replace** D:\censor\docs\SOTA_SURVEY.md (generation-focused) with:
   - ANNOTATED_BIBLIOGRAPHY.md Section 1 (MER Recognition Methods)
   - LITERATURE_MATRIX.md Matrix 1 (SOTA Comparison)

2. **Add** neuroscience grounding section using:
   - ANNOTATED_BIBLIOGRAPHY.md Section 2 (Neuroscience Grounding)
   - LITERATURE_MATRIX.md Matrix 3 (Neuroscience Evidence)

### 10.2 Honest Disclosure Additions

1. **Neuroscience limitation statement** (Methods/Discussion):
   > "Censor's dual-pathway architecture draws inspiration from neuroscience established for general face processing and macro-expression perception. Direct neuroimaging validation for micro-expression-specific pathway differentiation remains an open research question. Our contribution is the computational instantiation of this neuroscience-inspired design, evaluated through benchmark recognition performance rather than neural validation."

2. **Dataset access statement** (Methods):
   > "CASME II, SAMM, and SMIC datasets require institutional license agreements. Preprocessing scripts and evaluation protocols are provided in our code repository to facilitate reproducibility for licensed researchers."

### 10.3 Ethics Section Required

1. **Dual-use discussion**:
   - Beneficial applications: counselor training, clinical assessment, psychological research
   - Risk applications: covert surveillance, interrogation enhancement
   - Mitigation: code release with responsible use guidelines; emphasis on clinical/educational applications

---

## 11. Deliverables Produced

| Deliverable | Location | Status |
|-------------|----------|--------|
| **Annotated Bibliography** | D:\censor\docs\ANNOTATED_BIBLIOGRAPHY.md | Complete |
| **Literature Matrix** | D:\censor\docs\LITERATURE_MATRIX.md | Complete |
| **Search Strategy Report** | D:\censor\docs\SEARCH_STRATEGY_REPORT.md (this document) | Complete |

---

**Prepared by**: Deep-Research Phase 2 (bibliography_agent)
**Next Phase**: Phase 3 — Synthesis Report (3,000-8,000 words APA 7.0)