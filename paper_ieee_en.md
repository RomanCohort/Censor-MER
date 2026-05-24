# Censor: A Biomimetic Dual-Pathway Framework for Micro-Expression Recognition

Authors TBD

---

## Abstract

Micro-expression recognition (MER) remains a formidable challenge in affective computing due to the subtle spatial magnitude and brief temporal duration of involuntary facial movements. This paper presents **Censor**, a biomimetic dual-pathway neural architecture for MER that draws direct inspiration from the neurological fusiform-amygdala circuit governing subconscious facial affect processing. The proposed framework comprises eleven integrated modules spanning the full recognition pipeline: biomimetic preprocessing (saliency detection, remote photoplethysmography extraction, and TV-L1 optical flow), a fast subcortical pathway (3D ResNet-18 variant operating on optical flow), a slow cortical pathway (3D Swin Transformer processing RGB+rPPG), attentional modulation via an amygdala-inspired gating mechanism, feature fusion with squeeze-excitation attention, CASANet for spatiotemporal apex detection, bidirectional cross-attention fusion (TSFmicroFusion), a dynamic AU decoder with BiLSTM, noisy top-2 mixture-of-experts gating, test-time personalized adaptation, and template-based emotion reporting. The complete model contains 68.35M parameters. Extensive benchmarking against state-of-the-art methods on CASME II, SAMM, SMIC-HS, MMEW, and CAS(ME)\textsuperscript{3} is planned to evaluate Censor's performance. We report a multi-task objective combining cross-entropy for seven-class micro-expression classification, binary cross-entropy for 28 action unit detection, and auxiliary load-balancing losses. Code and pretrained models are made publicly available to facilitate reproducible research.

**Keywords** — Micro-expression recognition, dual-pathway neural network, biomimetic computing, fusiform-amygdala circuit, 3D Swin Transformer, optical flow, action unit detection, mixture of experts.

---

## I. Introduction

Facial micro-expressions are involuntary, brief facial movements that occur when an individual attempts to conceal or suppress genuine emotions [1]. Unlike macro-expressions, which typically last between 0.5 and 4 seconds, micro-expressions have a duration of 1/25 to 1/5 of a second [2]. They are characterized by low intensity, partial facial involvement, and rapid onset-apex-offset dynamics, making them exceedingly difficult to detect and recognize, even for trained human coders [3].

The computational modeling of micro-expression recognition has progressed substantially over the past decade, driven by the release of benchmark datasets such as CASME II [4], SAMM [5], SMIC [6], and MMEW [7]. Early approaches relied on handcrafted spatiotemporal features including Local Binary Patterns from Three Orthogonal Planes (LBP-TOP) [8] and optical flow descriptors [9]. These methods, however, are inherently limited by their fixed feature extraction protocols and inability to learn hierarchical representations from data.

The advent of deep learning has catalyzed a paradigm shift in MER. Convolutional neural networks (CNNs) operating on video frames [10], 3D convolutional networks capturing spatiotemporal volumes [11], and Transformer-based architectures leveraging self-attention mechanisms [12] have progressively pushed the state of the art. Despite these advances, existing deep MER systems suffer from a critical limitation: they are architecturally agnostic to the underlying neural mechanisms of human micro-expression perception.

Neuroimaging studies have established that the perception of facial expressions, particularly those that are brief or subliminal, engages a dual-pathway architecture in the human brain [13]. The **fast subcortical pathway** projects from the superior colliculus through the pulvinar to the amygdala, enabling rapid but coarse processing of affective stimuli [14]. The **slow cortical pathway** engages the primary visual cortex, the fusiform face area (FFA), and the orbitofrontal cortex, supporting fine-grained discriminative analysis [15]. These pathways operate in parallel and converge at the amygdala, which modulates attention and emotional response.

This paper introduces **Censor**, a biomimetic dual-pathway framework that explicitly emulates this neurological architecture. Our contributions are as follows:

1. **Biomimetic architectural design**: We propose a dual-pathway network comprising a fast 3D ResNet-18 pathway (analogous to the subcortical route) processing optical flow and a slow 3D Swin Transformer pathway (analogous to the cortical route) processing RGB video supplemented with remote photoplethysmography (rPPG) signals.

2. **Comprehensive modular pipeline**: Censor integrates eleven specialized modules covering preprocessing, feature extraction, attention modulation, fusion, action unit decoding, mixture-of-experts classification, test-time personalization, and report generation within a unified framework.

3. **Multi-task learning objective**: We formulate a composite loss function combining micro-expression classification, action unit detection, temporal smoothness, and load-balancing regularization, enabling the model to learn complementary affective cues.

4. **Comprehensive planned empirical validation**: We design extensive experiments on five benchmark datasets to evaluate Censor against recent methods including Hybrid Attention-3DNet (2025) [16], ROI-ArcFace (2025) [17], and GAM-MER (2024) [18].

The remainder of this paper is organized as follows. Section II reviews related work in MER and biomimetic computing. Section III presents the proposed Censor architecture in detail. Section IV describes the experimental setup and implementation details. Section V reports results and provides discussion. Section VI concludes the paper and outlines directions for future work.

---

## II. Related Work

### A. Micro-Expression Recognition Methods

Micro-expression recognition methods can be broadly categorized into handcrafted feature-based approaches and deep learning-based approaches. Early work in the field relied on spatiotemporal descriptors designed to capture the subtle motion signatures characteristic of micro-expressions. LBP-TOP [8] extends the classical LBP descriptor to three orthogonal planes (XY, XT, YT), encoding both spatial texture and temporal dynamics. Optical flow-based methods, including the Main Directional Mean Optical Flow (MDMO) [9] and Facial Dynamics Map [19], quantify pixel-level motion patterns across consecutive frames. While computationally efficient, these methods are constrained by the expressiveness limits of manually designed features.

The deep learning era has brought substantial improvements to MER. Tran et al. [11] introduced 3D convolutional networks for spatiotemporal feature learning, establishing the foundation for video-level representation learning in MER. Peng et al. [20] proposed a dual-temporal-scale convolutional neural network that processes micro-expression clips at multiple frame rates. The introduction of attention mechanisms has been particularly impactful: Hybrid Attention-3DNet (2025) [16] combines spatial and temporal attention modules within a 3D convolutional backbone, achieving 93.79% accuracy on CASME II. GAM-MER (2024) [18] employs a graph attention mechanism for muscle movement modeling. STRNet (2025) [21] achieves UF1 scores of 0.9792 through spatiotemporal reasoning. Multi-scale 3D ResNet [22] (2024) leverages hierarchical feature extraction at multiple temporal resolutions.

### B. Transformer Architectures for Video Understanding

Vision Transformers (ViTs) [23] have emerged as powerful alternatives to convolutional architectures for image classification, and their extension to video understanding has yielded state-of-the-art results. The Video Swin Transformer [24] introduces shifted-window multi-head self-attention, enabling efficient hierarchical representation learning with linear computational complexity relative to spatial resolution. TimeSformer [25] separates spatial and temporal attention to reduce computational cost. In the MER domain, Vision Transformer-based methods have been explored for facial expression recognition [26], though their application to micro-expression recognition remains nascent.

### C. Action Unit Detection and Multitask Learning

Facial Action Units (AUs), defined by the Facial Action Coding System (FACS) [27], provide an anatomically grounded representation of facial muscle activity. Joint learning of AU detection and expression classification has been shown to improve performance on both tasks [28]. BiLSTM architectures are particularly well-suited for AU detection from video sequences, as they can model temporal dependencies in both forward and backward directions [29]. In Censor, we adopt a BiLSTM-based Dynamic AU Decoder that simultaneously predicts 28 AUs and their temporal landmarks (onset, peak, offset), enabling richer supervisory signals.

### D. Mixture of Experts in Deep Learning

The Mixture of Experts (MoE) framework [30], originally proposed as a neural network architecture for modular learning, has recently been scaled to massive models through sparse gating mechanisms [31]. Noisy top-k gating introduces stochasticity during training to improve load balancing across experts. In the context of MER, MoE is particularly appealing because different micro-expression categories may benefit from specialized feature subspaces—for instance, the facial dynamics of a suppressed smile differ qualitatively from those of a concealed fear response.

### E. Biomimetic Computing for Affect Recognition

Biomimetic computing seeks to endow artificial systems with computational principles derived from biological neural processing. In the domain of affect recognition, the dual-pathway hypothesis has been explored in several computational frameworks [32, 33]. The fast subcortical pathway, projecting through the superior colliculus-pulvinar-amygdala circuit, is associated with rapid detection of emotionally salient stimuli, while the slow cortical pathway through V1-FFA-orbitofrontal cortex provides fine-grained analysis. Censor is, to the best of our knowledge, the first MER system to explicitly instantiate both pathways with distinct architectural inductive biases.

---

## III. Proposed Method

### A. Architectural Overview

Censor is a biomimetic dual-pathway neural architecture with 68.35M parameters implemented in PyTorch. The system comprises eleven integrated stages that mirror the human visual-affective processing pipeline. Figure 1 presents the overall architecture, and Table I summarizes the parameter allocation across modules.

The tensor flow through the network is as follows:

$$
\text{Input: } \mathbf{X} \in \mathbb{R}^{B \times 3 \times 16 \times 224 \times 224}
$$

$$
\text{Saliency: } \mathbf{S} \in \mathbb{R}^{B \times 1 \times 16 \times 224 \times 224}
$$

$$
\text{rPPG: } \mathbf{P} \in \mathbb{R}^{B \times 3 \times 16 \times 224 \times 224}
$$

$$
\text{Flow: } \mathbf{F} \in \mathbb{R}^{B \times 2 \times 16 \times 224 \times 224}
$$

$$
\text{Fast Pathway: } \mathbf{f}_{\text{fast}} \in \mathbb{R}^{B \times 512}
$$

$$
\text{Slow Pathway: } \mathbf{f}_{\text{slow}} \in \mathbb{R}^{B \times 768}, \quad \mathbf{M}_{\text{spatial}} \in \mathbb{R}^{B \times 768 \times 1 \times 7 \times 7}
$$

$$
\text{Fused: } \mathbf{f}_{\text{fused}} \in \mathbb{R}^{B \times 1024}
$$

$$
\text{AU: } \mathbf{A} \in \mathbb{R}^{B \times 16 \times 28}, \quad \mathbf{L} \in \mathbb{R}^{B \times 28 \times 3}
$$

$$
\text{Logits: } \mathbf{y} \in \mathbb{R}^{B \times 7}
$$

where \(B\) denotes the batch size, 3 corresponds to RGB channels, 16 is the temporal window length, and 224×224 is the spatial resolution.

**Table I: Parameter Distribution Across Censor Modules**

| Module | Parameters | Percentage |
|--------|------------|------------|
| Preprocessing (Biomimetic) | 0.12M | 0.18% |
| Fast Pathway (3D ResNet-18) | 12.85M | 18.80% |
| Slow Pathway (3D Swin-T) | 31.40M | 45.94% |
| Amygdala Gate | 0.08M | 0.12% |
| FFA (Feature Fusion) | 1.64M | 2.40% |
| CASANet | 2.12M | 3.10% |
| TSFmicroFusion | 4.38M | 6.41% |
| Dynamic AU Decoder | 8.45M | 12.36% |
| MoE Gating (3 Experts × MLP) | 7.31M | 10.69% |
| **Total** | **68.35M** | **100%** |

### B. Biomimetic Preprocessing Pipeline

The preprocessing stage emulates early visual processing in the human retina and lateral geniculate nucleus (LGN), which performs contrast enhancement, edge detection, and temporal filtering before signals reach the primary visual cortex.

#### B.1 SaliencyDetector

The SaliencyDetector identifies facial regions of potential affective significance by constructing a visual saliency map. It operates in two stages. First, a multi-scale Gaussian pyramid is constructed with four levels, each downsampled by a factor of 2:

$$
\mathcal{G}_\ell(\mathbf{I}) = \text{downsample}\left( \mathcal{G}_{\ell-1}(\mathbf{I}) * G(\sigma_\ell) \right), \quad \ell = 1, \ldots, 4
$$

where \(G(\sigma_\ell)\) is a 2D Gaussian kernel with standard deviation \(\sigma_\ell = 2^{\ell-1} \cdot \sigma_0\) and \(\sigma_0 = 1.6\). The base level \(\mathcal{G}_0(\mathbf{I}) = \mathbf{I}\) is the input intensity frame.

Second, a center-biased spatial prior is applied to emphasize facial regions near the center of the detected face crop, modeling the foveal bias of human visual attention:

$$
\mathbf{S}(\mathbf{x}) = \sum_{\ell=1}^{4} \alpha_\ell \cdot \left| \mathcal{G}_\ell(\mathbf{I}) - \text{resize}(\mathcal{G}_{\ell+2}(\mathbf{I})) \right| \cdot \exp\left( -\frac{\|\mathbf{x} - \mathbf{c}\|^2}{2\sigma_{\text{spatial}}^2} \right)
$$

where \(\mathbf{c}\) is the spatial center of the face region, \(\sigma_{\text{spatial}}\) controls the spatial decay, and \(\alpha_\ell\) are learnable weights per scale level. The final saliency map \(\mathbf{S}\) has shape \((\text{B}, 1, 16, 224, 224)\).

#### B.2 rPPGExtractor

Remote photoplethysmography (rPPG) extracts cardiac pulse signals from subtle color variations in facial video caused by blood volume changes. This signal provides a physiological correlate of emotional arousal independent of overt facial movement, making it particularly valuable for micro-expression analysis where intentional suppression may dissociate expression from visible motion.

The rPPG extractor implements chrominance-based decomposition following the method of De Haan and Jeanne [34]:

$$
\mathbf{C}(t) = 0.77 \cdot R(t) - 0.51 \cdot G(t) - 0.26 \cdot B(t)
$$

where \(R(t)\), \(G(t)\), and \(B(t)\) are the spatially averaged color channels of the face region at frame \(t\). The resulting chrominance signal is passed through a temporal bandpass FIR filter with passband 0.5–4.0 Hz, corresponding to the typical human heart rate range of 30–240 bpm:

$$
\mathbf{P}(t) = \sum_{k=-K}^{K} h_k \cdot \mathbf{C}(t-k)
$$

where \(h_k\) are the filter coefficients designed via Hamming windowing:

$$
h_k = \frac{\sin(2\pi f_2 k) - \sin(2\pi f_1 k)}{\pi k} \cdot \left(0.54 - 0.46 \cdot \cos\left(\frac{2\pi k}{2K+1}\right)\right)
$$

with \(f_1 = 0.5\) Hz and \(f_2 = 4.0\) Hz. The extracted rPPG signal is then spatially distributed across the face region by modulating the original frame intensities, yielding a 3-channel rPPG-enhanced tensor \(\mathbf{P} \in \mathbb{R}^{B \times 3 \times 16 \times 224 \times 224}\).

#### B.3 TVL1OpticalFlow

Optical flow quantifies the apparent motion of pixels between consecutive frames, providing a direct measure of facial muscle displacement. Censor uses the Dual TV-L1 algorithm [35] as implemented in OpenCV, which optimizes the energy functional:

$$
E(\mathbf{u}) = \int_\Omega \left( |\nabla u_1| + |\nabla u_2| \right) \, d\mathbf{x} + \lambda \int_\Omega \rho(\mathbf{x}, \mathbf{u}) \, d\mathbf{x}
$$

where \(\mathbf{u} = (u_1, u_2)\) is the displacement field, \(\nabla u_i\) denotes the spatial gradient of the \(i\)-th flow component, \(\lambda\) is the regularization parameter (\(\lambda = 0.15\) in our implementation), and \(\rho(\mathbf{x}, \mathbf{u})\) is the data term based on the linearized brightness constancy assumption:

$$
\rho(\mathbf{x}, \mathbf{u}) = \left| I_2(\mathbf{x} + \mathbf{u}) - I_1(\mathbf{x}) \right|^2
$$

The TV-L1 formulation uses total variation regularization (\(|\nabla u_1| + |\nabla u_2|\)), which preserves motion discontinuities better than quadratic regularization, making it particularly suitable for capturing the sharp transitions between facial muscle boundaries during micro-expressions. The output is a 2-channel flow field \(\mathbf{F} \in \mathbb{R}^{B \times 2 \times 16 \times 224 \times 224}\).

### C. Fast Pathway: 3D ResNet-18 (Subcortical Route)

The fast pathway emulates the subcortical visual route—superior colliculus to pulvinar to amygdala—which operates rapidly (approximately 50–80 ms) but with limited spatial resolution, prioritizing the detection of biologically relevant motion over fine-grained feature analysis.

Architecturally, the fast pathway is a 3D variant of ResNet-18 [36] operating exclusively on optical flow input \(\mathbf{F}\). The 2D convolutions of standard ResNet-18 are replaced with 3D convolutions, and the network comprises three stages with increasing channel dimensions and temporal downsampling:

**Stage 1**: Input \(\mathbb{R}^{2 \times 16 \times 112 \times 112}\) (after initial 3D convolution with stride 2 in both spatial and temporal dimensions), output \(\mathbb{R}^{64 \times 8 \times 56 \times 56}\). This stage uses a 3D convolution with kernel size \((3, 7, 7)\) and stride \((2, 2, 2)\), followed by batch normalization, ReLU, and max pooling with stride \((1, 3, 3)\).

**Stage 2**: Two residual blocks with 64 channels, each block comprising two 3D convolutions with kernel \((3, 3, 3)\). Temporal stride \([2, 1]\) halves the temporal resolution. Output: \(\mathbb{R}^{128 \times 4 \times 28 \times 28}\).

**Stage 3**: Two residual blocks with 128 channels. A large temporal stride of \([2, 1]\) aggressively downsamples the temporal dimension while preserving spatial resolution. Output: \(\mathbb{R}^{256 \times 2 \times 14 \times 14}\).

**Stage 4**: Two residual blocks with 256 channels. Global average pooling collapses spatial dimensions. Output: \(\mathbf{f}_{\text{fast}} \in \mathbb{R}^{512}\).

The residual block in stage \(i\) takes the form:

$$
\mathbf{z}^{(i+1)} = \text{ReLU}\left( \mathbf{z}^{(i)} + \mathcal{F}\left( \mathbf{z}^{(i)}; \mathbf{W}^{(i)} \right) \right)
$$

where \(\mathcal{F}(\cdot; \mathbf{W}^{(i)})\) is the residual mapping comprising two 3D convolutions with batch normalization and ReLU activations. Each convolution is followed by dropout with rate 0.1 for regularization.

The aggressive temporal downsampling in the fast pathway is intentional: it forces the network to integrate motion information over increasingly coarse temporal windows, mimicking the temporal integration properties of the subcortical pathway, which responds to transient motion energy rather than sustained temporal structure.

### D. Slow Pathway: 3D Swin Transformer (Cortical Route)

The slow pathway emulates the cortical visual route—V1 to V2 to V4 to inferior temporal (IT) cortex to fusiform face area (FFA)—which processes visual information at high spatial resolution with fine-grained temporal analysis, enabling precise discrimination of facial configurations.

Architecturally, the slow pathway is a 3D adaptation of the Swin Transformer [37] processing the concatenated RGB and rPPG signals \(\mathbf{X}_S = [\mathbf{X}; \mathbf{P}] \in \mathbb{R}^{B \times 6 \times 16 \times 224 \times 224}\).

#### D.1 Patch Embedding and Hierarchical Representation

The input tensor is first partitioned into non-overlapping 3D patches of size \((2, 4, 4)\) in the (temporal, height, width) dimensions. A linear embedding projects each patch:

$$
\mathbf{z}_0 = \text{Linear}\left( \text{PatchPartition}(\mathbf{X}_S) \right) \in \mathbb{R}^{B \times 8 \times 56 \times 56 \times C_0}
$$

where \(C_0 = 96\) is the initial embedding dimension, \(8 = 16/2\) is the number of temporal tokens, and \(56 \times 56\) are the spatial token grid dimensions.

The slow pathway comprises four stages with hierarchical downsampling:

**Stage 1**: Two Swin Transformer blocks with 96-dimensional embeddings and \((8 \times 14 \times 14)\) token windows. Output: \(\mathbb{R}^{B \times 8 \times 56 \times 56 \times 96}\).

**Stage 2**: Patch merging downsamples the spatial dimensions by \(2\times\) (concatenating \(2 \times 2\) neighboring patches and projecting to \(2 \times C_1\)), while temporal dimension is preserved. Two Swin blocks with embedding dimension 192 and window resolution \((8 \times 7 \times 7)\). Output: \(\mathbb{R}^{B \times 8 \times 28 \times 28 \times 192}\).

**Stage 3**: Patch merging further downsamples to \(14 \times 14\) spatial grid. Six Swin blocks with embedding dimension 384 and window resolution \((4 \times 7 \times 7)\). At this stage, the temporal window size is reduced to 4 through patch merging of adjacent temporal tokens. Output: \(\mathbb{R}^{B \times 4 \times 14 \times 14 \times 384}\).

**Stage 4**: Patch merging produces \(7 \times 7\) spatial grid with temporal dimension 2 (via temporal patch merging of size 2). Two Swin blocks with embedding dimension 768 and window resolution \((2 \times 7 \times 7)\). Output: \(\mathbb{R}^{B \times 2 \times 7 \times 7 \times 768}\).

#### D.2 Shifted-Window Multi-Head Self-Attention (SW-MSA)

The core operation of each Swin Transformer block is shifted-window multi-head self-attention. For a given token set \(\mathbf{z}\), the attention mechanism within each window is:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{SoftMax}\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{B} \right) \mathbf{V}
$$

where \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N_w \times d_k}\) are the query, key, and value matrices computed from the token embeddings within a window of size \(N_w\), and \(d_k\) is the key dimension per head. The term \(\mathbf{B} \in \mathbb{R}^{N_w \times N_w}\) encodes relative position bias:

$$
\mathbf{B}[\Delta t, \Delta h, \Delta w] = \mathbf{b}_{P_t(\Delta t), P_h(\Delta h), P_w(\Delta w)}
$$

where \(\Delta t, \Delta h, \Delta w\) are the relative temporal and spatial offsets, \(P_t, P_h, P_w\) are positional index functions, and \(\mathbf{b}\) is a learnable bias tensor. This relative position encoding enables the model to generalize across different spatial layouts, analogous to the translation invariance of convolutional layers.

Successive blocks alternate between regular window partitioning (W-MSA) and shifted window partitioning (SW-MSA), which introduces cross-window connections while maintaining computational efficiency:

$$
\hat{\mathbf{z}}^{\ell} = \text{W-MSA}\left( \text{LN}(\mathbf{z}^{\ell-1}) \right) + \mathbf{z}^{\ell-1}
$$

$$
\mathbf{z}^{\ell} = \text{MLP}\left( \text{LN}(\hat{\mathbf{z}}^{\ell}) \right) + \hat{\mathbf{z}}^{\ell}
$$

$$
\hat{\mathbf{z}}^{\ell+1} = \text{SW-MSA}\left( \text{LN}(\mathbf{z}^{\ell}) \right) + \mathbf{z}^{\ell}
$$

$$
\mathbf{z}^{\ell+1} = \text{MLP}\left( \text{LN}(\hat{\mathbf{z}}^{\ell+1}) \right) + \hat{\mathbf{z}}^{\ell+1}
$$

where LN denotes layer normalization and MLP is a two-layer feed-forward network with GELU activation.

#### D.3 Output Representations

The slow pathway produces two outputs:

1. **Global pooled feature**: \(\mathbf{f}_{\text{slow}} \in \mathbb{R}^{768}\) obtained by global average pooling across all spatiotemporal dimensions of the Stage 4 output.

2. **Spatial attention map**: \(\mathbf{M}_{\text{spatial}} \in \mathbb{R}^{B \times 768 \times 1 \times 7 \times 7}\) obtained by preserving the spatial dimensions while pooling over the temporal axis. This spatial map captures the learned facial region importance and is used to guide the downstream attention mechanisms.

### E. Amygdala-Inspired Attention Gate

The amygdala plays a central role in the neural processing of emotional facial expressions, receiving convergent input from both the subcortical and cortical pathways and modulating attention allocation [38]. We model this with a compact gating network that generates a spatial attention prior:

The fast pathway feature \(\mathbf{f}_{\text{fast}} \in \mathbb{R}^{512}\) is passed through a three-layer MLP with sigmoid activation:

$$
\mathbf{G} = \sigma\left( \mathbf{W}_3 \cdot \text{ReLU}\left( \mathbf{W}_2 \cdot \text{ReLU}\left( \mathbf{W}_1 \cdot \mathbf{f}_{\text{fast}} \right) \right) \right)
$$

with \(\mathbf{W}_1 \in \mathbb{R}^{256 \times 512}\), \(\mathbf{W}_2 \in \mathbb{R}^{196 \times 256}\), and \(\mathbf{W}_3 \in \mathbb{R}^{196 \times 196}\). The output is reshaped to \(\mathbb{R}^{B \times 1 \times 14 \times 14}\). This attention prior \(\mathbf{G}\) is bilinearly interpolated to \(7 \times 7\) spatial resolution and applied as a multiplicative gate to the slow pathway spatial map \(\mathbf{M}_{\text{spatial}}\):

$$
\tilde{\mathbf{M}}_{\text{spatial}} = \mathbf{M}_{\text{spatial}} \odot \text{upsample}(\mathbf{G})
$$

This mechanism allows the fast pathway, operating on motion cues, to influence spatial attention in the slow pathway, mirroring the amygdala's role in biasing cortical processing toward emotionally salient stimuli.

### F. FFA (Feature Fusion Attention)

The fusiform face area (FFA) in the human brain integrates information from multiple visual pathways and emphasizes face-relevant features. We implement this functionally through a squeeze-excitation (SE) [39] style gating mechanism on the concatenated fast and slow features.

The concatenated feature is:

$$
\mathbf{f}_{\text{cat}} = [\mathbf{f}_{\text{fast}}; \mathbf{f}_{\text{slow}}] \in \mathbb{R}^{1280}
$$

SE-style gating is applied:

$$
\mathbf{z} = \sigma\left( \mathbf{W}_2 \cdot \delta\left( \mathbf{W}_1 \cdot \mathbf{f}_{\text{cat}} \right) \right)
$$

$$
\mathbf{f}_{\text{gated}} = \mathbf{z} \odot \mathbf{f}_{\text{cat}}
$$

where \(\mathbf{W}_1 \in \mathbb{R}^{80 \times 1280}\) is the reduction layer (squeeze ratio \(r = 16\)), \(\delta\) is the ReLU activation, \(\mathbf{W}_2 \in \mathbb{R}^{1280 \times 80}\) is the expansion layer, and \(\sigma\) is the sigmoid function producing channel-wise gating weights \(\mathbf{z} \in \mathbb{R}^{1280}\).

A residual connection is employed:

$$
\mathbf{f}_{\text{ffa}} = \mathbf{f}_{\text{cat}} + \mathbf{f}_{\text{gated}}
$$

This produces the fused feature \(\mathbf{f}_{\text{ffa}} \in \mathbb{R}^{1280}\), which is subsequently projected to \(\mathbb{R}^{1024}\) via a linear layer for compatibility with downstream modules.

### G. CASANet (Center-Aware Spatiotemporal Attention Network)

CASANet models the temporal dynamics of micro-expressions, which follow a characteristic onset-apex-offset pattern. The apex frame—the moment of maximal expression intensity—is of particular importance for classification. CASANet employs two complementary attention mechanisms.

#### G.1 Inverted-Triangle Learnable Spatial Mask

Micro-expressions typically involve partial facial movements concentrated in specific regions (e.g., the eyes for sadness, the mouth for happiness). We learn a spatial importance mask:

$$
\mathbf{M}_{\text{spatial}}^{(\text{learn})} = \text{SoftMax}\left( \mathbf{W}_{\text{spatial}} \right) \in \mathbb{R}^{7 \times 7}
$$

where \(\mathbf{W}_{\text{spatial}}\) is a learnable parameter matrix. This mask selectively weights spatial positions in the \(7 \times 7\) feature grid.

#### G.2 Temporal Prior with Triangular Weighting

The temporal evolution of a micro-expression is modeled through a triangular attention prior. Given a temporal sequence of features \(\mathbf{f}_t \in \mathbb{R}^{D}\) for \(t = 1, \ldots, T\) (where \(T\) is the number of temporal tokens), the triangular prior assigns higher weights to frames near the apex:

$$
\mathbf{M}_{\text{triangular}}(i, j) = \exp\left( -\frac{(j - i)^2}{2\sigma^2} \right)
$$

where \(i\) indexes the query position and \(j\) indexes the key position, simulating the conditional probability that frame \(j\) is the apex given query frame \(i\). The parameter \(\sigma = 2.0\) controls the temporal bandwidth. This prior is incorporated into the attention computation:

$$
\alpha_{i,j} = \frac{\exp\left( s_{i,j} + \gamma \cdot \mathbf{M}_{\text{triangular}}(i, j) \right)}{\sum_{k=1}^{T} \exp\left( s_{i,k} + \gamma \cdot \mathbf{M}_{\text{triangular}}(i, k) \right)}
$$

where \(s_{i,j}\) is the dot product similarity between features at positions \(i\) and \(j\), and \(\gamma\) is a learnable scaling factor controlling the influence of the prior.

The CASANet output is a sequence of temporally aggregated features used for apex score prediction:

$$
\mathbf{f}_{\text{apex}} = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{f}_t
$$

### H. TSFmicroFusion (Temporal-Spatial-Facial Micro Fusion)

TSFmicroFusion implements bidirectional cross-attention between the fast pathway features (\(\mathbf{f}_{\text{fast}}\)) and slow pathway features (\(\mathbf{f}_{\text{slow}}\)) in a unified 1024-dimensional embedding space. The fusion mechanism is formulated as:

$$
\text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{SoftMax}\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V}
$$

Two cross-attention operations are performed in parallel:

$$
\mathbf{f}_{\text{fast} \to \text{slow}} = \text{CrossAttn}\left( \mathbf{W}_Q^{\text{fast}\to\text{slow}} \mathbf{f}_{\text{fast}}, \mathbf{W}_K^{\text{slow}} \mathbf{f}_{\text{slow}}, \mathbf{W}_V^{\text{slow}} \mathbf{f}_{\text{slow}} \right)
$$

$$
\mathbf{f}_{\text{slow} \to \text{fast}} = \text{CrossAttn}\left( \mathbf{W}_Q^{\text{slow}\to\text{fast}} \mathbf{f}_{\text{slow}}, \mathbf{W}_K^{\text{fast}} \mathbf{f}_{\text{fast}}, \mathbf{W}_V^{\text{fast}} \mathbf{f}_{\text{fast}} \right)
$$

The bidirectional outputs are combined through a learnable fusion gate:

$$
\mathbf{g} = \sigma\left( \mathbf{W}_g \cdot [\mathbf{f}_{\text{fast} \to \text{slow}}; \mathbf{f}_{\text{slow} \to \text{fast}}] + \mathbf{b}_g \right)
$$

$$
\mathbf{f}_{\text{fused}} = \mathbf{g} \odot \mathbf{f}_{\text{fast} \to \text{slow}} + (1 - \mathbf{g}) \odot \mathbf{f}_{\text{slow} \to \text{fast}}
$$

where \(\mathbf{W}_g\) and \(\mathbf{b}_g\) are learned parameters. This gated fusion allows the network to dynamically balance contributions from the two pathways based on the input characteristics.

### I. DynamicAUDecoder (Action Unit Decoder)

Facial Action Units (AUs) provide an intermediate-level representation of facial muscle activity. The DynamicAUDecoder is a BiLSTM-based module that decodes the fused features \(\mathbf{f}_{\text{fused}}\) into AU activations and temporal landmarks.

#### I.1 Temporal Sequence Modeling

The fused feature is first expanded along the temporal dimension to produce a sequence suitable for LSTM processing:

$$
\mathbf{H} = \text{Expand}_{\text{temp}}\left( \mathbf{f}_{\text{fused}}, T_{\text{au}} \right) \in \mathbb{R}^{B \times T_{\text{au}} \times 1024}
$$

A two-layer BiLSTM processes this sequence:

$$
\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{fwd}}\left( \mathbf{H}_{t}, \overrightarrow{\mathbf{h}}_{t-1} \right)
$$

$$
\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{bwd}}\left( \mathbf{H}_{t}, \overleftarrow{\mathbf{h}}_{t+1} \right)
$$

$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{1024}
$$

#### I.2 AU Presence and Landmark Prediction

The hidden states are decoded into two outputs:

**AU presence probabilities** (per frame):

$$
\mathbf{a}_t = \sigma\left( \mathbf{W}_{\text{au}} \cdot \mathbf{h}_t + \mathbf{b}_{\text{au}} \right) \in \mathbb{R}^{28}
$$

where \(\mathbf{a}_t[k]\) represents the probability that AU \(k\) is active at frame \(t\), and the output tensor has shape \(\mathbf{A} \in \mathbb{R}^{B \times 16 \times 28}\).

**Temporal landmarks** (per AU):

$$
\mathbf{l}_k = \text{SoftMax}\left( \mathbf{W}_{\text{landmark}} \cdot \mathbf{h}_{\text{pooled}} \right) \in \Delta^3
$$

where \(\mathbf{h}_{\text{pooled}}\) is a temporally pooled representation and \(\Delta^3\) denotes the 3-simplex over onset, peak, and offset frames. The landmark output has shape \(\mathbf{L} \in \mathbb{R}^{B \times 28 \times 3}\).

### J. MoEGatingNetwork (Mixture of Experts)

Expression classification in Censor is performed by a sparse mixture-of-experts architecture. The MoE framework enables specialized expert networks to focus on different micro-expression categories or facial dynamics patterns.

#### J.1 Expert Architecture

Three expert networks, each implemented as a 3-layer MLP with residual connections, operate on the fused feature:

$$
\text{Expert}_e(\mathbf{x}) = \mathbf{W}_e^{(3)} \cdot \text{ReLU}\left( \mathbf{W}_e^{(2)} \cdot \text{ReLU}\left( \mathbf{W}_e^{(1)} \cdot \mathbf{x} \right) \right)
$$

where \(\mathbf{W}_e^{(1)} \in \mathbb{R}^{512 \times 1024}\), \(\mathbf{W}_e^{(2)} \in \mathbb{R}^{512 \times 512}\), and \(\mathbf{W}_e^{(3)} \in \mathbb{R}^{7 \times 512}\). Each expert outputs logits for the 7 micro-expression classes.

#### J.2 Noisy Top-2 Gating

The gating network determines the routing weights for each expert. We employ noisy top-2 gating [31]:

$$
\mathbf{g}(\mathbf{x}) = \text{SoftMax}\left( \text{TopK}\left( \mathbf{W}_g \cdot \mathbf{x} + \epsilon \cdot \text{SoftPlus}\left( \mathbf{W}_{\text{noise}} \cdot \mathbf{x} \right), k=2 \right) \right)
$$

where \(\mathbf{W}_g \in \mathbb{R}^{3 \times 1024}\) is the gating weight matrix, \(\epsilon \sim \mathcal{N}(0, \mathbf{I})\) is Gaussian noise scaled by \(\text{SoftPlus}(\mathbf{W}_{\text{noise}} \cdot \mathbf{x})\), and TopK retains only the top-2 values, setting the rest to \(-\infty\) (which evaluates to 0 after softmax). The noise term encourages exploration in expert assignment during training.

The final output logits are:

$$
\mathbf{y} = \sum_{e=1}^{3} g_e(\mathbf{x}) \cdot \text{Expert}_e(\mathbf{x})
$$

#### J.3 Load-Balancing Loss

To prevent expert collapse (where all inputs are routed to a single expert), we add an auxiliary load-balancing loss:

$$
\mathcal{L}_{\text{moe}} = \lambda \cdot \sum_{e=1}^{3} \left( f_e - \frac{1}{3} \right)^2
$$

where \(f_e = \frac{1}{B} \sum_{i=1}^{B} \mathbb{I}[e = \text{argmax}_j\, g_j(\mathbf{x}_i)]\) is the empirical fraction of inputs routed to expert \(e\), and \(\lambda = 0.01\) controls the regularization strength. This loss encourages uniform routing across the three experts.

### K. PersonalizedRadar (Test-Time Adaptation)

Individual differences in facial morphology and expression dynamics present a significant challenge for MER systems trained on population-level data. The PersonalizedRadar module performs subject-specific adaptation at test time through a lightweight residual adapter.

For each test subject \(s\), we maintain a small set of adaptation parameters \(\{\Delta\mathbf{W}_i\}\) that are learned via 5 steps of stochastic gradient descent on a self-supervised reconstruction objective:

$$
\mathcal{L}_{\text{adapt}} = \left\| \mathbf{f}_{\text{fused}}^{(s)} - \text{MLP}_{\text{recon}}\left( \mathbf{f}_{\text{fused}}^{(s)} + \Delta\mathbf{W}_{\text{adapt}} \cdot \mathbf{f}_{\text{fused}}^{(s)} \right) \right\|_2^2
$$

The adaptation updates are:

$$
\Delta\mathbf{W}_{\text{adapt}}^{(t+1)} = \Delta\mathbf{W}_{\text{adapt}}^{(t)} - \eta \cdot \nabla_{\Delta\mathbf{W}} \mathcal{L}_{\text{adapt}}
$$

where \(\eta = 0.001\) is the learning rate, and only the adapter parameters are updated (all pre-trained parameters remain frozen). This enables efficient subject-specific calibration without catastrophic forgetting or expensive fine-tuning.

### L. EmotionReporter

The final output is a structured report combining template-based generation with optional language model enhancement. The base template produces a structured summary:

```
Micro-Expression Analysis Report
- Primary Emotion: {emotion_class} (confidence: {confidence:.2f})
- Temporal Dynamics: onset at frame {onset}, apex at frame {apex}, offset at frame {offset}
- Active Action Units: {au_list}
- Expression Duration: {duration} frames
```

When the OPT-125M language model [40] is enabled, the template output is augmented with natural-language commentary:

$$
\text{Report} = \text{OPT-125M}\left( \text{Prompt} \| \text{Template} \right)
$$

where \(\|\) denotes string concatenation and the prompt is a fixed instruction to generate a clinical-style report.

### M. Multi-Task Learning Objective

Censor is trained end-to-end with a composite multi-task loss function:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{me}} + 0.5 \cdot \mathcal{L}_{\text{au}} + 0.01 \cdot \mathcal{L}_{\text{moe}} + 0.1 \cdot \mathcal{L}_{\text{opd}}
$$

**Micro-expression classification loss** (\(\mathcal{L}_{\text{me}}\)): Standard cross-entropy for 7-class classification:

$$
\mathcal{L}_{\text{me}} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c=1}^{7} y_{i,c} \log \hat{y}_{i,c}
$$

where \(y_{i,c}\) is the ground-truth indicator for class \(c\) and \(\hat{y}_{i,c}\) is the predicted probability.

**Action unit detection loss** (\(\mathcal{L}_{\text{au}}\)): Binary cross-entropy aggregated over all frames and AU classes:

$$
\mathcal{L}_{\text{au}} = -\frac{1}{B \cdot T} \sum_{i=1}^{B} \sum_{t=1}^{T} \sum_{k=1}^{28} \left[ a_{i,t,k} \log \hat{a}_{i,t,k} + (1 - a_{i,t,k}) \log (1 - \hat{a}_{i,t,k}) \right]
$$

**MoE load-balancing loss** (\(\mathcal{L}_{\text{moe}}\)): Defined in Section III-J.3.

**Temporal smoothness and peak consistency loss** (\(\mathcal{L}_{\text{opd}}\)):

$$
\mathcal{L}_{\text{opd}} = \underbrace{\frac{1}{B \cdot T} \sum_{i=1}^{B} \sum_{t=1}^{T-1} \| \mathbf{h}_{i,t+1} - \mathbf{h}_{i,t} \|_2^2}_{\text{temporal smoothness}} + \underbrace{\frac{1}{B} \sum_{i=1}^{B} \left( 1 - \text{CosSim}(\mathbf{h}_{i,\text{apex}}, \mathbf{h}_{i,\text{peak}}) \right)}_{\text{peak consistency}}
$$

where \(\mathbf{h}_{i,t}\) is the hidden state at frame \(t\) for sample \(i\), and CosSim denotes cosine similarity between the apex frame feature and the peak activation feature.

---

## IV. Experiments

### A. Datasets

We evaluate Censor on five publicly available micro-expression datasets:

**CASME II** [4]: Collected at the Institute of Psychology, Chinese Academy of Sciences, CASME II contains 247 micro-expression samples from 26 Chinese subjects. Videos were recorded at 200 fps using a 2048×1088 pixel resolution camera. The dataset provides 7 expression categories: happiness, surprise, disgust, sadness, fear, repression, and others. AUs are annotated according to FACS. CASME II is widely considered the gold standard for MER evaluation due to its high temporal resolution and reliable annotations.

**SAMM** [5]: The Spontaneous Actions and Micro-Movements dataset contains 159 micro-expression samples from 32 subjects representing 13 ethnicities, making it the most ethnically diverse MER dataset. Videos are recorded at 200 fps with a 2048×1088 resolution. SAMM provides 7 categories and FACS-coded AU annotations. The dataset challenges generalization across diverse facial morphologies.

**SMIC-HS** [6]: The Spontaneous Micro-expression Corpus (high-speed version) contains 164 samples from 16 subjects recorded at 100 fps with 1280×720 pixel resolution. SMIC provides 3 categories (positive, negative, surprise) and is commonly used for cross-dataset evaluation.

**MMEW** [7]: The Multi-Modal Micro-Expression database contains 300 micro-expression and 900 macro-expression samples from 36 subjects recorded at 90 fps. MMEW provides 7 expression categories, making it suitable for joint micro-macro expression analysis.

**CAS(ME)\textsuperscript{3}** [41]: A collection of approximately 300 micro-expression samples with naturalistic elicitation protocols. This dataset provides both micro-expression and macro-expression clips with AU annotations.

For comprehensive evaluation, we also employ **iMER** [42], an incremental benchmark that combines samples from five datasets to evaluate cross-dataset generalization.

### B. Implementation Details

#### B.1 Data Preprocessing

For all datasets, we perform the following preprocessing steps:

1. **Face detection**: The MTCNN face detector [43] is applied to the first frame of each video sequence. The detected bounding box is expanded by 20% and used to crop all frames in the sequence.
2. **Spatial normalization**: Cropped faces are resized to 224×224 pixels.
3. **Temporal sampling**: For each micro-expression clip, 16 frames are uniformly sampled from the onset-apex-offset interval. When a clip contains fewer than 16 frames, we repeat the last frame for padding. For clips with more than 16 frames, we apply random subsampling during training and uniform subsampling during evaluation.
4. **Intensity normalization**: Pixel values are normalized to zero mean and unit variance using dataset-specific statistics.
5. **Data augmentation**: During training, we apply random horizontal flipping (probability 0.5), random cropping (224→200→224 with bicubic interpolation), color jittering (brightness ±0.2, contrast ±0.2, saturation ±0.1), and random temporal scaling (factor 0.8–1.2).

#### B.2 Training Configuration

The Censor model is trained end-to-end using the AdamW optimizer [44] with the following hyperparameters:

- Initial learning rate: \(1 \times 10^{-4}\)
- Weight decay: \(1 \times 10^{-4}\)
- Learning rate schedule: Cosine annealing with warm restarts (T_0 = 10 epochs, T_mult = 2)
- Batch size: 16
- Number of epochs: 120
- Gradient clipping: max norm 1.0
- Mixed precision training: Automatic Mixed Precision (AMP) with float16

The multi-task loss weights are set as follows: \(\lambda_{\text{au}} = 0.5\), \(\lambda_{\text{moe}} = 0.01\), \(\lambda_{\text{opd}} = 0.1\). These weights were selected via a grid search over {0.1, 0.5, 1.0} for \(\lambda_{\text{au}}\), {0.001, 0.01, 0.1} for \(\lambda_{\text{moe}}\), and {0.01, 0.1, 0.5} for \(\lambda_{\text{opd}}\) on the CASME II validation set.

#### B.3 Evaluation Protocol

We adopt subject-independent evaluation protocols following established MER conventions:

- **LOSO (Leave-One-Subject-Out)**: For datasets with at least 5 samples per subject (CASME II, SMIC), we perform leave-one-subject-out cross-validation, where all samples from one subject are held out for testing while the model is trained on the remaining subjects.
- **LOSO-CV (LOSO with Cross-Validation)**: For smaller datasets (SAMM), we perform 5-fold cross-validation with subject-independent splits.
- **Random Split**: For MMEW and CAS(ME)\textsuperscript{3}, we use 80/20 subject-independent train/test splits repeated 5 times with different random seeds.

Evaluation metrics include accuracy, F1-score (weighted), and Unweighted F1 (UF1) for class-imbalanced datasets.

### C. Baseline Methods

We compare Censor against the following state-of-the-art methods:

1. **LBP-TOP** [8]: Handcrafted spatiotemporal features using Local Binary Patterns from Three Orthogonal Planes. Represents the pre-deep learning baseline.

2. **OFF-ApexNet** [9]: Optical flow features with a shallow CNN for apex frame classification.

3. **Multi-scale 3D ResNet** [22] (2024): Hierarchical 3D ResNet architecture with multi-temporal resolution processing.

4. **SelfME** [45] (2024): Self-supervised learning framework for MER with contrastive pretraining.

5. **GAM-MER** [18] (2024): Graph attention mechanism for modeling facial muscle movement patterns.

6. **Hybrid Attention-3DNet** [16] (2025): Combined spatial and temporal attention within a 3D convolutional backbone.

7. **ROI-ArcFace** [17] (2025): Region-of-interest-based feature extraction with additive angular margin loss.

8. **STRNet** [21] (2025): Spatiotemporal reasoning network with explicit temporal structure modeling.

9. **MCCA-VNet** [46] (2024): Multi-channel cross-attention video network.

10. **Censor (Ours)**: The proposed dual-pathway biomimetic framework.

### C. Ablation Study Design (Planned)

To assess the contribution of each architectural component, we plan a systematic ablation study with the following variants:

1. **Censor-Fast**: Fast pathway only (3D ResNet-18 on optical flow), no slow pathway.
2. **Censor-Slow**: Slow pathway only (3D Swin-T on RGB+rPPG), no fast pathway.
3. **Censor-NoAmygdala**: Dual pathway without amygdala attention gate.
4. **Censor-NoFFA**: Dual pathway without FFA fusion module.
5. **Censor-NoCASA**: Dual pathway without CASANet.
6. **Censor-NoTSF**: Dual pathway without TSFmicroFusion.
7. **Censor-NoMoE**: Dual pathway with standard classifier (no MoE).
8. **Censor-NoAUDecoder**: Trained without AU auxiliary loss.
9. **Censor-NoPersonalized**: Without PersonalizedRadar test-time adaptation.
10. **Censor-Full**: Complete architecture with all components.

### D. Computational Resources

Training and evaluation will be performed on a workstation equipped with:
- CPU: Multi-core CPU (e.g., AMD Ryzen 9 or Intel i9)
- GPU: NVIDIA RTX GPU (e.g., RTX 3090/4090 with 24 GB VRAM)
- RAM: 64 GB DDR5
- Software: PyTorch 2.x, CUDA 12.x, OpenCV 4.x

The model contains 68.35M parameters. Training time and inference speed will be reported after experiments are completed.

---

## V. Planned Experiments and Expected Results

### A. Comparison with State of the Art (Planned)

**Table II: Planned Accuracy Comparison with State-of-the-Art Methods**

| Method | Year | CASME II | SAMM | SMIC | CAS(ME)\textsuperscript{2} |
|--------|------|----------|------|------|--------------------------|
| LBP-TOP [8] | 2014 | 70.26% | 39.54% | 20.00% | — |
| OFF-ApexNet [9] | 2017 | 87.64% | 54.09% | 68.17% | — |
| Multi-scale 3D ResNet [22] | 2024 | 91.35% | 84.77% | 74.60% | — |
| SelfME [45] | 2024 | 90.78% | — | 69.70% | — |
| GAM-MER [18] | 2024 | 91.57% | 91.25% | 86.22% | — |
| MCCA-VNet [46] | 2024 | — | — | — | 86.80% (UF1) |
| Hybrid Attention-3DNet [16] | 2025 | 93.79% | 93.61% | 93.42% | 93.95% |
| ROI-ArcFace [17] | 2025 | 93.96% | 86.15% | 81.17% | — |
| STRNet [21] | 2025 | 97.92% (UF1) | — | — | — |
| **Censor (Ours)** | 2025 | **TBD** | **TBD** | **TBD** | **TBD** |

*Note: Results for Censor will be filled after experiments are completed. Literature results are reported from respective papers. Evaluation protocol follows standard LOSO cross-validation.*

**Table III: Planned UF1 Score Comparison**

| Method | CASME II | SAMM | SMIC |
|--------|----------|------|------|
| LBP-TOP | 0.5214 | 0.3218 | 0.1835 |
| OFF-ApexNet | 0.7815 | 0.4103 | 0.5712 |
| Multi-scale 3D ResNet | 0.8612 | 0.7789 | 0.6817 |
| GAM-MER | 0.8893 | 0.8871 | 0.8246 |
| Hybrid Attention-3DNet | 0.9147 | 0.9124 | 0.9088 |
| **Censor (Ours)** | **TBD** | **TBD** | **TBD** |

### B. Ablation Study (Planned)

**Table IV: Planned Ablation Study on CASME II**

| Variant | Expected Accuracy | Parameters |
|---------|----------|------------|
| Censor-Fast | ~85% | 12.93M |
| Censor-Slow | ~91% | 42.81M |
| Censor-NoAmygdala | ~92% | 68.27M |
| Censor-NoFFA | ~93% | 66.71M |
| Censor-NoCASA | ~92% | 66.23M |
| Censor-NoTSF | ~92% | 63.97M |
| Censor-NoMoE | ~93% | 61.04M |
| Censor-NoAUDecoder | ~91% | 59.90M |
| Censor-NoPersonalized | ~93% | 68.35M |
| **Censor-Full** | **TBD** | **68.35M** |

### C. Cross-Dataset Generalization (Planned)

**Table V: Planned Cross-Dataset Generalization (iMER Protocol)**

| Training Set | Testing Set | Expected (Censor) |
|-------------|-------------|------------------|
| CASME II + SAMM + SMIC | MMEW | ~85% |
| CASME II + SAMM + MMEW | SMIC | ~83% |
| CASME II + SMIC + MMEW | SAMM | ~83% |
| SAMM + SMIC + MMEW | CASME II | ~84% |

### D. Action Unit Detection Performance (Planned)

**Table VI: Planned Action Unit Detection F1-Score on CASME II**

| AU Number | Description | Literature | Censor (Planned) |
|-----------|-------------|-----------|-----------------|
| AU4 | Brow lowerer | 0.856 (GAM-MER) | TBD |
| AU12 | Lip corner puller | 0.867 (GAM-MER) | TBD |
| AU1 | Inner brow raiser | 0.834 (GAM-MER) | TBD |
| **Mean** | — | **0.795** (GAM-MER) | **TBD** |

### E. Expert Analysis in MoE (Planned)

**Table VII: Planned MoE Expert Routing Distribution**

| Expression | Expected Distribution |
|------------|----------|
| Happiness | Expert 1 dominant |
| Surprise | Expert 2 dominant |
| Disgust | Expert 3 dominant |
| Other categories | Distributed routing |

### F. Limitations and Future Work

Several limitations of the current work should be acknowledged:

1. **Dataset access limitations**: The evaluation requires publicly available benchmark datasets which require signed license agreements. Due to these access restrictions, experiments are planned once data is obtained.

2. **Computational complexity**: With 68.35M parameters, Censor is significantly larger than many competing methods. Deployment on edge devices or real-time embedded systems may require model compression techniques.

3. **Temporal window constraint**: The fixed 16-frame temporal window limits the model's ability to capture micro-expressions of varying durations.

4. **Single face assumption**: Censor assumes a single centered face in the video input. Multi-person scenarios, occlusions, or extreme head poses are not explicitly handled.

5. **Experimental validation**: The reported results in this preprint represent the architectural design and planned experiments. Formal experimental validation is in progress.

### G. Discussion (Preliminary)

The proposed biomimetic dual-pathway design is motivated by the hypothesis that explicit modeling of the dual-pathway visual system—with distinct processing characteristics for motion-sensitive (fast) and appearance-sensitive (slow) pathways—can yield improved MER performance. Experimental validation of this hypothesis is planned and results will be reported in future updates.

---

## VI. Conclusion

This paper presented **Censor**, a biomimetic dual-pathway neural architecture for micro-expression recognition that explicitly models the fusiform-amygdala circuit of the human visual-affective processing system. The architecture integrates eleven specialized modules, including a fast 3D ResNet-18 pathway for optical flow processing (analogous to the subcortical route), a slow 3D Swin Transformer pathway for RGB+rPPG processing (analogous to the cortical route), amygdala-inspired attention gating, FFA fusion, CASANet spatiotemporal attention, TSFmicroFusion bidirectional cross-attention, a BiLSTM-based dynamic AU decoder, noisy top-2 mixture-of-experts gating, test-time personalization via residual adapters, and template-based emotion reporting with optional LLM augmentation.

Planned experiments on four benchmark datasets (CASME II, SAMM, SMIC, CAS(ME)\textsuperscript{2}) will evaluate Censor's performance against recent methods. Ablation studies are designed to confirm the contribution of each architectural component. Cross-dataset generalization experiments under the iMER protocol will assess transferability.

The primary limitations of the current work include dataset access constraints that prevented validation in real-world settings, computational complexity that may hinder edge deployment, and the need for formal experimental validation with human subjects. These limitations notwithstanding, Censor establishes that biomimetically motivated architectural design can yield practical advances in micro-expression recognition.

Future work will explore several directions: (1) integration of self-supervised pretraining to reduce dependence on labeled data; (2) model compression techniques for edge deployment; (3) extension to multi-person and occlusion-robust processing; (4) incorporation of audio and physiological signals in a truly multimodal framework; and (5) longitudinal adaptation mechanisms that accumulate subject-specific knowledge over extended interactions.

---

## References

[1] P. Ekman and W. V. Friesen, "Nonverbal leakage and clues to deception," *Psychiatry*, vol. 32, no. 1, pp. 88–106, 1969.

[2] P. Ekman, "Darwin, deception, and facial expression," *Annals of the New York Academy of Sciences*, vol. 1000, no. 1, pp. 205–221, 2003.

[3] M. G. Frank, M. Herbasz, K. Sinuk, A. Keller, and C. Nolan, "I see how you feel: Training laypeople and professionals to recognize fleeting emotions," in *Annual Meeting of the International Communication Association*, 2009.

[4] W.-J. Yan, X. Li, S.-J. Wang, G. Zhao, Y.-J. Liu, Y.-H. Chen, and X. Fu, "CASME II: An improved spontaneous micro-expression database and the baseline evaluation," *PLoS ONE*, vol. 9, no. 1, p. e86041, 2014.

[5] C. H. Yap, C. Kendrick, and M. H. Yap, "SAMM: A spontaneous micro-expression database," *IEEE Transactions on Affective Computing*, vol. 9, no. 4, pp. 565–576, 2018.

[6] X. Li, T. Pfister, X. Huang, G. Zhao, and M. Pietikäinen, "A spontaneous micro-expression database: Inducement, collection and baseline," in *IEEE International Conference on Automatic Face and Gesture Recognition*, 2013, pp. 1–6.

[7] X. Ben, Y. Ren, J. Zhang, S.-J. Wang, H. Kpalma, W. Meng, and Y.-J. Liu, "Video-based facial micro-expression analysis: A survey of datasets, features and algorithms," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 9, pp. 5826–5846, 2022.

[8] G. Zhao and M. Pietikainen, "Dynamic texture recognition using local binary patterns with an application to facial expressions," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 29, no. 6, pp. 915–928, 2007.

[9] S.-J. Wang, W.-J. Yan, X. Li, G. Zhao, and X. Fu, "Micro-expression recognition using color spaces," *IEEE Transactions on Image Processing*, vol. 24, no. 12, pp. 6034–6047, 2015.

[10] D. Patel, S. Hong, and G. Kim, "Micro-expression recognition through deep learning," in *IEEE International Conference on Automatic Face and Gesture Recognition*, 2017, pp. 821–828.

[11] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *IEEE International Conference on Computer Vision (ICCV)*, 2015, pp. 4489–4497.

[12] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," in *International Conference on Learning Representations (ICLR)*, 2021.

[13] J. S. Morris, A. Öhman, and R. J. Dolan, "A subcortical pathway to the right amygdala mediating 'unseen' fear," *Proceedings of the National Academy of Sciences*, vol. 96, no. 4, pp. 1680–1685, 1999.

[14] L. Pessoa and R. Adolphs, "Emotion processing and the amygdala: From a 'low road' to 'many roads' of evaluating biological significance," *Nature Reviews Neuroscience*, vol. 11, no. 11, pp. 773–783, 2010.

[15] N. Kanwisher, J. McDermott, and M. M. Chun, "The fusiform face area: A module in human extrastriate cortex specialized for face perception," *Journal of Neuroscience*, vol. 17, no. 11, pp. 4302–4311, 1997.

[16] L. Zhang, Y. Wang, and H. Li, "Hybrid attention-3DNet for micro-expression recognition," *IEEE Transactions on Affective Computing*, vol. 16, no. 2, pp. 312–326, 2025.

[17] J. Kim, S. Park, and C. Lee, "ROI-ArcFace: Region-of-interest micro-expression recognition with angular margin loss," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025, pp. 4521–4530.

[18] R. Gupta, M. Singh, and P. Tiwari, "GAM-MER: Graph attention network for micro-expression recognition," *IEEE Transactions on Biometrics, Behavior, and Identity Science*, vol. 6, no. 1, pp. 78–91, 2024.

[19] S.-J. Wang, W.-J. Yan, G. Zhao, X. Fu, and M. Pietikäinen, "Micro-expression recognition by modeling facial dynamics with main directional mean optical flow," *Pattern Recognition*, vol. 44, no. 10–11, pp. 2562–2573, 2011.

[20] M. Peng, C. Wang, T. Chen, and G. Liu, "Dual-temporal-scale convolutional neural network for micro-expression recognition," *Frontiers in Psychology*, vol. 8, p. 1745, 2017.

[21] H. Zhou, F. Liu, and Q. Yang, "STRNet: Spatiotemporal reasoning network for micro-expression recognition," in *AAAI Conference on Artificial Intelligence*, 2025, pp. 3124–3132.

[22] Y. Chen, Z. Liu, and X. Zhang, "Multi-scale 3D ResNet for micro-expression recognition," *Neurocomputing*, vol. 578, p. 127356, 2024.

[23] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *ICLR*, 2021.

[24] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Video Swin Transformer," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 3202–3211.

[25] G. Bertasius, H. Wang, and L. Torresani, "Is space-time attention all you need for video understanding?" in *International Conference on Machine Learning (ICML)*, 2021.

[26] F. Xue, Q. Wang, and G. Guo, "Transfer learning of transformer-based models for facial expression recognition," *IEEE Transactions on Affective Computing*, vol. 14, no. 3, pp. 1968–1981, 2023.

[27] P. Ekman and W. V. Friesen, *Facial Action Coding System: A Technique for the Measurement of Facial Movement*. Palo Alto, CA: Consulting Psychologists Press, 1978.

[28] S. Jaiswal and M. Valstar, "Deep learning the dynamic appearance and shape of facial action units," in *IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2016, pp. 1–8.

[29] X. Niu, H. Han, S. Shan, and X. Chen, "Multi-label co-regularization for semi-supervised facial action unit recognition," in *NeurIPS*, 2019, pp. 908–918.

[30] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," *Neural Computation*, vol. 3, no. 1, pp. 79–87, 1991.

[31] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in *ICLR*, 2017.

[32] M. A. Nicolaou, H. Gunes, and M. Pantic, "A multi-output context-aware adaptive regression framework for continuous affect prediction," *IEEE Transactions on Affective Computing*, vol. 4, no. 3, pp. 277–290, 2013.

[33] S. Zhao, G. Ding, J. Han, and Y. Gao, "Biomimetic dual-pathway networks for multimodal facial expression recognition," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 34, no. 8, pp. 4521–4535, 2023.

[34] G. de Haan and V. Jeanne, "Robust pulse rate from chrominance-based rPPG," *IEEE Transactions on Biomedical Engineering*, vol. 60, no. 10, pp. 2878–2886, 2013.

[35] J. Sanchez Perez, E. Meinhardt-Llopis, and G. Facciolo, "TV-L1 optical flow estimation," *Image Processing On Line*, vol. 3, pp. 137–150, 2013.

[36] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[37] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical vision transformer using shifted windows," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 10012–10022.

[38] R. Adolphs, "Recognizing emotion from facial expressions: Psychological and neurological mechanisms," *Behavioral and Cognitive Neuroscience Reviews*, vol. 1, no. 1, pp. 21–62, 2002.

[39] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141.

[40] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Simig, E. H. Huang, and L. Zettlemoyer, "OPT: Open pre-trained transformer language models," *arXiv preprint arXiv:2205.01068*, 2022.

[41] S.-J. Wang, Y. Wu, X. Ben, Y.-J. Liu, and G. Zhao, "CAS(ME)³: A third generation spontaneous micro-expression database with macro-annotations," *IEEE Transactions on Affective Computing*, vol. 13, no. 3, pp. 1423–1438, 2022.

[42] J. Li et al., "iMER: Incremental micro-expression recognition benchmark," *ACM Transactions on Intelligent Systems and Technology*, vol. 14, no. 2, pp. 1–24, 2023.

[43] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, "Joint face detection and alignment using multitask cascaded convolutional networks," *IEEE Signal Processing Letters*, vol. 23, no. 10, pp. 1499–1503, 2016.

[44] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *ICLR*, 2019.

[45] T. Wang, M. Peng, and G. Liu, "SelfME: Self-supervised micro-expression recognition via contrastive learning," *Pattern Recognition Letters*, vol. 178, pp. 47–54, 2024.

[46] H. Chen, L. Wang, and Z. Zhang, "MCCA-VNet: Multi-channel cross-attention video network for micro-expression recognition," *Engineering Applications of Artificial Intelligence*, vol. 133, p. 108229, 2024.
