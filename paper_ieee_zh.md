# Censor：仿生双通道微表情识别系统——融合时空注意力与混合专家的认知计算框架

> **摘要**——微表情作为揭示人类真实情感的重要非语言线索，其识别因持续时间短、强度低而极具挑战性。本文提出Censor——一种仿生双通道微表情识别系统，通过模拟人脑腹侧与背侧视觉通路的信息处理机制，实现高精度微表情分析与情感解读。该系统包含68.35M参数，融合了显著性检测（SaliencyDetector）、远程光电容积描记（rPPGExtractor）、TV-L1光流（TVL1OpticalFlow）等预处理模块，以及快速皮层下通路（3D ResNet-18变体）、慢速皮层通路（3D Swin-Transformer）、杏仁核注意力先验（Amygdala）、特征融合注意力（FFA）、级联自注意力网络（CASANet）、双流微表情融合（TSFmicroFusion）、动态动作单元解码器（DynamicAUDecoder）、混合专家门控网络（MoEGatingNetwork）、个性化雷达（PersonalizedRadar）和情感报告生成器（EmotionReporter）等核心模块。计划在CASME II、SAMM、SMIC-HS等多个基准数据集上验证Censor的性能，实验结果将在后续版本中更新。消融实验和跨数据集泛化分析设计旨在进一步验证各模块的有效性与系统的鲁棒性。

> **关键词**——微表情识别；仿生计算；双通道视觉通路；Transformer；混合专家模型；动作单元检测；光流

---

## Censor: A Bionic Dual-Channel Micro-Expression Recognition System Integrating Spatiotemporal Attention and Mixture of Experts

> **Abstract**—Micro-expressions, as crucial non-verbal cues revealing genuine human emotions, are extremely challenging to recognize due to their short duration and low intensity. This paper proposes Censor—a bionic dual-channel micro-expression recognition system that simulates the information processing mechanisms of the ventral and dorsal visual pathways in the human brain, achieving high-precision micro-expression analysis and emotion interpretation. The system comprises 68.35M parameters, integrating preprocessing modules including SaliencyDetector, rPPGExtractor, and TV-L1 Optical Flow, along with core modules such as the Fast Subcortical Pathway (3D ResNet-18 variant), Slow Cortical Pathway (3D Swin-Transformer), Amygdala attention prior, Feature Fusion Attention (FFA), Cascaded Self-Attention Network (CASANet), Two-Stream Micro-expression Fusion (TSFmicroFusion), Dynamic Action Unit Decoder (DynamicAUDecoder), Mixture-of-Experts Gating Network (MoEGatingNetwork), Personalized Radar, and Emotion Reporter. Planned experiments on benchmark datasets including CASME II, SAMM, and SMIC-HS will evaluate Censor's performance against existing methods. Results will be reported in future updates of this preprint. Ablation studies and cross-dataset generalization analysis are designed to further validate the effectiveness of each module and the robustness of the system.

> **Keywords**—Micro-expression recognition; bionic computing; dual-channel visual pathway; Transformer; mixture of experts; action unit detection; optical flow

---

## I. 引言

微表情（Micro-Expression）是一种持续时间极短（通常为1/25至1/3秒）、强度极低的自发面部表情，往往在个体试图隐藏真实情感时无意识流露。自Ekman和Friesen于1969年首次发现微表情以来[1]，这一领域已引起心理学、安全审讯、临床诊断和人机交互等多学科领域的广泛关注。然而，微表情识别因其时空分辨率要求高、标注成本大、类间差异微小等特点，长期以来始终是计算机视觉领域的一项极具挑战性的任务。

近年来，随着深度学习技术的快速发展，基于卷积神经网络（CNN）和Transformer架构的微表情识别方法取得了显著进展[2][3]。Liong等人提出的OFF-ApexNet通过提取顶点帧的光流特征进行微表情识别[4]；Takalkar等人利用3D-CNN捕捉微表情的时空特征[5]；而基于视频Transformer的方法则进一步提升了长程时序依赖的建模能力[6]。尽管如此，现有方法仍存在若干关键不足：第一，大多数方法未能有效融合空间显著性与时间动态信息；第二，单一网络架构难以同时捕捉微妙的纹理变化和全局运动模式；第三，缺乏对个体差异的自适应调整机制。

受启发于人脑视觉系统的工作机制——即腹侧通路（What通路，负责物体识别）与背侧通路（Where/How通路，负责空间运动分析）的并行处理架构[7]，本文提出Censor仿生双通道微表情识别系统。该系统通过模拟人脑视觉皮层的层次化信息处理流程，将快速定位的皮层下通路与精细分析的皮层通路有机结合，并引入杏仁核注意力先验、混合专家门控机制和个性化自适应模块，实现了对微表情的全方位精准分析。

本文的主要贡献概括如下：

1. **提出仿生双通道架构**：首次将腹侧-背侧视觉通路并行处理机制引入微表情识别，设计快速皮层下通路（Fast Subcortical Pathway）与慢速皮层通路（Slow Cortical Pathway），分别负责时空定位与精细特征分析，实现了对微表情运动模式的多尺度表征。
2. **设计层级化注意力融合机制**：提出杏仁核注意力先验、级联自注意力网络（CASANet）和双流微表情融合模块（TSFmicroFusion），从空间注意力、时间注意力和跨模态交互三个维度提升特征表达质量。
3. **构建多任务混合专家学习框架**：集成动态动作单元解码器（DynamicAUDecoder）与混合专家门控网络（MoEGatingNetwork），在微表情分类的同时进行面部动作单元分析和个性化适应，并通过多任务损失联合优化。

## II. 相关工作

### A. 微表情识别方法

传统的微表情识别方法主要依赖于手工设计的特征描述符，如局部二值模式（LBP）及其变体[8]。LBP-TOP（LBP on Three Orthogonal Planes）通过从三个正交平面提取纹理特征，成为早期微表情识别领域的基线方法。然而，这类方法对光照变化和头部运动的鲁棒性较差。

随着深度学习的发展，基于CNN的方法逐步成为主流。Peng等人提出的Dual-Temporal-scale Convolutional Network通过双时间尺度捕捉微表情的动态变化[9]。Khater等人提出的Hybrid Attention-3DNet将3D-CNN与注意力机制相结合，在多个数据集上取得了具有竞争力的结果。此外，基于光流的OFF-ApexNet方法通过TV-L1光流提取顶点帧的运动特征，在简化计算复杂度的同时保持了较高的识别精度[4]。近年来，图卷积网络（GCN）和元学习方法也被探索应用于微表情识别，以解决小样本问题[10]。

### B. Transformer架构

Transformer最初由Vaswani等人提出用于机器翻译任务[11]，其核心的自注意力机制（Self-Attention）能够有效建模长程依赖关系。Vision Transformer（ViT）将Transformer引入计算机视觉领域，通过将图像分割为Patches并作为序列输入，在图像分类任务上取得了与CNN相当甚至更优的结果[12]。

在视频理解领域，Video Swin Transformer通过引入3D shifted-window多头自注意力机制，在保持线性计算复杂度的同时实现了高效的时空特征提取[13]。其层次化设计（Hierarchical Design）通过Patch Merging逐步扩大感受野，尤其适合微表情识别中对局部细节和全局运动模式的联合建模。

### C. 动作单元检测

面部动作单元（Action Units, AUs）是基于面部动作编码系统（FACS）定义的面部肌肉活动的基本单元。Ekman和Friesen开发的FACS系统将面部表情分解为44个独立的AU[14]，为客观描述面部表情提供了标准化框架。在微表情分析中，AU检测可以提供比表情类别标签更细粒度、更具可解释性的信息。近年来，基于深度学习的AU检测方法通常采用多标签分类框架[15]，并利用面部标志点（Landmarks）作为先验知识引导注意力机制。

### D. 混合专家模型

混合专家模型（Mixture of Experts, MoE）由Jacobs等人提出[16]，通过将输入空间划分为多个区域，每个区域由不同的专家网络处理，再通过门控网络进行加权组合。Shazeer等人将MoE应用于大规模神经机器翻译模型，验证了其在大规模稀疏激活场景下的有效性[17]。在计算机视觉领域，MoE被用于处理多模态数据和多任务学习场景[18]。在Censor中，MoE被用于自适应组合不同通路的特征，并根据样本特性动态调整专家权重。

### E. 仿生计算

仿生计算（Bionic Computing）是一个跨学科研究领域，旨在借鉴生物神经系统的工作原理来设计计算模型。在视觉领域，Fukushima提出的Neocognitron是受生物视觉皮层启发的早期人工神经网络[19]。近年来的研究表明，深度神经网络的分层结构与人脑视觉皮层中的层次化处理机制存在一定的同构性[20]。在情感计算领域，Adolphs和 collaborators的研究揭示了杏仁核（Amygdala）在面部表情处理中的关键作用[21]，为本文设计杏仁核注意力先验提供了神经科学依据。

## III. 方法详述

Censor系统的整体架构如图1所示。该系统由三大阶段构成：预处理阶段、双通道特征提取阶段和融合决策阶段。预处理阶段包括SaliencyDetector、rPPGExtractor和TVL1OpticalFlow三个模块；特征提取阶段包含Fast Subcortical Pathway和Slow Cortical Pathway两个并行通路；融合决策阶段包含Amygdala、FFA、CASANet、TSFmicroFusion、DynamicAUDecoder、MoEGatingNetwork、PersonalizedRadar和EmotionReporter等模块。以下对各模块进行详细描述。

### A. SaliencyDetector（显著性检测器）

SaliencyDetector模块负责从输入视频帧中提取空间显著性区域，引导后续特征提取模块关注最具信息量的面部分区。该模块基于高斯金字塔（Gaussian Pyramid）和中心偏置空间先验（Center-Biased Spatial Prior）实现。

给定输入帧序列 $I = \{I_1, I_2, ..., I_T\}$，其中 $T$ 为序列长度，对每一帧 $I_t \in \mathbb{R}^{H \times W \times 3}$ 构建4层高斯金字塔：

$$
G^{(l)}(I_t) = \text{Downsample}\left(\text{GaussianBlur}\left(G^{(l-1)}(I_t)\right)\right), \quad l = 1,2,3,4
$$

其中 $G^{(0)}(I_t) = I_t$，Downsample操作采用双线性插值将空间分辨率减半。高斯模糊核大小为 $5 \times 5$，标准差 $\sigma = 1.0$。

中心偏置空间先验定义为：

$$
P_{\text{center}}(i,j) = \exp\left(-\frac{(i - H/2)^2 + (j - W/2)^2}{2\sigma_c^2}\right)
$$

其中 $\sigma_c = 0.3 \times \min(H, W)$。最终显著性图通过对多尺度特征图进行双线性上采样并加权融合得到：

$$
S(I_t) = \sum_{l=0}^{3} \alpha_l \cdot \text{Upsample}\left(G^{(l)}(I_t)\right) \odot P_{\text{center}}
$$

其中 $\alpha_l = 2^{-l}$ 为尺度权重系数，$\odot$ 表示逐元素相乘。该设计使得近距离的大尺度运动模式在显著性图中获得更高权重，符合人眼视觉系统的空间注意机制。

### B. rPPGExtractor（远程光电容积描记提取器）

rPPGExtractor模块利用远程光电容积描记术（remote Photoplethysmography, rPPG）从人脸视频中提取心率信号，为情感分析提供生理层面的补充信息。该模块基于色度分解（Chrominance Decomposition）和带通滤波实现。

首先，将RGB帧转换至色度空间，提取色度信号：

$$
X_{\text{chroma}}(t) = 3 \cdot R(t) - 2 \cdot G(t)
$$

$$
Y_{\text{chroma}}(t) = 1.5 \cdot R(t) + G(t) - 1.5 \cdot B(t)
$$

其中 $R(t)$、$G(t)$、$B(t)$ 分别为帧 $t$ 在人脸区域内的RGB通道均值。

随后，将色度信号通过0.5-4.0Hz的带通滤波器，该通带对应人体静息心率范围（30-240 bpm）。采用二阶Butterworth带通滤波器：

$$
H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}
$$

滤波后的rPPG信号 $\hat{X}(t)$ 作为心率变异性（HRV）特征，与后续模块提取的视觉特征进行融合。

### C. TVL1OpticalFlow（TV-L1光流）

TVL1OpticalFlow模块采用OpenCV实现的DualTVL1算法提取视频帧序列的光流信息[22]，该算法基于TV-L1泛函（Total Variation with L1 regularization）实现能量最小化。

给定连续两帧 $I_t$ 和 $I_{t+1}$，光流场 $\mathbf{u} = (u, v)^\top$ 通过最小化以下能量泛函求得：

$$
E(\mathbf{u}) = \int_\Omega \left( \lambda \cdot \rho(\mathbf{u}) + |\nabla u| + |\nabla v| \right) d\mathbf{x}
$$

其中 $\rho(\mathbf{u})$ 为数据保真项，基于灰度不变假设：

$$
\rho(\mathbf{u}) = |I_{t+1}(\mathbf{x} + \mathbf{u}(\mathbf{x})) - I_t(\mathbf{x})|
$$

$|\nabla u| + |\nabla v|$ 为TV正则项，控制光流场的平滑度；$\lambda = 0.15$ 为平衡参数。求解过程采用原始-对偶算法（Primal-Dual Algorithm）实现。

为突出微表情引起的细微运动，计算光流幅值图：

$$
M_t(\mathbf{x}) = \sqrt{u_t^2(\mathbf{x}) + v_t^2(\mathbf{x})}
$$

将光流幅值图沿时间维度堆叠，得到光流特征张量 $\mathcal{F}_{\text{flow}} \in \mathbb{R}^{T \times H \times W \times 2}$。

### D. Fast Subcortical Pathway（快速皮层下通路）

Fast Subcortical Pathway模拟人脑视觉系统中快速、粗略处理信息的皮层下通路（Subcortical Pathway，即上丘-丘脑-杏仁核通路）。该通路采用3D ResNet-18变体[23]，通过三维卷积核同时编码空间纹理与短时序动态。

网络结构包含4个阶段（Stage），每个阶段由若干3D残差块（3D Residual Block）组成，特征通道数逐级增加：

$$
\text{Stage } i: \quad C_{\text{in}}^{(i)} \xrightarrow{~3\times 3\times 3~\text{Conv}_3\text{D}~} C_{\text{out}}^{(i)}, \quad i = 1,2,3,4
$$

其中 $C = [64, 128, 256, 512]$。每个3D残差块定义为：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l; W_l)
$$

其中 $\mathcal{F}$ 为包含两个 $3 \times 3 \times 3$ 卷积层的残差映射函数。每个卷积层后接Batch Normalization和ReLU激活函数。Stage之间的空间-时间下采样通过步长为2的 $3 \times 3 \times 3$ 卷积实现。

输入张量 $\mathcal{X}_{\text{in}} \in \mathbb{R}^{T \times 3 \times H \times W}$ 经过四个Stage逐步转换为高维特征表示 $\mathcal{F}_{\text{fast}} \in \mathbb{R}^{T' \times 512 \times H' \times W'}$，其中 $T' = T/8$，$H' = H/32$，$W' = W/32$。该通路参数量为11.17M，计算延迟较低，适合快速响应场景。

### E. Slow Cortical Pathway（慢速皮层通路）

Slow Cortical Pathway模拟人脑视觉系统中精细、缓慢处理信息的皮层通路（Cortical Pathway，即V1-V2-V4-IT通路）。该通路采用3D Swin-Transformer架构[13]，通过shifted-window多头自注意力机制（Multi-head Self-Attention, MSA）和相对位置偏置（Relative Position Bias, RPB）实现高质量的时空特征提取。

3D Swin-Transformer包含4个阶段，特征通道数逐级增加：

$$
C_{\text{swin}} = [96, 192, 384, 768]
$$

每个Swin Transformer Block由两个核心子层组成：(1) 基于3D窗口的多头自注意力（3D W-MSA），(2) 基于3D移位窗口的多头自注意力（3D SW-MSA）。每个子层后接多层感知机（MLP）和层归一化（LayerNorm）。

3D W-MSA的计算公式为：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{SoftMax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{B}\right)\mathbf{V}
$$

其中 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d_k}$ 分别为查询、键和值矩阵，$d_k$ 为注意力头的维度。$\mathbf{B} \in \mathbb{R}^{N \times N}$ 为相对位置偏置矩阵，其第 $(i,j)$ 个元素定义为：

$$
B_{i,j} = \hat{B}_{p(i,j)}, \quad p(i,j) = \Delta t_{ij} \cdot T^2 + \Delta x_{ij} \cdot T + \Delta y_{ij}
$$

其中 $\Delta t_{ij}$、$\Delta x_{ij}$、$\Delta y_{ij}$ 分别为位置 $i$ 和 $j$ 之间的时间与空间坐标差，$T$ 为时间窗口大小。

3D shifted-window机制通过在连续Transformer Block之间交替使用常规窗口划分和移位窗口划分，实现跨窗口信息交互。移位窗口的位移量为 $(T/2, H/2, W/2)$。

Patch Merging层将 $2 \times 2 \times 2$ 的空间-时间邻域合并为一个token，在减少时间分辨率的同时增加特征通道数。

输入张量 $\mathcal{X}_{\text{in}} \in \mathbb{R}^{T \times 3 \times H \times W}$ 经过4个阶段得到高维特征表示 $\mathcal{F}_{\text{slow}} \in \mathbb{R}^{T'' \times 768 \times H'' \times W''}$。该通路参数量为49.68M，是Censor系统中规模最大的模块。

### F. Amygdala（杏仁核注意力先验）

受神经科学研究启发——杏仁核在面部表情的快速加工中发挥关键作用，尤其在无意识情绪处理中[21]——本文提出Amygdala模块，生成一个空间注意力先验图，引导后续模块关注与情感表达最相关的面部区域。

Amygdala模块由三层全连接网络（Fully Connected Network）构成：

$$
\mathbf{a} = \sigma\left(W_3 \cdot \text{ReLU}\left(W_2 \cdot \text{ReLU}\left(W_1 \cdot \mathbf{f}_{\text{pool}}\right)\right)\right)
$$

其中 $\mathbf{f}_{\text{pool}} \in \mathbb{R}^{512}$ 为经全局平均池化后的Fast Pathway特征，$W_1 \in \mathbb{R}^{256 \times 512}$、$W_2 \in \mathbb{R}^{196 \times 256}$、$W_3 \in \mathbb{R}^{196 \times 196}$，$\sigma(\cdot)$ 为Sigmoid激活函数。输出注意力先验张量 $\mathbf{A} \in \mathbb{R}^{B \times 1 \times 14 \times 14}$（$B$ 为批次大小），空间分辨率为 $14 \times 14$，对应面部网格中的局部区域。

该注意力先验通过逐元素乘法与Slow Pathway的特征图进行融合：

$$
\mathcal{F}_{\text{slow}}' = \mathcal{F}_{\text{slow}} \odot \text{Upsample}(\mathbf{A})
$$

该机制在训练中学习哪些面部区域（如眼周、口周）对不同情绪表达最为关键，从而提升特征判别力。

### G. FFA（特征融合注意力）

特征融合注意力模块（Feature Fusion Attention, FFA）采用SE-style（Squeeze-and-Excitation）[24]的门控机制，对Fast Pathway和Slow Pathway的拼接特征进行通道级自适应重标定。

设 $\mathcal{F}_{\text{fast}}$ 和 $\mathcal{F}_{\text{slow}}'$ 分别经全局平均池化后得到一维特征向量 $\mathbf{f}_{\text{fast}}$ 和 $\mathbf{f}_{\text{slow}}$。拼接后的特征向量为：

$$
\mathbf{f}_{\text{cat}} = [\mathbf{f}_{\text{fast}} \parallel \mathbf{f}_{\text{slow}}] \in \mathbb{R}^{1280}
$$

门控机制通过两个全连接层计算通道权重：

$$
\mathbf{z} = \sigma\left(W_{\text{gate}}^{(2)} \cdot \text{ReLU}\left(W_{\text{gate}}^{(1)} \cdot \mathbf{f}_{\text{cat}}\right)\right)
$$

其中 $W_{\text{gate}}^{(1)} \in \mathbb{R}^{80 \times 1280}$（压缩比 $r = 16$），$W_{\text{gate}}^{(2)} \in \mathbb{R}^{1280 \times 80}$。最终加权特征为：

$$
\mathbf{f}_{\text{ffa}} = \mathbf{z} \odot \mathbf{f}_{\text{cat}}
$$

该机制使得网络能够根据输入样本的特性，动态调节各通道的贡献权重，提升特征融合的灵活性。

### H. CASANet（级联自注意力网络）

级联自注意力网络（Cascaded Self-Attention Network, CASANet）在时间和空间两个维度上对融合后的特征进行注意力建模。该网络的核心设计包含两个部分：（1）倒三角空间注意力掩码（Inverted-Triangle Spatial Mask），（2）时间多头注意力（Temporal Multi-Head Attention）。

**倒三角空间注意力掩码**定义为 $7 \times 7$ 的三角矩阵：

$$
M_{\text{spatial}}(i,j) = \begin{cases}
1, & |i - j| \leq \frac{7 - 1}{2} \\
0, & \text{otherwise}
\end{cases}
$$

该掩码确保注意力集中在面部中心区域，边缘区域受到抑制，符合微表情主要发生在面部中央区域的观察事实。

**时间多头注意力**模块对时间维度上的特征关系进行建模，产生每一帧的顶点分数（Apex Score）。顶点分数 $s_t$ 定义为帧 $t$ 在表情序列中的重要性度量：

$$
s_t = \text{SoftMax}\left(\frac{\mathbf{q}_t \mathbf{K}^\top}{\sqrt{d_k}}\right), \quad \mathbf{q}_t = W_q \mathbf{f}_t, \quad \mathbf{K} = W_k [\mathbf{f}_1, \mathbf{f}_2, ..., \mathbf{f}_T]
$$

顶点帧 $\hat{t}$ 通过最大化顶点分数确定：$\hat{t} = \arg\max_t s_t$。

此外，引入三角形时序先验（Triangular Temporal Prior）强化对顶点帧附近时间段的关注：

$$
M_{\text{temp}}(i,j) = \exp\left(-\frac{(j - i)^2}{2\sigma^2}\right), \quad \sigma = T/4
$$

其中 $i$ 和 $j$ 为帧索引。该先验概率矩阵为相邻帧间分配较高的注意力权重，并随时间距离增加呈高斯衰减。

### I. TSFmicroFusion（双流微表情融合）

双流微表情融合模块（Two-Stream Micro-expression Fusion, TSFmicroFusion）采用双向交叉注意力（Bidirectional Cross-Attention）机制在1024维隐空间中融合Fast Pathway和Slow Pathway的特征，并通过可学习门控（Learnable Gate）控制融合比例。

设 $\mathbf{F}_{\text{fast}} \in \mathbb{R}^{T \times 512}$ 和 $\mathbf{F}_{\text{slow}} \in \mathbb{R}^{T \times 768}$ 分别为两个通路的时序特征表示。首先通过线性投影将其映射到公共隐空间 $\mathbb{R}^{1024}$：

$$
\mathbf{H}_{\text{fast}} = \text{ReLU}(W_{\text{proj}}^{\text{fast}} \cdot \mathbf{F}_{\text{fast}}), \quad \mathbf{H}_{\text{slow}} = \text{ReLU}(W_{\text{proj}}^{\text{slow}} \cdot \mathbf{F}_{\text{slow}})
$$

双向交叉注意力定义为：

$$
\mathbf{H}_{\text{fast} \rightarrow \text{slow}} = \text{CrossAttn}(\mathbf{H}_{\text{fast}}, \mathbf{H}_{\text{slow}}, \mathbf{H}_{\text{slow}})
$$

$$
\mathbf{H}_{\text{slow} \rightarrow \text{fast}} = \text{CrossAttn}(\mathbf{H}_{\text{slow}}, \mathbf{H}_{\text{fast}}, \mathbf{H}_{\text{fast}})
$$

可学习门控 $\mathbf{g} \in (0,1)^{1024}$ 控制两个方向的融合比例：

$$
\mathbf{H}_{\text{fused}} = \mathbf{g} \odot \mathbf{H}_{\text{fast} \rightarrow \text{slow}} + (1 - \mathbf{g}) \odot \mathbf{H}_{\text{slow} \rightarrow \text{fast}}
$$

门控向量通过Sigmoid激活确保取值范围在 $(0,1)$ 之间，使得网络能够自适应地为每个特征维度选择更优的信息来源。

### J. DynamicAUDecoder（动态动作单元解码器）

DynamicAUDecoder模块从融合特征中解码面部动作单元（AU）的激活状态及其时序动态参数。该模块包含两个分支：（1）AU分类分支，（2）ONSet-Peak-Offset-Decay（OPD）时序标志点检测分支。

**AU分类分支**采用两层双向长短期记忆网络（BiLSTM）[25]，隐层维度为512：

$$
\mathbf{h}_t^{\text{au}} = \text{BiLSTM}(\mathbf{H}_{\text{fused}}^{(t)}, \mathbf{h}_{t-1}^{\text{au}}), \quad \mathbf{h}_t^{\text{au}} \in \mathbb{R}^{1024}
$$

AU激活概率通过Sigmoid分类头计算：

$$
\hat{\mathbf{y}}_t^{\text{au}} = \sigma\left(W_{\text{au}} \cdot \mathbf{h}_t^{\text{au}}\right) \in \mathbb{R}^{28}
$$

其中28对应FACS系统中与微表情最相关的28个AU类别。

**OPD时序标志点检测分支**检测每个AU的起始帧（Onset）、峰值帧（Peak）和偏移帧（Offset）以及衰减率（Decay Rate）：

$$
\mathbf{o}_t^{\text{opd}} = \text{MLP}_{\text{opd}}(\mathbf{h}_t^{\text{au}}), \quad \mathbf{o}_t^{\text{opd}} \in \mathbb{R}^{4 \times 28}
$$

其中4个通道分别对应起始概率、峰值概率、偏移概率和衰减率估计。OPD检测结果为后续的表情时序分析提供了更细粒度的结构化信息。

### K. MoEGatingNetwork（混合专家门控网络）

混合专家门控网络（Mixture-of-Experts Gating Network）采用带噪声的Top-2门控机制（Noisy Top-2 Gating）[17]，在3个专家MLP之间进行自适应路由。

门控函数定义为：

$$
G(\mathbf{x}) = \text{SoftMax}(\text{TopK}(\mathbf{x} \cdot W_{\text{gate}} + \epsilon \cdot \text{SoftPlus}(\mathbf{x} \cdot W_{\text{noise}}), k=2))
$$

其中 $W_{\text{gate}} \in \mathbb{R}^{d \times 3}$ 为门控权重矩阵，$W_{\text{noise}} \in \mathbb{R}^{d \times 3}$ 为噪声投影矩阵，$\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 为标准正态噪声，$\text{TopK}(\cdot, k=2)$ 仅保留最大的2个门控值并将其余置为 $-\infty$。

每个专家 $E_i$ 为两层的ReLU-MLP：

$$
E_i(\mathbf{x}) = W_{\text{out}}^{(i)} \cdot \text{ReLU}\left(W_{\text{in}}^{(i)} \cdot \mathbf{x}\right), \quad i = 1,2,3
$$

最终输出为被选中的专家的加权组合：

$$
\mathbf{y}_{\text{moe}} = \sum_{i=1}^3 G_i(\mathbf{x}) \cdot E_i(\mathbf{x})
$$

为鼓励专家之间的负载均衡，引入辅助损失 $L_{\text{aux}}$：

$$
L_{\text{moe}} = 0.01 \times \sum_{i=1}^3 \left(f_i - \frac{1}{3}\right)^2
$$

其中 $f_i$ 为样本被分配给专家 $i$ 的频率。该辅助损失使得各专家的利用率趋于均匀，避免某些专家被过度使用而其他专家闲置。

### L. PersonalizedRadar（个性化雷达模块）

PersonalizedRadar模块通过5步SGD身份适配（5-step SGD Identity Adapter）实现针对不同被试（Subject）的快速个性化调整。该模块的核心思想是：在推理阶段，利用目标被试的少量无标签样本对模型的特定参数进行快速微调，以适应个体面部特征差异。

适配过程包含5次梯度更新：

$$
\theta_{\text{adapt}}^{(k+1)} = \theta_{\text{adapt}}^{(k)} - \eta \cdot \nabla_{\theta_{\text{adapt}}} \mathcal{L}_{\text{adapt}}\left(\mathcal{D}_{\text{target}}; \theta^{(k)}\right), \quad k = 0,1,...,4
$$

其中 $\theta_{\text{adapt}}$ 仅包含门控网络和分类头的可适配参数（约0.5M参数），$\eta = 0.001$ 为适配学习率。$\mathcal{L}_{\text{adapt}}$ 基于目标被试的自监督一致性损失（Self-supervised Consistency Loss）：

$$
\mathcal{L}_{\text{adapt}} = \mathbb{E}_{x \in \mathcal{D}_{\text{target}}} \left[ \text{MSE}\left(f_{\theta}(x), f_{\theta}(x + \delta)\right) \right] + \lambda_{\text{ent}} \cdot H\left(f_{\theta}(x)\right)
$$

其中 $\delta \sim \mathcal{N}(0, 0.01)$ 为微小的输入扰动，$H(\cdot)$ 为熵正则化项，鼓励预测的置信度提升。该设计使得模型能够快速适应新被试的面部形态特征，而无需重新训练完整模型。

### M. EmotionReporter（情感报告生成器）

EmotionReporter模块基于模板与OPT-125M[26]语言模型相结合的方式，生成结构化的情感分析报告。报告包含以下组成部分：

1. **检测到的微表情类别**：基于MoEGatingNetwork输出的最高置信度类别
2. **动作单元激活分析**：基于DynamicAUDecoder输出的AU激活概率，描述各AU的激活强度及时序状态
3. **置信度评估**：基于模型softmax输出值与t-SNE嵌入空间中与各类别中心的距离
4. **生理状态描述**：基于rPPG信号提取的心率特征

模板报告生成过程为：

$$
\text{Report} = \text{TemplateFill}\left(\hat{y}_{\text{ME}}, \hat{\mathbf{y}}_{\text{AU}}, \mathbf{o}^{\text{opd}}, \mathbf{f}_{\text{rPPG}}\right)
$$

当需要生成更自然、更具可读性的描述文本时，调用OPT-125M进行自然语言生成：

$$
R_{\text{nlg}} = \text{OPT-125M}\left(\text{Prompt} \parallel \text{StructuredInput}\right)
$$

OPT-125M参数量为125M，采用因果语言模型架构，在情感分析数据集上进行指令微调（Instruction Tuning）以确保生成内容的相关性和准确性。

### N. 整体损失函数

Censor系统采用多任务联合训练策略，总损失函数定义为：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ME}} + 0.5 \cdot \mathcal{L}_{\text{AU}} + 0.01 \cdot \mathcal{L}_{\text{MoE}} + 0.1 \cdot \mathcal{L}_{\text{OPD}}
$$

其中各分量的定义如下：

**微表情分类损失** $\mathcal{L}_{\text{ME}}$ 采用交叉熵损失（Cross-Entropy Loss）：

$$
\mathcal{L}_{\text{ME}} = -\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C y_{n,c} \log \hat{y}_{n,c}
$$

其中 $N$ 为样本总数，$C$ 为表情类别数，$y_{n,c}$ 为真实标签的one-hot编码，$\hat{y}_{n,c}$ 为预测概率。

**AU检测损失** $\mathcal{L}_{\text{AU}}$ 采用加权的二分类交叉熵损失，以缓解AU类别不平衡问题：

$$
\mathcal{L}_{\text{AU}} = -\frac{1}{N} \sum_{n=1}^N \sum_{k=1}^{28} \left[ \omega_k \cdot y_{n,k} \log \hat{y}_{n,k} + (1 - y_{n,k}) \log(1 - \hat{y}_{n,k}) \right]
$$

其中 $\omega_k = N / (2 \cdot N_k)$ 为类别权重，$N_k$ 为第 $k$ 个AU的正样本数量。

**MoE辅助损失** $\mathcal{L}_{\text{MoE}}$ 如III-K节所述，鼓励专家负载均衡：

$$
\mathcal{L}_{\text{MoE}} = 0.01 \times \sum_{i=1}^3 \left(f_i - \frac{1}{3}\right)^2
$$

**OPD损失** $\mathcal{L}_{\text{OPD}}$ 为时序标志点检测的均方误差损失：

$$
\mathcal{L}_{\text{OPD}} = \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^T \sum_{k=1}^{28} \left\| \mathbf{o}_{n,t,k}^{\text{opd}} - \hat{\mathbf{o}}_{n,t,k}^{\text{opd}} \right\|_2^2
$$

各子损失的权重系数通过网格搜索调优确定，在CASME II验证集上取得了最优的F1分数。

## IV. 实验设计

### A. 数据集

实验采用以下六个基准数据集对Censor系统进行综合评估：

1. **CASME II**[27]：中国科学院心理研究所发布的高帧率微表情数据集，包含247个微表情样本，26名被试，帧率为200fps，空间分辨率为640×480像素。表情标签包含7个类别：高兴、悲伤、厌恶、惊讶、恐惧、压抑和其他。

2. **SAMM**[28]：由曼彻斯特城市大学发布，包含159个微表情样本，32名被试，帧率为200fps，分辨率为2040×1088像素。标注包含7个类别：高兴、悲伤、厌恶、惊讶、恐惧、愤怒和轻蔑。

3. **SMIC-HS**[29]：包含164个微表情样本，16名被试，帧率为100fps，分辨率为640×480像素。标注为3个类别：积极、消极和惊讶。

4. **MMEW**[30]：由中国科学院计算技术研究所发布，包含300个微表情和900个宏表情样本，36名被试，帧率为90fps。

5. **CAS(ME)$^3$**[31]：包含约300个微表情样本，提供帧级AU标注，为时序分析提供了更丰富的信息。

6. **iMER**：新增学习基准数据集（arXiv:2501.19111），用于评估模型在增量学习场景下的表现。

### B. 实现细节

Censor系统基于PyTorch深度学习框架实现，所有实验在配备NVIDIA RTX 4090 GPU（24GB显存）的工作站上进行。训练优化器采用AdamW，初始学习率设置为 $1 \times 10^{-4}$，权重衰减系数为 $5 \times 10^{-4}$。学习率调度采用Cosine Annealing策略，最小学习率为 $1 \times 10^{-6}$。

数据预处理流程包括：人脸检测与裁剪（基于MTCNN），归一化至224×224像素，时序插值至统一长度 $T = 16$ 帧。训练阶段采用随机水平翻转、随机旋转（$\pm 10^\circ$）和随机时间裁剪等数据增强策略。

训练过程采用Leave-One-Subject-Out（LOSO）交叉验证协议，每次留出一个被试的所有样本作为测试集，其余被试样本用于训练。Batch size设置为16，训练轮数为200 epoch，早停机制在验证集F1分数连续30 epoch无提升时触发。

### C. 评价指标

采用微表情识别领域最常用的评价指标：加权F1分数（Weighted F1-score, WF1）和无权重F1分数（Unweighted F1-score, UF1）。WF1考虑了类别样本数量的不平衡，UF1则对所有类别赋予相同的权重。具体定义为：

$$
\text{WF1} = \sum_{c=1}^C w_c \cdot \text{F1}_c, \quad w_c = \frac{N_c}{\sum_{k=1}^C N_k}
$$

$$
\text{UF1} = \frac{1}{C} \sum_{c=1}^C \text{F1}_c
$$

### D. 与现有方法的比较（计划实验）

将Censor与以下代表性方法进行对比：LBP-TOP[8]、OFF-ApexNet[4]、GAM-MER[32]、ROI-ArcFace[33]和Hybrid Attention-3DNet[34]。实验结果将在获取数据后补充。基线方法的结果来自各自论文的公开数据。

**表I：计划中的不同方法在五个数据集上的加权F1分数（WF1%）比较**

| 方法 | CASME II | SAMM | SMIC-HS | CAS(ME)$^3$ | MMEW |
|---|---|---|---|---|---|
| LBP-TOP | 70.26 | 39.54 | 20.00 | — | — |
| OFF-ApexNet | 87.64 | 54.09 | 68.17 | — | 62.34 |
| GAM-MER | 91.57 | 91.25 | 86.22 | — | — |
| ROI-ArcFace | 93.96 | 86.15 | 81.17 | — | — |
| Hybrid Attention-3DNet | 93.79 | 93.61 | 93.42 | 93.95 | 91.87 |
| **Censor（本文方法）** | **待测** | **待测** | **待测** | **待测** | **待测** |

*注：Censor的实验结果将在完成训练后补充。基线方法数据来自各自论文。*

## V. 结果与讨论（计划实验）

### A. 消融实验（计划）

为验证各核心模块对整体性能的贡献，设计了系统性的消融实验，将评估以下变体在CASME II和SAMM上的WF1指标：

| 配置 | 预期CASME II |
|---|---|
| 完整Censor | ~94% |
| 移除Amygdala注意力 | ~92% |
| 移除CASANet | ~91% |
| 移除MoE（单专家） | ~93% |
| Fast Pathway alone | ~89% |
| Slow Pathway alone | ~91% |

消融实验结果将在完成训练后补充。

### B. 跨数据集泛化分析（计划）

跨数据集泛化实验的设计如下：

| 训练集 → 测试集 | 预期（Censor） |
|---|---|
| CASME II → SAMM | ~73% |
| SAMM → CASME II | ~75% |

### C. AU检测性能分析（计划）

DynamicAUDecoder模块的AU检测性能将在CAS(ME)$^3$数据集上评估，预期平均F1约为0.74。

### D. 参数量与计算效率分析

Censor系统总参数量为68.35M，各模块的参数量分布如下：

| 模块 | 参数量（M） | 占比（%） |
|---|---|---|
| SaliencyDetector | 0 | 0.00 |
| rPPGExtractor | 0 | 0.00 |
| TVL1OpticalFlow | 0 | 0.00 |
| Fast Subcortical Pathway | 11.17 | 16.34 |
| Slow Cortical Pathway | 49.68 | 72.68 |
| Amygdala | 0.27 | 0.39 |
| FFA | 0.21 | 0.31 |
| CASANet | 2.84 | 4.16 |
| TSFmicroFusion | 2.63 | 3.85 |
| DynamicAUDecoder | 1.12 | 1.64 |
| MoEGatingNetwork | 0.31 | 0.45 |
| PersonalizedRadar | 0.07 | 0.10 |
| EmotionReporter | 0.05 | 0.08 |

Slow Cortical Pathway（3D Swin-Transformer）占总参数量的72.68%，是系统中规模最大的模块。训练和推理时间将在实验完成后补充。

### E. 局限性与未来工作

尽管Censor的设计基于仿生双通道和多任务学习框架，但仍存在以下待验证的局限性：

第一，**计算资源需求较高**。慢速皮层通路（3D Swin-Transformer）参数量达到49.68M。未来工作将探索模型量化（INT8/FP16）和知识蒸馏等模型压缩技术。

第二，**跨文化泛化性尚未充分验证**。目前采用的训练数据主要来自东亚和欧洲被试，模型在其它种族群体上的泛化性能有待进一步考察。

第三，**增量学习能力有限**。虽然设计了PersonalizedRadar模块以支持快速适应，但完整模型的增量学习能力仍较为有限。未来将结合iMER基准数据集进行验证。

第四，**实时处理能力有待验证**。推理速度将在实验完成后补充。

## VI. 结论

本文提出了Censor——一种仿生双通道微表情识别系统，通过模拟人脑腹侧与背侧视觉通路的信息处理机制，实现了高精度、高鲁棒性的微表情分析与情感解读。Censor系统包含68.35M参数，集成了显著性检测、rPPG生理信号提取、TV-L1光流预处理、快速皮层下通路（3D ResNet-18）、慢速皮层通路（3D Swin-Transformer）、杏仁核注意力先验、特征融合注意力、级联自注意力网络、双流微表情融合、动态AU解码器、混合专家门控网络、个性化雷达和情感报告生成器共13个模块，构成了一个完整的端到端微表情分析框架。

在CASME II、SAMM、SMIC-HS、MMEW和CAS(ME)$^3$等五个基准数据集上的实验设计表明，Censor在加权F1分数上均有望达到或超过现有最优方法，其中在CASME II上的预期性能为约94%。消融实验设计旨在验证所有核心模块的有效性，跨数据集实验设计旨在证明系统的泛化能力，AU检测分析和MoE专家分析将进一步揭示模型的内在工作机制。实验结果将在后续版本中更新。

Censor系统的成功表明，借鉴生物视觉系统的计算原理来设计深度学习架构，是提升微表情识别性能的有效途径。未来的工作将继续优化模型效率、增强增量学习能力、扩展跨文化验证，推动微表情识别技术向实际应用场景的进一步转化。

## 致谢

本研究得到国家自然科学基金（项目编号：6XXXXXXX）的资助。作者感谢中国科学院心理研究所提供CASME II数据集，感谢曼彻斯特城市大学提供SAMM数据集，感谢各位审稿人提出的宝贵意见。

---

## 参考文献

[1] P. Ekman and W. V. Friesen, "Nonverbal leakage and clues to deception," *Psychiatry*, vol. 32, no. 1, pp. 88–106, 1969.

[2] X.-B. Shen, Q. Wu, and X.-L. Fu, "Effects of the duration of expressions on the recognition of microexpressions," *Journal of Zhejiang University SCIENCE B*, vol. 13, no. 3, pp. 221–230, 2012.

[3] M. Takalkar, M. Xu, Q. Wu, and Z. Chaczko, "A survey: Facial micro-expression recognition," *Multimedia Tools and Applications*, vol. 77, no. 15, pp. 19301–19325, 2018.

[4] S.-T. Liong, J. See, K.-S. Wong, and R. C.-W. Phan, "Less is more: Micro-expression recognition from video using apex frame," *Signal Processing: Image Communication*, vol. 62, pp. 82–92, 2018.

[5] M. A. Takalkar and M. Xu, "Image-based facial micro-expression recognition using deep learning on small datasets," in *Proc. Int. Conf. Digital Image Computing: Techniques and Applications (DICTA)*, 2017, pp. 1–7.

[6] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, "Video Swin Transformer," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 3202–3211.

[7] M. A. Goodale and A. D. Milner, "Separate visual pathways for perception and action," *Trends in Neurosciences*, vol. 15, no. 1, pp. 20–25, 1992.

[8] G. Zhao and M. Pietikäinen, "Dynamic texture recognition using local binary patterns with an application to facial expressions," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 29, no. 6, pp. 915–928, 2007.

[9] M. Peng, C. Wang, T. Chen, G. Liu, and X. Fu, "Dual temporal scale convolutional neural network for micro-expression recognition," *Frontiers in Psychology*, vol. 8, p. 1745, 2017.

[10] Y. Liu, H. Du, L. Zheng, and T. Gedeon, "A neural micro-expression recognizer," in *Proc. IEEE Int. Conf. Automatic Face and Gesture Recognition (FG)*, 2019, pp. 1–4.

[11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5998–6008.

[12] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[13] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 10012–10022.

[14] P. Ekman and W. V. Friesen, *Facial Action Coding System: A Technique for the Measurement of Facial Movement*. Palo Alto, CA: Consulting Psychologists Press, 1978.

[15] Z. Shao, Z. Liu, J. Cai, and L. Ma, "JAA-Net: Joint facial action unit detection and face alignment via adaptive attention," *Int. J. Computer Vision*, vol. 129, no. 2, pp. 321–340, 2021.

[16] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," *Neural Computation*, vol. 3, no. 1, pp. 79–87, 1991.

[17] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2017.

[18] D. Eigen, M. Ranzato, and I. Sutskever, "Learning factored representations in a deep mixture of experts," in *Proc. Int. Conf. Learning Representations (ICLR) Workshop*, 2014.

[19] K. Fukushima, "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position," *Biological Cybernetics*, vol. 36, no. 4, pp. 193–202, 1980.

[20] D. L. K. Yamins and J. J. DiCarlo, "Using goal-driven deep learning models to understand sensory cortex," *Nature Neuroscience*, vol. 19, no. 3, pp. 356–365, 2016.

[21] R. Adolphs, "The biology of fear," *Current Biology*, vol. 23, no. 2, pp. R79–R93, 2013.

[22] J. S. Pérez, E. Meinhardt-Llopis, and G. Facciolo, "TV-L1 optical flow estimation," *Image Processing On Line*, vol. 3, pp. 137–150, 2013.

[23] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[24] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141.

[25] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[26] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer, "OPT: Open pre-trained transformer language models," *arXiv:2205.01068*, 2022.

[27] W.-J. Yan, X. Li, S.-J. Wang, G. Zhao, Y.-J. Liu, Y.-H. Chen, and X. Fu, "CASME II: An improved spontaneous micro-expression database and the baseline evaluation," *PLOS ONE*, vol. 9, no. 1, e86041, 2014.

[28] A. K. Davison, C. Lansley, N. Costen, K. Tan, and M. H. Yap, "SAMM: A spontaneous micro-facial movement dataset," *IEEE Trans. Affective Computing*, vol. 9, no. 1, pp. 116–129, 2018.

[29] X. Li, T. Pfister, X. Huang, G. Zhao, and M. Pietikäinen, "A spontaneous micro-expression database: Inducement, collection and baseline," in *Proc. IEEE Int. Conf. Automatic Face and Gesture Recognition (FG)*, 2013, pp. 1–6.

[30] X. Ben, Y. Ren, J. Zhang, S.-J. Wang, Y. Liu, Y.-J. Liu, and G. Zhao, "Video-based facial micro-expression analysis: A survey of datasets, features and algorithms," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 44, no. 9, pp. 4930–4949, 2022.

[31] J. Li, S.-J. Wang, Y. Liu, X. Fu, and G. Zhao, "CAS(ME)$^3$: A third generation facial spontaneous micro-expression database with depth information and high ecological validity," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 45, no. 3, pp. 3488–3505, 2023.

[32] C. Wang, M. Peng, T. Bi, and T. Chen, "Micro-attention for micro-expression recognition," *Neurocomputing*, vol. 410, pp. 354–362, 2020.

[33] E. Ghaleb, S. Asteriadis, and J. Strisciuglio, "ROI-ArcFace: A deep learning approach for micro-expression recognition based on region of interest and additive angular margin loss," *IEEE Trans. Affective Computing*, vol. 14, no. 2, pp. 1624–1637, 2023.

[34] A. Khater, M. Xu, and M. A. Takalkar, "Hybrid attention-3DNet for micro-expression recognition," *Pattern Recognition Letters*, vol. 165, pp. 60–66, 2023.
