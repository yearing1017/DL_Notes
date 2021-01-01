### 1. 前言

- 本文是阅读的第二篇**涉及到Gate来融合low-level特征的语义分割的文章**
- 之前读的一篇论文：[Gated-SCNN: Gated Shape CNNs for Semantic Segmentation](http://xxx.itp.ac.cn/pdf/1907.05740)
- 本文的原文地址：[GFF: Gated Fully Fusion for Semantic Segmentation](https://arxiv.org/pdf/1907.05740.pdf)

### 2. Abstract

- 本文从**融合低级特征与高级特征问题出发**

> It is natural to consider importing low level features to compensate the lost detailed information in high level representations. Unfortunately, simply combining multi-level features is less effective due to the semantic gap existing among them.

- **提出本文的的网络结构**

> In this paper, we propose a new architecture, named Gated Fully Fusion(GFF), to selectively fuse features from multiple levels using gates in a fully connected way. Speciﬁcally, features at each level are enhanced by higher-level features with stronger semantics and lower-level features with more details, and gates are used to control the propagation of useful information which signiﬁcantly reduces the noises during fusion.

### 3. Introduction

- 论文先放出了一张图解释了**分割中由于物体尺度大小、远近灯因素导致的预测不准确的问题**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/33-1.jpg" style="zoom:50%;" />

- 提出本文的GFF:

> we propose Gated Fully Fusion (GFF) which uses a gate, a kind of operation commonly used in recurrent networks, at each pixel to measure the usefulness of each corresponding feature vector, and thus to control the propagation of the information through the gate.

- **我们提出了GFF 算法，该算法使用一个门控(一种递归网络中常用的操作)在每个像素点上测量每个对应特征向量的有用性，从而控制信息通过门控的传播。**

> On the other hand, considering contextual information in large receptive ﬁeld is also very important for semantic segmentation as proved by PSPNet [44], ASPP [3] and DenseASPP [37]. Therefore, we also model contextual information after GFF to achieve further performance improvement. Speciﬁcally, we propose a dense feature pyramid (DFP) module to encode the context information into each feature map. DFP reuses the contextual information for each feature level and aims to enhance the context modeling part while GFF operates on the backbone of network to capture more detailed information. Combining both components in a single end-to-end network, we achieve the state-of-the-art results on both Cityscapes and ADE20K datasets.

- 另一方面，PSPNet、ASPP 和 DenseASPP 证明，考虑大感受野的上下文信息对语义分割也非常重要。因此，我们也对GFF后的上下文信息进行建模，以达到进一步的性能提升。
- 具体来说，我们提出了一个**密集的特征金字塔(DFP)模块**来将上下文信息编码到每个特征图中。DFP为每个特征层重用上下文信息，旨在增强上下文建模部分，而**GFF则在网络的主干上运行**，以捕获更详细的信息。在一个单一的端到端网络中结合这两个组件，我们在城市景观和ADE20K数据集上都取得了最先进的结果。

- 论文贡献如下：

> The main contributions of our work can be summarized as follows:
>
> • Gated Fully Fusion is proposed to generate high-resolution and high-level feature map from multi-level feature maps.
>
> • Detailed analysis with visualization of gates learned in different layers intuitively shows the information regulation mechanism in GFF.
>
> • The proposed method is extensively veriﬁed on two standard semantic segmentation benchmarks including Cityscapes and ADE20K, and achieves new state-of-the-art performance. In particular, our model achieves 82.3% mIoU on Cityscapes test set trained only on the ﬁne labeled data.

### 4. Method

#### 4.1 Multi-level Feature Fusion

- 论文中指出**高级特征图中有丰富的语义，但是较低的分辨率，而低级特征图正好相反**

> feature maps of higher levels are with lower resolution due to the downsampling operations,
>
> In semantic segmentation, the top feature map X L with 1/8 resolution of the raw input image is mostly used for its rich semantics.
>
> In contrast, feature maps of low level from shallow layers are with high resolution, but with limited semantics.

- 论文给出了几种**多尺度特征融合的方式**：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/33-2.jpg)

- 论文指出了**concat、add、FPN三种融合方式的比较**

> Concatenation is a straightforward operation to aggregate all the information in multiple feature maps, but it mixes the useful information with large amount of noninformative features. 
>
> Addition is another simple way to combine feature maps by adding features at each position, while it suffers from the similar problem as concatenation. 
>
> FPN [23] conducts the fusion process through a top-down pathway with lateral connections, where semantic features in higher levels are gradually fused into lower levels. The three fusion strategies can be formulated as,

- 三种的融合公式大致如下：

$$
\begin{aligned} \text { Concat: } \tilde{X}_{l} &=\operatorname{concat}\left(X_{1}, \ldots, X_{L}\right) \\ \text { Addition: } \tilde{X}_{l} &=\sum_{i=1}^{L} X_{i} \\ \text { FPN: } \tilde{X}_{l} &=\tilde{X}_{l+1}+X_{l}, \text { where } \tilde{X}_{L}=X_{L} \end{aligned}
$$

- 又指出了**上面融合方式的存在问题**

> The problem of these basic fusion strategies is that feature maps are fused together without measuring the usefulness of each feature vector, and massive useless features are mixed with useful feature during fusion.

- 这些基本融合策略的问题在于，在融合过程中，**没有度量每个特征向量的有用性，大量无用的特征与有用的特征混合在一起。**

#### 4.2 Gated Fully Fusion

> The basic task in multi-level feature fusion is to aggregate useful information together under interference of massive useless information. Gating is a mature mechanism to measure the usefulness of each feature vector in a feature map and aggregates information accordingly.

- 多级特征融合的基本任务是在大量无用信息干扰下将有用信息聚合在一起。门机制的作用就是**在feature map中测量每个feature vector的有用性并相应地聚合信息。**

- 论文原文指出的GFF的实现如下：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/33-3.jpg" style="zoom:50%;" />

#### 4.3 Dense Feature Pyramid

> Context modeling aims to encode more global information, and it is orthogonal to the proposed GFF. Therefore, we further design a module to encode more contextual information from outputs of both PSPNet [44] and GFF. 
>
> Considering dense connections can strengthen feature propagation [13, 37], we also densely connect the feature maps in a top-down manner starting from feature map outputted from the PSPNet, and high-level feature maps are reused multiple times to add more contextual information to low levels, which was found important in our experiments for correctly segmenting large objects. 
>
> Since the feature pyramid is in a densely connected manner, we denote this module as Dense Feature Pyramid (DFP). Both GFF and DFP can be plugged into any existing FCNs for end-to-end training with only slightly extra computation cost.

#### 4.4 Network Architecture

> Our network architecture is designed based on previous state-of-the-art network PSPNet [44] with ResNet [11] as backbone for basic feature extraction, the last two stages in ResNet are modiﬁed with dilated convolution to make both strides to 1 and keep spatial information. Fig 3 shows the overall framework including both GFF and DFP.

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/33-4.jpg)

> PSPNet forms the bottom-up pathway with backbone network and pyramid pooling module (PPM), where PPM is at the top to encode contextual information. Feature maps from last residual blocks in each stage of backbone are used as the input for GFF module, and all feature maps are reduced to 256 channels with 1 × 1 convolutional layers. The output feature maps from GFF are further fused with two 3×3 convolutional layers in each level before feeding into the DFP module. All convolutional layers are followed by batch normalization [14] and ReLU activation function. After DFP, all feature maps are concatenated for ﬁnal semantic segmentation.
>

>Comparing with the basic PSPNet, the proposed method only slightly increases the number of parameters and computations. The entire network is trained in an endto-end manner driving by cross-entropy loss deﬁned on the segmentation benchmarks. To facilitate the training process, an auxiliary loss together with the main loss are used to help optimization following [19, 44], where the main loss is deﬁned on the ﬁnal output of the network and the auxiliary loss is deﬁned on the output feature map at stage3 of ResNet with weight of 0.4 [44].

### 5. Experiment

#### 5.1 Implementation Details

> Our implementation is based on PyTorch [26], and uses ResNet50 and ResNet101 pre-trained from ImageNet [29] as backbones.

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/33-5.jpg" style="zoom:50%;" />

- 有关详细实验结果见论文

### 6. Conclusion

> In this work, we propose Gated Fully Fusion (GFF) to fully fuse multi-level feature maps controlled by learned gate maps. The novel module bridges the gap between high resolution with low semantics and low resolution with high semantics. We explore the proposed GFF for the task of semantic segmentation and achieve new state-of-the-art results on Cityscapes and ADE20K datasets. In particular, we ﬁnd that the missing low-level features can be fused into each feature level in the pyramid, which indicates that our module can well handle small and thin objects in the scene. In our future work, we will verify the effectiveness of GFF in object detection tasks where ﬁne details are also important.
>