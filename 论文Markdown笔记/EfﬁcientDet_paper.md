### 1. 前言

- 本文为论文[EfﬁcientDet: Scalable and Efﬁcient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)的阅读笔记
- 最近在kaggle的目标检测比赛中用到了该网络，好好研读一下该论文

### 2. Abstract

> In this paper, we systematically study various neural network architecture design choices for object detection and propose several key optimizations to improve efﬁciency. First, we propose a weighted bi-directional feature pyramid network (BiFPN), which allows easy and fast multi-scale feature fusion;

- 本文系统地研究了用于目标检测的各种神经网络结构的设计选择，并提出了几个关键的优化方案以提高效率。**首先，我们提出了一种加权双向特征金字塔网络(BiFPN)，它可以方便快速地实现多尺度特征融合;**

> Second, we propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time.

- 其次，我们提出了一种**复合尺度变换方法**，该方法对所有主干、特征网络和box/class预测网络同时进行了分辨率、深度和宽度的均匀缩放。

> Based on these optimizations, we have developed a new family of object detectors, called EfﬁcientDet, which consistently achieve an order-of-magnitude better efﬁciency than prior art across a wide spectrum of resource constraints. In particular, without bells and whistles, our EfﬁcientDet-D7 achieves state-of-the-art 51.0 mAP on COCO dataset with 52M parameters and 326B FLOPS 1 , being 4x smaller and using 9.3x fewer FLOPS yet still more accurate (+0.3% mAP) than the best previous detector.

- 在这些优化的基础上，我们**开发了一个新的目标检测器系列，称为Efficiencydet**，它在广泛的资源约束条件下，始终能够达到比现有技术更好的数量级的效率。特别地，我们的Efficiencydet - d7在没有bells and whistles的情况下，在COCO dataset上实现了最先进的51.0 mAP，参数为52M, FLOPS 1为326B，比以前最好的检测器小4倍，少用9倍的FLOPS，但仍然比以前最好的检测器更精确(+0.3% mAP)。

### 3. Introduction

- **相要解决的问题**

> A natural question is: Is it possible to build a scalable detection architecture with both higher accuracy and better efﬁciency across a wide spectrum of resource constraints (e.g., from 3B to 300B FLOPS)?

- 一个自然的问题是:有没有可能**在广泛的资源限制范围内(比如从3B到300B)构建一个具有更高精确度和更高效率的可伸缩检测架构？**通俗一点就是**模型能不能扩展，在大机器上用大模型，小机器上用小模型**

- **两大挑战**

> Challenge 1: efﬁcient multi-scale feature fusion
>
> FPN has been widely used for multiscale feature fusion. Recently, PANet [19], NAS-FPN [5], and other studies [13, 12, 34] have developed more network structures for cross-scale feature fusion. While fusing different input features, most previous works simply sum them up without distinction; however, since these different input features are at different resolutions, we observe they usually contribute to the fused output feature unequally. To address this issue, we propose a simple yet highly effective weighted bi-directional feature pyramid network (BiFPN), which introduces learnable weights to learn the importance of different input features, while repeatedly applying top-down and bottom-up multi-scale feature fusion.

- PANet、NAS-FPN等研究开发了更多跨尺度特征融合的网络结构。**在融合不同的输入特性时，以往的作品大多是简单的归纳，没有区别；**
- 然而，**由于这些不同的输入特征在不同的分辨率下，我们观察到它们对融合输出特征的贡献是不平等的**。
- 为了解决这一问题，**我们提出了一种简单而高效的加权双向特征金字塔网络(BiFPN)，该网络在反复应用自顶向下和自底向上的多尺度特征融合的同时，引入可学习的权值来学习不同输入特征的重要性。**

> Challenge 2: model scaling 
>
>  While previous works mainly rely on bigger backbone networks [17, 27, 26, 5] or larger input image sizes [8, 37] for higher accuracy, we observe that scaling up feature network and box/class prediction network is also critical when taking into account both accuracy and efﬁciency. Inspired by recent works [31], we propose a compound scaling method for object detectors, which jointly scales up the resolution/depth/width for all backbone, feature network, box/class prediction network.

- 挑战2：**模型的尺度变换**
- 虽然之前的工作主要依靠更大的骨干网络或更大的输入图像尺寸来获得更高的精度，但我们观察到，在考虑精度和效率时，**scale up feature网络和box/class预测网络也是至关重要的**。受最近工作的启发，我们提出了一种用于目标检测的复合缩放方法，联合缩放所有主干、特征网络、box/class预测网络的分辨率/深度/宽度。

> Finally, we also observe that the recently introduced EfﬁcientNets [31] achieve better efﬁciency than previous commonly used backbones (e.g., ResNets [9], ResNeXt [33], and AmoebaNet [24]). Combining EfﬁcientNet backbones with our propose BiFPN and compound scaling, we have developed a new family of object detectors, named EfﬁcientDet,

- 最后，我们还观察到，最近引入的EfficientNets比以前常用的主干(如ResNets[9]、ResNeXt[33]和AmoebaNet[24])具有更好的效率。**将EfficiencyNet的骨干与我们提出的BiFPN和复合缩放相结合，我们开发了一个名为EfficiencyDet的新系列的目标检测器**

### 4. BiFPN

> In this section, we ﬁrst formulate the multi-scale feature fusion problem, and then introduce the two main ideas for our proposed BiFPN: efﬁcient bidirectional cross-scale connections and weighted feature fusion.

- 在本节中，我们首先阐述了**多尺度特征融合问题**，然后介绍了我们提出的BiFPN的两个主要思想:**有效的双向跨尺度连接和加权特征融合。**

#### 4.1 Problem Formulation

- 要说清BiFPN首先要说FPN，论文提供了一张图介绍了**6种多尺度特征融合的方式**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/31-1.jpg)

> Multi-scale feature fusion aims to aggregate features at different resolutions.

- **多尺度特征融合是将不同分辨率下的特征进行聚合**。

- 给定一组多尺度特征$\overrightarrow{P^{i n}}=\left(P_{l_{1}}^{i n}, P_{l_{2}}^{i n}, \cdots\right)$，其中 $P_{l_{i}}^{i n}$ 表示 $l_i$ 等级的特征，目标是要找到一个变换 f 可以高效的整合不同特征输出一个新的特征列表 $\overrightarrow{P^{o u t}}=f(\overrightarrow{P^{i n}})$

- 例如，3~7级，输入特征为$\overrightarrow{P^{i n}}=\left(P_{3}^{i n}, \cdots P_{7}^{i n}\right)$，其中 $P^{in}_{i}$ 表示特征分辨率为输入图像的 $1/2^i$，如果输入分辨率是640x640，那么 $P^{in}_{3}$ 的分辨率只有640/8 = 80。
- **FPN的top-down方法**可以表示为：

$$
\begin{aligned} P_{7}^{o u t} &=\operatorname{Conv}\left(P_{7}^{i n}\right) \\ P_{6}^{o u t} &=\operatorname{Conv}\left(P_{6}^{i n}+\operatorname{Resize}\left(P_{7}^{o u t}\right)\right) \\ \cdots & \\ P_{3}^{o u t} &=\operatorname{Conv}\left(P_{3}^{i n}+\operatorname{Resize}\left(P_{4}^{o u t}\right)\right) \end{aligned}
$$

#### 4.2 Cross-Scale Connections

> Conventional top-down FPN is inherently limited by the one-way information ﬂow. To address this issue, PANet [19] adds an extra bottom-up path aggregation network, as shown in Figure 2(b).

- 如上图2(b)所示，PANet针对FPN做的改进：**在单向信息流的基础上添加了自底向上的信息聚合**

> Recently, NAS-FPN [5] employs neural architecture search to search for better cross-scale feature network topology, but it requires thousands of GPU hours during search and the found network is irregular and difﬁcult to interpret or modify, as shown in Figure 2(c).

- 最近，NAS-FPN[5]使用神经架构搜索来搜索更好的跨尺度特征网络拓扑，但是在搜索过程中需要数千个GPU每小时，发现的网络不规则，难以解释或修改，如图2(c)所示。

- **BiFPN**是对PANet的改动，提出了三个优化：

  > First, we remove those nodes that only have one input edge. Our intuition is simple: if a node has only one input edge with no feature fusion, then it will have less contribution to feature network that aims at fusing different features. This leads to a simpliﬁed PANet as shown in Figure 2(e)

  - **删掉那些只有一个输入边的节点，因为只有一个输入边的节点就没有特征融合**，对特征网络的贡献少；改进如图二(e)所示，**简化版的PANet结构**

  > Second, we add an extra edge from the original input to output node if they are at the same level, in order to fuse more features without adding much cost, as shown in Figure 2(f)

  - 如果**原始输入节点与输出节点处于同一级别，我们就从原始输入节点添加一条额外的边到输出节点**，以便在不增加太多成本的情况下融合更多的特性，如图2(f)所示

  > Third, unlike PANet [19] that only has one top-down and one bottom-up path, we treat each bidirectional (top-down & bottom-up) path as one feature network layer, and repeat the same layer multiple times to enable more high-level feature fusion. Section 4.2 will discuss how to determine the number of layers for different resource constraints using a compound scaling method. With these optimizations, we name the new feature network as bidirectional feature pyramid network (BiFPN), as shown in Figure 2(f) and 3.

  - 与只有一个自顶向下和一个自底向上路径的PANet[19]不同，**我们将每个双向(自顶向下和自底向上)路径视为一个特性网络层，并多次重复同一层以实现更高级的特性融合**。4.2节将讨论如何使用复合缩放方法确定不同资源约束的层数。通过这些优化，我们将新的特征网络命名为双向特征金字塔网络(BiFPN)，如图2(f)和3所示。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/31-2.jpg)

#### 4.3 Weighted Feature Fusion

- 论文提出三种加权策略，最终选择了 $O=\sum_{i} \frac{w_{i}}{\epsilon+\sum_{j} w_{j}} I_{i}$ ，为了确保 $w_i>0$ 应用了ReLU函数，$\epsilon = 0.0001$，加权策略可以举例表示为：

$$
\begin{aligned} & \mathrm{P}_{6}^{t d}=\operatorname{Conv}\left(\frac{\mathrm{w}_{1} \cdot \mathrm{P}_{6}^{i n}+\mathrm{w}_{2} \cdot \operatorname{Resize}\left(\mathrm{P}_{7}^{i n}\right)}{\mathrm{w}_{1}+w_{2}+\epsilon}\right) \\ \mathrm{P}_{6}^{o u t}=& \operatorname{Conv}\left(\frac{\mathrm{w}_{1}^{\prime} \cdot \mathrm{P}_{6}^{i n}+\mathrm{w}_{2}^{\prime} \cdot \mathrm{P}_{6}^{t d}+\mathrm{w}_{3}^{\prime} \cdot \operatorname{Resize}\left(\mathrm{P}_{5}^{\text {out}}\right)}{\mathrm{w}_{1}^{\prime}+\mathrm{w}_{2}^{\prime}+\mathrm{w}_{3}^{\prime}+\epsilon}\right) \end{aligned}
$$

### 5. EfﬁcientDet

#### 5.1 EfﬁcientDet Architecture

> Figure 3 shows the overall architecture of EfﬁcientDet, which largely follows the one-stage detectors paradigm [20, 25, 16, 17]. We employ ImageNet-pretrained EfﬁcientNets as the backbone network. Our proposed BiFPN serves as the feature network, which takes level 3-7 features {P 3 , P 4 , P 5 , P 6 , P 7 } from the backbone network and repeatedly applies top-down and bottom-up bidirectional feature fusion. These fused features are fed to a class and box network to produce object class and bounding box predictions respectively. Similar to [17], the class and box network weights are shared across all levels of features.

- 图3显示了EfficiencyDet的总体架构，它很大程度上遵循了one-stage检测器的模式[20,25,16,17]。我们采用imagenet预训练的EfficientNets作为骨干网络。
- **我们提出的BiFPN作为特征网络，从骨干网中提取3-7级特征{p3, p4, p5, p6, p7}，反复应用自顶向下和自底向上的双向特征融合。将融合后的特征分别输入class网络和box网络**，分别生成目标类预测和边界盒预测。与[17]类似，class和box网络权重在所有级别的特性中共享。

#### 5.2 Compound Scaling

> Previous works mostly scale up a baseline detector by employing bigger backbone networks (e.g., ResNeXt [33] or AmoebaNet [24]), using larger input images, or stacking more FPN layers [5]. These methods are usually ineffective since they only focus on a single or limited scaling dimensions.

- 之前的工作主要是通过使用更大的骨干网络(例如ResNeXt或AmoebaNet)，使用更大的输入图像，或堆叠更多的FPN层来扩大基线检测器。这些方法通常是无效的，因为它们只关注单个或有限的scaling维度。

> Recent work [31] shows remarkable performance on image classiﬁcation by jointly scaling up all dimensions of network width, depth, and input resolution. Inspired by these works [5, 31], we propose a new compound scaling method for object detection, which uses a simple compound coefﬁcient φ to jointly scale up all dimensions of backbone network, BiFPN network, class/box network, and resolution.

- 论文提出用一个复合系数  同时对骨干网络、BiFPN网络、class/box网络和分辨率的所有维度尺度扩展。

> Backbone network – we reuse the same width/depth scaling coefﬁcients of EfﬁcientNet-B0 to B6 [31] such that we can easily reuse their ImageNet-pretrained checkpoints.

- 对于**骨干网络**，重用了EfficientNetB0-B6的深度/通道数的尺寸数

- 对于**BiFPN网络**，通道数使用指数增长，深度（层数）使用线性增长:

$$
W_{b i f p n}=64 \cdot\left(1.35^{\phi}\right), D_{b i f p n}=2+\phi
$$

- **box/class预测网络**，通道数与BiFPN相同，但是深度使用

$$
D_{b o x}=D_{c l a s s}+3+\lfloor\phi / 3\rfloor
$$

- 输入图像的分辨率，因为BiFPN使用3~7级的特征，输入分辨率要除以128，所以使用线性增长的分辨率

$$
R_{\text {input}}=512+\phi \cdot 128
$$

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/31-3.jpg" style="zoom:50%;" />

- 论文给出的在网络在coco数据上的表现：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/31-4.jpg)

### 6. Experiments

- 有关实验细节详见论文

### 7. Conclusion

> In this paper, we systematically study various network architecture design choices for efﬁcient object detection, and propose a weighted bidirectional feature network and a customized compound scaling method, in order to improve accuracy and efﬁciency. Based on these optimizations, we have developed a new family of detectors, named EfﬁcientDet, which consistently achieve better accuracy and efﬁciency than the prior art across a wide spectrum of resource constraints. In particular, our EfﬁcientDet-D7 achieves state-of-the-art accuracy with an order-of-magnitude fewer parameters and FLOPS than the best existing detector. Our EfﬁcientDet is also up to 3.2x faster on GPUs and 8.1x faster on CPUs. Code will be made public.

- 本文系统地研究了有效检测目标的各种网络架构设计选择，提出了一种加权的双向特征网络和自定义的compound scaling方法，以提高检测的精度和效率。在这些优化的基础上，我们开发了一个新的探测器系列，名为EfficientDet，它在广泛的资源约束范围内始终比现有技术获得更好的准确性和效率。