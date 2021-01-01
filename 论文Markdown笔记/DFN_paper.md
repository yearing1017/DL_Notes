### 1. 前言

- 本文为**判别特征网络DFN论文的学习笔记**
- 原文链接：[**Learning a Discriminative Feature Network for Semantic Segmentation**](cn.arxiv.org/pdf/1804.09337.pdf)

- 文章提出的判别特征网络（DFN）包含两个子网络**SmoothNetwork** 和 **BorderNetwork**致力于**解决两个图像多类分割出现的问题：**
  - **类内不一致**
  - **类间无差别**

### 2. Abstract

- `Most existing methods of semantic segmentation still suffer from two aspects of challenges: intra-class inconsistency and inter-class indistinction.`

- **intra-class inconsistency（类内不一致）和inter-class indistinction（类间无差别）**

- `Speciﬁcally, to handle the intra-class inconsistency problem, we specially design a Smooth Network with Channel Attention Block and global average pooling to select the more discriminative features.`
- **Smooth Network带有Channel Attention Block（通道注意力模块，CAB）和全局平均池化可以选择更有判别力的特征**
- `Furthermore, we propose a Border Network to make the bilateral features of boundary distinguishable with deep semantic boundary supervision.`

- **Border Network借助多层语义边界监督区分边界两边的特征。**

### 3. Introduction

- **语义分割中的两个问题：**
  - `1) the patches which share the same semantic label but different appearances, named intra-class inconsistency as shown in the ﬁrst row of Figure 1`
  - `2) the two adjacent patches which have different semantic labels but with similar appearances, named inter-class indistinction as shown in the second row of Figure 1.`

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/17-1.jpg" style="zoom:50%;" />

- `To address these two challenges, we rethink the semantic segmentation task from a more macroscopic point of view.`
- `In this way, we regard the semantic segmentation as a task to assign a consistent semantic label to a category of things, rather than to each single pixel.`

- **论文认为语义分割是将相同的语义标签分配给一类事物，而不是分配对单个像素分配单个标签。**

- `Our DFN involves two components: Smooth Network and Border Network, as Figure 2 illustrates.`

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/17-2.jpg)

- `The Smooth Network is designed to address the intra-class inconsistency issue. To learn a robust feature representation for intra-class consistency, we usually consider two crucial factors.`
  - `On the one hand, we need multi-scale and global context features to encode the local and global information.`
  - **需要多尺度和全局特征用于学习局部和全局的信息**

  - `On the other hand, as multi-scale context is introduced, for a certain scale of thing, the features have different extent of discrimination, some of which may predict a false label. Therefore, it is necessary to select the discriminative and effective features.`
  - **因为多尺度特征的引入，对于同一事物，因为尺度的原因可能会产生错误的预测。故需要学习更具区分力的有效的特征。**

  - `Motivated by these two aspects, our Smooth Network is presented based on the U-shape  structure to capture the multi-scale context information, with the global average pooling to capture the global context. Also, we propose a Channel Attention Block (CAB), which utilizes the high-level features to guide the selection of low-level features stage-by-stage.`

- `Border Network, on the other hand, tries to differentiate the adjacent patches with similar appearances but different semantic labels.`

- **区分具有相似外表但不同语义标签的相邻区域**

  - `Consider the example in Figure 1(d), if more and more global context is integrated into the classiﬁciation process, the computer case next to the monitor can be easily misclassiﬁed as a monitor due to the similar appearance. Thus, it is signiﬁcant to explicitly involve the semantic boundary to guide the learning of the features.`
  - `In our Border Network, we integrate semantic boundary loss during the training process to learn the discriminative features to enlarge the “inter-class distinction”.`

### 4. Related Work

- **Encoder-Decoder：SegNet、U-net、LRR、ReﬁneNet**
  - `The FCN model has inherently encoded different levels of feature. Naturally, some methods integrate them to reﬁne the ﬁnal prediction. This branch of methods mainly consider how to recover the reduced spatial information caused by consecutive pooling operator or convolution with stride.`
  - `In addition, most methods of this type are just summed up the features of adjacent stages without consideration of their diverse representation. This leads to some inconsistent results.`
- **Global Context：ParseNet、PSPNet、Deeplab v3**
  - `Some modern methods have proven the effectiveness of global average pooling.`
  - `ParseNet [24] ﬁrstly applies global average pooling in the semantic segmentation task. Then PSPNet [40] and Deeplab v3 [6] respectively extend it to the Spatial Pyramid Pooling [13] and Atrous Spatial Pyramid Pooling [5], resulting in great performance in different benchmarks.`
  - `However, to take advantage of the pyramid pooling module sufﬁciently, these two methods adopt the base feature network to 8 times downsample with atrous convolution [5, 38], which is time-consuming and memory intensive.`
- **Attention Module:**
  - `Attention is helpful to focus on what we want.`
  - `In this work, we utilize channel attention to select the features similar to SENet`
- **Semantic Boundary Detection:**
  - `Most of these methods straightly concatenate the different level of features to extract the boundary. However, in this work, our goal is to obtain the features with inter-class distinction as much as possible with accurate boundary supervision.`
  - `Therefore, we design a bottom-up structure to optimize the features on each stage.`

### 5. Method

#### 5.1 Smooth network

- `The intra-class inconsistency problem is mainly due to the lack of context.`
- **intra-class inconsistency主要是因为缺乏上下文，因此，论文使用全局平均池化来引入全局上下文信息。**
- **全局上下文信息只是具备高级的语义信息，这对于恢复空间信息没有多大帮助。因此，我们需要多尺度的感受野帮助恢复空间信息，**而许多方法在这里存在另一个问题，**不同尺度的感受野产生的特征会有一些不同尺度的区分力，这会造成错误的结果。 **
- 为了解决这个问题，我们需要选择更具区分力的特征来产生一致的语义标签。**论文引入了注意力机制来解决。**
- `In our proposed network, we use ResNet [14] as a base recognition model. This model can be divided into ﬁve stages according to the size of the feature maps. According to our observation, the different stages have different recognition abilities resulting in diverse consistency manifestation.`
- **论文的backbone是Resnet，Resnet依据输出特征的大小分成了5个阶段。论文观察到：不同阶段有着不同的特征观察能力，这导致了不同的表征(manifestation)：**
  - `In the lower stage, the network encodes ﬁner spatial information, however, it has poor semantic consistency because of its small receptive view and without the guidance of spatial context.`
  - **lower stage：网络编码更多的空间信息。由于网络的感受野较小并且没有空间上的上下文信息的指导，语义一致性表现欠佳。**
  - `While in the high stage, it has strong semantic consistency due to large receptive view, however, the prediction is spatially coarse.`
  - **high stage：由于感受野较大，语义一致性表现较佳，但是预测的空间信息较粗糙。**
- **总体而言，低级阶段有着更精确的空间预测，而高级阶段有着更精确的语义预测。基于这一观察，本文提出 Smooth Network 以整合两者的优势，利用高级阶段的一致性指导低级阶段获得最优的预测。**

- **当下流行的语义分割架构主要有两种 style，一种是 Backbone，如 PSPNet 和 Deeplab v3；另一种是 Encoder-Decoder，比如 RefineNet 和全局卷积网络。但上述架构并不完备。**
- `To remedy the defect, we ﬁrst embed a global average pooling layer [24] to extend the U-shape architecture [27, 36] to a Vshape architecture. With the global average pooling layer, we introduce the strongest consistency constraint into the network as a guidance. Furthermore, to enhance consistency, we design a Channel Attention Block, as shown in Figure 2(c). This design combines the features of adjacent stages to compute a channel attention vector 3(b).The features of high stage provide a strong consistency guidance, while the features of low stage give the different discrimination information of features. In this way, the channel attention vector can select the discriminative features.`
- **为此，本文首先嵌入一个全局平均池化层把 U 形架构扩展为 V 形架构，为网络引入最强的一致性约束作为指导；此外，本文提出通道注意力模块以优化一致性，如图 2(c) 所示。该设计结合相邻阶段的特征以计算通道注意力向量（图 3(b)）。高级阶段的特征给出一个强大的一致性指导，而低级阶段的特征给出特征的不同判别信息，从而通道注意力向量可以选择判别特征。**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/17-3.jpg" style="zoom:50%;" />

- **Channel attention block:**

  - **CAB 的设计目的是改变每一阶段的特征权重以优化一致性，如图 3 所示。**
  - **FCN网络的过程及结果如下所示：**
  - `In the FCN architecture, the convolution operator outputs a score map, which gives the probability of each class at each pixel. In Equation 1, the ﬁnal score at score map is just summed over all channels of feature maps.`

  $$
  y_{k}=F(x ; w)=\sum_{i=1, j=1}^{D} w_{i, j} x_{i, j}
  $$

  - `where x is the output feature of network. w represents the convolution kernel. And k ∈ {1, 2, . . . , K}. K is the number of channels. D is the set of pixel positions.`

  $$
  \delta_{i}\left(y_{k}\right)=\frac{\exp \left(y_{k}\right)}{\sum_{j=1}^{K} \exp \left(y_{j}\right)}
  $$

  - `where δ is the prediction probability. y is the output of network.`
  - `As shown in Equation 1 and Equation 2, the ﬁnal predicted label is the category with highest probability. Therefore, we assume that the prediction result is y 0 of a certain patch, while its true label is y 1 .`
  - **如上句，针对以下情况：真实label为$y_1$的预测结果为$y_0$，本文引入$\alpha$ 来进行转化**
  - `Consequently, we can introduce a parameter α to change the highest probability value from y 0 to y 1 , as Equation 3 shows.`

  $$
  \bar{y}=\alpha y=\left[\begin{array}{c}\alpha_{1} \\ \vdots \\ \alpha_{K}\end{array}\right] \cdot\left[\begin{array}{c}y_{1} \\ \vdots \\ y_{K}\end{array}\right]=\left[\begin{array}{c}\alpha_{1} w_{1} \\ \vdots \\ \alpha_{K} w_{K}\end{array}\right] \times\left[\begin{array}{c}x_{1} \\ \vdots \\ x_{K}\end{array}\right]
  $$

  - `where ¯y is the new prediction of network and α = Sigmoid(x; w)`
  - `In Equation 1, it implicitly indicates that the weights of different channels are equal.`
  - **然而，如上所述，不同阶段的特征判别力不同，造成预测的一致性各不相同。为实现类内一致预测，应该提取判别特征，并抑制非判别特征，如上公式所示，应用到特征图x上的$\alpha$ 代表了CAB模块的特征选取。从而可以逐阶段地获取判别特征以实现预测类内一致。**
  - `Therefore, in Equation 3, the α value applies on the feature maps x, which represents the feature selection with CAB. With this design, we can make the network to obtain discriminative features stage-wise to make the prediction intra-class consistent.`

- **Reﬁnement residual block:**
  - **特征网络中每一阶段的特征图全都经过 RRB，如图 2(b) 所示。该模块的第 1 个组件是 1 x 1 卷积层，作者用它把通道数量统一为 512。同时，它可以整合所有通道的信息。接着是一个基本的残差模块，它可以优化特征图。此外，受 ResNet 启发，该模块还可以强化每一阶段的识别能力。**

#### 5.2 Border network

- 在语义分割任务中，预测经常混淆外观相似的不同类别，尤其当它们在空间上相近之时，因此需要加大特征的差别。出于这一考虑，**本文采用语义边界指导特征学习，同时应用显式监督提取精确的语义边界，使网络学习类间差别能力强大的特征，进而提出 Border Network 加大特征的类间差别。**
- Border Network 直接通过显式语义边界监督学习语义边界，类似于语义边界检测任务。这使得语义边界两边的特征变得可区分。
- `In our work, we need semantic boundary with more semantic meanings. Therefore, we design a bottom-up Border Network. This network can simultaneously get accurate edge information from low stage and obtain semantic information from high stage, which eliminates some original edges lack of semantic information.`

- **本文的工作需要语义边界具有更多的语义含义。因此 Border Network 的设计是自下而上的。它可以同时从低级阶段获取精确的边界信息和从高级阶段获取语义信息，从而消除一些缺乏语义信息的原始边界。**

- **由此，高级阶段的语义信息可以逐阶段地优化低级阶段的细节边界信息。借助传统的图像处理方法，作者可以从语义分割的 groundtruth 中获得网络的监督信号。**
- `To remedy the imbalance of the positive and negative samples, we use focal loss [22] to supervise the output of the Border Network, as shown in Equation 4. We adjust the parameters α and γ of focal loss for better performance.`

$$
F L\left(p_{k}\right)=-\left(1-p_{k}\right)^{\gamma} \log p_{k}
$$

- `where p k is the estimated probability for class k, k ∈ {1, 2, . . . , K}. And K is the maximum value of class label.`

- **Border Network 主要关注分离边界两边的类别的语义分割。要精确地提取语义边界，需要两边的特征更加可区分，而这正是作者的目的所在。**

#### 5.3 Network Architecture

- **作者使用预训练的 ResNet 作为基础网络。Smooth Network 通过在网络顶部添加全局平均池化层以获得最强的一致性；接着利用 CAB 改变通道的权重进一步提升一致性。同时，Border Network 通过明确的语义边界监督获得精确的语义边界并使两边的特征更易区分。由此，类内特征更加一致，类间特征更易区分。**

- **对于显式的特征优化，需要使用多层监督以获取更佳性能，同时网络也更容易训练。Smooth Network 借助 softmax loss 监督每一阶段的上采样输出（全局平均池化层除外），而本文借助 focal loss 监督 Border Network 的输出。两个子网络在一起联合训练，其 loss 通过一个参数控制两者的权重。**

$$
\begin{aligned} \ell_{s} &=\text {SoftmaxLoss}(y ; w) \\ \ell_{b} &=\text {FocalLoss}(y ; w) \\ L &=\ell_{s}+\lambda \ell_{b} \end{aligned}
$$

### 6. 实验训练细节及结果略

### 7. 思考

- **可以考虑在网络中加入CAB模块**
- **BorderNetwork需要显式监督提取边界信息，目前不知如何实现**
- **重新思考数据增强，考虑加进去随机裁剪**

