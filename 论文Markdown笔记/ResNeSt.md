### 1. 前言

- 论文链接：[ResNeSt: Split-Attention Networks]([https://hangzhang.org/files/resnest.pdf](https://link.zhihu.com/?target=https%3A//hangzhang.org/files/resnest.pdf))
- 这篇文章是在ResNet基础上的工作，**融合了GoogleNet的Multi-path和SENet、SKNet中的attention思想**，将ResNeSt用作分类、分割、目标检测的backbone，大大提升了任务的性能。
- 作者给的代码链接：[GitHub](https://github.com/zhanghang1989/ResNeSt)

### 2. Abstract

- `We present a modular Split-Attention block that enables attention across feature-map groups. By stacking these Split-Attention blocks ResNet-style, we obtain a new ResNet variant which we call ResNeSt.`
- 在公开实验数据集上表现如下：
  - ResNeSt-50 在 ImageNet 上实现了81.13％ top-1 准确率
  - 用ResNeSt-50替换ResNet-50，可以将MS-COCO上的Faster R-CNN的mAP从39.25％提高到42.33％
  - 用ResNeSt-50替换ResNet-50，可以将ADE20K上的DeeplabV3的mIoU从42.1％提高到45.1％

### 3. Introduction

- `Recent work has signiﬁcantly boosted image classiﬁcation accuracy through large scale neural architecture search (NAS) [45, 55]. Despite their state-of-the-art performance, these NAS-derived models are usually not optimized for training eﬃciency or memory usage on general/commercial processing hardware (CPU/GPU)`
- **NAS-derived的模型在计算量和内存消耗上都比较大**
- `However, since ResNet models are originally designed for image classiﬁcation, they may not be suitable for various downstream applications because of the limited receptive-ﬁeld size and lack of cross-channel interaction.`
- **ResNet模型最初是为图像分类而设计的，由于感受野的有限性和缺乏跨通道的交互作用，它们可能不适合各种下游应用**
- `This means that boosting performance on a given computer vision task requires “network surgery” to modify the ResNet to be more eﬀective for that particular task.`
- **需要修改网络本身来提升特定任务的表现，例如之前的工作：**
- `For example, some methods add a pyramid module [8,69] or introduce long-range connections [56] or use cross-channel feature-map attention [15, 65].`

- `As the ﬁrst contribution of this paper, we explore a simple architectural modiﬁcation of the ResNet [23], incorporating feature-map split attention within the individual network blocks.`
- `More speciﬁcally, each of our blocks divides the feature-map into several groups (along the channel dimension) and ﬁner-grained subgroups or splits, where the feature representation of each group is determined via a weighted combination of the representations of its splits (with weights chosen based on global contextual information)`
- **在本文中，ResNeSt中每个块将特征图沿着channel维度划分为几个组（groups）和更细粒度的子组（splits），每个组的特征表示是由其splits的表示的加权组合来确定的（根据全局上下文信息来确定权重），将得到的这个单元称之为 Split-Attention block.**

- `By stacking several Split-Attention blocks, we create a ResNet-like network called ResNeSt (S stands for “split”)`
- 论文的第二个贡献：`The second contributions of this paper are large scale benchmarks on image classiﬁcation and transfer learning applications.`

### 4. Related work

- 论文的相关工作部分：分4个点讲述了本文的技术思路来源

#### 4.1 Modern CNN Architectures

- `ResNet [23] introduces an identity skip connection which alleviates the diﬃculty of vanishing gradient in deep neural network and allows network learning deeper feature representations. ResNet has become one of the most successful CNN architectures which has been adopted in various computer vision applications.`

#### 4.2 Multi-path and Feature-map Attention

- `Multi-path representation has shown success in GoogleNet [52], in which each network block consists of different convolutional kernels.`
- **GoogleNet：Multi-path表示**
- `ResNeXt [61] adopts group convolution [34] in the ResNet bottle block, which converts the multi-path structure into a uniﬁed operation.`

- **ResNeXt：group Convolution**

- `SE-Net [29] introduces a channel-attention mechanism by adaptively recalibrating the channel feature responses.`
- **SENet通过自适应地重新校准channel特征响应，引入了channel注意力机制**

- `SK-Net [38] brings the feature-map attention across two network branches`
- **SKNet通过两个网络分支引入了feature-map 注意力机制**

#### 4.3 Neural Architecture Search

- `Recent neural architecture search algorithms have adaptively produced CNN architectures that achieved state-of-the-art classiﬁcation performance, such as: AmoebaNet [45], MNASNet [54], and EﬃcientNet [55].`

### 5. Split-Attention Networks

#### 5.1 Split-Attention Block

- `Our Split-Attention block is a computational unit, consisting feature-map group and split attention operations. Figure 1 (Right) depicts an overview of a SplitAttention Block.`
- 论文中给的**ResNeSt 模块**图示：可以看到下方注释，此图展示的是**cardinality-major view，方便观看了解**，在真正的实现中使用的是**radix-major view，使用标准的CNN进行加速**。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/27-2.png)

#### Feature-map Group

- `As in ResNeXt blocks [61], the input feature-map can be divided into several groups along the channel dimension, and the number of feature-map groups is given by a cardinality hyperparameter K. We refer to the resulting feature-map groups as cardinal groups`
- **借鉴ResNeXt的想法，将feature map划分为几个不同的组， feature-map group的数量通过引入一个 cardinality 超参 K给定，每个划分的组称之为cardinal groups**

- `We introduce a new radix hyperparameter R that dictates the number of splits within a cardinal group. Then the block input X are split into G = KR groups along the channel dimension X = {X 1 , X 2 , ...X G } as shown in Figure 1.`
- **引入一个新的radix超参R来表示一个cardinal group的split数，所以总的特征组的个数是 $G = KR$，如上图所示，输入x将被分为G份。**

#### Split Attention in Cardinal Groups

- 论文中给的**Split Attention模块**图示：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/27-4.png" style="zoom:50%;" />

- `a combined representation for each cardinal group can be obtained by fusing via an element-wise summation across multiple splits.`
- **一个cardinal group的组合表示可以通过多个splits按元素求和进行融合来得到，第k个cardinal group表示为：**

$$
\hat{U}^{k}=\sum_{j=R(k-1)+1}^{R k} U_{j} ，其中：\hat{U}^{k} \in \mathbb{R}^{H \times W \times C / K} \text { for } k \in 1,2, \ldots K
$$

- `Global contextual information with embedded channel-wise statistics can be gathered with global average pooling across spatial dimensions `$s^k \in R^{C/K}$
- **基于channel的全局上下文信息可以通过在空间维度上的全局平均池化收集得到，第k组的第c通道的全局信息表示如下：**

$$
s_{c}^{k}=\frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \hat{U}_{c}^{k}(i, j)
$$

- `A weighted fusion of the cardinal group representation` $V_{c}^k \in R^{H \times W \times C /K}$`is aggregated using channel-wise soft attention, where each feature-map channel is produced using a weighted combination over splits.`
- **一个cardinal group的加权融合表示 V 利用channel-wise 的软注意力聚集得到，其中每个feature-map channel是通过对split的加权组合产生，第c个channel计算如下：**

$$
V_{c}^{k}=\sum_{i=1}^{R} a_{i}^{k}(c) U_{R(k-1)+r}
$$

- 如论文中，上式的各个部分代表：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/27-3.png)

- 通俗理解此流程：
  - 在基数组内部，**对每一个特征图小组进行 1x1和3x3 的卷积操作，得到R个特征图小组后进行Split-Attention操作。**
  - 首先**将R个特征图按对应元素相加汇聚成一个特征图小组，此后在其上施加全局平均池化，得到 c(C/K) 维的特征向量，表示各个channel的权重。**
  - **其后经过BN+ReLU的操作以及后续的softmax操作，对channel权重向量进行修正，然后与原始的特征小组相乘后对应元素相加得到此基数组的输出**。
  - 作者在文中也强调了每个组内部的$F_i$映射变换是通过1x1和3x3 的卷积操作实现的。上面的（3）式中注意力机制方法 $\mathcal{G}$ 是通过两个全连接层实现的。

#### ResNeSt Block

- cardinal group representations 沿着channel维度拼接为$V = Concat(V^1,V^2...V^K)$ 。
-  与标准残差块一样，如果输入和输出特征映射共享相同的形状，则使用快捷连接 Y=V+X 生成分割注意块的最终输出Y。对于大小不一样的块，将对快捷连接应用适当的变换T以对齐输出形状：Y=V+T（X）

### 6. Radix-major Split-Attention Block

- 论文中写的**radix-major和cardinality-major的选择原因：**
- `For easily visualizing the concept of Split-Attention, we employ cardinalitymajor implementation in the methods description of the main paper, where the groups with the same cardinal index reside next to each other physically. The cardinality-major implementation is straightforward and intuitive, but is diﬃcult to modularize and accelerate using standard CNN operators. Therefore, we adopt the radix-major implementation in our experiments.`

- 论文图示：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/27-5.jpg" style="zoom:50%;" />

### 7. Network and Training

- 在这一部分，本文介绍了网络训练中的一些trick，具体内容可以参照原文
  - **Large Mini-batch Distributed Training**
  - **Label Smoothing**
  - **Auto Augmentation**
  - **Mixup Training**
  - **Large Crop Size**
  - **Regularization**

### 8. 参考

- [ResNet最强改进版来了！ResNeSt：Split-Attention Networks](https://zhuanlan.zhihu.com/p/132655457)
- [ResNeSt 实现有误？](https://zhuanlan.zhihu.com/p/135220104)
- [关于ResNeSt的点滴疑惑](https://zhuanlan.zhihu.com/p/133805433?utm_source=qq&utm_medium=social&utm_oi=728200852833075200)

- [ResNeSt: Split-Attention Networks阅读笔记](https://zhuanlan.zhihu.com/p/133496926)