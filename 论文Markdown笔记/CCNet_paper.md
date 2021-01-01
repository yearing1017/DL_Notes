### 1. 前言

- 论文原地址： [CCNet: Criss-Cross Attention for Semantic Segmentation](http://cn.arxiv.org/pdf/1811.11721.pdf)
- 本文涉及到了与**Non-local的比较，详见[知乎回答](https://zhuanlan.zhihu.com/p/51393573)**
- 本文是发表于 ICCV2019 的一篇文章，文中指出特征之间的长距离依赖性可以提供更加密集的上下信息，以辅助更好的对图像进行理解。介于此，提出了 **CCNet ( Criss Cross Network )，其中最主要的工作就是重复十字交叉注意力模块 ( Recurrent Criss Cross Attention Moudle, 简称RCCA)，该模块通过计算目标特征像素点与特征图中其它所有点之间的相互关系，并用这样的相互关系对目标像素点的特征进行加权，以此获得更加有效的目标特征。**

### 2. Abstract

- `Long-range dependencies can capture useful contextual information to beneﬁt visual understanding problems.`
- `In this work, we propose a Criss-Cross Network (CCNet) for obtaining such important information through a more effective and efﬁcient way. `
- `Concretely, for each pixel, our CCNet can harvest the contextual information of its surrounding pixels on the criss-cross path through a novel crisscross attention module.  By taking a further recurrent operation, each pixel can ﬁnally capture the long-range dependencies from all pixels.`

### 3. Introduction

- `Recently, state-of-the-art semantic segmentation frameworks based on the fully convolutional network (FCN) [26] have made remarkable progress. Due to the ﬁxed geometric structures, they are inherently limited to local receptive ﬁelds and short-range contextual information. These limitations impose a great adverse effect on FCN-based methods due to insufﬁcient contextual information.`
- **由于固定的几何结构，一些FCN-based语义分割方法受限于局部感受野和short-range contextual information，因此效果上有所影响。**
- **为了获取较长距离的特征依赖性，Deeplab 系列的 ASPP 模块，PSPNet 的金字塔池化模块以及诸多基于空洞卷积的方法相继被提出。**
- `However, the dilated convolution based methods [7, 6, 13] collect information from a few surrounding pixels and can not generate dense contextual information actually.`

- **但是它们都是只能获取一定范围像素的特征依赖性，并不能生成密集的上下文信息。**

- **为了生成密集的，逐像素的上下文信息，Non-local Networks使用自注意力机制来使得特征图中的任意位置都能感知所有位置的特征信息，从而生成更有效的逐像素特征表达。**
- `Here, each position in the feature map is connected with all other ones through self-adaptively predicted attention maps, thus harvesting various range contextual information, see in Fig. 1 (a).`
- **如图1，特征图的每个位置都通过self-adaptively predicted attention maps与其他位置相关联，因此生成更丰富的特征表达。Non-local的简单介绍详见：[知乎回答](https://zhuanlan.zhihu.com/p/51393573)**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/18-1.jpg" style="zoom:50%;" />

- (a）是Non-local，含蓝色中心的feature map是输入，分为上下两个分支处理：
  - 深绿色分支代表已经完成Non-local操作，得到了$f(x_i,x_j)$ (**绿色的深浅则代表了当前位置与蓝色中心点的相关性大小**)；
  - 下面灰色分支代表进行了 $g(x_j)$操作。将两个结果相乘，得到 $y_i$(含红色中心的feature map).

- 但是，这种方法是时间和空间复杂度都为$O((H \times W) \times (H \times W))$，H和W代表特征图的宽和高。由于语义分割中特征图的分辨率都很大，因此这种方法需要消耗巨大的计算复杂度和占用大量的GPU内存。

- `We found that the current no-local operation adopted by [32] can be alternatively replaced by two consecutive criss-cross operations, in which each one only has sparse connections (H + W − 1) for each position in the feature maps.`
- **作者发现non-local操作可以被两个连续的criss-cross操作代替，对于每个pixel，一个criss-cross操作只与特征图中(H+W-1)个位置连接，而不是所有位置。**
- **这激发了作者提出criss-cross attention module来从水平和竖直方向聚合long-range上下文信息。通过两个连续的criss-cross attention module，使得每个pixel都可以聚合所有pixels的特征信息，并且将时间和空间复杂度由O((HxW)x(HxW))降低到O((HxW)x(H+W-1))。**
- `Concretely, our criss-cross attention module is able to harvest various information nearby and far away on the criss-cross path. As shown in Fig. 1`

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/18-2.jpg" style="zoom:50%;" />

- `In criss-cross attention module, each position (e.g., blue color) in the feature map is connected with other ones which are in the same row and the same column through predicted sparsely attention map. The predicted attention map only has H+W −1 weights rather than H×W in non-local module.`
- `Furthermore, we propose the recurrent crisscross attention module to capture the long-range dependencies from all pixels.`
- **进一步地，提出了recurrent criss-cross attention module来捕获所有pixels的长依赖关系，并且所有的criss-cross attention module都共享参数以便减少参数量。**

### 4. Related work

- 这部分作者介绍了一些语义分割和attention模型，详见论文。

### 5. Approach

- 该章节分三个部分介绍CCNet的细节部分：**general framework、criss-cross attention module、recurrent criss-cross attention module。**

#### 5.1 Overall

- 网络结构如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/18-3.jpg)

- `An input image is passed through a deep convolutional neural networks (DCNN), which is designed in a fully convolutional fashion [6], then, produces a feature map X. We denote the spatial size of X as H × W. In order to retain more details and efﬁciently produce dense feature maps, we remove the last two down-sampling operations and employ dilation convolutions in the subsequent convolutional layers, thus enlarging the width/height of the output feature maps X to 1/8 of the input image.`
- **一个输入图像经过全卷积式的深度卷积神经网 (DCNN)后得到一个feature map X，X的空间尺寸为HxW。为了获得更多的细节，DCNN作者采用了dilated FCN，并且得到的feature map X为原图的1/8。**

- `After obtaining feature maps X, we ﬁrst apply a convolution layer to obtain the feature maps H of dimension reduction, then, the feature maps H would be fed into the criss-cross attention (CCA) module and generate new feature maps H 0 which aggregate long-range contextual information together for each pixel in a criss-cross way. The feature maps H 0 only aggregate the contextual information in horizontal and vertical directions which are not powerful enough for semantic segmentation. To obtain richer and denser context information, we feed the feature maps H 0 into the criss-cross attention module again and output feature maps H 00 . Thus, each position in feature maps H 00 actually gather the information from all pixels. Two crisscross attention modules before and after share the same parameters to avoid adding too many extra parameters. We name this recurrent structure as recurrent criss-cross attention (RCCA) module.`
- **在得到feature map X之后，首先用一个卷积曾获得维度更低的feature map H。然后将feature map H输入到criss-cross attention module(CCA)中得到新的feature map H’，H’中的每个piexl都包含其criss-cross方向上的上下文信息。紧接着，feature map H’被再次输入到CCA中得到feature map H’’。因此，feature map H’‘中的每个pixel都考虑了所有pixels的特征信息。为了降低参数量，两个CCA模块式参数共享的，作者命名这两个连续的CCA模型为recurrent criss-cross attention (RCCA) module。**

- `Then we concatenate the dense contextual feature H 00 with the local representation feature X. It is followed by one or several convolutional layers with batch normalization and activation for feature fusion. Finally, the fused features are fed into the segmentation layer to generate the ﬁnal segmentation map.`

#### 5.2 Criss-Cross Attention

- **CCA模块如下图所示：**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/18-4.jpg" style="zoom:50%;" />

- **首先根据输入的特征H，作者采用了三个不同的1*1的卷积核来获取注意力模型中的Q、K、V，其中Q和K的作用是为了获取当前像素与该像素下横向和纵向的像素点之间的相关性。最后将相关性矩阵与V相整合并加上H特征，就得到了含有丰富语义的特征表示H’。**

- `After obtaining feature maps Q and K, we further generate attention maps A ∈ R (H+W −1)×W ×H via Afﬁnity operation.`

- **Affinity operation：在特征图Q的每个空间维度上的像素位置u得到向量$\mathbf{Q}_{\mathbf{u}} \in \mathbb{R}^{C^{\prime}}$，与此同时，也可从特征图K得到在位置u同行同列的向量集合$\boldsymbol{\Omega}_{\mathbf{u}}，\boldsymbol{\Omega}_{\mathbf{u}} \in \mathbb{R}(H+W-1) \times C^{\prime}$，`Ω i,u ∈ R C is ith element of Ω u .`**

- `The Afﬁnity operation is deﬁned as follows:`

$$
d_{i, u}=\mathbf{Q}_{\mathbf{u}} \mathbf{\Omega}_{\mathbf{i}, \mathbf{u}}^{\top}
$$

- `d i,u ∈ D denotes the degree of correlation between feature Q u and Ω i,u.D ∈ R (H+W −1)×W ×H`
- **上述过程如下动图所示：**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/18-6.gif)

- `Then, we apply a softmax layer on D along the channel dimension to calculate the attention map A.`
- **现需要对 D 进行 softmax 操作，由上知，D 的尺寸为 $[(W+H-1)*W*H]$，即对W+H-1 维度的特征向量进行 softmax . 上面说了，D 记录的是特征图中每个像素点与同行同列像素之间的关系，softmax 操作的目的是对这样的位置关系进行归一化，这样就得到新的特征图 A，使得每个位置的贡献度更明了。**

- **另外的操作为：将相关性张量A与V特征进行整合，最后利用残差思想加上输入的特征H，就获得了丰富的特征表示H’。论文中详细如下：**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/18-5.jpg" style="zoom:50%;" />

- 演示图如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/18-7.gif)

#### 5.3 Recurrent Criss-Cross Attention

- **使用该模块的原因：**

- `Despite a criss-cross attention module can capture long-range contextual information in horizontal and vertical direction, the connections between the pixel and around pixels are still sparse. It is helpful to obtain dense contextual information for semantic segmentation.`

- **针对这种情况，文中指出，只要紧接着 CCA 模块串联着再做一次就可以获得丰富的上下文信息。**
- 原因大致如下，先看论文给的图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/18-8.jpg" style="zoom:50%;" />

- **首先左下角的绿色像素在loop1中时，只会包含左上角像素点与右下角的像素点，这个时候并没有左下角绿色点的相关信息。**
- **在loop2时再次计算左下角的绿色像素点时，再次包含了左上角与右下角的像素点，但是由于loop1中的参数与loop2中共享，所以在此时这两个点已经不再是单纯的两个点，其包含了蓝色像素点的相关信息。也就是说在loop2时会获得左下角像素点与蓝色像素点之间的间接关系。**

### 6. 实验及数据

- 有关实验结果和数据对比详见论文。

