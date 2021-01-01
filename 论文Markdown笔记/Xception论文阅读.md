### 1. 前言

- 本文原文链接：[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### 2. Abstract

- 论文指出，Incpetion模块解释为规则卷积到depthwise separable convolutions操作中间的换代技术。
- Inception模块已经被depthwise separable convolutions替代。

### 3. Introduction

- 首段指出了CNN在计算机视觉领域的发展：简单的卷积层+池化层的堆叠、重复多次的卷积操作、加深的网络结构。后来就出现了GoogLeNet的Inception模块。
- 首先下图所示为InceptionV3的结构图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/14-1.png" style="zoom:50%;" />

- **在一层卷积中我们尝试训练的是一个3-D的kernel，kernel有两个spatial dimension，H和W，一个channel dimension，也就是C。这样一来，一个kernel就需要同时学习spatial correlations和cross-channel correlations。**
- 我把这里理解为，spatial correlations学习的是某个特征在空间中的分布，cross-channel correlations学习的是这些不同特征的组合方式。
- **Inception的理念：首先通过一系列的1x1卷积来学习cross-channel correlations，同时将输入的维度降下来；再通过常规的3x3和5x5卷积来学习spatial correlations。这样一来，两个卷积模块分工明确。**

- 首先考虑一个简版的Inception module，拿掉所有的pooling，并且只用一层3x3的卷积来提取spatial correlations，如下图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/14-2.jpg" style="zoom:50%;" />

- 可以将这些1x1的卷积用一个较大的1x1卷积来替代，再在这个较大卷积产生的feature map上分出三个不重叠的部分，进行separable convolution，如下图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/14-3.jpg" style="zoom:50%;" />

- 这样一来就自然而然地引出：**为什么不是分出多个不重叠的部分，而是分出三个部分来进行separable convolution呢？如果加强一下Inception的假设，假设cross-channel correlations和spatial correlations是完全无关的呢？**
- 沿着上面的思路，一种极端的情况就是，**在每个channel上进行separable convolution，假设1x1卷积输出的feature map的channel有128个，那么极端版本的inception就是在每个channel上进行3x3的卷积，而不是学习一个3x3x128的kernel，取而代之的是学习128个3x3的kernel。将spatial correlations的学习细化到每一个channel，完全假设spatial correlations的学习于cross-channel correlations的学习无关，如下图所示：**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/14-4.jpg" style="zoom:50%;" />

- **接下里论文讨论了极端Inception和深度分离卷积的区别：**
  - 操作的顺序：depthwise separable convolution的通常实现首先执行通道空间卷积，然后执行1 x 1卷积；而Inception首先执行1 x 1卷积。
  - 第一次操作之后是否存在非线性操作。在Inception中，两个操作后都跟着ReLU非线性操作；然而depthwise separable convolution的通常实现没有非线性操作。

### 4. The Xception architecture

- 论文提出了一种完全基于depthwise separable convolution层的卷积神经网络架构。实际上，我们做出了以下假设：`that the mapping of cross-channels correlations and spatial correlations in the feature maps of convolutional neural networks can be entirely decoupled.`
- 因为这个假设是在Inception架构假设的基础上的假设增强版本，我们将我们提出的架构命名为Xception，它代表“极端Inception”。
- 下图给出了网络结构的描述：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/14-5.jpg)

- 简单地说，**Xception架构是一个带有残差连接的depthwise separable convolution层的线性堆叠，数据依次流过Entry flow, Middle flow和Exit flow。。**

- 论文剩余部分为实验结果评价与对比。
- **deeplabv3中对于Xception的改进：**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/12-4.jpg)

