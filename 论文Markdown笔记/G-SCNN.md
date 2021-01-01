### 1. 前言

- 本文为 ICCV2019 的一篇语义分割的文章
- 原文地址：[Gated-SCNN: Gated Shape CNNs for Semantic Segmentation](http://xxx.itp.ac.cn/pdf/1907.05740)

### 2. Abstract

- 图像分割的数据集包括颜色，形状和文本信息，但是现存的最优的网络架构都是将所有信息直接输入给网络，但由于数据集的形式是多样的，所以这样的处理当然不是最佳的。

> Here, we propose a new two-stream CNN architecture for semantic segmentation that explicitly wires shape information as a separate processing branch, i.e. shape stream, that processes information in parallel to the classical stream.

- 因此作者提出了一种新的思路，通过两个并行CNN结构来分别进行常规特征抽取和抽取图像的边界相关信息。作者将他们分别称为regular stream和shape stream。
- **Regular steam的结构与传统的语义分割模型相似。而Shape stream的主要作用是获取图像中的边界信息，最终将两者信息进行融合，从而得到最终的分割结果。**

### 3. Introduction

> A standard practice is to adapt an image classiﬁcation CNN architecture for the task of semantic segmentation by converting fully-connected layers into convolutional layers [37]. However, using classiﬁcation architectures for dense pixel prediction has several drawbacks [51, 37, 58, 11].

- 使用全卷积的分类网络进行dense pixel prediction有以下的缺陷：

  > One eminent drawback is the loss in spatial resolution of the output due to the use of pooling layers.

  - 一个显著的缺点是由于使用池化层而导致输出的空间分辨率降低。

> In this work, we propose a new two-stream CNN architecture for semantic segmentation that explicitly wires shape information as a separate processing branch. In particular, we keep the classical CNN in one stream, and add a so-called shape stream that processes information in parallel. We explicitly do not allow fusion of information between the two streams until the very top layers.

- 模型描述如上，模型图示如下：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/32-1.png" style="zoom:50%;" />

> Key to our architecture are a new type of gates that allow the two branches to interact. In particular, we exploit the higher-level information contained in the classical stream to denoise activations in the shape stream in its very early stages of processing. By doing so, the shape stream focuses on processing only the relevant information. This allows the shape stream to adopt a very effective shallow architecture that operates on the full image resolution.

- 作者提出的two-steam是并行的，这种架构的关键是一种新型的门，它连接两个流的中间层。具体而言，作者使用regular steam中的较高级特征的激活来控制形状流中的较低级特征的激活，有效地消除噪声并帮助shape steam仅关注处理相关的边界相关信息。

> To achieve that the shape information gets directed to the desired stream, we supervise it with a semantic boundary loss. We further exploit a new loss function that encourages the predicted semantic segmentation to correctly align with the groundtruth semantic boundaries, which further encourages the fusion layer to exploit information coming from the shape stream. We call our new architecture GSCNN.

- 为了实现形状信息被定向到所需的流，我们用语义边界损失来监督它。我们进一步开发一个新的损失函数，鼓励预测的语义分割正确地与groundtruth语义边界对齐，这进一步鼓励融合层利用来自形状流的信息。我们称我们的新架构为GSCNN。

### 4. Gated Shape CNN

> In this section, we present our Gated-Shape CNN architecture for semantic segmentation. As depicted in Fig. 2, our network consists of two streams of networks followed by a fusion module.

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/32-2.jpg)

> The ﬁrst stream of the network (“regular stream”) is a standard segmentation CNN, and the second stream (“shape stream”) processes shape information in the form of semantic boundaries.

> We enforce shape stream to only process boundary-related information by our carefully designed Gated Convolution Layer (GCL) and local supervision. We then fuse semantic-region features from the regular stream and boundary features from the shape stream to produce a reﬁned segmentation result, especially around boundaries. Next, we describe, in detail, each of the modules in our framework followed by our novel GCL.

- 我们通过精心设计的门控卷积层(GCL)和局部监控，强制shape stream只处理边界相关的信息。然后将规则流中的语义区域特征和shape stream中的边界特征进行融合，得到精确的分割结果，特别是边界附近的分割结果。

#### 4.1 Regular Stream

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/32-3.jpg" style="zoom:50%;" />

- Regular steam的输入是一个3 x H x W的图像，该图像经过全卷积网络来得到他的高级特征，最终输出的特征维度为C x (H/m) x (W/m)其中m是Regular steam中的stride设置。

#### 4.2 Shape Stream

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/32-4.jpg" style="zoom:50%;" />

- Shape steam**将regular steam的第一层卷积的输出与图像的梯度进行输入**，并产生语义边界信息。网络结构是结合残差块和门控卷积层组成，最终**用真实的图像中的语义边界对其进行监督**，从而产生最终的边界相关信息。

#### 4.3 Fusion Module

- 融合模型将将Regular steam和Shape steam产生的结果进行融合，最终产生维度为K x H x W的语义分割图的输出，K表示的是语义类别。作者在融合时使用了ASPP多尺度信息融合的方法将两个steam的输出进行多尺度融合最终得到的语义分割结果。

#### 4.4 Gated Convolutional Layer

> GCL is a core component of our architecture and helps the shape stream to only process relevant information by ﬁltering out the rest. Note that the shape stream does not incorporate features from the regular stream. Rather, it uses GCL to deactivate its own activations that are not deemed relevant by the higher-level information contained in the regular stream. One can think of this as a collaboration between two streams, where the more powerful one, which has formed a higher-level semantic understanding of the scene, helps the other stream to focus only on the relevant parts since start. This enables the shape stream to adopt an effective shallow architecture that processes the image at a very high resolution.

- 门控卷积层(GCL)的目的是**为了帮助Shape steam 只处理和边界相关的信息而过滤掉其他的信息**。且要注意的一点就是shape stream并不会整合来自regular steam的信息，相反，它使用GCL来停用其自身的激活，这些激活被regular steam中包含的更高级别信息认为不相关。作者对两个steam之间的多个地方使用GCL，其具体的过程如下图：**标注的*号为GCL**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/32-5.jpg" style="zoom:50%;" />

- 论文的原文描述如下：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/32-6.jpg" style="zoom:50%;" />

- 首先要**将两个steam得到的r_t 和 s_t 输出进行拼接，然后利用1x1卷积核和sigmoid函数得到一个注意力分布**，其公式如下：

$$
\alpha_{t}=\sigma\left(C_{1 \times 1}\left(s_{t} \| r_{t}\right)\right)
$$

- 针对上一层得到的shape分布将其与得到的注意力分布进行点乘，在加上之前的shape分布（这个就是残差计算部分）最后转置后与通道的权重核进行计算。通过一个GCL层得到一个新的shape 分布，公式如下：

$$
\begin{aligned} \hat{s}_{t}^{(i, j)} &=\left(s_{t} \text { * } w_{t}\right)_{(i, j)} \\ &=\left(\left(s_{t_{(i, j)}} \odot \alpha_{t_{(i, j)}}\right)+s_{t_{(i, j)}}\right)^{T} w_{t} \end{aligned}
$$

#### 4.5 Joint Multi-Task Learning

- 针对整体的任务，作者利用了两个损失函数，一个是regular steam得到的特征与真实分割结果之间的损失。一个是shape steam得到的边界结果与真实边界结果的损失，将这两者结合得到了最终的损失函数：

$$
\mathcal{L}^{\theta \phi, \gamma}=\lambda_{1} \mathcal{L}_{B C E}^{\theta, \phi}(s, \hat{s})+\lambda_{2} \mathcal{L}_{C E}^{\theta \phi, \gamma}(\hat{y}, f)
$$

#### 4.6 Dual Task Regularizer

- 作者为了防止模型过拟合，因此在损失函数中加入了正则化项。输出代表某个像素是否属于某张图片的某个类别，它可以由输出图片的空间的导数计算得到，公式如下：
- 其中p代表`categorical distribution output of the fusion module`

$$
\zeta=\frac{1}{\sqrt{2}}\left\|\nabla\left(G * \arg \max _{k} p\left(y^{k} \mid r, s\right)\right)\right\|
$$

- G为高斯滤波器。作者通过预测值与真实值之间的差的绝对值作为regular steam的正则项，其公式如下：

$$
\mathcal{L}_{r e g \rightarrow}^{\theta \phi, \gamma}=\lambda_{3} \sum_{p^{+}}\left|\zeta\left(p^{+}\right)-\hat{\zeta}\left(p^{+}\right)\right|
$$

- 作者希望确保当与GT边界不匹配时边界像素受到惩罚，并且避免非边界像素支配损失函数。所以在下面的正则项中利用的是边界预测和在边界区域的语义分割的二元性。其公式:

$$
\mathcal{L}_{r e g_{\leftarrow}}^{\theta \phi, \gamma}=\lambda_{4} \sum_{k_{n}} \mathbb{1}_{s_{p}}\left[\hat{y}_{p}^{k} \log p\left(y_{p}^{k} \mid r, s\right)\right]
$$

- 最终将两个子任务的正则项进行融合得到最终双任务的正则项：

$$
\mathcal{L}^{\theta \phi, \gamma}=\mathcal{L}_{r e g_{\rightarrow}}^{\theta \phi, \gamma}+\mathcal{L}_{r e g_{\leftarrow}}^{\theta \phi, \gamma}
$$

### 5. Experimental Results

- 实验结果及对比详见论文图标。

### 6. Conclusion

- 论文原文如下：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/32-7.jpg" style="zoom:50%;" />

