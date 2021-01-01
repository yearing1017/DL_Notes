### 1. 前言

- **Squeeze-and-Excitation Networks（SENet）通过对特征通道间的相关性进行建模，把重要的特征进行强化来提升准确率。**
- 2017 ILSVR竞赛的冠军，top5的错误率达到了2.251%，比2016年的第一名还要低25%，可谓提升巨大。

- 论文地址：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

### 2. Abstract

- 卷积神经网络主要依赖于卷积操作，在局部感受野中融合空间信息和通道信息，来提取有用的特征。**有很多工作从增强空间编码的角度来提升网络的表示能力**
- 本文主要聚焦于通道角度，并**提出了一种新的结构单元——“Squeeze-and Excitation(SE)”模块，可以自适应的调整各通道的特征响应值，对通道间的内部依赖关系进行建模**
- 如果将SE block添加到之前的先进网络中，只会增加很小的计算消耗，但却可以极大地提升网络性能

### 3. Introduction

- 本文主要探索网络架构设计的另一个方面：**特征通道之间的关系**
- 希望能够显式地**建模特征通道之间的相互依赖关系**
- 通过学习的方式来自动获取到每个特征通道的重要程度，然后依照这个重要程度去提升有用的特征并抑制对当前任务用处不大的特征

- **SE block的基本结构如下图所示：**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/26-1.jpg)

- 第一步**squeeze**操作：**将各通道的全局空间特征作为该通道的表示，形成一个通道描述符**
- 第二步**excitation**操作：**基于特征通道间的相关性，每个特征通道生成一个权重，代表特征通道的重要程度**

- 第三步**Reweight：将Excitation输出的权重看做每个特征通道的重要性，然后通过乘法逐通道加权到之前的特征上，完成在通道维度上的对原始特征的重标定**

### 4. Squeeze-and-Excitation Blocks

- 传统的卷积神经网络**卷积层的输出并没有考虑对各通道的依赖性**
- 本文的目标是**让网络有选择性的增强信息量大的特征，使得后续处理可以充分利用这些特征，并对无用特征进行抑制。**

#### 4.1 Squeeze: Global Information Embedding

- 将全局空间信息压缩为一个通道描述符
- 通过使用**全局平均池化**来生成通道统计信息，实现通道描述
- Squeeze部分的作用是获得Feature Map U 的每个通道的全局信息嵌入（特征向量）。
- 在SE block中，这一步通过VGG中引入的**Global Average Pooling（GAP）实现**的。也就是通过求每个通道 c 的Feature Map的平均值：

$$
z_{c}=\mathbf{F} s q\left(\mathbf{u}_{c}\right)=\frac{1}{W \times H} \sum_{i=1}^{W} \sum_{j=1}^{H} u_{c}(i, j)
$$

- 通过GAP得到的特征值是全局的（虽然比较粗糙）。另外，$z_c$ 也可以通过其它方法得到，要求只有一个，得到的特征向量具有全局性。

#### 4.2 Excitation: Adaptive Recalibration

- Excitation部分的作用是通过 $z_c$ 学习 C 中每个通道的特征权值，要求有三点：
  - 要足够灵活，这样能保证学习到的权值比较具有价值；
  - 要足够简单，这样不至于添加SE blocks之后网络的训练速度大幅降低；
  - 通道之间的关系是non-exclusive的，也就是说学习到的特征能够激励重要的特征，抑制不重要的特征。
- 根据上面的要求，SE blocks使用了两层全连接构成的门机制（gate mechanism）。门控单元 s (即结构图中彩色的1x1xC的特征向量)的计算方式表示为：

$$
\mathbf{s}=\mathbf{F}_{e x}(\mathbf{z}, \mathbf{W})=\sigma(g(\mathbf{z}, \mathbf{W}))=\sigma\left(g\left(\mathbf{W}_{2} \delta\left(\mathbf{W}_{1} \mathbf{z}\right)\right)\right)
$$

- 其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta) 表示ReLU激活函数， ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma) 表示sigmoid激活函数。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_1+%5Cin+%5Cmathbb%7BR%7D%5E%7B%5Cfrac%7BC%7D%7Br%7D%5Ctimes+C%7D) , ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D_2+%5Cin+%5Cmathbb%7BR%7D%5E%7BC%5Ctimes%5Cfrac%7BC%7D%7Br%7D%7D) 分别是两个全连接层的权值矩阵。 ![[公式]](https://www.zhihu.com/equation?tex=r) 则是中间层的隐层节点数，论文中指出这个值是16。

- 得到门控单元 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bs%7D) 后，最后的输出 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7BX%7D%7D) 表示为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bs%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BU%7D) 的向量积，即图1中的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BF%7D_%7Bscale%7D%28%5Ccdot%2C%5Ccdot%29) 操作：

$$
\tilde{x}_{c}=\mathbf{F}_{s c a l e}\left(\mathbf{u}_{c}, s_{c}\right)=s_{c} \cdot \mathbf{u}_{c}
$$

- 其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D_c) 是 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7BX%7D%7D) 的一个特征通道的一个Feature Map， ![[公式]](https://www.zhihu.com/equation?tex=s_c) 是门控单元 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bs%7D) （是个向量）中的一个标量值。
- 第一个全连接把C个通道压缩成了C/r个通道来降低计算量（后面跟了RELU），第二个全连接再恢复回C个通道（后面跟了Sigmoid），r是指压缩的比例。作者尝试了r在各种取值下的性能 ，最后得出结论r=16时整体性能和计算量最平衡。
- 有全连接层的原因：**没有全连接层，某个通道的调整值完全基于单个通道GAP的结果，事实上只有GAP的分支是完全没有反向计算、没有训练的过程的，就无法基于全部数据集来训练得出通道增强、减弱的规律。**

#### 4.3 SE-Inception 和 SE-ResNet

- 在Inception网络和ResNet网络中加入SE block，具体见图2、图3。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/26-2.jpg)

### 5. 实验及细节

- 有关实验及相关结果详见论文。