### 前言

- 论文：[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) CVPR 2017 (Best Paper Award)
- **作者从feature入手，通过对feature的极致利用达到更好的效果和更少的参数。**

### Abstract

- **原文**

  `Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections—one between each layer and its subsequent layer—our network has L(L+1) 2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance. Code and pre-trained models are available at https://github.com/liuzhuang13/DenseNet.`

- **翻译理解**

  - 最近的工作表明，如果卷积网络在靠近输入的层与靠近输出的层之间包含更短的连接，那么卷积网络的深度可以显著增加，准确度更高，并且更易于训练。
  - 在本文中，我们采纳这一观点，并提出了密集连接卷积网络(DenseNet)，它以前馈的方式将每个层与每个其它层连接起来。
  - 具有L层的传统卷积网络具有L个连接（每个层与其后续层之间），而我们的网络具有的连接个数为：$\frac{L(L+1)}{2}$

  - 对于每个层，所有先前层的特征图都被用作本层的输入，并且本层输出的特征图被用作所有后续层的输入。
  - DenseNet有几个优点：减轻梯度弥散问题，加强特征传播，鼓励特征重用，大大减少参数数量。

  - 我们在四个竞争激烈的对象识别基准任务(CIFAR-10, CIFAR-100, SVHN, ImageNet)上对我们提出的网络架构进行评估。相较于其它最先进的方法，DenseNet在大多数情况下都有显著的改善，同时要求较少的算力以实现高性能。代码和预训练的模型在https://github.com/liuzhuang13/DenseNet

### 1. Introduction

- **ph_1**

  `Convolutional neural networks (CNNs) have become the dominant machine learning approach for visual object recognition. Although they were originally introduced over 20 years ago [18], improvements in computer hardware and network structure have enabled the training of truly deep CNNs only recently. The original LeNet5 [19] consisted of 5 layers, VGG featured 19 [29], and only last year Highway Networks [34] and Residual Networks (ResNets) [11] have surpassed the 100-layer barrier.`

- **理解**
  - 各种卷积神经网络(CNN)已经成为视觉对象识别的主要机器学习方法。
  - 虽然它们最早是在20多年前被提出的，但是直到近年来，随着计算机硬件和网络结构的改进，才使得训练真正深层的CNN成为可能。
  - 最早的LeNet5由5层组成，VGG有19层，去年的Highway Network和残差网络(ResNet)已经超过了100层。

- **ph_2**

  `As CNNs become increasingly deep, a new research problem emerges: as information about the input or gradient passes through many layers, it can vanish and “wash out” by the time it reaches the end (or beginning) of the network. Many recent publications address this or related problems. ResNets [11] and Highway Networks [34] bypass signal from one layer to the next via identity connections. Stochastic depth [13] shortens ResNets by randomly dropping layers during training to allow better information and gradient flow. FractalNets [17] repeatedly combine several parallel layer sequences with different number of convolutional blocks to obtain a large nominal depth, while maintaining many short paths in the network. Although these different approaches vary in network topology and training procedure, they all share a key characteristic: they create short paths from early layers to later layers.`

- **理解**

  - 随着CNN越来越深，出现了一个新的研究问题：梯度弥散。
  - 许多最近的研究致力于解决这个问题或相关的问题。ResNet和Highway Network通过恒等连接将信号从一个层传递到另一层。
  - Stochastic depth通过在训练期间随机丢弃层来缩短ResNets，以获得更好的信息和梯度流。FractalNet重复将几个并行层序列与不同数量的约束块组合，获得大的标称深度，同时在网络中保持许多短路径。

  - 虽然这些不同的方法在网络拓扑和训练过程中有所不同，但它们都具有一个关键特性：**它们创建从靠近输入的层与靠近输出的层的短路径。**

- **ph_3**

  `In this paper, we propose an architecture that distills this insight into a simple connectivity pattern: to ensure maximum information flow between layers in the network, we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers. Figure 1 illustrates this layout schematically. Crucially, in contrast to ResNets, we never combine features through summation before they are passed into a layer; instead, we combine features by concatenating them. Hence, the  th layer has  inputs, consisting of the feature-maps of all preceding convolutional blocks. Its own feature-maps are passed on to all L− subsequent layers. This introduces L(L+1) 2 connections in an L-layer network, instead of just L, as in traditional architectures. Because of its dense connectivity pattern, we refer to our approach as Dense Convolutional Network (DenseNet).`

- **理解**

  - 在本文中，我们提出了一种将这种关键特性简化为简单连接模式的架构：为了确保网络中各层之间的最大信息流，我们将所有层（匹配的特征图大小）直接连接在一起。
  - 为了保持前馈性质，**每个层从所有先前层中获得额外的输入，并将其自身的特征图传递给所有后续层。**

  - 如下图（图1）所示：（5层的密集连接块，增长率k=4k=4，每一层都把先前层的输出作为输入）

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/2-1.png)
  - 最重要的是，与ResNet相比，我们从不将特征通过求和合并后作为一层的输入，我们将特征串联成一个更长的特征。
  - 因此，**第ℓ层(不包括input层)有ℓ个输入**，由所有先前卷积块的特征图组成。它自己的特征图传递给所有L−ℓ个后续层。这样，在L层网络中，有$\frac{L(L+1)}{2}$个连接，而不是传统架构中仅仅有LL个连接。

  - 由于其密集的连接模式，我们将我们的方法称为密集连接卷积网络(DenseNet)。

- **ph_4**

  `A possibly counter-intuitive effect of this dense connectivity pattern is that it requires fewer parameters than traditional convolutional networks, as there is no need to relearn redundant feature-maps. Traditional feed-forward architectures can be viewed as algorithms with a state, which is passed on from layer to layer. Each layer reads the state from its preceding layer and writes to the subsequent layer. It changes the state but also passes on information that needs to be preserved. ResNets [11] make this information preservation explicit through additive identity transformations. Recent variations of ResNets [13] show that many layers contribute very little and can in fact be randomly dropped during training. This makes the state of ResNets similar to (unrolled) recurrent neural networks [21], but the number of parameters of ResNets is substantially larger because each layer has its own weights. Our proposed DenseNet architecture explicitly differentiates between information that is added to the network and information that is preserved. DenseNet layers are very narrow (e.g., 12 filters per layer), adding only a small set of feature-maps to the “collective knowledge” of the network and keep the remaining featuremaps unchanged—and the final classifier makes a decision based on all feature-maps in the network.`

- **理解**

  - 这种密集连接模式的可能的反直觉效应是，它比传统卷积网络需要的参数少，因为不需要重新学习冗余特征图。
  - 传统的前馈架构可以被看作是具有状态的算法，它是从一层传递到另一层的。
  - 每个层从上一层读取状态并写入后续层。它改变状态，但也传递需要保留的信息。
  - ResNet通过加性恒等转换使此信息保持明确。ResNet的最新变化表明，许多层次贡献很小，实际上可以在训练过程中随机丢弃。这使得ResNet的状态类似于（展开的）循环神经网络(recurrent neural network)，但是ResNet的参数数量太大，因为每个层都有自己的权重。

  - 我们提出的DenseNet架构明确区分添加到网络的信息和保留的信息。
  - DenseNet层非常窄（例如，每层12个卷积核），仅将一小组特征图添加到网络的“集体知识”，并保持剩余的特征图不变。最终分类器基于网络中的所有特征图。

- **ph_5**

  `Besides better parameter efficiency, one big advantage of DenseNets is their improved flow of information and gradients throughout the network, which makes them easy to train. Each layer has direct access to the gradients from the loss function and the original input signal, leading to an implicit deep supervision [20]. This helps training of deeper network architectures. Further, we also observe that dense connections have a regularizing effect, which reduces overfitting on tasks with smaller training set sizes.`

- **理解**

  - 除了更好的参数效率，DenseNet的一大优点是改善了整个网络中信息流和梯度流，从而使其易于训练。
  - 每个层都可以直接从损失函数和原始输入信号中获取梯度，从而进行了深入的监督。
  - 这有助于训练更深层的网络架构。此外，我们还观察到密集连接具有正则化效应，这减少了具有较小训练集大小的任务的过拟合。

- **ph_6**

  `We evaluate DenseNets on four highly competitive benchmark datasets (CIFAR-10, CIFAR-100, SVHN, and ImageNet). Our models tend to require much fewer parameters than existing algorithms with comparable accuracy. Further, we significantly outperform the current state-ofthe-art results on most of the benchmark tasks.`

- **理解**

  - 我们在四个高度竞争的基准数据集(CIFAR-10，CIFAR-100，SVHN和ImageNet)上评估DenseNet。在准确度相近的情况下，我们的模型往往需要比现有算法少得多的参数。
  - 此外，我们的模型在大多数测试任务中，准确度明显优于其它最先进的方法。

### 2. Related Work

- **ph_1**

  `The exploration of network architectures has been a part of neural network research since their initial discovery. The recent resurgence in popularity of neural networks has also revived this research domain. The increasing number of layers in modern networks amplifies the differences between architectures and motivates the exploration of different connectivity patterns and the revisiting of old research ideas.`

- **理解**

  - 网络架构的探索一直是神经网络研究的一部分。
  - 最近神经网络流行的复苏也使得这个研究领域得以恢复。
  - 现代网络中越来越多的层扩大了架构之间的差异，激发了对不同连接模式的探索和对旧的研究思想的重新审视。

- **ph_2**

  `A cascade structure similar to our proposed dense network layout has already been studied in the neural networks literature in the 1980s [3]. Their pioneering work focuses on fully connected multi-layer perceptrons trained in a layerby-layer fashion. More recently, fully connected cascade networks to be trained with batch gradient descent were proposed [40]. Although effective on small datasets, this approach only scales to networks with a few hundred parameters. In [9, 23, 31, 41], utilizing multi-level features in CNNs through skip-connnections has been found to be effective for various vision tasks. Parallel to our work, [1] derived a purely theoretical framework for networks with cross-layer connections similar to ours.`

- **理解**

  - 类似于我们提出的密集网络布局的级联结构已经在20世纪80年代的神经网络文献中被研究。
  - 他们的开创性工作着重于以逐层方式训练的全连接多层感知机。
  - 最近，用批量梯度下降训练的全连接的级联网络的方法被提出。虽然对小数据集有效，但这种方法只能扩展到几百个参数的网络。
  - 在[9,23,31,41]中，通过跳跃连接在CNN中利用多层次特征已被发现对于各种视觉任务是有效的。
  - 与我们的工作同时进行的，1为具有类似于我们的跨层连接的网络衍生了一个纯粹的理论框架。

- **ph_3**

  `Highway Networks [34] were amongst the first architectures that provided a means to effectively train end-to-end networks with more than 100 layers. Using bypassing paths along with gating units, Highway Networks with hundreds of layers can be optimized without difficulty. The bypassing paths are presumed to be the key factor that eases the training of these very deep networks. This point is further supported by ResNets [11], in which pure identity mappings are used as bypassing paths. ResNets have achieved impressive, record-breaking performance on many challenging image recognition, localization, and detection tasks, such as ImageNet and COCO object detection [11]. Recently, stochastic depth was proposed as a way to successfully train a 1202-layer ResNet [13]. Stochastic depth improves the training of deep residual networks by dropping layers randomly during training. This shows that not all layers may be needed and highlights that there is a great amount of redundancy in deep (residual) networks. Our paper was partly inspired by that observation. ResNets with pre-activation also facilitate the training of state-of-the-art networks with > 1000 layers [12].`

- **理解**

  - Highway Network是其中第一个提供了有效训练100多层的端对端网络的方案。
  - 使用旁路与门控单元，可以无困难地优化具有数百层深度的Highway Network。旁路是使这些非常深的网络训练变得简单的关键因素。
  - ResNet进一步支持这一点，其中纯恒等映射用作旁路。

  - ResNet已经在许多挑战性的图像识别，定位和检测任务（如ImageNet和COCO对象检测）上取得了显著的创纪录的表现。
  - 最近，Stochastic depth被提出作为一种成功地训练1202层ResNet的方式。Stochastic depth通过在训练期间随机丢弃层来改善深层ResNet的训练。这表明不是所有的层都是有必要存在的，并且突显了在深层网络中存在大量的冗余。
  - 本文一定程度上受到了这一观点的启发。具有预激活的ResNet还有助于对具有超过1000层的最先进网络的训练。

- **ph_4**

  `An orthogonal approach to making networks deeper (e.g., with the help of skip connections) is to increase the network width. The GoogLeNet [36, 37] uses an “Inception module” which concatenates feature-maps produced by filters of different sizes. In [38], a variant of ResNets with wide generalized residual blocks was proposed. In fact, simply increasing the number of filters in each layer of ResNets can improve its performance provided the depth is sufficient [42]. FractalNets also achieve competitive results on several datasets using a wide network structure [17].`

- **理解**

  - 另一种使网络更深的方法（例如，借助于跳连接）是增加网络宽度。
  - GoogLeNet使用一个“Inception模块”，它连接由不同大小的卷积核产生的特征图。
  - 在[38]中，提出了具有宽广残差块的ResNet变体。事实上，只要深度足够，简单地增加每层ResNets中的卷积核数量就可以提高其性能。
  - FractalNet也可以使用更宽的网络结构在几个数据集上达到不错的效果。

- **ph_5**

  `Instead of drawing representational power from extremely deep or wide architectures, DenseNets exploit the potential of the network through feature reuse, yielding condensed models that are easy to train and highly parameterefficient. Concatenating feature-maps learned by different layers increases variation in the input of subsequent layers and improves efficiency. This constitutes a major difference between DenseNets and ResNets. Compared to Inception networks [36, 37], which also concatenate features from different layers, DenseNets are simpler and more efficient.`

- **理解**

  - **DenseNet不是通过更深或更宽的架构来获取更强的表示学习能力，而是通过特征重用来发掘网络的潜力**，产生易于训练和高效利用参数的浓缩模型。
  - 由不同层次学习的串联的特征图增加了后续层输入的变化并提高效率。
  - 这是DenseNet和ResNet之间的主要区别。与Inception网络相比，它也连接不同层的特征，DenseNet更简单和更高效。

- **ph_6**

  `There are other notable network architecture innovations which have yielded competitive results. The Network in Network (NIN) [22] structure includes micro multi-layer perceptrons into the filters of convolutional layers to extract more complicated features. In Deeply Supervised Network (DSN) [20], internal layers are directly supervised by auxiliary classifiers, which can strengthen the gradients received by earlier layers. Ladder Networks [27, 25] introduce lateral connections into autoencoders, producing impressive accuracies on semi-supervised learning tasks. In [39], Deeply-Fused Nets (DFNs) were proposed to improve information flow by combining intermediate layers of different base networks. The augmentation of networks with pathways that minimize reconstruction losses was also shown to improve image classification models [43].`

- **理解**

  - 还有其他显著的网络架构创新产生了有竞争力的效果。
  - NIN的结构包括将微多层感知机插入到卷积层的卷积核中，以提取更复杂的特征。
  - 在深度监督网络（DSN）中，隐藏层由辅助分类器直接监督，可以加强先前层次接收的梯度。
  - 梯形网络引入横向连接到自动编码器，在半监督学习任务上产生了令人印象深刻的准确性。
  - 在[39]中，提出了通过组合不同基网络的中间层来提高信息流的深度融合网络（DFN）。
  - 通过增加最小化重建损失路径的网络也使得图像分类模型得到改善。

### 3. DenseNets

- **ph_1**

  `Consider a single image x0 that is passed through a convolutional network. The network comprises L layers, each of which implements a non-linear transformation H(·), where indexes the layer. H(·) can be a composite function of operations such as Batch Normalization (BN) [14], rectified linear units (ReLU) [6], Pooling [19], or Convolution (Conv). We denote the output of the  th layer as x.`

- **理解**

  - 考虑在一个卷积网络中传递的单独图像$x_0$，这个网络包含L层，每层都实现了一个非线性变换$H_l(.)$，其中ℓ表示层的索引。
  - $H_l(.)$可以是诸如批量归一化(BN)、线性整流单元(ReLU)、池化(Pooling)或卷积(Conv)等操作的复合函数。我们将第ℓ层输出表示为xℓ。

- **ph_2**

  `ResNets. Traditional convolutional feed-forward networks connect the output of the  th layer as input to the ( + 1)th layer [16], which gives rise to the following layer transition: x= H(x−1). ResNets [11] add a skip-connection that bypasses the non-linear transformations with an identity function:`
  $$
  x_l = H_l(x_{l-1})+x_{l-1}
  $$
  `An advantage of ResNets is that the gradient can flow directly through the identity function from later layers to the earlier layers. However, the identity function and the output of H are combined by summation, which may impede the information flow in the network`

- **理解**

  - **ResNet：**传统的前馈卷积神经网络将第ℓ层的输出作为第ℓ+1层的输入，可表示为：$x_l = H_l(x_{l-1})$

  - ResNet添加了一个跳连接，即使用恒等函数跳过非线性变换：

  - $$
    x_l = H_l(x_{l-1})+x_{l-1}
    $$

  - ResNets的一个优点是梯度可以通过从后续层到先前层的恒等函数直接流动。然而，恒等函数与Hℓ的输出是通过求和组合，这可能阻碍网络中的信息流。

- **ph_3**

  `Dense connectivity. To further improve the information flow between layers we propose a different connectivity pattern: we introduce direct connections from any layer to all subsequent layers. Figure 1 illustrates the layout of the resulting DenseNet schematically. Consequently, the  th layer receives the feature-maps of all preceding layers, x0, . . . , x−1, as input:`
  $$
  x_l = H_l([x_0,x_1,...x_{l-1}])
  $$
  `where [x0, x1, . . . , x−1] refers to the concatenation of the feature-maps produced in layers 0, . . . , −1. Because of its dense connectivity we refer to this network architecture as Dense Convolutional Network (DenseNet). For ease of implementation, we concatenate the multiple inputs of H(·) in eq. (2) into a single tensor.`

- **理解**

  - **密集连接：**为了进一步改善层之间的信息流，我们提出了不同的连接模式：我们提出从任何层到所有后续层的直接连接。

  - 因此，第ℓ层接收所有先前图层的特征图，$x_0,x_1,…,x_{ℓ−1}$，作为输入:

  - $$
    x_l = H_l([x_0,x_1,...x_{l-1}])
    $$

  - 其中$[x_0,x_1,…,x_{ℓ−1}]$表示0,…,ℓ−1层输出的特征图的串联。由于其密集的连接，我们将此网络架构称为密集卷积网络(DenseNet)。为了便于实现，我们将公式(2)中的$H_ℓ(⋅)$的多个输入连接起来变为单张量。

- **ph_4**

  `Composite function. Motivated by [12], we define H(·) as a composite function of three consecutive operations: batch normalization (BN) [14], followed by a rectified linear unit (ReLU) [6] and a 3 × 3 convolution (Conv).`

- **理解**

  - **复合函数**，受到12的启发，我们将Hℓ(⋅)定义为进行三个连续运算的复合函数：**先批量归一化(BN)，然后是线性整流单元(ReLU)，最后接一个3×3的卷积(Conv)。**

- **ph_5**

  `Pooling layers. The concatenation operation used in Eq. (2) is not viable when the size of feature-maps changes. However, an essential part of convolutional networks is down-sampling layers that change the size of feature-maps. To facilitate down-sampling in our architecture we divide the network into multiple densely connected dense blocks; see Figure 2. We refer to layers between blocks as transition layers, which do convolution and pooling. The transition layers used in our experiments consist of a batch normalization layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer.`

- **理解**

  - **池化层：**当特征图的大小变化时，公式(2)中的级串联运算是不可行的。所以，卷积网络的一个必要的部分是改变特征图尺寸的下采样层。
  - 为了便于我们的架构进行下采样，我们将网络分为多个密集连接的密集块。
  - 如下图（图2）所示：（一个有三个密集块的DenseNet。两个相邻块之间的层被称为过渡层，并通过卷积和池化来改变特征图大小。）

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/2-2.jpg)

  - 我们将块之间的层称为过渡层，它们进行卷积和池化。我们实验中使用的过渡层由一个批量归一化层(BN)，一个1×1的卷积层和一个2×2平均池化层组成。

- **ph_6**

  `Growth rate. If each function H produces k featuremaps, it follows that the  th layer has k0 +k ×(−1) input feature-maps, where k0 is the number of channels in the input layer. An important difference between DenseNet and existing network architectures is that DenseNet can have very narrow layers, e.g., k = 12. We refer to the hyperparameter k as the growth rate of the network. We show in Section 4 that a relatively small growth rate is sufficient to obtain state-of-the-art results on the datasets that we tested on. One explanation for this is that each layer has access to all the preceding feature-maps in its block and, therefore, to the network’s “collective knowledge”. One can view the feature-maps as the global state of the network. Each layer adds k feature-maps of its own to this state. The growth rate regulates how much new information each layer contributes to the global state. The global state, once written, can be accessed from everywhere within the network and, unlike in traditional network architectures, there is no need to replicate it from layer to layer.`

- **理解**
  - 如果每个Hℓ函数输出k个特征图，那么第ℓ层有k0+k×(ℓ−1)个输入特征图，其中k0是输入层的通道数。
  - 与其它网络架构的一个重要的区别是，DenseNet可以有非常窄的层，比如k=12。我们将超参数kk称为网络的增长率。
  - 在实验这一节中可以看到，一个相对小的增长率已经足矣在我们测试的数据集上达到领先的结果。
  - 对此的一个解释是，每个层都可以访问其块中的所有先前的特征图，即访问网络的“集体知识”。
  - 可以将特征图视为网络的全局状态。每个层都将自己的k个特征图添加到这个状态。增长率调节每层对全局状态贡献多少新信息。
  - 与传统的网络架构不同的是，一旦写入，从网络内的任何地方都可以访问全局状态，无需逐层复制。

- **ph_7**

  `Bottleneck layers. Although each layer only produces k output feature-maps, it typically has many more inputs. It has been noted in [37, 11] that a 1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency. We find this design especially effective for DenseNet and we refer to our network with such a bottleneck layer, i.e., to the BN-ReLU-Conv(1× 1)-BN-ReLU-Conv(3×3) version of H, as DenseNet-B. In our experiments, we let each 1×1 convolution produce 4k feature-maps.`

- **理解**

  - 尽管每层只产生kk个输出特征图，但它通常具有更多的输入。
  - 在[37,11]中提到，一个1×1的卷积层可以被看作是瓶颈层，放在一个3×3的卷积层之前可以起到减少输入数量的作用，以提高计算效率。**（类似inception中1x1卷积核的减少参数运算作用）**

  - 我们发现这种设计对DenseNet尤其有效，我们的网络也有这样的瓶颈层，比如另一个名为DenseNet-B版本的Hℓ是这样的：BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)，在我们的实验中，我们让1×1的卷积层输出4k个特征图。

- **ph_8**

  `Compression. To further improve model compactness, we can reduce the number of feature-maps at transition layers. If a dense block contains m feature-maps, we let the following transition layer generate bθmc output featuremaps, where 0 <θ ≤1 is referred to as the compression factor. When θ = 1, the number of feature-maps across transition layers remains unchanged. We refer the DenseNet with θ <1 as DenseNet-C, and we set θ = 0.5 in our experiment. When both the bottleneck and transition layers with θ < 1 are used, we refer to our model as DenseNet-BC.`

- **理解**

  - 为了进一步提高模型的紧凑性，我们可以在过渡层减少特征图的数量。
  - 如果密集块包含m个特征图，我们让后续的过渡层输出⌊θm⌋个特征图，其中θ为压缩因子，且0<θ≤1。
  - 当θ=1时，通过过渡层的特征图的数量保持不变。我们将θ<1的DenseNet称作DenseNet-C，我们在实验设置θ=0.5。
  - 我们将同时使用瓶颈层和压缩的模型称为DenseNet-BC。

- **ph_9**

  `Implementation Details. On all datasets except ImageNet, the DenseNet used in our experiments has three dense blocks that each has an equal number of layers. Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC) output channels is performed on the input images. For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded by one pixel to keep the feature-map size fixed. We use 1×1 convolution followed by 2×2 average pooling as transition layers between two contiguous dense blocks. At the end of the last dense block, a global average pooling is performed and then a softmax classifier is attached. The feature-map sizes in the three dense blocks are 32× 32, 16×16, and 8×8, respectively. We experiment with the basic DenseNet structure with configurations {L = 40, k = 12}, {L = 100, k = 12} and {L = 100, k = 24}. For DenseNetBC, the networks with configurations {L = 100, k = 12}, {L= 250, k= 24} and {L= 190, k= 40} are evaluated.`

- **理解**

  - **实现细节：**在除ImageNet之外的所有数据集上，我们实验中使用的DenseNet具有三个密集块，每个具有相等层数。
  - 在进入第一个密集块之前，对输入图像执行卷积，输出16（或者DenseNet-BC的增长率的两倍）通道的特征图。
  - 对于卷积核为3×3的卷积层，输入的每一边都添加1像素宽度的边，以0填充，以保持特征图尺寸不变。

  - 在两个连续的密集块之间，我们使用一个1×1的卷积层接一个2×2的平均池化层作为过渡层。
  - 在最后一个密集块的后面，执行全局平均池化，然后附加一个softmax分类器。
  - 三个密集块的特征图尺寸分别是32×32，16×16和8×8。
  - 在实验中，基础版本的DenseNet超参数配置为：L=40，k=12，L=100，k=12和L=100，k=24。DenseNet-BC的超参数配置为：L=100,k=12，L=250，k=24和L=190，k=40。

- **ph_10**

  `In our experiments on ImageNet, we use a DenseNet-BC structure with 4 dense blocks on 224×224 input images. The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2; the number of feature-maps in all other layers also follow from setting k. The exact network configurations we used on ImageNet are shown in Table 1`

- **理解**

  - 在ImageNet上的实验，我们使用的DenseNet-BC架构包含4个密集块，输入图像大小为224×224。
  - 初始卷积层包含2k个大小为7×7，步长为2的卷积。其它层的特征图数量都由超参数k决定。
  - 具体的网络架构如下表（表1）所示：（用于ImageNet的DenseNet网络架构。所有网络的增长率都是k=32k=32。注意，**表格中的Conv层表示BN-ReLU-Conv的组合**）

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/2-3.jpg)

### 4. Experiments

	- `We empirically demonstrate DenseNet’s effectiveness on several benchmark datasets and compare with state-of-theart architectures, especially with ResNet and its variants.`

- 我们经验性地证明了DenseNet在几个基准数据集上的有效性，并与最先进的架构进行比较，特别是ResNet及其变体。

#### 4.1 Datasets

- **ph_1**

  `CIFAR. The two CIFAR datasets [15] consist of colored natural images with 32×32 pixels. CIFAR-10 (C10) consists of images drawn from 10 and CIFAR-100 (C100) from 100 classes. The training and test sets contain 50,000 and 10,000 images respectively, and we hold out 5,000 training images as a validation set. We adopt a standard data augmentation scheme (mirroring/shifting) that is widely used for these two datasets [11, 13, 17, 22, 28, 20, 32, 34]. We denote this data augmentation scheme by a “+” mark at the end of the dataset name (e.g., C10+). For preprocessing, we normalize the data using the channel means and standard deviations. For the final run we use all 50,000 training images and report the final test error at the end of training.`

- **理解**

  - **CIFAR：**两个CIFAR数据集都是由32×32像素的彩色照片组成的。CIFAR-10(C10)包含10个类别，CIFAR-100(C100)包含100个类别。训练和测试集分别包含50,000和10,000张照片，我们将5000张训练照片作为验证集。
  - 我们采用广泛应用于这两个数据集的标准数据增强方案（镜像/移位）。我们通过数据集名称末尾的“+”标记（例如，C10+）表示该数据增强方案。
  - 对于预处理，我们使用各通道的均值和标准偏差对数据进行归一化。对于最终的训练，我们使用所有50,000训练图像，作为最终的测试结果。

- **ph_2**

  `SVHN. The Street View House Numbers (SVHN) dataset [24] contains 32×32 colored digit images. There are 73,257 images in the training set, 26,032 images in the test set, and 531,131 images for additional training. Following common practice [7, 13, 20, 22, 30] we use all the training data without any data augmentation, and a validation set with 6,000 images is split from the training set. We select the model with the lowest validation error during training and report the test error. We follow [42] and divide the pixel values by 255 so they are in the [0, 1] range.`

- **理解**

  - **SVHN：**街景数字(SVHN)数据集由32×32像素的彩色数字照片组成。训练集有73,257张照片，测试集有26,032张照片，以及531,131张照片进行额外的训练。
  - 按常规做法，我们使用所有的训练数据，没有任何数据增强，使用训练集中的6,000张照片作为验证集。
  - 我们选择在训练期间具有最低验证错误的模型，作最终的测试。我们遵循[42]并将像素值除以255，使它们在[0,1]范围内。

- **ph_3**

  `ImageNet. The ILSVRC 2012 classification dataset [2] consists 1.2 million images for training, and 50,000 for validation, from 1, 000 classes. We adopt the same data augmentation scheme for training images as in [8, 11, 12], and apply a single-crop or 10-crop with size 224×224 at test time. Following [11, 12, 13], we report classification errors on the validation set.`

- **理解**

  - **ImageNet**：ILSVRC 2012分类数据集包含1000个类，训练集120万张照片，验证集50,000张照片。
  - 我们采用与[8,11,12]相同的数据增强方案来训练照片，并在测试时使用尺寸为224×224的single-crop和10-crop。与[11,12,13]一样，我们报告验证集上的分类错误。

#### 4.2 Training

- **ph_1**

  `All the networks are trained using stochastic gradient descent (SGD). On CIFAR and SVHN we train using batch size 64 for 300 and 40 epochs, respectively. The initial learning rate is set to 0.1, and is divided by 10 at 50% and 75% of the total number of training epochs. On ImageNet, we train models for 90 epochs with a batch size of 256. The learning rate is set to 0.1 initially, and is lowered by 10 times at epoch 30 and 60. Note that a naive implementation of DenseNet may contain memory inefficiencies. To reduce the memory consumption on GPUs, please refer to our technical report on the memory-efficient implementation of DenseNets [26].`

- **理解**

  - **所有网络都使用随机梯度下降(SGD)进行训练。**
  - 在CIFAR和SVHN上，我们训练批量为64，分别训练300和40个周期。初始学习率设置为0.1，在训练周期数达到50％和75％时除以10。
  - 在ImageNet上，训练批量为256，训练90个周期。学习速率最初设置为0.1，并在训练周期数达到30和60时除以10。
  - 由于GPU内存限制，我们最大的模型(DenseNet-161)以小批量128进行训练。为了补偿较小的批量，我们训练该模型的周期数为100，并在训练周期数达到90时将学习率除以10。

- **ph_2**

  `Following [8], we use a weight decay of 10−4 and a Nesterov momentum [35] of 0.9 without dampening. We adopt the weight initialization introduced by [10]. For the three datasets without data augmentation, i.e., C10, C100 and SVHN, we add a dropout layer [33] after each convolutional layer (except the first one) and set the dropout rate to 0.2. The test errors were only evaluated once for each task and model setting.`

- **理解**

  - 根据[8]，我们使用的权重衰减为$10^{−4}$，Nesterov动量[35]为0.9且没有衰减。我们采用[10]中提出的权重初始化。
  - 对于没有数据增强的三个数据集，即C10，C100和SVHN，我们在每个卷积层之后（除第一个层之外）添加一个Dropout层，并将Dropout率设置为0.2。只对每个任务和超参数设置评估一次测试误差。

#### 4.3  Classification Results on CIFAR and SVHN

- **ph_1**

  `We train DenseNets with different depths, L, and growth rates, k. The main results on CIFAR and SVHN are shown in Table 2. To highlight general trends, we mark all results that outperform the existing state-of-the-art in boldface and the overall best result in blue.`

- **理解**

  - 我们训练不同深度L和增长率k的DenseNet。
  -  CIFAR和SVHN的主要结果如下表（表2）所示：

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/2-4.jpg)
  - CIFAR和SVHN数据集的错误率(％)。 k表示网络的增长率。
  - 超越所有竞争方法的结果以粗体表示，整体最佳效果标记为蓝色。
  -  “+”表示标准数据增强（翻转和/或镜像）。“*”表示我们的运行结果。
  - 没有数据增强(C10，C100，SVHN)的DenseNets测试都使用Dropout。
  - 使用比ResNet少的参数，DenseNets可以实现更低的错误率。没有数据增强，DenseNet的表现更好。

- **ph_2**

  `Accuracy. Possibly the most noticeable trend may originate from the bottom row of Table 2, which shows that DenseNet-BC with L = 190 and k = 40 outperforms the existing state-of-the-art consistently on all the CIFAR datasets. Its error rates of 3.46% on C10+ and 17.18% on C100+ are significantly lower than the error rates achieved by wide ResNet architecture [42]. Our best results on C10 and C100 (without data augmentation) are even more encouraging: both are close to 30% lower than FractalNet with drop-path regularization [17]. On SVHN, with dropout, the DenseNet with L = 100 and k = 24 also surpasses the current best result achieved by wide ResNet. However, the 250-layer DenseNet-BC doesn’t further improve the performance over its shorter counterpart. This may be explained by that SVHN is a relatively easy task, and extremely deep models may overfit to the training set.`

- **理解**

  - **准确率**：可能最明显的趋势在表2的底行，L=190，k=40的DenseNet-BC优于所有CIFAR数据集上现有的一流技术。
  - 它的C10+错误率为3.46％，C100+的错误率为17.18％，明显低于Wide ResNet架构的错误率。
  - 我们在C10和C100（没有数据增强）上取得的最佳成绩更令人鼓舞：两者的错误率均比带有下降路径正则化的FractalNet下降了接近30％。
  - 在SVHN上，在使用Dropout的情况下，L=100，k=24的DenseNet也超过Wide ResNet成为了当前的最佳结果。
  - 然而，相对于层数较少的版本，250层DenseNet-BC并没有进一步改善其性能。这可以解释为SVHN是一个相对容易的任务，极深的模型可能会过拟合训练集。

- **ph_3**

  `Capacity. Without compression or bottleneck layers, there is a general trend that DenseNets perform better as L and k increase. We attribute this primarily to the corresponding growth in model capacity. This is best demonstrated by the column of C10+ and C100+. On C10+, the error drops from 5.24% to 4.10% and finally to 3.74% as the number of parameters increases from 1.0M, over 7.0M to 27.2M. On C100+, we observe a similar trend. This suggests that DenseNets can utilize the increased representational power of bigger and deeper models. It also indicates that they do not suffer from overfitting or the optimization difficulties of residual networks [11].`

- **理解**

  - **模型容量：**在没有压缩或瓶颈层的情况下，总体趋势是DenseNet在LL和kk增加时表现更好。我们认为这主要是模型容量相应地增长。
  - 从表2中C10+和C100+列可以看出。在C10+上，随着参数数量从1.0M，增加到7.0M，再到27.2M，误差从5.24％，下降到4.10％，最终降至3.74％。
  - 在C100 +上，我们也可以观察到类似的趋势。这表明DenseNet可以利用更大更深的模型提高其表达学习能力。也表明它们不会受到类似ResNet那样的过度拟合或优化困难的影响。

- **ph_4**

  `Parameter Efficiency. The results in Table 2 indicate that DenseNets utilize parameters more efficiently than alternative architectures (in particular, ResNets). The DenseNetBC with bottleneck structure and dimension reduction at transition layers is particularly parameter-efficient. For example, our 250-layer model only has 15.3M parameters, but it consistently outperforms other models such as FractalNet and Wide ResNets that have more than 30M parameters. We also highlight that DenseNet-BC with L = 100 and k = 12 achieves comparable performance (e.g., 4.51% vs 4.62% error on C10+, 22.27% vs 22.71% error on C100+) as the 1001-layer pre-activation ResNet using 90% fewer parameters. Figure 4 (right panel) shows the training loss and test errors of these two networks on C10+. The 1001-layer deep ResNet converges to a lower training loss value but a similar test error. We analyze this effect in more detail below.`

- **理解**

  - **参数效率：**表2中的结果表明，DenseNet比其它架构（特别是ResNet）更有效地利用参数。
  - 具有压缩和瓶颈层结构的DenseNet-BC参数效率最高。例如，我们的250层模型只有15.3M个参数，但它始终优于其他模型，如FractalNet和具有超过30M个参数的Wide ResNet。
  - 还需指出的是，与1001层的预激活ResNet相比，具有L=100，k=12的DenseNet-BC实现了相当的性能（例如，对于C10+，错误率分别为4.62％和4.51％，而对于C100+，错误率分别为22.71％和22.27％）但参数数量少90％。
  - 图4中右图显示了这两个网络在C10+上的训练误差和测试误差。 1001层深ResNet收敛到了更低的训练误差，但测试误差却相似。我们在下面更详细地分析这个效果。

- **ph_5**

  `Overfitting. One positive side-effect of the more efficient use of parameters is a tendency of DenseNets to be less prone to overfitting. We observe that on the datasets without data augmentation, the improvements of DenseNet architectures over prior work are particularly pronounced. On C10, the improvement denotes a 29% relative reduction in error from 7.33% to 5.19%. On C100, the reduction is about 30% from 28.20% to 19.64%. In our experiments, we observed potential overfitting in a single setting: on C10, a 4× growth of parameters produced by increasing k = 12 to k = 24 lead to a modest increase in error from 5.77% to 5.83%. The DenseNet-BC bottleneck and compression layers appear to be an effective way to counter this trend.`

- **理解**

  - **过拟合：**更有效地使用参数的一个好处是DenseNet不太容易过拟合。
  - 我们发现，在没有数据增强的数据集上，DenseNet相比其它架构的改善特别明显。
  - 在C10上，错误率从7.33％降至5.19％，相对降低了29％。在C100上，错误率从28.20％降至19.64％，相对降低约30％。
  - 在我们的实验中，我们观察到了潜在的过拟合：在C10上，通过将k=12增加到k = 24使参数数量增长4倍，导致误差略微地从5.77％增加到5.83％。 DenseNet-BC的压缩和瓶颈层似乎是抑制这一趋势的有效方式。

#### 4.4 Classification Results on ImageNet

- **ph_1**

  `We evaluate DenseNet-BC with different depths and growth rates on the ImageNet classification task, and compare it with state-of-the-art ResNet architectures. To ensure a fair comparison between the two architectures, we eliminate all other factors such as differences in data preprocessing and optimization settings by adopting the publicly available Torch implementation for ResNet by [8] 1 . We simply replace the ResNet model with the DenseNetBC network, and keep all the experiment settings exactly the same as those used for ResNet.`

- **理解**

  - 我们在ImageNet分类任务上评估不同深度和增长率的DenseNet-BC，并将其与最先进的ResNet架构进行比较。
  - 为了确保两种架构之间的公平对比，我们采用Facebook提供的ResNet的Torch实现来消除数据预处理和优化设置之间的所有其他因素的影响。
  - 我们只需将ResNet替换为DenseNet-BC，并保持所有实验设置与ResNet所使用的完全相同。

- **ph_2**

  `We report the single-crop and 10-crop validation errors of DenseNets on ImageNet in Table 3. Figure 3 shows the single-crop top-1 validation errors of DenseNets and ResNets as a function of the number of parameters (left) and FLOPs (right). The results presented in the figure reveal that DenseNets perform on par with the state-of-the-art ResNets, whilst requiring significantly fewer parameters and computation to achieve comparable performance. For example, a DenseNet-201 with 20M parameters model yields similar validation error as a 101-layer ResNet with more than 40M parameters. Similar trends can be observed from the right panel, which plots the validation error as a function of the number of FLOPs: a DenseNet that requires as much computation as a ResNet-50 performs on par with a ResNet-101, which requires twice as much computation.`

- **理解**

  - 在ImageNet上DenseNets的测试结果如下表（表3）所示：（ImageNet验证集上的top-1和top-5错误率，测试分别使用了single-crop和10-crop）

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/2-5.png)
  - 将DenseNet和ResNet在验证集上使用single-crop进行测试，把top-1错误率作为参数数量（左）和计算量（右）的函数，结果如下图（图3）所示：

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/2-6.png)
  - 图中显示的结果表明，在DenseNet与最先进的ResNet验证误差相当的情况下，DensNet需要的参数数量和计算量明显减少。
  - 例如，具有20M个参数的DenseNet-201与具有超过40M个参数的101层ResNet验证误差接近。
  - 从右图可以看出类似的趋势，它将验证错误作为计算量的函数：DenseNet只需要与ResNet-50相当的计算量，就能达到与ResNet-101接近的验证误差，而ResNet-101需要2倍的计算量。

- **ph_3**

  `It is worth noting that our experimental setup implies that we use hyperparameter settings that are optimized for ResNets but not for DenseNets. It is conceivable that more extensive hyper-parameter searches may further improve the performance of DenseNet on ImageNet.`

- **理解**

  - 值得注意的是，我们的实验设置意味着我们使用针对ResNet优化的超参数设置，而不是针对DenseNet。
  - 可以想象，可以探索并找到更好的超参数设置以进一步提高DenseNet在ImageNet上的性能。（我们的DenseNet实现显存使用效率不高，暂时不能进行超过30M参数的实验。）

### 5. Discussion

- `Superficially, DenseNets are quite similar to ResNets: Eq. (2) differs from Eq. (1) only in that the inputs to H(·) are concatenated instead of summed. However, the implications of this seemingly small modification lead to substantially different behaviors of the two network architectures.`
- 表面上，DenseNet与ResNet非常相似：公式(2)与公式(1)的区别仅仅是输入被串联而不是相加。然而，这种看似小的修改导致这两种网络架构产生本质上的不同。
- `Model compactness. As a direct consequence of the input concatenation, the feature-maps learned by any of the DenseNet layers can be accessed by all subsequent layers. This encourages feature reuse throughout the network, and leads to more compact models.`
- **模型紧凑性：**作为输入串联的直接结果，任何DenseNet层学习的特征图可以由所有后续层访问。这有助于整个网络中的特征重用，并产生更紧凑的模型。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/2-7.png)

- `The left two plots in Figure 4 show the result of an experiment that aims to compare the parameter efficiency of all variants of DenseNets (left) and also a comparable ResNet architecture (middle). We train multiple small networks with varying depths on C10+ and plot their test accuracies as a function of network parameters. In comparison with other popular network architectures, such as AlexNet [16] or VGG-net [29], ResNets with pre-activation use fewer parameters while typically achieving better results [12]. Hence, we compare DenseNet (k = 12) against this architecture. The training setting for DenseNet is kept the same as in the previous section.`
- 如上图（图4）所示，左边两个图显示了一个实验的结果，其目的是比较DenseNet（左）的所有变体的参数效率以及可比较的ResNet架构（中间）。我们对C10 +上的不同深度的多个小网络进行训练，并将其测试精度作为网络参数量的函数进行绘制。与其他流行的网络架构（如AlexNet 或VGG）相比，虽然具有预激活的ResNet使用更少的参数，但通常会获得更好的结果。因此，我们将这个架构和DenseNet(k=12)进行比较。 DenseNet的训练设置与上一节保持一致。
- `The graph shows that DenseNet-BC is consistently the most parameter efficient variant of DenseNet. Further, to achieve the same level of accuracy, DenseNet-BC only requires around 1/3 of the parameters of ResNets (middle plot). This result is in line with the results on ImageNet we presented in Figure 3. The right plot in Figure 4 shows that a DenseNet-BC with only 0.8M trainable parameters is able to achieve comparable accuracy as the 1001-layer (pre-activation) ResNet [12] with 10.2M parameters.`
- 图表显示，DenseNet-BC始终是DenseNet最具参数效率的变体。此外，为了达到相同的准确度，DenseNet-BC只需要ResNet参数数量的大约1/3（中间图）。这个结果与图3所示的ImageNet结果一致。图4中的右图显示，只有0.8M可训练参数的DenseNet-BC能够实现与1001层（预激活）ResNet相当的精度， ResNet 具有10.2M参数。
- `Implicit Deep Supervision. One explanation for the improved accuracy of dense convolutional networks may be that individual layers receive additional supervision from the loss function through the shorter connections. One can interpret DenseNets to perform a kind of “deep supervision”. The benefits of deep supervision have previously been shown in deeply-supervised nets (DSN; [20]), which have classifiers attached to every hidden layer, enforcing the intermediate layers to learn discriminative features.`
- 隐性深度监督：密集卷积网络提高精度的一个可能的解释是，各层通过较短的连接从损失函数中接收额外的监督。可以认为DenseNet执行了一种“深度监督”。在深度监督的网络（DSN）中显示了深度监督的好处，其中每个隐藏层都附有分类器，迫使中间层去学习不同的特征。
- `DenseNets perform a similar deep supervision in an implicit fashion: a single classifier on top of the network provides direct supervision to all layers through at most two or three transition layers. However, the loss function and gradient of DenseNets are substantially less complicated, as the same loss function is shared between all layers.`
- 密集网络以一种简单的方式进行类似的深度监督：网络上的单一分类器通过最多两个或三个过渡层对所有层进行直接监督。然而，DenseNet的损失函数和形式显然不那么复杂，因为所有层之间共享相同的损失函数。
- `Stochastic vs. deterministic connection. There is an interesting connection between dense convolutional networks and stochastic depth regularization of residual networks [13]. In stochastic depth, layers in residual networks are randomly dropped, which creates direct connections be-tween the surrounding layers. As the pooling layers are never dropped, the network results in a similar connectivity pattern as DenseNet: there is a small probability for any two layers, between the same pooling layers, to be directly connected—if all intermediate layers are randomly dropped. Although the methods are ultimately quite different, the DenseNet interpretation of stochastic depth may provide insights into the success of this regularizer.`
- 随机连接与确定连接：密集卷积网络与残差网络的随机深度正则化之间有一个有趣的联系。在随机深度中，通过层之间的直接连接，残差网络中的层被随机丢弃。由于池化层不会被丢弃，网络会产生与DenseNet类似的连接模式：如果所有中间层都是随机丢弃的，那么在相同的池化层之间的任何两层直接连接的概率很小，尽管这些方法最终是完全不同的，但DenseNet对随机深度的解释可以为这种正规化的成功提供线索。

- `Feature Reuse. By design, DenseNets allow layers access to feature-maps from all of its preceding layers (although sometimes through transition layers). We conduct an experiment to investigate if a trained network takes advantage of this opportunity. We first train a DenseNet on C10+ with L = 40 and k = 12. For each convolutional layer  within a block, we compute the average (absolute) weight assigned to connections with layer s. Figure 5 shows a heat-map for all three dense blocks. The average absolute weight serves as a surrogate for the dependency of a convolutional layer on its preceding layers. A red dot in position (, s) indicates that the layer  makes, on average, strong use of feature-maps produced s-layers before. Several observations can be made from the plot:`
- 特征重用：通过设计，DenseNet允许层访问来自其所有先前层的特征映射（尽管有时通过过渡层）。我们进行实验，探查训练后的网络是否利用了这一特性。我们首先在C10+上，使用超参数：L=40,k=12训练了一个DenseNet。对于任一块内的每个卷积层ℓ，我们计算分配给与层s连接的平均（绝对）权重。
- 如下图（图5）所示：（卷积层在训练后的DenseNet中的平均绝对卷积核权重。像素(s,ℓ)表示在同一个密集块中，连接卷积层s与ℓ的权重的平均L1范数（由输入特征图的数量归一化）。由黑色矩形突出显示的三列对应于两个过渡层和分类层。第一行表示的是连接到密集块的输入层的权重。）

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/2-8.png)

- 图5显示了所有三个密集块的热度图。平均绝对权重反映了层之间的依赖性。在坐标(s,ℓ)上的红点表示层ℓ高度依赖前s层生成的特征图。从图中可以看出：
  - 任一层都在同一密集块内的许多输入上更新它们的权重。这表明，在同一密集块中，由先前层提取的特征实际上被后续层直接使用。
  - 过渡层也在先前密集块内的所有层的输出上更新它的权重，这表明，信息从DenseNet的第一层到最后层进通过很少的间接传播。
  - 第二和第三密集块内的层一致地将最小的权重分配给过渡层（三角形的顶行）的输出，表明过渡层输出许多冗余特征（平均权重较小） 。这与DenseNet-BC的强大结果保持一致，其中这些输出被压缩。
  - 虽然最右边的分类层也在整个密集块中使用权重，但似乎集中在最终的特征图上，这表明最终的特征图中可能会出现更多的高级特征。

### 6. Conclusion

- `We proposed a new convolutional network architecture, which we refer to as Dense Convolutional Network (DenseNet). It introduces direct connections between any two layers with the same feature-map size. We showed that DenseNets scale naturally to hundreds of layers, while exhibiting no optimization difficulties. In our experiments,DenseNets tend to yield consistent improvement in accuracy with growing number of parameters, without any signs of performance degradation or overfitting. Under multiple settings, it achieved state-of-the-art results across several highly competitive datasets. Moreover, DenseNets require substantially fewer parameters and less computation to achieve state-of-the-art performances. Because we adopted hyperparameter settings optimized for residual networks in our study, we believe that further gains in accuracy of DenseNets may be obtained by more detailed tuning of hyperparameters and learning rate schedules.`
- 我们提出了一个新的卷积网络架构，我们称之为密集卷积网络（DenseNet）。它引入了具有相同特征图大小的任意两个层之间的直接连接。我们发现，DenseNet可以轻易地扩展到数百层，而没有优化困难。在我们的实验中，DenseNet趋向于在不断增加的参数数量上提高准确性，没有任何性能下降或过度拟合的迹象。在多种不同的超参数设置下，在多个竞争激烈的数据集上获得了领先的结果。此外，DenseNet需要更少的参数和更少的计算来达到领先的性能。因为我们在研究中采用了针对残差网络优化的超参数设置，我们认为，通过更详细地调整超参数和学习速率表，可以获得DenseNet的精度进一步提高。
- `Whilst following a simple connectivity rule, DenseNets naturally integrate the properties of identity mappings, deep supervision, and diversified depth. They allow feature reuse throughout the networks and can consequently learn more compact and, according to our experiments, more accurate models. Because of their compact internal representations and reduced feature redundancy, DenseNets may be good feature extractors for various computer vision tasks that build on convolutional features, e.g., [4, 5]. We plan to study such feature transfer with DenseNets in future work.`
- 虽然遵循简单的连接规则，DenseNet自然地整合了恒等映射，深度监督和多样化深度的属性。它们允许在整个网络中进行特征重用，从而可以学习更紧凑的，并且根据我们的实验，更准确的模型。由于其紧凑的内部表示和减少了特征冗余，DenseNet可能是建立在卷积特征上的各种计算机视觉任务的良好特征提取器，例如，我们计划在未来的工作中使用DenseNets研究这种特征的转移。