### 1. 前言

- 本文为Deeplab系列的v1学习笔记，后续有v2、v3、v3+
- 本文工作**结合了深度卷积神经网络(DCNNs)和概率图模型(DenseCRFs)**的方法。
- 原文链接：[Semantic image segmentation with deep convolutional nets and fully connected CRFs](https://arxiv.org/pdf/1412.7062v3.pdf)

### 2. Abstract

- DCNNs做语义分割时精准度不够的问题，根本原因是**DCNNs的高级特征的平移不变性**，这对图像识别、分类来说是优点，但对于分割问题来讲，**分割需要精确的位置信息**。
- `This is due to the very invariance properties that make DCNNs good for high level tasks.`

- 解决这一问题的方法是通过**将DCNNs层的响应和完全连接的条件随机场（CRF）结合**
- `by combining the responses at the ﬁnal DCNN layer with a fully connected Conditional Random Field (CRF).`
- 同时模型创新性的将Hole（即**空洞卷积**）算法应用到DCNNs模型上

### 3. Introduction

- DCNN对图像变换局部区域的不变性，从而可以更好的学习抽象的信息。但另一方面却削弱了低层次类型的任务，像姿态估计，语义分割等需要精细定位的任务。
- DCNN应用于图像标记任务主要存在两个技术障碍，**下采样和空间不变性**。
- `signal downsampling, and spatial ‘insensitivity’ (invariance).`

- 第一个问题是在标准的DCNN中由于连续的池化和下采样导致单一分辨率的缺失，从而使得**位置信息丢失难以恢复，**为此，该文**引用了空洞卷积算法**

- 第二个问题是实际中我们分类器所作的是以目标物体中心决定分类的，这就决定需要空间信息的不变性，这就限制了DCNN的空间信息的准确性。**空间不变性导致细节信息丢失**。
- 该文通过**后接一个全连接的条件随机场(CRF)**来获得更加较好的细节。
- `We boost our model’s ability to capture ﬁne details by employing a fully-connected Conditional Random Field (CRF)`

- 论文的三个主要贡献：
  - 速度：借用空洞算法，可以使DCNN在8fps
  - 准确率：当时在PASCAL的语义分割集上效果最好
  - 简单的结构：DCNN和CRF的组合

### . 网络结构部分

#### 4.1 基于空洞算法的密集滑动窗来进行特征提取

- **首先将VGG-16的全连接层替换为卷积层**，然而，结果是生成的检测scores很稀疏
- 为了改善结果，于是该文**在VGG-16最后两个最大池化层后跳过下采样，同时，改变最后三层卷积层与全连接层的卷积核，在他们之间添加0来增加他们的长度**，即采用了**空洞卷积**。
- 把**最后三个卷积层`（conv5_1、conv5_2、conv5_3）`的dilate rate设置为2，且第一个全连接层的dilate rate设置为4（保持感受野）**
- 这样**可以通过保持过滤器不加改动**从而更高效的实现这个操作，而**不是分别使用2或4像素的步幅稀疏地对其应用的特征图进行采样**。

- **对VGG-16的finetune：**
  - 将其最后一层的类别数为1000的分类器替换为类别数为21的分类器（分割的一共21类）。
  - `loss function is the sum of cross-entropy terms for each spatial position in the CNN output map`
  - 运用标准的SGD优化每一层网络的权重。

#### 4.2  控制网络感受野大小

- 使用我们的网络进行密集分数计算的另一个关键因素是**明确控制网络的感受野大小**。
- **感受野越大，导致目标位置信息的损失越严重，进而定位信息不准确。**
- 最新的基于DCNN的图像识别方法依赖于在Imagenet大规模分类任务上预先训练的网络。这些网络通常具有很大的感受野大小：在我们考虑的VGG-16网络的情况下，如果网络应用卷积，其感受野是224×224（零填充）和404×404像素。
- 此外，在将网络转换为完全卷积的网络之后，第一个全连接层具有4,096个大的7×7空间大小的滤波器，并且成为我们密集分数图计算中的计算瓶颈。
- 我们已经通过将第一FC层空间抽样到4×4空间大小来解决这两个严重的实际问题。 这将网络的感受野减少到128×128（零填充）或308×308（卷积模式），并将第一个FC层的计算时间缩短了3倍。

### 5. DETAILED BOUNDARY RECOVERY : FULLY-CONNECTED CONDITIONAL RANDOM FIELDS AND MULTI-SCALE PREDICTION

#### 5.1 DEEP CONVOLUTIONAL NETWORKS AND THE LOCALIZATION CHALLENGE

- 如图2所示，**DCNN分数图可以可靠地预测图像中对象的存在和粗略位置，但不太适合用于勾画其精确轮廓。**

- 卷积网络在分类精度和定位精度之间有自然的权衡：**具有多个最大池化层的深度模型已被证明在分类任务中最成功，然而，他们增加的不变性和大的感受野使得在其最高输出层分数推断位置的问题更具挑战性**。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/9-1.jpg)

- 近期工作从两个方向来解决这个定位的挑战。 第一种方法是利用卷积网络中多层的信息来更好地估计对象边界。 第二种方法是采用超像素表示，实质上是将定位任务委托给低级分割方法。
- 在下一章节中，我们通过**结合DCNN的识别能力和完全连接的CRF的细粒度定位精度来寻求新的替代方向**，并表明它在解决定位挑战方面非常成功，产生了准确的语义分割结果， 以超出现有方法的详细程度恢复对象边界。

#### 5.2 FULLY-CONNECTED CONDITIONAL RANDOM FIELDS FOR ACCURATE LOCALIZATION



#### 5.3 MULTI-SCALE PREDICTION

- 本文探讨了一种**多尺度预测方法**来提高边界定位精度。

- 具体来说，我们**将input image和前四个最大池化层中的每一个的输出附加到一个两层MLP**，(MLP：多层感知机)，即：**在input的后面加一个两层的MLP，在四个最大池化层后面各加一个MLP，**MLP的两层参数如下：`ﬁrst layer: 128 3x3 convolutional ﬁlters, second layer: 128 1x1 convolutional ﬁlters)`，**其特征图连接`concatenated to`到主网络的最后一层特征图**。
- 因此，通过**5 x 128 = 640个通道增强了馈送到softmax层的聚合特征图**。5即一个input+4个maxpooling。
- **个人感觉这的思想有点类似UNet 和 FCN的skip思想。**
- 我们只调整新添加的权重，保留其他网络参数在之前学习到的值。如实验部分所述，从精细分辨率层引入这些额外的直接连接可以提高定位性能，但效果并不像用完全连接的CRF所获得的那样大。

### 6. EXPERIMENTAL EVALUATION 

#### 6.1 Dataset

- PASCAL VOC 2012分割基准，包含20个目标类别,1个背景类别。
- 原始数据：1464(training)，1449(validation)，1456(testing)。数据增强：通过Hariharan等提供的额外标注扩增至10582(training images)。

#### 6.2 Training

#### 6.3 Evaluation on Validation set

- 我们在PASCAL的'val'集进行了大部分的评估，并在增强的PASCAL'train'集上训练。如下表，将完全连接的CRF结合到我们的模型（由DeepLab-CRF表示）产生了显着的性能提升，大约4％的提高。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/9-2.jpg)

- 转向定性结果，我们提供了下图中的DeepLab和DeepLab-CRF之间的视觉比较。使用完全连接的CRF可显著提高结果，从而允许模型准确地捕获复杂的对象边界。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/9-3.jpg" style="zoom:50%;" />

#### 6.4 Multi-Scale features

- 我们还利用中间层的特征。
- 如6.3中 表a所示，将多尺度特征添加到我们的DeepLab模型（表示为DeepLab-MSc）可提高约1.5％的性能，并且进一步整合的完全连接的CRF（表示为DeepLab-MSc-CRF）提高约4％。
- DeepLab和DeepLab-MSc的定性比较如图4所示。利用多尺度特征可以稍微改进对象边界。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/9-5.jpg)

#### 6.5 Field of View

- 我们采用的atrous算法**允许我们通过调整输入步长来任意地控制（感受野）**（FOV）。**这里的input stride即现在的dilated rate**。
- 如下表：我们在第一个完全连接的层上尝试了几种kernel size和input stride。DeepLab-CRF-7x7是VGG-16网络的直接修改，kernel size为7x7，input stride为4。该模型在'val'集上产生了67.64％的性能，但是相对较慢（训练期间每秒1.44张图像）。
- 通过将kernel size减小到4x4，我们将模型速度提高到每秒2.9张图像。我们已经尝试了两种具有不同FOV尺寸的网络变体，DeepLab-CRF和DeepLab-CRF-4x4；后者具有大的FOV（即大的输入步长）并获得更好的性能。
- 最后，我们使用内核大小3x3，输入步幅为12，并进一步将过滤器数量从4096更改为最后两层的1024。有趣的是，由此产生的DeepLab-CRF-LargeFOV型号与昂贵的DeepLab-CRF-7x7的性能相当。同时，运行速度快3.36倍，参数明显减少（20.5M而不是134.3M）。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/9-6.jpg)

- 下面还有几个实验的比较结果综述，详见论文。
- **最好的模型是DeepLab-MSc-CRF-LargeFOV，通过采用多尺度功能和大型FOV，达到71.6％的最佳性能**

### 7. DISCUSSION

- 我们的工作**结合了卷积神经网络和完全连接的条件随机场**的想法，产生了一种新颖的方法，能够产生准确预测和详细的语义分割图，同时具有计算效率。我们的实验结果表明所提出的方法在PASCAL VOC 2012语义图像分割挑战中显著提高了结果。
- 我们打算改进模型的多个方面，例如完全整合其两个主要组件（CNN和CRF），并实现端到端的方式对整个系统进行训练。我们还计划尝试更多数据集，并将我们的方法应用于其他数据源，如深度图或视频。最近，我们还**使用弱监督标注，以边界框或图像级标签的形式进行模型训练**。

- 在较高层次上，我们的工作依赖于卷积神经网络和概率图形模型。我们计划进一步研究这两种强大的方法的相互作用，并探讨其解决具有挑战性的计算机视觉任务的协同潜力。