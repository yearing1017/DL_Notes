### 1. 前言

- 入坑R-CNN系列第二篇：[Fast R-CNN](http://cn.arxiv.org/pdf/1504.08083v2)

- **Fast R-CNN较R-CNN的改进：**
  - 很大程度上实现了**end to end**（除了region proposals的产生还是用的selective search）
  - 不再是将region proposals依次通过CNN，而是**直接输入原图，来提取特征**
  - 网络**同时输出类别判断以及bbox回归建议（即两个同级输出）**，不再分开训练SVM和回归器

### 2. SPP-Net

- 在**R-CNN网络结构模型中，由于卷积神经网络的全连接层对于输入的图像尺寸有限制，所以所有候选区域的图像都必须经过变形转换后才能交由卷积神经网络模型进行特征提取，但是无论采用剪切(crop)还是采用变形(warp)的方式，都无法完整保留原始图像信息。**
- 何凯明等人提出的空间金字塔池化层(Spatial Pyramid Pooling Layer)有效地解决了传统卷积神经网络对输入图像的尺寸的限制。

- 论文链接：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)

- R-CNN需要对候选区域进行缩放的原因是全连接层的输入维度必须固定。整个网络包含底部的卷积层和顶部的全连接层，卷积层能够适应任意尺寸的输入图像，产生相应维度的特征图，但是全连接层不同，**全连接层的参数是神经元对于所有输入的连接权重，即如果输入维度不固定，全连接层的参数数量也无法确定，网络将无法训练。**为了既能固定全连接层的输入维度又不让候选区域产生畸变，很自然的想法就是**在卷积层和全连接层的衔接处加入一个新的网络层，使得通过该层后特征的维度可以固定**，在SPP-net中引入的空间金字塔池化层(Spatial Pyramid Pooling Layer, SPP Layer)就是这样一种网络层，SPP-net也因此得名。

- SPP-net的结构如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/23-1.png)

- 如上图所示，输入任意尺度的待测图像，用CNN提取得到卷积层特征(例如VGG16最后的卷积层为Conv5_3，得到256幅特征图)。然后将不同大小候选区域的坐标投影到特征图上得到对应的窗口(window)，**将每个window均匀划分为4x4, 2x2, 1x1的块，然后对每个块使用Max-Pooling下采样，这样无论window大小如何，经过SPP层之后都得到了一个固定长度为(4x4+2x2+1)x256维的特征向量，将这个特征向量作为全连接层的输入进行后续操作。**这样就能保证**只对图像提取一次卷积层特征，同时全连接层的输入维度固定。**
- 如下图，可以清晰看出**使用SPP模块与R-CNN的区别：**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/23-2.png)

### 3. Fast R-CNN

- **论文中给出的Fast R-CNN实验流程图：**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/23-3.jpg" style="zoom:50%;" />

- **对于上图的解释如下：**
  - Fast R-CNN的输入由两部分组成：**原图**和在**原图上提取到的P个ROI**
  - **ROI的解释如下：**由于存在多个候选区域，系统会有一个甄别，判断出感兴趣区域，也就是Region of Interest, RoI，**ROI的产生还是用的selective search**
  - **RoI池化层是SSP(Spatial Pyramid Pooling)层的特殊情况**，它可以**从特征图中提取一个固定长度的特征向量。每个特征向量，都会被输送到全连接(FC)层序列中，这个FC分支成两个同级输出层**
  - 其中一层的功能是**进行分类，对目标关于K个对象类(包括全部”背景”类)输出每一个RoI的概率分布，也就是产生softmax概率估计；**
  - 另一层是为了输出K个对象中每一个类的四个实数值(bbox regression)。每4个值编码K个类中的每个类的精确边界框(bounding-box)位置
  - 整个结构是使用**多任务损失的端到端训练**(trained end-to-end with a multi-task loss)(除去Region Proposal提取阶段)。
- **Fast R-CNN结构简图：**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/23-4.jpg)

- **RoI Pooling Layer**：实际上是SPP Layer的简化版，**SPP Layer对每个候选区域使用了不同大小的金字塔映射，即SPP Layer采用多个尺度的池化层进行池化操作；而RoI Pooling Layer只需将不同尺度的特征图下采样到一个固定的尺度(例如7x7)。**例如对于VGG16网络conv5_3有512个特征图，虽然输入图像的尺寸是任意的，但是通过RoI Pooling Layer后，均会产生一个7x7x512维度的特征向量作为全连接层的输入，**即RoI Pooling Layer只采用单一尺度进行池化。**如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/23-5.png)

- 对于RoI pooling层则采用一种尺度的池化层进行下采样，将每个RoI区域的卷积特征分成4x4个bin，然后对每个bin内采用max pooling，这样就得到一共16维的特征向量。SPP层和RoI pooling层使得网络对输入图像的尺寸不再有限制，同时RoI pooling解决了SPP无法进行权值更新的问题。
- **RoI pooling层有两个主要作用：第一个：将图像中的RoI区域定位到卷积特征中的对应位置；第二个：将这个对应后的卷积特征区域通过池化操作固定到特定长度的特征，然后将该特征送入全连接层。**

- **Fast R-CNN目标检测主要流程**如下：
  - 输入一张待检测图像；
  - **提取候选区域：**利用Selective Search算法在输入图像中提取出候选区域，并把这些候选区域按照空间位置关系映射到最后的卷积特征层
  - **区域归一化：**对于卷积特征层上的每个候选区域进行RoI Pooling操作，得到固定维度的特征；
  - **分类与回归：**将提取到的特征输入全连接层，然后用Softmax进行分类，对候选区域的位置进行回归。

