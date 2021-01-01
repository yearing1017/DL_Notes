### 1. 前言

- 本文为R-CNN系列第三篇：Faster R-CNN，在Faster R-CNN基础上进行改进
- 原文地址：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

### 2. Faster R-CNN

#### 2.1 改进点

- **Fast R-CNN中region proposals是用SS算法产生的，跑在CPU上，消耗了大量时间，作者就想：为什么不用神经网络来选择region proposals呢。这样就能都跑在GPU上。注意不单单是训练一个网络来产生region proposals，更重要的是单独训练一个网络就错过了共享计算的好处。于是，面临的两个问题：**
  - 使用名为RPN的网络产生region proposals
  - RPN还要和之前的Fast共享底层计算
- 下面一张图展示了R-CNN系列三者的差异：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-1.jpg)

#### 2.2 Faster R-CNN基本结构

- 论文中给出的网络基础结构如下图所示：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-2.jpg" style="zoom: 50%;" />

- Faster RCNN其实可以分为**4个主要内容：**
  - Conv layers。作为一种CNN网络目标检测方法，**Faster RCNN首先使用一组基础的conv + relu +pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。**
  - Region Proposal Networks。**RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。**
  - Roi Pooling。**该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。**
  - Classification。**利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。**
- 下图展示了python版本中的VGG16模型中的faster_rcnn的网络结构：**其中红色部分就是新增的RPN（用来产生Region Proposals），绿色部分是Fast中就有的。**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-3.jpg)

#### 2.3 Conv layers

- Conv layers包含了conv，pooling，relu三种层。以上图中的网络结构为例，如图2，Conv layers部分共有13个conv层，13个relu层，4个pooling层。在Conv layers中：
  - **所有的conv层都是：kernel_size=3，pad=1，stride=1**
  - **所有的pooling层都是：kernel_size=2，pad=0，stride=2**

- 在Faster RCNN Conv layers中对所有的卷积都做了扩边处理（ pad=1，即填充一圈0），原图变为 (M+2)x(N+2)大小，再做3x3卷积后输出MxN 。Conv layers中的conv层不改变输出矩阵大小。如下图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-4.jpg" style="zoom:50%;" />

- 类似的是，Conv layers中的pooling层kernel_size=2，stride=2。这样每个经过pooling层的MxN矩阵，都会变为(M/2)x(N/2)大小。综上所述，**在整个Conv layers中，conv和relu层不改变输入输出大小，只有pooling层使输出长宽都变为输入的1/2。一个MxN大小的矩阵经过Conv layers固定变为(M/16)x(N/16)**

#### 2.4 Region Proposal Networks(RPN)

- 经典的检测方法生成检测框都非常耗时，如OpenCV adaboost使用滑动窗口+图像金字塔生成检测框；或如R-CNN使用SS(Selective Search)方法生成检测框。而**Faster RCNN则抛弃了传统的滑动窗口和SS方法，直接使用RPN生成检测框，这也是Faster R-CNN的巨大优势，能极大提升检测框的生成速度。**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-5.jpg)

- 上图展示了RPN网络的具体结构。可以看到**RPN网络实际分为2条线**：
  - 上面一条**通过softmax分类anchors获得positive和negative分类**
  - 下面一条用于**计算对于anchors的bounding box regression偏移量，以获得精确的proposal**
  - 而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。

##### Anchors的概念

- RPN中一个重要的概念。在feature map中**每个点都会预测k个anchor boxes**，这些box是在架构图中MxN的图像上的，相当于预选的ROI。同时这些**box都是以feature map的每点为中心(可以映射到输入图上)，且其大小和长宽比都是事先固定的(论文中使用了三种面积以及三种长宽比，即k=9)**，所以这些anchor boxes都是确定的。下面是一个点的九个anchor boxes：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-6.jpg" style="zoom:50%;" />

- **使用多个anchor的目的是引入多尺度，论文中实际上在feature map上使用了nXn的sliding window，所以anchor boxes的中心是sliding window的中心：**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-7.jpg" style="zoom:50%;" />

- 红色的框就是sliding window（注意sliding window只是选择位置，除此之外没有其它作用，和后面的3x3的卷积区分），大小为n x n（实现时n=3）

- 如上图：Convs提取到的归一化图像（resize过后的图像，即架构中M x N的图）的feature map有256个通道。然后**RPN想在这个feature map（视为一张有256个通道的图片）上滑动一个小网络来对其每个点在归一化图像上对应的9个anchor boxes进行打分（判断每个box是否为前景）和回归（每个box的位置修正意见）。**对应上图就是**2k和4k的输出（k=9），2代表了前景背景，4代表了修正意见的4个值（见R-CNN）。**

- **解释结构图中18和36两个数字的来源：**在特征图上每个点都会取9个anchor，而对于第一条支线，即判断前背景来讲是一个2分类问题，对256-d的特征图使用1x1x18的卷积，得到每个位置9个anchor的分类，所以是18。而对于第二条支线来讲，会得到修正意见的4个坐标，对256-d的特征图使用1x1x36的卷积，得到每个位置的anchor的修正意见。大致如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-8.jpg)

- 上面所谓的第一个位置，第几个anchor，anchor的哪个参数都是我自己理解的一种方式，实际的排序可能不是这样，这样写只是想说**最后那18个卷积核和36个卷积核都有自己的任务：针对同一位置不同Anchor或者同一Anchor的不同的指标（是否是前景，以及回归意见等）**

##### Proposal Layer

- Proposal Layer负责**综合修正变换量和positive anchors**，计算出精准的proposal，送入后续RoI Pooling Layer。Proposal Layer有**3个输入：positive vs negative anchors分类器结果、对应的bbox reg的修正坐标、以及im_info。**

- **im_info：**对于一副任意大小PxQ图像，传入Faster RCNN前首先reshape到固定MxN，im_info=[M, N, scale_factor]则保存了此次缩放的所有信息。

##### RPN网络结构总结

- **生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals**

#### 2.5 RoI pooling

- 有关此部分见[博文Fast R-CNN](http://yearing1017.cn/2020/04/27/Fast-R-CNN/)

#### 2.6 Classification

- Classification部分利用已经获得的proposal feature maps，通过full connect层与softmax计算每个proposal具体属于那个类别（如人，车，电视等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个proposal的位置偏移量bbox_pred，用于回归更加精确的目标检测框。Classification部分网络结构如下图。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-9.jpg)

### 3. 训练过程

- 在网上找到的一张训练流程图：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/24-10.jpg)

