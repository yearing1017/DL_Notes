### 1. 前言

- 本文是Yolo系列的第二篇[YOLO9000: Better, Faster, Stronger](http://xxx.itp.ac.cn/pdf/1612.08242)学习笔记

- Yolov2是基于Yolov1的一系列改进，之前的笔记：[Yolov1解读笔记及参考](http://yearing1017.cn/2020/07/17/YoloV1-paper/)

- 本文参考文章：[<机器爱学习>YOLOv2 / YOLO9000 深入理解](https://zhuanlan.zhihu.com/p/47575929)；[Yolo三部曲解读——Yolov2](https://zhuanlan.zhihu.com/p/74540100)

### 2. 概述

- 在论文中一开始Abstract先出现了Yolov2和Yolo9000两个概念，区别如下解释：**Yolov2和Yolo9000算法内核相同，区别是训练方式不同：Yolov2用coco数据集训练后，可以识别80个种类。而Yolo9000可以使用coco数据集 + ImageNet数据集联合训练，可以识别9000多个种类**
- Introduction中对于Yolov2、Yolo9000、和联合分类+检测作为训练集的想法：

> We propose a new method to harness the large amount of classiﬁcation data we already have and use it to expand the scope of current detection systems. Our method uses a hierarchical view of object classiﬁcation that allows us to combine distinct datasets together.
>
> We also propose a joint training algorithm that allows us to train object detectors on both detection and classiﬁcation data. Our method leverages labeled detection images to learn to precisely localize objects while it uses classiﬁcation images to increase its vocabulary and robustness.
>
> Using this method we train YOLO9000, a real-time object detector that can detect over 9000 different object categories. First we improve upon the base YOLO detection system to produce YOLOv2, a state-of-the-art, real-time detector. Then we use our dataset combination method and joint training algorithm to train a model on more than 9000 classes from ImageNet as well as detection data from COCO.

### 3. Better

- 论文中给出了下面的**改进trick列表**，列出各项改进对mAP的提升效果。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/30-1.jpg)

#### 3.1 Batch Normalization

- 批归一化使mAP有2.4的提升。
- 批归一化有**助于解决反向传播过程中的梯度消失和梯度爆炸问题**，降低对一些超参数（比如学习率、网络参数的大小范围、激活函数的选择）的敏感性，并且每个batch分别进行归一化的时候，起到了一定的正则化效果（**YOLO2不再使用dropout**），从而能够获得更好的收敛速度和收敛效果。

#### 3.2 High Resolution Classiﬁer

- mAP提升了3.7。
- 图像分类的训练样本很多，而标注了边框的用于训练对象检测的样本相比而言就比较少了，因为标注边框的人工成本比较高。所以**对象检测模型通常都先用图像分类样本训练卷积层，提取图像特征。但这引出的另一个问题是，图像分类样本的分辨率不是很高。所以YOLO v1使用ImageNet的图像分类样本采用 224 x 224 作为输入，来训练CNN卷积层。然后在训练对象检测时，检测用的图像样本采用更高分辨率的 448 x 448 的图像作为输入。但这样切换对模型性能有一定影响**。
- 所以**YOLOv2在采用 224 x 224 图像进行分类模型预训练后，再采用 448 x 448 的高分辨率样本对分类模型进行微调（10个epoch），使网络特征逐渐适应 448 x 448 的分辨率。然后再使用 448 x 448 的检测样本进行训练，缓解了分辨率突然切换造成的影响**。

#### 3.3 Convolutional With Anchor Boxes

- 召回率大幅提升到88%，同时mAP轻微下降了0.2。
- **借鉴Faster RCNN的做法，YOLOv2也尝试采用先验框（anchor）。在每个grid预先设定一组不同大小和宽高比的边框，来覆盖整个图像的不同位置和多种尺度**，这些先验框作为预定义的候选区在神经网络中将检测其中是否存在对象，以及微调边框的位置。
- 同时YOLOv2移除了全连接层。另外去掉了一个池化层，使网络卷积层输出具有更高的分辨率。
- 之前**YOLOv1并没有采用先验框，并且每个grid只预测两个bounding box，整个图像98个。YOLO2如果每个grid采用9个先验框，总共有13x13x9=1521个先验框**。所以，相对YOLO1的81%的召回率，YOLO2的召回率大幅提升到88%。同时mAP有0.2%的轻微下降。不过YOLO2接着进一步对先验框进行了改良。

#### 3.4 Dimension Clusters

- 之前先验框都是手工设定的，YOLOv2尝试统计出更符合样本中对象尺寸的先验框，这样就可以减少网络微调先验框到实际位置的难度。
- YOLOv2的做法是**对训练集中标注的边框进行聚类分析，以寻找尽可能匹配样本的边框尺寸**。原文：

> Instead of choosing priors by hand, we run k-means clustering on the training set bounding boxes to automatically ﬁnd good priors. If we use standard k-means with Euclidean distance larger boxes generate more error than smaller boxes. However, what we really want are priors that lead to good IOU scores, which is independent of the size of the box. Thus for our distance metric we use:

$$
d(\text { box }, \text { centroid })=1-\text { IOU }(\text { box }, \text { centroid })
$$

- **聚类算法最重要的是选择如何计算两个边框之间的“距离”**，对于常用的欧式距离，大边框会产生更大的误差，但我们关心的是边框的IOU。所以，YOLOv2在聚类时采用以上公式来计算两个边框之间的“距离”。 
- centroid是聚类时被选作中心的边框，box就是其它边框，d就是两者间的“距离”。**IOU越大，“距离”越近;**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/30-2.png" style="zoom:50%;" />

- 上图左边是选择不同的聚类k值情况下，得到的k个centroid边框，计算样本中标注的边框与各centroid的Avg IOU。显然，边框数k越多，Avg IOU越大。YOLOv2选择k=5作为边框数量与IOU的折中。对比手工选择的先验框，使用5个聚类框即可达到61 Avg IOU，相当于9个手工设置的先验框60.9 Avg IOU。

#### 3.5 Direct location prediction

- 借鉴于Faster RCNN的先验框方法，在训练的早期阶段，其位置预测容易不稳定。其位置预测公式为： 

$$
\begin{array}{l}x=\left(t_{x} * w_{a}\right)-x_{a} \\ y=\left(t_{y} * h_{a}\right)-y_{a}\end{array}
$$

- 其中，x, y 是预测边框的中心， $x_a, y_a$ 是先验框（anchor）的中心点坐标， $w_a,h_a$ 是先验框（anchor）的宽和高， $t_x,t_y$ 是要学习的参数。
- 由于 $t_x,t_y$ 的取值没有任何约束，因此预测边框的中心可能出现在任何位置，训练早期阶段不容易稳定。YOLO调整了预测公式，**将预测边框的中心约束在特定gird网格内**。

$$
\begin{array}{c}b_{x}=\sigma\left(t_{x}\right)+c_{x} \\ b_{y}=\sigma\left(t_{y}\right)+c_{y} \\ b_{w}=p_{w} e^{t_{w}} \\ b_{h}=p_{h} e^{t_{h}} \\ \operatorname{Pr}(\text {object}) * I O U(b, \text {object})=\sigma\left(t_{o}\right)\end{array}
$$

- 其中， $b_x,b_y,b_w,b_h$ 是预测边框的中心和宽高。 $Pr(object) * IOU(b, object)$ 是预测边框的置信度，YOLOv1是直接预测置信度的值，这里对预测参数  $t_o$ 进行σ变换后作为置信度的值。 $c_x,c_y$ 是当前网格左上角到图像左上角的距离，要先将网格大小归一化，即令一个网格的宽=1，高=1。 $p_w,p_h$ 是先验框的宽和高。 σ是sigmoid函数。 $t_x,t_y,t_w,t_h,t_o$ 是要学习的参数，分别用于预测边框的中心和宽高，以及置信度。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/30-3.png" style="zoom:50%;" />

- 参考上图，由于σ函数将 $t_x, t_y$ 约束在(0,1)范围内，所以根据上面的计算公式，预测边框的蓝色中心点被约束在蓝色背景的网格内。约束边框位置使得模型更容易学习，且预测更为稳定。

#### 3.6 Fine-Grained Features

- 对象检测面临的一个问题是图像中对象会有大有小，输入图像经过多层网络提取特征，最后输出的特征图中（比如YOLO2中输入416 x 416经过卷积网络下采样最后输出是13 x 13），较小的对象可能特征已经不明显甚至被忽略掉了。为了更好的检测出一些比较小的对象，最后输出的特征图需要保留更细节的信息。

- YOLOv2引入**一种称为passthrough层的方法在特征图中保留一些细节信息**。具体来说，就是在最后一个pooling之前，特征图的大小是26 x 26 x 512，将其1拆4，直接传递（passthrough）到pooling后（并且又经过一组卷积）的特征图，两者叠加到一起作为输出的特征图。**类似ResNet的思想。**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/30-4.png" style="zoom:50%;" />

#### 3.7 Multi-Scale Training

- 因为去掉了全连接层，YOLOv2可以输入任何尺寸的图像。因为整个网络下采样倍数是32，作者采用了{320,352,...,608}等10种输入图像的尺寸，这些尺寸的输入图像对应输出的特征图宽和高是{10,11,...19}。**训练时每10个batch就随机更换一种尺寸，使网络能够适应各种大小的对象检测。**

### 4. Faster

- 为了进一步提升速度，YOLO2提出了Darknet-19（有19个卷积层和5个MaxPooling层）网络结构。DarkNet-19比VGG-16小一些，精度不弱于VGG-16，但浮点运算量减少到约1/5，保证更快的运算速度。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/30-5.jpg)

- YOLOv2的训练主要包括三个阶段。
- 第一阶段就是**先在ImageNet分类数据集上预训练Darknet-19，此时模型输入为 224*224 ，共训练160个epochs。**
- 第二阶段将**网络的输入调整为 448x448 ，继续在ImageNet数据集上finetune分类模型，训练10个epochs，此时分类模型的top-1准确度为76.5%，而top-5准确度为93.3%。**
- 第三个阶段就是**修改Darknet-19分类模型为检测模型，移除最后一个卷积层、global avgpooling层以及softmax层，并且新增了三个 3x3x1024卷积层，同时增加了一个passthrough层，最后使用 1x1 卷积层输出预测结果，输出的channels数为：num_anchors x (5+num_classes)** ，和训练采用的数据集有关系。由于anchors数为5，对于VOC数据集（20种分类对象）输出的channels数就是125，最终的预测矩阵T的shape为 (batch_size, 13, 13, 125)，可以先将其reshape为 (batch_size, 13, 13, 5, 25) ，其中 T[:, :, :, :, 0:4] 为边界框的位置和大小 $t_x,t_y,t_w,t_h$，T[:, :, :, :, 4] 为边界框的置信度，而 T[:, :, :, :, 5:] 为类别预测值。

### 5. YOLOv2 输入->输出

- 综上所述，虽然YOLOv2做出了一些改进，但总的来说网络结构依然很简单，**就是一些卷积+pooling，从416 x 416 x 3 变换到 13 x 13 x 5 x 25。稍微大一点的变化是增加了batch normalization，增加了一个passthrough层，去掉了全连接层，以及采用了5个先验框。**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/30-6.jpg" style="zoom:50%;" />

- 对比YOLOv1的输出张量，**YOLOv2的主要变化就是会输出5个先验框，且每个先验框都会尝试预测一个对象。输出的 13*13*5*25 张量中，25维向量包含 20个对象的分类概率+4个边框坐标+1个边框置信度。**

### 6. Stronger

- Joint classification and detection
- 物体分类，是对整张图片打标签，比如这张图片中含有人，另一张图片中的物体为狗；而物体检测不仅对物体的类别进行预测，同时需要框出物体在图片中的位置。物体分类的数据集，最著名的ImageNet，物体类别有上万个，而物体检测数据集，例如coco，只有80个类别，因为物体检测、分割的打标签成本比物体分类打标签成本要高很多。所以在这里，作者提出了分类、检测训练集联合训练的方案。

- 联合训练方法思路简单清晰，Yolov2中物体矩形框生成，不依赖于物理类别预测，二者同时独立进行。
- **当输入是检测数据集时，标注信息有类别、有位置，那么对整个loss函数计算loss，进行反向传播；当输入图片只包含分类信息时，loss函数只计算分类loss，其余部分loss为零。**当然，一般的训练策略为，先在检测数据集上训练一定的epoch，待预测框的loss基本稳定后，再联合分类数据集、检测数据集进行交替训练，同时为了分类、检测数据量平衡，作者对coco数据集进行了上采样，使得coco数据总数和ImageNet大致相同。
- 联合分类与检测数据集，这里不同于将网络的backbone在ImageNet上进行预训练，预训练只能提高卷积核的鲁棒性，而分类检测数据集联合，可以扩充识别物体种类。例如，在检测物体数据集中，有类别人，当网络有了一定的找出人的位置的能力后，可以通过分类数据集，添加细分类别：男人、女人、小孩、成人、运动员等等。这里会遇到一个问题，类别之间并不一定是互斥关系，可能是包含（例如人与男人）、相交（运动员与男人），那么在网络中，该怎么对类别进行预测和训练呢？
- **Dataset combination with WordTree**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/30-7.jpg" style="zoom:50%;" />

- 树结构表示物体之间的从属关系非常合适，第一个大类，物体，物体之下有动物、人工制品、自然物体等，动物中又有更具体的分类。此时，在类别中，不对所有的类别进行softmax操作，而对同一层级的类别进行softmax：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/30-8.jpg" style="zoom:50%;" />

- 如图中所示，同一颜色的位置，进行softmax操作，使得同一颜色中只有一个类别预测分值最大。在预测时，从树的根节点开始向下检索，每次选取预测分值最高的子节点，直到所有选择的节点预测分值连乘后小于某一阈值时停止。在训练时，如果标签为人，那么只对人这个节点以及其所有的父节点进行loss计算，而其子节点，男人、女人、小孩等，不进行loss计算。
- 最后的结果是，Yolo v2可以识别超过9000个物体，作者美其名曰Yolo9000。当然原文中也提到，只有当父节点在检测集中出现过，子节点的预测才会有效。如果子节点是裤子、T恤、裙子等，而父节点衣服在检测集中没有出现过，那么整条预测类别支路几乎都是检测失效的状态。

- Yolo v2将同期业界的很多深度学习技巧结合进来，将Yolo的性能与精度又提升了一个层次。Yolo9000的物体检测分类技巧，可以很好的被运用到工程中，并不需要对所有的物体都进行费时费力的检测标注，而是同时利用分类标签与检测标签达到同样的效果，从而减轻了标注成本与时间。