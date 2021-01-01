### 1. 前言

- 目标检测的**two-stage算法代表有R-CNN系列，one-stage算法代表有Yolo系列。**
- 本文为yolov1论文学习笔记，论文地址：[You Only Look Once: Uniﬁed, Real-Time Object Detection](http://xxx.itp.ac.cn/pdf/1506.02640v5)
- 参考文章：[Yolov1详细解读](https://zhuanlan.zhihu.com/p/46691043)、[Yolo三部曲解读——Yolov1](https://zhuanlan.zhihu.com/p/70387154)

### 2. Introduction

- 核心思想：**整张图片作为网络的输入（类似于Faster-RCNN），在输出层对BBox的位置和类别进行回归。**

  > We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. Using our system, you only look once (YOLO) at an image to predict what objects are present and where they are.

- **Yolo对比R-CNN的优势：**

  > First, YOLO is extremely fast

  > Second, YOLO reasons globally about the image when making predictions. Unlike sliding window and region proposal-based techniques, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance. Fast R-CNN, a top detection method [14], mistakes background patches in an image for objects because it can’t see the larger context. YOLO makes less than half the number of background errors compared to Fast R-CNN.

  - **与基于滑动窗口和区域提议的技术不同，YOLO在训练和测试期间会看到整个图像，因此它隐式地编码有关类及其外观的上下文信息。 Faster R-CNN是一种top检测方法[14]，因为它看不到较大 的上下文，所以将图像中的背景色块误认为是对象。 与Fast R-CNN相比，YOLO产生的背景错误少于一半。**

  > Third, YOLO learns generalizable representations of objects. When trained on natural images and tested on artwork, YOLO outperforms top detection methods like DPM and R-CNN by a wide margin.

  - **第三，YOLO学习对象的可概括表示。 在自然图像上进行训练并在艺术品上进行测试时，YOLO在很大程度上优于DPM和R-CNN等顶级检测方法。**

### 3. Uniﬁed Detection

> Our system divides the input image into an S × S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.

- **我们的系统将输入图像划分为S×S网格。 如果对象的中心落入网格单元，则该网格单元负责检测该对象。**

- **每个网络需要预测B个BBox的位置信息和confidence（置信度）信息，一个BBox对应着四个位置信息和一个confidence信息。confidence代表了所预测的box中含有object的置信度和这个box预测的有多准两重信息：公式定义如下：**

$$
\operatorname{Pr}(\text { Object }) * \mathrm{IOU}_{\text {pred }}^{\text {truth }}
$$

- 其中如果有object落在一个grid cell里，第一项取1，否则取0。 第二项是预测的bounding box和实际的groundtruth之间的IoU值。
- **每个bounding box要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类**。则SxS个网格，每个网格要预测B个bounding box还要预测C个categories。输出就是**S x S x (5*B+C)**的一个tensor。（**注意：class信息是针对每个网格的，confidence信息是针对每个bounding box的。**）

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/yearing1017/j2.jpg" style="zoom:50%;" />

- 举例说明: 在PASCAL VOC中，图像输入为448x448，取S=7，B=2，一共有20个类别(C=20)。则输出就是7x7x30的一个tensor。整个网络结构如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/yearing1017/j3.png)

- 在test的时候，每个网格预测的class信息和bounding box预测的confidence信息相乘，就得到每个bounding box的class-specific confidence score:

$$
\operatorname{Pr}\left(\text { Class }_{i} \mid \text { Object }\right) * \operatorname{Pr}(\text { Object }) * \text { IOU }_{\text {pred }}^{\text {truth }}=\operatorname{Pr}\left(\text { Class }_{i}\right) * IOU_{pred}^{truth}
$$

- 等式**左边第一项就是每个网格预测的类别信息，第二三项就是每个bounding box预测的confidence。这个乘积即encode了预测的box属于某一类的概率，也有该box准确度的信息**。

- 得到每个box的class-specific confidence score以后，设置阈值，滤掉得分低的boxes，对保留的boxes进行NMS处理，就得到最终的检测结果。

#### 3.1 Network Design

> We implement this model as a convolutional neural network and evaluate it on the PASCAL VOC detection dataset [9]. The initial convolutional layers of the network extract features from the image while the fully connected layers predict the output probabilities and coordinates.

- **网络的初始卷积层从图像中提取特征，而完全连接层则预测输出概率和坐标**。

- 完整的网络结构图如上图3所示。

#### 3.2 Training

- **损失函数**

> We optimize for sum-squared error in the output of our model. We use sum-squared error because it is easy to optimize, however it does not perfectly align with our goal of maximizing average precision. It weights localization error equally with classiﬁcation error which may not be ideal. Also, in every image many grid cells do not contain any object. This pushes the “conﬁdence” scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability, causing training to diverge early on.

- **我们使用平方和误差是因为它易于优化，但它与我们实现平均精度最大化的目标并不完全一致**。 它对定位误差和分类误差的权重相等，这可能不理想。 同样，在每个图像中，许多网格单元都不包含任何对象。 这将这些单元格的“置信度”得分推向零，这通常会overpower确实包含对象的单元格的梯度。 这可能会导致模型不稳定，从而导致训练在早期就出现分歧。

> To remedy this, we increase the loss from bounding box coordinate predictions and decrease the loss from conﬁdence predictions for boxes that don’t contain objects. We use two parameters, λ coord and λ noobj to accomplish this. We set λ coord = 5 and λ noobj = .5.

- 为了解决这个问题，对于不包含对象的box，我们增加了边界框坐标预测的损失，并减少了置信度预测的损失。 我们使用两个参数λcoord和λnoobj来完成此操作。 我们将λcoord = 5和λnoobj = 0.5设置。
- 作者简单粗暴的全部采用了**sum-squared error loss**来做损失函数。
- **这种做法存在以下几个问题：**
  - 8维的localization error和20维的classification error同等重要显然是不合理的；
  - 一个网格中没有object（一幅图中这种网格很多），那么就会将这些网格中的box的confidence push到0，相比于较少的有object的网格，这种做法是overpowering的，这会导致网络不稳定甚至发散。

- **解决办法：**
  - 更重视8维的坐标预测，给这些损失前面赋予更大的loss weight。
  - 对没有object的box的confidence loss，赋予小的loss weight。
  - 有object的box的confidence loss和类别的loss的loss weight正常取1。
- 对不同大小的box预测中，相比于大box预测偏一点，小box预测偏一点肯定更不能被忍受的。而sum-square error loss中对同样的偏移loss是一样。
- 为了缓和这个问题，作者用了一个比较取巧的办法，就是将box的width和height取平方根代替原本的height和width。

- 一个网格预测多个box，希望的是每个box predictor专门负责预测某个object。具体做法就是看当前预测的box与ground truth box中哪个IoU大，就负责哪个。这种做法称作box predictor的specialization。
- 最后整个的损失函数如下所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/yearing1017/j4.png)

- 这个损失函数中：
- 只有当某个网格中有object的时候才对classification error进行惩罚。
- 只有当某个box predictor对某个ground truth box负责的时候，才会对box的coordinate error进行惩罚，而对哪个ground truth box负责就看其预测值和ground truth box的IoU是不是在那个cell的所有box中最大。

#### 3.3 缺点

- 由于**输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率**。
- 虽然每个格子可以预测B个bounding box，但是**最终只选择只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体**。当物体占画面比例较小，如图像中包含畜群或鸟群时，每个格子包含多个物体，但却只能检测出其中一个。这是YOLO方法的一个缺陷。
- YOLO loss函数中，大物体IOU误差和小物体IOU误差对网络训练中loss贡献值接近（虽然采用求平方根方式，但没有根本解决问题）。因此，对于小物体，小的IOU误差也会对网络优化过程造成很大的影响，从而降低了物体检测的定位准确性。