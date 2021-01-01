### 1. 前言

- 本篇为R-CNN系列第4篇：[Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
- Mask R-CNN在 Faster-RCNN 的基础上添加一个分支网络，在实现目标检测的同时分割目标像素

### 2. Introduction

- Mask R-CNN的方法通过**添加一个与现有目标检测框回归并行的，用于预测目标掩码的分支**来扩展Faster R-CNN，通过添加一个用于在每个感兴趣区域（RoI）上预测分割掩码的分支来扩展Faster R-CNN，就是在每个感兴趣区域（RoI）进行一个二分类的语义分割。
- 在这个感兴趣区域同时做目标检测和分割，这个**分支与用于分类和目标检测框回归的分支并行执行**，如下图所示（用于目标分割的Mask R-CNN框架）：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/25-1.jpg" style="zoom:50%;" />

- **掩码分支是作用于每个RoI的小FCN，以像素到像素的方式预测分割掩码，可是要在ROI区域进行一个mask分割**，存在一个问题，**Faster R-CNN不是为网络输入和输出之间的像素到像素对齐而设计的**，如果直接拿Faster R-CNN得到的ROI进行mask分割，那么像素到像素的分割可能不精确，因为**应用到目标检测上的核心操作执行的是粗略的空间量化特征提取，直接分割出来的mask存在错位的情况**。
- 所以作者提出了简单的，量化无关的层，称为**RoIAlign(ROI对齐)**，可以保留精确的空间位置，可以将掩码(mask)准确度提高10％至50％。

### 3. Mask R-CNN

- Faster R-CNN为每个候选目标输出**类标签和边框偏移量**。之前的[Faster R-CNN论文笔记](http://yearing1017.cn/2020/04/29/Faster-R-CNN/)

- Faster R-CNN由两个阶段组成。称**为区域提议网络（RPN）的第一阶段提出候选目标边界框**。**第二阶段，本质上是Fast R-CNN，使用RoIPool从每个候选框中提取特征，并进行分类和边界回归**。两个阶段使用的特征可以共享，以便更快的推理。

- Mask R-CNN采用相同的两个阶段，具有**相同的第一阶段（即RPN），**此步骤提出了候选对象边界框。第二阶段本质上就是Fast R-CNN，它使用来自候选框架中的**RoIPool来提取特征并进行分类和边界框回归**，**Mask R-CNN还为每个RoI输出二进制掩码。**
- **损失函数：**每个采样后的RoI上的多任务损失函数定义为：$L=L_{cls}+L_{box}+L_{mask}$
- 分类损失 $L_{cls}$和检测框损失$L_{box}$与Fast RCNN中定义的相同，掩码分支对于每个RoI的输出维度为$K \times m^2$的数目，即K个分辨率为m×m的二进制掩码(矩阵)，每个类别一个，K表示类别数量。我们为每个像素应用Sigmoid，并将$L_{mask}$定义为平均二分类交叉熵损失。对于真实类别为k的RoI，仅在第k个掩码上计算$L_{mask}$ （其他掩码输出不计入）
- 这里有一个思考的点：**这里的mask在预测时，使用的二分类损失函数+sigmoid；和一般的使用FCN网络进行分割使用多分类损失函数+softmax；这里论文中也进行了解释如下：**
- `Our deﬁnition of L mask allows the network to generate masks for every class without competition among classes; we rely on the dedicated classiﬁcation branch to predict the class label used to select the output mask. This decouples mask and class prediction. This is different from common practice when applying FCNs [30] to semantic segmentation, which typically uses a per-pixel softmax and a multinomial cross-entropy loss. In that case, masks across classes compete; in our case, with a per-pixel sigmoid and a binary loss, they do not. We show by experiments that this formulation is key for good instance segmentation results.`
- **ROIAlign**：**ROIPooling和ROIAlign的分析与比较**
- 先看下图，左图代表原始ROI对应的区域，右图代表经过中间卷积操作之后的ROI对应比例缩减后对应区域，可以发现：**与预想区域有偏差**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/25-2.jpg)

- **ROI Pooling和ROIAlign最大的区别是：前者使用了两次量化操作，而后者并没有采用量化操作，使用了线性插值算法，具体的解释如下所示**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/25-3.jpg)

- **ROI Pooling流程：**

  - 如上图所示，为了得到固定大小（7X7）的feature map，**我们需要做两次量化操作：1）图像坐标 — feature map坐标，2）feature map坐标 — ROI feature坐标。**
  - 如图我们输入的是一张800x800的图像，在图像中有两个目标（猫和狗），**狗的BB(bounding box)大小为665x665**，经过VGG16网络后，我们可以获得对应的feature map，如果我们对卷积层进行Padding操作，我们的图片经过卷积层后保持原来的大小，但是由于池化层的存在，我们最终获得feature map 会比原图缩小一定的比例，这和Pooling层的个数和大小有关。**在该VGG16中，我们使用了5个池化操作**，每个池化操作都是**2x2 Pooling**，因此我们最终获得feature map的大小为800/32 x 800/32 = 25x25（是整数）。
  - **但是将狗的BB对应到feature map上面，我们得到的结果是665/32 x 665/32 = 20.78 x 20.78，结果是浮点数，含有小数**，但是我们的像素值可没有小数，那么作者就对其进行了量化操作（即取整操作），即其结果变为20 x 20，**在这里引入了第一次的量化误差；**
  - 我们的feature map中有不同大小的ROI，但是我们后面的网络却要求我们有固定的输入，因此，我们需要将不同大小的ROI转化为固定的ROI feature，在这里使用的是7x7的ROI feature，那么我们需要将**20 x 20的ROI映射成7 x 7的ROI feature**，其结果是 **20 /7 x 20/7 = 2.86 x 2.86，同样是浮点数**，含有小数点，我们采取同样的操作对其进行取整吧，在这里引入了第二次量化误差。**其实，这里引入的误差会导致图像中的像素和特征中的像素的偏差，即将feature空间的ROI对应到原图上面会出现很大的偏差**。
  - 用我们第二次引入的误差来分析，本来是2,86，我们将其量化为2，这期间引入了0.86的误差，看起来是一个很小的误差呀，但是你要记得这是在feature空间，我们的**feature空间和图像空间是有比例关系的，在这里是1:32，那么对应到原图上面的差距就是0.86 x 32 = 27.52。**这个差距不小，这还是仅仅考虑了第二次的量化误差。这会大大影响整个检测算法的性能。

- **ROIAlign流程：**

  - 为了得到为了得到固定大小（7X7）的feature map，**ROIAlign技术并没有使用量化操作，即我们不想引入量化误差，比如665 / 32 = 20.78，我们就用20.78，不用什么20来替代它**，比如20.78 / 7 = 2.97，我们就用2.97，而不用2来代替它。
  - **解决思路是使用“双线性值”算法**。**双线性插值是一种比较好的图像缩放算法，它充分的利用了原图中虚拟点（比如20.56这个浮点数，像素位置都是整数值，没有浮点值）四周的四个真实存在的像素值来共同决定目标图中的一个像素值，即可以将20.56这个虚拟的位置点对应的像素值估计出来**。

- **网络架构：**

  - 使用ResNet的Faster R-CNN从第四阶段的最终卷积层提取特征，我们称之为C4。例如，使用ResNet-50的下层网络由ResNet-50-C4表示，还探讨了一个有效的下层网络，称为特征金字塔网络（FPN）。**FPN使用具有横旁路连接的自顶向下架构，以从单尺度输入构建网络中的特征金字塔。使用FPN的Faster R-CNN根据其尺度提取不同级别的金字塔的RoI特征**，不过其它部分和平常的ResNet类似。使用ResNet-FPN进行特征提取的Mask R-CNN可以在精度和速度方面获得极大的提升。

  <img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/25-4.jpg" style="zoom:50%;" />

  - 如上图所示，为了产生对应的Mask，文中提出了两种架构，即左边的Faster R-CNN/ResNet和右边的Faster R-CNN/FPN。对于左边的架构，**backbone使用的是预训练好的ResNet，使用了ResNet倒数第4层的网络。输入的ROI首先获得7x7x1024的ROI feature**，然后将其升维到2048个通道（这里修改了原始的ResNet网络架构），然后有两个分支，上面的分支负责分类和回归，下面的分支负责生成对应的mask。由于前面进行了多次卷积和池化，减小了对应的分辨率，**mask分支开始利用反卷积进行分辨率的提升**，同时减少通道的个数，变为14x14x256，最后输出了14x14x80的mask模板。
  - 右边使用到的backbone是FPN网络，这是一个新的网络，通过输入单一尺度的图片，最后可以对应的特征金字塔，得到证实的是，该网络可以在一定程度上面提高检测的精度，当前很多的方法都用到了它。**由于FPN网络已经包含了res5，可以更加高效的使用特征，因此这里使用了较少的filters**。该架构也分为两个分支，作用于前者相同，但是分类分支和mask分支和前者相比有很大的区别。可能是因为**FPN网络可以在不同尺度的特征上面获得许多有用信息**，因此分类时使用了更少的滤波器。而mask分支中进行了多次卷积操作，首先将ROI变化为14x14x256的feature，然后进行了5次相同的操作，然后进行反卷积操作，最后输出28x28x80的mask。即输出了更大的mask，与前者相比可以获得更细致的mask。

### 4.  实验及结果部分

- 有关实验实现细节及相关结果见原文。