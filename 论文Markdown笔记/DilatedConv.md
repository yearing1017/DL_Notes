### 1. 前言

- 本文的主要学习内容为**Dilated Convolution(空洞卷积、扩张卷积)**

- 原论文链接为[MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS](https://arxiv.org/pdf/1511.07122.pdf)

### 2. 问题

- 本文中最核心的概念就是空洞卷积。不过空洞卷积的概念并不是由本文提出的，它最早出现在小波分解算法当中，将空洞卷积应用在卷积网络架构本文也不是最早的。
- 空洞卷积的最大特点就是**能够增大感受野，同时还不损失分辨率。**
- 回顾之前学过的FCN网络，**FCN的一个不足之处在于**，由于**池化层的存在，响应张量的大小（长和宽）越来越小**，但是FCN的设计初衷则**需要和输入大小一致的输出**，因此**FCN做了上采样**。**但是上采样并不能将丢失的信息全部无损地找回来。**
- 对上而言，dilated convolution是一种很好的解决方案——**既然池化的下采样操作会带来信息损失，那么就把池化层去掉。**
- 但是**池化层去掉随之带来的是网络各层的感受野（Receptive field）变小。**这样会降低预测的精度。
- 这里解释**为什么池化层会增大感受野**。我们知道在实际训练中，我们的卷积核一般就是比较小的，如3 x 3，这些卷积核本质就是在特征图上进行滤波窗口计算并滑动。如果要保持卷积核大小不变，同时增大卷积核覆盖区域（感受野增大，便于提取高层语义），那么就可以**对图片尺寸进行下采样，相对与之前来看，同样大小的卷积核使得感受野增大。**
- Dilated convolution的主要贡献就是，如何**在去掉池化下采样操作的同时，而不降低网络的感受野**。

### 3. 空洞卷积

#### 3.1 论文公式

- 在论文中，提出了**标准卷积的公式**如下：

$$
(F * k)(\mathbf{p})=\sum_{\mathbf{s}+\mathbf{t}=\mathbf{p}} F(\mathbf{s}) k(\mathbf{t})
$$

- **空洞卷积的公式：**

$$
\left(F *_{l} k\right)(\mathbf{p})=\sum_{\mathbf{s}+l \mathbf{t}=\mathbf{p}} F(\mathbf{s}) k(\mathbf{t})
$$

- 提到的两个公式的区别：**加入了一个 $l$ 因子，从而改变了标准卷积。当 $l$ = 1的时候，两者一致**

> `We use the term “dilatedconvolution” instead of “convolutionwitha dilated filter” to clarify that no “dilated filter” is constructed or represented.`

- 如上句，论文中提出，所谓的**空洞卷积并不是构建了特有的dilated filter，而是以另外的方式改变了卷积的参数，即加入了 $l$ 因子。**

#### 3.2 图示

- **标准卷积图示**：该卷积操作对原图做了填充（padding=1），卷积核大小为3x3，卷积的步长为2。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/3-1.jpg" style="zoom:50%;" />

- **空洞卷积图示：**空洞卷积操作的卷积核大小为3x3，空洞率（dilation rate）=2，卷积的步长为1

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/3-2.jpg" style="zoom:50%;" />

- 我们可以看到，**标准的卷积操作中，卷积核的元素之间都是相邻的**，但是**在空洞卷积中，卷积核的元素是间隔的，间隔的大小取决于空洞率**。

#### 3.3 论文中的举例

- 论文中给出下面的三个图片，为了直观得出**空洞卷积增大了感受野**这一结论。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/3-3.jpg)

- 如图a，**红色圆点为卷积核对应的输入“像素”，绿色为其在原输入中的感知野**。我们可以理解红点就为卷积核的“核”，每个核对应一块绿色的“感受野”。
- **以$3 \times 3$的卷积核为例，传统卷积核在做卷积操作时，是将卷积核与输入张量中“连续”的$3\times3$的patch逐点相乘再求和**。
- 图b，在**去掉一层池化层后，需要在去掉的池化层后将传统卷积层换做一个“dilation=2”的dilated convolution层，**此时卷积核**将输入张量每隔一个“像素”的位置作为输入patch**进行卷积计算，可以发现这时对应到原输入的感知野已经扩大（dilate）为$7\times7$。
- 这时我们可以像上面那样理解，当然这里只是为了更容易理解，这里的话并不是准确的：红色的卷积核的点，之前一个对应一个绿色，现在每隔一个之后，核变大（卷积核不会变，还是3 x 3），感受野自然变大。

- 同理，如果再去掉一个池化层，就要**将其之后的卷积层换成“dilation=4”的dilated convolution层**，如图c所示。这样一来，即使去掉池化层也能保证网络的感受野，从而确保图像语义分割的精度。

- 从下面的几个图像语义分割效果图可以看出，**在使用了dilated convolution这一技术后可以大幅提高语义类别的辨识度以及分割细节的精细度**：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/3-4.jpg)

### 4. 总结

- 本文对**空洞卷积进行了学习总结**
- 接下来的学习计划是去实际应用一下空洞卷积

