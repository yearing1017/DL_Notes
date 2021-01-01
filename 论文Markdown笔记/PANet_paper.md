### 1. 前言

- 本文论文地址：[Pyramid Attention Network for Semantic Segmentation](https://arxiv.org/abs/1805.10180v1)

- 本文提出了**结合注意力机制和空间金字塔去提取精准的密集特征而用于像素级标注任务**
- 论文没有开源的实现，github上的实现代码：[Pyramid-Attention-Networks-pytorch](https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch)

### 2. Abstract

- `Different from most existing works, we combine attention mechanism and spatial pyramid to extract precise dense features for pixel labeling instead of complicated dilated convolution and artiﬁcially designed decoder networks.`
- `Speciﬁcally, we introduce a Feature Pyramid Attention module to perform spatial pyramid attention structure on high-level output and combining global pooling to learn a better feature representation, and a Global Attention Upsample module on each decoder layer to provide global context as a guidance of low-level features to select category localization details.`

### 3. Introduction

- 文章在`Introduction`一节中阐述了语义分割任务中的两个 `issue`:
  - `The ﬁrst issue is that the existence of objects at multiple scales cause difﬁculty in classiﬁcation of categories.` 概括讲即：**分割对象的多尺度问题。**
  - 解决方案：`PSPNet` 中的 `spatial pyramid pooling`， `Deeplab`中的 `Atrous spatial pyramid pooling`.
- 第二个问题：
  - `Another issue is that high-level features are skilled in making category classiﬁcation, while weak in restructuring original resolution binary prediction.`
  - 概括讲：**高层语义特征更适合分类任务，不适合 `dense prediction` 的语义分割任务**。
  - 解决方案就是**引入`decoder`模块利用低层特征信息辅助高层特征进行细节恢复。**
- 本文贡献：
  - `Firstly, We propose a Feature Pyramid Attention module to embed different scale context features in an FCN based pixel prediction framework.`
  - `Then, We develop Global Attention Upsample, an effective decoder module for semantic segmentation.`
  - `Lastly, Combined Feature Pyramid Attention module and Global Attention Upsample, our Pyramid Attention Network architecture archieves state-of-the-art accuracy on VOC2012 and cityscapes benchmark.`

### 4. Method

- 文章提出的整个网络框架仍是常见的 `encoder - decoder` 的结构。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/21-1.jpg)

#### 4.1 Feature Pyramid Attention

- **当前考虑的问题：**
  - `In the current semantic segmentation architecture, the pyramid structure can extract different scale of feature information and increase respective ﬁeld effectively in pixel-level, while this kind of structure lacks global context prior attention to select the features channel-wisely as in SENet[8] and EncNet[31].`
  - 在目前的语义分割架构中，金字塔结构可以提取出不同尺寸的特征信息并增加像素级的感受野，但是这样的结构缺少全局上下文先验注意力去选择在 SENet 和 EncNet 中的对应通道的特征。
  - `On the other hand, using channel-wise attention vector is not enough to extract multi-scale features effectively and lack pixel-wise information.`
  - 另一方面，使用对应通道注意力向量还不足以有效提取多个尺度的特征且缺少像素级的信息。

- **SPA与FPA**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/21-2.jpg)

- `To better extract context from different pyramid scales, we use 3×3, 5×5, 7×7 convolution in pyramid structure respectively. Since the resolution of high-level feature maps is small, using large kernel size doesn’t bring too much computation burden. `
- `Then the pyramid structure integrates information of different scales step-by-step, which can incorporate neighbor scales of context features more precisely.`
- `Then the origin features from CNNs is multiplied pixel-wisely by the pyramid attention features after passing through a 1×1 convolution.`

- `We also introduce global pooling branch concatenated with the output features, which improve our FPA module performance further.`

- 这个模块被安插在 `encoder` 的顶端， 产生跟 `high-level feature map` 同样形状的 `accurate pixel-wise attention`. 在笔者看来，**这是给顶层**`feature maps`**的每一个通道都产生一个** `spatial attention map`，并且这个 `attention map`融合了针对不同尺度（大尺度物体，中尺度物体，小尺度物体）的三条支路产生的 `attention map`. 除此之外，效仿 `ParseNet`的 `GAP` ,加入 `global context information`, 有助于减轻 **类内异质性**。

#### 4.2 Global Attention Upsample

- `We consider that the main character of decoder module is to repair category pixel localization. Furthermore, high-level features with abundant category information can be used to weight low-level information to select precise resolution details.`
- **GAU结构图**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/21-3.jpg" style="zoom:50%;" />

- `Our Global Attention Upsample module performs global pooling to provide global context as a guidance of low-level features to select category localization details.`

- `In detail, we perform 3×3 convolution on the low-level features to reduce channels of feature maps from CNNs. The global context generated from high-level features is through a 1×1 convolution with batch normalization and nonlinearity, then multiplied by the low-level features. Finally high-level features are added with the weighted low-level features and upsampled gradually.`

### 5. Conclusion

- `We designed Feature Pyramid Attention module and an effective decoder module Global Attention Upsample.`
- `Feature Pyramid Attention module provides pixel-level attention information and increases respective ﬁeld by performing pyramid structure.`
- `Global Attention Upsample module exploits high-level feature map to guide low-level features recovering pixel localization.`

- `Our experimental results show that the proposed approach can achieve comparable performance with other state-of-art models on the PASCAL VOC 2012 semantic image segmentation benchmark.`

