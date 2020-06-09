# DL_Notes

## 📚 专栏介绍

- 学习笔记的归类
- 有关DL、ML的资料的积累
- 相关领域：神经网络、计算机视觉

## 1.神经网络与深度学习

- [1-1 神经网络与深度学习（1）-深度学习概论](http://yearing1017.cn/2019/04/12/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-1-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A6%82%E8%AE%BA/)
- [1-2 神经网络与深度学习（2）-逻辑回归](http://yearing1017.cn/2019/04/27/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-2-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/)
- [1-3 神经网络与深度学习（3）-Python与向量化](http://yearing1017.cn/2019/05/02/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-3-Python%E4%B8%8E%E5%90%91%E9%87%8F%E5%8C%96/)
- [机器学习常见评价指标及分割指标MIoU](http://yearing1017.cn/2020/02/07/语义分割指标MIoU/)
- [MIoU基于PyTorch的计算](http://yearing1017.cn/2020/02/17/MIoU-PyTorch/)
- [深度可分离卷积](http://yearing1017.cn/2020/02/15/Depthwise-separable-convolution/)
- [Kappa系数基于PyTorch的计算](http://yearing1017.cn/2020/02/27/基于混淆矩阵的Kappa系数的计算/)
- [过拟合问题原因](https://www.cnblogs.com/eilearn/p/9203186.html)
- [k折交叉验证](https://zhuanlan.zhihu.com/p/67563863)
- [sklearn中k折交叉验证的实现](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) 
- [离线数据增强与在线数据增强](https://zhuanlan.zhihu.com/p/56139575)
- [语义分割中的三种上采样方式：转置卷积、线性插值、Unpooling](https://zhuanlan.zhihu.com/p/92123010)
- [上采样-双线性插值的理解](https://www.zhihu.com/search?type=content&q=%E4%B8%8A%E9%87%87%E6%A0%B7%20%E5%8F%8C%E7%BA%BF%E6%80%A7%E6%8F%92%E5%80%BC)
- [上采样-最邻近插值](https://zhuanlan.zhihu.com/p/89421892)
- [某大佬对align_corners的理解](https://zhuanlan.zhihu.com/p/87572724)
- [深度学习的11个技巧](https://zhuanlan.zhihu.com/p/138024517)
- [Focal-Loss损失函数的理解](https://www.zhihu.com/search?q=focal%20loss%20pytorch&type=content)
- [labelme标注工具的安装及使用总结](http://yearing1017.cn/2020/05/30/labelme%E7%9A%84%E5%AE%89%E8%A3%85%E5%8F%8A%E4%BD%BF%E7%94%A8/)


## 2.卷积神经网络原理与视觉实践

- [CNN学习笔记（1）-CNN基本结构简介](http://yearing1017.cn/2019/07/28/CNN%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/)
- [CNN学习笔记（2）-CNN基本流程](http://yearing1017.cn/2019/08/04/CNN%E5%9F%BA%E6%9C%AC%E6%B5%81%E7%A8%8B/)
- [CNN学习笔记（3）-CNN_卷积层](http://yearing1017.cn/2019/08/11/CNN基本部件-卷积层/)
- [CNN学习笔记（4）-CNN_汇合层（池化层）](http://yearing1017.cn/2019/08/13/CNN%E5%9F%BA%E6%9C%AC%E9%83%A8%E4%BB%B6-%E6%B1%87%E5%90%88%E5%B1%82/)
- [CNN学习笔记（5）-CNN_激活函数与全连接层](http://yearing1017.cn/2019/08/14/CNN-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E4%B8%8E%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82/)
- [CNN学习笔记（6）-CNN_填充和步幅](http://yearing1017.cn/2019/09/05/CNN-%E5%A1%AB%E5%85%85%E5%92%8C%E6%AD%A5%E5%B9%85/)
- [CNN学习笔记（7）-CNN_channels](http://yearing1017.cn/2019/09/07/CNN-channels/)
- [CNN学习笔记（8）-张量](http://yearing1017.cn/2019/11/12/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-%E5%BC%A0%E9%87%8F/)
- [DL知识随笔记录](http://yearing1017.cn/2019/11/14/CNN-%E7%9F%A5%E8%AF%86%E7%82%B9%E9%9A%8F%E7%AC%94%E7%A7%AF%E7%B4%AF/)
  - 全连接层流程及softmax loss
  - 过拟合欠拟合问题
  - 权重衰减的推导
  - 归一化
  - 网络参数的几种初始化方式
  - 机器学习模型的三种评估方法
  - 感受野
  - 全局平均池化
  - 张量的维度判断技巧
  - Focal Loss
  
## 3.经典CNN网络模型学习
- [1. LeNet-5](http://yearing1017.cn/2019/09/09/CNN-LeNet-5/)
- [2. AlexNet](http://yearing1017.cn/2019/09/10/CNN-AlexNet/)
- [3. VGGNet-16](http://yearing1017.cn/2019/09/13/CNN-VGGNet16/)
- [4. GoogLeNet_v1-v3](http://yearing1017.cn/2019/09/24/GoogLeNet-V1-V3/)
- [5. ResNet_DRN](http://yearing1017.cn/2019/09/26/ResNet-DRN/)
- [7. DenseNet](http://yearing1017.cn/2019/10/29/DenseNet-CVPR2017/)

## 4.语义分割网络学习
- [1. FCN](http://yearing1017.cn/2019/10/17/FCN-%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/)
- [2. UNet](http://yearing1017.cn/2019/11/21/U-Net-paper/)
- [3. Deeplabv3](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/Deeplab_v3.md)
- [4. Deeplabv3+](https://github.com/yearing1017/Paper_Note/blob/master/论文Markdown笔记/deeplabv3%2B_paper.md)
- [5. DFN](http://yearing1017.cn/2020/03/19/DFN-paper/)
- [6. Non-local-Net](http://yearing1017.cn/2020/04/05/Non-local-paper/)
- [7. CCNet](http://yearing1017.cn/2020/03/26/CCNet-paper/)
- [8. DAN](http://yearing1017.cn/2020/04/06/DAN-paper/#more)
- [9. PANet](http://yearing1017.cn/2020/04/10/PAN-paper/)
- [10. ResNeSt](http://yearing1017.cn/2020/05/17/ResNeSt-Split-Attention-Networks/)


## 5.实例分割

### 5.1 相关资料积累
- [实例分割最新最全面综述：从Mask R-CNN到BlendMask](https://zhuanlan.zhihu.com/p/110132002)

### 5.2 R-CNN系列
- R-CNN:[Rich feature hierarchies for accurate object detection and semantic segmentation Tech report (v5)](http://cn.arxiv.org/pdf/1311.2524.pdf)
- [R-CNN学习笔记](http://yearing1017.cn/2020/04/26/R-CNN-paper/)
- Fast R-CNN:[Fast R-CNN](http://cn.arxiv.org/pdf/1504.08083v2)
- [Fast R-CNN学习笔记](http://yearing1017.cn/2020/04/27/Fast-R-CNN/)
- Faster R-CNN:[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://cn.arxiv.org/pdf/1506.01497.pdf)
- [Faster R-CNN学习笔记](http://yearing1017.cn/2020/04/29/Faster-R-CNN/)
- Mask R-CNN:[Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
- [Mask R-CNN学习笔记](http://yearing1017.cn/2020/05/04/Mask-R-CNN/)

## 6.计算机视觉中的Attention机制
- [简单认识CV中的注意力机制](https://blog.csdn.net/paper_reader/article/details/81082351)
- [Attention机制的文章总结](https://blog.csdn.net/humanpose/article/details/85332392)
- [深入理解注意力机制](https://zhuanlan.zhihu.com/p/40197380)
- [CV中的Attention机制-SENet中的SE模块]( https://zhuanlan.zhihu.com/p/102035721)
- [视觉注意力机制 | Non-local模块与Self-attention的之间的关系与区别](https://zhuanlan.zhihu.com/p/110130098)
- SENet论文笔记：[博文地址](http://yearing1017.cn/2020/05/11/SENet-paper/)
- SKNet论文笔记：[博文地址](http://yearing1017.cn/2020/05/14/SKNet/)
- [融合SENet、SKNet、ResNeXt、ResNet的backbone网络：ResNeSt](http://yearing1017.cn/2020/05/17/ResNeSt-Split-Attention-Networks/)
- [x] CCNet论文：[论文地址](http://cn.arxiv.org/pdf/1811.11721.pdf)
- [x] CCNet论文的实现：[官方实现](https://github.com/speedinghzl/CCNet)、[非官方简单实现](https://github.com/Serge-weihao/CCNet-Pure-Pytorch)
- [x] Non-local-Net论文：[论文地址](https://arxiv.org/abs/1711.07971)
- [x] DAN论文：[Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983)
- [x] DAN代码实现：[代码地址1](https://github.com/yiskw713/DualAttention_for_Segmentation)、[代码地址2](https://github.com/junfu1115/DANet/)
- [x] PANet论文：[Pyramid Attention Network for Semantic Segmentation](http://cn.arxiv.org/pdf/1805.10180v1.pdf)
- [x] PAN的代码实现：[参考1](https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch/)、[参考2](https://github.com/Andy-zhujunwen/pytorch-Pyramid-Attention-Networks-PAN-/)，后者是前者的改进
- [x] DFN的实现：[参考](https://github.com/ycszen/TorchSeg/tree/master/model/dfn)
- [ ] EMANet论文：[ Expectation-Maximization Attention Networks for Semantic Segmentation](https://zhuanlan.zhihu.com/p/78018142)


## 7. 目标检测
- R-CNN:[Rich feature hierarchies for accurate object detection and semantic segmentation Tech report (v5)](http://cn.arxiv.org/pdf/1311.2524.pdf)
- [R-CNN学习笔记](http://yearing1017.cn/2020/04/26/R-CNN-paper/)
- Fast R-CNN:[Fast R-CNN](http://cn.arxiv.org/pdf/1504.08083v2)
- [Fast R-CNN学习笔记](http://yearing1017.cn/2020/04/27/Fast-R-CNN/)
- Faster R-CNN:[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://cn.arxiv.org/pdf/1506.01497.pdf)
- [Faster R-CNN学习笔记](http://yearing1017.cn/2020/04/29/Faster-R-CNN/)


## 8. GAN

### 8.1 GAN入门
- [通俗理解生成对抗网络GAN](https://zhuanlan.zhihu.com/p/33752313)
