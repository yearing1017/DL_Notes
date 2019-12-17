# DL_Notes
⏰DL&ML&经典网络学习笔记

## 简介

此专栏记录DL和ML的学习笔记，以及经典的CNN神经网络学习，把自己的笔记贴到这边，坚持更新...

## 1.神经网络与深度学习

- [1-1 神经网络与深度学习（1）-深度学习概论](https://yearing1017.site/2019/04/12/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-1-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A6%82%E8%AE%BA/)
- [1-2 神经网络与深度学习（2）-逻辑回归](https://yearing1017.site/2019/04/27/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-2-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/)
- [1-3 神经网络与深度学习（3）-Python与向量化](https://yearing1017.site/2019/05/02/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-3-Python%E4%B8%8E%E5%90%91%E9%87%8F%E5%8C%96/)

## 2.卷积神经网络原理与视觉实践

- [CNN学习笔记（1）-CNN基本结构简介](https://yearing1017.site/2019/07/28/CNN%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/)
- [CNN学习笔记（2）-CNN基本流程](https://yearing1017.site/2019/08/04/CNN%E5%9F%BA%E6%9C%AC%E6%B5%81%E7%A8%8B/)
- [CNN学习笔记（3）-CNN_卷积层](https://yearing1017.site/2019/08/11/CNN基本部件-卷积层/)
- [CNN学习笔记（4）-CNN_汇合层（池化层）](https://yearing1017.site/2019/08/13/CNN%E5%9F%BA%E6%9C%AC%E9%83%A8%E4%BB%B6-%E6%B1%87%E5%90%88%E5%B1%82/)
- [CNN学习笔记（5）-CNN_激活函数与全连接层](https://yearing1017.site/2019/08/14/CNN-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E4%B8%8E%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82/)
- [CNN学习笔记（6）-CNN_填充和步幅](https://yearing1017.site/2019/09/05/CNN-%E5%A1%AB%E5%85%85%E5%92%8C%E6%AD%A5%E5%B9%85/)
- [CNN学习笔记（7）-CNN_channels](https://yearing1017.site/2019/09/07/CNN-channels/)
- [CNN学习笔记（8）-张量](https://yearing1017.site/2019/11/12/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-%E5%BC%A0%E9%87%8F/)
- [CNN知识随笔记录](https://yearing1017.site/2019/11/14/CNN-%E7%9F%A5%E8%AF%86%E7%82%B9%E9%9A%8F%E7%AC%94%E7%A7%AF%E7%B4%AF/)
  - 全连接层流程及softmax loss
  - 过拟合欠拟合问题
  - 权重衰减的推导
  - 归一化
  - 网络参数的几种初始化方式
  - 机器学习模型的三种评估方法
  - 感受野
  
## 3.经典网络模型学习
- [1. LeNet-5](https://yearing1017.site/2019/09/09/CNN-LeNet-5/)
- [2. LeNet-5_MINST](https://yearing1017.site/2019/09/10/Tensorflow-LeNet-5-MNIST/)
- [3. AlexNet](https://yearing1017.site/2019/09/10/CNN-AlexNet/)
- [4. AlexNet_MINST](https://yearing1017.site/2019/09/10/Tensorflow-AlexNet-MNIST/)
- [5. VGGNet-16](https://yearing1017.site/2019/09/13/CNN-VGGNet16/)
- [6. GoogLeNet_v1-v3](https://yearing1017.site/2019/09/24/GoogLeNet-V1-V3/)
- [7. ResNet_DRN](https://yearing1017.site/2019/09/26/ResNet-DRN/)
- [8. FCN](https://yearing1017.site/2019/10/17/FCN-%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/)
- [9. DenseNet](https://yearing1017.site/2019/10/29/DenseNet-CVPR2017/)
- [10. UNet](https://yearing1017.site/2019/11/21/U-Net-paper/)

## 4.Numpy积累

### 4.1 有关`np.argmin/argmax`在axis维度上的认识

- 先看一段有关argmin的代码：
```python
import numpy as np
a = np.random.randint(24, size=[4,2,3,3])

print("a本身为:")
print(a)

a_argmin = np.argmin(a, axis=1)
print("argmin: axis=1")
print(a_argmin)

print("加上None:")
print(a_argmin[:,None,:,:])
```

- 输出结果
```python
a本身为:
[[[[ 3 14  4]
   [ 5 23 12]
   [ 0 22 20]]

  [[12 17 10]
   [ 1 15  1]
   [ 2 18  7]]]


 [[[19  6  5]
   [ 9 22  7]
   [ 4  8 11]]

  [[10  8 12]
   [21  7  4]
   [17 21  8]]]


 [[[ 5 22  0]
   [ 0 13 13]
   [14  5 13]]

  [[11 18  4]
   [ 0 10  7]
   [22  0  9]]]


 [[[15 21  2]
   [13 20 12]
   [14  3  7]]

  [[12 23 22]
   [11  1 21]
   [11 20 14]]]]
argmin: axis=1
[[[0 0 0]
  [1 1 1]
  [0 1 1]]

 [[1 0 0]
  [0 1 1]
  [0 0 1]]

 [[0 1 0]
  [0 1 1]
  [0 1 1]]

 [[1 0 0]
  [1 1 0]
  [1 0 0]]]
加上None:
[[[[0 0 0]
   [1 1 1]
   [0 1 1]]]


 [[[1 0 0]
   [0 1 1]
   [0 0 1]]]


 [[[0 1 0]
   [0 1 1]
   [0 1 1]]]


 [[[1 0 0]
   [1 1 0]
   [1 0 0]]]]
[Finished in 0.2s]
```
- a本身是一个numpy的多维数组，shape:[4, 2, 3, 3]

- 理解：4代表第0维，其中输出a的结果首尾有4个`[`或`]`就代表该维度的数值

- 理解：2代表第1维，其中每3个`[`或`]`的中间就有两块数据，如下：
```python
[[[19  6  5]
   [ 9 22  7]
   [ 4  8 11]]

  [[10  8 12]
   [21  7  4]
   [17 21  8]]]
```
- 理解：3代表第2维和第3维，shape的最后两个数字表示行列。即每两个`[`或`]`包起来的数据：
```python
  [[10  8 12]
   [21  7  4]
   [17 21  8]]
```
- `np.argmin`表示在某维度上比较大小，取最小的下标组成，返回array，shape为原ndarray去掉axis维度后的shape[4, 3, 3]

- 在上面的例子中，axis=1表示第1维，即在每两个3x3的矩阵上下比较大小

- `np.argmax`同理，即取较大的下标返回。

- 在shape中加None，相当于加了一个新的维度，但是没有新的数值插入，仅仅多了表示维度的`[`和`]`
