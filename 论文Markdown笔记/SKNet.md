### 1. 前言

- 论文链接：[Selective Kernel Networks](https://arxiv.org/pdf/1903.06586.pdf)

- 皮质神经元根据不同的刺激可动态调节其自身的感受野。因此，论文提出**一种动态选择机制使每一个神经元可以针对目标物体的大小选择不同的感受野。**
- 参考原作者的讲解：[知乎-SKNet——SENet孪生兄弟篇](https://zhuanlan.zhihu.com/p/59690223)

### 2. SKNet结构

- 本文设计**SK单元用不同卷积核提取特征，然后通过每个分支引导的不同信息构成的softmax进行融合**。
- SK单元包括三个方面：Split， Fuse， Select
  - **Split：阶段使用不同的卷积核对原图进行卷积；**
  - **Fuse：组合并聚合来自多个路径的信息，以获得选择权重的全局和综合表示；**
  - **Select：根据选择权重聚合不同大小的内核的特征映射。**

- 结构图如下所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/27-1.jpg)

#### 2.1 Split

- 对于任意输入的feature map，首先进行两个卷积，得到$\widetilde{\mathbf{U}}$和$\widehat{\mathrm{U}}$，使用的kernel size分别为 3x3 和 5x5，其中 5x5 的卷积核替换为一个dilation为2的3x3的卷积核。

#### 2.2 Fuse

- 该步骤通过门控机制将上一层的输出进行有选择的筛选，使每个分支都携带不同的信息流进入下个神经元。

- 对不同分支的输出进行融合，即**逐元素进行相加**（输出的尺寸和通道数必须是一样的）：
  $$
  \mathbf{U}=\tilde{\mathbf{U}}+\widehat{\mathbf{U}}
  $$

- 对两个输出进行**全局平均池化**（global average pooling ）操作，获得每一个通道上的全局信息：

$$
s_{c}=\mathcal{F}_{g p}\left(\mathbf{U}_{c}\right)=\frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{U}_{c}(i, j)
$$

- **对输出 s 做全连接找到每一个通道占的比重大小：**

$$
\mathbf{z}=\mathcal{F}_{f c}(\mathbf{s})=\delta(\mathcal{B}(\mathbf{W} \mathbf{s}))
$$

> ​	δ 是relu函數，B表示批正则化处理.

- 上述的全连接操作：**将C维转为了d维，后如图所示，又做了两个线性变换，将d维变为了C维，这样完成了针对channel维度的信息提取。**
- 然后**使用Softmax进行归一化，这时候每个channel对应一个分数，代表其channel的重要程度**

#### 2.3 Select

- 将这三个分别得到的mask（即分数）分别乘以对应的U1,U2,U3，得到A1,A2,A3。然后三个模块相加，进行信息融合，得到最终模块A， 模块A相比于最初的X经过了信息的提炼，融合了多个感受野的信息。

- 其中，如下公式：

$$
\begin{array}{l}a_{c}=\frac{e^{\mathbf{A}_{c} \mathbf{z}}}{e^{\mathbf{A}_{c} \mathbf{z}}+e^{\mathbf{B}_{c} \mathbf{z}}}, b_{c}=\frac{e^{\mathbf{B}_{c} \mathbf{z}}}{e^{\mathbf{A}_{c} \mathbf{z}}+e^{\mathbf{B}_{c} \mathbf{z}}} \\ \mathbf{V}_{c}=a_{c} \cdot \widetilde{\mathbf{U}}_{c}+b_{c} \cdot \widehat{\mathbf{U}}_{c}, \quad a_{c}+b_{c}=1 \\ \mathbf{V}=\left[\mathbf{V}_{1}, \mathbf{V}_{2}, \ldots, \mathbf{V}_{C}\right], \mathbf{V}_{c} \in \mathbb{R}^{H \times W}\end{array}
$$

### 3. SK模块的构成

- 从C线性变换为Z维，再到C维度，这个部分与SE模块的实现是一致的
- 多分支的操作借鉴自：inception
- 整个流程类似merge-and-run mapping

### 4. 代码实现

- 可与上图对应来看代码实现：

```python
作者：pprp
链接：https://zhuanlan.zhihu.com/p/102034839
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

import torch.nn as nn
import torch

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))
            
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

if __name__ == "__main__":
    t = torch.ones((32, 256, 24,24))
    sk = SKConv(256,WH=1,M=2,G=1,r=2)
    out = sk(t)
    print(out.shape)
```

