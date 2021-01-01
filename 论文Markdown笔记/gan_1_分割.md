### 1. 前言

- 问题：**深度学习的模型（比如卷积神经网络）有时候并不能很好地学到训练数据中的一些特征**。比如，在图像分割中，现有的模型通常对每个像素的类别进行预测，像素级别的准确率可能会很高，**但是像素与像素之间的相互关系就容易被忽略，使得分割结果不够连续或者明显地使某一个物体在分割结果中的尺寸、形状与在金标准中的尺寸、形状差别较大。**

- 对抗学习（adversarial learning）就是为了解决上述问题而被提出的一种方法。
- 本篇论文是**首次将GAN与语义分割联系在一起的文章，发表于2016**

- 论文链接：[Semantic Segmentation using Adversarial Networks](https://arxiv.org/pdf/1611.08408.pdf) 

### 2. 网络理解

- 这篇文章第一个将对抗网络（adversarial network）应用到图像分割中，该文章中的方法如下图。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/28-1.png)

- 在GAN中，有生成器和判别器，生成器生成fake样本然后判别器进行鉴别，随着训练的进行，生成器的fake样本越接近与数据真实分布，判别器也越难分辨真伪。
- 将GAN用于分割，实际上，基本的分割网络（FCN, DeepLab, PSPNet...）就是GAN中的生成器。换句话说，GAN用于分割不需要另外再构造一个生成网络，**传统分割网络就是生成网络。然后在生成网络之后加一个判别网络结构，如上图所示**。

- 左边Segmentor就是传统的CNN-based分割网络，Convnet中可以看到有convolution和deconvolution过程；右边Adversarial network是GAN中的判别器，最后用sigmoid activation进行二分类。
- Segmentor这个部分会有两种训练图片输入到右边的Adversarial network中：
  - 原始训练图片+ground truth, 这时候判别器判别为 1 标签；
  - 原始训练图片+Segmentor分割结果， 这时候判别器判别为0标签。
- 然后训练过程就是经典的博弈思想（**生成器生成图片通过判别器进行判别直到判别网络无法识别出图像的真假为止。**），相互提高网络的ability, 提高分割精度，提高鉴别能力。

### 3. 对抗训练

- 论文提出了混合损失函数，将Cross Entropy和GAN的损失结合起来。看一下论文中对两个loss函数的解释：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/28-2.png)

- **输入为x_n，对应的label为y_n，则loss的表达式如下：**

$$
\ell\left(\boldsymbol{\theta}_{s}, \boldsymbol{\theta}_{a}\right)=\sum_{n=1}^{N} \ell_{\mathrm{mce}}\left(s\left(\boldsymbol{x}_{n}\right), \boldsymbol{y}_{n}\right)-\lambda\left[\ell_{\mathrm{bce}}\left(a\left(\boldsymbol{x}_{n}, \boldsymbol{y}_{n}\right), 1\right)+\ell_{\mathrm{bce}}\left(a\left(\boldsymbol{x}_{n}, s\left(\boldsymbol{x}_{n}\right)\right), 0\right)\right]
$$

- 论文中对上公式附了一段解释如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/28-3.png)

- 理解上段的主要内容如下：
  - **mce代表多类别的交叉熵损失函数；bce代表二分类的交叉熵损失函数；**
  - 希望最小化mce损失，同时最大化bce损失；这里的maximizing代表的含义：看公式得知，**总的loss为mce-bce，当mce最小，bce最大时，整个loss的值最小；**这里期盼bce损失大可这样理解，对抗网络来分辨真实label和生成预测结果，当bce损失最大时，对抗网络无法辨别，说明生成网络较好。
- 对抗训练分成两个步骤迭代，**需要找到两个参数最优相互调整，先固定一个参数，调整另一个参数，再循环迭代。**

#### 3.1 训练对抗模型

- 相当于训练GAN辨别真伪的能力，所以要尽量的min：`training the adversarial model is equivalent to minimizing the following binary classiﬁcation loss`

$$
\sum_{n=1}^{N} \ell_{\mathrm{bce}}\left(a\left(\boldsymbol{x}_{n}, \boldsymbol{y}_{n}\right), 1\right)+\ell_{\mathrm{bce}}\left(a\left(\boldsymbol{x}_{n}, s\left(\boldsymbol{x}_{n}\right)\right), 0\right)
$$

#### 3.2 训练分割模型

- **分割网络模型的训练：需要mce损失降到最小，使得输入和label逼近一致；同时需要对抗模型的反馈，于是加上下面一项，使得bce的损失大一些，对抗模型无法辨别；**

$$
\sum_{n=1}^{N} \ell_{\mathrm{mce}}\left(s\left(\boldsymbol{x}_{n}\right), \boldsymbol{y}_{n}\right)-\lambda \ell_{\mathrm{bce}}\left(a\left(\boldsymbol{x}_{n}, s\left(\boldsymbol{x}_{n}\right)\right), 0\right)
$$

