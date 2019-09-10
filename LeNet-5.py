import tensorflow as tf   
from tensorflow.examples.tutorials.mnist import input_data  
'''
	1.input_data是TensorFlow的一个自带模块，在使用read_data_sets方法读取数据，若无该数据，则会自动下载
	2.第二个参数 one_hot = True
'''
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)  #读取数据
sess = tf.InteractiveSession()  # 注册默认session 后面操作无需指定session 不同sesson之间的数据是独立的
 
def weight_variable(shape):                          # 权值w初始化设置 stddev=0.1：给一些偏差0.1防止死亡节点
    initial = tf.truncated_normal(shape,stddev=0.1)  # tf.truncated_normal函数返回指定形状的张量填充随机截断的正常值
    return tf.Variable(initial)                      # 创建一个变量
  
def bias_variable(shape):                            # 偏置bias初始化设置
    initial = tf.constant(0.1,shape = shape)         # 定义一个常量，shape即常量维数，当前面value是数字时，使用数字填充
    return tf.Variable(initial)  					 # 返回以该常量变量化的变量
'''
	1.x是输入，W是卷积参数
	2.如[5,5,1,30] 前两个表示卷积核的尺寸；第三个表示通道channel；  第四个表示提取多少类特征
	3.strides 表示卷积模板移动的步长都是 1代表不遗漏的划过图片每一个点
	4.padding 表示边界扩充方式这里的SAME代表给边界加上padding让输出和输入保持相同尺寸
'''
def conv2d(x,W):  
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')  
    # 卷积strides=[首位默认为1,平行步长=1,竖直步长=1,尾位默认为1]

# 池化层
def max_pool_2x2(x):  
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 
    # ksize=[1,2,2,1] 池化核size: 2×2
    # 池化strides=[首位默认为1,平行步长=2,竖直步长=2,尾位默认为1]

# placeholder：等待输入数据，x为占位符，接受型号为float32的数据，输入格式为矩阵[None,784]
x = tf.placeholder(tf.float32,[None,784])   #784 = 28×28 只对输入矩阵的列数有要求

y_ = tf.placeholder(tf.float32,[None,10])   
											# reshape变换张量shape 2维张量变4维 [None, 784] to [-1,28,28,1] 
x_image = tf.reshape(x,[-1,28,28,1])  		# [batch=-1, height=28, width=28, in_channels=1] -1表示样本数量不固定


# Conv1 Layer 卷积层 初始化第一个卷积层的权值和偏置 28x28x1 -> 28x28x32 （因为padding补0所以尺寸大小不变）
W_conv1 = weight_variable([5,5,1,32]) # [5×5卷积核, in_channels=1, out_channels=32=卷积核个数] 32表示提取32类特征
b_conv1 = bias_variable([32])         # [32=卷积核个数=bias个数]

# 把x_image和权值向量W_conv1进行卷积，再加上偏置值，然后用relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)       # 池化方式：max pool 取最大值 max_pooling 28x28x32 -> 14x14x32

# Conv2 Layer  14x14x32 -> 14x14x64
W_conv2 = weight_variable([5,5,32,64])# [5×5卷积核, in_channels=32, out_channels=64=卷积核个数]
b_conv2 = bias_variable([64])  		  # [64=卷积核个数=bias个数]

#把h_pool1和权值向量W_conv2进行卷积，再加上偏置值，然后用relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)       # 进行max_pooling 14x14x64 -> 7x7x64

# 28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
# 第二次卷积后为14*14，第二次池化后变为了7*7
# => 进过上面操作后得到7*7*64

# 初始化第一个FC全连接层的权值和偏置
W_fc1 = weight_variable([7*7*64,1024])  
b_fc1 = bias_variable([1024])

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])  # -1表示扁平化为1维（1行，7*7*64列）

# 第一个连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)  #matmul 表示矩阵相乘 得到1*1024的矩阵

# 为了减轻过拟合使用一个Dropout层
keep_prob = tf.placeholder(tf.float32)  #占位符
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)  # 使输入tensor中某些元素变为0，其它没变0的元素值变为原来的1/keep_prob大小

# Dropout层 softmax连接输出层，初始化第二个FC全连接层
W_fc2 = weight_variable([1024,10])  
b_fc2 = bias_variable([10]) 

# 计算输出 
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)  #矩阵相乘 得到1*10矩阵

# 损失函数loss function: 交叉熵cross_entropy 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))  

# 优化算法Adam函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 优化方法：AdamOptimizer| 学习速率：(1e-4)| 交叉熵：最小化

# 存放结果到一个布尔列表 ：tf.argmax表示找到最大值的位置(也就是预测的分类和实际的分类)，然后看看他们是否一致，是就返回true,不是就返回false,这样得到一个boolean数组
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))  # tf.equal返回布尔值 | tf.argmax(y_,1)：数字1代表最大值
# 准确率：tf.cast将boolean数组转成int数组，最后求平均值，得到分类的准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  
  
tf.global_variables_initializer().run() #所有变量初始化
# 训练20000次 每次大小为50的mini-batch 每1000次训练查看训练结果 用以实时监测模型性能 
for i in range(20000):  
    batch = mnist.train.next_batch(50)          # 喂入训练集的数据
    if i % 1000 == 0:                           # 批量梯度下降，把1000改成1：随机梯度下降
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}) # train accuracy: accuracy.eval
        print("step %d, training accuracy %g"%(i,train_accuracy))  
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})                     # test accuracy: accuracy.eval
  
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
