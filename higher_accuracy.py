import input_data
#下载和导入MNIST数据集。它会自动创建一个'MNIST_data'的目录来存储数据
#mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
#InteractiveSession类，它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的
import tensorflow as tf
sess = tf.InteractiveSession()
 
"""为了创建这个模型，我们需要创建大量的权重和偏置项。这个模型中的权重在初始化时应该
加入少量的噪声来打破对称性以及避免0梯度。由于我们使用的是ReLU神经元，因此比较好的做法
是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。
为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。"""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
 
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

"""我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。
我们的池化用简单传统的2x2大小的模板做max pooling。为了代码更简洁，我们把这部分抽象成一个函数。""" 
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
 
 
# Create the model
# placeholder
  """这里的x和y并不是特定的值，相反，他们都只是一个占位符.输入图片x是一个2维的浮点数张量。
  这里，分配给它的shape为[None, 784]，其中784是一张展平的MNIST图片的维度。
  None表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小，意即x的数量不定。
  输出类别值y_也是一个2维张量，其中每一行为一个10维的one-hot向量,
  用于代表对应某一MNIST图片的类别。"""
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
 

#构建卷积层
"""weight_variable的两个5表明卷积核大小为5，1应该还是图像的通道数，
32表明该卷积层会提取32个特征，也就是会输出32个maps。 """
#first
#第一层卷积
"""卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，
最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。"""
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
 
"""输入的图片本来是一维的，需要把它调整为四维的。-1表示任意多个，两个28指的是图片的长和宽，
1是说图片是灰度图，只有一个通道。
为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1"""
x_image = tf.reshape(x, [-1,28,28,1])
 
"""我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。"""
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
 
#second第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
 
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
 
 
#密集连接层
"""现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。"""
 
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
 
 
 
#dropout
 #为了减少过拟合，我们在输出层之前加入dropout。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
 
#softmax输出层,最后，我们添加一个softmax层
 
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
 
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 

 #训练和评估模型
 
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "double"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %f"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
 
print ("test accuracy %f"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
