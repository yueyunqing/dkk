import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNUST_data/", one_hot = True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# Extracting MNUST_data/train-images-idx3-ubyte.gz
# Extracting MNUST_data/train-labels-idx1-ubyte.gz
# Extracting MNUST_data/t10k-images-idx3-ubyte.gz
# Extracting MNUST_data/t10k-labels-idx1-ubyte.gz
# (55000, 784) (55000, 10)
# (10000, 784) (10000, 10)
# (5000, 784) (5000, 10)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax regression
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

# SGD 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 全局参数初始化
tf.global_variables_initializer().run()

# train
for i in range(1000):
    # 每次抽取100条数据
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_ : batch_ys})

# 返回分类是否正确的类别    
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# cast
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x : mnist.test.images, y_ : mnist.test.labels}))