from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_relu(input, kernel_shape, bias_shape):
  w = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
  b = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
  conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding = 'SAME')
  return tf.nn.relu(conv+b)


def conv_layers(x_image):
  # conv1 layer
  with tf.variable_scope("conv1"):
    h_conv1 = conv_relu(x_image, [4, 4, 1, 32], [32])

  # max_pool1
  h_pool1 = max_pool_2x2(h_conv1)

  # conv2 layer
  with tf.variable_scope("conv2"):
    h_conv2 = conv_relu(h_pool1, [3, 3, 32, 64], [64])

  # max_pool2
  h_pool2 = max_pool_2x2(h_conv2)

  return h_pool2


def main():
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, 784])  # 28x28 images
  y_ = tf.placeholder(tf.float32, shape=[None, 10])  # class labels

  #image input reshape
  x_image = tf.reshape(x, [-1,28,28,1])

  h_pool2 = conv_layers(x_image)

  #fc1
  W_fc1 = weight_variable([7*7*64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  #Dropout
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  #fc2
  W_fc2 = W_fc1 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  #cross_entropy & softmax
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

  #choose optimizer
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  #accuracy
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  #start session
  sess.run(tf.initialize_all_variables())


  #train
  for i in range(20000):
    batch = mnist.train.next_batch(50)

    #train step
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #train and val error
    if i % 100 == 0:
      print('step:', i,
            'train_accuracy:', accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),
            'val_accuracy:', accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
