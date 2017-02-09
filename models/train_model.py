from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import tf_test
import model


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, None, 128, 128, 1]) # 28x28 images
y_ = tf.placeholder(tf.float32, shape=[None, 8, 8, 16]) # class labels


y = model.encoder_model(x, 10)


#Mean Squared Error - Loss Function
loss = mean_squared_error(y_,y)

#choose optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#start session
sess.run(tf.global_variables_initializer())


#train
for i in range(20000):
  batch = np.random.rand(50,10,128,128,1)
  random_y = np.random.rand(50,8,8,16)

  train_step.run(feed_dict={x: batch, y_: random_y})
  print(i)

