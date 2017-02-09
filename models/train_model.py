from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import tf_test
import model

ENCODER_LENGTH = 5
DECODER_LENGTH = 5

def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))

def decoder_mse(frames_gen, frames_original):
  loss = 0.0
  for i in range(len(frames_gen)):
    loss += mean_squared_error(frames_original[:,i,:,:,:], frame_gen[i])
  return loss


#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, None, 128, 128, 1]) # 128x128 images

frame_gen = model.composite_model(x, ENCODER_LENGTH, DECODER_LENGTH, num_channels=1)

#Mean Squared Error - Loss Function
frames_original = x[:,(ENCODER_LENGTH-1):(ENCODER_LENGTH+DECODER_LENGTH-1),:,:,:] # x[(ENCODER_LENGTH-1):(ENCODER_LENGTH+DECODER_LENGTH-1),:,:,:]
loss = decoder_mse(frame_gen, frames_original)

#choose optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#start session
sess.run(tf.global_variables_initializer())


#train
for i in range(20000):
  batch = np.random.rand(50,10,128,128,1)
  assert(batch.shape[2] >= (ENCODER_LENGTH + DECODER_LENGTH))
  train_step.run(feed_dict={x: batch})
  print(i)

