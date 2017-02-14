from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import tf_test
import model

ENCODER_LENGTH = 5
DECODER_FUTURE_LENGTH = 5
DECODER_RECONST_LENGTH = 5
LOSS_FUNCTIONS = ['mse']

def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))

def decoder_loss(frames_gen, frames_original, loss_fun): #TODO make more efficient by replacing loop by reshape
  """Sum of parwise l2 distance between frames of frames_gen and frames_original
    Args:
    frames_gen: array of length=sequence_length of Tensors with each having the shape=(batch size, frame_height, frame_width, num_channels)
    frames_original: Tensor with shape=(batch size, sequence_length, frame_height, frame_width, num_channels)
    loss_fun: loss function type ['mse',...]
  Returns:
    sum of mean squared error between ground truth and predicted frames of provided sequence.
  """
  assert loss_fun in LOSS_FUNCTIONS
  loss = 0.0
  if loss_fun == 'mse':
    for i in range(len(frames_gen)):
      loss += mean_squared_error(frames_original[:,i,:,:,:], frames_gen[i])
  return loss

def composite_loss(original_frames, frames_pred, frames_reconst, loss_fun='mse',
                   encoder_length=ENCODER_LENGTH, decoder_future_length=DECODER_FUTURE_LENGTH, decoder_reconst_length=DECODER_RECONST_LENGTH):
  assert encoder_length<=decoder_reconst_length
  assert loss_fun in LOSS_FUNCTIONS
  frames_original_future = original_frames[:, (encoder_length):(encoder_length + decoder_future_length ), :, :, :]
  frames_original_reconst = original_frames[:, (encoder_length - decoder_reconst_length):encoder_length, :, :, :]
  pred_loss = decoder_loss(frames_pred, frames_original_future, loss_fun)
  reconst_loss = decoder_loss(frames_reconst, frames_original_reconst, loss_fun)
  return pred_loss + reconst_loss

def train():

  #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, None, 128, 128, 1]) # 128x128 images

  frames_pred, frames_reconst = model.composite_model(x, ENCODER_LENGTH, DECODER_FUTURE_LENGTH, DECODER_RECONST_LENGTH, num_channels=1)

  #Mean Squared Error - Loss Function
  loss = composite_loss(x, frames_pred, frames_reconst)

  #choose optimizer
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

  #start session
  sess.run(tf.global_variables_initializer())


  #train
  for i in range(20000):
    batch = np.random.rand(50,10,128,128,1)
    assert(batch.shape[2] >= (ENCODER_LENGTH + DECODER_FUTURE_LENGTH))
    train_step.run(feed_dict={x: batch})
    print(i)

train()
