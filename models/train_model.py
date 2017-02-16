from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import model

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

tf.logging.set_verbosity(tf.logging.INFO)

ENCODER_LENGTH = 5  # TODO: define sequence length as flag
DECODER_FUTURE_LENGTH = 5
DECODER_RECONST_LENGTH = 5
LOSS_FUNCTIONS = ['mse', 'gdl']

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('loss_function', 'gdl', 'loss function to minimize')


def gradient_difference_loss(true, pred, alpha=2.0):
  """"Gradient differnce loss (GDL) implementation (see Mathieu et al. (2015) http://arxiv.org/abs/1511.05440)
  Args:
    true: the ground truth image. 4D tensor -> shape=(batch size, frame_height, frame_width, num_channels)
    pred: the predicted image. 4D tensor -> shape=(batch size, frame_height, frame_width, num_channels)
    alpha: parameter of alpha-norm (if 2.0 -> l2 norm between true and pred pixel gradients)
  Returns:
    reduced tensor corresponding to the gradient difference loss between ground truth and predicted image.
  """
  tf.assert_equal(tf.shape(true), tf.shape(pred))
  #vertical
  true_pred_diff_vert = tf.pow(tf.abs(difference_gradient(true, vertical=True) - difference_gradient(pred, vertical=True)), alpha)
  # horizontal
  true_pred_diff_hor = tf.pow(tf.abs(difference_gradient(true, vertical=False) - difference_gradient(pred, vertical=False)), alpha)
  # return mean diff
  return (tf.reduce_sum(true_pred_diff_vert) + tf.reduce_sum(true_pred_diff_hor)) / tf.to_float(2*tf.size(pred))

def difference_gradient(image, vertical=True):
  """computes pixel difference gradient in an image (Tensor)
  Args:
    image: 4D tensor corresponding to to image shape=(batch size, frame_height, frame_width, num_channels)
  Returns:
    reduced tensor corresponding to the gradient difference loss between ground truth and predicted image.
  """

  # careful! begin is zero-based; size is one-based
  if vertical:
    begin0 = [0, 0, 0, 0]
    begin1 = [0, 1, 0, 0]
    size = [tf.shape(image)[0], tf.shape(image)[1] - 1, tf.shape(image)[2], tf.shape(image)[3]]
  else: # horizontal
    begin0 = [0, 0, 0, 0]
    begin1 = [0, 0, 1, 0]
    size = [tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2] - 1, tf.shape(image)[3]]

  slice0 = tf.slice(image[:, :, :, :], begin0, size)
  slice1 = tf.slice(image[:, :, :, :], begin1, size)
  return tf.abs(tf.sub(slice0, slice1))


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def decoder_loss(frames_gen, frames_original, loss_fun):
  """Sum of parwise loss between frames of frames_gen and frames_original
    Args:
    frames_gen: array of length=sequence_length of Tensors with each having the shape=(batch size, frame_height, frame_width, num_channels)
    frames_original: Tensor with shape=(batch size, sequence_length, frame_height, frame_width, num_channels)
    loss_fun: loss function type ['mse',...]
  Returns:
    loss: sum (specified) loss between ground truth and predicted frames of provided sequence.
  """
  assert loss_fun in LOSS_FUNCTIONS
  loss = 0.0
  if loss_fun == 'mse':
    for i in range(len(frames_gen)):
      loss += mean_squared_error(frames_original[:, i, :, :, :], frames_gen[i])
  if loss_fun == 'gdl':
    for i in range(len(frames_gen)):
      loss += gradient_difference_loss(frames_original[:, i, :, :, :], frames_gen[i])
  return loss


def decoder_psnr(frames_gen, frames_original, loss_fun):
  """Sum of peak_signal_to_noise_ratio loss between frames of frames_gen and frames_original
     Args:
       frames_gen: array of length=sequence_length of Tensors with each having the shape=(batch size, frame_height, frame_width, num_channels)
       frames_original: Tensor with shape=(batch size, sequence_length, frame_height, frame_width, num_channels)
       loss_fun: loss function type ['mse',...]
     Returns:
       loss: sum of mean squared error between ground truth and predicted frames of provided sequence.
  """
  psnr = 0.0
  for i in range(len(frames_gen)):
    psnr += peak_signal_to_noise_ratio(frames_original[:, i, :, :, :], frames_gen[i])
  return psnr


def composite_loss(original_frames, frames_pred, frames_reconst, loss_fun='mse',
                   encoder_length=ENCODER_LENGTH, decoder_future_length=DECODER_FUTURE_LENGTH,
                   decoder_reconst_length=DECODER_RECONST_LENGTH):
  assert encoder_length <= decoder_reconst_length
  assert loss_fun in LOSS_FUNCTIONS
  frames_original_future = original_frames[:, (encoder_length):(encoder_length + decoder_future_length), :, :, :]
  frames_original_reconst = original_frames[:, (encoder_length - decoder_reconst_length):encoder_length, :, :, :]
  pred_loss = decoder_loss(frames_pred, frames_original_future, loss_fun)
  reconst_loss = decoder_loss(frames_reconst, frames_original_reconst, loss_fun)
  return pred_loss + reconst_loss


def main(unused_argv):
  # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  x = tf.placeholder(tf.float32, shape=[None, None, 128, 128, 1])  # 128x128 images

  frames_pred, frames_reconst = model.composite_model(x, ENCODER_LENGTH, DECODER_FUTURE_LENGTH, DECODER_RECONST_LENGTH,
                                                      num_channels=1)

  # Loss Function
  loss = composite_loss(x, frames_pred, frames_reconst, loss_fun=FLAGS.loss_function)

  # choose optimizer
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

  sess = tf.InteractiveSession()

  # start session
  sess.run(tf.global_variables_initializer())

  # train
  for i in range(FLAGS.num_iterations):
    batch = np.random.rand(50, 10, 128, 128, 1)
    assert (batch.shape[2] >= (ENCODER_LENGTH + DECODER_FUTURE_LENGTH))
    train_step.run(feed_dict={x: batch})
    tf.logging.info(str(i))


if __name__ == '__main__':
  app.run()

