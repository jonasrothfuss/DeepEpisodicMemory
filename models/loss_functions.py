import tensorflow as tf
import numpy as np

def gradient_difference_loss(true, pred, alpha=2.0):
  """
  computes gradient difference loss of two images
  :param ground truth image: Tensor of shape (batch_size, frame_height, frame_width, num_channels)
  :param predicted image: Tensor of shape (batch_size, frame_height, frame_width, num_channels)
  :param alpha parameter of the used l-norm
  """
  #tf.assert_equal(tf.shape(true), tf.shape(pred))
  # vertical
  true_pred_diff_vert = tf.pow(tf.abs(difference_gradient(true, vertical=True) - difference_gradient(pred, vertical=True)), alpha)
  # horizontal
  true_pred_diff_hor = tf.pow(tf.abs(difference_gradient(true, vertical=False) - difference_gradient(pred, vertical=False)), alpha)
  # normalization over all dimensions
  return (tf.reduce_mean(true_pred_diff_vert) + tf.reduce_mean(true_pred_diff_hor)) / tf.to_float(2)


def difference_gradient(image, vertical=True):
  """
  :param image: Tensor of shape (batch_size, frame_height, frame_width, num_channels)
  :param vertical: boolean that indicates whether vertical or horizontal pixel gradient shall be computed
  :return: difference_gradient -> Tenor of shape (:, frame_height-1, frame_width, :) if vertical and (:, frame_height, frame_width-1, :) else
  """
  s = tf.shape(image)
  if vertical:
    return tf.abs(image[:, 0:s[1] - 1, :, :] - image[:, 1:s[1], :, :])
  else:
    return tf.abs(image[:, :, 0:s[2]-1,:] - image[:, :, 1:s[2], :])

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


def vae_error(V):
  def KL(x, y):
    X = tf.contrib.distributions.Categorical(p=x)
    Y = tf.contrib.distributions.Categorical(p=y)
    return tf.contrib.distributions.kullback_leibler.kl(X, Y)

  N = tf.random_normal(shape=tf.shape(V), mean=0., stddev=1.)

  return KL(V, N)


def decoder_loss(frames_gen, frames_original, loss_fun, V=None):
  """Sum of parwise loss between frames of frames_gen and frames_original
    Args:
    frames_gen: array of length=sequence_length of Tensors with each having the shape=(batch size, frame_height, frame_width, num_channels)
    frames_original: Tensor with shape=(batch size, sequence_length, frame_height, frame_width, num_channels)
    loss_fun: loss function type ['mse',...]
  Returns:
    loss: sum (specified) loss between ground truth and predicted frames of provided sequence.
  """
  loss = 0.0
  if loss_fun == 'mse':
    for i in range(len(frames_gen)):
      loss += mean_squared_error(frames_original[:, i, :, :, :], frames_gen[i])
  elif loss_fun == 'gdl':
    for i in range(len(frames_gen)):
      loss += gradient_difference_loss(frames_original[:, i, :, :, :], frames_gen[i])
  elif loss_fun == 'mse_gdl':
    for i in range(len(frames_gen)):
      loss += 0.4 * gradient_difference_loss(frames_original[:, i, :, :, :], frames_gen[i]) + 0.6 * mean_squared_error(frames_original[:, i, :, :, :], frames_gen[i])
  elif loss_fun == 'vae':
      assert V is not None
      for i in range(len(frames_gen)):
        loss += mean_squared_error(frames_original[:, i, :, :, :], frames_gen[i]) + vae_error(V)
  else:
    raise Exception('Unknown loss funcion type')
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


def composite_loss(original_frames, frames_pred, frames_reconst, loss_fun='vae',
                   encoder_length=5, decoder_future_length=5,
                   decoder_reconst_length=5, hidden_repr=None):

  assert encoder_length <= decoder_reconst_length
  frames_original_future = original_frames[:, (encoder_length):(encoder_length + decoder_future_length), :, :, :]
  frames_original_reconst = original_frames[:, (encoder_length - decoder_reconst_length):encoder_length, :, :, :]
  pred_loss = decoder_loss(frames_pred, frames_original_future, loss_fun, V=hidden_repr)
  reconst_loss = decoder_loss(frames_reconst, frames_original_reconst, loss_fun, V=hidden_repr)
  return pred_loss + reconst_loss
