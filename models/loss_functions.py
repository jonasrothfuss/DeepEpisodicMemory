import tensorflow as tf

def gradient_difference_loss(true, pred, alpha=2.0):
  """description here"""
  tf.assert_equal(tf.shape(true), tf.shape(pred))
  # vertical
  true_pred_diff_vert = tf.pow(tf.abs(difference_gradient(true, vertical=True) - difference_gradient(pred, vertical=True)), alpha)
  # horizontal
  true_pred_diff_hor = tf.pow(tf.abs(difference_gradient(true, vertical=False) - difference_gradient(pred, vertical=False)), alpha)
  # normalization over all dimensions
  return tf.reduce_sum(true_pred_diff_vert) + tf.reduce_sum(true_pred_diff_hor) / tf.to_float(2*tf.size(pred))


def difference_gradient(image, vertical=True):
  # two dimensional tensor
  # rank = ndim in numpy
  #tf.assert_rank(tf.rank(image), 4)

  # careful! begin is zero-based; size is one-based
  if vertical:
    begin0 = [0, 0, 0]
    begin1 = [1, 0, 0]
    size = [tf.shape(image)[1] - 1, tf.shape(image)[2], tf.shape(image)[3]]
  else: # horizontal
    begin0 = [0, 0, 0]
    begin1 = [0, 1, 0]
    size = [tf.shape(image)[1], tf.shape(image)[2] - 1, tf.shape(image)[3]]

  slice0 = tf.slice(image[0, :, :, :], begin0, size)
  slice1 = tf.slice(image[0, :, :, :], begin1, size)
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


def composite_loss(original_frames, frames_pred, frames_reconst, loss_fun='mse',
                   encoder_length=5, decoder_future_length=5,
                   decoder_reconst_length=5):

  assert encoder_length <= decoder_reconst_length
  frames_original_future = original_frames[:, (encoder_length):(encoder_length + decoder_future_length), :, :, :]
  frames_original_reconst = original_frames[:, (encoder_length - decoder_reconst_length):encoder_length, :, :, :]
  pred_loss = decoder_loss(frames_pred, frames_original_future, loss_fun)
  reconst_loss = decoder_loss(frames_reconst, frames_original_reconst, loss_fun)
  return pred_loss + reconst_loss