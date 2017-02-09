import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from conv_lstm import basic_conv_lstm_cell

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


def encoder_model(frames, sequence_length, iter_num=-1.0, k=-1, use_state=True, context_frames=2):
  """
  Args:
    images: tensor of ground truth image sequences
    actions: tensor of action sequences
    states: tensor of ground truth state sequences
    iter_num: tensor of the current training iteration (for sched. sampling)
    k: constant used for scheduled sampling. -1 to feed in own prediction.
    use_state: True to include state and action in prediction
    num_masks: the number of different pixel motion predictions (and
               the number of masks for each of those predictions)
    context_frames: number of ground truth frames to pass in before
                    feeding in own predictions
  Returns:
    gen_images: predicted future image frames
    gen_states: predicted future states
  Raises:
    ValueError: if more than one network option specified or more than 1 mask
    specified for DNA model.
  """

  lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None

  for i in range(sequence_length):

    frame = frames[:,i,:,:,:]

    reuse = (i > 0)
    with slim.arg_scope(
            [basic_conv_lstm_cell, slim.layers.conv2d, slim.layers.fully_connected,
             tf_layers.layer_norm, slim.layers.conv2d_transpose],
            reuse=reuse):

      #LAYER 1: conv1
      conv1 = slim.layers.conv2d(frame, 16, [5, 5], stride=2, scope='conv1', normalizer_fn=tf_layers.layer_norm,
          normalizer_params={'scope': 'layer_norm1'})

      #LAYER 2: convLSTM1
      hidden1, lstm_state1 = basic_conv_lstm_cell(conv1, lstm_state1, 16, filter_size=5, scope='convlstm1')
      hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')

      #LAYER 3: conv2
      conv2 = slim.layers.conv2d(hidden1, hidden1.get_shape()[3], [5, 5], stride=2, scope='conv2', normalizer_fn=tf_layers.layer_norm,
                                  normalizer_params={'scope': 'layer_norm3'})

      #LAYER 4: convLSTM2
      hidden2, lstm_state2 = basic_conv_lstm_cell(conv2, lstm_state2, 16, filter_size=5, scope='convlstm2')
      hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm4')

      #LAYER 5: conv3
      conv3 = slim.layers.conv2d(hidden2, hidden2.get_shape()[3], [5, 5], stride=2, scope='conv3', normalizer_fn=tf_layers.layer_norm,
                                  normalizer_params={'scope': 'layer_norm5'})

      #LAYER 6: convLSTM3
      hidden3, lstm_state3 = basic_conv_lstm_cell(conv3, lstm_state3, 16, filter_size=3, scope='convlstm3')
      hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm6')


      #LAYER 7: conv4
      conv4 = slim.layers.conv2d(hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv4', normalizer_fn=tf_layers.layer_norm,
                                 normalizer_params={'scope': 'layer_norm7'})

      #LAYER 8: convLSTM4 (8x8 featuremap size)
      hidden4, lstm_state4 = basic_conv_lstm_cell(conv4, lstm_state4, 16, filter_size=3, scope='convlstm4')
      hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm8')

  return hidden4

'''
def decoder_model(hidden, scope='decoder')

  # LAYER 9: upconv1 (8x8 -> 16x16)
  upconv1 = slim.layers.conv2d_transpose(hidden, hidden.get_shape()[3], 3, stride=2, scope='upconv1',
                                         normalizer_fn=tf_layers.layer_norm,
                                         normalizer_params={'scope': 'layer_norm9'})

  # LAYER 10: convLSTM5
  hidden5, lstm_state5 = basic_conv_lstm_cell(upconv1, lstm_state5, 16, filter_size=3, scope='convlstm5')
  hidden5 = tf_layers.layer_norm(hidden1, scope='layer_norm10')

  # LAYER 11: upconv2 (16x16 -> 32x32)
  upconv2 = slim.layers.conv2d_transpose(hidden5, hidden5.get_shape()[3], 3, stride=2, scope='upconv2',
                                         normalizer_fn=tf_layers.layer_norm,
                                         normalizer_params={'scope': 'layer_norm11'})

  # LAYER 12: convLSTM6
  hidden6, lstm_state6 = basic_conv_lstm_cell(upconv2, lstm_state6, 16, filter_size=3, scope='convlstm6')
  hidden6 = tf_layers.layer_norm(hidden1, scope='layer_norm12')

  # LAYER 13: upconv3 (32x32 -> 64x64)
  upconv3 = slim.layers.conv2d_transpose(hidden6, hidden6.get_shape()[3], 3, stride=2, scope='upconv3',
                                         normalizer_fn=tf_layers.layer_norm,
                                         normalizer_params={'scope': 'layer_norm13'})

  # LAYER 14: convLSTM7
  hidden7, lstm_state7 = basic_conv_lstm_cell(upconv3, lstm_state7, 16, filter_size=3, scope='convlstm6')
  hidden7 = tf_layers.layer_norm(hidden1, scope='layer_norm12')

  # LAYER 15: upconv3 (32x32 -> 64x64)
  upconv3 = slim.layers.conv2d_transpose(hidden6, hidden6.get_shape()[3], 3, stride=2, scope='upconv3',
                                         normalizer_fn=tf_layers.layer_norm,
                                         normalizer_params={'scope': 'layer_norm13'})

'''


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
  """Sample batch with specified mix of ground truth and generated data points.
  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    num_ground_truth: number of ground-truth examples to include in batch.
  Returns:
    New batch with num_ground_truth sampled from ground_truth_x and the rest
    from generated_x.
  """
  idx = tf.random_shuffle(tf.range(int(batch_size)))
  ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
  generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

  ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
  generated_examps = tf.gather(generated_x, generated_idx)
  return tf.dynamic_stitch([ground_truth_idx, generated_idx],
[ground_truth_examps, generated_examps])