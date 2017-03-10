import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from models.conv_lstm import basic_conv_lstm_cell

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12
FC_LAYER_SIZE = 512

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


def encoder_model(frames, sequence_length, scope='encoder', fc_conv_layer=False):
  """
  Args:
    frames: 5D array of batch with videos - shape(batch_size, num_frames, frame_width, frame_higth, num_channels)
    sequence_length: number of frames that shall be encoded
    scope: tensorflow variable scope name
  Returns:
    hidden4: hidden state of highest ConvLSTM layer
    fc_conv_layer: indicated whether a Fully Convolutional (8x8x16 -> 1x1x1024) shall be added
  """

  lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None

  for i in range(sequence_length):

    frame = frames[:,i,:,:,:]

    reuse = (i > 0)

    with tf.variable_scope(scope, reuse=reuse):
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
      conv3 = slim.layers.conv2d(hidden2, hidden2.get_shape()[3], [3, 3], stride=2, scope='conv3', normalizer_fn=tf_layers.layer_norm,
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

      #LAYER 9: Fully Convolutional Layer (8x8x16 --> 1x1xFC_LAYER_SIZE)
      if fc_conv_layer:
        fc_conv = slim.layers.conv2d(hidden4, FC_LAYER_SIZE, [8,8], stride=1, scope='fc_conv', padding='VALID')
        hidden_repr = fc_conv
      else:
        hidden_repr = hidden4

  return hidden_repr


def decoder_model(hidden_repr, sequence_length, num_channels=3, scope='decoder', fc_conv_layer=False):
  """
  Args:
    hidden_repr: Tensor of latent space representation
    sequence_length: number of frames that shall be decoded from the hidden_repr
    num_channels: number of channels for generated frames
  Returns:
    frame_gen: array of generated frames (Tensors)
    fc_conv_layer: indicates whether hidden_repr is 1x1xdepth tensor a and fully concolutional layer shall be added
  """
  frame_gen = []

  lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
  assert (not fc_conv_layer) or (hidden_repr.get_shape()[1] == hidden_repr.get_shape()[2] == 1)

  for i in range(sequence_length):
    reuse = (i > 0) #reuse variables (recurrence) after first time step

    with tf.variable_scope(scope, reuse=reuse):

      #Fully Convolutional Layer (1x1xFC_LAYER_SIZE -> 8x8x16)
      if fc_conv_layer:
        fc_conv = slim.layers.conv2d_transpose(hidden_repr, 16, [8, 8], stride=1, scope='fc_conv', padding='VALID')
        hidden1_input = fc_conv
      else:
        hidden1_input = hidden_repr

      #LAYER 1: convLSTM1
      hidden1, lstm_state1 = basic_conv_lstm_cell(hidden1_input, lstm_state1, 16, filter_size=3, scope='convlstm1')
      hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm1')

      #LAYER 2: upconv1 (8x8 -> 16x16)
      upconv1 = slim.layers.conv2d_transpose(hidden1, hidden1.get_shape()[3], 3, stride=2, scope='upconv1',
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm2'})

      #LAYER 3: convLSTM2
      hidden2, lstm_state2 = basic_conv_lstm_cell(upconv1, lstm_state2, 16, filter_size=3, scope='convlstm2')
      hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')

      #LAYER 4: upconv2 (16x16 -> 32x32)
      upconv2 = slim.layers.conv2d_transpose(hidden2, hidden2.get_shape()[3], 3, stride=2, scope='upconv2',
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm4'})

      #LAYER 5: convLSTM3
      hidden3, lstm_state3 = basic_conv_lstm_cell(upconv2, lstm_state3, 16, filter_size=5, scope='convlstm3')
      hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm5')

      # LAYER 6: upconv3 (32x32 -> 64x64)
      upconv3 = slim.layers.conv2d_transpose(hidden3, hidden3.get_shape()[3], 5, stride=2, scope='upconv3',
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm6'})

      #LAYER 7: convLSTM4
      hidden4, lstm_state4 = basic_conv_lstm_cell(upconv3, lstm_state4, 16, filter_size=5, scope='convlstm4')
      hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm7')

      #Layer 8: upconv4 (64x64 -> 128x128)
      upconv4 = slim.layers.conv2d_transpose(hidden4, num_channels, 5, stride=2, scope='upconv4')

      frame_gen.append(upconv4)

  assert len(frame_gen)==sequence_length
  return frame_gen


def composite_model(frames, encoder_len=5, decoder_future_len=5, decoder_reconst_len=5, num_channels=3, fc_conv_layer=False):
  """
  Args:
    frames: 5D array of batch with videos - shape(batch_size, num_frames, frame_width, frame_higth, num_channels)
    encoder_len: number of frames that shall be encoded
    decoder_future_sequence_length: number of frames that shall be decoded from the hidden_repr
    num_channels: number of channels for generated frames
    fc_conv_layer: indicates whether fully connected layer shall be added between encoder and decoder
  Returns:
    frame_gen: array of generated frames (Tensors)
  """
  assert all([len > 0 for len in [encoder_len, decoder_future_len, decoder_reconst_len]])
  hidden_repr = encoder_model(frames, encoder_len, fc_conv_layer=fc_conv_layer)
  frames_pred = decoder_model(hidden_repr, decoder_future_len, num_channels=num_channels,
                              scope='decoder_pred', fc_conv_layer=fc_conv_layer)
  frames_reconst = decoder_model(hidden_repr, decoder_reconst_len, num_channels=num_channels,
                                 scope='decoder_reconst', fc_conv_layer=fc_conv_layer)
  return frames_pred, frames_reconst