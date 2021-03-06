import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from models.conv_lstm import basic_conv_lstm_cell

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12
FC_LAYER_SIZE = 1024
FC_LSTM_LAYER_SIZE = 1024

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


def encoder_model(frames, sequence_length, initializer, scope='encoder', fc_conv_layer=False):
  """
  Args:
    frames: 5D array of batch with videos - shape(batch_size, num_frames, frame_width, frame_higth, num_channels)
    sequence_length: number of frames that shall be encoded
    scope: tensorflow variable scope name
    initializer: specifies the initialization type (default: contrib.slim.layers uses Xavier init with uniform data)
    fc_conv_layer: adds an fc layer at the end of the encoder
  Returns:
    hidden4: hidden state of highest ConvLSTM layer
    fc_conv_layer: indicated whether a Fully Convolutional (8x8x16 -> 1x1x1024) shall be added
  """

  lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6 = None, None, None, None, None, None

  for i in range(sequence_length):

    frame = frames[:,i,:,:,:]

    reuse = (i > 0)

    with tf.variable_scope(scope, reuse=reuse):
      #LAYER 1: conv1
      conv1 = slim.layers.conv2d(frame, 16, [5, 5], stride=2, scope='conv1', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
          normalizer_params={'scope': 'layer_norm1'})

      #LAYER 2: convLSTM1
      hidden1, lstm_state1 = basic_conv_lstm_cell(conv1, lstm_state1, 16, initializer, filter_size=5, scope='convlstm1')
      hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')

      #LAYER 3: conv2
      conv2 = slim.layers.conv2d(hidden1, hidden1.get_shape()[3], [5, 5], stride=2, scope='conv2', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
                                  normalizer_params={'scope': 'layer_norm3'})

      #LAYER 4: convLSTM2
      hidden2, lstm_state2 = basic_conv_lstm_cell(conv2, lstm_state2, 16, initializer, filter_size=5, scope='convlstm2')
      hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm4')

      #LAYER 5: conv3
      conv3 = slim.layers.conv2d(hidden2, hidden2.get_shape()[3], [5, 5], stride=2, scope='conv3', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
                                  normalizer_params={'scope': 'layer_norm5'})

      #LAYER 6: convLSTM3
      hidden3, lstm_state3 = basic_conv_lstm_cell(conv3, lstm_state3, 16, initializer, filter_size=3, scope='convlstm3')
      hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm6')


      #LAYER 7: conv4
      conv4 = slim.layers.conv2d(hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv4', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
                                 normalizer_params={'scope': 'layer_norm7'})

      #LAYER 8: convLSTM4 (8x8 featuremap size)
      hidden4, lstm_state4 = basic_conv_lstm_cell(conv4, lstm_state4, 32, initializer, filter_size=3, scope='convlstm4')
      hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm8')

      #LAYER 8: conv5
      conv5 = slim.layers.conv2d(hidden4, hidden4.get_shape()[3], [3, 3], stride=2, scope='conv5', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer, 
                                 normalizer_params={'scope': 'layer_norm9'})

      # LAYER 9: convLSTM5 (4x84 featuremap size)
      hidden5, lstm_state5 = basic_conv_lstm_cell(conv5, lstm_state5, 32, initializer, filter_size=3, scope='convlstm5')
      hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm10')

      #LAYER 10: Fully Convolutional Layer (4x4x32 --> 1x1xFC_LAYER_SIZE)
      fc_conv = slim.layers.conv2d(hidden5, FC_LAYER_SIZE, [4,4], stride=1, scope='fc_conv', padding='VALID', weights_initializer=initializer)

      #LAYER 11: Fully Convolutional LSTM (1x1x256 -> 1x1x128)
      hidden6, lstm_state6 = basic_conv_lstm_cell(fc_conv, lstm_state6, FC_LSTM_LAYER_SIZE, initializer, filter_size=1, scope='convlstm6')

      hidden_repr = hidden6

  return hidden_repr


def decoder_model(hidden_repr, sequence_length, initializer, num_channels=3, scope='decoder', fc_conv_layer=False):
  """
  Args:
    hidden_repr: Tensor of latent space representation
    sequence_length: number of frames that shall be decoded from the hidden_repr
    num_channels: number of channels for generated frames
    initializer: specifies the initialization type (default: contrib.slim.layers uses Xavier init with uniform data)
    fc_conv_layer: adds an fc layer at the end of the encoder
  Returns:
    frame_gen: array of generated frames (Tensors)
    fc_conv_layer: indicates whether hidden_repr is 1x1xdepth tensor a and fully concolutional layer shall be added
  """
  frame_gen = []

  lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state0 = None, None, None, None, None, None
  assert (not fc_conv_layer) or (hidden_repr.get_shape()[1] == hidden_repr.get_shape()[2] == 1)

  for i in range(sequence_length):
    reuse = (i > 0) #reuse variables (recurrence) after first time step

    with tf.variable_scope(scope, reuse=reuse):

      #Fully Convolutional Layer (1x1xFC_LAYER_SIZE -> 4x4x16)
      hidden0, lstm_state0 = basic_conv_lstm_cell(hidden_repr, lstm_state0, FC_LAYER_SIZE, initializer, filter_size=1,
                                                  scope='convlstm0')


      fc_conv = slim.layers.conv2d_transpose(hidden0, 32, [4, 4], stride=1, scope='fc_conv', padding='VALID', weights_initializer=initializer)


      #LAYER 1: convLSTM1
      hidden1, lstm_state1 = basic_conv_lstm_cell(fc_conv, lstm_state1, 32, initializer, filter_size=3, scope='convlstm1')
      hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm1')

      #LAYER 2: upconv1 (8x8 -> 16x16)
      upconv1 = slim.layers.conv2d_transpose(hidden1, hidden1.get_shape()[3], 3, stride=2, scope='upconv1', weights_initializer=initializer,
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm2'})

      #LAYER 3: convLSTM2
      hidden2, lstm_state2 = basic_conv_lstm_cell(upconv1, lstm_state2, 32, initializer, filter_size=3, scope='convlstm2')
      hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')

      #LAYER 4: upconv2 (16x16 -> 32x32)
      upconv2 = slim.layers.conv2d_transpose(hidden2, hidden2.get_shape()[3], 3, stride=2, scope='upconv2', weights_initializer=initializer,
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm4'})

      #LAYER 5: convLSTM3
      hidden3, lstm_state3 = basic_conv_lstm_cell(upconv2, lstm_state3, 16, initializer, filter_size=3, scope='convlstm3')
      hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm5')

      # LAYER 6: upconv3 (32x32 -> 64x64)
      upconv3 = slim.layers.conv2d_transpose(hidden3, hidden3.get_shape()[3], 5, stride=2, scope='upconv3', weights_initializer=initializer,
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm6'})

      #LAYER 7: convLSTM4
      hidden4, lstm_state4 = basic_conv_lstm_cell(upconv3, lstm_state4, 16, initializer, filter_size=5, scope='convlstm4')
      hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm7')

      #Layer 8: upconv4 (64x64 -> 128x128)
      upconv4 = slim.layers.conv2d_transpose(hidden4, 16, 5, stride=2, scope='upconv4', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
                                             normalizer_params={'scope': 'layer_norm8'})

      #LAYER 9: convLSTM5
      hidden5, lstm_state5 = basic_conv_lstm_cell(upconv4, lstm_state5, 16, initializer, filter_size=5, scope='convlstm5')
      hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm9')

      upconv5 = slim.layers.conv2d_transpose(hidden5, num_channels, 5, stride=2, scope='upconv5', weights_initializer=initializer)

      frame_gen.append(upconv5)

  assert len(frame_gen)==sequence_length
  return frame_gen


def composite_model(frames, encoder_len=5, decoder_future_len=5, decoder_reconst_len=5, uniform_init=True, num_channels=3, fc_conv_layer=True):
  """
  Args:
    frames: 5D array of batch with videos - shape(batch_size, num_frames, frame_width, frame_higth, num_channels)
    encoder_len: number of frames that shall be encoded
    decoder_future_sequence_length: number of frames that shall be decoded from the hidden_repr
    uniform_init: specifies if the weight initialization should be drawn from gaussian or uniform distribution (default:uniform)
    num_channels: number of channels for generated frames
    fc_conv_layer: indicates whether fully connected layer shall be added between encoder and decoder
  Returns:
    frame_gen: array of generated frames (Tensors)
  """
  assert all([len > 0 for len in [encoder_len, decoder_future_len, decoder_reconst_len]])
  initializer = tf_layers.xavier_initializer(uniform=uniform_init)
  hidden_repr = encoder_model(frames, encoder_len, initializer, fc_conv_layer=fc_conv_layer)
  frames_pred = decoder_model(hidden_repr, decoder_future_len, initializer, num_channels=num_channels,
                              scope='decoder_pred', fc_conv_layer=fc_conv_layer)
  frames_reconst = decoder_model(hidden_repr, decoder_reconst_len, initializer, num_channels=num_channels,
                                 scope='decoder_reconst', fc_conv_layer=fc_conv_layer)
  return frames_pred, frames_reconst, hidden_repr
