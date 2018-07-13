import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from models.conv_lstm import basic_conv_lstm_cell, conv_lstm_cell_no_input

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12
FC_LAYER_SIZE = 1000
FC_LSTM_LAYER_SIZE = 1000
VAE_REPR_SIZE = 1000

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


def encoder_model(frames, sequence_length, initializer, keep_prob_dropout=0.9, scope='encoder', fc_conv_layer=False):
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
      conv1 = slim.layers.conv2d(frame, 32, [5, 5], stride=2, scope='conv1', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
          normalizer_params={'scope': 'layer_norm1'})
      conv1 = tf.nn.dropout(conv1, keep_prob_dropout)

      #LAYER 2: convLSTM1
      hidden1, lstm_state1 = basic_conv_lstm_cell(conv1, lstm_state1, 32, initializer, filter_size=5, scope='convlstm1')
      hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
      hidden1 = tf.nn.dropout(hidden1, keep_prob_dropout)

      #LAYER 3: conv2
      conv2 = slim.layers.conv2d(hidden1, hidden1.get_shape()[3], [5, 5], stride=2, scope='conv2', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
                                  normalizer_params={'scope': 'layer_norm3'})
      conv2 = tf.nn.dropout(conv2, keep_prob_dropout)

      #LAYER 4: convLSTM2
      hidden2, lstm_state2 = basic_conv_lstm_cell(conv2, lstm_state2, 32, initializer, filter_size=5, scope='convlstm2')
      hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm4')
      hidden2 = tf.nn.dropout(hidden2, keep_prob_dropout)

      #LAYER 5: conv3
      conv3 = slim.layers.conv2d(hidden2, hidden2.get_shape()[3], [5, 5], stride=2, scope='conv3', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
                                  normalizer_params={'scope': 'layer_norm5'})
      conv3 = tf.nn.dropout(conv3, keep_prob_dropout)

      #LAYER 6: convLSTM3
      hidden3, lstm_state3 = basic_conv_lstm_cell(conv3, lstm_state3, 32, initializer, filter_size=3, scope='convlstm3')
      hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm6')
      hidden3 = tf.nn.dropout(hidden3, keep_prob_dropout)

      #LAYER 7: conv4
      conv4 = slim.layers.conv2d(hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv4', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer,
                                 normalizer_params={'scope': 'layer_norm7'})
      conv4 = tf.nn.dropout(conv4, keep_prob_dropout)

      #LAYER 8: convLSTM4 (8x8 feature map size)
      hidden4, lstm_state4 = basic_conv_lstm_cell(conv4, lstm_state4, 64, initializer, filter_size=3, scope='convlstm4')
      hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm8')
      hidden4 = tf.nn.dropout(hidden4, keep_prob_dropout)

      #LAYER 8: conv5
      conv5 = slim.layers.conv2d(hidden4, hidden4.get_shape()[3], [3, 3], stride=2, scope='conv5', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer, 
                                 normalizer_params={'scope': 'layer_norm9'})
      conv5 = tf.nn.dropout(conv5, keep_prob_dropout)

      # LAYER 9: convLSTM5 (4x4 feature map size)
      hidden5, lstm_state5 = basic_conv_lstm_cell(conv5, lstm_state5, 64, initializer, filter_size=3, scope='convlstm5')
      hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm10')
      hidden5 = tf.nn.dropout(hidden5, keep_prob_dropout)

      # LAYER 10: Fully Convolutional Layer (4x4x128 --> 1x1xFC_LAYER_SIZE)
      # necessary for dimension compatibility with conv lstm cell
      fc_conv = slim.layers.conv2d(hidden5, FC_LAYER_SIZE, [4,4], stride=1, scope='fc_conv', padding='VALID', weights_initializer=initializer)
      fc_conv = tf.nn.dropout(fc_conv, keep_prob_dropout)

      # LAYER 11: Fully Convolutional LSTM (1x1x256 -> 1x1x128)
      hidden6, lstm_state6 = basic_conv_lstm_cell(fc_conv, lstm_state6, FC_LSTM_LAYER_SIZE, initializer, filter_size=1, scope='convlstm6')
      #no dropout since its the last encoder layer --> hidden repr should be steady

      # mu and sigma for sampling latent variable
      sigma = slim.layers.fully_connected(inputs=lstm_state6, num_outputs=VAE_REPR_SIZE, activation_fn=tf.nn.softplus)
      mu = slim.layers.fully_connected(inputs=lstm_state6, num_outputs=VAE_REPR_SIZE, activation_fn=None)

      # do reparamazerization trick to allow backprop flow through deterministic nodes sigma and mu
      z = mu + sigma * tf.random_normal(tf.shape(mu), mean=0., stddev=1.)

  return z, mu, sigma


def decoder_model(hidden_repr, sequence_length, initializer, num_channels=3, keep_prob_dropout=0.9, scope='decoder', fc_conv_layer=False):
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

      hidden0 = tf.nn.dropout(hidden_repr, keep_prob_dropout)

      fc_conv = slim.layers.conv2d_transpose(hidden0, 64, [4, 4], stride=1, scope='fc_conv', padding='VALID', weights_initializer=initializer)
      fc_conv = tf.nn.dropout(fc_conv, keep_prob_dropout)


      #LAYER 1: convLSTM1
      hidden1, lstm_state1 = basic_conv_lstm_cell(fc_conv, lstm_state1, 64, initializer, filter_size=3, scope='convlstm1')
      hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm1')
      hidden1 = tf.nn.dropout(hidden1, keep_prob_dropout)

      #LAYER 2: upconv1 (8x8 -> 16x16)
      upconv1 = slim.layers.conv2d_transpose(hidden1, hidden1.get_shape()[3], 3, stride=2, scope='upconv1', weights_initializer=initializer,
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm2'})
      upconv1 = tf.nn.dropout(upconv1, keep_prob_dropout)

      #LAYER 3: convLSTM2
      hidden2, lstm_state2 = basic_conv_lstm_cell(upconv1, lstm_state2, 64, initializer, filter_size=3, scope='convlstm2')
      hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')
      hidden2 = tf.nn.dropout(hidden2, keep_prob_dropout)

      #LAYER 4: upconv2 (16x16 -> 32x32)
      upconv2 = slim.layers.conv2d_transpose(hidden2, hidden2.get_shape()[3], 3, stride=2, scope='upconv2', weights_initializer=initializer,
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm4'})
      upconv2 = tf.nn.dropout(upconv2, keep_prob_dropout)

      #LAYER 5: convLSTM3
      hidden3, lstm_state3 = basic_conv_lstm_cell(upconv2, lstm_state3, 32, initializer, filter_size=3, scope='convlstm3')
      hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm5')
      hidden3 = tf.nn.dropout(hidden3, keep_prob_dropout)

      # LAYER 6: upconv3 (32x32 -> 64x64)
      upconv3 = slim.layers.conv2d_transpose(hidden3, hidden3.get_shape()[3], 5, stride=2, scope='upconv3', weights_initializer=initializer,
                                             normalizer_fn=tf_layers.layer_norm,
                                             normalizer_params={'scope': 'layer_norm6'})
      upconv3 = tf.nn.dropout(upconv3, keep_prob_dropout)

      #LAYER 7: convLSTM4
      hidden4, lstm_state4 = basic_conv_lstm_cell(upconv3, lstm_state4, 32, initializer, filter_size=5, scope='convlstm4')
      hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm7')
      hidden4 = tf.nn.dropout(hidden4, keep_prob_dropout)

      #Layer 8: upconv4 (64x64 -> 128x128)
      upconv4 = slim.layers.conv2d_transpose(hidden4, hidden4.get_shape()[3], 5, stride=2, scope='upconv4', normalizer_fn=tf_layers.layer_norm, weights_initializer=initializer, normalizer_params={'scope': 'layer_norm8'})
      upconv4 = tf.nn.dropout(upconv4, keep_prob_dropout)

      #LAYER 9: convLSTM5
      hidden5, lstm_state5 = basic_conv_lstm_cell(upconv4, lstm_state5, 32, initializer, filter_size=5, scope='convlstm5')
      hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm9')
      hidden5 = tf.nn.dropout(hidden5, keep_prob_dropout)

      upconv5 = slim.layers.conv2d_transpose(hidden5, num_channels, 5, stride=2, scope='upconv5', weights_initializer=initializer)
      # no dropout since this layer finally generates the output image

      frame_gen.append(upconv5)

  assert len(frame_gen)==sequence_length
  return frame_gen


def composite_model(frames, encoder_len=5, decoder_future_len=5, decoder_reconst_len=5, noise_std=0.0,
                    uniform_init=True, num_channels=3, keep_prob_dropout=0.9, fc_conv_layer=True):
  """
  Args:
    frames: 5D array of batch with videos - shape(batch_size, num_frames, frame_width, frame_higth, num_channels)
    encoder_len: number of frames that shall be encoded
    decoder_future_sequence_length: number of frames that shall be decoded from the hidden_repr
    noise_std: standard deviation of the gaussian noise to be added to the hidden representation
    uniform_init: specifies if the weight initialization should be drawn from gaussian or uniform distribution (default:uniform)
    num_channels: number of channels for generated frames
    fc_conv_layer: indicates whether fully connected layer shall be added between encoder and decoder
  Returns:
    frame_gen: array of generated frames (Tensors)
  """
  assert all([len > 0 for len in [encoder_len, decoder_future_len, decoder_reconst_len]])
  initializer = tf_layers.xavier_initializer(uniform=uniform_init)
  hidden_repr, mu, sigma = encoder_model(frames, encoder_len, initializer, fc_conv_layer=fc_conv_layer)

  # add noise
  if noise_std != 0.0:
    hidden_repr = hidden_repr + tf.random_normal(shape=hidden_repr.get_shape(), mean=0.0, stddev=noise_std,
                                                 dtype=tf.float32)

  frames_pred = decoder_model(hidden_repr, decoder_future_len, initializer, num_channels=num_channels, keep_prob_dropout=keep_prob_dropout,
                              scope='decoder_pred', fc_conv_layer=fc_conv_layer)
  frames_reconst = decoder_model(hidden_repr, decoder_reconst_len, initializer, num_channels=num_channels, keep_prob_dropout=keep_prob_dropout,
                                 scope='decoder_reconst', fc_conv_layer=fc_conv_layer)
  return frames_pred, frames_reconst, [hidden_repr, mu, sigma]
