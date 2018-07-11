import tensorflow as tf
import math
from pprint import pprint
from settings import FLAGS, model

from models import loss_functions
import data_prep.model_input as input


class Model:
  def __init__(self):

    self.learning_rate = tf.placeholder_with_default(FLAGS.learning_rate, ())
    self.iter_num = tf.placeholder_with_default(FLAGS.num_iterations, ())
    self.summaries = []
    self.noise_std = tf.placeholder_with_default(FLAGS.noise_std, ())
    self.opt = tf.train.AdamOptimizer(self.learning_rate)
    self.model_name = model.__file__

    assert FLAGS.image_range_start + FLAGS.encoder_length + FLAGS.decoder_future_length <= FLAGS.overall_images_count and FLAGS.image_range_start >= 0, \
            "settings for encoder/decoder lengths along with starting range exceed number of available images"
    assert FLAGS.encoder_length >= FLAGS.decoder_reconst_length, "encoder must be at least as long as reconstructer"

  def add_image_summary(self, summary_prefix, frames, encoder_length, decoder_future_length, decoder_reconst_length):
    for i in range(decoder_future_length):
      self.summaries.append(tf.summary.image(summary_prefix + '_future_gen_' + str(i + 1),
                                             self.frames_pred[i], max_outputs=1))
      self.summaries.append(tf.summary.image(summary_prefix + '_future_orig_' + str(i + 1),
                                             frames[:, encoder_length + i, :, :, :], max_outputs=1))
    for i in range(decoder_reconst_length):
      self.summaries.append(tf.summary.image(summary_prefix + '_reconst_gen_' + str(i + 1),
                                             self.frames_reconst[i], max_outputs=1))
      self.summaries.append(tf.summary.image(summary_prefix + '_reconst_orig_' + str(i + 1),
                                             frames[:, i, :, :, :], max_outputs=1))


class TrainModel(Model):
  def __init__(self, summary_prefix, scope_name='train_model'):
    print("Constructing TrainModel")

    with tf.variable_scope(scope_name, reuse=None) as training_scope:
      Model.__init__(self)
      self.scope = training_scope

      tower_grads = []
      tower_losses = []
      for i in range(FLAGS.num_gpus):
        train_batch, _, _ = input.create_batch(FLAGS.tf_records_dir, 'train', FLAGS.batch_size,
                                               int(math.ceil(
                                                 FLAGS.num_iterations / (FLAGS.batch_size * 20))),
                                               False)
        train_batch = tf.cast(train_batch, tf.float32)
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('tower', i)):
            tower_loss, _, _, _ = tower_operations(train_batch[:,FLAGS.image_range_start:,:,:,:], train=True)
            tower_losses.append(tower_loss)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            if FLAGS.fine_tuning_weights_list is not None:
              train_vars = []
              for scope_i in FLAGS.fine_tuning_weights_list:
                train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_i)
              pprint('Finetuning. Training only specified weights: %s' % (FLAGS.fine_tuning_weights_list))
              grads = self.opt.compute_gradients(tower_loss, var_list=train_vars)
            else:
              grads = self.opt.compute_gradients(tower_loss)
            tower_grads.append(grads)

      with tf.device('/cpu:0'):
        #copmute average loss
        self.loss = average_losses(tower_losses)

        #compute average over gradients of all towers
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        self.train_op= self.opt.apply_gradients(grads)

      #measure batch time
      self.elapsed_time = tf.placeholder(tf.float32, [])
      self.summaries.append(tf.summary.scalar('batch_duration', self.elapsed_time))

      self.summaries.append(tf.summary.scalar(summary_prefix + '_loss', self.loss))
      self.sum_op = tf.summary.merge(self.summaries)


class ValidationModel(Model):
  def __init__(self, summary_prefix, scope_name='valid_model', reuse_scope=None):
    print("Constructing ValidationModel")

    with tf.variable_scope(scope_name, reuse=None):
      Model.__init__(self)

      assert reuse_scope is not None

      with tf.variable_scope(reuse_scope, reuse=True):
        tower_losses, frames_pred_list, frames_reconst_list, hidden_repr_list, label_batch_list, metadata_batch_list, val_batch_list = [], [], [], [], [], [], []

        for i in range(FLAGS.num_gpus):
          val_batch, label_batch, metadata_batch = input.create_batch(FLAGS.tf_records_dir, 'valid', FLAGS.valid_batch_size, int(
            math.ceil(FLAGS.num_iterations / (FLAGS.batch_size * 20))), False)

          val_batch = tf.cast(val_batch, tf.float32)
          self.val_batch = val_batch

          with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ('tower', i)):
              tower_loss, frames_pred, frames_reconst, hidden_repr = tower_operations(
                val_batch[:, FLAGS.image_range_start:, :, :, :], train=False)
              tower_losses.append(tower_loss)

              frames_pred_list.append(tf.pack(frames_pred))
              frames_reconst_list.append(tf.pack(frames_reconst))
              hidden_repr_list.append(hidden_repr)

              val_batch_list.append(val_batch)
              label_batch_list.append(label_batch)
              metadata_batch_list.append(metadata_batch)
              # Reuse variables for the next tower.
              tf.get_variable_scope().reuse_variables()

        with tf.device('/cpu:0'):
          # compute average loss
          self.loss = average_losses(tower_losses)
          # concatenate outputs of towers to one large tensor each
          self.frames_pred = tf.unstack(tf.concat(1, frames_pred_list))
          self.frames_reconst = tf.unstack(tf.concat(1, frames_reconst_list))
          self.hidden_repr = tf.concat(0, hidden_repr_list)
          self.label = tf.concat(0, label_batch_list)
          self.metadata = tf.concat(0, metadata_batch_list)
          val_set = tf.concat(0, val_batch_list)

      self.add_image_summary(summary_prefix, val_set, FLAGS.encoder_length, FLAGS.decoder_future_length,
                             FLAGS.decoder_reconst_length)

      # evaluate frame predictions for storing on disk
      self.output_frames = self.frames_reconst + self.frames_pred  # join arrays of tensors


      self.summaries.append(tf.summary.scalar(summary_prefix + '_loss', self.loss))
      self.sum_op = tf.summary.merge(self.summaries)


class FeedingValidationModel(Model):
  def __init__(self, scope_name='feeding_model', reuse_scope=None):
    print("Constructing FeedingModel")
    with tf.variable_scope(scope_name, reuse=None):
      Model.__init__(self)

      assert reuse_scope is not None

      with tf.variable_scope(reuse_scope, reuse=True):
        "5D array of batch with videos - shape(batch_size, num_frames, frame_width, frame_higth, num_channels)"
        self.feed_batch = tf.placeholder(tf.float32, shape=(1, FLAGS.encoder_length, FLAGS.height, FLAGS.width, FLAGS.num_channels), name='feed_batch')

        self.frames_pred, self.frames_reconst, self.hidden_repr = \
          tower_operations(self.feed_batch[:, FLAGS.image_range_start:, :, :, :], train=False, compute_loss=False)



def tower_operations(video_batch, train=True, compute_loss=True):
  """
  Build the computation graph from input frame sequences till loss of batch
  :param device number for assining queue runner to CPU
  :param train: boolean that indicates whether train or validation mode
  :param compute_loss: boolean that specifies whether loss should be computed (for feed mode / production compute_loss might be disabled)
  :return batch loss (scalar)
  """
  #only dropout in train mode
  keep_prob_dropout = FLAGS.keep_prob_dopout if train else 1.0

  frames_pred, frames_reconst, hidden_repr = model.composite_model(video_batch, FLAGS.encoder_length,
                                                                   FLAGS.decoder_future_length,
                                                                   FLAGS.decoder_reconst_length,
                                                                   keep_prob_dropout=keep_prob_dropout,
                                                                   noise_std=FLAGS.noise_std,
                                                                   uniform_init=FLAGS.uniform_init,
                                                                   num_channels=FLAGS.num_channels,
                                                                   fc_conv_layer=FLAGS.fc_layer)

  if compute_loss:
    tower_loss = loss_functions.composite_loss(video_batch, frames_pred, frames_reconst, loss_fun=FLAGS.loss_function,
                                  encoder_length=FLAGS.encoder_length,
                                  decoder_future_length=FLAGS.decoder_future_length,
                                  decoder_reconst_length=FLAGS.decoder_reconst_length,
                                  hidden_repr=hidden_repr)
    return tower_loss, frames_pred, frames_reconst, hidden_repr
  else:
    return frames_pred, frames_reconst, hidden_repr


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """

  average_grads = []

  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def average_losses(tower_losses):
  """Calculate the average loss among all towers
  Args:
    tower_losses: List of tf.Tensor skalars denoting the loss at each tower.
  Returns:
     loss: tf.Tensor skalar which is the mean over all losses
  """
  losses = []
  for l in tower_losses:
    # Add 0 dimension to the gradients to represent the tower.
    expanded_l = tf.expand_dims(l, 0)

    # Append on a 'tower' dimension which we will average over below.
    losses.append(expanded_l)

  # Average over the 'tower' dimension.
  loss = tf.concat(0, losses)
  loss = tf.reduce_mean(loss, 0)
  return loss


def create_model(mode=None, train_model_scope=None):
  model = None

  if mode is "train":
    model = TrainModel('train', scope_name='train_model')
  elif mode is 'valid':
    assert train_model_scope is not None, "train_model_scope is None, valid mode requires a train scope"
    model = ValidationModel('valid', reuse_scope=train_model_scope)
  elif mode is 'feeding':
    assert train_model_scope is not None, "train_model_scope is None, valid mode requires a train scope"
    model = FeedingValidationModel(reuse_scope=train_model_scope)

  assert model is not None

  return model