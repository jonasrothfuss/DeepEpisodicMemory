import tensorflow as tf
import numpy as np
from models import model
import math
import data_prep.model_input as input
import os
import datetime as dt
import moviepy.editor as mpy


from tensorflow.python.platform import app
from tensorflow.python.platform import flags

tf.logging.set_verbosity(tf.logging.INFO)

LOSS_FUNCTIONS = ['mse', 'gdl']

# constants for developing
FLAGS = flags.FLAGS
#DATA_PATH = '/home/jonasrothfuss/Dropbox/Deep_Learning_for_Object_Manipulation/4_Data/Datasets/ArtificialFlyingBlobs'
#LOG_PATH = '/home/jonasrothfuss/Desktop/'
#OUT_DIR = '/home/jonasrothfuss/Desktop/'
#DATA_PATH = '/localhome/rothfuss/data/ArtificialFlyingBlobs/tfrecords/'
#OUT_DIR = '/localhome/rothfuss/training/'
DATA_PATH = '/Users/fabioferreira/Dropbox/Deep_Learning_for_Object_Manipulation/4_Data/Datasets/ArtificialFlyingBlobs'
OUT_DIR = '/Users/fabioferreira/Dropbox/Deep_Learning_for_Object_Manipulation/4_Data/Datasets/ArtificialFlyingBlobs/out_dir'


# use pretrained model
PRETRAINED_MODEL = '' #/localhome/rothfuss/training/02-20-17_23-21'

# use pre-trained model and run validation only
VALID_ONLY = False


# hyperparameters
flags.DEFINE_integer('num_iterations', 1000000, 'specify number of training iterations, defaults to 100000')
flags.DEFINE_integer('learning_rate', 0.0001, 'learning rate for Adam optimizer')
flags.DEFINE_string('loss_function', 'mse', 'specify loss function to minimize, defaults to gdl')
flags.DEFINE_string('batch_size', 50, 'specify the batch size, defaults to 50')

flags.DEFINE_string('encoder_length', 5, 'specifies how many images the encoder receives, defaults to 5')
flags.DEFINE_string('decoder_future_length', 5, 'specifies how many images the future prediction decoder receives, defaults to 5')
flags.DEFINE_string('decoder_reconst_length', 5, 'specifies how many images the reconstruction decoder receives, defaults to 5')

#IO specifications
flags.DEFINE_string('path', DATA_PATH, 'specify the path to where tfrecords are stored, defaults to "../data/"')
flags.DEFINE_integer('num_channels', 3, 'number of channels in the input frames')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('pretrained_model', PRETRAINED_MODEL, 'filepath of a pretrained model to initialize from.')
flags.DEFINE_string('valid', VALID_ONLY, 'Set to "True" if you want to validate a pretrained model only (no training involved). Defaults to False.')

# intervals
flags.DEFINE_integer('valid_interval', 500, 'number of training steps between each validation')
flags.DEFINE_integer('summary_interval', 100, 'number of training steps between summary is stored')
flags.DEFINE_integer('save_interval', 1, 'number of training steps between session/model dumps')

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
                   encoder_length=FLAGS.encoder_length, decoder_future_length=FLAGS.decoder_future_length,
                   decoder_reconst_length=FLAGS.decoder_reconst_length):

  assert encoder_length <= decoder_reconst_length
  assert loss_fun in LOSS_FUNCTIONS
  frames_original_future = original_frames[:, (encoder_length):(encoder_length + decoder_future_length), :, :, :]
  frames_original_reconst = original_frames[:, (encoder_length - decoder_reconst_length):encoder_length, :, :, :]
  pred_loss = decoder_loss(frames_pred, frames_original_future, loss_fun)
  reconst_loss = decoder_loss(frames_reconst, frames_original_reconst, loss_fun)
  return pred_loss + reconst_loss

class Model:

  def __init__(self,
               frames,
               summary_prefix,
               encoder_length=FLAGS.encoder_length,
               decoder_future_length=FLAGS.decoder_future_length,
               decoder_reconst_length=FLAGS.decoder_reconst_length,
               loss_fun=FLAGS.loss_function,
               reuse_scope=None,
               out_dir = None):

    #self.output_dir = out_dir  # for validation only
    self.learning_rate = tf.placeholder_with_default(FLAGS.learning_rate, ())
    #self.prefix = tf.placeholder(tf.string, []) #string for summary that denotes whether train or val
    self.iter_num = tf.placeholder(tf.int32, [])
    self.summaries = []
    self.output_dir = out_dir

    if reuse_scope is None: #train model
      frames_pred, frames_reconst = model.composite_model(frames, encoder_length,
                                                          decoder_future_length,
                                                          decoder_reconst_length,
                                                          num_channels=FLAGS.num_channels)
    else: # -> validation or test model
      with tf.variable_scope(reuse_scope, reuse=True):
        frames_pred, frames_reconst = model.composite_model(frames, encoder_length,
                                                            decoder_future_length,
                                                            decoder_reconst_length,
                                                            num_channels=FLAGS.num_channels)

    self.frames_pred = frames_pred
    self.frames_reconst = frames_reconst
    self.loss = composite_loss(frames, frames_pred, frames_reconst, loss_fun=loss_fun)
    self.summaries.append(tf.summary.scalar(summary_prefix + '_loss', self.loss))

    if reuse_scope: # only image summary if validation or test model
      self.add_image_summary(summary_prefix, frames, encoder_length, decoder_future_length, decoder_reconst_length) #TODO: add more summaries

    if reuse_scope and FLAGS.valid:
      self.export_image_summary(summary_prefix, frames, encoder_length, decoder_future_length, decoder_reconst_length)

    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    self.sum_op = tf.summary.merge(self.summaries)


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


  def export_image_summary(self, summary_prefix, frames, encoder_length, decoder_future_length, decoder_reconst_length):
    for i in range(decoder_future_length):
      clip = mpy.ImageSequenceClip(list(self.frames_pred), fps=10)
      clip.write_gif(self.output_dir + 'decoder_future_' + str(self.iter_num) + '.gif')
      #TODO: not tested

class Initializer:

  def __init__(self, out_dir=None):
    self.status = False
    self.sess = None
    self.threads = None
    self.coord = None
    self.saver = None

  def start_session(self):
    """Starts a session and initializes all variables. Provides access to session and coordinator"""
    # Start Session and initialize variables
    self.status = True
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    self.sess = sess
    sess.run(init_op)


    # Start input enqueue threads
    coord = tf.train.Coordinator()
    self.coord = coord
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    self.threads = threads

  def stop_session(self):
    """Stops a current session."""
    if self.sess and self.coord:
      self.coord.join(self.threads)
      self.sess.close()
      self.status = False

  def start_saver(self):
    """Constructs a saver and if pretrained model given, loads the model."""
    print('Constructing saver')
    self.saver = tf.train.Saver(max_to_keep=0)

    # restore dumped model if provided
    if FLAGS.pretrained_model:
      print('Restore model from: ' + str(FLAGS.pretrained_model))
      self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.pretrained_model))

    return self.saver


def create_session_dir():
  assert(FLAGS.output_dir)
  dir_name = str(dt.datetime.now().strftime("%m-%d-%y_%H-%M"))
  output_dir = os.path.join(FLAGS.output_dir, dir_name)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  print('Created custom directory for session:', dir_name)
  return output_dir

def create_subfolder(dir, name):
  if os.path.isdir(dir + name):
    os.mkdir(dir)
    print('Created subdir', name)
    return dir
  else:
    return None


#TODO implement validation run
def valid_run(output_dir):
  val_model = create_model()

  initializer = Initializer(output_dir)
  initializer.start_session()
  initializer.start_saver()

  summary_writer = tf.summary.FileWriter(output_dir, graph=initializer.sess.graph, flush_secs=10)

  try:
    for itr in range (FLAGS.num_iterations):
      if initializer.coord.should_stop():
        break

      feed_dict = {val_model.learning_rate: 0.0}

      # summary and log
      val_loss, val_summary_str = initializer.sess.run([val_model.loss, val_model.sum_op], feed_dict)
      val_model.iter_num = itr

      summary_writer.add_summary(val_summary_str, itr)

      #TODO:

  except tf.errors.OutOfRangeError:
    tf.logging.info('Done producing validation results -- iterations limit reached')
  finally:
    # When done, ask the threads to stop.
    initializer.coord.request_stop()

  # Wait for threads to finish.
  initializer.stop_session()

def create_model():
  if FLAGS.valid:
    print('Constructing validation model and input')
    with tf.variable_scope('val_model', reuse=None):
      val_set = input.create_batch(FLAGS.path, 'valid', 1000, int(math.ceil(FLAGS.num_iterations/FLAGS.valid_interval)+10))
      val_set = tf.cast(val_set, tf.float32)
      val_model = Model(val_set, 'valid')
    return val_model

  else:
    print('Constructing train model and input')
    with tf.variable_scope('train_model', reuse=None) as training_scope:
      train_batch = input.create_batch(FLAGS.path, 'train', FLAGS.batch_size, int(math.ceil(FLAGS.num_iterations/(FLAGS.batch_size * 20))))
      train_batch = tf.cast(train_batch, tf.float32)
      train_model = Model(train_batch, 'train')

    print('Constructing validation model and input')
    with tf.variable_scope('val_model', reuse=None):
      val_set = input.create_batch(FLAGS.path, 'valid', 1000, int(math.ceil(FLAGS.num_iterations/FLAGS.valid_interval)+10))
      val_set = tf.cast(val_set, tf.float32)
      val_model = Model(val_set, 'valid', reuse_scope=training_scope)

    return train_model, val_model


def train_valid_run(output_dir):
  train_model, val_model = create_model()


  initializer = Initializer()
  initializer.start_session()

  saver = initializer.start_saver()


  summary_writer = tf.summary.FileWriter(output_dir, graph=initializer.sess.graph, flush_secs=10)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info(' --- Start Training --- ')
  tf.logging.info(' Iteration, Train_Loss ')


  ''' main training loop '''
  try:
    for itr in range(FLAGS.num_iterations):
      if initializer.coord.should_stop():
        break

      #Training Step on batch
      feed_dict = {train_model.learning_rate: FLAGS.learning_rate}
      train_loss, _, train_summary_str = initializer.sess.run([train_model.loss, train_model.train_op, train_model.sum_op], feed_dict)
      #Print Interation and loss
      tf.logging.info(' ' + str(itr) + ':    ' + str(train_loss))

      #validation
      if itr % FLAGS.valid_interval == 1:
        feed_dict = {val_model.learning_rate: 0.0}

        # summary and log
        val_loss, val_summary_str = initializer.sess.run([val_model.loss, val_model.sum_op], feed_dict)

        summary_writer.add_summary(val_summary_str, itr)
        #Print validation loss
        tf.logging.info(' Validation loss at step ' + str(itr) + ':    ' + str(val_loss))

      #dump summary
      if itr % FLAGS.summary_interval == 1:
        summary_writer.add_summary(train_summary_str, itr)

      #save model checkpoint
      if itr % FLAGS.save_interval == 1:
        save_path = saver.save(initializer.sess, os.path.join(output_dir, 'model'), global_step=itr) #TODO also implement save operation in Initializer class
        tf.logging.info(' Saved Model to: ' + str(save_path))

  except tf.errors.OutOfRangeError:
    tf.logging.info('Done training -- iterations limit reached')
  finally:
    # When done, ask the threads to stop.
    initializer.coord.request_stop()

  tf.logging.info(' Saving Model ... ')
  saver.save(initializer.sess, output_dir + '/model')

  # Wait for threads to finish.
  initializer.stop_session()


def main(unused_argv):

  # run validation only
  if FLAGS.valid:
    assert FLAGS.pretrained_model
    output_dir = FLAGS.pretrained_model
    tf.logging.info(' --- VALIDATION MODE ONLY --- ')
    print('Reusing provided session directory:', output_dir)
    subdir = create_subfolder(output_dir, 'valid_run')
    print('Storing validation data in:', output_dir)
    valid_run(subdir)

  # run training + validation
  else:
    if not FLAGS.pretrained_model:
      # create new session directory
      output_dir = create_session_dir()
    else:
      output_dir = FLAGS.pretrained_model
      print('Reusing provided session directory:', output_dir)

    tf.logging.info(' --- TRAIN+VALID MODE --- ')
    train_valid_run(output_dir)


if __name__ == '__main__':
  app.run()


