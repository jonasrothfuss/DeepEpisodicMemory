import tensorflow as tf
import numpy as np
from models import model
from models import loss_functions
import math
import data_prep.model_input as input
import os
import datetime as dt
import moviepy.editor as mpy
import re
from datetime import datetime


from tensorflow.python.platform import app
from tensorflow.python.platform import flags

tf.logging.set_verbosity(tf.logging.INFO)

LOSS_FUNCTIONS = ['mse', 'gdl']

# constants for developing
FLAGS = flags.FLAGS
DATA_PATH = '/localhome/rothfuss/data/ArtificialFlyingBlobs/tfrecords/'
OUT_DIR = '/localhome/rothfuss/training/'
#DATA_PATH = '/Users/fabioferreira/Dropbox/Deep_Learning_for_Object_Manipulation/4_Data/Datasets/ArtificialFlyingBlobs'
#OUT_DIR = '/Users/fabioferreira/Dropbox/Deep_Learning_for_Object_Manipulation/4_Data/Datasets/ArtificialFlyingBlobs/out_dir'


# use pretrained model
PRETRAINED_MODEL = '' #''/localhome/rothfuss/training/02-28-17_18-25'

# use pre-trained model and run validation only
VALID_ONLY = False


# hyperparameters
flags.DEFINE_integer('num_iterations', 1000000, 'specify number of training iterations, defaults to 100000')
flags.DEFINE_integer('learning_rate', 0.001, 'learning rate for Adam optimizer')
flags.DEFINE_string('loss_function', 'mse', 'specify loss function to minimize, defaults to gdl')
flags.DEFINE_string('batch_size', 50, 'specify the batch size, defaults to 50')

flags.DEFINE_string('encoder_length', 5, 'specifies how many images the encoder receives, defaults to 5')
flags.DEFINE_string('decoder_future_length', 5, 'specifies how many images the future prediction decoder receives, defaults to 5')
flags.DEFINE_string('decoder_reconst_length', 5, 'specifies how many images the reconstruction decoder receives, defaults to 5')
flags.DEFINE_bool('fc_layer', False, 'indicates whether fully connected layer shall be added between encoder and decoder')

#IO specifications
flags.DEFINE_string('path', DATA_PATH, 'specify the path to where tfrecords are stored, defaults to "../data/"')
flags.DEFINE_integer('num_channels', 3, 'number of channels in the input frames')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('pretrained_model', PRETRAINED_MODEL, 'filepath of a pretrained model to initialize from.')
flags.DEFINE_string('valid', VALID_ONLY, 'Set to "True" if you want to validate a pretrained model only (no training involved). Defaults to False.')

# intervals
flags.DEFINE_integer('valid_interval', 500, 'number of training steps between each validation')
flags.DEFINE_integer('summary_interval', 100, 'number of training steps between summary is stored')
flags.DEFINE_integer('save_interval', 2000, 'number of training steps between session/model dumps')


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
                                                          num_channels=FLAGS.num_channels,
                                                          fc_conv_layer=FLAGS.fc_layer)
    else: # -> validation or test model
      with tf.variable_scope(reuse_scope, reuse=True):
        frames_pred, frames_reconst = model.composite_model(frames, encoder_length,
                                                            decoder_future_length,
                                                            decoder_reconst_length,
                                                            num_channels=FLAGS.num_channels,
                                                            fc_conv_layer=FLAGS.fc_layer)

    self.frames_pred = frames_pred
    self.frames_reconst = frames_reconst
    self.loss = loss_functions.composite_loss(frames, frames_pred, frames_reconst, loss_fun=loss_fun,
                                        encoder_length=FLAGS.encoder_length,
                                        decoder_future_length=FLAGS.decoder_future_length,
                                        decoder_reconst_length=FLAGS.decoder_future_length)
    self.summaries.append(tf.summary.scalar(summary_prefix + '_loss', self.loss))

    if reuse_scope: # only image summary if validation or test model
      self.add_image_summary(summary_prefix, frames, encoder_length, decoder_future_length, decoder_reconst_length) #TODO: add more summaries

    if reuse_scope and FLAGS.valid: # only valid mode - evaluate frame predictions for storage on disk
      self.output_frames = self.frames_reconst + self.frames_pred #join arrays of tensors

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

class Initializer:

  def __init__(self, out_dir=None):
    self.status = False
    self.sess = None
    self.threads = None
    self.coord = None
    self.saver = None
    self.itr_start = 0

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
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.pretrained_model)
      self.itr_start = get_iter_from_pretrained_model(latest_checkpoint) + 1
      print('Start with iteration: ' + str(self.itr_start))
      self.saver.restore(self.sess, latest_checkpoint)

    return self.saver

def get_iter_from_pretrained_model(checkpoint_file_name):
  ''' extracts the iterator count of a dumped checkpoint from the checkpoint file name
  :param checkpoint_file_name: name of checkpoint file - must contain a
  :return: iterator number
  '''
  file_basename = os.path.basename(checkpoint_file_name)
  assert re.compile('[A-Za-z0-9]+[-][0-9]+').match(file_basename)
  idx = re.findall(r'-\b\d+\b', file_basename)[0][1:]
  return int(idx)

def create_session_dir(): #TODO move to utils
  assert(FLAGS.output_dir)
  dir_name = str(dt.datetime.now().strftime("%m-%d-%y_%H-%M"))
  output_dir = os.path.join(FLAGS.output_dir, dir_name)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  print('Created custom directory for session:', dir_name)
  return output_dir

def create_subfolder(dir, name): #TODO move to utils
  subdir = os.path.join(dir, name)
  if not os.path.isdir(subdir):
    os.mkdir(subdir)
    print('Created subdir:', subdir)
    return subdir
  else:
    return os.path.join(dir, name)

def create_model():
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
    for itr in range(initializer.itr_start, initializer.itr_start + FLAGS.num_iterations):
      if initializer.coord.should_stop():
        break

      #Training Step on batch
      feed_dict = {train_model.learning_rate: FLAGS.learning_rate}
      train_loss, _, train_summary_str = initializer.sess.run([train_model.loss, train_model.train_op, train_model.sum_op], feed_dict)
      #Print Interation and loss
      tf.logging.info(' ' + str(itr) + ':    ' + str(train_loss))

      #validation
      if itr % FLAGS.valid_interval == 1:

        with open(os.path.join(output_dir, 'timestamps.log'), 'a') as f:  # TODO delete
          f.write(str(itr) + ': ' + str(datetime.now()) + '\n')

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

def store_output_frames_as_gif(output_frames, output_dir):
  """ Stores frame sequence produced by model as gif
    Args:
      output_frames:  list with Tensors of shape [batch_size, frame_height, frame_width, num_channels],
                      each element corresponds to one frame in the produced gifs
      output_dir:     path to output directory
  """
  assert os.path.isdir(output_dir)
  batch_size = output_frames[0].shape[0]
  for i in range(batch_size): #iterate over validation instances
    clip_array = [frame[i,:,:,:] for frame in output_frames]
    print(type(clip_array))
    print(type(clip_array[0]))
    print(clip_array[0].shape)
    clip = mpy.ImageSequenceClip(clip_array, fps=10)
    clip.to_gif(os.path.join(output_dir, 'generated_clip_' + str(i) + '.gif'))

def valid_run(output_dir):
  """ feeds validation batch through the model and stores produced frame sequence as gifs to output_dir
    :param
      output_dir: path to output directory where validation summary and gifs are stored
  """
  _, val_model = create_model()

  initializer = Initializer(output_dir)
  initializer.start_session()
  initializer.start_saver()

  summary_writer = tf.summary.FileWriter(output_dir, graph=initializer.sess.graph, flush_secs=10)

  tf.logging.info(' --- Start Validation --- ')

  try:
    feed_dict = {val_model.learning_rate: 0.0}

    # summary and log
    val_loss, val_summary_str, output_frames = initializer.sess.run([val_model.loss, val_model.sum_op, val_model.output_frames], feed_dict)
    val_model.iter_num = 1

    tf.logging.info('Converting validation frame sequences to gif')
    store_output_frames_as_gif(output_frames, output_dir)
    tf.logging.info('Dumped validation gifs in: ' + str(output_dir))

    summary_writer.add_summary(val_summary_str, 1)

  except tf.errors.OutOfRangeError:
    tf.logging.info('Done producing validation results -- iterations limit reached')
  finally:
    # When done, ask the threads to stop.
    initializer.coord.request_stop()

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
    print('Storing validation data in:', subdir)
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

