import tensorflow as tf
import math
import data_prep.model_input as input
import data_postp.similarity_computations as similarity_computations
import os
import time

from utils.helpers import get_iter_from_pretrained_model, learning_rate_decay
from utils.io_handler import create_session_dir, create_subfolder, store_output_frames_as_gif, write_metainfo, store_latent_vectors_as_df, store_encoder_latent_vector



from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from models import loss_functions

""" Set Model From Model Zoo"""
from models.model_zoo import model_conv5_fc_lstm_256 as model
""""""


LOSS_FUNCTIONS = ['mse', 'gdl', 'mse_gdl']

# constants for developing
FLAGS = flags.FLAGS
OUT_DIR = '/common/homes/students/rothfuss/Documents/training'
DATA_PATH = '/localhome/rothfuss/data/ucf101/tf_records'
#OUT_DIR = '/home/ubuntu/training'
#DATA_PATH = '/home/ubuntu/Dropbox-Uploader/tf_records_activity_net'


# use pretrained model
PRETRAINED_MODEL = ''
# use pre-trained model and run validation only
VALID_ONLY = False
VALID_MODE = 'data_frame' # 'vector', 'gif', 'similarity', 'data_frame'


# hyperparameters
flags.DEFINE_integer('num_iterations', 1000000, 'specify number of training iterations, defaults to 100000')
flags.DEFINE_string('loss_function', 'mse', 'specify loss function to minimize, defaults to gdl')
flags.DEFINE_string('batch_size', 30, 'specify the batch size, defaults to 50')
flags.DEFINE_integer('valid_batch_size', 200, 'specify the validation batch size, defaults to 50')
flags.DEFINE_bool('uniform_init', False, 'specifies if the weights should be drawn from gaussian(false) or uniform(true) distribution')
flags.DEFINE_integer('num_gpus', 8, 'specifies the number of available GPUs of the machine')

flags.DEFINE_string('encoder_length', 5, 'specifies how many images the encoder receives, defaults to 5')
flags.DEFINE_string('decoder_future_length', 5, 'specifies how many images the future prediction decoder receives, defaults to 5')
flags.DEFINE_string('decoder_reconst_length', 5, 'specifies how many images the reconstruction decoder receives, defaults to 5')
flags.DEFINE_bool('fc_layer', True, 'indicates whether fully connected layer shall be added between encoder and decoder')
flags.DEFINE_float('learning_rate_decay', 0.000008, 'learning rate decay factor')
flags.DEFINE_integer('learning_rate', 0.0005, 'initial learning rate for Adam optimizer')
flags.DEFINE_float('noise_std', 0.0, 'defines standard deviation of gaussian noise to be added to the hidden representation during training')

#IO specifications
flags.DEFINE_string('path', DATA_PATH, 'specify the path to where tfrecords are stored, defaults to "../data/"')
flags.DEFINE_integer('num_channels', 3, 'number of channels in the input frames')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('pretrained_model', PRETRAINED_MODEL, 'filepath of a pretrained model to initialize from.')
flags.DEFINE_string('valid_only', VALID_ONLY, 'Set to "True" if you want to validate a pretrained model only (no training involved). Defaults to False.')
flags.DEFINE_string('valid_mode', VALID_MODE, 'When set to '
                                              '"vector": encoder latent vector for each validation is exported to "output_dir" (only when VALID_ONLY=True) '
                                              '"gif": gifs are generated from the videos'
                                              '"similarity": compute (cos) similarity matrix')

# intervals
flags.DEFINE_integer('valid_interval', 500, 'number of training steps between each validation')
flags.DEFINE_integer('summary_interval', 100, 'number of training steps between summary is stored')
flags.DEFINE_integer('save_interval', 2000, 'number of training steps between session/model dumps')


class Model:

  def __init__(self,
               frames,
               video_id,
               summary_prefix,
               encoder_length=FLAGS.encoder_length,
               decoder_future_length=FLAGS.decoder_future_length,
               decoder_reconst_length=FLAGS.decoder_reconst_length,
               loss_fun=FLAGS.loss_function,
               reuse_scope=None,
               out_dir=None,
               metadata=None
               ):

    self.learning_rate = tf.placeholder_with_default(FLAGS.learning_rate, ())
    self.iter_num = tf.placeholder(tf.int32, [])
    self.summaries = []
    self.output_dir = out_dir
    self.label = video_id
    self.metadata = metadata
    self.noise_std = tf.placeholder_with_default(FLAGS.noise_std, ())


    if reuse_scope is None: #train model
      if FLAGS.noise_std > 0.0: #noise mode
        frames_pred, frames_reconst, hidden_repr = model.composite_model(frames, encoder_length,
                                                            decoder_future_length,
                                                            decoder_reconst_length,
                                                            noise_std=self.noise_std,
                                                            uniform_init=FLAGS.uniform_init,
                                                            num_channels=FLAGS.num_channels,
                                                            fc_conv_layer=FLAGS.fc_layer)
      else: #no noise
        frames_pred, frames_reconst, hidden_repr = model.composite_model(frames, encoder_length,
                                                                        decoder_future_length,
                                                                        decoder_reconst_length,
                                                                        uniform_init=FLAGS.uniform_init,
                                                                        num_channels=FLAGS.num_channels,
                                                                        fc_conv_layer=FLAGS.fc_layer)
    else: # -> validation or test model
      with tf.variable_scope(reuse_scope, reuse=True):
        frames_pred, frames_reconst, hidden_repr = model.composite_model(frames, encoder_length,
                                                            decoder_future_length,
                                                            decoder_reconst_length,
                                                            uniform_init=FLAGS.uniform_init,
                                                            num_channels=FLAGS.num_channels,
                                                            fc_conv_layer=FLAGS.fc_layer)

    self.frames_pred = frames_pred
    self.frames_reconst = frames_reconst
    self.hidden_repr = hidden_repr
    self.loss = loss_functions.composite_loss(frames, frames_pred, frames_reconst, loss_fun=loss_fun,
                                        encoder_length=FLAGS.encoder_length,
                                        decoder_future_length=FLAGS.decoder_future_length,
                                        decoder_reconst_length=FLAGS.decoder_reconst_length)

    self.summaries.append(tf.summary.scalar(summary_prefix + '_loss', self.loss)) # TODO: add video_id to summary

    if reuse_scope is None: #only measure time DURING training step
      self.elapsed_time = tf.placeholder(tf.float32, [])
      self.summaries.append(tf.summary.scalar('batch_duration', self.elapsed_time))

    if reuse_scope: # only image summary if validation or test model
      self.add_image_summary(summary_prefix, frames, encoder_length, decoder_future_length, decoder_reconst_length) #TODO: add more summaries

    if reuse_scope and FLAGS.valid_only: # only valid mode - evaluate frame predictions for storage on disk
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

    # limit gpu RAM occupation
    config = tf.ConfigProto(
      #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
      #gpu_options = tf.GPUOptions(allow_growth=True)
    )

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    self.sess = tf.Session(config=config)
    self.sess.run(init_op)

    # Start input enqueue threads
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

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

def create_model():
  print('Constructing train model and input')
  with tf.variable_scope('train_model', reuse=None) as training_scope:
    # TODO: convert video_id's to utf 8 string (e.g. b'002328 to 002328)
    train_batch, video_id_batch, _ = input.create_batch(FLAGS.path, 'train', FLAGS.batch_size,
                                                        int(math.ceil(FLAGS.num_iterations/(FLAGS.batch_size * 20))),
                                                        False)
    train_batch = tf.cast(train_batch, tf.float32)
    train_model = Model(train_batch, video_id_batch, 'train')

  print('Constructing validation model and input')
  with tf.variable_scope('val_model', reuse=None):
    val_set, video_id_batch, metadata_batch = input.create_batch(FLAGS.path, 'valid', FLAGS.valid_batch_size,
                                                                 int(math.ceil(FLAGS.num_iterations/FLAGS.valid_interval)+10),
                                                                 False)
    val_set = tf.cast(val_set, tf.float32)
    val_model = Model(val_set, video_id_batch, 'valid', reuse_scope=training_scope, metadata=metadata_batch)
  
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

  elapsed_time = 0

  ''' main training loop '''
  try:
    for itr in range(initializer.itr_start, initializer.itr_start + FLAGS.num_iterations):
      if initializer.coord.should_stop():
        break

      #Training Step on batch
      learning_rate = learning_rate_decay(FLAGS.learning_rate, itr, decay_factor=FLAGS.learning_rate_decay)
      feed_dict = {train_model.learning_rate: learning_rate, train_model.elapsed_time: float(elapsed_time)}

      t = time.time()
      train_loss, _, train_summary_str, _ = initializer.sess.run([train_model.loss, train_model.train_op, train_model.sum_op, train_model.label], feed_dict)
      elapsed_time = time.time() - t

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

      #Print Interation and loss
      tf.logging.info(' ' + str(itr) + ':    ' + str(train_loss))

  except tf.errors.OutOfRangeError:
    tf.logging.info('Done training -- iterations limit reached')
  finally:
    # When done, ask the threads to stop.
    initializer.coord.request_stop()

  tf.logging.info(' Saving Model ... ')
  saver.save(initializer.sess, output_dir + '/model')

  # Wait for threads to finish.
  initializer.stop_session()

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

  tf.logging.info(' --- Start validation --- ')

  try:
    feed_dict = {val_model.learning_rate: 0.0}
    val_summary_str = []


    val_loss, val_summary_str, output_frames, hidden_representations, labels, metadata = initializer.sess.run(
      [val_model.loss, val_model.sum_op, val_model.output_frames, val_model.hidden_repr, val_model.label, val_model.metadata], feed_dict)

    if FLAGS.valid_mode == 'vector':
      # store encoder latent vector for analysing

      hidden_repr_dir = create_subfolder(output_dir, 'hidden_repr')
      store_encoder_latent_vector(hidden_repr_dir, hidden_representations, labels, True)

    if FLAGS.valid_mode == 'gif':
      # summary and log
      val_model.iter_num = 1
      tf.logging.info('Converting validation frame sequences to gif')
      store_output_frames_as_gif(output_frames, labels, output_dir)
      tf.logging.info('Dumped validation gifs in: ' + str(output_dir))

    if FLAGS.valid_mode == 'similarity':
      print(str(similarity_computations.compute_hidden_representation_similarity(hidden_representations, labels, 'cos')))

    if FLAGS.valid_mode == 'data_frame':
      store_latent_vectors_as_df(output_dir, hidden_representations, labels, metadata)

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
  if FLAGS.valid_only:
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
      output_dir = create_session_dir(FLAGS.output_dir)
    else:
      output_dir = FLAGS.pretrained_model
      print('Reusing provided session directory:', output_dir)

    tf.logging.info(' --- TRAIN+VALID MODE --- ')
    write_metainfo(output_dir, model, FLAGS)
    train_valid_run(output_dir)


if __name__ == '__main__':
  app.run()

