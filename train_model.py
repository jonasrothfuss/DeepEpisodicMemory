import tensorflow as tf
import numpy as np
from models import model
import math
import data_prep.model_input as input
import io
import matplotlib.pyplot as plt



from tensorflow.python.platform import app
from tensorflow.python.platform import flags

tf.logging.set_verbosity(tf.logging.INFO)

LOSS_FUNCTIONS = ['mse', 'gdl']

FLAGS = flags.FLAGS
#DATA_PATH = '/home/jonasrothfuss/Dropbox/Deep_Learning_for_Object_Manipulation/4_Data/Datasets/ArtificialFlyingBlobs'
#LOG_PATH = '/home/jonasrothfuss/Desktop/'
#OUT_DIR = '/home/jonasrothfuss/Desktop/'
DATA_PATH = '/Users/fabioferreira/Dropbox/Deep_Learning_for_Object_Manipulation/4_Data/Datasets/ArtificialFlyingBlobs'
LOG_PATH = '/Users/fabioferreira/Desktop'
OUT_DIR = '/Users/fabioferreira/Desktop'

# hyperparameters
flags.DEFINE_integer('num_iterations', 100000, 'specify number of training iterations, defaults to 10 000')
flags.DEFINE_integer('learning_rate', 0.0001, 'learning rate for Adam optimizer')
flags.DEFINE_string('loss_function', 'mse', 'specify loss function to minimize, defaults to gdl')
flags.DEFINE_string('batch_size', 50, 'specify the batch size, defaults to 50')
flags.DEFINE_integer('valid_interval', 500, 'number of training steps between each validation')
flags.DEFINE_integer('summary_interval', 100, 'number of training steps between summary is stored')
flags.DEFINE_integer('save_interval', 2000, 'number of training steps between session/model dumps')

flags.DEFINE_string('encoder_length', 5, 'specifies how many images the encoder receives, defaults to 5')
flags.DEFINE_string('decoder_future_length', 5, 'specifies how many images the future prediction decoder receives, defaults to 5')
flags.DEFINE_string('decoder_reconst_length', 5, 'specifies how many images the reconstruction decoder receives, defaults to 5')

#IO specifications
flags.DEFINE_string('path', DATA_PATH, 'specify the path to where tfrecords are stored, defaults to "../data/"')
flags.DEFINE_string('event_log_dir', LOG_PATH, 'specify the path where logger files are dumped')
flags.DEFINE_integer('num_channels', 3, 'number of channels in the input frames')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('pretrained_model', '', 'filepath of a pretrained model to initialize from.')


def gen_plot(predicted_frame):
  """Create a pyplot plot and save to buffer."""
  plt.figure()
  plt.plot([1, 2])
  plt.title("test")
  buf = predicted_frame
  plt.savefig(buf, format='png')
  buf.seek(0)
  return buf


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
               reuse_scope=None):

    self.learning_rate = tf.placeholder_with_default(FLAGS.learning_rate, ())
    #self.prefix = tf.placeholder(tf.string, []) #string for summary that denotes whether train or val
    self.iter_num = tf.placeholder(tf.float32, [])
    summaries = []

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
    summaries.append(tf.summary.scalar(summary_prefix + '_loss', self.loss)) #TODO: add more summaries

    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    self.sum_op = tf.summary.merge(summaries)


def main(unused_argv):

  print('Constructing train model and input')
  with tf.variable_scope('train_model', reuse=None) as training_scope:
    train_batch = input.create_batch(FLAGS.path, 'train', FLAGS.batch_size, int(math.ceil(FLAGS.num_iterations/(FLAGS.batch_size * 20))))
    train_batch = tf.cast(train_batch, tf.float32)
    train_model = Model(train_batch, 'train')

  print('Constructing validation model and input')
  with tf.variable_scope('val_model', reuse=None):
    val_set = input.create_batch(FLAGS.path, 'valid', FLAGS.batch_size, 1)  # TODO: ensure that validation set data doesn't change (--> Fabio)
    val_set = tf.cast(val_set, tf.float32)
    val_model = Model(val_set, 'valid', reuse_scope=training_scope)

  print('Constructing saver')
  saver = tf.train.Saver(
    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)

  # Start Session and initialize variables
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess = tf.Session()
  sess.run(init_op)

  #restore dumped model if provided
  if FLAGS.pretrained_model:
    print('Restore model from: ' + str(FLAGS.pretrained_model))
    saver.restore(sess, FLAGS.pretrained_model)


  summary_writer = tf.summary.FileWriter(FLAGS.event_log_dir, graph=sess.graph, flush_secs=10)

  # Start input enqueue threads
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  tf.logging.info(' --- Start Training --- ')
  tf.logging.info(' Iteration, Train_Loss ')


  ''' main training loop '''
  try:
    for itr in range(FLAGS.num_iterations):
      if coord.should_stop():
        break

      #Training Step on batch
      feed_dict = {train_model.learning_rate: FLAGS.learning_rate}
      train_loss, _, train_summary_str = sess.run([train_model.loss, train_model.train_op, train_model.sum_op], feed_dict)
      #Print Interation and loss
      tf.logging.info(' ' + str(itr) + ':    ' + str(train_loss))

      #validation
      if itr % FLAGS.valid_interval == 1:
        feed_dict = {val_model.learning_rate: 0.0}

        # prepare image view in TensorBoard
        #predicted_frames = val_model.frames_pred
        #pred_frame_batch = predicted_frames[-1] #last batch
        #pred_frame = pred_frame_batch[-1] #last predicted image of batch
        #pred_frame = pred_frame.eval(session=sess)
        #plot_buf = gen_plot(pred_frame)
        #image = tf.decode_raw(plot_buf, tf.uint8)
        #image_summary_t = tf.image_summary("plot", pred_img)
        #TODO Add image summary

        #summary and log
        val_loss, val_summary_str = sess.run([val_model.loss, val_model.sum_op], feed_dict)
        summary_writer.add_summary(val_summary_str, itr)
        #Print validation loss
        tf.logging.info(' Validation loss at step ' + str(itr) + ':    ' + str(val_loss))

      #dump summary
      if itr % FLAGS.summary_interval == 1:
        summary_writer.add_summary(train_summary_str, itr)

      if itr % FLAGS.save_interval == 1:
        tf.logging.info(' Saving Model ... ')
        saver.save(sess, FLAGS.output_dir + '/model' + str(itr))

  except tf.errors.OutOfRangeError:
    tf.logging.info('Done training -- iterations limit reached')
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()

  tf.logging.info(' Saving Model ... ')
  saver.save(sess, FLAGS.output_dir + '/model')

  # Wait for threads to finish.
  coord.join(threads)
  sess.close()


if __name__ == '__main__':
  app.run()


