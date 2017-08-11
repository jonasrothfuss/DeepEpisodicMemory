"""Convert data to TFRecords file format with example protos. An Example is a mostly-normalized data format for
 storing data for training and inference. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

#statics for data
NUM_IMAGES = 10
NUM_DEPTH = 4
WIDTH = 128
HEIGHT = 128
# specifies the number of pre-processing threads
NUM_THREADS = 32

# Constants used for dealing with the tf records files, aligned with convertToRecords.
flags.DEFINE_string('train_files', 'train*.tfrecords', 'Regex for filtering train tfrecords files.')
flags.DEFINE_string('valid_files', 'valid*.tfrecords', 'Regex for filtering valid tfrecords files.')
flags.DEFINE_string('test_files', 'test*.tfrecords', 'Regex for filtering test tfrecords files.')


def read_and_decode(filename_queue):
    """Creates one image sequence"""

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq = []
    video_id = None

    for imageCount in range(NUM_IMAGES):
        path = 'blob' + '/' + str(imageCount)

        feature_dict = {
          path: tf.FixedLenFeature([], tf.string),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'id': tf.FixedLenFeature([], tf.string),
          'metadata': tf.FixedLenFeature([], tf.string)
        }

        features = tf.parse_single_example(
            serialized_example,
            features=feature_dict)

        image_buffer = tf.reshape(features[path], shape=[])
        image = tf.decode_raw(image_buffer, tf.uint8)
        image = tf.reshape(image, tf.pack([HEIGHT, WIDTH, NUM_DEPTH]))
        image = tf.reshape(image, [1, HEIGHT, WIDTH, NUM_DEPTH])


        image_seq.append(image)

    if features:
      video_id = features['id']

    image_seq = tf.concat(0, image_seq)

    return image_seq, video_id, features


def create_batch(directory, mode, batch_size, num_epochs, overall_images_count, standardize=True):

    """ If mode equals 'train": Reads input data num_epochs times and creates batch
        If mode equals 'valid': Creates one large batch with all validation tensors.
        batch_size will be ignored and num_epochs will be set to 1 in this case.
        If mode equals 'test': #TODO

    :arg
        ;param directory: path to directory where train/valid tfrecord files are stored
        ;param modus: for differentiating data (train|valid|test)
        ;param batch_size: number of batches that will be created
        ;param num_epochs: number of times to read the input data, or 0/None for endless

    :returns
        A batch array of shape(s, i, h, w, c) where:
        s: batch size
        i: length of image sequence
        h: height of image
        w: width of image
        c: depth of image
    """
    path = os.path.abspath(directory)
    if mode == 'train':
      data_filter = FLAGS.train_files
    elif mode == 'valid':
      data_filter = FLAGS.valid_files
    elif mode == 'test':
      data_filter = FLAGS.test_files

    filenames = gfile.Glob(os.path.join(path, data_filter))

    if not filenames:
        raise RuntimeError('No data files found.')

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
          filenames, num_epochs=num_epochs)

        # sharing the same file even when multiple reader threads used
        image_seq_tensor, video_id, features = read_and_decode(filename_queue)


        if mode == 'valid' or mode == 'test':
          if not batch_size:
            batch_size = get_number_of_records(filenames)
          assert batch_size > 0
          image_seq_batch, video_id_batch, metadata_batch = tf.train.batch(
              [image_seq_tensor, video_id, features['metadata']], batch_size=batch_size, num_threads=NUM_THREADS, capacity=100 * batch_size)


        # -- training -- get shuffled batches
        else:
          # Shuffle the examples and collect them into batch_size batches.
          # (Internally uses a RandomShuffleQueue.)
          # We run this in two threads to avoid being a bottleneck.
          image_seq_batch, video_id_batch = tf.train.shuffle_batch(
            [image_seq_tensor, video_id], batch_size=batch_size, num_threads=NUM_THREADS,
            capacity=60*8* batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=10*8*batch_size)
          metadata_batch = None

        return image_seq_batch, video_id_batch, metadata_batch


def get_number_of_records(filenames):

  """Iterates through the tfrecords files given by the list 'filenames'
  and returns the number of available videos
  :param filenames a list with absolute paths to the .tfrecords files
  :return number of found videos (int)
  """

  filename_queue_val = tf.train.string_input_producer(
    filenames, num_epochs=1)
  image_seq_tensor_val = read_and_decode(filename_queue_val)

  num_examples = 0

  # create new session to determine batch_size for validation/test data
  with tf.Session() as sess_valid:
    init_op = tf.group(tf.local_variables_initializer())
    sess_valid.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      while True:
        sess_valid.run([image_seq_tensor_val])
        num_examples += 1
    except tf.errors.OutOfRangeError as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)

  return num_examples
