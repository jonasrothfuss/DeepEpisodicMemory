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
NUM_IMAGES = 20
NUM_DEPTH = 3
WIDTH = 128
HEIGHT = 128
# specifies the number of pre-processing threads
NUM_THREADS = 4

# Constants used for dealing with the tf records files, aligned with convertToRecords.
flags.DEFINE_string('train_files', 'train*.tfrecords', 'Regex for filtering train tfrecords files.')
flags.DEFINE_string('valid_files', 'valid*.tfrecords', 'Regex for filtering valid tfrecords files.')
flags.DEFINE_string('test_files', 'test*.tfrecords', 'Regex for filtering test tfrecords files.')
flags.DEFINE_string('input', '/tmp/data', 'Directory to tfrecord files')


def read_and_decode(filename_queue):
    """Creates one image sequence"""

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq = []

    for imageCount in range(NUM_IMAGES):
        path = 'blob' + '/' + str(imageCount)

        features = tf.parse_single_example(
            serialized_example,
            features={
                path: tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64)
            })

        image_buffer = tf.reshape(features[path], shape=[])
        image = tf.decode_raw(image_buffer, tf.uint8)
        image = tf.reshape(image, tf.pack([HEIGHT, WIDTH, NUM_DEPTH]))
        image = tf.reshape(image, [1, HEIGHT, WIDTH, NUM_DEPTH])

        image_seq.append(image)

    image_seq = tf.concat(0, image_seq)

    return image_seq


def create_batch(directory, mode, batch_size, num_epochs):

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
        image_seq_tensor = read_and_decode(filename_queue)

        # -- validation -- read all validation data
        if mode == 'valid':
          valid_data = []
          with tf.Session() as sess_valid:
            init_op = tf.group(tf.local_variables_initializer())
            sess_valid.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
              # tf.parse_example requires number of examples beforehand. Therefore, using more flexible
              # reader solution that throws an error when all tfrecords depleted
              while True:
                sample = sess_valid.run([image_seq_tensor])
                images = tf.reshape(sample[0], tf.pack([NUM_IMAGES, HEIGHT, WIDTH, NUM_DEPTH]))
                images = tf.reshape(images, [1, NUM_IMAGES, HEIGHT, WIDTH, NUM_DEPTH])
                valid_data.append(images)
            except tf.errors.OutOfRangeError as e:
              coord.request_stop(e)
            finally:
              coord.request_stop()
              coord.join(threads)
              valid_data = tf.reshape(valid_data, [len(valid_data), 1, NUM_IMAGES, HEIGHT, WIDTH, NUM_DEPTH])
              # improve speed with creating empty tensor with shape(len(valid_data, 1, NUM_IMAGES, HEIGHT, WIDTH, NUM_DEPTH)) and then reshape valid_data into it
              # returning tensor with shape (i, 1, NUM_IMAGES, HEIGHT, WIDTH, NUM_DEPTH)
              return valid_data
        # -- training -- get shuffled batches
        else:
          # Shuffle the examples and collect them into batch_size batches.
          # (Internally uses a RandomShuffleQueue.)
          # We run this in two threads to avoid being a bottleneck.
          image_seq_batch = tf.train.shuffle_batch(
              [image_seq_tensor], batch_size=batch_size, num_threads=NUM_THREADS,
              capacity=1000 + 3 * batch_size,
              # Ensures a minimum amount of shuffling of examples.
              min_after_dequeue=1000)
          return image_seq_batch


#def main(args):
    #test run
    #path = os.path.abspath(FLAGS.input)
    #filenames = gfile.Glob(os.path.join(path, FLAGS.train_files))


#if __name__ == '__main__':
  # FLAGS, unparsed = parser.parse_known_args()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)