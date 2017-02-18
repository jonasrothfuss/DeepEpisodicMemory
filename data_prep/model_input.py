"""Convert data to TFRecords file format with example protos. An Example is a mostly-normalized data format for
 storing data for training and inference. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf

from tensorflow.python.platform import gfile



FLAGS = None

#statics for data
NUM_IMAGES = 20
NUM_DEPTH = 3
WIDTH = 128
HEIGHT = 128
BATCH_SIZE = 25
NUM_EPOCHS = 1000
# specifies the number of pre-processing threads
NUM_THREADS = 4

# Constants used for dealing with the tf records files, aligned with convertToRecords.
TRAIN_FILES = 'train*.tfrecords'
VALIDATION_FILES = 'valid*.tfrecords'
TEST_FILES = 'test*.tfrecords'


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

    """Reads input data num_epochs times and creates batch

    :arg
        ;param directory: path to directory where train/valid tfrecord files are stored
        ;param modus: for differentiating data (train|valid|test)
        ;param batch_size: number of batches that will be created
        ;param num_epochs: number of times to read the input data, or 0/None to for endless

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
      data_filter = TRAIN_FILES
    elif mode == 'valid':
      data_filter = VALIDATION_FILES
    elif mode == 'test':
      data_filter = TEST_FILES

    filenames = gfile.Glob(os.path.join(path, data_filter))

    if not filenames:
        raise RuntimeError('No data files found.')

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs)

        # sharing the same file even when multiple reader threads used
        image_seq_tensor = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.

        #TODO: use shuffle
        image_seq_batch = tf.train.shuffle_batch(
            [image_seq_tensor], batch_size=batch_size, num_threads=NUM_THREADS,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        return image_seq_batch


def main(args):
    #test run
    path = os.path.abspath(FLAGS.input)
    filenames = gfile.Glob(os.path.join(path, TRAIN_FILES))
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs='None')

    image_seq_tensor = read_and_decode(filenames)
    #createInputs(FLAGS.input, True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--input',
      type=str,
      default='/tmp/data',
      help='Directory to tfrecord files'
    )

    #FLAGS, unparsed = parser.parse_known_args()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)