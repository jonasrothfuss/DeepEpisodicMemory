"""Convert data to TFRecords file format with example protos. An Example is a mostly-normalized data format for
 storing data for training and inference. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import tensorflow as tf


FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data, name):
    """Converts a dataset to tfrecords.
    :param data: ndarray(uint8) of shape (v,i,c,h,w) with v=number of videos, i=number of images, c=number of image
    channels, h=image height, w=image width
    :param name: filename; data samples type (train|valid|test)
    """
    num_videos = data.shape[0]
    num_images = data.shape[1]
    num_channels = data.shape[2]
    height = data.shape[3]
    width = data.shape[4]

    filename = os.path.join(FLAGS.outputPath, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for videoCount in range(num_videos):
        for imageCount in range(num_images):
            for depthCount in range(num_channels):
                # more compatible than tobytes()
                image_raw = data[videoCount, imageCount, depthCount, :, :].tostring()
                print("Processing of video: " + str(videoCount) + " image: " + str(imageCount))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'videoNumber': _int64_feature(videoCount),
                    'imageNumber': _int64_feature(imageCount),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'depth': _int64_feature(depthCount),
                    'image_raw': _bytes_feature(image_raw)}))
                #image_back = np.fromstring(image_raw, dtype='uint8')
                writer.write(example.SerializeToString())
    writer.close()

def main(args):
    # Get the data.
    data_train = np.load(FLAGS.filePath)
    convert_to(data_train, FLAGS.type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--filePath',
      type=str,
      default='/tmp/data',
      help='Directory to numpy (train|valid|test) file'
    )

    parser.add_argument(
      '--outputPath',
      type=str,
      default='/tmp/data',
      help='Directory for storing tf records'
    )

    parser.add_argument(
      '--type',
      type=str,
      default='train',
      help='Type of data (train|valid|test)'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)