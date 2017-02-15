"""Convert data to TFRecords file format with example protos. An Example is a mostly-normalized data format for
 storing data for training and inference. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_to_tfrecord(data, name, fragmentSize):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
    :param data: ndarray(uint8) of shape (v,i,h,w,c) with v=number of videos, i=number of images, c=number of image
    channels, h=image height, w=image width
    :param name: filename; data samples type (train|valid|test)
    ;param fragmentSize: specifies how many videos are stored in one tfrecords file
    """
    num_videos = data.shape[0]
    num_images = data.shape[1]
    num_channels = data.shape[4]
    height = data.shape[2]
    width = data.shape[3]

    i=0
    writer = None
    feature = {}


    for videoCount in range(num_videos):

        if videoCount % fragmentSize == 0:
            if writer is not None:
                writer.close()
            i += 1
            filename = os.path.join(FLAGS.outputPath, name + str(i) + '-of-' + str(int(math.ceil(num_videos/fragmentSize))) + '.tfrecords')
            print('Writing', filename)
            writer = tf.python_io.TFRecordWriter(filename)

        for imageCount in range(num_images):
            path = 'blob' + '/' + str(imageCount)
            image = data[videoCount, imageCount, :, :, :]

            #test = tf.image.encode_png(image)
            #plt.imshow(image_new)
            #plt.show()
            image = image.astype(np.uint8)
            image_raw = image.tostring()


            print("Processing of video: " + str(videoCount) + " image: " + str(imageCount))

            feature[path]= _bytes_feature(image_raw)
            #feature[path] = _bytes_feature(test)
            feature['height'] = _int64_feature(height)
            feature['width'] = _int64_feature(width)
            feature['depth'] = _int64_feature(num_channels)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def main(args):
    # Get the data.
    data_train = np.load(FLAGS.filePath)
    save_to_tfrecord(data_train, FLAGS.type, 1)


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