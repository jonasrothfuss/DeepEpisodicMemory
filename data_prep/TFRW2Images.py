from PIL import Image
import argparse
import PIL.Image
import os
import sys
import numpy as np
import tensorflow as tf
import moviepy.editor as mpy



from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

generateGif = True
storeSingle = False
numVideos = 5
BATCH_SIZE = 25

def createImages(filenames):
    if not filenames:
        raise RuntimeError('No data files found.')
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq = []

    for i in range(20):
        #image_name = 'move/' + str(i) + '/image/encoded'
        image_name = 'blob' + '/' + str(i)
        features = {image_name: tf.FixedLenFeature([], tf.string)}
        features = tf.parse_single_example(serialized_example, features=features)

        #image = tf.image.decode_png(features[image_name], channels=0)

        # save some storage through uint8 as [0, 255] is sufficient
        image_data = tf.decode_raw(features[image_name], out_type=tf.uint8)
        # oe dimension on top of the extra dimension given by tf.pack for the batch size
        image = tf.reshape(image_data, tf.pack([1, 128, 128, 3]))
        image_seq.append(image)

    image_seq = tf.concat(0, image_seq)

    # create the batch
    image_batch = tf.train.batch(
        [image_seq],
        BATCH_SIZE,
        num_threads=1,
        capacity=1)
    return image_batch


def createGif(videos, labels, output_dir):
    import moviepy.editor as mpy
    for i in range(videos.shape[0]):
        npy = videos[i]
        clip = mpy.ImageSequenceClip(list(npy), fps=10)
        clip.write_gif(os.path.join(output_dir, 'original_clip_' + str(labels[i].decode('utf-8')) + '.gif'),
                       program='ffmpeg')

def storeSingleImages(numVideos, videos, path):
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for j in range(numVideos):
            video = videos[j]
            for i in range(25):
                image = video[i]
                img = Image.fromarray(image, 'RGB')
                img.save(path + str(j) + '_' + str(i) + '.png')

    coord.request_stop()
    coord.join(threads)


def main(unparsed):
    inputPath = os.path.abspath(FLAGS.input)
    inputPath += '/'
    outputPath = os.path.abspath(FLAGS.output)
    outputPath += '/'
    filenames = gfile.Glob(os.path.join(inputPath, '*.tfrecords'))

    train_image_tensor = createImages(filenames)
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())
    train_videos = sess.run(train_image_tensor)

    if storeSingle:
        storeSingleImages(numVideos, train_videos, outputPath)

    if generateGif:
        createGif(numVideos, train_videos, outputPath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--input',
      type=str,
      default='/tmp/data',
      help='Directory with .tfrecords files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/tmp/data',
        help='Directory where new files should be stored.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

