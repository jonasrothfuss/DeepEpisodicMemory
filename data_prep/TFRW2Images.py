from PIL import Image
#from Image import Image
#import PIL.Image
import os
import numpy as np
import tensorflow as tf



from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

generateGif = True
storeSingle = False
numVideos = 5

def main():
  filenames = (['/Users/fabioferreira/Downloads/push_testnovel.tfrecord-00000-of-00005'])# list of files to read
  path = '/Users/fabioferreira/Downloads/PushExtracted/'

  train_image_tensor = createImages(filenames)
  sess = tf.InteractiveSession()
  tf.train.start_queue_runners(sess)
  sess.run(tf.initialize_all_variables())
  train_videos = sess.run(train_image_tensor)

  if storeSingle:
    storeSingleImages(numVideos, train_videos, path)

  if generateGif:
    createGif(numVideos, train_videos, path)


def createImages(filenames):
    if not filenames:
        raise RuntimeError('No data files found.')
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq = []


    for i in range(25):
        image_name = 'move/' + str(i) + '/image/encoded'
        features = {image_name: tf.FixedLenFeature([1], tf.string)}
        features = tf.parse_single_example(serialized_example, features=features)


        image_buffer = tf.reshape(features[image_name], shape=[])
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image.set_shape([512, 640, 3])

        image = tf.reshape(image, [1, 512, 640, 3])
        image_seq.append(image)

    image_seq = tf.concat(0, image_seq)

    image_batch = tf.train.batch(
        [image_seq],
        25, #batch size
        num_threads=1,
        capacity=1)
    return image_batch


def createGif(numVideos, videos, path):
    import moviepy.editor as mpy
    for i in range(numVideos):
        npy = videos[i]
        clip = mpy.ImageSequenceClip(list(npy), fps=10)
        clip.write_gif(path + 'train' + str(i) +'.gif')

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


if __name__ == '__main__':
    main()
