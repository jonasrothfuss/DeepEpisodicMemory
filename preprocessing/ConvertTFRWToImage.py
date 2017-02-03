from PIL import Image
#from Image import Image
#import PIL.Image
import os
import numpy as np
import tensorflow as tf



from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

def main():
  filename_queue = tf.train.string_input_producer(['/Users/fabioferreira/Downloads/push_train.tfrecord-00001-of-00264'])  # list of files to read
  read_and_decode(filename_queue)
  #get_all_records('/Users/fabioferreira/Downloads/push_testnovel.tfrecord-00000-of-00005')


def read_and_decode(filename_queue):
 reader = tf.TFRecordReader()
 _, serialized_example = reader.read(filename_queue)

 image_seq, state_seq, action_seq = [], [], []

 init_op = tf.initialize_all_variables()
 #with tf.Session() as sess:
 with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1000):#FLAGS.sequence_length):
        image_name = 'move/' + str(i) + '/image/encoded'
        features = {image_name: tf.FixedLenFeature([1], tf.string)}
        features = tf.parse_single_example(serialized_example, features=features)

     #features = tf.parse_single_example(serialized_example, features={'image': tf.FixedLenFeature([3], tf.string)})

        image_buffer = tf.reshape(features[image_name], shape=[])
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        print(i)
        image_npy = image.eval()
        print(image_npy.shape)
        img = Image.fromarray(image_npy, 'RGB')
        img.save('/Users/fabioferreira/Downloads/PushExtracted/'+str(i)+'.png')
        #img.show()


        #print(Image.fromarray(np.asarray(image_npy)))

        #my_image = image.eval()

        #image.save("output/" + str(i) + '-train.png')
        #Image.show(image)
        #print(np.asarray(image))
        #Image.show(Image.fromarray(np.asarray(image)))

    coord.request_stop()
    coord.join(threads)
     #image = tf.decode_raw(features['image'], tf.uint8)




# def get_all_records(FILE):
#  with tf.Session() as sess:
#    filename_queue = tf.train.string_input_producer([ FILE ])
#    image = read_and_decode(filename_queue)
#
#    init_op = tf.initialize_all_variables()
#    sess.run(init_op)
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    for i in range(2053):
#      example, l = sess.run([image])
#      img = Image.fromarray(example, 'RGB')
#      img.save( "output/" + str(i) + '-train.png')
#
#      print (example,l)
#    coord.request_stop()
#    coord.join(threads)
#



if __name__ == '__main__':
    main()
