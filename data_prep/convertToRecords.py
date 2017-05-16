"""Convert data to TFRecords file ((avi -> numpy -> tfrecords) format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import math
import warnings
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import tensorflow as tf
import json
from data_prep.video_preparation import create_ucf101_metadata_dicts
import utils.io_handler as io_handler

FLAGS = None
FILE_FILTER = '*.avi'
NUM_FRAMES_PER_VIDEO = 20
NUM_CHANNELS_VIDEO = 3
WIDTH_VIDEO = 128
HEIGHT_VIDEO = 128

SOURCE = '/data/rothfuss/data/ucf101_prepared_videos'
DESTINATION = '/data/rothfuss/data/test_records'
METADATA_SUBCLIPS_DICT = '/common/homes/students/rothfuss/Downloads/clips/metadata_subclips.json'
METADATA_TAXONOMY_DICT = '/common/homes/students/rothfuss/Downloads/metadata.json'

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_videos', 1000, 'Number of videos stored in one single tfrecords file')
flags.DEFINE_string('source', SOURCE, 'Directory with avi files')
flags.DEFINE_string('file_path', '/tmp/data', 'Directory to numpy (train|valid|test) file')
flags.DEFINE_string('output_path', DESTINATION, 'Directory for storing tf records')
flags.DEFINE_boolean('use_meta', True, 'indicates whether meta-information shall be extracted from filename')



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_numpy_to_tfrecords(data, destination_path, meta_info, name, fragmentSize, current_batch_number, total_batch_number, type='fylingshapes'):
  """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
  :param data: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos, i=number of images, c=number of image
  channels, h=image height, w=image width
  :param meta_info: contains file information (typically extracted from video file names) in format: (num_videos, 7)
            7: id, shape, color, start_location, end_location, motion_location, eucl_distance
  :param name: filename; data samples type (train|valid|test)
  :param fragmentSize: specifies how many videos are stored in one tfrecords file
  :param current_batch_number: indicates the current batch index (function call within loop)
  :param total_batch_number: indicates the total number of batches
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
          filename = os.path.join(destination_path, name + str(current_batch_number) + '_of_' + str(total_batch_number) + '.tfrecords')
          print('Writing', filename)
          writer = tf.python_io.TFRecordWriter(filename)

      for imageCount in range(num_images):
          path = 'blob' + '/' + str(imageCount)
          image = data[videoCount, imageCount, :, :, :]
          image = image.astype(np.uint8)
          image_raw = image.tostring()

          feature[path]= _bytes_feature(image_raw)
          feature['height'] = _int64_feature(height)
          feature['width'] = _int64_feature(width)
          feature['depth'] = _int64_feature(num_channels)
          if type == 'flyingshapes':
            feature['id'] = _bytes_feature(meta_info[videoCount][0])
            meta_dict = {'height': height, 'width': width, 'depth': num_channels, 'id': meta_info[videoCount][0]}
          elif type == 'UCF101':
            feature['id'] = _bytes_feature(meta_info[videoCount]['video_id'])
            meta_dict = {'height': height, 'width': width, 'depth': num_channels}


          if len(meta_info[0]) > 1 and type == 'fylingshapes':
            feature['shape'] = _bytes_feature(meta_info[videoCount][1])
            feature['color'] = _bytes_feature(meta_info[videoCount][2])
            feature['start_location'] = _bytes_feature(meta_info[videoCount][3])
            feature['end_location'] = _bytes_feature(meta_info[videoCount][4])
            feature['motion_location'] = _bytes_feature(meta_info[videoCount][5])
            feature['eucl_distance'] = _bytes_feature(meta_info[videoCount][6])

            meta_dict['shape'] = meta_info[videoCount][1]
            meta_dict['color'] = meta_info[videoCount][2]
            meta_dict['start_location'] = meta_info[videoCount][3]
            meta_dict['end_location'] = meta_info[videoCount][4]
            meta_dict['motion_location'] = meta_info[videoCount][5]
            meta_dict['eucl_distance'] = meta_info[videoCount][6]
          elif len(meta_info[0]) > 1 and type == 'UCF101':
            meta_dict.update(meta_info[videoCount])
          else:
            warnings.warn("no meta info stored in tf records")

          feature['metadata'] = _bytes_feature(json.dumps(meta_dict))

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
  writer.close()


def convert_avi_to_numpy(filenames, dict = None, use_meta=FLAGS.use_meta):
  """Generates an ndarray from multiple avi files given by filenames.
  Implementation chooses frame step size automatically for a equal separation distribution of the avi images.

  :param filenames
  :param use_meta: indicates whether meta-information shall be extracted from filename
  :return ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos, i=number of images, c=number of image"""
  if not filenames:
    raise RuntimeError('No data files found.')

  number_of_videos = len(filenames)
  data = np.zeros((number_of_videos, NUM_FRAMES_PER_VIDEO, HEIGHT_VIDEO, WIDTH_VIDEO, NUM_CHANNELS_VIDEO), dtype = np.uint32)
  image = np.zeros((HEIGHT_VIDEO, WIDTH_VIDEO, NUM_CHANNELS_VIDEO), dtype=np.uint8)
  video = np.zeros((NUM_FRAMES_PER_VIDEO, HEIGHT_VIDEO, WIDTH_VIDEO, NUM_CHANNELS_VIDEO), dtype=np.uint32)
  meta_info = list()


  for i in range(number_of_videos):
    #print(i, filenames[i])
    cap = getVideoCapture(filenames[i])

    if cap is None:
      print("Couldn't load video capture:" + filenames[i] + ". Moving to next video.")
      break

    if use_meta:
      #meta_info_entry = get_meta_info(filenames[i])
      meta_info_entry = get_meta_info(filenames[i], dict=dict)
      meta_info.append(meta_info_entry)
    else: #only include filename as id
      meta_info.append([os.path.basename(filenames[i]).split('.')[0]])

    # compute meta data of video
    frameCount = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    # returns nan, if fps needed a measurement must be implemented
    # frameRate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    steps = frameCount / NUM_FRAMES_PER_VIDEO
    j = 0

    restart = True
    if frameCount < 1:
      restart = False

    while restart:
      for f in range(int(frameCount)):
        # get next frame after 'steps' iterations:
        # floor used after modulo operation because rounding module before leads to
        # unhandy partition of data (big gab in the end)
        if math.floor(f % steps) == 0:
          frame = getNextFrame(cap)
          # special case handling: opencv's frame count != real frame count, reiterate over same video
          if frame is None and j < NUM_FRAMES_PER_VIDEO:
            warnings.warn("reducing step size due to error")
            # repeat with smaller step size
            steps -= 1
            j = 0
            cap.release()
            cap = getVideoCapture(filenames[i])
            video.fill(0)
            break
          else:
            if j >= NUM_FRAMES_PER_VIDEO:
              restart = False
              break

            # iterate over channels
            if frame.ndim == 2:
              # cv returns 2 dim array if gray
              resizedImage = cv2.resize(frame[:, :], (HEIGHT_VIDEO, WIDTH_VIDEO))
            else:
              for k in range(NUM_CHANNELS_VIDEO):
                resizedImage = cv2.resize(frame[:, :, k], (HEIGHT_VIDEO, WIDTH_VIDEO))
                image[:, :, k] = resizedImage

            video[j, :, :, :] = image
            j += 1
            #print('total frames: ' + str(j) + " frame in video: " + str(f))
        else:
          getNextFrame(cap)

    #print(str(i + 1) + " of " + str(number_of_videos) + " videos processed")
    data[i, :, :, :, :] = video
    cap.release()
  return data, meta_info

def chunks(l, n):
  """Yield successive n-sized chunks from l.
  Used to create n sublists from a list l"""
  for i in range(0, len(l), n):
    yield l[i:i + n]

def save_avi_to_tfrecords(source_path, destination_path, videos_per_file=FLAGS.num_videos, use_meta=FLAGS.use_meta):
  """calls sub-functions convert_avi_to_numpy and save_numpy_to_tfrecords in order to directly export tfrecords files
  :param source_path: directory where avi videos are stored
  :param destination_path: directory where tfrecords should be stored
  :param videos_per_file: specifies the number of videos within one tfrecords file
  """
  filenames = gfile.Glob(os.path.join(source_path, FILE_FILTER))
  if not filenames:
    raise RuntimeError('No data files found.')

  print('Total videos found: ' + str(len(filenames)))
  i = 1
  filenames_split = list(chunks(filenames, videos_per_file))

  meta_dict = create_ucf101_metadata_dicts(FLAGS.source, METADATA_SUBCLIPS_DICT, METADATA_TAXONOMY_DICT,
                                           FILE_FILTER)

  for batch in filenames_split:
    data, meta_info = convert_avi_to_numpy(batch, dict=meta_dict, use_meta=use_meta)
    total_batch_number = int(math.ceil(len(filenames)/videos_per_file))
    print('Batch ' + str(i) + '/' + str(total_batch_number))
    #save_numpy_to_tfrecords(data, destination_path, meta_info, 'train_blobs_batch_', videos_per_file, i, total_batch_number, type='flyingshapes)
    save_numpy_to_tfrecords(data, destination_path, meta_info, 'train_blobs_batch_', videos_per_file, i, total_batch_number, type='UCF101')
    meta_info = []
    i += 1

def get_meta_info(filename, dict = None):

  """extracts meta information from video file names or from dictionary
  @param filename: one absolute path to a file of type string
  @returns tuple of strings containing: id, shape, color, start_location, end_location, motion_direction, eucl_distance_between_objs"""
  base = os.path.basename(filename)
  if dict is None:
    m = re.search('(\d+)_(\w+)_(\w+)_(\w+)_(\w+)_(\w+)_(\d+\.\d+)', base)
    assert m.lastindex >= 7
    return [m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)]
  else:
    result = io_handler.get_video_id_from_path(base, 'UCF101')
    if result:
      return dict[result.group(1)]




def getVideoCapture(path):
    cap = None
    if path:
      cap = cv2.VideoCapture(path)
      # set capture settings here:
      # cap.set(0, 0)  # (0,x) POS_MSEC, (1,x)
    return cap

def getNextFrame(cap):
  ret, frame = cap.read()
  if ret == False:
    return None

  return np.asarray(frame)


def main(argv):
  save_avi_to_tfrecords(FLAGS.source, FLAGS.output_path, FLAGS.num_videos, use_meta=FLAGS.use_meta)

if __name__ == '__main__':
  app.run()




