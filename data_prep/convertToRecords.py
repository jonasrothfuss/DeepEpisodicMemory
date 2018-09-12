"""Convert data to TFRecords file ((avi -> numpy -> tfrecords) format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import multiprocessing
import os

import cv2 as cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

from data_prep.video_preparation import create_activity_net_metadata_dicts, create_youtube8m_metadata_dicts, \
  create_20bn_metadata_dicts, create_ucf101_metadata_dicts
from utils.helpers import chunks
from utils.io_handler import get_meta_info
from utils.video import compute_dense_optical_flow, getVideoCapture, getNextFrame

NUM_CORES = multiprocessing.cpu_count()
FLAGS = None
FILE_FILTER = '*.avi'
NUM_FRAMES_PER_VIDEO = 20
NUM_CHANNELS_VIDEO = 3
WIDTH_VIDEO = 128
HEIGHT_VIDEO = 128
ALLOWED_TYPES = [None, 'flyingshapes', 'activity_net', 'UCF101', 'youtube8m', '20bn_train', '20bn_valid']

SOURCE = '/PDFData/scrambled_eggs/augment/rgb/valid/'
DESTINATION = '/localhome/rothfuss/data/segmented_scrambled_eggs_augmented/tfrecords_all'
METADATA_SUBCLIPS_DICT = '/common/homes/students/rothfuss/Downloads/ucf101_prepared_clips/metadata_subclips.json'
METADATA_TAXONOMY_DICT = '/common/homes/students/rothfuss/Downloads/ucf101_prepared_clips/metadata.json'
METADATA_y8m_027 = '/PDFData/rothfuss/data/youtube8m/videos/pc027/metadata.json'
METADATA_y8m_031 = '/PDFData/rothfuss/data/youtube8m/videos/pc031/metadata.json'
METADATA_DICT = '/PDFData/rothfuss/data/youtube8m/videos/pc031/metadata.json'
CSV_20BN_TRAIN = '/PDFData/rothfuss/data/20bn-something/new_label_files/something-something-v1-train_test.csv'
CSV_20BN_VALID = '/PDFData/rothfuss/data/20bn-something/something-something-v1-validation.csv'
METADATA_DICT_UCF101 = '/PDFData/rothfuss/data/UCF101/prepared_videos/metadata.json'


FLAGS = flags.FLAGS
flags.DEFINE_integer('num_video', 50, 'Number of videos stored in one single tfrecords file')
flags.DEFINE_string('source', SOURCE, 'Directory with avi files')
flags.DEFINE_string('file_path', '/tmp/data', 'Directory to numpy (train|valid|test) file')
flags.DEFINE_string('output_path', DESTINATION, 'Directory for storing tf records')
flags.DEFINE_boolean('use_meta', False, 'indicates whether meta-information shall be extracted from filename')
flags.DEFINE_boolean('optical_flow', False, 'Indictes whether optical flow shall be computed and added as fourth channel')
flags.DEFINE_string('typ', None, 'Processing type for video data - Allowed values: ' + str(ALLOWED_TYPES))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_numpy_to_tfrecords(data, destination_path, meta_info, name, fragmentSize, current_batch_number, total_batch_number):
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

  assert all(['id' in entry for entry in meta_info])

  writer = None
  feature = {}

  for videoCount, meta_info_entry in zip(range(num_videos), meta_info):

      if videoCount % fragmentSize == 0:
          if writer is not None:
              writer.close()
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
          feature['id'] = _bytes_feature(meta_info_entry['id'].encode())

          meta_info_entry.update({'height': height, 'width': width, 'depth': num_channels})
          feature['metadata'] = _bytes_feature(json.dumps(meta_info_entry).encode())

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
  if writer is not None:
    writer.close()

def convert_avi_to_numpy(filenames, typ=None, meta_dict = None, dense_optical_flow=False):
  """Generates an ndarray from multiple avi files given by filenames.
  Implementation chooses frame step size automatically for a equal separation distribution of the avi images.

  :param filenames
  :param type: processing type for video data
  :param meta_dict: dictionary with meta-information about the video files - keys of the dict must be the video_ids
  :return if no optical flow is used: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos, i=number of images,
  (h,w)=height and width of image, c=channel, if optical flow is used: ndarray(uint32) of (v,i,h,w,
  c+1)"""
  assert typ in ALLOWED_TYPES
  global NUM_CHANNELS_VIDEO
  if not filenames:
    raise RuntimeError('No data files found.')

  number_of_videos = len(filenames)

  if dense_optical_flow:
    # need an additional channel for the optical flow with one exception:
    global NUM_CHANNELS_VIDEO
    NUM_CHANNELS_VIDEO = 4
    num_real_image_channel = 3
  else:
    # if no optical flow, make everything normal:
    num_real_image_channel = NUM_CHANNELS_VIDEO

  data = []
  meta_info = list()

  def avi_file_to_ndarray(i, filename):
    image = np.zeros((HEIGHT_VIDEO, WIDTH_VIDEO, num_real_image_channel), dtype=np.uint8)
    video = np.zeros((NUM_FRAMES_PER_VIDEO, HEIGHT_VIDEO, WIDTH_VIDEO, NUM_CHANNELS_VIDEO), dtype=np.uint32)
    imagePrev = None
    assert os.path.isfile(filename), "Couldn't find video file"
    cap = getVideoCapture(filename)
    assert cap is not None, "Couldn't load video capture:" + filename + ". Moving to next video."

    # compute meta data of video
    frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # returns nan, if fps needed a measurement must be implemented
    # frameRate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    steps = math.floor(frameCount / NUM_FRAMES_PER_VIDEO)
    j = 0
    prev_frame_none = False

    restart = True
    assert not (frameCount < 1 or steps < 1), str(filename) + " does not have enough frames. Moving to next video."

    while restart:
      for f in range(int(frameCount)):
        # get next frame after 'steps' iterations:
        # floor used after modulo operation because rounding module before leads to
        # unhandy partition of data (big gab in the end)
        if math.floor(f % steps) == 0:
          frame = getNextFrame(cap)
          # special case handling: opencv's frame count != real frame count, reiterate over same video
          if frame is None and j < NUM_FRAMES_PER_VIDEO:
            if frame and prev_frame_none: break
            prev_frame_none = True
            # repeat with smaller step size
            steps -= 1
            if steps == 0: break
            print("reducing step size due to error")
            j = 0
            cap.release()
            cap = getVideoCapture(filenames[i])
            # wait for image retrieval to be ready
            cv2.waitKey(3000)
            video.fill(0)
            continue
          else:
            if j >= NUM_FRAMES_PER_VIDEO:
              restart = False
              break
            # iterate over channels
            if frame.ndim == 2:
              # cv returns 2 dim array if gray
              resizedImage = cv2.resize(frame[:, :], (HEIGHT_VIDEO, WIDTH_VIDEO))
            else:
              for k in range(num_real_image_channel):
                resizedImage = cv2.resize(frame[:, :, k], (HEIGHT_VIDEO, WIDTH_VIDEO))
                image[:, :, k] = resizedImage

              if dense_optical_flow:
                # optical flow requires at least two images
                if imagePrev is not None:
                  frameFlow = np.zeros((HEIGHT_VIDEO, WIDTH_VIDEO))
                  frameFlow = compute_dense_optical_flow(imagePrev, image)
                  frameFlow = cv2.cvtColor(frameFlow, cv2.COLOR_BGR2GRAY)
                else:
                  frameFlow = np.zeros((HEIGHT_VIDEO, WIDTH_VIDEO))

                imagePrev = image.copy()

            if dense_optical_flow:
              image_with_flow = image.copy()
              image_with_flow = np.concatenate((image_with_flow, np.expand_dims(frameFlow, axis=2)), axis=2)
              video[j, :, :, :] = image_with_flow
            else:
              video[j, :, :, :] = image
            j += 1
            # print('total frames: ' + str(j) + " frame in video: " + str(f))
        else:
          getNextFrame(cap)

    print(str(i + 1) + " of " + str(number_of_videos) + " videos processed", filenames[i])

    v = video.copy()
    cap.release()
    return v

  for i, file in enumerate(filenames):
    try:
      v = avi_file_to_ndarray(i, file)
      data.append(v)
    except Exception as e:
      print(e)

    #get meta info and append to meta_info list
    meta_info.append(get_meta_info(file, typ, meta_dict=meta_dict).copy())

  return np.array(data), meta_info


def save_avi_to_tfrecords(source_path, destination_path, videos_per_file, typ=None, video_filenames=None, dense_optical_flow=False):
  """calls sub-functions convert_avi_to_numpy and save_numpy_to_tfrecords in order to directly export tfrecords files
  :param source_path: directory where avi videos are stored
  :param destination_path: directory where tfrecords should be stored
  :param videos_per_file: specifies the number of videos within one tfrecords file
  :param use_meta: boolean that indicates whether to use meta information
  """
  global NUM_CHANNELS_VIDEO
  assert (NUM_CHANNELS_VIDEO == 3 and (not dense_optical_flow)) or (NUM_CHANNELS_VIDEO == 4 and dense_optical_flow), "correct NUM_CHANNELS_VIDEO"
  assert typ in ALLOWED_TYPES, str(typ) + " is not an allowed type"

  if video_filenames is not None:
    filenames = video_filenames
  else:
    filenames = gfile.Glob(os.path.join(source_path, FILE_FILTER))
  if not filenames:
    raise RuntimeError('No data files found.')

  print('Total videos found: ' + str(len(filenames)))

  filenames_split = list(chunks(filenames, videos_per_file))

  if typ == 'activity_net':
    meta_dict = create_activity_net_metadata_dicts(FLAGS.source, METADATA_SUBCLIPS_DICT, METADATA_TAXONOMY_DICT, FILE_FILTER)
  elif typ == 'youtube8m':
    meta_dict = create_youtube8m_metadata_dicts(FLAGS.source, METADATA_DICT, FILE_FILTER)
  elif typ == '20bn_train':
    meta_dict = create_20bn_metadata_dicts(FLAGS.source, CSV_20BN_TRAIN, FILE_FILTER)
  elif typ == '20bn_valid':
    meta_dict = create_20bn_metadata_dicts(FLAGS.source, CSV_20BN_VALID, FILE_FILTER)
  elif typ == 'UCF101':
    meta_dict = create_ucf101_metadata_dicts(FLAGS.source, METADATA_DICT_UCF101, FILE_FILTER)
  else:
    meta_dict = None

  for i, batch in enumerate(filenames_split):
    data, meta_info = convert_avi_to_numpy(batch, typ=typ, meta_dict=meta_dict, dense_optical_flow=dense_optical_flow)
    total_batch_number = int(math.ceil(len(filenames)/videos_per_file))
    print('Batch ' + str(i+1) + '/' + str(total_batch_number))
    save_numpy_to_tfrecords(data, destination_path, meta_info, 'train_blobs_batch_', videos_per_file, i+1,
                            total_batch_number)


def main(argv):
  #_, all_files_shuffled = io_handler.shuffle_files_in_list([FLAGS.source])

  #categories = ['Train', 'Dish (food)']
  #_, all_files_shuffled = io_handler.shuffle_files_in_list_from_categories([FLAGS.source], categories, METADATA_y8m_027, type='youtube8m')

  #with open('/common/homes/students/rothfuss/Downloads/shuffled_videos.txt', 'r') as f:
  #  content = f.read()
  #  all_files_shuffled = content.split('\n')
  #print('Collected %i Video Files'%len(all_files_shuffled))
  #all_files_shuffled = all_files_shuffled[0:140000]

  save_avi_to_tfrecords(FLAGS.source, FLAGS.output_path, FLAGS.num_video, typ=None, video_filenames=None, dense_optical_flow=FLAGS.optical_flow)

if __name__ == '__main__':
  app.run()




