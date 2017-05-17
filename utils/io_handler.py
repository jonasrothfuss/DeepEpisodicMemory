import os
from tensorflow.python.platform import gfile
import tensorflow as tf
import re


def files_from_directory(dir_str, file_type):
  file_paths = gfile.Glob(os.path.join(dir_str, file_type))
  return [os.path.basename(i) for i in file_paths]


def get_filename_and_filetype_from_path(path):
  """extracts and returns both filename (e.g. 'video_1') and filetype (e.g. 'mp4') from a given absolute path"""
  filename = os.path.basename(path)
  video_id, filetype = filename.split(".")
  return video_id, filetype


def get_metadata_dict_as_bytes(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_video_id_from_path(path_str, type=None):
  video_name = os.path.basename(path_str)
  if type == 'UCF101':
    p = re.compile('^([a-zA-Z0-9_-]+_[0-9]+)_\d{3}')
    video_id = p.match(video_name).group(1)
    return video_id
  elif type == 'youtube8m':
    # TODO
    return None
  elif type == 'flyingshapes':
    video_id = video_name.split('_')[0]
    return video_id
  else: #just return filename without extension
    return video_name.replace('.avi', '').replace('.mp4', '')