import os
from tensorflow.python.platform import gfile
import tensorflow as tf
import re
import random


def files_from_directory(dir_str, file_type):
  file_paths = gfile.Glob(os.path.join(dir_str, file_type))
  return [os.path.basename(i) for i in file_paths]

def file_paths_from_directory(dir_str, file_type):
  file_paths = gfile.Glob(os.path.join(dir_str, file_type))
  return file_paths


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
    print(video_name)
    p = re.compile('^([a-zA-Z0-9_-]+)_[0-9]+x[0-9]+')
    video_id = p.match(video_name).group(1)
    return video_id
  elif type == 'flyingshapes':
    video_id = video_name.split('_')[0]
    return video_id
  else: #just return filename without extension
    return video_name.replace('.avi', '').replace('.mp4', '')

def shuffle_files_in_list(paths_list, seed=5):
  """
  generates a list of randomly shuffled paths of the files contained in the provided directories
  :param paths_list: list with different path locations containing the files
  :return: returns two lists, one with the content of all the given directories in the provided order and another
  containing the same list randomly shuffled
  """
  assert paths_list is not None
  all_files = []
  for path_entry in paths_list:
    print(path_entry)
    all_files.extend(file_paths_from_directory(path_entry, '*.avi'))
  random.seed(a=seed)	
  return all_files, random.sample(all_files, len(all_files))
