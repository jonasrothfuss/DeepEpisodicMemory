import math, pafy, json, sys, os, os.path, moviepy, imageio, os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path
from moviepy.editor import *
import tensorflow as tf


def crop_and_resize_video(video_path, output_dir, target_format=(128, 128), relative_crop_displacement=0.0):
  assert -1 <= relative_crop_displacement <= 1
  assert os.path.isfile(video_path)
  assert os.path.isdir(output_dir)

  video_name = os.path.basename(video_path).replace('.mp4', '').replace('.avi', '')
  result_video_name = video_name + '_' + str(target_format[0]) + 'x' + str(target_format[1]) + '_' \
                      + str(relative_crop_displacement) + '.mp4'
  result_video_path = os.path.join(output_dir, result_video_name)

  clip = VideoFileClip(video_path)
  width, height = clip.size
  if width >= height:
    x1 = math.floor((width - height) / 2) + relative_crop_displacement * math.floor((width - height) / 2)
    y1 = None
    size = height
  else:
    x1 = None
    y1 = math.floor((height - width) / 2) + relative_crop_displacement * math.floor((height - width) / 2)
    size = width

  clip_crop = moviepy.video.fx.all.crop(clip, x1=x1, y1=y1, width=size, height=size)
  clip_resized = moviepy.video.fx.all.resize(clip_crop, newsize=target_format)

  clip_resized.write_videofile(result_video_path)


def get_ucf_video_category(taxonomy_list, label):
  """requires the ucf101 taxonomy tree structure (from json file) and the clip label (e.g. 'Surfing)
  in order to find the corresponding category (e.g. 'Participating in water sports)''"""
  return search_list_of_dicts(taxonomy_list, 'nodeName', label)[0]['parentName']


def search_list_of_dicts(list_of_dicts, dict_key, dict_value):
  """parses through a list of dictionaries in order to return an entire dict entry that matches a given value (dict_value)
  for a given key (dict_key)"""
  return [entry for entry in list_of_dicts if entry[dict_key] == dict_value]


def get_metadata_dict_as_bytes(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_filename_filetype_from_path(path):
  """extracts and returns both filename (e.g. 'video_1') and filetype (e.g. 'mp4') from a given absolute path"""
  filename = os.path.basename(path)
  video_id, filetype = filename.split(".")
  return video_id, filetype


def get_metadata_dict_entry(clip_dict_entry, taxonomy_list):
  """constructs and returns a dict entry consisting of the following entries (with UCF 101 example values):
  - label (Scuba diving)
  - category (Participating in water sports)
  - video_id (m_ST2LDe5lA_0)
  - filetype (mp4)
  - duration (12.627)
  - mode (training)
  - url (https://www.youtube.com/watch?v=m_ST2LDe5lA)"""
  meta_dict_entry = {}

  label = clip_dict_entry['label']
  video_path = clip_dict_entry['path']
  duration = clip_dict_entry['duration']
  url = clip_dict_entry['url']
  mode = clip_dict_entry['subset']
  video_id, filetype = get_filename_filetype_from_path(video_path)
  category = get_ucf_video_category(taxonomy_list, label)

  meta_dict_entry['label'] = label
  meta_dict_entry['category'] = category
  meta_dict_entry['video_id'] = video_id
  meta_dict_entry['filetype'] = filetype
  meta_dict_entry['duration'] = duration
  meta_dict_entry['mode'] = mode
  meta_dict_entry['url'] = url

  return meta_dict_entry

def create_ucf101_metadata_dicts(subclip_dict, database_dict):
  """construcs a list of dicts. Requires the json.load returns of the metadata files;
  single clip dict and database / taxonomy dict (e.g. UCF101: metadata_subclips.json and metadata.json"""
  taxonomy_list = database_dict['taxonomy']
  meta_dict = []

  for i, (key, clip) in enumerate(subclip_dict.items()):
    entry = get_metadata_dict_entry(clip, taxonomy_list)
    meta_dict.append(entry)

  return meta_dict



if __name__ == '__main__':
  # load subclip dict
  json_file_location = '/common/homes/students/rothfuss/Downloads/clips/metadata_subclips.json'
  json_file_location_taxonomy = '/common/homes/students/rothfuss/Downloads/metadata.json'
  output_dir = '/common/homes/students/rothfuss/Videos'
  categories = []
  with open(json_file_location) as file:
    subclip_dict = json.load(file)

  with open(json_file_location_taxonomy) as file:
    database_dict = json.load(file)

  metadata = create_ucf101_metadata_dicts(subclip_dict, database_dict)



  for i, (key, clip) in enumerate(subclip_dict.items()):

    label = clip['label']
    video_path = clip['path']
    crop_and_resize_video(video_path, output_dir)
    crop_and_resize_video(video_path, output_dir, relative_crop_displacement=-1)
    crop_and_resize_video(video_path, output_dir, relative_crop_displacement=1)

    if i > 2:
      break


