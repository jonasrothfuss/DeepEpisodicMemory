import math, json, os.path, moviepy, os, multiprocessing, itertools
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
from utils import io_handler
from tensorflow.python.platform import gfile
import random, multiprocessing, subprocess, scenedetect
import pandas as pd
from pprint import pprint
from joblib import Parallel, delayed
import traceback
import shutil


NUM_CORES = multiprocessing.cpu_count()

CSV_TRAIN_20BN = '/PDFData/rothfuss/data/20bn-something/something-something-v1-train_test.csv'
CSV_VALID_20BN = '/PDFData/rothfuss/data/20bn-something/something-something-v1-validation.csv'

DATASETS = ['youtube8m', 'UCF101']
NUM_TIME_CROPS = 1

NUM_CORES = multiprocessing.cpu_count()

def crop_and_resize_video_clip(video_path=None, video_file_clip=None, target_format=(128, 128), relative_crop_displacement=0.0):
  """
  :param video_path: specifies the full (absolute) path to the video file
  :param video_file_clip: provide a moviepy.VideoFileClip if this should be used instead of the video specified by video_path
  :param target_format: a tuple (width, height) specifying the dimensions of the returned video
  :param relative_crop_displacement: augmentation parameter, adjusts the clipping in either
  y (when width < height) or x (when width >= height) dimension
  :return: returns the cropped and resized VideoFileClip instance
  """

  assert video_path or video_file_clip
  assert (-1 <= relative_crop_displacement <= 1), "relative_crop_displacement must be in interval [0,1]"

  if video_file_clip:
    clip = video_file_clip
  else:
    assert os.path.isfile(video_path), "video_path must be a file"
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
  return moviepy.video.fx.all.resize(clip_crop, newsize=target_format)

def get_activity_net_video_category(taxonomy_list, label):
  """requires the ucf101 taxonomy tree structure (from json file) and the clip label (e.g. 'Surfing)
  in order to find the corresponding category (e.g. 'Participating in water sports)''"""
  return search_list_of_dicts(taxonomy_list, 'nodeName', label)[0]['parentName']

def search_list_of_dicts(list_of_dicts, dict_key, dict_value):
  """parses through a list of dictionaries in order to return an entire dict entry that matches a given value (dict_value)
  for a given key (dict_key)"""
  return [entry for entry in list_of_dicts if entry[dict_key] == dict_value]

def get_metadata_dict_entry(subclip_dict_entry, taxonomy_list=None, type='activity_net'):
  """constructs and returns a dict entry consisting of the following entries (with UCF 101 example values):
  - label (Scuba diving)
  - category (Participating in water sports)
  - video_id (m_ST2LDe5lA_0)
  - filetype (mp4)
  - duration (12.627)
  - mode (training)
  - url (https://www.youtube.com/watch?v=m_ST2LDe5lA)"""
  meta_dict_entry = {}
  assert type in DATASETS
  assert taxonomy_list or not type=='activity_net'

  label = subclip_dict_entry['label']
  video_path = subclip_dict_entry['path']
  duration = subclip_dict_entry['duration']
  url = subclip_dict_entry['url']

  video_id, filetype = io_handler.get_filename_and_filetype_from_path(video_path)

  meta_dict_entry['label'] = str(label)
  meta_dict_entry['video_id'] = str(video_id)
  meta_dict_entry['filetype'] = str(filetype)
  meta_dict_entry['duration'] = str(duration)
  meta_dict_entry['url'] = str(url)

  if type is 'activity_net':
    meta_dict_entry['category'] = str(get_activity_net_video_category(taxonomy_list, label))
    meta_dict_entry['mode'] = str(subclip_dict_entry['subset'])

  return meta_dict_entry

def create_activity_net_metadata_dicts(video_dir, subclip_json_file_location, json_file_location_taxonomy, file_type="*.avi"):
  """
  construcs a list of dicts. Requires the json.load returns of the metadata files;
  single clip dict and database / taxonomy dict (e.g. UCF101: metadata_subclips.json and metadata.json

  :param: video_dir: path to directory containing the video files
  :param: subclip_json_file_location: path to json file with meta information about the subclips in video_dir
  :param: json_file_location_taxonomy: path to json file containing the ucf101 class hirarchy
  :param: file_type: postfix of video files in video_dir -> indicates the video format used
  :returns:
  """
  assert os.path.isdir(video_dir)
  assert os.path.isfile(subclip_json_file_location) and os.path.isfile(json_file_location_taxonomy)
  assert file_type in ["*.avi", "*.mp4"]

  filenames = io_handler.files_from_directory(video_dir, file_type)

  if not filenames:
    raise RuntimeError('No data files found.')

  # load dictionaries
  with open(json_file_location_taxonomy) as file:
    database_dict = json.load(file)

  with open(subclip_json_file_location) as file:
    subclip_dict = json.load(file)


  taxonomy_list = database_dict['taxonomy']
  meta_dict = {}

  for i, video_path in enumerate(filenames):
    video_id = io_handler.get_video_id_from_path(video_path, 'activity_net')
    subclip_dict_entry = subclip_dict[video_id + ".mp4"]
    new_entry = get_metadata_dict_entry(subclip_dict_entry, taxonomy_list)
    meta_dict.update({video_id : new_entry})

  assert len(meta_dict) == len(set([io_handler.get_video_id_from_path(video_path, 'UCF101') for video_path in filenames]))
  return meta_dict

def create_youtube8m_metadata_dicts(video_dir, metadata_json_file, file_type="*.avi"):
  """
  construcs a list of metadata dicts

  :param: video_dir: path to directory containing the video files
  :param: metadata_json_file: path to json file with meta information about the clips in video_dir
  :param: file_type: postfix of video files in video_dir -> indicates the video format used
  :returns:
  """
  assert os.path.isdir(video_dir) and os.path.isfile(metadata_json_file)
  assert file_type in ["*.avi", "*.mp4"]

  filenames = io_handler.files_from_directory(video_dir, file_type)

  if not filenames:
    raise RuntimeError('No data files found.')

  with open(metadata_json_file) as file:
    clip_dict = json.load(file)

  meta_dict = {}

  for i, video_path in enumerate(filenames):
    video_id = io_handler.get_video_id_from_path(video_path, 'youtube8m')
    subclip_dict_entry = clip_dict[video_id + ".mp4"]
    new_entry = get_metadata_dict_entry(subclip_dict_entry, type='youtube8m')
    meta_dict.update({video_id: new_entry})

  assert len(meta_dict) == len(
    set([io_handler.get_video_id_from_path(video_path, 'youtube8m') for video_path in filenames]))
  return meta_dict

def create_ucf101_metadata_dicts(video_dir, metadata_json_file, file_type="*.avi"):
  """
  construcs a list of metadata dicts

  :param: video_dir: path to directory containing the video files
  :param: metadata_json_file: path to json file with meta information about the clips in video_dir
  :param: file_type: postfix of video files in video_dir -> indicates the video format used
  :returns:
  """
  assert os.path.isdir(video_dir) and os.path.isfile(metadata_json_file)
  assert file_type in ["*.avi", "*.mp4"]

  filenames = io_handler.files_from_directory(video_dir, file_type)

  if not filenames:
    raise RuntimeError('No data files found.')

  with open(metadata_json_file) as file:
    clip_dict = json.load(file)

  meta_dict = {}

  for i, video_path in enumerate(filenames):
   video_id = io_handler.get_video_id_from_path(video_path, 'UCF101')
   subclip_dict_entry = clip_dict[video_id]
   subclip_dict_entry.update({'label': subclip_dict_entry['category'], 'filetype': 'avi', 'video_id': video_id })
   meta_dict.update({video_id: subclip_dict_entry})

  assert len(meta_dict) == len(
    set([io_handler.get_video_id_from_path(video_path, 'youtube8m') for video_path in filenames]))
  return meta_dict

def create_20bn_metadata_dicts(video_dir, csv_file, file_type="*.avi"):
  """
  construcs a list of metadata dicts

  :param: video_dir: path to directory containing the video files
  :param: csv_file: path to csv which list the labels correpsonding to the video ids
  :param: file_type: postfix of video files in video_dir -> indicates the video format used
  :returns:
  """
  assert os.path.isdir(video_dir) and os.path.isfile(csv_file)
  assert file_type in ["*.avi", "*.mp4"]

  filenames = io_handler.files_from_directory(video_dir, file_type)
  assert filenames, 'There must be %s files in the provided directory' % file_type
  # valid csv files are separated by ; whereas train csv files are by ','
  separator = ',' if 'valid' in csv_file else ';'
  label_id_df = pd.read_csv(csv_file, sep=separator, names=['video_id', 'label'])

  label_id_df = label_id_df.set_index('video_id')

  meta_dict = {}
  for i, video_path in enumerate(filenames):
    video_id = io_handler.get_video_id_from_path(video_path, type='20bn_valid')
    try:
      label = label_id_df.loc[int(video_id)]['label']
      subclip_dict_entry = {'label': label, 'filetype': 'avi', 'video_id': video_id, 'category': label}
      meta_dict.update({video_id: subclip_dict_entry})
    except Exception as e:
      print(e)
  return meta_dict

def extract_subvideo(video_path, target_time_interval=(1, 4)):
  """ Returns an instance of VideoFileClip of the initial video with the content between times start and end_time.
  In the case end_time exceeds the true end_time of the clip the entire clip starting from start_time is returned.
  Should the video's duration be smaller than the given start_time, the original video is returned immediately without
  trimming. Also, should the specified subvideo length (end_time - start_time) exceed the video duration, the original
  video is returned.

  :param video_path: specifies the full (absolute) path to the video file
  :param target_time_interval(x,y): x: start time in s (e.g. 6.5) y: end time in s (e.g. 6.5)
  :return: the trimmed sub video (VideoFileClip)
  """

  start_time = target_time_interval[0]
  end_time = target_time_interval[1]

  assert os.path.isfile(video_path), "video_path does not contain a file"
  assert start_time < end_time, "invalid target time interval - start_time must be smaller than end_time"

  clip = VideoFileClip(video_path)

  assert end_time < clip.duration, "video to short to crop (duration=%.3f, end_time=%.3f)" % (clip.duration, end_time)

  sub_clip = clip.subclip(start_time, end_time)

  assert abs(sub_clip.duration - end_time + start_time) < 0.001 # ensure that sub_clip has desired length
  # returning both for killing since ffmpeg implementation produces zombie processes
  return sub_clip

def prepare_and_store_video(source_video_path, output_dir, target_time_interval, target_format=(128,128), relative_crop_displacement=0.0):
  sub_clip = None
  source_video_name = os.path.basename(source_video_path).replace('.mp4', '').replace('.avi', '')

  assert isinstance(target_time_interval, tuple), "provided target_time_interval is not a tuple"
  # do time trimming, else leave video at original length
  sub_clip = extract_subvideo(source_video_path, target_time_interval=(target_time_interval[0], target_time_interval[1]))
  assert sub_clip is not None

  if sub_clip is not None and sub_clip.duration < (target_time_interval[1] - target_time_interval[0]):
    # skip video if it is shorter than specified
    print('Video too short, skipping.')
    kill_process(sub_clip)
    subprocess.call(["pkill -9 -f " + source_video_path], shell=True)
    return None

  target_video_name = generate_video_name(source_video_name, target_format, relative_crop_displacement, target_time_interval)
  target_video_path = os.path.join(output_dir, target_video_name)

  clip_resized = crop_and_resize_video_clip(source_video_path, video_file_clip=sub_clip, target_format=target_format,
                                       relative_crop_displacement=relative_crop_displacement)

  print("writing video: " + target_video_path)
  clip_resized.write_videofile(target_video_path, codec='rawvideo', progress_bar=False, verbose=False)
  kill_process(clip_resized)

def generate_video_name(source_video_name, target_format, relative_crop_displacement, time_interval):
  return source_video_name + '_' + str(target_format[0]) + 'x' + str(target_format[1]) + '_' \
                      + str("%.2f" % relative_crop_displacement) + '_' + "(%.1f,%.1f)" % time_interval\
                      + '.avi'

def get_metadata_dict(input_dir, output_dir, subclip_json_file_location, type):
  if type is 'json':
    # load dictionaries
    with open(subclip_json_file_location) as file:
      metadata_dict = json.load(file)

  elif type is 'folders':
    if os.path.isfile(os.path.join(output_dir, 'metadata.json')):
      with open(os.path.join(output_dir, 'metadata.json'), 'r')as f:
        metadata_dict = json.load(f)
      print('Loaded metadata dict from:', os.path.join(output_dir, 'metadata.json'))

    else:
      # generate metadata dict for videos
      ff_dict = io_handler.folder_files_dict(input_dir)
      metadata_dict = {}

      for i, (category, videos) in enumerate(ff_dict.items()):
        print(i, category)
        for video in videos:
          video_path = os.path.join(input_dir, category, video)
          duration = io_handler.video_length(video_path)
          metadata_dict[video.replace('.avi', '')] = {'category': category, 'path': video_path, 'duration': duration}

      # dump metadata dict as json to output dir
      with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata_dict, f)
      print('Dumped metadata dict to:', os.path.join(output_dir, 'metadata.json'))
  return metadata_dict

def generate_video_snippets(i, key, clip, file_names, output_dir, target_format, num_clips):
  if not key[0] == '-':  # file system cannot handle filenames that start with '-'
    try:
      video_path, duration = clip['path'], clip['duration']
      video_name = os.path.basename(video_path).replace('.mp4', '').replace('.avi', '')
      if any(str(video_name) in x for x in file_names):
        print("Skipping video (already_exists): " + video_name)
      else:
        interval_suggestions = video_time_interval_suggestions(duration, max_num_suggestions=NUM_TIME_CROPS)
        if len(interval_suggestions) == 1:
          num_random_crops = 4
        elif len(interval_suggestions) == 2:
          num_random_crops = 1
        else:
          num_random_crops = 1

        for time_interval in interval_suggestions:
          for _ in range(num_random_crops):
            sample_rel_crop_displacement = random.uniform(-0.7, 0.7)
            try:
              prepare_and_store_video(video_path, output_dir, target_time_interval=time_interval,
                                      relative_crop_displacement=sample_rel_crop_displacement,
                                      target_format=target_format)
            except Exception as e:
              print('Failed to process video (' + str(video_path) + ') ---' + str(e))
            finally:
              subprocess.call(["pkill -9 -f " + video_path], shell=True)

        print('[%d of %d]  ' % (i, num_clips) + 'Successfully processed video (' + str(video_path) + ')')
    except Exception as e:
      print('[%d of %d]  ' % (i, num_clips) + 'Failed to process video (' + str(video_path) + ') ---' + str(e))

def prepare_and_store_all_videos(output_dir, subclip_json_file_location=None, input_dir=None, target_format=(128,128), type='json'):
  assert type in ['json', 'folders']
  assert (type != 'json' or subclip_json_file_location), 'If type is json, subclip_json_file_location must be provided'
  assert (type != 'folders' or input_dir), 'If type is folders, input_dir must be specified'

  metadata_dict = get_metadata_dict(input_dir, output_dir, subclip_json_file_location, type)

  #get files that were already processed
  file_names = io_handler.files_from_directory(output_dir, '*.avi')

  num_clips = len(metadata_dict)

  Parallel(n_jobs=NUM_CORES)(
    delayed(generate_video_snippets)(i, key, clip, file_names, output_dir, target_format, num_clips)
    for i, (key, clip) in enumerate(metadata_dict.items()))

def kill_process(process):
  try:
    if hasattr(process, 'reader'):
      process.reader.close()
    if hasattr(process, 'audio'):
      process.audio.reader.close_proc()
    process.__del__()
  except:
    pass

def video_time_interval_suggestions(video_duration, max_num_suggestions=4):
  """
  :param video_length: duration of video in seconds
  :return: array of tuples representing time intervals [(t1_start, t1_end), (t2_start, t2_end), ...]
  """
  assert (video_duration > 3), "video too short to crop (duration < 3 sec)"
  suggestions = []
  if video_duration < 4:
    margin = (4-video_duration)/2
    suggestions.append((margin, 4-margin))
  elif video_duration < 5:
    suggestions.append((1, 4))
  else:
    num_suggestuions = min(max_num_suggestions, int((video_duration-2)//2.5))
    left_margin, right_margin= 1, video_duration - 1
    for i in range(num_suggestuions):
      offset = (video_duration-2)/num_suggestuions * i
      suggestions.append((left_margin + offset, left_margin + offset + 3))

  assert len(suggestions) > 0
  assert all([t_end <= video_duration and abs(t_end - t_start - 3) < 0.001 for t_start, t_end in suggestions])

  return suggestions

def crap_detect(video_dir):
  detector_list = [
    scenedetect.detectors.ThresholdDetector(threshold=16, min_percent=0.6)
  ]

  file_paths = gfile.Glob(os.path.join(video_dir, '*.avi'))
  l = []
  for file_path in file_paths:
    try:
      print(file_path)
      scene_list = []
      video_framerate, frames_read = scenedetect.detect_scenes_file(
        file_path, scene_list, detector_list)

      # scene_list now contains the frame numbers of scene boundaries.
      print(l)
      if len(scene_list) >= 1:
        l.append(file_path)
    except:
      pass

def convert_frames_to_avi(frames_dir, output_dir, image_type='.jpg'):
  """ converts a folder with images to an avi file"""
  file_names = sorted((os.path.join(frames_dir, fn) for fn in os.listdir(frames_dir) if fn.endswith(image_type)))
  #images = [imageio.imread(fn) for fn in file_names]
  filename = os.path.join(output_dir, os.path.basename(frames_dir) + '.avi')
  #imageio.mimsave(filename, images)
  clip = ImageSequenceClip(file_names, fps=24)
  clip = crop_and_resize_video_clip(video_file_clip=clip,
                                       target_format=(128,128),
                                       relative_crop_displacement=0.0)
  clip.write_videofile(filename, codec='rawvideo', verbose=False, progress_bar=False)

def video_id_train_val_dict(csv_file_train, csv_file_val):
  train_ids = pd.read_csv(csv_file_train, names=['video_id', 'label'])['video_id']
  valid_ids = pd.read_csv(csv_file_val, names=['video_id', 'label'])['video_id']
  return dict([(int(id), 'train') for id in train_ids]+[(int(id), 'valid') for id in valid_ids])

def process_folder_with_frames(i, frames_dir, target_dir, id_dict, num_dirs):
  video_id = int(os.path.basename(frames_dir))
  output_dir = os.path.join(target_dir, 'train') if id_dict[int(video_id)] == 'train' else os.path.join(target_dir, 'valid')
  convert_frames_to_avi(frames_dir, os.path.join(target_dir, output_dir))
  print('%i [%.2f %%]: Converted' % (i, (i / float(num_dirs)) * 100), frames_dir)

def convert_20bn_dataset_to_videos(source_dir, target_dir):
  already_converted_ids = [fn.replace('.avi', '') for fn in os.listdir(os.path.join(target_dir, 'train')) + os.listdir(os.path.join(target_dir, 'valid'))]
  frames_directories = [os.path.join(source_dir, fn) for fn in os.listdir(source_dir) if fn not in already_converted_ids]
  num_dirs = len(frames_directories)
  id_dict = video_id_train_val_dict(CSV_TRAIN_20BN, CSV_VALID_20BN)

  Parallel(n_jobs=NUM_CORES)(
    delayed(process_folder_with_frames)(i, frames_dir, target_dir, id_dict, num_dirs) for i, frames_dir in enumerate(frames_directories))

def split_ucf_in_train_valid(source_dir, train_list, test_list):
  train_videos = pd.read_csv(train_list, sep=' ', names=['vid_name', 'label'])['vid_name']
  valid_videos = pd.read_csv(test_list, names=['vid_name'])['vid_name']
  train_val_dict = {}
  for v in train_videos:
    train_val_dict[v.split('/')[1].replace('.avi', '')] = 'train'
  for v in valid_videos:
    train_val_dict[v.split('/')[1].replace('.avi', '')] = 'valid'
  videos = io_handler.files_from_directory(source_dir, file_type='*.avi')

  for i, v in enumerate(videos):
    v_id = io_handler.get_video_id_from_path(v, type='UCF101')
    os.rename(os.path.join(source_dir, v), os.path.join(os.path.join(source_dir, train_val_dict[v_id]), v))
    print(i, '- moved ', v_id)

def select_subset_from_20bn(source_dir, goal_dir, csv_file, classes):
  df = pd.read_csv(csv_file, sep=',', names=['video_id', 'label'])
  print(df)
  video_count = 0
  for id, label in zip(df['video_id'], df['label']):
    if label in classes:
      video_count += 1
      try:
        shutil.copy2(os.path.join(source_dir, str(id) + '.avi'), os.path.join(goal_dir, str(id) + '.avi'))
        print('Successfully copied ', str(id) + '.avi')
      except:
        print('Failed to copy ', str(id) + '.avi')
  print(video_count) 

def image_to_stationary_avi(image_file, output_dir, n_frames=20):
  image_files = [image_file for _ in range(n_frames)]
  filename = os.path.join(output_dir, os.path.basename(image_file).replace('.bmp','').replace('.jpg','') + '.avi')
  clip = ImageSequenceClip(image_files, fps=24)
  clip = moviepy.video.fx.all.crop(clip, x_center=320, y_center=240,
       width=380, height=380)
  clip = crop_and_resize_video_clip(video_file_clip=clip,
                                       target_format=(128,128),
                                       relative_crop_displacement=0.0)
  clip.write_videofile(filename, codec='rawvideo', verbose=False, progress_bar=False)

def generate_stationary_videos_from_dir(images_dir, output_dir, n_frames=20, image_type='.bmp'):
  file_names = (os.path.join(images_dir, fn) for fn in os.listdir(images_dir) if fn.endswith(image_type))
  for image_file in file_names:
    print(image_file)
    image_to_stationary_avi(image_file, output_dir, n_frames=n_frames)




def main():
  # load subclip dict
  json_file_location = '/PDFData/rothfuss/data/youtube8m/videos/pc031/metadata.json'
  #json_file_location_taxonomy = '/common/homes/students/rothfuss/Downloads/metadata.json'
  #output_dir = '/data/rothfuss/data/ucf101_prepared_videos/'
  input_dir = '/common/homes/students/rothfuss/Downloads/UCF-101'
  output_dir = '/PDFData/rothfuss/data/UCF101/prepared_videos'

  #frames_dir = '/PDFData/rothfuss/data/20bn-something-something-v1'
  #traget_dir = '/PDFData/rothfuss/data/20bn-something/videos'

  #convert_20bn_dataset_to_videos(frames_dir, traget_dir)

  #prepare_and_store_all_videos(output_dir,input_dir=input_dir, subclip_json_file_location=json_file_location, type='folders')

  source_dir= '/PDFData/rothfuss/data/20bn-something/videos/valid'
  goal_dir = '/PDFData/rothfuss/data/20bn-something/selected_subset_10classes_eren/videos_valid'
  # classes = [
  #   'Moving something up', #replaced Folding something
  #   'Throwing something', #replaced Covering something
  #   'Turning something upside down', #replaced Hitting something with something
  #   'Pulling something from left to right',
  #   'Pulling something from right to left',
  #   'Moving something and something away from each other', #replaced seperates into two pieces
  #   'Taking something from somewhere', #replaced pushing something off of something
  #   'Moving something and something closer to each other', #replaced Plugging something into something
  #   'Turning something upside down', #replaced Closing something
  #   'Moving something up' #replaced Tipping something over
  # ]
  classes = [
    'Lifting a surface with something on it but not enough for it to slide down',
    'Lifting a surface with something on it until it starts sliding down',
    'Lifting something up completely',
    'Lifting something up completely without letting it drop down',
    'Lifting something with something on it',
    'Lifting up one end of something',
    'Lifting up one end of something without letting it drop down'
    'Moving part of something'
    'Moving something across a surface until it falls down',
    'Moving something across a surface without it falling down',
    'Moving something and something away from each other',
    'Moving something and something closer to each other',
    'Moving something and something so they collide with each other',
    'Moving something and something so they pass each other',
    'Moving something away from something',
    'Moving something closer to something',
    'Moving something down',
    'Moving something up'
    'Poking a stack of something so the stack collapses',
    'Poking something so it slightly moves',
    'Poking something so that it falls over',
    'Poking something so that it spins around'
    'Pouring something into something',
    'Pouring something into something until it overflows',
    'Pouring something out of something'
    'Pulling something from left to right',
    'Pulling something from right to left'
    'Pulling something onto something',
    'Pushing something from left to right',
    'Pushing something from right to left',
    'Pushing something off of something',
    'Pushing something onto something',
    'Pushing something so it spins',
    'Pushing something so that it falls off the table',
    'Pushing something so that it slightly moves',
    'Pushing something with something'
    'Putting number of something onto something',
    'Putting something',
    'Putting something into something',
    'Putting something next to something',
    'Putting something on a flat surface without letting it roll',
    'Putting something on a surface',
    'Putting something on the edge of something so it is not supported and falls down',
    'Putting something onto something'
    'Putting something onto something else that cannot support it so it falls down',
    'Putting something similar to other things that are already on the table',
    'Putting something that cannot actually stand upright upright on the table',
    'Putting something upright on the table'
    'Throwing something',
    'Throwing something in the air and catching it',
    'Throwing something in the air and letting it fall',
    'Throwing something onto a surface'
    'Squeezing something'
    'Taking one of many similar things on the table',
    'Taking something from somewhere',
    'Taking something out of something'
  ]



  with open('/PDFData/rothfuss/data/20bn-something/selected_subset_10classes_eren/classes.json', 'w') as f:
    json.dump(classes, f)


  #select_subset_from_20bn(source_dir, goal_dir, CSV_VALID_20BN, classes)

  #generate_stationary_videos_from_dir('/common/temp/toEren/4PdF_ArmarSampleImages/input/', '/common/temp/toEren/4PdF_ArmarSampleImages/stationary_image_videos_cropped')

if __name__ == '__main__':
 main()
