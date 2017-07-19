import os
from tensorflow.python.platform import gfile
import tensorflow as tf
import re
import random
import datetime as dt
import moviepy.editor as mpy
import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sn
from moviepy.editor import VideoFileClip


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
  if type == 'activity_net':
    p = re.compile('^([a-zA-Z0-9_-]+_[0-9]+)_\d{3}')
  elif type == 'youtube8m':
    p = re.compile('^([a-zA-Z0-9_-]+)_[0-9]+x[0-9]+')
  elif type == 'UCF101':
    p = re.compile('^([a-zA-Z0-9_-]+)_[0-9]+x[0-9]+')
  elif type == 'flyingshapes':
    video_id = video_name.split('_')[0]
    return video_id
  else: #just return filename without extension
    return video_name.replace('.avi', '').replace('.mp4', '')
  video_id = p.match(video_name).group(1)
  return video_id

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


def shuffle_files_in_list_from_categories(paths_list, categories, metadata_path, type='youtube8m', seed=5):
  """
  generates a list of randomly shuffled paths of the files contained in the provided directories which match at least one
  of the given categories from the 'categories' list. metadata (json file) must be provided to determine a file's category.
  :param paths_list: list with different path locations containing the files
  :param categories: list that works as a filter, e.g. ['Train'] only gives files that are of category train
  :param metadata_path: path to the json file (mostly provided by the dataset authors)
  :param type: specifies the dataset (e.g. 'UCF101')
  :return: returns two lists, one with the content of all the given directories in the provided order and another
  containing the same list randomly shuffled
  """

  assert paths_list is not None
  assert categories is not None
  assert os.path.isfile(metadata_path)

  with open(metadata_path) as file:
    metadata_file = json.load(file)

  all_files = []
  # first get all possible files
  for path_entry in paths_list:
    print(path_entry)
    all_files.extend(file_paths_from_directory(path_entry, '*.avi'))

  # then discard all files from the list not belonging to one of the given categories
  for file_path in all_files:
    file_prefix = get_video_id_from_path(file_path, type)
    value = next(v for (k, v) in metadata_file.items() if file_prefix + '.mp4' in k)
    file_category = []

    if value is not None and 'label' in value:
      file_category = value['label']

    if file_category not in categories or file_category is None:
      print(file_path + ' removed (category: ' + file_category + ')')
      all_files.remove(file_path)

  random.seed(a=seed)
  return all_files, random.sample(all_files, len(all_files))


def create_session_dir(output_dir): #TODO move to utils
  assert(output_dir)
  dir_name = str(dt.datetime.now().strftime("%m-%d-%y_%H-%M"))
  output_dir = os.path.join(output_dir, dir_name)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  print('Created custom directory for session:', dir_name)
  return output_dir


def create_subfolder(dir, name): #TODO move to utils
  subdir = os.path.join(dir, name)
  if not os.path.isdir(subdir):
    os.mkdir(subdir)
    print('Created subdir:', subdir)
    return subdir
  else:
    return os.path.join(dir, name)


def store_output_frames_as_gif(output_frames, labels, output_dir):
  """ Stores frame sequence produced by model as gif
    Args:
      output_frames:  list with Tensors of shape [batch_size, frame_height, frame_width, num_channels],
                      each element corresponds to one frame in the produced gifs
      labels:         list with video_id's of shape [batch_size, label]
      output_dir:     path to output directory
  """
  assert os.path.isdir(output_dir)
  batch_size = output_frames[0].shape[0]
  for i in range(batch_size): #iterate over validation instances
    clip_array = [bgr_to_rgb(frame[i,:,:,:]) for frame in output_frames]
    clip = mpy.ImageSequenceClip(clip_array, fps=10).to_RGB()
    clip.write_gif(os.path.join(output_dir, 'generated_clip_' + str(labels[i].decode('utf-8')) + '.gif'), program='ffmpeg')


def bgr_to_rgb(frame):
  blue_channel = frame[:,:,0]
  red_channel = frame[:,:,2]
  frame[:, :, 2] = red_channel
  frame[:, :, 0] = blue_channel
  return frame


def write_metainfo(output_dir, model, flags):
  with open(os.path.join(output_dir, 'metainfo.txt'), 'a') as f:
    f.write('\n' + '---- Training: ' + str(dt.datetime.now()) + ' ----' + '\n')
    f.write('model' + ':  ' + str(os.path.basename(model.__file__)) + '\n') #print model name
    for key, value in flags.__flags.items():
      f.write(str(key) + ':  ' + str(value) + '\n')


def store_latent_vectors_as_df(output_dir, hidden_representations, labels, metadata, filename = None):
  """" exports the latent representation of the last encoder layer (possibly activations of fc layer if fc-flag activated)
  and the video metadata as a pandas dataframe in python3 pickle format

  :param output_dir: the path where the pickle file should be stored
  :param hidden_representations: numpy array containing the activations
  :param labels: the corresponding video id's
  :param shapes: the corresponding shape of the object in the video
  :param filename: name of the pickle file - if not provided, a filename is created automatically

  Example shape of stored objects if no fc layer used
  (each ndarray)
  hidden repr file: shape(1000,8,8,16), each of 0..1000 representing the activation of encoder neurons
  labels file: shape(1000,), each of 0..1000 representing the video_id for the coresponding activations
  """

  # create 2 column df including hidden representations and labels/ids
  hidden_representations = [hidden_representations[i] for i in
                            range(hidden_representations.shape[0])]  # converts 2d ndarray to list of 1d ndarrays

  hidden_rep_df = pd.DataFrame({'label': labels, 'hidden_repr': hidden_representations})
  hidden_rep_df['label'] = hidden_rep_df['label'].map(lambda x: x.decode('utf-8'))

  # create dataframe from metadata
  f = lambda x: x.decode('utf-8')
  metadata_df = pd.DataFrame.from_dict([json.loads(f(e)) for e in list(metadata)], orient='columns')

  #merge dataframes to one
  df = pd.merge(hidden_rep_df, metadata_df, left_on='label', right_on='id')

  if not filename:
    filename = os.path.join(output_dir, 'metadata_and_hidden_rep_df_' + str(dt.datetime.now().strftime("%m-%d-%y_%H-%M-%S")) +'.pickle')
  df.to_pickle(filename)
  print("Dumped df pickle to", filename)
  return df


def store_encoder_latent_vector(output_dir, hidden_representations, labels, produce_single_files=True):
  """" exports the latent representation of the last encoder layer (possibly activations of fc layer if fc-flag activated)
  and the video labels that created the activations as npy files.

  :param output_dir: the path where the files should be stored
  :param hidden_representations: numpy array containing the activations
  :param labels: the corresponding video id's
  :param produce_single_files: export representations as single files

  Example shape of stored objects if no fc layer used
  (each ndarray)
  hidden repr file: shape(1000,8,8,16), each of 0..1000 representing the activation of encoder neurons
  labels file: shape(1000,), each of 0..1000 representing the video_id for the coresponding activations
  """
  assert os.path.isdir(output_dir) and hidden_representations.size > 0 and labels.size > 0, \
    'Storing latent representation failed: Output dir does not exist or latent vector or/and label vector empty'

  if produce_single_files:
    for single_rep_itr in range(hidden_representations.shape[0]):
      file_name = os.path.join(output_dir, labels[single_rep_itr].decode('utf-8'))
      np.save(file_name, hidden_representations[single_rep_itr])
  else:
    tag = str(dt.datetime.now().strftime("%m-%d-%y_%H-%M-%S"))
    file_name_hidden = os.path.join(output_dir, tag + '_hidden_repr')
    np.save(file_name_hidden, hidden_representations)

    file_name_label = os.path.join(output_dir, tag + '_label')
    np.save(file_name_label, labels)


def store_plot(output_dir, name1, name2="", name3="", suffix=".png"):
  assert output_dir is not None
  assert name1 is not None
  file_name = os.path.join(os.path.dirname(output_dir),
                                   name1 + name2 + name3 + suffix)

  plt.savefig(file_name, dpi=100)
  print('Dumped plot to:', file_name)


def export_plot_from_pickle(pickle_file_path, plot_options=((64, 64), 15, 15), show=False):
  """
  Loads a pickle file, generates a seaborn heatmap from its data and saves it to the dir of the specified pickle_file.

  :param pickle_file_path: the full path to the pickle file.
  :param plot_options: list of settings for matplotlib and seaborn. First list element specifies figure size as a
  tuple e.g. 64x64. Second list element specifies font_scale for seaborn as a single integer, e.g. 15. Third list
  element specifies annotation font size as a single integer, e.g. 15)
  :param show: defines wheter this function should also show the generated plot in the GUI.
  :return: the plot
  """
  assert os.path.isfile(pickle_file_path)
  df = pd.read_pickle(pickle_file_path)
  plt.figure(figsize=plot_options[0])
  sn.set(font_scale=plot_options[1])
  ax = sn.heatmap(df, annot=True, annot_kws={"size": plot_options[2]})

  if show:
    plt.show()

  heatmap_file_name = os.path.join(os.path.dirname(pickle_file_path),
                                   'pickle_heatmap_plot.png')
  plt.savefig(heatmap_file_name, dpi=100)

  return plt


def folder_names_from_dir(directory):
  return [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]


def folder_files_dict(directory):
  ff_dict = {}
  for folder in folder_names_from_dir(directory):
    folder_path = os.path.join(directory, folder)
    ff_dict[folder] = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
  return ff_dict


def video_length(video_path):
  clip = VideoFileClip(video_path)
  duration = clip.duration
  clip.__del__()
  return duration


def get_class_mapping(mapping_document_path):
  """returns a pandas data frame that specifies the mapping given by a csv document that defines an association.

  @:param mapping_document: the csv file, contains two columns (1st column specifies the dict key, 2nd specifies the dict value)"""
  df_mapping = pd.read_csv(mapping_document_path, sep=',', names=['subclass', 'class'])
  df_mapping.set_index('subclass')
  return df_mapping

def insert_general_classes_to_20bn_dataframe(dataframe_path, mapping_document_path):
  """According to a mapping (given by the 2nd argument) that specifies the relation 'subclass -> general class', 
  this function inserts a new column 'class' into the dataframe (given by the 1st argument) with the corresponding 
  class value from the mapping. 
  
  @:param dataframe_path: path to the dataframe
  @:param mapping_document_path: path to the mapping document (csv file) with two columns (subclass, class)

  """
  assert os.path.exists(dataframe_path), "invalid path to dataframe"
  assert os.path.exists(mapping_document_path), "invalid path to mapping document"
  dataframe = pd.read_pickle(dataframe_path)
  df_mapping = get_class_mapping(mapping_document_path)

  count = 0
  for index, row in dataframe.iterrows():
    # assuming the sub class label is named 'category' and mother category 'class'
    sub_label = row['category']
    result = df_mapping.loc[df_mapping['subclass']==sub_label, 'class']
    try:
     label = result.item()
    except ValueError:
     label = 'test'
     count += 1
     
    dataframe.loc[index, 'class'] = label
  print("%d times subclass could not be found in mapping" % (count)) 

  return dataframe
  
