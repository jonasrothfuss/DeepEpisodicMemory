import pafy, json, sys, os, os.path, moviepy, imageio
import pandas as pd
from pprint import pprint
import numpy as np
import random
import requests
from collections import defaultdict
from io import StringIO
from data_prep.activity_net_download import meta_already_downloaded
import glob

TARGET_DIR = '/data/rothfuss/youtube-8m/videos'
CATEGORY_VIDEO_FILE = '/data/rothfuss/youtube-8m/category_video_df.pickle'
CATEGORY = 'Food & Drink'
CSV_PREFIX = "http://www.yt8m.org/csv"

def download_video(url, destination_directory, file_name=None):
  assert os.path.isdir(destination_directory)
  video = pafy.new(url)
  duration = 3600 * int(str(video.duration)[0:2]) + 60 * int(str(video.duration)[3:5]) + int(str(video.duration)[6:8])
  stream = select_stream(video)
  if file_name:
    file_path = stream.download(filepath=os.path.join(destination_directory, file_name))
  else:
    file_path = stream.download(filepath=destination_directory)
  return file_path, duration

def download_youtube_videos(video_ids, destination_directory):
  files_in_destination_dir = os.listdir(destination_directory)
  n_videos = len(video_ids)
  for i, id in enumerate(video_ids):
    progress_str = ("[ %.2f" % ((i+1)/n_videos*100)) + "% ]"
    if not (id + '.mp4') in files_in_destination_dir:
      try:
        url = 'https://www.youtube.com/watch?v=' + str(id)
        file_path, duration = download_video(url, destination_directory, file_name=id)
        print(progress_str, 'Successfully downloaded: ', id)
      except Exception as e:
        print(progress_str, 'Failed to download video: ', id, ' --- ', str(e))
    else:
      print(progress_str, 'Video already downloaded: ', id)

def select_stream(video, desired_quality='640x360'):
  mp4_streams = [s for s in video.streams if s.extension == 'mp4']
  if desired_quality in [s.resolution for s in mp4_streams]:
    return [s for s in mp4_streams if s.resolution == desired_quality][0]
  else:
    raise Exception('No mp4 stream available for ' + video.videoid)

def get_youtube8m_category_df():
  csv_string = requests.get('https://research.google.com/youtube8m/csv/vocabulary.csv').content
  csv_io = StringIO(csv_string.decode('utf-8'))
  df = pd.read_csv(csv_io, sep=",", index_col=0)
  return df

def get_youtube_ids_of_category(category):
  category_df = get_youtube8m_category_df()
  assert category in list(category_df['Name'])
  category_id = str(category_df['KnowledgeGraphId'].loc[category_df['Name'] == category].values[0])
  jsurl = str(CSV_PREFIX + '/j/' + category_id.split("/")[-1] + '.js')
  r = requests.get(jsurl)
  idlist = r.content.decode('utf-8').split("\"")[3]
  ids = [n for n in idlist.split(";") if len(n) > 3]
  return ids

def get_video_id_category_tuples(categoeries, max_num_videos_per_cat=3000):
  video_id_category_tuples = []
  for category in categoeries:
    ids = get_youtube_ids_of_category(category)
    if len(ids) > max_num_videos_per_cat:
      ids = random.sample(ids, max_num_videos_per_cat)
    video_id_category_tuples.extend([(id, category)for id in ids])
  return video_id_category_tuples

def remove_downloaded_videos_from_tuple_list(video_id_category_tuples, metadata_dict):
  if len(metadata_dict) == 0:  # no videos downloaded yet
    return video_id_category_tuples
  else:
    download_dict = dict(video_id_category_tuples)
    for video_name in metadata_dict.keys():
      if video_name.replace('.mp4','') in download_dict.keys():
        print(video_name.replace('.mp4',''))
        del download_dict[video_name.replace('.mp4','')]
    return download_dict.items()

def download_youtube_categories(categoeries, destination_directory, metadata_file_name="metadata.json",
                                max_num_videos_per_cat=3000):
  #prepare data to download
  video_id_category_tuples = get_video_id_category_tuples(categoeries, max_num_videos_per_cat=max_num_videos_per_cat)

  success_count, fail_count = 0, 0
  metadata_dict = meta_already_downloaded(destination_directory, metadata_file_name)

  video_id_category_tuples = remove_downloaded_videos_from_tuple_list(video_id_category_tuples, metadata_dict)

  video_count = len(video_id_category_tuples)
  for video_id, category in video_id_category_tuples:
    try:
      # attempt to download and store video
      file_name = str(video_id)+'.mp4'
      file_path, duration = download_video(video_id, destination_directory, file_name=file_name)
      # extend metadata dict
      metadata_dict[file_name] = {'path': file_path, 'label': category, 'duration': duration,
                                  'url': 'https://www.youtube.com/watch?v=' + video_id}

      # report success
      print(str(success_count + fail_count) + ' of ' + str(video_count) + ': ' + file_path)
      success_count += 1

    except Exception as e:
      fail_count += 1
      try:
        print(str(success_count + fail_count) + ' of ' + str(video_count) + ': Failed to download video:' + str(e))
      except:
        print(str(success_count + fail_count) + ' of ' + str(video_count) + ': Failed to download video:')

    # dump metadata dict as json file
    if (success_count % 20 == 0) or (success_count + fail_count == video_count):
      with open(os.path.join(destination_directory, metadata_file_name), 'w') as f:
        json.dump(metadata_dict, f)
      print('Dumped metadata file to: ' + str(destination_directory + metadata_file_name))
      # dump metadata dict as json file
  return metadata_dict


def main():
  categories_to_download = ['Dressage', 'Surfing', 'Roasting', 'Basketball', 'Skateboard', 'Piano', 'Dish (food)',
                            'Train', 'Airplane', 'Acoustic guitar', 'Roasting', 'Cooking show']
  download_youtube_categories(categories_to_download, '/common/temp/PdfData...')



  #video_df = pd.read_pickle(CATEGORY_VIDEO_FILE)
  #video_ids = list(video_df.index[video_df[CATEGORY] == 1.0])

  #download_youtube_videos(video_ids[:20000], TARGET_DIR)

if __name__ == '__main__':
  main()
