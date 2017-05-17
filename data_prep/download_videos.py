import pafy, json, sys, os, os.path, moviepy, imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path
from moviepy.editor import *
import pandas as pd
from pprint import pprint
import numpy as np

TARGET_DIR = '/data/rothfuss/youtube-8m/videos'
CATEGORY_VIDEO_FILE = '/data/rothfuss/youtube-8m/category_video_df.pickle'
CATEGORY = 'Food & Drink'

def download_video(url, destination_directory, file_name=None):
  assert os.path.isdir(destination_directory)
  video = pafy.new(url)
  stream = select_stream(video)
  if file_name:
    file_path = stream.download(filepath=os.path.join(destination_directory, file_name + '.' + stream.extension))
  else:
    file_path = stream.download(filepath=destination_directory)
  return file_path

def download_youtube_videos(video_ids, destination_directory):
  files_in_destination_dir = os.listdir(destination_directory)
  n_videos = len(video_ids)
  for i, id in enumerate(video_ids):
    progress_str = ("[ %.2f" % ((i+1)/n_videos*100)) + "% ]"
    if not (id + '.mp4') in files_in_destination_dir:
      try:
        url = 'https://www.youtube.com/watch?v=' + str(id)
        download_video(url, destination_directory, file_name=id)
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


def main():
  video_df = pd.read_pickle(CATEGORY_VIDEO_FILE)
  video_ids = list(video_df.index[video_df[CATEGORY] == 1.0])

  download_youtube_videos(video_ids[:20000], TARGET_DIR)

if __name__ == '__main__':
  main()