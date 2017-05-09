import math, pafy, json, sys, os, os.path, moviepy, imageio, os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path
from moviepy.editor import *

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


if __name__ == '__main__':
  # load subclip dict
  json_file_location = '/common/homes/students/rothfuss/Downloads/clips/metadata_subclips.json'
  output_dir = '/common/homes/students/rothfuss/Videos'
  with open(json_file_location) as file:
    subclip_dict = json.load(file)
  for i, (key, clip) in enumerate(subclip_dict.items()):
    video_path = clip['path']
    crop_and_resize_video(video_path, output_dir)
    crop_and_resize_video(video_path, output_dir, relative_crop_displacement=-1)
    crop_and_resize_video(video_path, output_dir, relative_crop_displacement=1)
    if i > 2:
      break