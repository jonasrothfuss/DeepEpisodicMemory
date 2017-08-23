import os, subprocess
import utils.io_handler as io
import itertools

AVI_SOURCE_DIR = '/data/rothfuss/data/ArmarExperiences/videos/memory'
TARGET_DIR = '/data/rothfuss/data/ArmarExperiences/videos/memory_augmented'

GAMMA_VALS = [0.7, 1.0]
BRIGHTNESS_VALS = [-0.1, 0.0, 0.2]
SATURATION = [0.6, 1.0]


avi_files = io.file_paths_from_directory(AVI_SOURCE_DIR, '*.avi')

def augment_video(video_path, target_dir, gamma, brightness, saturation, i):
  video_id = os.path.basename(video_path).split('.')[0]
  new_video_path = os.path.join(target_dir, video_id + '_%i.avi'%i)
  ffmpeg_str = "/common/homes/students/rothfuss/Documents/ffmpeg/bin/ffmpeg -i %s -vf eq=gamma=%.1f:brightness=%.1f:saturation=%.1f -c:a copy %s"%(video_path, gamma, brightness, saturation, new_video_path)
  print(ffmpeg_str)
  subprocess.Popen(ffmpeg_str, shell=True)


for file in avi_files:
  for i, (gamma, brightness, saturation) in enumerate(itertools.product(GAMMA_VALS, BRIGHTNESS_VALS, SATURATION)):
    try:
      augment_video(file, TARGET_DIR, gamma, brightness, saturation, i)
      print("%i: "%i +  "Sucessfully augmented " + file)
    except Exception as e:
      print("%i: "%i + "Could not augment " + file + " --- " + str(e))
