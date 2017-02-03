import pafy, json, sys, os, os.path, moviepy, imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path
from moviepy.editor import *

#returns activity net database as dict given the corresponding json file
def activity_net_db_from_json(json_file_location):
    assert os.path.isfile(json_file_location)
    with open(json_file_location) as file:
        activity_net_db = json.load(file)['database']
    return activity_net_db

# extracts and saves subclip from video file starting at t1 and ending at t2
# t1 and t2 in seconds
def extract_subclip(file_location, t1, t2, target_location):
    try:
        #moviepy.video.io.ffmpeg_tools.ffmpeg_extract_subclip(file_location, t1, t2, target_location)
        video = VideoFileClip(file_location).subclip(t1, t2).write_videofile(target_location)
    except imageio.core.NeedDownloadError:
        imageio.plugins.ffmpeg.download()


def download_activity_net(activity_net_dict, destination_directory, metadata_file_name = "metadata.json"):
    success_count = 0
    fail_count = 0
    video_count = len(activity_net_dict.keys())
    metadata_dict = {}
    for video_key, video_meta in activity_net_dict.items():
        try:
            #attempt to download and store video
            file_path = download_video(video_meta['url'], destination_directory, file_name=str(video_key))

            #extend metadata dict
            metadata_dict[video_key] = video_meta
            metadata_dict[video_key]['path'] = file_path

            #report success
            print('video ' + str(success_count + fail_count) + ' of ' + str(video_count) + ' (' + str(fail_count) + 'fails): ' + file_path)
            success_count += 1
        except Exception as e:
            fail_count += 1
            try: print('Failed to download: ' + video_meta['url'] + ' --- Message: ' + str(e))
            except: print('Failed to download')
        if success_count > 4:
            break

    #dump metadata dict as json file
    with open(destination_directory + metadata_file_name, 'w') as f:
        json.dump(metadata_dict, f)
    return metadata_dict


def download_video(url, destination_directory, file_name = None):
    assert os.path.isdir(destination_directory)
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    if file_name:
        file_path = best.download(filepath=destination_directory + file_name + '.' + best.extension)
    else:
        file_path = best.download(filepath=destination_directory)
    return file_path

def extract_subclips(metadata_dict, target_dir, metadata_file_name = "metadata_subclips.json"):
    metadata_dict_subclips = {}
    for video_key, video_meta in metadata_dict.items():
        try:
            for idx, subclip_meta in enumerate(video_meta['annotations']):

                assert len(subclip_meta['segment']) == 2

                #build subclip path
                subclip_name = str(video_key) + '_' + str(idx) + Path(video_meta['path']).suffix
                subclip_path = os.path.join(str(target_dir), subclip_name)

                #cut out subclip and it at subclip_path
                extract_subclip(video_meta['path'], subclip_meta['segment'][0], subclip_meta['segment'][1], subclip_path)

                #metadata corresponding to subclip
                metadata_dict_subclips[subclip_name] = {
                    'url': video_meta['url'],
                    'subset': video_meta['subset'],
                    'resolution': video_meta['resolution'],
                    'label': subclip_meta['label'],
                    'duration': subclip_meta['segment'][1] - subclip_meta['segment'][0],
                    'path': subclip_path
                }
        except:
            print("Failed to generate subclips from: " + video_key)

    #dump metadata dict as json file
    with open(target_dir + metadata_file_name, 'w') as f:
        json.dump(metadata_dict_subclips, f)

    return metadata_dict_subclips

def handle_prompt_input():
    assert(len(sys.argv[1:]) in (1,2))
    activity_net_db_location = sys.argv[1]
    assert(os.path.isfile(activity_net_db_location))
    activity_net_db_location = os.path.abspath(activity_net_db_location)
    try:
        activity_net_target_dir = sys.argv[2]
        assert (os.path.isdir(activity_net_target_dir))
    except:
        print('No valid target directory provided - Take current working directory')
        activity_net_target_dir = os.getcwd()
    return activity_net_db_location, activity_net_target_dir

def create_directories_if_necessary(activity_net_target_dir):
    download_dir = os.path.join(activity_net_target_dir, 'download/')
    subclip_dir = os.path.join(activity_net_target_dir, 'clips/')
    for dir in [download_dir, subclip_dir]:
        try:
            os.stat(dir)
        except:
            os.mkdir(dir)
    return download_dir, subclip_dir


if __name__ == '__main__':
    activity_net_db_location, activity_net_target_dir = handle_prompt_input()
    print(activity_net_db_location, activity_net_target_dir)
    download_dir, subclip_dir = create_directories_if_necessary(activity_net_target_dir)

    activity_net_dict = activity_net_db_from_json(activity_net_db_location)
    metadata_dict = download_activity_net(activity_net_dict, download_dir)
    extract_subclips(metadata_dict, target_dir=subclip_dir)