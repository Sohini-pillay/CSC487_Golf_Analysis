import os
import math

from scipy.io import loadmat
from pytubefix import YouTube
from moviepy.editor import *

def extract_frames(start_frame, end_frame, video_path, output_dir, video_name):
    clip = VideoFileClip(video_path)
    fps = math.ceil(clip.fps)
    start_time = start_frame/fps
    end_time = end_frame/fps

    edited_clip = clip.subclip(start_time, end_time)
    edited_clip.write_videofile(output_dir + "/" + video_name + ".mp4")

if __name__ == "__main__":
    YOUTUBE_URL = "https://youtube.com/watch?v="

    VIDEO_ID_IDX = 1
    ANGLE_IDX = 5
    KEY_FRAMES_IDX = 7
    BOUNDING_BOX_IDX = 8

    output_dir = "youtube"
    edited_clip_dir = "shortened"

    if output_dir not in os.listdir():
        os.mkdir(output_dir)
    if edited_clip_dir not in os.listdir():
        os.mkdir(edited_clip_dir)

    data = loadmat('golfDB.mat')

    dump = data['golfDB'][0]

    video_num = 0

    write_data = {}

    for video in dump:
        try:
            video_url = YOUTUBE_URL + video[VIDEO_ID_IDX][0]
            yt_video = YouTube(video_url).streams.get_highest_resolution()
            video_path = yt_video.download(output_path=output_dir)
            
            start_frame = video[KEY_FRAMES_IDX][0][0]
            end_frame = video[KEY_FRAMES_IDX][0][-1]
        
            extract_frames(start_frame, end_frame, video_path, edited_clip_dir, str(video_num))
            write_data[video_num] = video
        except:
            print("Video Cropping Failed")
        video_num += 1

    with open("video_data.txt", "w") as write_file:
        write_file.write(write_data)