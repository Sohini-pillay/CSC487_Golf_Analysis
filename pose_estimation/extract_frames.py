import os
import math
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.io import loadmat
import csv


def plot_frames(events, video_path):
    clip = VideoFileClip(video_path)
    fps = math.ceil(clip.fps)  

    fig, axes = plt.subplots(1, len(events), figsize=(15, 5)) 
    fig.suptitle("Key Event Frames", fontsize=16)

    for idx, frame_index in enumerate(events):
        time = frame_index / fps
        frame = clip.get_frame(time)
        
        axes[idx].imshow(frame)
        axes[idx].axis("off")
        axes[idx].set_title(f"Frame {frame_index}")

    plt.tight_layout()
    plt.show()


def get_events_for_video(video_id):
    events = []
    with open('key_frames.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            if int(row[0]) == video_id:
                events.append(int(row[1]))
    return events

def get_vid_frames(dump, video_id, VIDEO_ID, KEY_FRAMES_IDX):
    for video in dump:
        if video[VIDEO_ID] == int(video_id):
            frames = video[KEY_FRAMES_IDX][0]
            return frames[1:9]
    return None

# save frames to video corresponding directory
def save_frames(frames, video_path, output_dir, video_id):
    img_frames_dir = os.path.join(output_dir, video_id)
    os.makedirs(img_frames_dir, exist_ok=True)
    
    clip = VideoFileClip(video_path)
    fps = math.ceil(clip.fps)  
    
    for frame_ind in frames:
        time = frame_ind / fps
        frame = clip.get_frame(time)
        frame_path = os.path.join(img_frames_dir, f"{video_id}_frame_{frame_ind}.png")
        plt.imsave(frame_path, frame)
        print(f"saved {frame_ind} to {frame_path}")

def save_vid_frames(video_path, output_dir, video_id):
    img_frames_dir = os.path.join(output_dir, video_id)
    os.makedirs(img_frames_dir, exist_ok=True)

    clip = VideoFileClip(video_path)
    # print(f"Video {video_id} FPS: {clip.fps}, Duration: {clip.duration}")
    fps = math.ceil(clip.fps)
    total_frames = int(clip.duration * fps)

    for frame_idx in range(total_frames):
        time = frame_idx / fps
        frame = clip.get_frame(time)
        frame_path = os.path.join(img_frames_dir, f"{video_id}_{frame_idx:04d}.png")
        plt.imsave(frame_path, frame)
        print(f"saved {frame_idx} to {frame_path}")


if __name__ == "__main__":

    # for testing key events

    # data = loadmat('../golfDB.mat')
    # dump = data['golfDB'][0]

    # video_id = 1013

    # video_entry = next((video for video in dump if video[0] == video_id), None)

    # key_frames = video_entry[7][0]
    # start = key_frames[0]
    # end = key_frames[9]
    # key_frames = key_frames[1:9]

    # print(f"clip {video_id} with key frames {key_frames} and clip start {start} and end {end}")

    # clip = VideoFileClip(f"../youtube/{video_id}.mp4")
    # print(f"Video {video_id} FPS: {clip.fps}, Duration: {clip.duration}")

        
    # fps = math.ceil(clip.fps)
    # total_frames = int(clip.duration * fps)
    # print(f"video {video_id} with key frames {key_frames} and total frames {total_frames}")

    # # test key event CSV for videos_160
    # translated_events = get_events_for_video(1013)
    # print(f"video {video_id} with key frames {translated_events} and total frames {total_frames}")

    # plot_frames(translated_events, "../videos_160/1013.mp4")

    # plot_frames(key_frames, "../youtube/1013.mp4")

    # VIDEO_ID = 0
    # KEY_FRAMES_IDX = 7

    video_dir = "../videos_160"

    frames_dir = "./vid_frames"

    os.makedirs(frames_dir, exist_ok=True)

    for vid in os.listdir(video_dir):
        video_id = vid.split(".")[0]
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        print(video_path)
        
        # get the frames array
        # key_events = get_vid_frames(dump, video_id, VIDEO_ID, KEY_FRAMES_IDX) 
        # save_frames(key_events, video_path, frames_dir, video_id)

        save_vid_frames(video_path, frames_dir, video_id)
        clip = VideoFileClip(video_path)
        fps = math.ceil(clip.fps)
        print(f"video {video_id} has {fps} fps")