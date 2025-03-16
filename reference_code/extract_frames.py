import os
import math
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
from concurrent.futures import ProcessPoolExecutor

def save_vid_frames(video_path, output_dir, video_id):
    """Saves every frame of the video as a PNG image."""
    img_frames_dir = os.path.join(output_dir, video_id)
    os.makedirs(img_frames_dir, exist_ok=True)

    clip = VideoFileClip(video_path)
    fps = math.ceil(clip.fps)
    total_frames = int(clip.duration * fps)

    for frame_idx in range(total_frames):
        time = frame_idx / fps
        frame = clip.get_frame(time)
        frame_path = os.path.join(img_frames_dir, f"{video_id}_{frame_idx:04d}.png")
        plt.imsave(frame_path, frame)
        print(f"Saved frame {frame_idx} to {frame_path}")

def process_video(video_file, video_dir, frames_dir):
    """Process a single video by saving its frames."""
    video_id = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_dir, video_file)
    print(f"Processing {video_path}")

    save_vid_frames(video_path, frames_dir, video_id)

    clip = VideoFileClip(video_path)
    fps = math.ceil(clip.fps)
    print(f"Video {video_id} has {fps} fps")
    return video_id

if __name__ == "__main__":
    video_dir = "pose_estimation/videos_160"
    frames_dir = "./vid_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # List all mp4 files in the video directory
    video_files = [vid for vid in os.listdir(video_dir) if vid.endswith('.mp4')]

    # Use ProcessPoolExecutor to process videos concurrently
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_video, video_file, video_dir, frames_dir)
                   for video_file in video_files]

        for future in futures:
            try:
                video_id = future.result()
                print(f"Finished processing video {video_id}")
            except Exception as e:
                print(f"Error processing video: {e}")