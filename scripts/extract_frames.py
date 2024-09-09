import subprocess
import os
import sys

def extract_frames_ffmpeg(video_path, output_folder, interval=0.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fps = 1 / interval
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps}',
        os.path.join(output_folder, 'frame_%04d.jpg')
    ]

  
    subprocess.run(command, check=True)
    print(f"Frames extracted to {output_folder}")

args = sys.argv[1:]

if len(args) < 2:
    print("No args received. Expected video path and output folder.")
else:
    video_path = args[0]
    output_folder = args[1]
    extract_frames_ffmpeg(video_path, output_folder)