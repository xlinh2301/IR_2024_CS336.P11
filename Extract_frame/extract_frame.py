from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
import os

def extract_keyframes(video_path, save_folder):
    """Tách keyframe từ video và lưu trữ, lấy 1 keyframe mỗi 30 frame."""
    if not os.path.exists(save_folder):
       os.makedirs(save_folder)
    # Extract video name from the path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    keyframe_paths = []
    saved_frame_count = 0
    while success:
        if count % 10 == 0:
            keyframe_path = os.path.join(save_folder, f"{video_name}_frame_{count}.jpg")
            cv2.imwrite(keyframe_path, image)
            keyframe_paths.append(keyframe_path)
            saved_frame_count += 1
        success, image = vidcap.read()
        count += 1
    print(f"Extracted {saved_frame_count} keyframes from {video_path}.")
    vidcap.release()

def main():
    for num in range(1, 2):
        path = f'E:/AIC/AIC2024/src/assets/videos/Videos_L{str(num).zfill(2)}'
        for video in os.listdir(path):
            video_name,_ = os.path.splitext(video)
            output_dir = f'D:/AIC/Frames/Videos_L{str(num).zfill(2)}/{video_name}'
            video_path = os.path.join(path, video)
            print(output_dir)
            extract_keyframes(video_path, output_dir)
if __name__ == "__main__":
    main()
