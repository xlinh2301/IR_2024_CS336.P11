import cv2
import numpy as np
import os
from tqdm import tqdm

def calculate_histogram_difference(frame1, frame2):
    """Tính sự khác biệt giữa 2 frame bằng histogram."""
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def extract_representative_frames_from_folder(frame_folder, output_folder, threshold=0.95):
    """Lọc frame đại diện từ folder và lưu lại giữ nguyên tên file."""
    frame_files = sorted(os.listdir(frame_folder))
    frame_paths = [os.path.join(frame_folder, f) for f in frame_files]

    if len(frame_paths) == 0:
        print("Thư mục không có frame.")
        return

    # Đọc tất cả các frame
    frames = []
    for path in tqdm(frame_paths, desc="Đang đọc frame"):
        frame = cv2.imread(path)
        if frame is not None:
            frames.append((path, frame))

    # Chọn frame đầu tiên
    selected_frames = [frames[0]]
    group_start_index = 0

    for i in range(1, len(frames)):
        diff = calculate_histogram_difference(frames[group_start_index][1], frames[i][1])
        if diff < threshold:
            # Thêm frame đầu, giữa và cuối của cụm
            if group_start_index != i - 1:
                mid_index = (group_start_index + i - 1) // 2
                selected_frames.extend([frames[mid_index], frames[i - 1]])
            selected_frames.append(frames[i])
            group_start_index = i

    # Đảm bảo không trùng lặp frame
    selected_paths = sorted(set(path for path, _ in selected_frames), key=lambda x: frame_files.index(os.path.basename(x)))

    # Tạo thư mục lưu frame kết quả
    os.makedirs(output_folder, exist_ok=True)
    for path in selected_paths:
        frame = cv2.imread(path)
        frame_name = os.path.basename(path).replace("frame_", "")
        output_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_path, frame)
        print(f"Đã lưu: {output_path}")


# Đường dẫn tới thư mục chứa frame và thư mục lưu kết quả
for i in range(1, 2):
    frame_folder = f"D:/AIC/Frames/Videos_L{str(i).zfill(2)}/"
    for video in os.listdir(frame_folder):
        video_name = os.path.splitext(video)[0]
        video_folder = os.path.join(frame_folder, video)
        output_folder = f"../AIC2024/public/assets/images/Videos_L{str(i).zfill(2)}/{video_name}"
        extract_representative_frames_from_folder(video_folder, output_folder, threshold=0.8)
