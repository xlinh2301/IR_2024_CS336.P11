import os
import json

def extract_frame_data(root_folder, output_file):
    # Danh sách để lưu dữ liệu
    frame_data = []

    # Duyệt qua tất cả thư mục con
    for video_folder in os.listdir(root_folder):
        if video_folder == 'archive.zip':
            continue
        video_folder_path = os.path.join(root_folder, video_folder)
        for video in os.listdir(video_folder_path):
            print(video)
            for frame_file in os.listdir(os.path.join(video_folder_path, video)):
                print(frame_file)
                if frame_file.endswith(".jpg"):  # Kiểm tra định dạng frame
                    # Lấy thông tin từ tên file
                    frame_id = frame_file.split("_")[-1].split(".")[0]
                    video_id = "_".join(frame_file.split("_")[:2])
                    
                    # Thêm dữ liệu vào danh sách
                    frame_data.append({
                        "frame_id": frame_id,
                        "video_id": f"{video_id}_{frame_id}",
                        "video_folder": video_id
                    })

    # Lưu dữ liệu vào file JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(frame_data, f, indent=4, ensure_ascii=False)

    print(f"Data saved to {output_file}")

# Đường dẫn thư mục gốc và file xuất
root_folder = r"D:\AIC\Frames"
output_file = "./file_name.json"

# Gọi hàm
extract_frame_data(root_folder, output_file)
