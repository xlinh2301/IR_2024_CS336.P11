import json
from collections import defaultdict
import os

for i in range(1, 32):
    # Đường dẫn tới file JSON
    input_file = f"E:/AIC/vietocr/output/reg/reg_L01_V0{str(i).zfill(2)}.json"
    output_file = f"E:/AIC/vietocr/output/group/reg_L01_V0{str(i).zfill(2)}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Đọc dữ liệu từ file .json
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Gom nhóm dữ liệu theo video_name và frame_id
    grouped_data = defaultdict(lambda: {"video_folder": None, "video_name": None, "frame_id": None, "text": [], "prob": []})

    for item in data:
        key = (item["video_name"], item["frame_id"])
        grouped_data[key]["video_folder"] = item["video_folder"]
        grouped_data[key]["video_name"] = item["video_name"]
        grouped_data[key]["frame_id"] = item["frame_id"]
        grouped_data[key]["text"].append(item["text"])
        grouped_data[key]["prob"].append(item["prob"])

    # Kết hợp các đoạn văn bản và tính trung bình xác suất
    result = [
        {
            "video_folder": value["video_folder"],
            "video_name": value["video_name"],
            "frame_id": value["frame_id"],
            "text": " ".join(value["text"]),
            "avg_prob": sum(value["prob"]) / len(value["prob"])
        }
        for value in grouped_data.values()
    ]

    # Lưu kết quả vào file JSON
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được lưu vào {output_file}")
