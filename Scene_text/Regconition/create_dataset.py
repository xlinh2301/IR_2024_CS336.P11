import os
import json
import cv2

# Đường dẫn file label gốc và thư mục chứa ảnh gốc
label_path = r"E:\Docs\CVA\PaddleOCR\train_data\vietnamese\train_bkai_label_2.txt"
image_dir = r"E:\Docs\CVA\PaddleOCR\train_data\vietnamese"

# Thư mục để lưu ảnh cắt và file label mới
output_image_dir = r"E:\AIC\vietocr\data\img"
output_label_path = r"E:\AIC\vietocr\data\label_train.txt"

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_image_dir, exist_ok=True)

# Đọc file label gốc
with open(label_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Mở file label mới để ghi
with open(output_label_path, "w", encoding="utf-8") as out_label:
    for line in lines:
        try:
            # Tách đường dẫn ảnh và thông tin bounding box
            image_path, annotations = line.strip().split("\t")
            annotations = json.loads(annotations)
            full_image_path = os.path.join(image_dir, image_path)

            # Đọc ảnh gốc
            img = cv2.imread(full_image_path)
            if img is None:
                print(f"Không thể đọc ảnh: {full_image_path}")
                continue

            # Xử lý từng bounding box
            for i, annotation in enumerate(annotations):
                transcription = annotation["transcription"]
                points = annotation["points"]

                # Kiểm tra nếu transcription không rỗng
                if not transcription.strip():  # Nếu transcription rỗng thì bỏ qua
                    continue

                # Lấy tọa độ bounding box
                x_min = int(min(p[0] for p in points))
                y_min = int(min(p[1] for p in points))
                x_max = int(max(p[0] for p in points))
                y_max = int(max(p[1] for p in points))

                # Kiểm tra tọa độ bounding box
                if x_min < 0 or y_min < 0 or x_max > img.shape[1] or y_max > img.shape[0]:
                    print(f"Tọa độ bounding box không hợp lệ: {x_min, y_min, x_max, y_max} trong {image_path}")
                    continue

                # Cắt ảnh
                cropped_img = img[y_min:y_max, x_min:x_max]

                # Kiểm tra ảnh cắt
                if cropped_img is None or cropped_img.size == 0:
                    print(f"Không thể cắt ảnh: {x_min, y_min, x_max, y_max} trong {image_path}")
                    continue

                # Tạo tên file mới
                cropped_file_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}.png"
                cropped_file_path = os.path.join(output_image_dir, cropped_file_name)

                # Lưu ảnh cắt
                cv2.imwrite(cropped_file_path, cropped_img)

                # Ghi vào file label mới
                out_label.write(f"{cropped_file_name}\t{transcription}\n")

        except Exception as e:
            print(f"Lỗi khi xử lý dòng: {line.strip()}. Chi tiết: {e}")
