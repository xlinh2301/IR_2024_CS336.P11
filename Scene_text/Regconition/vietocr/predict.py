import json
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np

# Load VietOCR configuration
config = Cfg.load_config_from_file(r'E:/AIC/vietocr/config/config.yml')
detector = Predictor(config)



# Path to Vietnamese font
font_path = r"E:/AIC/vietocr/font/Roboto-Light.ttf"  # Đường dẫn font hỗ trợ tiếng Việt

# Output directory
output = r"./output/reg"
os.makedirs(output, exist_ok=True)

for i in range(2, 32):
    # Path to the results file
    results_file = f"E:/AIC/Remake/OCR/detection_results/L01_V0{str(i).zfill(2)}_results.txt"
    print(f"Processing L01_V0{str(i).zfill(2)}")
    # Load the detection results
    with open(results_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process each line in the results file
    for line in lines:
        try:
            # Parse file name and bounding boxes
            parts = line.strip().split(" ", 1)
            image_path = os.path.join(f"D:/AIC/Frames/Videos_L01/L01_V0{str(i).zfill(2)}", parts[0])
            boxes = json.loads(parts[1])

            # Open the image using OpenCV
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for compatibility

            # Convert OpenCV image to PIL
            pil_image = Image.fromarray(image_rgb)

            for box in boxes:
                # Convert coordinates to integers
                x_min = int(min(point[0] for point in box))
                y_min = int(min(point[1] for point in box))
                x_max = int(max(point[0] for point in box))
                y_max = int(max(point[1] for point in box))

                # Crop the image using PIL
                cropped_img = pil_image.crop((x_min, y_min, x_max, y_max))

                # Predict with VietOCR
                prediction, prob = detector.predict(cropped_img, return_prob=True)
                if "°" in prediction:
                    prediction = prediction.replace("°", " ")
                if prob > 0.5:
                    # Extract details
                    video_folder = parts[0].split("_")[0]
                    video_name = parts[0].split("_")[0] + "_" + parts[0].split("_")[1]
                    frame_id = parts[0].split("_")[2].split(".")[0]
                    doc = {
                        "video_folder": video_folder,
                        "video_name": video_name,
                        "frame_id": frame_id,
                        "text": prediction,
                        "prob": prob
                    }

                    # File path for current video
                    json_file_path = os.path.join(output, f"reg_{video_name}.json")

                    # Read existing data (if file exists)
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        data = []

                    # Append new record and save back to file
                    data.append(doc)
                    with open(json_file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
            #     if prob > 0.5:
            #         # Draw bounding box on the image
            #         color = (0, 255, 0)  # Green color for bounding box
            #         draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

            #         # Draw label using Pillow
            #         label = f"{prediction} ({prob:.2f})"
            #         font = ImageFont.truetype(font_path, 8)  # Font size 20
            #         text_bbox = font.getbbox(label)  # Get bounding box of the text
            #         text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            #         text_position = (x_min, y_min - text_height if y_min - text_height > 10 else y_min + 10)
            #         draw.rectangle(
            #             [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
            #             fill=color
            #         )
            #         draw.text(text_position, label, font=font, fill=(0, 0, 0))

            #         print(f"{prediction}, {prob}")

            # # Convert PIL image back to OpenCV format
            # image_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # # Display the image
            # cv2.imshow("Image with Bounding Boxes", image_with_boxes)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")
