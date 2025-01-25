# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import time
import sys

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
import json


class TextDetector(object):
    def __init__(self, args, logger=None):
        if logger is None:
            logger = get_logger()
        self.args = args
        self.use_onnx = args.use_onnx
        pre_process_list = [
            {
                "DetResizeForTest": {
                    "limit_side_len": args.det_limit_side_len,
                    "limit_type": args.det_limit_type,
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        postprocess_params = {}
        postprocess_params["name"] = "DBPostProcess"
        postprocess_params["thresh"] = args.det_db_thresh
        postprocess_params["box_thresh"] = args.det_db_box_thresh
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
        postprocess_params["use_dilation"] = args.use_dilation
        postprocess_params["score_mode"] = args.det_db_score_mode
        postprocess_params["box_type"] = args.det_box_type
       
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = utility.create_predictor(args, "det", logger)

        if self.use_onnx:
            img_h, img_w = self.input_tensor.shape[2:]
            if isinstance(img_h, str) or isinstance(img_w, str):
                pass
            elif img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
                pre_process_list[0] = {
                    "DetResizeForTest": {"image_shape": [img_h, img_w]}
                }
        self.preprocess_op = create_operators(pre_process_list)

        if args.benchmark:
            import auto_log

            pid = os.getpid()
            gpu_id = utility.get_infer_gpuid()
            self.autolog = auto_log.AutoLogger(
                model_name="det",
                model_precision=args.precision,
                batch_size=1,
                data_shape="dynamic",
                save_path=None,  # not used if logger is not None
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=["preprocess_time", "inference_time", "postprocess_time"],
                warmup=2,
                logger=logger,
            )

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def pad_polygons(self, polygon, max_points):
        padding_size = max_points - len(polygon)
        if padding_size == 0:
            return polygon
        last_point = polygon[-1]
        padding = np.repeat([last_point], padding_size, axis=0)
        return np.vstack([polygon, padding])

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)

        if len(dt_boxes_new) > 0:
            max_points = max(len(polygon) for polygon in dt_boxes_new)
            dt_boxes_new = [
                self.pad_polygons(polygon, max_points) for polygon in dt_boxes_new
            ]

        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def predict(self, img):
        ori_im = img.copy()
        # # Áp dụng mask
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # # print(img.shape[:2])
        # # Define the regions to mask
        # regions_to_mask = [
        #     [[0, 0], [200, 0], [200, 23], [0, 23]],
        #     [[391, 23], [450, 23], [450, 41], [391, 41]],
        #     [[0, 245], [480, 245], [480, 258], [0, 258]]
        # ]

        # # Fill the mask with the defined regions
        # for coords in regions_to_mask:
        #     polygon = np.array(coords, np.int32)
        #     polygon = polygon.reshape((-1, 1, 2))
        #     cv2.fillPoly(mask, [polygon], 255)  # 255 for white mask (to cover)

        # # Apply the mask to the image
        # img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        data = {"image": img}

        st = time.time()

        if self.args.benchmark:
            self.autolog.times.start()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        if self.args.benchmark:
            self.autolog.times.stamp()
        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
        else:
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            if self.args.benchmark:
                self.autolog.times.stamp()

        preds = {}
    
        preds["maps"] = outputs[0]

        post_result = self.postprocess_op(preds, shape_list)
        # print(post_result)
        dt_boxes = post_result[0]["points"]

        if self.args.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        if self.args.benchmark:
            self.autolog.times.end(stamp=True)
        et = time.time()
        return dt_boxes, et - st

    def __call__(self, img, use_slice=False):
        # For image like poster with one side much greater than the other side,
        # splitting recursively and processing with overlap to enhance performance.
        MIN_BOUND_DISTANCE = 50
        dt_boxes = np.zeros((0, 4, 2), dtype=np.float32)
        elapse = 0
        if (
            img.shape[0] / img.shape[1] > 2
            and img.shape[0] > self.args.det_limit_side_len
            and use_slice
        ):
            start_h = 0
            end_h = 0
            while end_h <= img.shape[0]:
                end_h = start_h + img.shape[1] * 3 // 4
                subimg = img[start_h:end_h, :]
                if len(subimg) == 0:
                    break
                sub_dt_boxes, sub_elapse = self.predict(subimg)
                offset = start_h
                # To prevent text blocks from being cut off, roll back a certain buffer area.
                if (
                    len(sub_dt_boxes) == 0
                    or img.shape[1] - max([x[-1][1] for x in sub_dt_boxes])
                    > MIN_BOUND_DISTANCE
                ):
                    start_h = end_h
                else:
                    sorted_indices = np.argsort(sub_dt_boxes[:, 2, 1])
                    sub_dt_boxes = sub_dt_boxes[sorted_indices]
                    bottom_line = (
                        0
                        if len(sub_dt_boxes) <= 1
                        else int(np.max(sub_dt_boxes[:-1, 2, 1]))
                    )
                    if bottom_line > 0:
                        start_h += bottom_line
                        sub_dt_boxes = sub_dt_boxes[
                            sub_dt_boxes[:, 2, 1] <= bottom_line
                        ]
                    else:
                        start_h = end_h
                if len(sub_dt_boxes) > 0:
                    if dt_boxes.shape[0] == 0:
                        dt_boxes = sub_dt_boxes + np.array(
                            [0, offset], dtype=np.float32
                        )
                    else:
                        dt_boxes = np.append(
                            dt_boxes,
                            sub_dt_boxes + np.array([0, offset], dtype=np.float32),
                            axis=0,
                        )
                elapse += sub_elapse
        elif (
            img.shape[1] / img.shape[0] > 3
            and img.shape[1] > self.args.det_limit_side_len * 3
            and use_slice
        ):
            start_w = 0
            end_w = 0
            while end_w <= img.shape[1]:
                end_w = start_w + img.shape[0] * 3 // 4
                subimg = img[:, start_w:end_w]
                if len(subimg) == 0:
                    break
                sub_dt_boxes, sub_elapse = self.predict(subimg)
                offset = start_w
                if (
                    len(sub_dt_boxes) == 0
                    or img.shape[0] - max([x[-1][0] for x in sub_dt_boxes])
                    > MIN_BOUND_DISTANCE
                ):
                    start_w = end_w
                else:
                    sorted_indices = np.argsort(sub_dt_boxes[:, 2, 0])
                    sub_dt_boxes = sub_dt_boxes[sorted_indices]
                    right_line = (
                        0
                        if len(sub_dt_boxes) <= 1
                        else int(np.max(sub_dt_boxes[:-1, 1, 0]))
                    )
                    if right_line > 0:
                        start_w += right_line
                        sub_dt_boxes = sub_dt_boxes[sub_dt_boxes[:, 1, 0] <= right_line]
                    else:
                        start_w = end_w
                if len(sub_dt_boxes) > 0:
                    if dt_boxes.shape[0] == 0:
                        dt_boxes = sub_dt_boxes + np.array(
                            [offset, 0], dtype=np.float32
                        )
                    else:
                        dt_boxes = np.append(
                            dt_boxes,
                            sub_dt_boxes + np.array([offset, 0], dtype=np.float32),
                            axis=0,
                        )
                elapse += sub_elapse
        else:
            dt_boxes, elapse = self.predict(img)
        return dt_boxes, elapse


def save_predictions_to_txt(save_results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for save_pred in save_results:
            img_name = save_pred["image_name"]
            boxes = save_pred["boxes"]
            # Convert to string and write to file
            f.write(f"{img_name} {str(boxes)}\n")
            f.write(f"{coords}\n")

if __name__ == "__main__":
    args = utility.parse_args()

    # Hardcode the model directory
    args.det_model_dir = "./inference/resnet50_finetune"
    # args.det_box_type = "poly"
    args.det_db_box_thresh = 0.18
    args.det_db_thresh = 0.1
    # args.det_db_score_mode = "slow"   
    # Get image file list
    for i in range(1, 32):
        image_file_list = [
            os.path.join(f"D:\AIC\Frames\Videos_L01\L01_V0{str(i).zfill(2)}", file)
            for file in os.listdir(f"D:\AIC\Frames\Videos_L01\L01_V0{str(i).zfill(2)}")
        ]
        total_time = 0

        # create logger
        log_file = args.save_log_path
        if os.path.isdir(args.save_log_path) or (
            not os.path.exists(args.save_log_path) and args.save_log_path.endswith("/")
        ):
            log_file = os.path.join(log_file, "benchmark_detection.log")
        logger = get_logger(log_file=log_file)

        # create text detector
        text_detector = TextDetector(args, logger)

        # Warmup (optional)
        if args.warmup:
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(2):
                res = text_detector(img)

        save_results = []
        for idx, image_file in enumerate(image_file_list):
            img, flag_gif, flag_pdf = check_and_read(image_file)
            if not flag_gif and not flag_pdf:
                img = cv2.imread(image_file)
            if not flag_pdf:
                if img is None:
                    logger.debug("Error loading image: {}".format(image_file))
                    continue
                imgs = [img]
            else:
                page_num = args.page_num
                if page_num > len(img) or page_num == 0:
                    page_num = len(img)
                imgs = img[:page_num]
            for index, img in enumerate(imgs):
                st = time.time()
                dt_boxes, _ = text_detector(img)
                elapse = time.time() - st
                total_time += elapse

                # Generate the prediction result for logging
                if len(dt_boxes) > 0:
                    if len(imgs) > 1:
                        save_pred = (
                            os.path.basename(image_file)
                            + "_"
                            + str(index)
                            + " "
                            + str(json.dumps([x.tolist() for x in dt_boxes]))
                            # + "\n"
                        )
                    else:
                        save_pred = (
                            os.path.basename(image_file)
                            + " "
                            + str(json.dumps([x.tolist() for x in dt_boxes]))
                            # + "\n"
                        )
                    save_results.append(save_pred)
                    name = f"L01_V0{str(i).zfill(2)}_results"
                    output_file = f"E:/Docs/CVA/PaddleOCR/train_data/results/{name}.txt"
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "w", encoding="utf-8") as f:
                        for result in save_results:
                            f.write(result + "\n")
                    # logger.info(save_pred)
                    # print(save_pred)
                # else:
                #     logger.info(f"No bounding boxes found for {image_file}_{index}")
                # Display the results using OpenCV
                # src_im = utility.draw_text_det_res(dt_boxes, img)
                # cv2.imshow("Detected Text", src_im)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                

        logger.info("The predict total time is {}".format(total_time))