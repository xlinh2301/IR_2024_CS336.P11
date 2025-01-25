from typing import List, Dict, Any, Optional
import torch
import clip
import os
import numpy as np
import json
import cv2
import faiss
from googletrans import Translator
from langdetect import detect
import requests
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from app.config import INDEX_FILE_PATH, ID_MAP_FILE_PATH, FILE_LIST, FILE_VIDEO_LIST, FILE_FPS_LIST
import traceback

# Kiểm tra xem PyTorch có nhận diện được GPU không
print("CUDA is available:", torch.cuda.is_available())

# In ra tên GPU nếu có
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Cấu hình môi trường và tải mô hình CLIP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-L/14", device=device)
translator = Translator()
index_load = faiss.read_index(INDEX_FILE_PATH)

# Nếu GPU có sẵn, chuyển FAISS index sang GPU
if torch.cuda.is_available():
    res = faiss.StandardGpuResources()  # Tạo tài nguyên GPU
    index_load = faiss.index_cpu_to_gpu(res, 0, index_load)  # Chuyển index sang GPU
    # print(f"index_load: {index_load}")
class SearchResult(BaseModel):
    frame_id: int
    video_id: str
    video_folder: str
    image_path: str
    video_path: str
    fps: int

def translate_query(query: str) -> str:
    """Dịch câu truy vấn từ tiếng Việt sang tiếng Anh"""
    translated_query = translator.translate(query, src='vi', dest='en').text
    print(f"Translated query: {translated_query}")
    return translated_query

def detect_language(query: str) -> Optional[str]:
    """Nhận diện ngôn ngữ của câu truy vấn"""
    try:
        lang = detect(query)
        print(f"Detected language: {lang}")
        return lang
    except Exception as e:
        print(f"Language detection failed: {e}")
        return None

def process_image(image_path: str) -> torch.Tensor:
    """Tải và tiền xử lý hình ảnh cho mô hình CLIP, hỗ trợ URL"""
    
    if image_path.startswith("http"):
        # Nếu image_path là URL của Google Drive, chuyển đổi nó thành URL tải về trực tiếp
        if "drive.google.com" in image_path:
            file_id = image_path.split("id=")[-1]
            image_path = f"https://drive.google.com/uc?export=download&id={file_id}"
            print(f"Download URL: {image_path}")
        # Tải ảnh về từ URL
        response = requests.get(image_path)
        if response.status_code != 200:
            raise ValueError(f"Unable to download image from {image_path}")
        image = Image.open(BytesIO(response.content))
    else:
        # Nếu image_path là đường dẫn file local
        image = Image.open(image_path)

    # Tiền xử lý ảnh cho CLIP
    image = preprocess(image).unsqueeze(0).to(device)
    return image


def search_text(text_query: str, top_k: int = 300) -> List[int]:
    """Tìm kiếm văn bản, dịch nếu cần thiết và thực hiện tìm kiếm"""
    try:
        print(f"Received query: {text_query}")
        
        # Xác định ngôn ngữ
        lang = detect_language(text_query)
        print(f"Detected language: {lang}")
        translated_query = text_query

        if lang == "vi":
            # Dịch tiếng Việt sang tiếng Anh
            translated_query = translate_query(text_query)
            print(f"Translated query: {translated_query}")
        
        # Tokenize văn bản
        text = clip.tokenize([translated_query]).to(device)
        
        # Encode văn bản bằng mô hình
        with torch.no_grad():
            text_features = model.encode_text(text)
        
        # Chuyển đổi sang numpy và tìm kiếm trong FAISS
        text_features = text_features.cpu().numpy().astype('float32').flatten()
        
        distances, indices = index_load.search(np.expand_dims(text_features, axis=0), top_k)
        
        return indices[0]
    
    except Exception as e:
        print(f"An error occurred during text search: {e}")
        print("Traceback:")
        traceback.print_exc()  # In toàn bộ stack trace để debug lỗi
        return []

def search_image(image_path: str, top_k: int = 300) -> List[int]:
    """Tìm kiếm bằng hình ảnh"""
    image = process_image(image_path)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features = image_features.cpu().numpy().astype('float32').flatten()
    distances, indices = index_load.search(np.expand_dims(image_features, axis=0), top_k)
    return indices[0]

def load_file_list(file_list_path: str) -> Dict[str, str]:
    """Tải tệp JSON chứa ID và tên tệp vào một dictionary."""
    with open(file_list_path, 'r') as f:
        file_list = json.load(f)
    
    # Kiểm tra cấu trúc của file_list và xây dựng dictionary
    file_dict = {}
    for item in file_list:
        title = item.get('title')
        file_id = item.get('id')  # Hoặc tên trường chứa ID
        if title and file_id:
            file_dict[title] = file_id
    
    return file_dict

def load_fps_list(fps_list_path: str) -> Dict[str, float]:
    """Tải tệp JSON chứa FPS vào một dictionary."""
    with open(fps_list_path, 'r') as f:
        fps_list = json.load(f)
    
    # Xây dựng dictionary từ danh sách FPS
    fps_dict = {}
    for item in fps_list:
        title = item.get('title')
        fps = item.get('fps')
        if title and fps is not None:
            fps_dict[title] = fps
    
    return fps_dict

def get_fps(video_name: str, file_fps_list: Dict[str, float]) -> Optional[float]:
    """Lấy FPS từ từ điển dựa trên video_name."""
    return file_fps_list.get(video_name, None)

def construct_image_path_and_video_path(file_list: Dict[str, str], file_video_list: Dict[str, str], file_fps_list: Dict[str, int], image_info: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    Xây dựng đường dẫn ảnh và video từ thông tin ảnh sử dụng tệp JSON chứa ID và tên tệp.
    
    :param file_list: Dictionary chứa ID và tên tệp.
    :param file_video_list: Dictionary chứa ID và tên video.
    :param image_info: Thông tin ảnh bao gồm 'frame_id', 'video_id', và 'video_folder'.
    :return: Một dictionary chứa đường dẫn ảnh và video.
    """
    base_image_url = "https://drive.google.com/thumbnail?export=view&sz=w160-h160&id="
    base_video_url = "https://drive.google.com/file/d/"  # Base URL for video
    
    # Extract information from image_info
    frame_id = image_info.get('frame_id')
    video_id = image_info.get('video_id')
    video_folder = image_info.get('video_folder')
    
    result = {}
    
    if frame_id and video_id and video_folder:
        # Build the file name for the image
        file_name = f"{video_id}_{frame_id}.jpg"
        video_name = f"{video_id}.mp4"
        file_id = file_list.get(file_name)
        video_file_id = file_video_list.get(video_name)  # Get video file ID using video_id
        fps = file_fps_list.get(video_name, None)

        if file_id:
            # Build the image path from file_id
            image_path = f"{base_image_url}{file_id}"
            result['image_path'] = image_path
        else:
            print(f"File ID not found for file: {file_name}")
            result['image_path'] = None
        
        if video_file_id:
            # Construct the video path from video_id
            video_path = f"{base_video_url}{video_file_id}/preview"
            result['video_path'] = video_path
        else:
            print(f"Video ID not found for video: {video_id}")
            result['video_path'] = None

        if fps:
            result['fps'] = fps
    else:
        print(f"Required information not found in image_info: {image_info}")
        result['image_path'] = None
        result['video_path'] = None
        result['fps'] = None
        
    return result


def search_faiss(query: Optional[str] = None, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Tìm kiếm trong FAISS dựa trên văn bản hoặc hình ảnh và trả về kết quả dưới dạng danh sách từ điển"""
    # Tải chỉ mục FAISS và bản đồ ID
    print("Loading FAISS index and ID map...")
    # Đọc bản đồ ID từ tệp
    try:
        with open(ID_MAP_FILE_PATH, 'r') as f:
            id_map_load = json.load(f)
        
        # Đảm bảo id_map_load là một dictionary
        if isinstance(id_map_load, list):
            id_map_load = {str(i): item for i, item in enumerate(id_map_load)}        
        
    except Exception as e:
        print(f"Error loading ID map file: {e}")
        return []

    file_list = load_file_list(FILE_LIST) 
    file_video_list = load_file_list(FILE_VIDEO_LIST)    
    file_fps_list = load_fps_list(FILE_FPS_LIST)
    if query:
        # Tìm kiếm theo văn bản
        result_indices = search_text(query, 400)
        print(f"Found {len(result_indices)} results for query: {query}")
    elif image_path:
        # Tìm kiếm theo hình ảnh
        result_indices = search_image(image_path, 400)
    else:
        raise ValueError("Either query or image_path must be provided.")

    # Chuyển đổi các chỉ số thành kết quả và xây dựng đường dẫn hình ảnh
    results = []
    for idx in result_indices:
        image_info = id_map_load.get(str(idx), None)
        if image_info:
            # Sử dụng hàm construct_image_path_and_video_path để tạo đường dẫn hình ảnh
            paths = construct_image_path_and_video_path(file_list, file_video_list, file_fps_list, image_info)
            img_path = paths.get('image_path')
            # video_path = paths.get('video_path')
            fps = paths.get('fps')
            
            # if img_path and video_path:
            # if img_path:
            #     # Thêm thông tin hình ảnh, video, và FPS vào kết quả
            #     results.append({
            #         'frame_id': image_info['frame_id'],
            #         'video_id': image_info['video_id'], 
            #         'video_folder': image_info['video_folder'],
            #         # 'image_path': img_path,
            #         # 'video_path': video_path,
            #         'fps': fps  # Thêm FPS vào kết quả
            #     })
            # else:
            #     if not img_path:
            #         print(f"Failed to construct image path for index { idx }: Image path not found")
            #     if not video_path:
            #         print(f"Failed to construct video path for index {idx}: Video path not found")

            results.append({
                    'frame_id': image_info['frame_id'],
                    'video_id': image_info['video_id'], 
                    'video_folder': image_info['video_folder'],
                    'image_path': img_path,
                    # 'video_path': video_path,
                    'fps': fps  # Thêm FPS vào kết quả
                })
        else:
            print(f"Video ID not found for index {idx}")

    return results   