from typing import List, Dict, Optional
from elasticsearch import Elasticsearch, exceptions
import json
import os
from rapidfuzz import fuzz  # type: ignore
from bisect import bisect_right
from app.config import ASR_BACKUP_FILE_PATH, OCR_BACKUP_FILE_PATH, OBJECT_BACKUP_FILE_PATH, FILE_LIST, FILE_VIDEO_LIST, FILE_FPS_LIST, FILE_NAME_FRAME

def load_json_file(file_path: str) -> List[Dict]:
    """Tải dữ liệu từ tệp JSON và trả về dưới dạng danh sách các dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

def load_file_dict(file_list_path: str) -> Dict[str, str]:
    """Tải danh sách file vào dictionary với key là tên file và value là ID."""
    file_list = load_json_file(file_list_path)
    return {item.get('title'): item.get('id') for item in file_list if item.get('title') and item.get('id')}

def construct_paths(file_list: Dict[str, str], file_video_list: Dict[str, str], file_fps_list: Dict[str, float], image_info: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    Xây dựng đường dẫn ảnh và video từ thông tin ảnh sử dụng tệp JSON chứa ID và tên tệp.
    
    :param file_list: Dictionary chứa ID và tên tệp ảnh.
    :param file_video_list: Dictionary chứa ID và tên video.
    :param file_fps_list: Dictionary chứa FPS của video.
    :param image_info: Thông tin ảnh bao gồm 'frame_id', 'video_id', và 'video_folder'.
    :return: Một dictionary chứa đường dẫn ảnh, video và FPS.
    """
    base_image_url = "https://drive.google.com/thumbnail?export=view&sz=w160-h160&id="
    base_video_url = "https://drive.google.com/file/d/"
    
    frame_id = image_info.get('frame_id')
    video_id = image_info.get('video_id')
    video_folder = image_info.get('video_folder')
    
    result = {
        'image_path': None,
        'video_path': None,
        'fps': None
    }

    if frame_id and video_id and video_folder:
        file_name = f"{video_id}_{frame_id}.jpg"
        video_name = f"{video_id}.mp4"

        file_id = file_list.get(file_name)
        video_file_id = file_video_list.get(video_name)
        fps = file_fps_list.get(video_name)
        
        if file_id:
            result['image_path'] = f"{base_image_url}{file_id}"
        if video_file_id:
            result['video_path'] = f"{base_video_url}{video_file_id}/preview"
        if fps is not None:
            result['fps'] = fps

    return result

def search_from_elasticsearch(es: Elasticsearch, index_name: str, query: str, field: str) -> List[Dict]:
    """Tìm kiếm từ Elasticsearch dựa trên trường và truy vấn cụ thể."""
    search_query = {
        "query": {
            "match": {
                field: query
            }
        },
        "size": 100
    }
    try:
        response = es.search(index=index_name, body=search_query)
        return response['hits']['hits']
    except (exceptions.ConnectionError, exceptions.TransportError) as e:
        print(f"Elasticsearch connection error: {e}")
        return []

def search_in_backup(query: str, backup_data: List[Dict], threshold: int = 70, field: str = 'text') -> List[Dict]:
    """Tìm kiếm trong dữ liệu backup bằng cách sử dụng fuzzy matching."""
    results = []
    for entry in backup_data:
        text_list = entry.get(field, [])
        if isinstance(text_list, list):
            for text in text_list:
                score = fuzz.partial_ratio(query.lower(), text.lower())
                if score >= threshold:
                    results.append(entry)
                    break
        if len(results) >= 300:
            break
    return results

def search_ocr(es: Elasticsearch, index_name: str, query: str) -> List[Dict]:
    """Tìm kiếm OCR trong Elasticsearch hoặc trong backup nếu không có kết quả từ Elasticsearch."""
    file_list = load_file_dict(FILE_LIST)
    file_video_list = load_file_dict(FILE_VIDEO_LIST)
    file_fps_list = {item['title']: item['fps'] for item in load_json_file(FILE_FPS_LIST)}

    es_results = search_from_elasticsearch(es, index_name, query, 'text')
    print(es_results)
    results = []
    if es_results:
        for hit in es_results:
            video_name = hit['_source'].get('video_name', '')
            video_folder = f"Videos_{video_name.split('_')[0]}"
            frame_id = hit['_source'].get('frame', '')
            image_info = {
                'frame_id': frame_id,
                'video_id': video_name,
                'video_folder': video_folder,
                'text': hit['_source'].get('text', ''),
                'prob': hit['_source'].get('prob', 0.0),
                'fps': 25
            }
            # hit['_source'].update(construct_paths(file_list, file_video_list, file_fps_list, image_info))
            # hit['_source']['video_folder'] = video_folder  # Thêm trường video_folder vào kết quả
            results.append(image_info)
        return results


    backup_data = load_json_file(OCR_BACKUP_FILE_PATH)
    return search_in_backup(query, backup_data)


def find_matching_frame(frames_data: List[Dict], sampled_frames: List[int], video_id: str, video_folder: str) -> Optional[Dict]:
    """
    Tìm frame đầu tiên trong sampled_frames mà tồn tại trong frames_data, đồng thời 
    kiểm tra video_id và video_folder khớp nhau.

    :param frames_data: Danh sách các frame dữ liệu.
    :param sampled_frames: Danh sách các frame được lấy mẫu.
    :param video_id: ID của video từ Elasticsearch.
    :param video_folder: Thư mục chứa video từ Elasticsearch.
    :return: Frame đầu tiên khớp nếu tìm thấy, hoặc None nếu không tìm thấy.
    """
    for sampled_frame in sampled_frames:
        for frame in frames_data:
            if (
                int(frame['frame_id']) == sampled_frame and
                frame.get('video_folder') == video_id
            ):
                print(f"Matching frame found: {frame}")
                return frame
        print(f"Frame {sampled_frame} not found in frames_data.")
    return None


def search_asr(es: Elasticsearch, index_name: str, query: str) -> List[Dict]:
    """
    Tìm kiếm ASR trong Elasticsearch hoặc trong backup nếu không có kết quả từ Elasticsearch.

    :param es: Elasticsearch client.
    :param index_name: Tên index để tìm kiếm.
    :param query: Chuỗi tìm kiếm.
    :return: Danh sách kết quả tìm kiếm.
    """
    file_list = load_file_dict(FILE_LIST)
    file_video_list = load_file_dict(FILE_VIDEO_LIST)
    file_fps_list = {item['title']: item['fps'] for item in load_json_file(FILE_FPS_LIST)}
    frames_data = load_json_file(FILE_NAME_FRAME)
    es_results = search_from_elasticsearch(es, index_name, query, 'text')
    results = []

    if es_results:
        for hit in es_results:
            video_name = hit['_source'].get('video_id', '')
            sampled_frames = hit['_source'].get('sampled_frames', [])
            video_folder = hit['_source'].get('video_folder', '')

            # Tìm frame khớp trong frames_data
            matching_frame = find_matching_frame(frames_data, sampled_frames, video_name, video_folder)

            if matching_frame:
                image_info = {
                    'frame_id': matching_frame.get('frame_id', ''),
                    'video_id': video_name,
                    'video_folder': video_folder,
                    'start_frame': hit['_source'].get('start_frame', ''),
                    'fps': hit['_source'].get('fps', ''),
                    'text': hit['_source'].get('text', '')
                }
                results.append(image_info)
            # break
        return results

    # Nếu không có kết quả từ Elasticsearch, tìm trong backup
    backup_data = load_json_file(ASR_BACKUP_FILE_PATH)
    return search_in_backup(query, backup_data, field='text')


def search_object(es: Elasticsearch, index_name: str, query: str, operator: str, value: int) -> List[Dict]:
    """Tìm kiếm đối tượng trong Elasticsearch hoặc trong backup nếu không có kết quả từ Elasticsearch."""
    file_list = load_file_dict(FILE_LIST)
    file_video_list = load_file_dict(FILE_VIDEO_LIST)
    file_fps_list = {item['title']: item['fps'] for item in load_json_file(FILE_FPS_LIST)}

    search_body = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"labels.keyword": query}},
                    {"bool": {"must": [{"exists": {"field": f"label_counts.{query}"}}]}}
                ]
            }
        },
        "size": 300
    }

    # Add the range query only if the value is not None
    if value is not None:
        search_body["query"]["bool"]["must"].append({
            "range": {f"label_counts.{query}": {operator: value}}
        })

    try:
        response = es.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        results = []
        for hit in hits:
            image_info = {
                'frame_id': hit['_source'].get('frame_id', ''),
                'video_id': hit['_source'].get('video_id', ''),
                'video_folder': hit['_source'].get('video_folder', '')
            }
            hit['_source'].update(construct_paths(file_list, file_video_list, file_fps_list, image_info))
            results.append(hit['_source'])
        return results
    except (exceptions.ConnectionError, exceptions.TransportError) as e:
        print(f"Elasticsearch connection error: {e}")
        backup_data = load_json_file(OBJECT_BACKUP_FILE_PATH)
        return search_in_backup(query, backup_data, threshold=50)
