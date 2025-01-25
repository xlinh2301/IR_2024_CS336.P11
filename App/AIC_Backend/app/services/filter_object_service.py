from typing import List, Dict, Optional
from elasticsearch import Elasticsearch, exceptions
import json
from app.config import OBJECT_BACKUP_FILE_PATH, FILE_LIST, FILE_VIDEO_LIST, FILE_FPS_LIST
from rapidfuzz import fuzz  # type: ignore

def load_json_file(file_path: str) -> List[Dict]:
    """Load data from a JSON file and return it as a list of dictionaries."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

def load_file_dict(file_list_path: str) -> Dict[str, str]:
    """Load a file list into a dictionary with filenames as keys and IDs as values."""
    file_list = load_json_file(file_list_path)
    return {item.get('title'): item.get('id') for item in file_list if item.get('title') and item.get('id')}

def construct_paths(file_list: Dict[str, str], file_video_list: Dict[str, str], file_fps_list: Dict[str, float], image_info: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    Construct image and video paths from image information using JSON files containing IDs and filenames.
    
    :param file_list: Dictionary containing image IDs and filenames.
    :param file_video_list: Dictionary containing video IDs and filenames.
    :param file_fps_list: Dictionary containing FPS of videos.
    :param image_info: Image information including 'frame_id', 'video_id', and 'video_folder'.
    :return: A dictionary containing image path, video path, and FPS.
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

def search_filter_object_from_elasticsearch(
    es: Elasticsearch, 
    index_name: str, 
    query: str, 
    top: int, 
    operator: str, 
    value: Optional[int], 
    clip_results: List[Dict]
) -> List[Dict]:
    """Search for objects in Elasticsearch with filters based on provided query and clip results."""
    file_list = load_file_dict(FILE_LIST)
    file_video_list = load_file_dict(FILE_VIDEO_LIST)
    file_fps_list = {item['title']: item['fps'] for item in load_json_file(FILE_FPS_LIST)}
    
    # Extract frame_ids and video_ids from clip results
    frame_ids = [clip['frame_id'] for clip in clip_results]
    video_ids = [clip['video_id'] for clip in clip_results]
    print(f"Frame IDs: {frame_ids}")
    print(f"Video IDs: {video_ids}")
    
    # Initialize search body
    search_body = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"labels.keyword": query}},  # Tìm các đối tượng liên quan tới query
                    {"terms": {"frame_id.keyword": frame_ids}},   # Chỉ tìm trong frame_id được trả về từ CLIP
                    {"terms": {"video_id.keyword": video_ids}}    # Chỉ tìm trong video_id được trả về từ CLIP
                ]
            }
        },
        "size": top  # Số lượng kết quả tối đa
    }

    # Define range operators mapping
    range_operators = {
        "gt": "gt",
        "lt": "lt",
        "gte": "gte",
        "lte": "lte",
        "eq": "eq"
    }
    
    # Add the range query only if the value is not None and operator is valid
    if value is not None and operator in range_operators:
        op = range_operators[operator]
        search_body["query"]["bool"]["must"].append({
            "range": {f"label_counts.{query}": {op: value}}
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
        print(f"Error searching Elasticsearch: {str(e)}")
        return []



def search_filter_object_in_backup(
    query: str, 
    backup_data: List[Dict], 
    top: int, 
    operator: str, 
    value: int, 
    clip_results: List[Dict]
) -> List[Dict]:
    """Search for objects in the backup data with filters based on provided query and clip results."""
    results = []
    file_list = load_file_dict(FILE_LIST)
    file_video_list = load_file_dict(FILE_VIDEO_LIST)
    file_fps_list = {item['title']: item['fps'] for item in load_json_file(FILE_FPS_LIST)}
    
    allowed_pairs = {(clip['frame_id'], clip['video_id']) for clip in clip_results}
    
    for entry in backup_data:
        entry_frame_id = entry.get('frame_id')
        entry_video_id = entry.get('video_id')
        
        if (entry_frame_id, entry_video_id) in allowed_pairs:
            if query.lower() in [name.lower() for name in entry.get('labels', [])]:
                count_value = entry['label_counts'].get(query, 0)

                # Apply the operator to filter based on the count_value
                if operator == 'lt' and count_value < value:
                    results.append(entry)
                elif operator == 'lte' and count_value <= value:
                    results.append(entry)
                elif operator == 'gt' and count_value > value:
                    results.append(entry)
                elif operator == 'gte' and count_value >= value:
                    results.append(entry)
                elif operator == 'eq' and count_value == value:
                    results.append(entry)

                # Add image path information
                frame_id = entry.get('frame_id', '')
                video_id = entry.get('video_id', '')
                video_folder = entry.get('video_folder', '')
                image_info = {
                    'frame_id': frame_id,
                    'video_id': video_id,
                    'video_folder': video_folder
                }
                image_info.update(construct_paths(file_list, file_video_list, file_fps_list, image_info))
                entry.update(image_info)

                # Append the entry to the results
                results.append(entry)
        
        # Break when enough results are found
        if len(results) >= top:
            break

    return results

def search_filter_object(
    es: Elasticsearch, 
    index_name: str, 
    query: str, 
    operator: str, 
    value: Optional[int], 
    clip_results: List[Dict]
) -> List[Dict]:
    """Search for objects either from Elasticsearch or backup data based on query and filters."""
    top = 200
    
    try:
        es_results = search_filter_object_from_elasticsearch(es, index_name, query, top, operator, value, clip_results)
        if es_results:
            print("Successfully connected to Elasticsearch server")
            return es_results
        else:
            raise Exception("No results from Elasticsearch.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Attempting to load data from backup...")
        backup_data = load_json_file(OBJECT_BACKUP_FILE_PATH)
        backup_results = search_filter_object_in_backup(query, backup_data, top, operator, value, clip_results)
        return backup_results
