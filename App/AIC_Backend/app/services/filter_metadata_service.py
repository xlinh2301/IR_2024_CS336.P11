import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import META_DATA
# META_DATA = "E:\\CODE\\AIC_2024\\Fastapi\\app\\data\\metadata"

def load_all_metadata() -> Dict[str, Any]:
    all_metadata = {}
    try:
        for filename in os.listdir(META_DATA):
            if filename.endswith('.json'):
                video_id = os.path.splitext(filename)[0]
                # print("video_id: ", video_id)
                metadata_file = os.path.join(META_DATA, filename)
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    all_metadata[video_id] = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading all metadata: {str(e)}")
    return all_metadata

def filter_by_metadata(search_results: Dict[str, List[Dict[str, Any]]], 
                       search_type: str, 
                       publish_day: Optional[int] = None,
                       publish_month: Optional[int] = None,
                       publish_year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Filter search results based on publish_date from metadata of each video for a specific search type.

    Parameters:
    - search_results: Dictionary containing search results for different types (clip, ocr, object, asr).
    - search_type: The type of search results to filter (clip, ocr, object, asr).
    - publish_day: Optional day of the publish date to filter results.
    - publish_month: Optional month of the publish date to filter results.
    - publish_year: Optional year of the publish date to filter results.

    Returns:
    - Filtered results based on publish_date for the specified search_type.
    """
    filtered_results = []

    if search_type in search_results:
        # Load all metadata
        all_metadata = load_all_metadata()

        # Filter results based on publish_date for the selected search type
        for info in search_results[search_type]:
            # Extract video_id depending on the search type
            if search_type == "clip":
                video_id = info['video_id']
            elif search_type == "ocr" or search_type == "asr":
                video_id = info['_source']['video_name']
            elif search_type == "object":
                video_id = info['_source']['video_id']

            # Check if video_id exists in metadata
            if video_id in all_metadata:
                metadata = all_metadata[video_id]
                video_publish_date = datetime.strptime(metadata['publish_date'], "%d/%m/%Y")
                
                # Check the filtering conditions based on day, month, and year
                if ((publish_year is None or video_publish_date.year == publish_year) and
                    (publish_month is None or video_publish_date.month == publish_month) and
                    (publish_day is None or video_publish_date.day == publish_day)):
                    print("publish_year: ", publish_year)
                    print("publish_month: ", publish_month)
                    print("publish_day: ", publish_day)
                    filtered_results.append(info)

    return filtered_results


# # Ví dụ sử dụng hàm
# search_results = {
#     'clip': [
#         {'frame_id': 10840, 'video_id': 'L04_V016', 'video_folder': 'Videos_L04', 'image_path': 'https://drive.google.com/uc?export=view&id=1LmvSKjBsAkvzmD1rpBUBCvKuGN3BCGR9'},
#         {'frame_id': 31024, 'video_id': 'L01_V001', 'video_folder': 'Videos_L01', 'image_path': 'https://drive.google.com/uc?export=view&id=11SEA5mG2DC3wHDHe15XdZia-DVFi9p22'},
#         {'frame_id': 18632, 'video_id': 'L02_V008', 'video_folder': 'Videos_L02', 'image_path': 'https://drive.google.com/uc?export=view&id=1EP5GiyUZZz-i2enN_o0AYTxyUbdazb16'}
#     ],
#     'ocr': [],
#     'object': [],
#     'asr': [],
#     'image': []
# }

# publish_date = '31/10/2023'
# filtered_results = filter_by_metadata(search_results, publish_date)

# for result in filtered_results:
#     print(result)
