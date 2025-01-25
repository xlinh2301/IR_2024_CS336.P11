from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from app.services.faiss_service import search_faiss, search_image
from app.services.elasticsearch_service import search_ocr, search_object, search_asr
from app.services.filter_metadata_service import filter_by_metadata  # Import directly
from app.services.filter_object_service import search_filter_object  # Import directly
from elasticsearch import Elasticsearch
from datetime import datetime
from app.config import CLIENT_SECRETS, CREDENTIALS_PATH, FILE_LIST
from pydrive.auth import GoogleAuth, RefreshError
from pydrive.drive import GoogleDrive
from oauth2client.client import HttpAccessTokenRefreshError
import os
import numpy as np 
app = FastAPI()

if not os.path.exists(CLIENT_SECRETS):
    raise FileNotFoundError(f"Client secrets file not found at {CLIENT_SECRETS}")

# # Cấu hình PyDrive và xác thực
# gauth = GoogleAuth()
# gauth.LoadClientConfigFile(CLIENT_SECRETS)

# # Kiểm tra xem tệp credentials.json có tồn tại không
# if os.path.exists(CREDENTIALS_PATH):
#     gauth.LoadCredentialsFile(CREDENTIALS_PATH)
#     if gauth.credentials is None or gauth.access_token_expired:
#         try:
#             # Nếu token hết hạn, làm mới token
#             gauth.Refresh()
#         except (RefreshError, HttpAccessTokenRefreshError):
#             # Nếu không thể làm mới token, thực hiện xác thực qua trình duyệt web
#             gauth.LocalWebserverAuth()
#             # Lưu token vào tệp credentials.json
#             gauth.SaveCredentialsFile(CREDENTIALS_PATH)
#     else:
#         # Nếu token hợp lệ, sử dụng token hiện tại
#         gauth.Authorize()
# else:
#     # Nếu không có tệp credentials.json, thực hiện xác thực qua trình duyệt web
#     gauth.LocalWebserverAuth()
#     # Lưu token vào tệp credentials.json
#     gauth.SaveCredentialsFile(CREDENTIALS_PATH)

# # Tạo đối tượng GoogleDrive
# drive = GoogleDrive(gauth)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Elasticsearch
es = Elasticsearch(['http://localhost:9200'])

@app.post("/app/search")
async def search_all(
    queries: Dict[str, Optional[str]], 
    operator: Optional[str] = "gte", 
    value: Optional[int] = 1, 
    publish_day: Optional[int] = None,
    publish_month: Optional[int] = None,
    publish_year: Optional[int] = None,
    object_as_filter: Optional[bool] = False  # Thêm cờ để quyết định cách sử dụng object search
):
    """
    Endpoint to perform combined search from multiple sources: CLIP, OCR, Object, ASR, and Image.

    Parameters:
    - queries: Dictionary containing queries for CLIP, OCR, Object, ASR, and Image.
    - operator: Comparison operator for the range condition in object queries. Can be "gte", "gt", "lte", "lt", or "eq".
    - value: Value of label_count to compare in the object query.
    - publish_day: Day of the publish date to filter results.
    - publish_month: Month of the publish date to filter results.
    - publish_year: Year of the publish date to filter results.

    Returns:
    - Combined search results from various sources.
    """
    print(f"Queries: {queries}")
    print(f"Operator: {operator}")
    print(f"Value: {value}")

    results = {
        "clip": [],
        "ocr": [],
        "object": [],
        "asr": [],
        "image": []
    }

    try:
        print("Vô đây")
        # Perform searches in respective services
        if "clip" in queries and queries["clip"]:
            results["clip"] = search_faiss(queries["clip"])
            # print("Clip: ", results["clip"])

        if "ocr" in queries and queries["ocr"]:
            results["ocr"] = search_ocr(es, "ocr_video_1", queries["ocr"])

        if "asr" in queries and queries["asr"]:
            results["asr"] = search_asr(es, "asr_video_1", queries["asr"])

        if "image_url" in queries and queries["image_url"]:
            results["image"] = search_image(queries["image_url"])

        # # Ensure operator and value are defined
        # operator = queries.get("operator")
        # value = queries.get("value")

        # Ensure results["clip"] is initialized
        results["clip"] = results.get("clip", [])

        # Kiểm tra cách sử dụng object search
        if object_as_filter:
            print("Vô đây")
            if "object" in queries and queries["object"] and results["clip"]:
                print("Results clip: ", results["clip"])
                print("operator: ", operator)
                print("value: ", value)
                combined_results = search_filter_object(es, "object_detection", queries["object"], operator, value, results["clip"])
            else:
                combined_results = combine_results(results)
        else:
            if "object" in queries and queries["object"]:
                results["object"] = search_object(es, "object_detection", queries["object"], operator, value)
            combined_results = combine_results(results)

        # Filter results by publish_date if any date components are provided
        if publish_day or publish_month or publish_year:
            # Construct the publish_date to use for filtering
            try:
                publish_date = datetime(
                    year=publish_year if publish_year else 1, 
                    month=publish_month if publish_month else 1, 
                    day=publish_day if publish_day else 1
                ).date()  # Use .date() to get just the date part
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date provided.")
            
            publish_day = publish_day if publish_day else None
            publish_month = publish_month if publish_month else None
            publish_year = publish_year if publish_year else None

            # Filter results by publish_date
            for key in combined_results:
                if queries.get(key):
                    combined_results[key] = filter_by_metadata(combined_results, key, publish_day, publish_month, publish_year)

        return combined_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def combine_results(results: Dict[str, list]) -> Dict[str, list]:
    """
    Combine search results from various sources.

    Parameters:
    - results: Dictionary containing search results from CLIP, OCR, Object, ASR, and Image.

    Returns:
    - Dictionary containing combined results from various sources.
    """
    combined = {
        "clip": results["clip"],
        "ocr": results["ocr"],
        "object": results["object"],
        "asr": results["asr"],
        "image": results["image"]
    }
    return combined


@app.post("/app/search-image-similar")
async def search_image_similar(image_path: str):
    """
    Tìm kiếm hình ảnh tương tự sử dụng CLIP và FAISS.

    Parameters:
    - image_path: Đường dẫn của hình ảnh cần truy vấn.

    Returns:
    - Danh sách kết quả hình ảnh tương tự.
    """
    try:
        print(f"Searching similar images for: {image_path}")
        
        # Tìm kiếm hình ảnh tương tự bằng CLIP qua FAISS
        similar_images = search_faiss(None,image_path)
        
        # Chuyển đổi kết quả (nếu là mảng NumPy) thành danh sách Python
        similar_images = similar_images.tolist() if isinstance(similar_images, np.ndarray) else similar_images
        
        # Kiểm tra nếu không tìm thấy hình ảnh tương tự
        if not similar_images:
            raise HTTPException(status_code=404, detail="No similar images found.")
        
        return {"similar_images": similar_images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))