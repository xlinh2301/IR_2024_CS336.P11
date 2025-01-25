import os
import json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Đảm bảo rằng đường dẫn đến tệp client_secrets.json là chính xác
client_secrets_path = 'E:\\AIC\\aic_backend\\app\\data\\client_secrets.json'

# Cấu hình PyDrive và xác thực
gauth = GoogleAuth()
gauth.LoadClientConfigFile(client_secrets_path)
gauth.LocalWebserverAuth()  # Cho phép xác thực qua trình duyệt web

# Tạo đối tượng GoogleDrive
drive = GoogleDrive(gauth)

def get_video_ids(drive, folder_id):
    """Lấy ID và tên của các video (MP4) trong thư mục và các thư mục con có tên chứa từ 'video'"""
    video_info_list = []
    folders = [folder_id]
    
    while folders:
        current_folder_id = folders.pop()
        print(f"Đang xử lý thư mục: {current_folder_id}")
        query = f"'{current_folder_id}' in parents and trashed=false"
        page_token = None
        
        while True:
            params = {'q': query}
            if page_token:
                params['pageToken'] = page_token
            
            file_list = drive.ListFile(params).GetList()
            print(f"Đã tìm thấy {len(file_list)} tệp trong thư mục {current_folder_id}")
            for file in file_list:
                file_id = file['id']
                file_title = file['title']
                mime_type = file['mimeType']
                
                if mime_type == 'application/vnd.google-apps.folder':
                    # Nếu là thư mục và tên chứa từ 'video', thêm vào danh sách để xử lý tiếp
                    if 'video' in file_title.lower():
                        folders.append(file_id)
                        print(f"Thêm thư mục con: {file_title}, ID: {file_id}")
                elif mime_type == 'video/mp4':
                    # Chỉ thêm các video có định dạng MP4
                    video_info = {
                        'id': file_id,
                        'title': file_title
                    }
                    video_info_list.append(video_info)
                    print(f"Thêm video: {file_title}, ID: {file_id}")
                else:
                    print(f"Bỏ qua tệp: {file_title} (mimeType: {mime_type})")
            
            # Lấy pageToken cho trang tiếp theo nếu có
            page_token = drive.auth.service.files().list(q=query, pageToken=page_token).execute().get('nextPageToken')
            if not page_token:
                break
    
    return video_info_list

# Thay thế `folder_id` bằng ID của thư mục chứa các tệp video của bạn
folder_id = '1pi5sKtD_PTEijpyUD49oX_NGrB5cc59u'  # ID thư mục gốc
video_info_list = get_video_ids(drive, folder_id)

# Lưu thông tin video vào file JSON mới
output_file_path = 'E:\\AIC\\aic_backend\\app\\file_video_list.json'
with open(output_file_path, 'w') as f:
    json.dump(video_info_list, f, indent=4)

print(f"File JSON mới đã được lưu tại {output_file_path}")