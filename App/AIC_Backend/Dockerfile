# Sử dụng hình ảnh cơ sở của Python
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file yêu cầu và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
COPY . .

# Cung cấp lệnh để chạy ứng dụng
CMD ["uvicorn", "main:app", "--reload","--host", "0.0.0.0", "--port", "8000"]
