# Sử dụng một base image Python chính thức, gọn nhẹ
FROM python:3.11-slim

# Thiết lập các biến môi trường cho Python
ENV PYTHONDONTWRITEBYTECODE 1  # Ngăn Python tạo file .pyc
ENV PYTHONUNBUFFERED 1         # Buộc stdout/stderr không bị buffer, log xuất hiện ngay

# Tạo và đặt thư mục làm việc bên trong container
WORKDIR /app

# (Tùy chọn nhưng khuyến nghị cho security) Tạo user và group không phải root để chạy ứng dụng
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Cài đặt các gói hệ thống cần thiết (nếu có)
# Ví dụ: nếu một thư viện Python nào đó cần gcc hoặc các thư viện C khác
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements.txt trước để tận dụng Docker layer caching
COPY requirements.txt .

# Cài đặt các Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của ứng dụng vào thư mục /app trong image
# QUAN TRỌNG: Hãy đảm bảo bạn có file .dockerignore để loại trừ các file/thư mục không cần thiết
COPY . .
# Ví dụ: nếu bạn muốn copy có chọn lọc hơn:
# COPY ./src ./src
# COPY ./config.py .
# COPY ./knowledge_graph.graphml ./knowledge_graph.graphml # Nếu bạn muốn copy file KG vào image

# Thay đổi chủ sở hữu của thư mục ứng dụng cho user đã tạo (nếu bạn tạo non-root user)
RUN chown -R appuser:appgroup /app

# Chuyển sang non-root user
USER appuser

# Expose cổng mà ứng dụng FastAPI sẽ chạy bên trong container
EXPOSE 8000

# Lệnh để chạy ứng dụng FastAPI khi container khởi động
# Các API keys và cấu hình nhạy cảm sẽ được truyền vào qua biến môi trường khi chạy container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]