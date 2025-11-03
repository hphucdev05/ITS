
 5. Khởi Chạy Ứng Dụng Streamlit

# (CHỈ DÀNH CHO WINDOWS/PowerShell): Gỡ bỏ giới hạn thực thi tạm thời 

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# KHỞI CHẠY APP
streamlit run app.py

# Kích hoạt

.\venv\Scripts\activate

Dự án này sử dụng mô hình AI (YOLO/Roboflow) để nhận diện các thành phần món ăn từ hình ảnh và tích hợp với Streamlit (GUI của Python) để cung cấp giao diện người dùng thân thiện.

I. Yêu Cầu Hệ Thống

Hệ điều hành: Windows, macOS, hoặc Linux.

Ngôn ngữ: Python 3.8+ (Đã được kiểm thử với Python 3.11).

Cấu hình: Nên có GPU (ít nhất 4GB VRAM) để chạy các mô hình AI như YOLO/Ultralytics nhanh hơn.

II. Hướng Dẫn Cài Đặt và Khởi Chạy

Các lệnh dưới đây được viết cho môi trường PowerShell (Windows) và Terminal (macOS/Linux).

1. Tải Mã Nguồn (Clone Repository)

git clone <URL_repository_của_bạn>
cd <tên_thư_mục_dự_án>


2. Thiết Lập Môi Trường Ảo (Virtual Environment)

Sử dụng môi trường ảo giúp cô lập thư viện và tránh xung đột.
PowerShell (Windows)
# Tạo venv

python -m venv venv




💡 Lưu ý: Sau khi kích hoạt thành công, bạn sẽ thấy (venv) xuất hiện ở đầu dòng lệnh.

3. Cài Đặt Thư Viện Phụ Thuộc (Dependencies)

# Sử dụng file requirements.txt (chứa hơn 200 thư viện, bao gồm streamlit, torch, opencv-python, ultralytics, và roboflow).

pip install -r requirements.txt


4. Thiết Lập API Key (Roboflow)

CẢNH BÁO: Không bao giờ nhập khóa API thật vào file mã nguồn hoặc README. Hãy thiết lập biến môi trường.

# Windows (PowerShell)

# $env:ROBOFLOW_API_KEY="YOUR_API_KEY_HERE"





Ứng dụng sẽ tự động mở trong trình duyệt của bạn (thường là: http://localhost:8501).

III. Cấu Trúc Dự Án

app.py: File chính chứa logic giao diện Streamlit và xử lý gọi mô hình AI.

requirements.txt: Danh sách các thư viện cần thiết.

assets/: Chứa các tài nguyên như ảnh mẫu, icon, v.v. (Tùy chọn)