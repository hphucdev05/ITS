TS - Pothole Detection System (YOLOv8 + Streamlit + Firebase + Colab GPU)
🧩 Giới thiệu

Ứng dụng phát hiện ổ gà (pothole detection) trong video đường giao thông bằng mô hình YOLOv8.
Dự án hỗ trợ:

Chạy trên CPU (local hoặc Streamlit Cloud)

Kết nối Colab GPU qua API (nhanh hơn nhiều)

Tích hợp Firebase để lưu trữ video và metadata

⚙️ 1. Cấu trúc thư mục
ITS-Pothole-Detection/
│
├── app.py                # Ứng dụng Streamlit chính
├── best.pt                   # File model YOLO đã huấn luyện
├── colab_api_server.ipynb    # Colab notebook chạy API GPU
├── requirements.txt          # Thư viện cần thiết
└── README.md                 # File hướng dẫn này

🧠 2. Cài đặt môi trường
Nếu bạn chạy LOCAL / CODESPACE / STREAMLIT CLOUD

Cài thư viện:

pip install -r requirements.txt


File requirements.txt nên gồm:

streamlit
ultralytics
opencv-python
firebase-admin
requests
flask
flask-cors
pyngrok

🚀 3. Chạy ứng dụng (CPU mode - bình thường)
streamlit run app.py


👉 Ứng dụng mở ở địa chỉ:
http://localhost:8501

Sau đó:

Upload video đường (mp4, mov, avi, mkv)

Ứng dụng sẽ xử lý bằng YOLOv8 (CPU)

Hiển thị kết quả phát hiện ổ gà và upload lên Firebase

⚡ 4. Chạy nhanh với GPU (Colab Integration)
Bước 1️⃣: Mở notebook colab_api_server.ipynb trong Google Colab

Upload các file cần thiết:

best.pt

colab_api_server.ipynb

Bước 2️⃣: Chạy các cell trong notebook

Khi chạy, bạn sẽ thấy đoạn log như sau:

🚀 API running on: https://your-ngrok-url.ngrok-free.dev
 * Running on http://127.0.0.1:5000


➡️ Copy link https://your-ngrok-url.ngrok-free.dev (đường link thật của bạn).

Bước 3️⃣: Sửa file app.py (ở dòng 20–21)

Thay:

API_URL = "https://alaina-debentured-earnestine.ngrok-free.dev/predict"


Bằng:

API_URL = "https://your-ngrok-url.ngrok-free.dev/predict"

Bước 4️⃣: Chạy Streamlit (client)

Trong Codespace hoặc máy bạn:

streamlit run app.py


Ứng dụng Streamlit sẽ gửi video tới Colab GPU, Colab xử lý YOLOv8,
và trả lại video đã có bounding boxes (ổ gà được khoanh vùng).

☁️ 5. Firebase Integration (tùy chọn)

Để bật upload video & lưu kết quả, thêm file secret trong Streamlit:

Vào Streamlit Cloud → Settings > Secrets

Dán key Firebase JSON vào:

[FIREBASE_KEY]
{
  "type": "...",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "...",
  "client_email": "...",
  ...
}


Ứng dụng sẽ tự động:

Upload video gốc / kết quả lên Firebase Storage

Lưu metadata vào Firestore (detections collection)

🧠 6. Nguyên lý hoạt động
Thành phần	Vai trò
Streamlit	Giao diện web, upload video, hiển thị kết quả
YOLOv8	Mô hình phát hiện ổ gà
Firebase	Lưu trữ video và kết quả
Flask (Colab)	API chạy YOLO trên GPU
Ngrok	Tạo public link để Streamlit gửi request tới Colab
🧪 7. API Test nhanh (không cần Streamlit)

Nếu bạn muốn test riêng API trên Colab GPU:

curl -X POST -F "file=@road.mp4" https://your-ngrok-url.ngrok-free.dev/predict --output result.mp4


→ File result.mp4 sẽ là video có bounding boxes.

🧭 8. Kết quả đầu ra

Video gốc có bounding boxes quanh các ổ gà.

Thống kê số ổ gà (potholes) và số frame (frames) phát hiện.

Kết quả upload Firebase (nếu bật).

GPU inference ~30 FPS (so với ~3 FPS CPU).

🧩 9. Ghi chú hiệu năng
Mode	Tốc độ	Phù hợp khi
🧠 CPU (Codespace/Cloud)	~3–5 FPS	demo nhẹ, không có GPU
⚡ GPU (Colab)	~25–40 FPS	demo đồ án, video thực tế
🏁 10. Liên kết quan trọng

Streamlit App: chạy bằng streamlit run app.py

Colab API (GPU): colab_api_server.ipynb

Firebase Console: https://console.firebase.google.com

YOLO Docs: https://docs.ultralytics.com
