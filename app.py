import os
import streamlit as st
import cv2
import tempfile
import numpy as np
import datetime
import random
import pandas as pd
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import gpxpy
import gpxpy.gpx
import torch

# --- HÀM DETECT ---
def detect_video(model, cap, conf_threshold=0.4):
    """Generator xử lý từng frame bằng YOLOv8."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        yield frame, results[0]

# --- CẤU HÌNH YOLO ---
@st.cache_resource
def load_yolo_model():
    """Tải model YOLOv8 một lần."""
    try:
        model = YOLO("best.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        st.sidebar.success(f"Đã tải model 'best.pt' ({device}).")
        return model
    except Exception as e:
        st.sidebar.error(f"Lỗi tải model: {e}")
        return None

# --- CẤU HÌNH FIREBASE ---
@st.cache_resource
def init_firebase():
    """Kết nối Firebase."""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        st.sidebar.success("Đã kết nối Firebase!")
        return db
    except Exception as e:
        st.sidebar.error(f"Lỗi Firebase: {e}")
        return None

model = load_yolo_model()
db = init_firebase()

# --- HÀM GPX ---
@st.cache_data
def parse_gpx(gpx_file_data):
    gpx = gpxpy.parse(gpx_file_data)
    if not gpx.tracks:
        return None
    track = gpx.tracks[0]
    segment = track.segments[0]
    points = []
    start_time = None
    for point in segment.points:
        if point.time is None:
            continue
        if start_time is None:
            start_time = point.time
        current_sec = (point.time - start_time).total_seconds()
        points.append({
            "sec": current_sec,
            "lat": point.latitude,
            "lon": point.longitude
        })
    return points if points else None

def get_gps_for_second(gpx_data, video_second):
    if not gpx_data:
        return None, None
    closest_point = min(gpx_data, key=lambda p: abs(p['sec'] - video_second))
    return closest_point['lat'], closest_point['lon']

# --- XÓA DỮ LIỆU ---
def delete_all_potholes(db_client):
    batch_size = 50
    docs = db_client.collection("potholes").limit(batch_size).stream()
    deleted = 0
    for doc_batch in iter(lambda: list(docs), []):
        batch = db_client.batch()
        for doc in doc_batch:
            batch.delete(doc.reference)
            deleted += 1
        batch.commit()
        docs = db_client.collection("potholes").limit(batch_size).stream()
    st.cache_data.clear()
    return deleted

# --- GIAO DIỆN ---
st.set_page_config(layout="wide")
st.title("🕳️ Dashboard Phát hiện Ổ gà (ITS) - YOLOv8 (Tối ưu CPU)")

st.sidebar.header("⚙️ Cấu hình")
CONF_THRESHOLD = st.sidebar.slider("Ngưỡng tin cậy:", 0.1, 1.0, 0.4, 0.05)

uploaded_video_file = st.file_uploader("🎥 Tải video (.mp4)", type=["mp4"])
uploaded_gpx_file = st.file_uploader("📍 Tải file GPX (tùy chọn)", type=["gpx"])

# 🚀 TỰ ĐỘNG XỬ LÝ KHI CÓ VIDEO
if uploaded_video_file and model and db:
    st.header("🔄 Đang xử lý video (tự động)...")
    progress = st.progress(0, text="Khởi tạo mô hình...")

    temp_video_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video_file.read())

    # GPX (nếu có)
    gpx_data = None
    if uploaded_gpx_file is not None:
        try:
            gpx_data = parse_gpx(uploaded_gpx_file.getvalue())
            st.sidebar.success("Đã đọc file GPX.")
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc GPX: {e}")
    else:
        st.sidebar.warning("Không có GPX, dùng tọa độ giả định.")

    # Thông tin video
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ghi video output
    output_path = os.path.join(tempfile.gettempdir(), "output_detected.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    stframe = st.empty()
    base_lat, base_lon = 10.7769, 106.6954
    potholes = []

    # Duyệt từng frame
    for idx, (frame, result) in enumerate(detect_video(model, cap, CONF_THRESHOLD)):
        annotated = frame.copy()
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "O ga", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                video_second = idx / fps
                lat, lon = get_gps_for_second(gpx_data, video_second)
                if lat is None:
                    lat = base_lat + random.uniform(-0.00005, 0.00005)
                    lon = base_lon + random.uniform(-0.00005, 0.00005)
                potholes.append({
                    "latitude": lat,
                    "longitude": lon,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "confidence": conf
                })

        out.write(annotated)
        if idx % 5 == 0:
            stframe.image(annotated, channels="BGR", use_container_width=True)
        progress.progress(min(idx / total_frames, 1.0), text=f"Đang xử lý {idx}/{total_frames}...")

    cap.release()
    out.release()
    progress.empty()

    # Lưu Firebase
    if potholes:
        for p in potholes:
            db.collection("potholes").add(p)
        st.success(f"✅ Hoàn tất! Phát hiện {len(potholes)} ổ gà.")
    else:
        st.warning("Không phát hiện ổ gà nào.")

    st.video(output_path)
    st.info("Video đã được xử lý và lưu. Dashboard dữ liệu sẽ xuất hiện tự động sau khi có kết quả.")

# 🧭 HIỂN THỊ DASHBOARD SAU KHI XONG
elif db:
    potholes_ref = db.collection("potholes")
    docs = potholes_ref.stream()
    pothole_data = [{"id": doc.id, **doc.to_dict()} for doc in docs]
    if pothole_data:
        st.header("📊 Dữ liệu Ổ gà từ Firestore")
        df = pd.DataFrame(pothole_data)
        st.dataframe(df)
        if st.button("🗑️ Xóa toàn bộ dữ liệu"):
            deleted = delete_all_potholes(db)
            st.warning(f"Đã xóa {deleted} bản ghi khỏi Firestore.")
    else:
        st.info("📭 Chưa có dữ liệu ổ gà nào — hãy tải video để bắt đầu xử lý.")
