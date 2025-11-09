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

# --- CẤU HÌNH (CHO LOCAL) ---
@st.cache_resource
def load_yolo_model():
    """Tải model YOLOv8 'best.pt' một lần và cache lại."""
    try:
        model = YOLO("best.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        st.sidebar.success(f"Đã tải model 'best.pt' ({device}).")
        return model
    except Exception as e:
        st.sidebar.error(f"Lỗi tải model: {e}")
        st.error("Không tìm thấy file 'best.pt'.")
        return None

# --- CẤU HÌNH FIREBASE ---
@st.cache_resource
def init_firebase():
    """Khởi tạo kết nối Firebase một lần và cache lại."""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        st.sidebar.success("Đã kết nối Firebase!")
        return db
    except Exception as e:
        st.sidebar.error(f"Lỗi kết nối Firebase: {e}")
        st.error("Không tìm thấy file 'firebase-key.json'.")
        return None

model = load_yolo_model()
db = init_firebase()

# --- HÀM GPX ---
@st.cache_data
def parse_gpx(gpx_file_data):
    gpx = gpxpy.parse(gpx_file_data)
    if not gpx.tracks:
        st.error("File GPX không chứa 'tracks'!")
        return None
    track = gpx.tracks[0]
    segment = track.segments[0]
    points = []
    start_time = None
    for point in segment.points:
        if point.time is None:
            st.error("File GPX không có timestamp.")
            return None
        if start_time is None:
            start_time = point.time
        current_sec = (point.time - start_time).total_seconds()
        points.append({
            "sec": current_sec,
            "lat": point.latitude,
            "lon": point.longitude
        })
    if not points:
        st.error("Không tìm thấy điểm GPS.")
        return None
    st.sidebar.success(f"Đã đọc {len(points)} điểm GPS.")
    return points

def get_gps_for_second(gpx_data, video_second):
    if not gpx_data:
        return None, None
    closest_point = min(gpx_data, key=lambda p: abs(p['sec'] - video_second))
    return closest_point['lat'], closest_point['lon']

# --- XÓA DATABASE ---
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

tab1, tab2 = st.tabs(["Xử lý Video (Nhanh)", "Bản đồ Ổ gà (Dữ liệu)"])

# --- TAB 1 ---
with tab1:
    st.header("Xử lý Video (Nhanh hơn 10–30x, không cần GPU)")
    st.sidebar.header("⚙️ Tùy chỉnh")

    CONF_THRESHOLD = st.sidebar.slider(
        "Ngưỡng tin cậy:",
        min_value=0.1, max_value=1.0, value=0.4, step=0.05
    )

    uploaded_video_file = st.file_uploader("1️⃣ Chọn video (.mp4)", type=["mp4"])
    uploaded_gpx_file = st.file_uploader("2️⃣ Chọn file GPX (tùy chọn)", type=["gpx"])
    start_button = st.button("🚀 Bắt đầu xử lý")

    if start_button and uploaded_video_file and model and db:
        temp_video_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video_file.read())

        # --- Đọc GPX ---
        gpx_data = None
        if uploaded_gpx_file is not None:
            try:
                gpx_data = parse_gpx(uploaded_gpx_file.getvalue())
            except Exception as e:
                st.error(f"Lỗi đọc GPX: {e}")
                gpx_data = None
        else:
            st.warning("Không có file GPX — sẽ dùng tọa độ giả định.")

        stframe = st.empty()
        progress = st.progress(0, text="Đang xử lý video...")

        # --- Thông tin video ---
        cap = cv2.VideoCapture(temp_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # --- VideoWriter ---
        output_path = os.path.join(tempfile.gettempdir(), "output_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        base_lat, base_lon = 10.7769, 106.6954
        potholes = []
        frame_idx = 0  # ✅ Khởi tạo biến đếm frame

        results = model.predict(
            source=temp_video_path,
            conf=CONF_THRESHOLD,
            stream=True
        )

        # --- Xử lý từng frame ---
        for r in results:
            frame = r.orig_img.copy()
            boxes = r.boxes

            # --- Vẽ khung và chữ “Ổ gà” ---
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "O ga", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            out.write(frame)

            # Hiển thị mỗi 5 frame để đỡ lag
            if frame_idx % 5 == 0:
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            progress.progress(min(frame_idx / total_frames, 1.0),
                              text=f"Đang xử lý {frame_idx}/{total_frames} frames")

            # --- Lưu kết quả ổ gà ---
            if len(r.boxes) > 0:
                video_second = frame_idx / fps
                lat, lon = get_gps_for_second(gpx_data, video_second)
                if lat is None:
                    lat = base_lat + random.uniform(-0.00005, 0.00005)
                    lon = base_lon + random.uniform(-0.00005, 0.00005)
                for box in r.boxes:
                    potholes.append({
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "confidence": float(box.conf[0])
                    })

            frame_idx += 1  # ✅ Tăng frame index

        out.release()
        os.remove(temp_video_path)
        progress.empty()

        # --- Lưu Firebase ---
        if potholes:
            for p in potholes:
                db.collection("potholes").add(p)
            st.success(f"✅ Hoàn tất! Phát hiện {len(potholes)} ổ gà.")
        else:
            st.warning("Không phát hiện ổ gà nào.")

        st.video(output_path)

    elif not uploaded_video_file:
        st.info("Vui lòng tải lên video để bắt đầu.")
    elif db is None or model is None:
        st.error("Lỗi khởi tạo model hoặc Firebase.")

# --- TAB 2 ---
with tab2:
    st.header("Bản đồ Ổ gà (Dữ liệu)")
    if db:
        potholes_ref = db.collection("potholes")
        docs = potholes_ref.stream()
        pothole_data = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        if pothole_data:
            df = pd.DataFrame(pothole_data)
            st.dataframe(df)
        else:
            st.info("Chưa có dữ liệu ổ gà nào trong Firestore.")
        if st.button("🗑️ Xóa toàn bộ dữ liệu"):
            deleted = delete_all_potholes(db)
            st.warning(f"Đã xóa {deleted} bản ghi khỏi Firestore.")
    else:
        st.error("Firebase chưa sẵn sàng.")
