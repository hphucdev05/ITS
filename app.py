import os
import streamlit as st
import cv2
import tempfile
import numpy as np
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError

# --- CẤU HÌNH ---
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="5akRXX55svon1GhOHBLl"  # Private API Key của bạn
)

MODEL_ID = "o-ga_-18k-imges-wwo6c/1"  # Model ID chính xác của bạn


# --- HÀM VẼ ---
def draw_predictions(frame, predictions, conf_threshold=0.8):
    """Vẽ các bounding box và nhãn lên frame."""
    for pred in predictions:
        confidence = pred['confidence']
        if confidence < conf_threshold:
            continue  # Bỏ qua dự đoán có độ tin cậy thấp

        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        class_name = pred['class']

        # Màu theo mức độ tin cậy
        color = (0, 255, 0) if confidence > 0.9 else (0, 165, 255)

        # Vẽ hộp và nhãn
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(layout="wide")
st.title("🕳️ Ứng dụng phát hiện ổ gà")
st.write("Tải lên ảnh hoặc video để phát hiện ổ gà.")

# --- CÀI ĐẶT THAM SỐ ---
st.sidebar.header("⚙️ Tùy chỉnh hiển thị")
CONF_THRESHOLD = st.sidebar.slider(
    "Ngưỡng tin cậy tối thiểu để hiển thị ổ gà:",
    min_value=0.5, max_value=1.0, value=0.8, step=0.05
)

uploaded_file = st.file_uploader(
    "Chọn ảnh (.jpg, .png) hoặc video (.mp4)", type=["jpg", "png", "mp4"]
)

if uploaded_file is not None:
    file_type = uploaded_file.type

    col1, col2 = st.columns(2)

    with col1:
        st.header("File gốc")
        if "image" in file_type:
            st.image(uploaded_file, caption="Ảnh gốc")
        else:
            st.video(uploaded_file)

    with col2:
        st.header("Kết quả nhận diện")

        try:
            # Xử lý ảnh
            if "image" in file_type:
                with st.spinner("Đang xử lý ảnh..."):
                    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(img_bytes, 1)
                    result = CLIENT.infer(image, model_id=MODEL_ID)
                    predictions = result.get('predictions', [])
                    image_with_boxes = draw_predictions(image, predictions, CONF_THRESHOLD)
                    st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB),
                             caption="Ảnh đã nhận diện")

            # Xử lý video
            elif "video" in file_type:
                with st.spinner("Đang xử lý video..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_file.read())
                    cap = cv2.VideoCapture(tfile.name)

                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out_path = os.path.join(tempfile.gettempdir(), "output_detected.mp4")
                    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                    if not out.isOpened():
                        st.error("❌ Không thể mở VideoWriter. Kiểm tra codec hoặc đường dẫn.")

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0, text="Đang xử lý...")

                    frame_count = 0
                    predictions = []

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Giảm tần suất infer để tăng tốc
                        if frame_count % 5 == 0:
                            try:
                                result = CLIENT.infer(frame.copy(), model_id=MODEL_ID)
                                predictions = result.get('predictions', [])
                            except Exception:
                                predictions = []

                        frame_with_boxes = draw_predictions(frame.copy(), predictions, CONF_THRESHOLD)
                        out.write(frame_with_boxes)

                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames,
                                              text=f"Đã xử lý {frame_count}/{total_frames} frames")

                    cap.release()
                    out.release()
                    tfile.close()
                    os.remove(tfile.name)

                    progress_bar.empty()
                    st.video(out_path)
                    st.success("✅ Video đã xử lý xong và có kết quả nhận diện.")

        except HTTPCallErrorError as e:
            if e.status_code == 403:
                st.error(
                    "**Lỗi xác thực (403 Forbidden)**\n\n"
                    "Vui lòng kiểm tra lại 2 thông tin sau trong file `app.py`:\n"
                    "1. **`api_key`**: Đảm bảo bạn đã dán đúng Private API Key.\n"
                    "2. **`MODEL_ID`**: Đảm bảo đây là ID model của bạn (ví dụ: `pothole-project/3`).\n\n"
                    f"Chi tiết lỗi từ server: *{e.api_message}*"
                )
            else:
                st.error(f"Đã xảy ra lỗi khi kết nối đến Roboflow: {e}")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi không xác định: {e}")
