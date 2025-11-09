import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import uvicorn
import os # Cần cho uvicorn

# --- CẤU HÌNH FIREBASE ---
# (KHÔNG DÙNG CACHE STREAMLIT NỮA)
def init_firebase():
    """Khởi tạo kết nối Firebase."""
    try:
        # Kiểm tra xem app đã được khởi tạo chưa
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase-key.json") 
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Đã kết nối Firebase!") # Báo cho terminal biết
        return db
    except Exception as e:
        print(f"Lỗi kết nối Firebase: {e}")
        return None

db = init_firebase()
app = FastAPI() # Khởi tạo app FastAPI

# --- ĐỊNH NGHĨA DỮ LIỆU ĐẦU VÀO ---
class PotholeReport(BaseModel):
    latitude: float
    longitude: float
    confidence: float = 0.0 

# --- TẠO "CỬA" (ENDPOINT) CHÍNH ---

@app.get("/")
def read_root():
    """Endpoint gốc để kiểm tra xem API có 'sống' không."""
    return {"message": "Chào mừng đến với API Phát hiện Ổ gà! (ITS Project)"}

@app.post("/report_pothole")
def report_pothole(report: PotholeReport):
    """
    Đây là 'cửa' để các app di động của tài xế gửi dữ liệu về.
    Nó nhận (lat, lon) và lưu vào Firebase.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Lỗi server: Không kết nối được database.")
        
    try:
        pothole_data = {
            "latitude": report.latitude,
            "longitude": report.longitude,
            "confidence": report.confidence,
            "source": "mobile_app", # Đánh dấu dữ liệu này đến từ app
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        db.collection("potholes").add(pothole_data)
        
        return {"status": "success", "message": "Đã nhận và lưu báo cáo ổ gà."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi server khi lưu dữ liệu: {e}")

# --- CẤU HÌNH CHẠY SERVER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)