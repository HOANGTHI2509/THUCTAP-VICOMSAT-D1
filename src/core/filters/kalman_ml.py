import os
import pickle
import numpy as np
import pandas as pd
import warnings

# Tắt cảnh báo của sklearn khi dự đoán 1 mẫu đơn lẻ
warnings.filterwarnings("ignore", category=UserWarning)

# Load mô hình AI một lần duy nhất ở cấp module (để tránh load lại hàng ngàn lần)
GLOBAL_MODEL = None
model_path = "models/fuel_state_classifier.pkl"
if os.path.exists(model_path):
    try:
        import pickle
        with open(model_path, "rb") as f:
            GLOBAL_MODEL = pickle.load(f)
    except Exception as e:
        print(f"Lỗi khi load mô hình AI: {e}")

def predict_batch(df):
    if GLOBAL_MODEL is None:
        return np.zeros(len(df), dtype=int)
    
    features = ['Speed', 'Acceleration', 'DeltaFuel', 'RollingStd', 'TimeGapMinutes']
    X = df[features].copy()
    X['Speed'] = X['Speed'].fillna(0.0)
    if 'Acceleration' in X.columns:
        X['Acceleration'] = X['Acceleration'].fillna(0.0)
    else:
        X['Acceleration'] = 0.0
    X['DeltaFuel'] = X['DeltaFuel'].fillna(0.0)
    X['RollingStd'] = X['RollingStd'].fillna(0.0)
    X['TimeGapMinutes'] = X['TimeGapMinutes'].fillna(5.0)
    
    try:
        return GLOBAL_MODEL.predict(X)
    except:
        return np.zeros(len(df), dtype=int)

class BoLocKalmanAI:
    def __init__(
        self,
        trang_thai_ban_dau: float,
        r_co_ban: float = 9.0,
        sai_so_uoc_luong_ban_dau: float = 9.0,
        q_co_ban: float = 1.0,
        nhip_cho_xac_nhan: int = 3
    ):
        self.x = float(trang_thai_ban_dau)
        self.P = float(sai_so_uoc_luong_ban_dau)
        self.r_base = r_co_ban
        self.q_base = q_co_ban
        self.nhip_cho_xac_nhan = nhip_cho_xac_nhan
        self.so_nhip_thay_doi = 0
                
    def cap_nhat(
        self, 
        gia_tri_do: float, 
        ty_le_dt: float = 1.0,
        nhan_ai: int = 0
    ) -> float:
        z = float(gia_tri_do)
        
        # ÁP DỤNG PERSISTENCE CHO LABEL AI
        if nhan_ai == 2:
            self.so_nhip_thay_doi += 1
        else:
            self.so_nhip_thay_doi = 0
        
        # AI QUYẾT ĐỊNH THAM SỐ VÀ CÓ CHỜ NHỊP
        if nhan_ai == 1:
            # Nhiễu gai (Spike): Ép R cực to để từ chối điểm dữ liệu này
            R_thich_nghi = 1000.0
            Q_thich_nghi = self.q_base * max(ty_le_dt, 0.1)
        elif nhan_ai == 2 and self.so_nhip_thay_doi >= self.nhip_cho_xac_nhan:
            # Tụt/Bơm thật (True Change) VÀ Đã xác nhận: Ép R nhỏ, Q to để bám theo ngay
            R_thich_nghi = 1.0
            Q_thich_nghi = 100.0
        elif nhan_ai == 2 and self.so_nhip_thay_doi < self.nhip_cho_xac_nhan:
            # Tụt/Bơm nghi ngờ, chưa đủ 3 nhịp -> Coi như nhiễu
            R_thich_nghi = 1000.0
            Q_thich_nghi = self.q_base * max(ty_le_dt, 0.1)
        else:
            # Ổn định (Stable): Dùng tham số cơ bản
            R_thich_nghi = self.r_base
            Q_thich_nghi = self.q_base * max(ty_le_dt, 0.1)
            
        # Các bước Kalman Filter tiêu chuẩn
        x_du_doan = self.x
        P_du_doan = self.P + Q_thich_nghi
        
        phan_du = z - x_du_doan
        
        K = P_du_doan / (P_du_doan + R_thich_nghi)
        self.x = x_du_doan + K * phan_du
        self.x = max(0.0, self.x)
        
        self.P = (1.0 - K)**2 * P_du_doan + K**2 * R_thich_nghi
        
        return self.x
