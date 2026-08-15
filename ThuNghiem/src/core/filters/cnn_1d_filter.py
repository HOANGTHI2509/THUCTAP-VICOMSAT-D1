import os
import torch
import numpy as np
from collections import deque
import pandas as pd
import sys

# Import kiến trúc mô hình (đảm bảo src nằm trong PYTHONPATH)
try:
    from src.pipeline.train_cnn_1d import CNN1DDenoisingNet
except ImportError:
    # Nếu chạy script từ bên trong src/core/filters, fallback:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from src.pipeline.train_cnn_1d import CNN1DDenoisingNet

# Biến toàn cục để lưu mô hình (tránh việc đọc file nhiều lần)
GLOBAL_CNN_MODEL = None
GLOBAL_WINDOW_SIZE = 20

def load_cnn_model():
    global GLOBAL_CNN_MODEL, GLOBAL_WINDOW_SIZE
    if GLOBAL_CNN_MODEL is not None:
        return True
        
    model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')), "models", "cnn_1d_denoising.pth")
    if not os.path.exists(model_path):
        print(f"[CNN1D] Không tìm thấy mô hình tại {model_path}. Cần chạy train_cnn_1d.py trước.")
        return False
        
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        GLOBAL_WINDOW_SIZE = checkpoint.get('window_size', 20)
        
        model = CNN1DDenoisingNet(window_size=GLOBAL_WINDOW_SIZE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Chuyển sang chế độ inference
        
        GLOBAL_CNN_MODEL = model
        return True
    except Exception as e:
        print(f"[CNN1D] Lỗi khi tải mô hình: {e}")
        return False

class BoLocCNN1D:
    def __init__(self):
        self.is_ready = load_cnn_model()
        self.window_size = GLOBAL_WINDOW_SIZE if self.is_ready else 20
        # Hàng đợi lưu N giá trị nhiên liệu thô
        self.window = deque(maxlen=self.window_size)
        
    def cap_nhat(self, gia_tri_do: float) -> float:
        """
        Nhận giá trị mới và trả về giá trị đã lọc (nếu đủ dữ liệu).
        Nếu chưa đủ dữ liệu cửa sổ, trả về chính giá trị đó.
        """
        # Xử lý giá trị không hợp lệ
        if pd.isna(gia_tri_do) or gia_tri_do <= 0:
            self.window.clear()
            return np.nan
            
        gia_tri_do = float(gia_tri_do)
        self.window.append(gia_tri_do)
        
        # Nếu mô hình chưa sẵn sàng hoặc chưa đủ cửa sổ -> trả về giá trị thô
        if not self.is_ready or len(self.window) < self.window_size:
            return gia_tri_do
            
        # Đã đủ cửa sổ N điểm, tiến hành Inference
        with torch.no_grad():
            # Chuyển thành tensor (Batch=1, Channels=1, SeqLen=window_size)
            # Áp dụng Local Normalization dựa vào điểm GẦN NHẤT (hiện tại) của cửa sổ
            offset = self.window[-1]
            diff = np.array(self.window, dtype=np.float32) - offset
            # CLAMPING
            diff = np.clip(diff, -50.0, 50.0)
            input_array = diff / 100.0
            input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            output_tensor = GLOBAL_CNN_MODEL(input_tensor)
            
            # Khôi phục lại giá trị thực tế
            gia_tri_da_loc = output_tensor.item() * 100.0 + offset
            
        # Chặn giá trị âm (bình xăng không thể âm)
        return max(0.0, gia_tri_da_loc)

def chay_cnn_1d_cho_tat_ca_xe(mau_ten_file: str = "data/processed/CarFuelHistory_Processed_*.csv") -> None:
    """Hàm này chỉ dùng để test nhanh riêng bộ lọc CNN 1D."""
    import glob
    print("=== BẮT ĐẦU CHẠY CNN 1D FILTER ===")
    
    if not load_cnn_model():
        return
        
    danh_sach_file = [f for f in sorted(glob.glob(mau_ten_file)) if "_CNN1D" not in f]
    if not danh_sach_file: return
    
    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang chạy CNN 1D cho xe {ma_xe}...")
        
        df = pd.read_csv(duong_dan_file)
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        df["CNN1D_Filter"] = np.nan
        
        for ma_doan, nhom in df.groupby("SegmentID", sort=False, dropna=False):
            bo_loc = BoLocCNN1D()
            for dong in nhom.itertuples():
                vi_tri = dong.Index
                gia_tri_do = getattr(dong, "FuelLevel", np.nan)
                
                gia_tri_da_loc = bo_loc.cap_nhat(gia_tri_do)
                df.loc[vi_tri, "CNN1D_Filter"] = gia_tri_da_loc
                
        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
        duong_dan_xuat = duong_dan_file.replace(".csv", "_CNN1D.csv")
        df.to_csv(duong_dan_xuat, index=False, encoding="utf-8-sig")
        print(f"Hoàn tất {ma_xe}! Đã xuất: {duong_dan_xuat}")
        
    print("=== ĐÃ CHẠY XONG CHO TẤT CẢ CÁC XE ===")

if __name__ == "__main__":
    chay_cnn_1d_cho_tat_ca_xe()
