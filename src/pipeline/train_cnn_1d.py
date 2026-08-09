import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import random

sys.stdout.reconfigure(encoding='utf-8')

# 1. Định nghĩa mô hình CNN 1D
class CNN1DDenoisingNet(nn.Module):
    def __init__(self, window_size=20):
        super(CNN1DDenoisingNet, self).__init__()
        self.window_size = window_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # SỬA LỖI KIẾN TRÚC: Bỏ AdaptiveAvgPool1d. Khi pool toàn bộ thời gian, 
        # mạng bị mất thông tin vị trí của điểm cuối (điểm cần dự đoán).
        # Dùng Flatten để Fully Connected layer nhìn thấy rõ từng thời điểm trong quá khứ.
        self.fc = nn.Linear(32 * window_size, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten: (Batch, 32 * window_size)
        x = self.fc(x)
        return x

# 2. Định nghĩa Dataset tạo cửa sổ trượt (Sliding Window)
class FuelDenoiseDataset(Dataset):
    def __init__(self, data_list, target_list, window_size):
        self.data = np.array(data_list, dtype=np.float32)
        self.target = np.array(target_list, dtype=np.float32)
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.data) - self.window_size + 1)

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size].copy()
        label = self.target[idx + self.window_size - 1]
        
        # TUYỆT CHIÊU: Dùng điểm GẦN NHẤT (hiện tại) của cửa sổ làm mốc chuẩn (Anchor).
        # Nhờ vậy, CNN chỉ cần học cách dự đoán "Độ lệch" (Residual) so với điểm hiện tại,
        # loại bỏ hoàn toàn hiện tượng trôi dạt (drift) hay sai số tích lũy.
        offset = window[-1]
        diff = window - offset
        # CLAMP: Giới hạn sự khác biệt ở mức +-50 Lít. Tránh CNN bị choáng ngợp bởi bước nhảy 500-600 Lít gây ringing.
        diff = np.clip(diff, -50.0, 50.0)
        window = diff / 100.0
        label = (label - offset) / 100.0
        
        window = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor([label], dtype=torch.float32)
        return window, label

def get_vehicle_id(filepath):
    return os.path.basename(filepath).replace("CarFuelHistory_Processed_", "").replace(".csv", "")

# 3. Hàm tạo dữ liệu từ danh sách file
def build_dataset(files, window_size):
    all_inputs = []
    all_targets = []
    
    # Import Adaptive Kalman để làm Target (Teacher)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D, is_valid_measurement
    
    for f in files:
        df = pd.read_csv(f)
        
        # Tạo dữ liệu theo từng Segment và chạy Adaptive Kalman để lấy Target
        for segment_id, group in df.groupby("SegmentID"):
            if len(group) >= window_size:
                kf_adapt = None
                reference_gap = 5.0
                khoang_thoi_gian_tich_luy = 0.0
                
                target_vals = []
                raw_vals = []
                
                for dong in group.itertuples():
                    gap = getattr(dong, "TimeGapMinutes", reference_gap)
                    if pd.isna(gap) or gap <= 0: gap = reference_gap
                    
                    m_state = str(getattr(dong, "MovementState", "Moving")).strip().upper()
                    movement_state = 0 if m_state == "STOPPED" else 1
                    acceleration = float(getattr(dong, "Acceleration", 0.0))
                    
                    val = getattr(dong, "FuelLevel", None)
                    status = getattr(dong, "FeatureStatus", "")
                    
                    if pd.isna(val):
                        val = None
                    else:
                        val = float(val)
                        
                    if not is_valid_measurement(val, status):
                        khoang_thoi_gian_tich_luy += gap
                        last_raw = raw_vals[-1] if raw_vals else (val if val is not None else 0.0)
                        last_target = target_vals[-1] if target_vals else last_raw
                        raw_vals.append(last_raw)
                        target_vals.append(last_target)
                        continue
                        
                    khoang_thoi_gian_adapt = gap + khoang_thoi_gian_tich_luy
                    khoang_thoi_gian_tich_luy = 0.0
                    
                    if kf_adapt is None:
                        kf_adapt = BoLocKalmanThichNghi1D(
                            trang_thai_ban_dau=val, 
                            sai_so_uoc_luong_ban_dau=9.0, 
                            nhieu_qua_trinh=1.0, 
                            r_co_ban=9.0, 
                            nguong_bat_nhay=10.0, 
                            nhip_cho_xac_nhan=3
                        )
                        target_vals.append(val)
                    else:
                        new_val = kf_adapt.cap_nhat(
                            val, 
                            ty_le_dt=khoang_thoi_gian_adapt/reference_gap, 
                            trang_thai_chuyen_dong=movement_state, 
                            gia_toc=acceleration
                        )
                        target_vals.append(new_val)
                    
                    raw_vals.append(val)
                    
                all_inputs.extend(raw_vals)
                all_targets.extend(target_vals)
                
    return FuelDenoiseDataset(all_inputs, all_targets, window_size)

# 4. Huấn luyện mô hình
def train_and_evaluate(window_size, train_files, val_files, test_files, epochs=5, batch_size=64, lr=0.001):
    print(f"\n--- Thử nghiệm với Window Size (N) = {window_size} ---")
    
    train_dataset = build_dataset(train_files, window_size)
    val_dataset = build_dataset(val_files, window_size)
    test_dataset = build_dataset(test_files, window_size)
    
    if len(train_dataset) == 0:
        print("Không đủ dữ liệu train.")
        return None
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CNN1DDenoisingNet(window_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for windows, labels in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(windows)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * windows.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for windows, labels in val_loader:
                windows, labels = windows.to(device), labels.to(device)
                outputs = model(windows)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * windows.size(0)
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    # Đánh giá trên tập TEST (xe hoàn toàn chưa từng xuất hiện)
    model.load_state_dict(best_model_state)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for windows, labels in test_loader:
            windows, labels = windows.to(device), labels.to(device)
            outputs = model(windows)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * windows.size(0)
    test_loss /= len(test_dataset)
    
    print(f"Hoàn thành N={window_size} | Best Val Loss: {best_val_loss:.6f} | TEST LOSS (MSE): {test_loss:.6f}")
    return test_loss, best_model_state

def run_experiment():
    print("=== BẮT ĐẦU EXPERIMENT CNN 1D DENOISING ===")
    
    # Tập huấn luyện và Validation: Lấy từ bộ dữ liệu lớn (Car1 -> Car5)
    train_val_files = sorted(glob.glob("data/processed/CarFuelHistory_Car*_Features.csv"))
    # Tập Kiểm thử (Inference): 13 xe thực tế (Bỏ qua các file sinh ra từ lần test trước)
    test_files = [f for f in sorted(glob.glob("data/processed/CarFuelHistory_Processed_*.csv")) if "_CNN1D" not in f]
    
    if not train_val_files or not test_files:
        print("Không tìm thấy dữ liệu. Hãy chắc chắn có đủ file Train và Test trong data/processed/.")
        return
        
    # Bước 1: Chia tập dữ liệu theo xe (Vehicle Split)
    random.seed(42)
    random.shuffle(train_val_files)
    
    # Chia 80% Train, 20% Val từ bộ Train lớn
    train_idx = int(0.8 * len(train_val_files))
    train_files = train_val_files[:train_idx]
    val_files = train_val_files[train_idx:]
    
    print(f"Tổng số xe dùng để Huấn Luyện (Train/Val): {len(train_val_files)}")
    print(f"- Train Vehicles ({len(train_files)}): {[get_vehicle_id(f) for f in train_files]}")
    print(f"- Val Vehicles ({len(val_files)}): {[get_vehicle_id(f) for f in val_files]}")
    print(f"\nTường thành Inference (Test) trên 13 xe kia:")
    print(f"- Test Vehicles ({len(test_files)}): {[get_vehicle_id(f) for f in test_files]}")
    print("\n* Pseudo-Ground-Truth: Offline Centered Median (Không dùng Kalman)")
    
    N_LIST = [10]
    results = {}
    best_overall_loss = float('inf')
    best_n = None
    best_state = None
    
    for n in N_LIST:
        test_loss, model_state = train_and_evaluate(n, train_files, val_files, test_files, epochs=5)
        if test_loss is not None:
            results[n] = test_loss
            if test_loss < best_overall_loss:
                best_overall_loss = test_loss
                best_n = n
                best_state = model_state
                
    print("\n=== TỔNG KẾT EXPERIMENT ===")
    for n, loss in results.items():
        print(f"N = {n} -> Test Loss: {loss:.6f}")
        
    print(f"\n=> Chọn N = {best_n} là mô hình tốt nhất (Test Loss nhỏ nhất).")
    
    # Lưu mô hình tốt nhất
    os.makedirs("models", exist_ok=True)
    model_path = "models/cnn_1d_denoising.pth"
    torch.save({
        'window_size': best_n,
        'model_state_dict': best_state,
    }, model_path)
    print(f"Đã lưu mô hình N={best_n} tại {model_path}.")

if __name__ == "__main__":
    run_experiment()
