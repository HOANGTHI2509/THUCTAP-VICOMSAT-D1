import os
import glob
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def tao_nhan_gia(df):
    """
    Sử dụng dữ liệu tương lai (t+1, t+2) để tạo nhãn chính xác cho thời điểm t.
    Nhãn: 
      0: Ổn định (Stable)
      1: Nhiễu gai (Spike) - bị lệch nhưng ngay lập tức phục hồi
      2: Thay đổi thật (True Drop/Fill) - bị lệch và duy trì
    """
    baseline = df['RollingMedian'].fillna(df['FuelLevel'])
    dev = df['FuelLevel'] - baseline
    
    threshold = np.where(df['Speed'] <= 5, 3.0, 10.0)
    is_large_dev = dev.abs() > threshold
    
    future_2 = df['FuelLevel'].shift(-2)
    dev_future = future_2 - baseline
    
    is_recovered = dev_future.abs() <= threshold
    
    labels = np.zeros(len(df))
    labels[is_large_dev & is_recovered] = 1 # Spike
    labels[is_large_dev & ~is_recovered] = 2 # True Change
    
    return labels

def train_model():
    print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH KALMAN AI (RANDOM FOREST) ===")
    file_list = glob.glob("data/processed/CarFuelHistory_Processed_*.csv")
    
    if not file_list:
        print("Không tìm thấy dữ liệu processed.")
        return
        
    all_data = []
    for f in file_list:
        df = pd.read_csv(f)
        df['Label'] = tao_nhan_gia(df)
        
        if 'Acceleration' not in df.columns:
            time_gap_sec = df["TimeGapMinutes"].fillna(5.0) * 60.0
            safe_time_gap = time_gap_sec.replace(0, 1.0)
            df["Acceleration"] = df.groupby("SegmentID")["Speed"].diff().fillna(0.0) / safe_time_gap
            
        all_data.append(df)
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    features = ['Speed', 'Acceleration', 'DeltaFuel', 'RollingStd', 'TimeGapMinutes']
    ml_df = full_df.dropna(subset=features + ['Label']).copy()
    
    X = ml_df[features]
    y = ml_df['Label']
    
    print(f"Tổng số mẫu huấn luyện: {len(X)}")
    print(f"Phân phối nhãn: \n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nĐang huấn luyện Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    
    print("\nĐánh giá trên tập Test:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Stable (0)', 'Spike (1)', 'True Change (2)']))
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/fuel_state_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
        
    print(f"\n[THÀNH CÔNG] Đã lưu mô hình ML tại: {model_path}")

if __name__ == "__main__":
    train_model()
