import pandas as pd
import numpy as np
import glob
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

def profile_vehicles():
    print("Bắt đầu thống kê đặc tính dữ liệu của 22 xe...")
    
    # Tìm tất cả các file đã qua tiền xử lý nhưng không phải file output của mô hình
    all_files = glob.glob('data/processed/CarFuelHistory_Processed_*.csv')
    base_files = [f for f in all_files if '_CNN' not in f and '_Kalman' not in f]
    
    results = []
    
    for file in base_files:
        try:
            ma_xe = os.path.basename(file).replace('CarFuelHistory_Processed_', '').replace('.csv', '')
            df = pd.read_csv(file)
            
            if df.empty:
                continue
                
            # Đảm bảo các cột cần thiết
            df['FuelTime'] = pd.to_datetime(df['FuelTime'], errors="coerce")
            df['FuelLevel'] = pd.to_numeric(df['FuelLevel'], errors="coerce")
            df['Speed'] = pd.to_numeric(df.get('Speed', 0), errors="coerce")
            df['TimeGapMinutes'] = pd.to_numeric(df.get('TimeGapMinutes', 5.0), errors="coerce")
            
            df = df.dropna(subset=['FuelLevel', 'FuelTime'])
            if len(df) < 10:
                continue
                
            # 1. Thông số cơ bản
            total_samples = len(df)
            duration_days = (df['FuelTime'].max() - df['FuelTime'].min()).total_seconds() / (24 * 3600)
            capacity = df['FuelLevel'].quantile(0.99)
            if pd.isna(capacity) or capacity < 10: capacity = 200.0
            
            # 2. Đặc tính nhiễu
            df['RollingStd'] = df['FuelLevel'].rolling(5, min_periods=1).std().fillna(0)
            noise_level = df['RollingStd'].median()
            noise_max = df['RollingStd'].max()
            
            # 3. Đặc tính gián đoạn (Gap)
            gaps_over_10m = (df['TimeGapMinutes'] > 10).sum()
            gap_ratio = gaps_over_10m / total_samples * 100
            
            # 4. Đặc tính di chuyển
            moving_ratio = (df['Speed'] > 0).mean() * 100
            
            # 5. Phân loại mức độ nhiễu (Dựa trên dung tích)
            noise_pct = (noise_level / capacity) * 100
            if noise_pct < 0.5:
                noise_class = "Ít nhiễu (Mượt)"
            elif noise_pct < 1.5:
                noise_class = "Nhiễu vừa"
            else:
                noise_class = "Nhiễu mạnh"
                
            results.append({
                "Mã Xe": ma_xe,
                "Dung tích (L)": round(capacity, 1),
                "Thời gian đo (Ngày)": round(duration_days, 1),
                "Số mẫu": total_samples,
                "Tỷ lệ mất tín hiệu (%)": round(gap_ratio, 2),
                "Tỷ lệ di chuyển (%)": round(moving_ratio, 1),
                "Nhiễu trung bình (L)": round(noise_level, 2),
                "Max dao động (L)": round(noise_max, 2),
                "Phân loại": noise_class
            })
            
        except Exception as e:
            print(f"Lỗi khi xử lý {file}: {e}")
            
    df_results = pd.DataFrame(results)
    
    # Sắp xếp theo mức độ nhiễu
    df_results = df_results.sort_values("Nhiễu trung bình (L)", ascending=False)
    
    # Lưu ra CSV
    os.makedirs('artifacts', exist_ok=True)
    out_path = 'artifacts/vehicle_profiles.csv'
    df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"\nĐã lưu thống kê ra: {out_path}")
    print("\n--- BẢNG TÓM TẮT ĐẶC TÍNH XE ---")
    print(df_results.to_markdown(index=False))
    
if __name__ == "__main__":
    profile_vehicles()
