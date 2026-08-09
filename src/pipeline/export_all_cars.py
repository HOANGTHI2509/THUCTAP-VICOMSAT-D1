import pandas as pd
import json
import os
import glob
import sys
import numpy as np

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.filters.kalman_adaptive import BoLocKalmanThichNghi1D, is_valid_measurement
from src.core.filters.kalman_traditional import BoLocKalmanTieuChuan1D
from src.core.filters import kalman_ml as kml
from src.core.filters import cnn_1d_filter as cnn


def export_car_data():
    os.makedirs('project/public', exist_ok=True)
    files = sorted(glob.glob('data/processed/CarFuelHistory_Processed_*.csv'))
    
    if not files:
        print("No CSV files found!")
        return
        
    for file_path in files:
        car_id_str = os.path.basename(file_path).replace('CarFuelHistory_Processed_', '').replace('.csv', '')
        print(f"Processing {car_id_str}...")
        
        df = pd.read_csv(file_path)
        df['FuelTime'] = pd.to_datetime(df['FuelTime'])
        df = df.sort_values('FuelTime')
        
        # Tính số giây kể từ thời điểm bắt đầu
        start_time = df['FuelTime'].iloc[0]
        t_seconds = (df['FuelTime'] - start_time).dt.total_seconds().astype(int)
        
        speed = df['Speed'].fillna(0).astype(int).tolist()
        rawFuel = df['FuelLevel'].fillna(0).round(2).tolist()
        
        # Tính toán gia tốc
        if "Acceleration" not in df.columns:
            if "Speed" in df.columns:
                time_gap_sec = df["TimeGapMinutes"].fillna(5.0) * 60.0
                safe_time_gap = time_gap_sec.replace(0, 1.0)
                df["Acceleration"] = df.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / safe_time_gap
            else:
                df["Acceleration"] = 0.0
                
        # Dự đoán nhãn AI cho toàn bộ dataframe trước (Tối ưu hiệu năng)
        df["_ML_Label"] = kml.predict_batch(df)

        # Chạy thuật toán Kalman Adaptive, Traditional, ML
        adaptiveFuel = []
        traditionalFuel = []
        mlFuel = []
        cnnFuel = []
        kf_adapt = None
        kf_trad = None
        kf_ml = None
        kf_cnn = cnn.BoLocCNN1D()
        reference_gap = 5.0
        khoang_thoi_gian_tich_luy_adapt = 0.0
        khoang_thoi_gian_tich_luy_trad = 0.0
        khoang_thoi_gian_tich_luy_ml = 0.0
        
        for i_loc, dong in enumerate(df.itertuples()):
            gap = getattr(dong, "TimeGapMinutes", reference_gap)
            if pd.isna(gap) or gap <= 0: gap = reference_gap
            
            movement_state = 0 if str(getattr(dong, "MovementState", "Moving")).strip().upper() == "STOPPED" else 1
            acceleration = float(getattr(dong, "Acceleration", 0.0))
            
            if not is_valid_measurement(getattr(dong, 'FuelLevel', None), getattr(dong, 'FeatureStatus', '')):
                khoang_thoi_gian_tich_luy_adapt += gap
                khoang_thoi_gian_tich_luy_trad += gap
                khoang_thoi_gian_tich_luy_ml += gap
                adaptiveFuel.append(adaptiveFuel[-1] if adaptiveFuel else rawFuel[i_loc])
                traditionalFuel.append(traditionalFuel[-1] if traditionalFuel else rawFuel[i_loc])
                mlFuel.append(mlFuel[-1] if mlFuel else rawFuel[i_loc])
                cnnFuel.append(cnnFuel[-1] if cnnFuel else rawFuel[i_loc])
                continue
                
            measurement = float(dong.FuelLevel)
            
            # Adaptive Kalman
            khoang_thoi_gian_adapt = gap + khoang_thoi_gian_tich_luy_adapt
            khoang_thoi_gian_tich_luy_adapt = 0.0
            if kf_adapt is None:
                kf_adapt = BoLocKalmanThichNghi1D(trang_thai_ban_dau=measurement, sai_so_uoc_luong_ban_dau=9.0, nhieu_qua_trinh=1.0, r_co_ban=9.0, nguong_bat_nhay=10.0, nhip_cho_xac_nhan=3)
                adaptiveFuel.append(measurement)
            else:
                adaptiveFuel.append(kf_adapt.cap_nhat(measurement, ty_le_dt=khoang_thoi_gian_adapt/reference_gap, trang_thai_chuyen_dong=movement_state, gia_toc=acceleration))
                
            # Traditional Kalman
            khoang_thoi_gian_trad = gap + khoang_thoi_gian_tich_luy_trad
            khoang_thoi_gian_tich_luy_trad = 0.0
            if kf_trad is None:
                kf_trad = BoLocKalmanTieuChuan1D(trang_thai_ban_dau=measurement, sai_so_uoc_luong_ban_dau=9.0, nhieu_qua_trinh=1.0, nhieu_do_luong=9.0)
                traditionalFuel.append(measurement)
            else:
                traditionalFuel.append(kf_trad.cap_nhat(measurement, ty_le_dt=khoang_thoi_gian_trad/reference_gap))
                
            # ML Kalman
            khoang_thoi_gian_ml = gap + khoang_thoi_gian_tich_luy_ml
            khoang_thoi_gian_tich_luy_ml = 0.0
            nhan_ai_hien_tai = int(getattr(dong, "_ML_Label", 0))
            if kf_ml is None:
                kf_ml = kml.BoLocKalmanAI(trang_thai_ban_dau=measurement, r_co_ban=9.0)
                mlFuel.append(measurement)
            else:
                mlFuel.append(kf_ml.cap_nhat(measurement, ty_le_dt=khoang_thoi_gian_ml/reference_gap, nhan_ai=nhan_ai_hien_tai))
                
            # CNN 1D Filter
            cnnFuel.append(kf_cnn.cap_nhat(measurement))
        
        isSpike = (df['QualityFlag'] == 1).tolist() if 'QualityFlag' in df.columns else [False]*len(df)
        
        points = []
        for i in range(len(df)):
            noise = rawFuel[i] - adaptiveFuel[i]
            points.append({
                "t": int(t_seconds.iloc[i]),
                "speed": int(speed[i]),
                "rawFuel": float(rawFuel[i]),
                "adaptiveKalman": float(adaptiveFuel[i]),
                "traditionalKalman": float(traditionalFuel[i]),
                "mlKalman": float(mlFuel[i]),
                "cnn1DFuel": float(cnnFuel[i]),
                "noise": float(round(noise, 2)),
                "isSpike": bool(isSpike[i])
            })
            
        out_path = f'project/public/data_{car_id_str}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(points, f)
        
        print(f"Saved {len(points)} points to {out_path}")

if __name__ == "__main__":
    export_car_data()
