import os
import sys
import glob
import pandas as pd
import numpy as np

# Thêm đường dẫn gốc để import từ thư mục src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.filters import kalman_traditional as kt
from src.core.filters import kalman_adaptive as ka
from src.core.filters import kalman_ml as kml

def export_kalman_results():
    input_files = glob.glob("data/processed/CarFuelHistory_Processed_*.csv")
    if not input_files:
        print("Không tìm thấy file processed nào!")
        return

    output_dir = "data/exported"
    os.makedirs(output_dir, exist_ok=True)

    print("=== BẮT ĐẦU XUẤT DỮ LIỆU KALMAN & ADAPTIVE KALMAN ===")

    for file_path in input_files:
        car_id = os.path.basename(file_path).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang xử lý {car_id}...")

        df = pd.read_csv(file_path)
        df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
        df["FuelLevel"] = pd.to_numeric(df["FuelLevel"], errors="coerce")
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")

        # Chuẩn bị cột
        df["Kalman_Standard"] = np.nan
        df["Kalman_Adaptive"] = np.nan
        df["Kalman_ML"] = np.nan

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

        thoi_gian_chuan_phut = 5.0
        r_kalman = 9.0
        
        for ma_doan, nhom in df.groupby("SegmentID", sort=False, dropna=False):
            kf_std = None
            kf_adapt = None
            kf_ml = None
            
            khoang_thoi_gian_tich_luy_std = 0.0
            khoang_thoi_gian_tich_luy_adapt = 0.0
            khoang_thoi_gian_tich_luy_ml = 0.0

            for vi_tri, dong in nhom.iterrows():
                gap = getattr(dong, "TimeGapMinutes", thoi_gian_chuan_phut)
                if pd.isna(gap) or gap <= 0: gap = thoi_gian_chuan_phut

                is_valid = kt.is_valid_measurement(getattr(dong, "FuelLevel", None), getattr(dong, "FeatureStatus", ""))
                
                if not is_valid:
                    khoang_thoi_gian_tich_luy_std += gap
                    khoang_thoi_gian_tich_luy_adapt += gap
                    khoang_thoi_gian_tich_luy_ml += gap
                    continue

                measurement = float(dong.FuelLevel)

                # Cập nhật Standard Kalman
                khoang_thoi_gian_std = gap + khoang_thoi_gian_tich_luy_std
                khoang_thoi_gian_tich_luy_std = 0.0
                ty_le_dt_std = khoang_thoi_gian_std / thoi_gian_chuan_phut

                if kf_std is None:
                    kf_std = kt.BoLocKalmanTieuChuan1D(trang_thai_ban_dau=measurement, nhieu_do_luong=r_kalman)
                    df.loc[vi_tri, "Kalman_Standard"] = measurement
                else:
                    df.loc[vi_tri, "Kalman_Standard"] = kf_std.cap_nhat(measurement, ty_le_dt=ty_le_dt_std)

                # Cập nhật Adaptive Kalman
                khoang_thoi_gian_adapt = gap + khoang_thoi_gian_tich_luy_adapt
                khoang_thoi_gian_tich_luy_adapt = 0.0
                ty_le_dt_adapt = khoang_thoi_gian_adapt / thoi_gian_chuan_phut
                
                trang_thai_chuyen_dong_thay_the = str(getattr(dong, "MovementState", "Moving")).strip().upper()
                trang_thai_chuyen_dong_int = 0 if trang_thai_chuyen_dong_thay_the == "STOPPED" else 1
                
                gia_toc_hien_tai = float(getattr(dong, "Acceleration", 0.0))
                if pd.isna(gia_toc_hien_tai): gia_toc_hien_tai = 0.0

                if kf_adapt is None:
                    kf_adapt = ka.BoLocKalmanThichNghi1D(
                        trang_thai_ban_dau=measurement, 
                        r_co_ban=r_kalman
                    )
                    df.loc[vi_tri, "Kalman_Adaptive"] = measurement
                else:
                    df.loc[vi_tri, "Kalman_Adaptive"] = kf_adapt.cap_nhat(
                        measurement, 
                        ty_le_dt=ty_le_dt_adapt, 
                        trang_thai_chuyen_dong=trang_thai_chuyen_dong_int,
                        gia_toc=gia_toc_hien_tai
                    )

                # Cập nhật Kalman ML
                khoang_thoi_gian_ml = gap + khoang_thoi_gian_tich_luy_ml
                khoang_thoi_gian_tich_luy_ml = 0.0
                ty_le_dt_ml = khoang_thoi_gian_ml / thoi_gian_chuan_phut
                
                nhan_ai_hien_tai = int(getattr(dong, "_ML_Label", 0))
                
                if kf_ml is None:
                    kf_ml = kml.BoLocKalmanAI(trang_thai_ban_dau=measurement, r_co_ban=r_kalman)
                    df.loc[vi_tri, "Kalman_ML"] = measurement
                else:
                    df.loc[vi_tri, "Kalman_ML"] = kf_ml.cap_nhat(
                        measurement, 
                        ty_le_dt=ty_le_dt_ml,
                        nhan_ai=nhan_ai_hien_tai
                    )

        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns=["_OriginalOrder", "Acceleration", "_ML_Label"])
        
        # Làm tròn kết quả Kalman filter về 1 chữ số thập phân cho dễ nhìn (vd: 191.1)
        df["Kalman_Standard"] = df["Kalman_Standard"].round(1)
        df["Kalman_Adaptive"] = df["Kalman_Adaptive"].round(1)
        df["Kalman_ML"] = df["Kalman_ML"].round(1)
        
        # Chỉ giữ lại các cột quan trọng theo yêu cầu
        columns_to_export = [
            "VehicleID", "FuelTime", "FuelLevel", "Lat", "Lng", "Address", "Speed", 
            "SegmentID", "MovementState", "FeatureStatus", 
            "Kalman_Standard", "Kalman_Adaptive", "Kalman_ML"
        ]
        
        # Chỉ lấy những cột thực sự tồn tại trong df (đề phòng file cũ chưa chạy lại preprocess)
        columns_to_export = [c for c in columns_to_export if c in df.columns]
        
        df_export = df[columns_to_export]

        out_path = os.path.join(output_dir, f"Kalman_Results_{car_id}.csv")
        try:
            df_export.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  -> Đã lưu: {out_path}")
        except PermissionError:
            out_path = os.path.join(output_dir, f"Kalman_Results_{car_id}_v2.csv")
            df_export.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  -> Đã lưu (bản mới do file cũ bị khóa): {out_path}")

    print("=== HOÀN TẤT XUẤT DỮ LIỆU ===")

if __name__ == "__main__":
    export_kalman_results()
