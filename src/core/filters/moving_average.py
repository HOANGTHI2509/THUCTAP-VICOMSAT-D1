import glob
import os
import sys
from collections import deque
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

def tinh_loc_trung_binh_dong(du_lieu: list[float], kich_thuoc_cua_so: int) -> list[float]:
    """
    Moving Average dạng causal/trailing window (Triển khai luồng thời gian thực).
    Khi gặp giá trị không hợp lệ, cửa sổ được reset để tránh nối dữ liệu.
    """
    if kich_thuoc_cua_so <= 0:
        raise ValueError("kich_thuoc_cua_so phải lớn hơn 0.")

    ket_qua: list[float] = []
    cua_so: deque[float] = deque(maxlen=kich_thuoc_cua_so)
    tong_hien_tai = 0.0

    for gia_tri in du_lieu:
        # Chuẩn hóa sang số; dữ liệu không chuyển đổi được sẽ thành NaN
        gia_tri = pd.to_numeric(gia_tri, errors="coerce")

        # FuelLevel thiếu, bằng 0 hoặc âm được xem là không hợp lệ
        if pd.isna(gia_tri) or gia_tri <= 0:
            ket_qua.append(np.nan)
            # KHÔNG CHO CỬA SỔ ĐI XUYÊN QUA ĐIỂM LỖI (Tránh bù trừ sai khi ra khỏi hầm)
            cua_so.clear()
            tong_hien_tai = 0.0
            continue

        gia_tri = float(gia_tri)

        # Nếu cửa sổ đã đầy, loại phần tử cũ nhất khỏi tổng (O(1))
        if len(cua_so) == cua_so.maxlen:
            tong_hien_tai -= cua_so[0]

        cua_so.append(gia_tri)
        tong_hien_tai += gia_tri

        # Chấp nhận Burn-in: Trả về trung bình của các mẫu đã có kể cả khi chưa đủ N
        trung_binh_hien_tai = tong_hien_tai / len(cua_so)
        ket_qua.append(trung_binh_hien_tai)

    return ket_qua

def tinh_loc_trung_vi_dong(du_lieu: list[float], kich_thuoc_cua_so: int) -> list[float]:
    """
    Median Filter dạng causal/trailing window (Triển khai luồng thời gian thực).
    Khi gặp giá trị không hợp lệ, cửa sổ được reset để tránh nối dữ liệu.
    Rất hiệu quả trong việc loại bỏ nhiễu đột biến (outliers) 1-2 nhịp.
    """
    if kich_thuoc_cua_so <= 0:
        raise ValueError("kich_thuoc_cua_so phải lớn hơn 0.")

    ket_qua: list[float] = []
    cua_so: deque[float] = deque(maxlen=kich_thuoc_cua_so)

    for gia_tri in du_lieu:
        gia_tri = pd.to_numeric(gia_tri, errors="coerce")

        if pd.isna(gia_tri) or gia_tri <= 0:
            ket_qua.append(np.nan)
            cua_so.clear()
            continue

        cua_so.append(float(gia_tri))
        
        # Tính trung vị (Median)
        trung_vi_hien_tai = np.median(cua_so)
        ket_qua.append(trung_vi_hien_tai)

    return ket_qua

def chay_trung_binh_dong_cho_tat_ca_xe(mau_ten_file: str = "CarFuelHistory_Processed_*.csv", kich_thuoc_cua_so: int = 10) -> None:
    print("=== BẮT ĐẦU CHẠY MOVING AVERAGE (CHUYÊN GIA) ===")
    danh_sach_file = sorted(glob.glob(mau_ten_file))
    
    if not danh_sach_file: return
        
    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang chạy Moving Average (N={kich_thuoc_cua_so}) cho xe {ma_xe}...")
        
        df = pd.read_csv(duong_dan_file, parse_dates=["FuelTime"])
        
        # Giữ lại thứ tự ban đầu để khôi phục
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        df["Moving_Average"] = np.nan
        
        # Lọc độc lập trong từng segment
        for ma_doan, nhom in df.groupby("SegmentID", sort=False, dropna=False):
            gia_tri_da_loc = tinh_loc_trung_binh_dong(nhom["FuelLevel"].tolist(), kich_thuoc_cua_so)
            # Gán theo index an toàn
            df.loc[nhom.index, "Moving_Average"] = gia_tri_da_loc
            
        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
        
        duong_dan_xuat = duong_dan_file.replace(".csv", f"_MA_N{kich_thuoc_cua_so}.csv")
        df.to_csv(duong_dan_xuat, index=False, encoding="utf-8-sig")
        print(f"Hoàn tất {ma_xe}! Đã xuất: {duong_dan_xuat}")

    print("=== ĐÃ CHẠY XONG CHO TẤT CẢ CÁC XE ===")

if __name__ == "__main__":
    chay_trung_binh_dong_cho_tat_ca_xe(kich_thuoc_cua_so=10)
