import glob
import os
import sys
from collections import deque
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

def tinh_loc_trung_vi(du_lieu: list[float], kich_thuoc_cua_so: int) -> list[float]:
    """
    Thuật toán Median Filter (Lọc Trung vị) dạng causal window.
    Khi gặp giá trị không hợp lệ, cửa sổ được reset để tránh nối dữ liệu.
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

        gia_tri = float(gia_tri)
        cua_so.append(gia_tri)

        cua_so_da_sap_xep = sorted(cua_so)
        giua = len(cua_so_da_sap_xep) // 2
        
        if len(cua_so_da_sap_xep) % 2 == 0:
            ket_qua.append((cua_so_da_sap_xep[giua-1] + cua_so_da_sap_xep[giua]) / 2.0)
        else:
            ket_qua.append(cua_so_da_sap_xep[giua])

    return ket_qua

def chay_loc_trung_vi_cho_tat_ca_xe(mau_ten_file: str = "CarFuelHistory_Processed_*.csv", kich_thuoc_cua_so: int = 10) -> None:
    print("=== BẮT ĐẦU CHẠY MEDIAN FILTER ===")
    danh_sach_file = sorted(glob.glob(mau_ten_file))
    
    if not danh_sach_file: return
        
    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang chạy Median Filter (N={kich_thuoc_cua_so}) cho xe {ma_xe}...")
        
        df = pd.read_csv(duong_dan_file, parse_dates=["FuelTime"])
        
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        df["Median_Filter"] = np.nan
        
        for ma_doan, nhom in df.groupby("SegmentID", sort=False, dropna=False):
            gia_tri_da_loc = tinh_loc_trung_vi(nhom["FuelLevel"].tolist(), kich_thuoc_cua_so)
            df.loc[nhom.index, "Median_Filter"] = gia_tri_da_loc
            
        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
        
        duong_dan_xuat = duong_dan_file.replace(".csv", f"_Median_N{kich_thuoc_cua_so}.csv")
        df.to_csv(duong_dan_xuat, index=False, encoding="utf-8-sig")
        print(f"Hoàn tất {ma_xe}! Đã xuất: {duong_dan_xuat}")

    print("=== ĐÃ CHẠY XONG CHO TẤT CẢ CÁC XE ===")

if __name__ == "__main__":
    chay_loc_trung_vi_cho_tat_ca_xe(kich_thuoc_cua_so=10)
