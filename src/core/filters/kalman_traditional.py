import glob
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

class BoLocKalmanTieuChuan1D:
    """
    Standard Kalman Filter 1D với mô hình random walk:
        x_t = x_(t-1) + w_t
        z_t = x_t + v_t
    Trong đó:
        Q: phương sai nhiễu quá trình, đơn vị lít²
        R: phương sai nhiễu phép đo, đơn vị lít²
    """
    def __init__(
        self,
        trang_thai_ban_dau: float,
        sai_so_uoc_luong_ban_dau: float = 1.0,
        nhieu_qua_trinh: float = 1.0,
        nhieu_do_luong: float = 9.0,
    ) -> None:
        if sai_so_uoc_luong_ban_dau <= 0:
            raise ValueError("sai_so_uoc_luong_ban_dau phải lớn hơn 0.")
        if nhieu_qua_trinh < 0:
            raise ValueError("nhieu_qua_trinh không được âm.")
        if nhieu_do_luong <= 0:
            raise ValueError("nhieu_do_luong phải lớn hơn 0.")

        self.x = float(trang_thai_ban_dau)
        self.P = float(sai_so_uoc_luong_ban_dau)
        self.Q = float(nhieu_qua_trinh)
        self.R = float(nhieu_do_luong)

    def cap_nhat(self, gia_tri_do: float, ty_le_dt: float = 1.0) -> float:
        """Cập nhật bộ lọc bằng một phép đo FuelLevel hợp lệ."""
        z = float(gia_tri_do)

        # Predict
        x_du_doan = self.x
        # Mở rộng P theo thời gian thực tế trôi qua
        P_du_doan = self.P + self.Q * max(ty_le_dt, 0.1)

        # Kalman Gain
        K = P_du_doan / (P_du_doan + self.R)

        # Update state
        phan_du = z - x_du_doan
        self.x = x_du_doan + K * phan_du
        
        # Ràng buộc vật lý: Xăng không thể âm
        self.x = max(0.0, self.x)

        # Joseph-form (Tăng độ ổn định số học, tránh P âm)
        self.P = (1.0 - K)**2 * P_du_doan + K**2 * self.R

        return self.x

def is_valid_measurement(fuel, feature_status) -> bool:
    fuel_val = pd.to_numeric(fuel, errors="coerce")
    if pd.isna(fuel_val) or fuel_val <= 0:
        return False
        
    status = str(feature_status).strip().upper()
    invalid_statuses = {"FUEL_ZERO", "PREVIOUS_INVALID", "INVALID"}
    return status not in invalid_statuses

def chay_kalman_tieu_chuan_cho_tat_ca_xe(
    mau_ten_file: str = "CarFuelHistory_Processed_*.csv",
    nhieu_qua_trinh: float = 1.0,
    nhieu_do_luong: float = 9.0,
) -> None:
    print("=== BẮT ĐẦU CHẠY STANDARD KALMAN FILTER (CHUYÊN GIA) ===")
    danh_sach_file = sorted(glob.glob(mau_ten_file))
    if not danh_sach_file: return

    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang chạy Standard Kalman cho xe {ma_xe}...")

        df = pd.read_csv(duong_dan_file)
        
        df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
        df["FuelLevel"] = pd.to_numeric(df["FuelLevel"], errors="coerce")
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        df["Kalman_Standard"] = np.nan

        thoi_gian_chuan_phut = 5.0 # Mặc định 5 phút mỗi mẫu

        for ma_doan, nhom in df.groupby("SegmentID", sort=False, dropna=False):
            kf = None
            khoang_thoi_gian_tich_luy = 0.0
            for dong in nhom.itertuples():
                vi_tri = dong.Index
                
                # Tích lũy thời gian
                gap = getattr(dong, "TimeGapMinutes", thoi_gian_chuan_phut)
                if pd.isna(gap) or gap <= 0: gap = thoi_gian_chuan_phut
                
                # Sửa lỗi mất trí nhớ: chỉ bỏ qua cập nhật, không set kf = None
                if not is_valid_measurement(getattr(dong, "FuelLevel", None), getattr(dong, "FeatureStatus", "")):
                    khoang_thoi_gian_tich_luy += gap
                    df.loc[vi_tri, "Kalman_Standard"] = np.nan
                    continue

                gia_tri_do = float(dong.FuelLevel)
                
                khoang_thoi_gian = gap + khoang_thoi_gian_tich_luy
                khoang_thoi_gian_tich_luy = 0.0 # reset sau khi dùng
                ty_le_dt = khoang_thoi_gian / thoi_gian_chuan_phut

                if kf is None:
                    kf = BoLocKalmanTieuChuan1D(
                        trang_thai_ban_dau=gia_tri_do,
                        sai_so_uoc_luong_ban_dau=nhieu_do_luong,
                        nhieu_qua_trinh=nhieu_qua_trinh,
                        nhieu_do_luong=nhieu_do_luong,
                    )
                    gia_tri_da_loc = gia_tri_do
                else:
                    gia_tri_da_loc = kf.cap_nhat(gia_tri_do, ty_le_dt=ty_le_dt)

                df.loc[vi_tri, "Kalman_Standard"] = gia_tri_da_loc

        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
        duong_dan_xuat = duong_dan_file.replace(".csv", "_Kalman_Standard.csv")
        df.to_csv(duong_dan_xuat, index=False, encoding="utf-8-sig")
        print(f"Hoàn tất {ma_xe}! Đã xuất: {duong_dan_xuat}")

    print("=== ĐÃ CHẠY XONG STANDARD KALMAN ===")

if __name__ == "__main__":
    chay_kalman_tieu_chuan_cho_tat_ca_xe()
