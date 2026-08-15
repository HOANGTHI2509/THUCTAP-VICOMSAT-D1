import glob
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

class BoLocKalmanThichNghi1D:
    def __init__(
        self,
        trang_thai_ban_dau: float,
        sai_so_uoc_luong_ban_dau: float = 4.0,
        nhieu_qua_trinh: float = 0.1, # Siêu mượt
        r_co_ban: float = 64.0,       # Tăng R base lên 64 để đè phẳng mọi nhấp nhô
        r_nhieu_dot_bien: float = 128.0,
        nguong_bat_nhay_co_ban: float = 15.0, # Nới ngưỡng lên 15L để không bị nảy theo quả núi lớn
        nguong_toi_da: float = 25.0,
        nhip_cho_xac_nhan: int = 3,           # Đợi 3 nhịp (15p) mới cho phép giật lên/xuống
        muc_tieu_thu_100km: float = 20.0,     # Mức tiêu thụ dự kiến (L/100km)
    ) -> None:
        self.x = float(trang_thai_ban_dau)
        self.P = float(sai_so_uoc_luong_ban_dau)
        self.Q = float(nhieu_qua_trinh)
        self.muc_tieu_thu_100km = float(muc_tieu_thu_100km)

        self.r_base = float(r_co_ban)
        self.r_spike = float(r_nhieu_dot_bien)
        self.threshold_base = float(nguong_bat_nhay_co_ban)
        self.threshold_max = float(nguong_toi_da)
        self.persistence_required = int(nhip_cho_xac_nhan)

        self.so_nhip_nhieu_lon = 0
        self.dau_nhieu_truoc_do = 0
        self.so_nhip_cung_chieu = 0
        
    def cap_nhat(
        self, 
        gia_tri_do: float, 
        ty_le_dt: float = 1.0, 
        trang_thai_chuyen_dong: int = 1, 
        gia_toc: float = 0.0,
        rolling_std: float = 0.0,
        van_toc: float = 0.0  # Vận tốc (km/h)
    ) -> tuple:
        z = float(gia_tri_do)

        # 1. Dự đoán (Standard Kalman)
        x_du_doan = self.x
        
        phan_du = z - x_du_doan
        phan_du_tuyet_doi = abs(phan_du)
        dau_hien_tai = int(np.sign(phan_du))

        # Dynamic Threshold
        nguong_hien_tai = min(self.threshold_max, max(self.threshold_base, 2.0 * rolling_std))

        if phan_du_tuyet_doi > nguong_hien_tai:
            if dau_hien_tai == self.dau_nhieu_truoc_do:
                self.so_nhip_nhieu_lon += 1
            else:
                self.so_nhip_nhieu_lon = 1
        else:
            self.so_nhip_nhieu_lon = 0

        self.dau_nhieu_truoc_do = dau_hien_tai

        # 2. Adaptive Q & R
        R_thich_nghi = self.r_base
        Q_thich_nghi = self.Q * max(ty_le_dt, 0.1)
        
        # Nếu nhiễu kéo dài quá mức cho phép -> Đổ xăng thật hoặc sụt giảm thật
        if self.so_nhip_nhieu_lon >= self.persistence_required:
            R_thich_nghi = 1.0
            Q_thich_nghi = 100.0
        else:
            # Chế độ bình thường: Hành xử giống hệt Kalman truyền thống để bám xu hướng mượt mà
            # Tự động tăng mức độ làm phẳng (R) khi xe đi nhanh để khử độ rung lắc của sóng xăng
            if van_toc > 0:
                R_thich_nghi *= max(1.0, van_toc / 30.0)
            
            if abs(gia_toc) > 0.1:
                R_thich_nghi *= 3.0

        # 3. Cập nhật (Update)
        P_du_doan = self.P + Q_thich_nghi
        K = P_du_doan / (P_du_doan + R_thich_nghi)

        self.x = x_du_doan + K * phan_du
        self.x = max(0.0, self.x)
        self.P = (1 - K) * P_du_doan

        return (self.x, self.P, x_du_doan, P_du_doan)
        
def rts_smooth_1d(x_forward, P_forward, x_predict, P_predict):
    n = len(x_forward)
    x_smooth = np.zeros(n)
    if n == 0: return x_smooth
    
    x_smooth[-1] = x_forward[-1]
    
    for k in range(n - 2, -1, -1):
        if P_predict[k+1] > 0:
            C = P_forward[k] / P_predict[k+1]
        else:
            C = 0.0
        x_smooth[k] = x_forward[k] + C * (x_smooth[k+1] - x_predict[k+1])
        
    return x_smooth

def is_valid_measurement(fuel, feature_status) -> bool:
    fuel_val = pd.to_numeric(fuel, errors="coerce")
    if pd.isna(fuel_val) or fuel_val <= 0:
        return False
        
    status = str(feature_status).strip().upper()
    invalid_statuses = {"FUEL_ZERO", "PREVIOUS_INVALID", "INVALID"}
    return status not in invalid_statuses

def chay_kalman_thich_nghi_cho_tat_ca_xe(
    mau_ten_file: str = "CarFuelHistory_Processed_*.csv",
) -> None:
    print("=== BẮT ĐẦU CHẠY ADAPTIVE KALMAN FILTER (INNOVATION GATING) ===")
    danh_sach_file = sorted(glob.glob(mau_ten_file))
    if not danh_sach_file: return

    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang chạy Adaptive Kalman cho xe {ma_xe}...")

        df = pd.read_csv(duong_dan_file)
        
        df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
        df["FuelLevel"] = pd.to_numeric(df["FuelLevel"], errors="coerce")
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        df["Kalman_Adaptive"] = np.nan
        
        # Luôn tính lại gia tốc (m/s^2) chuẩn xác
        if "Speed" in df.columns:
            time_gap_sec = df["TimeGapMinutes"].fillna(5.0) * 60.0
            safe_time_gap = time_gap_sec.replace(0, 1.0)
            df["Acceleration"] = (df.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / 3.6) / safe_time_gap
        else:
            df["Acceleration"] = 0.0
        
        thoi_gian_chuan_phut = 5.0 # Mặc định 5 phút mỗi mẫu

        for ma_doan, nhom in df.groupby("SegmentID", sort=False, dropna=False):
            kf = None
            khoang_thoi_gian_tich_luy = 0.0
            for dong in nhom.itertuples():
                vi_tri = dong.Index
                
                # Tích lũy thời gian
                gap = getattr(dong, "TimeGapMinutes", thoi_gian_chuan_phut)
                if pd.isna(gap) or gap <= 0: gap = thoi_gian_chuan_phut
                
                if not is_valid_measurement(getattr(dong, "FuelLevel", None), getattr(dong, "FeatureStatus", "")):
                    khoang_thoi_gian_tich_luy += gap
                    df.loc[vi_tri, "Kalman_Adaptive"] = np.nan
                    continue

                gia_tri_do = float(dong.FuelLevel)
                
                khoang_thoi_gian = gap + khoang_thoi_gian_tich_luy
                khoang_thoi_gian_tich_luy = 0.0 # reset sau khi dùng
                ty_le_dt = khoang_thoi_gian / thoi_gian_chuan_phut

                # Đọc MovementState
                trang_thai_chuyen_dong_thay_the = str(getattr(dong, "MovementState", "Moving")).strip().upper()
                trang_thai_chuyen_dong_int = 0 if trang_thai_chuyen_dong_thay_the == "STOPPED" else 1
                
                # Đọc Gia tốc (Acceleration) đã tính sẵn (m/s^2)
                gia_toc_hien_tai = float(getattr(dong, "Acceleration", 0.0))
                if pd.isna(gia_toc_hien_tai): gia_toc_hien_tai = 0.0

                # Đọc RollingStd
                rolling_std_hien_tai = float(getattr(dong, "RollingStd", 0.0))
                if pd.isna(rolling_std_hien_tai): rolling_std_hien_tai = 0.0

                if kf is None:
                    kf = BoLocKalmanThichNghi1D(
                        trang_thai_ban_dau=gia_tri_do,
                        sai_so_uoc_luong_ban_dau=4.0,
                        nhieu_qua_trinh=0.2,
                        r_co_ban=16.0,
                        r_nhieu_dot_bien=64.0,
                        nguong_bat_nhay_co_ban=10.0,
                        nguong_toi_da=25.0,
                        nhip_cho_xac_nhan=3
                    )
                    gia_tri_da_loc = gia_tri_do
                else:
                    gia_tri_da_loc = kf.cap_nhat(
                        gia_tri_do, 
                        ty_le_dt=ty_le_dt, 
                        trang_thai_chuyen_dong=trang_thai_chuyen_dong_int,
                        gia_toc=gia_toc_hien_tai,
                        rolling_std=rolling_std_hien_tai
                    )

                df.loc[vi_tri, "Kalman_Adaptive"] = gia_tri_da_loc

        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
        duong_dan_xuat = duong_dan_file.replace(".csv", "_Kalman_Adaptive.csv")
        df.to_csv(duong_dan_xuat, index=False, encoding="utf-8-sig")
        print(f"Hoàn tất {ma_xe}! Đã xuất: {duong_dan_xuat}")

    print("=== ĐÃ CHẠY XONG ADAPTIVE KALMAN (GATING) ===")

if __name__ == "__main__":
    chay_kalman_thich_nghi_cho_tat_ca_xe()
