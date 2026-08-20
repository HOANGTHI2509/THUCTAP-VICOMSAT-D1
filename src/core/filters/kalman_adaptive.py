import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

class BoLocKalmanThichNghi1D:
    def __init__(
        self,
        trang_thai_ban_dau: float,
        capacity: float = 200.0,
        sai_so_uoc_luong_ban_dau: float = 4.0,
        nhieu_qua_trinh: float = 0.1, 
        r_co_ban: float = 9.0,
        nhip_cho_xac_nhan: int = 3,           
    ) -> None:
        self.x = float(trang_thai_ban_dau)
        self.P = float(sai_so_uoc_luong_ban_dau)
        self.Q = float(nhieu_qua_trinh)
        self.capacity = float(capacity)
        self.r_base = float(r_co_ban)
        self.threshold = max(5.0, 0.02 * self.capacity)
        self.persistence_required = int(nhip_cho_xac_nhan)
        self.so_nhip_nhieu_lon = 0
        self.dau_nhieu_truoc_do = 0
        
    def cap_nhat(
        self, 
        gia_tri_do: float, 
        ty_le_dt: float = 1.0, 
        trang_thai_chuyen_dong: int = 1, 
        gia_toc: float = 0.0,
        rolling_std: float = 0.0,
        van_toc: float = 0.0,
        xac_nhan_nap_nhanh: bool = False,
    ) -> float:
        z = float(gia_tri_do)

        phan_du = z - self.x
        phan_du_tuyet_doi = abs(phan_du)
        dau_hien_tai = int(np.sign(phan_du))

        nguong_hien_tai = max(self.threshold, 2.0 * rolling_std)

        if xac_nhan_nap_nhanh and phan_du > max(0.8, nguong_hien_tai * 0.20):
            if phan_du > nguong_hien_tai:
                self.x = z
            else:
                self.x = self.x + 0.88 * phan_du
            self.P = 4.0
            self.so_nhip_nhieu_lon = 0
            self.dau_nhieu_truoc_do = 0
            return self.x

        noise_context = (
            van_toc > 10.0
            or rolling_std > max(3.0, self.threshold * 0.75)
            or abs(gia_toc) > 0.12
        )

        if phan_du_tuyet_doi > nguong_hien_tai and noise_context:
            if dau_hien_tai == self.dau_nhieu_truoc_do or self.dau_nhieu_truoc_do == 0:
                self.so_nhip_nhieu_lon += 1
            else:
                self.so_nhip_nhieu_lon = 1
        else:
            self.so_nhip_nhieu_lon = 0

        self.dau_nhieu_truoc_do = dau_hien_tai

        trend_drop_candidate = (
            phan_du < -nguong_hien_tai
            and noise_context
            and self.so_nhip_nhieu_lon >= 2
        )

        if self.so_nhip_nhieu_lon >= self.persistence_required:
            self.x = z
            self.P = 4.0
            self.so_nhip_nhieu_lon = 0
            self.dau_nhieu_truoc_do = 0
            return self.x
            
        elif self.so_nhip_nhieu_lon > 0 and not trend_drop_candidate:
            return self.x
            
        else:
            R_thich_nghi = self.r_base
            Q_thich_nghi = self.Q * max(ty_le_dt, 0.1)

            stable_sensor_context = (
                van_toc <= 3.0
                and rolling_std <= max(3.0, self.threshold * 0.75)
                and abs(gia_toc) <= 0.12
            )
            if stable_sensor_context:
                R_thich_nghi = max(3.0, R_thich_nghi * 0.35)
                Q_thich_nghi *= 4.0
            elif trend_drop_candidate:
                R_thich_nghi = max(4.0, R_thich_nghi * 0.55)
                Q_thich_nghi *= 8.0
            
            # Phân dải Vận Tốc (Speed Buckets) thay vì tuyến tính
            if van_toc > 0:
                if van_toc <= 5.0:
                    pass # Đã được ưu tiên giảm R ở stable_sensor_context nếu dao động thấp
                elif van_toc <= 40.0:
                    R_thich_nghi *= 2.0  # Đi phố, stop-and-go -> Sóng sánh nhiều
                elif van_toc <= 70.0:
                    R_thich_nghi *= 1.5  # Tốc độ vừa, đường thoáng -> Ít sóng sánh hơn đi phố
                else:
                    R_thich_nghi *= 2.5  # Tốc độ cao (>70) -> Rung xóc và sức cản gió lớn

            if abs(gia_toc) > 0.1:
                R_thich_nghi *= 3.0

            P_du_doan = self.P + Q_thich_nghi
            K = P_du_doan / (P_du_doan + R_thich_nghi)
            
            phan_du_z = z - self.x
            self.x = self.x + K * phan_du_z
            self.x = max(0.0, self.x)
            self.P = (1 - K) * P_du_doan

            return self.x
        


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
    
    da_ve_bieu_do = False

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
                        capacity=200.0, # Will be handled properly in dashboard
                        sai_so_uoc_luong_ban_dau=4.0,
                        nhieu_qua_trinh=0.2,
                        r_co_ban=9.0,
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

        if not da_ve_bieu_do:
            plt.figure(figsize=(12, 6))
            plt.plot(df['FuelTime'], df['FuelLevel'], label='Xăng Gốc (Thô)', color='red', alpha=0.5)
            plt.plot(df['FuelTime'], df['Kalman_Adaptive'], label='Adaptive Kalman (Khử nhiễu)', color='blue', linewidth=2)
            plt.title(f"So sánh Xăng Gốc và Adaptive Kalman - Xe {ma_xe}")
            plt.xlabel("Thời gian")
            plt.ylabel("Lít")
            plt.legend()
            plt.grid(True)
            plt.show()
            da_ve_bieu_do = True

    print("=== ĐÃ CHẠY XONG ADAPTIVE KALMAN (GATING) ===")

if __name__ == "__main__":
    chay_kalman_thich_nghi_cho_tat_ca_xe()
