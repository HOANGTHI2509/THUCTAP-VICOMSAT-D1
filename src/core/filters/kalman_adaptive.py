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
        sai_so_uoc_luong_ban_dau: float = 9.0,
        nhieu_qua_trinh: float = 1.0,
        r_co_ban: float = 100.0,
        r_nhieu_dot_bien: float = 1000.0,
        nguong_bat_nhay: float = 10.0,
        nhip_cho_xac_nhan: int = 3,
    ) -> None:
        self.x = float(trang_thai_ban_dau)
        self.P = float(sai_so_uoc_luong_ban_dau)
        self.Q = float(nhieu_qua_trinh)

        self.r_base = float(r_co_ban)
        self.r_spike = float(r_nhieu_dot_bien)
        self.threshold = float(nguong_bat_nhay)
        self.persistence_required = int(nhip_cho_xac_nhan)

        self.so_nhip_nhieu_lon = 0
        self.dau_nhieu_truoc_do = 0
        self.trang_thai_xe_truoc_do = 1  # Mặc định là xe đang chạy
        
        # Khóa cảm biến (Sensor Failure Lock)
        self.is_sensor_locked = False
        self.sensor_lock_counter = 0

    def cap_nhat(self, gia_tri_do: float, ty_le_dt: float = 1.0, trang_thai_chuyen_dong: int = 1, gia_toc: float = 0.0) -> float:
        z = float(gia_tri_do)

        # Đặc xá: Nếu chu kỳ trước xe đang dừng, sự thay đổi mức nhiên liệu thường là sự thay đổi vật lý thực tế.
        trang_thai_thuc_te = 0 if self.trang_thai_xe_truoc_do == 0 else trang_thai_chuyen_dong

        # Tự động điều chỉnh Nhịp chờ (Adaptive Persistence)
        if trang_thai_thuc_te == 0:
            # Xe đỗ: Cực kỳ nhạy với mất xăng (trộm). Ép nhịp chờ xuống mức tối thiểu (1 nhịp).
            nguong_cho_phep_hien_tai = 1
        else:
            # Xe chạy: Cực kỳ bảo thủ với biến động (dốc dài, xóc). Đẩy nhịp chờ lên cao để lọc nhiễu 4 điểm.
            # Lấy cài đặt của người dùng (thường là 3) cộng thêm 1 hoặc 2.
            nguong_cho_phep_hien_tai = max(4, self.persistence_required + 1)

        r_dot_bien_hien_tai = self.r_spike
        
        # Khi xe dừng, nhiễu sóng sánh ít hơn, nên hạ ngưỡng phát hiện (threshold) xuống để nhạy với bơm/rút nhỏ
        nguong_bat_nhay_hien_tai = 3.0 if trang_thai_thuc_te == 0 else self.threshold

        # Predict
        x_du_doan = self.x
        P_du_doan = self.P + self.Q * max(ty_le_dt, 0.1)

        phan_du = z - x_du_doan
        phan_du_tuyet_doi = abs(phan_du)
        dau_hien_tai = int(np.sign(phan_du))

        # Kiểm tra innovation lớn có duy trì cùng chiều không
        if phan_du_tuyet_doi > nguong_bat_nhay_hien_tai:
            if dau_hien_tai == self.dau_nhieu_truoc_do:
                self.so_nhip_nhieu_lon += 1
            else:
                self.so_nhip_nhieu_lon = 1
        else:
            self.so_nhip_nhieu_lon = 0

        self.dau_nhieu_truoc_do = dau_hien_tai

        # --- KIỂM TRA SENSOR FAILURE LOCK ---
        is_locked_this_tick = False
        if self.is_sensor_locked:
            if phan_du > -15.0:
                # Xăng đã hồi lại (sai số < 15L), mở khóa cảm biến
                self.is_sensor_locked = False
                self.sensor_lock_counter = 0
            else:
                self.sensor_lock_counter += 1
                if self.sensor_lock_counter > 6:
                    # Timeout (hơn 30 phút). Chấp nhận sự thật (Trộm siêu tốc hoặc Vỡ bình)
                    self.is_sensor_locked = False
                    self.sensor_lock_counter = 0
                else:
                    is_locked_this_tick = True

        if not self.is_sensor_locked and not is_locked_this_tick and phan_du < -50.0:
            # Tụt một phát hơn 50 lít -> Chập cảm biến / Mất kết nối
            self.is_sensor_locked = True
            self.sensor_lock_counter = 1
            is_locked_this_tick = True
        # ------------------------------------

        # 0. Lỗi cảm biến (Sensor Failure Lock)
        if is_locked_this_tick:
            R_thich_nghi = r_dot_bien_hien_tai * 10.0  # Đóng băng tuyệt đối đường dự đoán
            Q_thich_nghi = self.Q * 0.1
        # 1. Innovation Gating (Gai nhiễu đơn lẻ)
        elif (
            phan_du_tuyet_doi > nguong_bat_nhay_hien_tai
            and self.so_nhip_nhieu_lon < nguong_cho_phep_hien_tai
        ):
            R_thich_nghi = r_dot_bien_hien_tai
            Q_thich_nghi = self.Q * max(ty_le_dt, 0.1)
        # 2. Persistence Memory (Thay đổi kéo dài)
        elif (
            phan_du_tuyet_doi > nguong_bat_nhay_hien_tai
            and self.so_nhip_nhieu_lon >= nguong_cho_phep_hien_tai
        ):
            R_thich_nghi = 1.0     # Ép R cực nhỏ để bám theo ngay lập tức
            Q_thich_nghi = 100.0   # Bơm Q cực lớn để K tiến về 1, bứt tốc xóa bỏ sai số tụt hậu
        # 3. Tín hiệu ổn định (Innovation nhỏ)
        else:
            if trang_thai_thuc_te == 0:
                R_thich_nghi = self.r_base * 3.0  # Tĩnh: Tăng nhẹ R để chống nhiễu lượng tử
            else:
                if abs(gia_toc) > 0.5:
                    R_thich_nghi = self.r_base * 5.0  # Dằn xóc: Gia tốc lớn, tăng mạnh R để làm trơn nhiễu sóng sánh
                else:
                    R_thich_nghi = self.r_base  # Ổn định: Gia tốc nhỏ, giảm R để bám sát thực tế
            Q_thich_nghi = self.Q * max(ty_le_dt, 0.1)

        # Cập nhật P_pred với Q_thich_nghi mới
        P_du_doan = self.P + Q_thich_nghi

        K = P_du_doan / (P_du_doan + R_thich_nghi)

        self.x = x_du_doan + K * phan_du
        
        # Ràng buộc vật lý: Xăng không thể âm
        self.x = max(0.0, self.x)

        # Joseph-form: P = (I - K)P_pred(I - K)^T + KRK^T
        self.P = (1.0 - K)**2 * P_du_doan + K**2 * R_thich_nghi

        # Lưu lại trạng thái của chu kỳ này
        self.trang_thai_xe_truoc_do = trang_thai_chuyen_dong

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

    for duong_dan_file in danh_sach_file:
        ma_xe = os.path.basename(duong_dan_file).replace("CarFuelHistory_Processed_", "").replace(".csv", "")
        print(f"Đang chạy Adaptive Kalman cho xe {ma_xe}...")

        df = pd.read_csv(duong_dan_file)
        
        df["FuelTime"] = pd.to_datetime(df["FuelTime"], errors="coerce")
        df["FuelLevel"] = pd.to_numeric(df["FuelLevel"], errors="coerce")
        df["_OriginalOrder"] = np.arange(len(df))
        df = df.sort_values(["SegmentID", "FuelTime", "_OriginalOrder"], kind="stable")
        
        df["Kalman_Adaptive"] = np.nan
        
        # Tiền xử lý: Tính Gia tốc (Acceleration) nếu chưa có
        if "Acceleration" not in df.columns:
            if "Speed" in df.columns:
                # Tránh chia cho 0 hoặc NaN
                time_gap_sec = df["TimeGapMinutes"].fillna(5.0) * 60.0
                safe_time_gap = time_gap_sec.replace(0, 1.0)
                # Tính diff theo từng SegmentID để tránh tính gia tốc vượt ranh giới qua đêm
                df["Acceleration"] = df.groupby("SegmentID", dropna=False)["Speed"].diff().fillna(0.0) / safe_time_gap
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
                
                # Đọc Gia tốc (Acceleration) đã tính sẵn
                gia_toc_hien_tai = float(getattr(dong, "Acceleration", 0.0))
                if pd.isna(gia_toc_hien_tai): gia_toc_hien_tai = 0.0

                if kf is None:
                    kf = BoLocKalmanThichNghi1D(
                        trang_thai_ban_dau=gia_tri_do,
                        sai_so_uoc_luong_ban_dau=9.0,
                        nhieu_qua_trinh=1.0,
                        r_co_ban=9.0,
                        r_nhieu_dot_bien=49.0,
                        nguong_bat_nhay=10.0,
                        nhip_cho_xac_nhan=3
                    )
                    gia_tri_da_loc = gia_tri_do
                else:
                    gia_tri_da_loc = kf.cap_nhat(
                        gia_tri_do, 
                        ty_le_dt=ty_le_dt, 
                        trang_thai_chuyen_dong=trang_thai_chuyen_dong_int,
                        gia_toc=gia_toc_hien_tai
                    )

                df.loc[vi_tri, "Kalman_Adaptive"] = gia_tri_da_loc

        df = df.sort_values("_OriginalOrder", kind="stable").drop(columns="_OriginalOrder")
        duong_dan_xuat = duong_dan_file.replace(".csv", "_Kalman_Adaptive.csv")
        df.to_csv(duong_dan_xuat, index=False, encoding="utf-8-sig")
        print(f"Hoàn tất {ma_xe}! Đã xuất: {duong_dan_xuat}")

    print("=== ĐÃ CHẠY XONG ADAPTIVE KALMAN (GATING) ===")

if __name__ == "__main__":
    chay_kalman_thich_nghi_cho_tat_ca_xe()
