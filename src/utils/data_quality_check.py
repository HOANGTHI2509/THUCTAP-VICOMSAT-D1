import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import sys

# Redirect stdout to a file for the report
report_path = "quality_report.txt"
sys.stdout = open(report_path, "w", encoding="utf-8")

print("================ BÁO CÁO KIỂM TRA CHẤT LƯỢNG DỮ LIỆU TỪNG XE ================\n")

file_path = "CarFuelHistory.xlsx"
print(f"Đang đọc file {file_path}...")
sheets = pd.read_excel(file_path, sheet_name=None)
print("Danh sách sheet (VehicleID):", list(sheets.keys()))
print("\n" + "="*70 + "\n")

required_columns = {"FuelTime", "FuelLevel", "Lat", "Lng", "Address", "Speed"}

def haversine_distance(lat1, lon1, lat2, lon2):
    earth_radius_m = 6371000
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_m * c

def most_common_gap(series):
    series = series.dropna()
    if series.empty: return np.nan
    return series.mode().iloc[0]

# Chạy riêng cho từng xe
for vehicle_id, data in sheets.items():
    print(f"################################################################")
    print(f"                     PHÂN TÍCH XE: {vehicle_id} ")
    print(f"################################################################\n")
    
    data = data.copy()
    data["VehicleID"] = vehicle_id
    
    print("=== BƯỚC 2: Kiểm tra cấu trúc chung ===")
    print("Số dòng:", len(data))
    print("Các cột hiện có:", data.columns.tolist())
    missing_columns = required_columns - set(data.columns)
    print("Cột bị thiếu:", missing_columns if missing_columns else "Không thiếu cột nào")
    print("\n")
    
    print("=== BƯỚC 3 & 6: Kiểm tra FuelTime ===")
    
    data["FuelTime"] = pd.to_datetime(data["FuelTime"], errors="coerce")
    
    invalid_time = data[data["FuelTime"].isna()]
    print("Số dòng thời gian trống hoặc không đọc được:", len(invalid_time))
    
    duplicate_time = data[data.duplicated(subset=["FuelTime"], keep=False)].sort_values(["FuelTime"])
    print("Số dòng có timestamp trùng:", len(duplicate_time))
    
    time_conflict = data.groupby("FuelTime")["FuelLevel"].nunique().reset_index(name="FuelLevelCount")
    time_conflict = time_conflict[time_conflict["FuelLevelCount"] > 1]
    print("Số thời điểm có nhiều FuelLevel khác nhau (xung đột):", len(time_conflict))
    print("\n")

    print("=== BƯỚC 4 & 5: Chuẩn hóa cột số và sắp xếp ===")
    data["FuelLevel"] = pd.to_numeric(data["FuelLevel"], errors="coerce")
    data["Lat"] = pd.to_numeric(data["Lat"], errors="coerce")
    data["Lng"] = pd.to_numeric(data["Lng"], errors="coerce")
    data["Speed"] = pd.to_numeric(data["Speed"], errors="coerce")
    
    data = data.sort_values(["FuelTime"]).reset_index(drop=True)
    print("Đã sắp xếp dữ liệu theo thời gian.\n")

    print("=== BƯỚC 7 & 8: TimeGap và Tần suất gửi ===")
    data["TimeGapMinutes"] = data["FuelTime"].diff().dt.total_seconds().div(60)
    gaps = data["TimeGapMinutes"].dropna()
    if not gaps.empty:
        print("Khoảng trung vị:", gaps.median(), "phút")
        print("Các TimeGap phổ biến:")
        print(gaps.value_counts().head(5))
        
        for threshold in [10, 30, 60, 1440]:
            count = (data["TimeGapMinutes"] > threshold).sum()
            print(f"Số khoảng mất kết nối > {threshold} phút: {count}")
    print("\n")

    print("=== BƯỚC 9 & 10: Kiểm tra FuelLevel ===")
    print("Số FuelLevel bị thiếu:", data["FuelLevel"].isna().sum())
    print("Số FuelLevel âm:", (data["FuelLevel"] < 0).sum())
    print("Số bản ghi FuelLevel = 0:", (data["FuelLevel"] == 0).sum())
    
    if not data["FuelLevel"].isna().all():
        print("Thống kê FuelLevel:")
        print(data["FuelLevel"].describe())
    
    data["PreviousFuel"] = data["FuelLevel"].shift(1)
    data["NextFuel"] = data["FuelLevel"].shift(-1)
    data["DeltaFuel"] = data["FuelLevel"].diff()
    data["AbsDeltaFuel"] = data["DeltaFuel"].abs()
    
    largest_changes = data.sort_values("AbsDeltaFuel", ascending=False)
    print("\nBiến động Fuel lớn nhất (Top 5):")
    print(largest_changes[["FuelTime", "PreviousFuel", "FuelLevel", "DeltaFuel", "TimeGapMinutes"]].head(5))
    print("\n")

    print("=== BƯỚC 11 & 12: Kiểm tra Lat/Lng và tính quãng đường ===")
    data["PreviousLat"] = data["Lat"].shift(1)
    data["PreviousLng"] = data["Lng"].shift(1)
    data["DistanceMeters"] = haversine_distance(data["PreviousLat"], data["PreviousLng"], data["Lat"], data["Lng"])
    
    invalid_lat = ~data["Lat"].between(-90, 90)
    invalid_lng = ~data["Lng"].between(-180, 180)
    print("Số Lat ngoài phạm vi:", invalid_lat.sum())
    print("Số Lng ngoài phạm vi:", invalid_lng.sum())
    zero_coordinate = data[(data["Lat"] == 0) & (data["Lng"] == 0)]
    print("Số bản ghi tọa độ 0,0:", len(zero_coordinate))
    print("\n")

    print("=== BƯỚC 13 & 14: Kiểm tra Speed ===")
    print("Speed bị thiếu:", data["Speed"].isna().sum())
    print("Số Speed âm:", (data["Speed"] < 0).sum())
    if not data["Speed"].isna().all():
        print("Thống kê Speed:")
        print(data["Speed"].describe())
        speed_zero_ratio = (data["Speed"] == 0).mean() * 100
        print(f"Tỷ lệ Speed = 0 (%): {speed_zero_ratio:.2f}%")
        
        speed_zero_but_moved = data[(data["Speed"] == 0) & (data["DistanceMeters"] > 100)]
        print(f"Số điểm Speed = 0 nhưng di chuyển > 100m: {len(speed_zero_but_moved)}")
        speed_positive_but_no_move = data[(data["Speed"] > 5) & (data["DistanceMeters"] < 5) & (data["TimeGapMinutes"] > 0)]
        print(f"Số điểm Speed > 5 nhưng GPS di chuyển < 5m: {len(speed_positive_but_no_move)}")
    print("\n")

    print("=== BƯỚC 15: Chuẩn hóa Address ===")
    if "Address" in data.columns:
        data["Address"] = data["Address"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        print("Số Address bị thiếu:", data["Address"].isna().sum())
    print("\n")

    print("=== BƯỚC 17: Gắn cờ lỗi và lưu file ===")
    data["FlagMissingTime"] = data["FuelTime"].isna()
    data["FlagMissingFuel"] = data["FuelLevel"].isna()
    data["FlagFuelNegative"] = data["FuelLevel"] < 0
    data["FlagFuelZero"] = data["FuelLevel"] == 0
    data["FlagMissingGPS"] = data["Lat"].isna() | data["Lng"].isna()
    data["FlagInvalidGPS"] = ~data["Lat"].between(-90, 90) | ~data["Lng"].between(-180, 180)
    data["FlagZeroCoordinate"] = (data["Lat"] == 0) & (data["Lng"] == 0)
    data["FlagMissingSpeed"] = data["Speed"].isna()
    data["FlagNegativeSpeed"] = data["Speed"] < 0
    data["FlagDuplicateTime"] = data.duplicated(subset=["FuelTime"], keep=False)
    data["FlagLongGap"] = data["TimeGapMinutes"] > 30
    data["FlagSpeedGpsConflict"] = (data["Speed"] == 0) & (data["DistanceMeters"] > 100)

    out_file = f"{vehicle_id}_Flags.csv"
    # To avoid issues with invalid characters in filenames, we sanitize vehicle_id
    sanitized_id = "".join(c for c in str(vehicle_id) if c.isalnum() or c in ('_', '-')).strip()
    out_file = f"CarFuelHistory_{sanitized_id}_Flags.csv"
    
    data.to_csv(out_file, index=False)
    print(f"-> Đã hoàn thành xe {vehicle_id} và lưu kết quả vào file {out_file}")
    print("\n" + "="*70 + "\n")

sys.stdout.close()
