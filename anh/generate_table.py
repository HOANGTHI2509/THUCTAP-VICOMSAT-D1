import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import glob
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

csv_files = sorted(glob.glob("d:\\THUCTAP_VICOMSAT\\CarFuelHistory_Car*_Flags.csv"))
if not csv_files:
    print("Không tìm thấy file CSV nào!")
    sys.exit(0)

stats = []

for f in csv_files:
    df = pd.read_csv(f)
    vehicle_id = df["VehicleID"].iloc[0] if "VehicleID" in df.columns else "Unknown"
    
    total_records = len(df)
    
    # Flags statistics
    fuel_zero = df["FlagFuelZero"].sum() if "FlagFuelZero" in df.columns else 0
    missing_gps = df["FlagMissingGPS"].sum() if "FlagMissingGPS" in df.columns else 0
    invalid_gps = df["FlagInvalidGPS"].sum() if "FlagInvalidGPS" in df.columns else 0
    zero_coord = df["FlagZeroCoordinate"].sum() if "FlagZeroCoordinate" in df.columns else 0
    long_gap = df["FlagLongGap"].sum() if "FlagLongGap" in df.columns else 0
    speed_gps_conflict = df["FlagSpeedGpsConflict"].sum() if "FlagSpeedGpsConflict" in df.columns else 0
    speed_zero = (df["Speed"] == 0).sum() if "Speed" in df.columns else 0
    speed_zero_rate = (speed_zero / total_records * 100) if total_records > 0 else 0
    
    # Value ranges
    min_fuel = df["FuelLevel"].min()
    max_fuel = df["FuelLevel"].max()
    median_fuel = df["FuelLevel"].median()
    
    stats.append({
        "Xe (VehicleID)": vehicle_id,
        "Tổng bản ghi": f"{total_records:,}",
        "Mức NL (Min - Median - Max)": f"{min_fuel:.1f} - {median_fuel:.1f} - {max_fuel:.1f}",
        "Số điểm NL = 0": f"{fuel_zero:,}",
        "Khoảng ngắt > 30p": f"{long_gap:,}",
        "Tọa độ lỗi (0,0)": f"{zero_coord:,}",
        "Tỷ lệ dừng (%)": f"{speed_zero_rate:.1f}%",
        "Xung đột GPS/Vận tốc": f"{speed_gps_conflict:,}"
    })

# Generate Markdown table manually
headers = list(stats[0].keys())
print("| " + " | ".join(headers) + " |")
print("|" + "|".join(["---" for _ in headers]) + "|")
for row in stats:
    print("| " + " | ".join(str(row[h]) for h in headers) + " |")
