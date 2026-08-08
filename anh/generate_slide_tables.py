import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import glob
import sys

sys.stdout.reconfigure(encoding='utf-8')
csv_files = sorted(glob.glob("d:\\THUCTAP_VICOMSAT\\CarFuelHistory_*_Flags.csv"))

all_stats_1 = []
all_stats_2 = []
all_stats_3 = []
all_stats_5 = []
raw_fuel_stats = {}

print("Đang xử lý dữ liệu...")
for f in csv_files:
    df = pd.read_csv(f)
    df["FuelTime"] = pd.to_datetime(df["FuelTime"])
    car = df["VehicleID"].iloc[0] if "VehicleID" in df.columns else "Unknown"
    
    if car not in ["Car 1", "Car 2", "Car 3", "Car 4", "Car 5"]:
        # Try to infer car name from filename if not in dataframe
        if "Car1" in f: car = "Car 1"
        elif "Car2" in f: car = "Car 2"
        elif "Car3" in f: car = "Car 3"
        elif "Car4" in f: car = "Car 4"
        elif "Car5" in f: car = "Car 5"
        else: continue
    
    # Table 1
    total_records = len(df)
    time_range = f"{df['FuelTime'].min().strftime('%Y-%m')} đến {df['FuelTime'].max().strftime('%Y-%m')}"
    days = (df['FuelTime'].max() - df['FuelTime'].min()).days
    fuel_min = df["FuelLevel"].min()
    fuel_med = df["FuelLevel"].median()
    fuel_max = df["FuelLevel"].max()
    speed_0_rate = (df["Speed"] == 0).mean() * 100 if "Speed" in df.columns else 0
    
    all_stats_1.append({
        "Xe": car, "Số bản ghi": f"{total_records:,}", "Thời gian dữ liệu": time_range, "Số ngày": days,
        "Fuel Min": f"{fuel_min:.1f}", "Fuel Median": f"{fuel_med:.1f}", "Fuel Max": f"{fuel_max:.1f}", 
        "Tỷ lệ Speed = 0": f"{speed_0_rate:.1f}%"
    })
    
    # Table 2
    missing_data = df["FlagMissingTime"].sum() + df["FlagMissingFuel"].sum() + df["FlagMissingGPS"].sum() + df["FlagMissingSpeed"].sum() if "FlagMissingTime" in df.columns else 0
    dup_time = df["FlagDuplicateTime"].sum() if "FlagDuplicateTime" in df.columns else 0
    fuel_0 = df["FlagFuelZero"].sum() if "FlagFuelZero" in df.columns else 0
    invalid_gps = df["FlagInvalidGPS"].sum() + df["FlagZeroCoordinate"].sum() if "FlagInvalidGPS" in df.columns else 0
    abnormal_speed = df["FlagNegativeSpeed"].sum() if "FlagNegativeSpeed" in df.columns else 0
    long_gap = df["FlagLongGap"].sum() if "FlagLongGap" in df.columns else 0
    
    def pct(val): return f"{val:,} ({(val/total_records)*100:.2f}%)" if val > 0 else "0"
    
    all_stats_2.append({
        "Xe": car, "Thiếu dữ liệu": pct(missing_data), "Timestamp trùng": pct(dup_time),
        "Fuel = 0": pct(fuel_0), "GPS không hợp lệ": pct(invalid_gps),
        "Speed bất thường": pct(abnormal_speed), "Khoảng mất dữ liệu": pct(long_gap)
    })
    
    # Table 3
    if "TimeGapMinutes" in df.columns:
        mode_gap = df["TimeGapMinutes"].mode()
        mode_val = mode_gap.iloc[0] if not mode_gap.empty else 0
        med_gap = df["TimeGapMinutes"].median()
        max_gap = df["TimeGapMinutes"].max()
        on_beat = (df["TimeGapMinutes"] == mode_val).mean() * 100 if not mode_gap.empty else 0
    else:
        mode_val, med_gap, max_gap, on_beat = 0, 0, 0, 0
        
    all_stats_3.append({
        "Xe": car, "TimeGap phổ biến": f"{mode_val:.1f} ph", "TimeGap trung vị": f"{med_gap:.1f} ph",
        "Tỷ lệ đúng nhịp": f"{on_beat:.1f}%", "Khoảng lớn nhất": f"{max_gap:.0f} ph", "Số gap > 30 phút": f"{long_gap:,}"
    })
    
    # Table 4 logic
    if "AbsDeltaFuel" in df.columns:
        delta = df["AbsDeltaFuel"].dropna()
        raw_fuel_stats[car] = {
            "Median |DeltaFuel|": f"{delta.median():.3f}",
            "P95 |DeltaFuel|": f"{delta.quantile(0.95):.3f}",
            "Max |DeltaFuel|": f"{delta.max():.1f}"
        }
    
    # Table 5 logic
    if "Speed" in df.columns and "AbsDeltaFuel" in df.columns:
        stopped = df[df["Speed"] == 0]
        moving = df[df["Speed"] > 0]
        med_delta_stop = stopped["AbsDeltaFuel"].median()
        med_delta_move = moving["AbsDeltaFuel"].median()
        gps_moved_when_stop_count = ((df["Speed"] == 0) & (df["DistanceMeters"] > 100)).sum() if "DistanceMeters" in df.columns else 0
        conflict = df["FlagSpeedGpsConflict"].sum() if "FlagSpeedGpsConflict" in df.columns else 0
    else:
        med_delta_stop, med_delta_move, gps_moved_when_stop_count, conflict = 0, 0, 0, 0
        
    all_stats_5.append({
        "Xe": car, "Median |ΔFuel| khi dừng": f"{med_delta_stop:.3f}", "Median |ΔFuel| khi chạy": f"{med_delta_move:.3f}",
        "GPS dịch chuyển khi Speed = 0 (lần)": f"{gps_moved_when_stop_count:,}", "Số mâu thuẫn Speed-GPS": f"{conflict:,}"
    })

def dict_to_md(df_list):
    if not df_list: return ""
    df = pd.DataFrame(df_list)
    headers = list(df.columns)
    md = "| " + " | ".join(headers) + " |\n"
    md += "|" + "|".join(["---" for _ in headers]) + "|\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
    return md

with open("d:\\THUCTAP_VICOMSAT\\slide_tables.md", "w", encoding="utf-8") as f:
    f.write("## 1. Bảng tổng quan dữ liệu\n")
    f.write(dict_to_md(all_stats_1) + "\n\n")
    
    f.write("## 2. Bảng chất lượng dữ liệu\n")
    f.write(dict_to_md(all_stats_2) + "\n\n")
    
    f.write("## 3. Phân tích tần suất truyền\n")
    f.write(dict_to_md(all_stats_3) + "\n\n")
    
    f.write("## 4. Đặc điểm tín hiệu nhiên liệu thô\n")
    df4 = pd.DataFrame(raw_fuel_stats)
    df4.index.name = "Chỉ số"
    df4 = df4.reset_index()
    f.write(dict_to_md(df4.to_dict('records')) + "\n\n")
    
    f.write("## 5. Mối liên hệ Fuel - Speed - GPS\n")
    f.write(dict_to_md(all_stats_5) + "\n\n")

print("Hoàn tất xuất bảng ra slide_tables.md")
