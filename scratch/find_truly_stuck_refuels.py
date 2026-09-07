import glob
import os
import sys
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

from src.core.filters.ai_state_filter import load_fuel_state_classifier, filter_with_ai_state
from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime as new_filter


def main():
    model, metadata = load_fuel_state_classifier("models/fuel_state_classifier")
    files = sorted(glob.glob("TienXuLy/*_processed.csv"))
    vehicle_files = [f for f in files if "Bao_cao_lich_su" not in f]

    print("Tìm kiếm các ca THỰC SỰ BỊ KẸT THẤP (sau 40 mẫu / ~80 phút vẫn không nhận mức mới):")
    truly_stuck = []

    for fpath in vehicle_files:
        v_id = Path(fpath).name.replace("_processed.csv", "")
        df = pd.read_csv(fpath)
        if len(df) < 100:
            continue
        df_ai = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")
        clean_new = np.array(new_filter(df_ai))
        raw = df_ai["FuelLevel"].to_numpy()
        times = df_ai["FuelTime"].to_numpy()

        i = 10
        while i < len(raw) - 50:
            # Kiểm tra xem có bước tăng thật không:
            # median tại i-5..i so với median tại i+30..i+45
            pre_raw = float(np.median(raw[max(0, i-5):i+1]))
            post_raw = float(np.median(raw[i+30:i+45]))
            jump = post_raw - pre_raw

            if jump >= 30.0:
                # Kiểm tra mức clean sau 40 mẫu
                clean_pre = float(clean_new[i])
                clean_post = float(clean_new[i+40])
                clean_gain = clean_post - clean_pre

                if clean_gain < 0.5 * jump:
                    truly_stuck.append({
                        "vehicle_id": v_id,
                        "time": str(times[i]),
                        "time_post": str(times[i+40]),
                        "pre_raw": round(pre_raw, 1),
                        "post_raw": round(post_raw, 1),
                        "jump": round(jump, 1),
                        "clean_pre": round(clean_pre, 1),
                        "clean_post": round(clean_post, 1),
                        "clean_gain": round(clean_gain, 1),
                    })
                i += 45
            else:
                i += 1

    print(f"\nTổng số ca bị kẹt thấp thực sự: {len(truly_stuck)}")
    for s in truly_stuck:
        print(f"  Xe {s['vehicle_id']} tại {s['time']} -> {s['time_post']}: Raw {s['pre_raw']} -> {s['post_raw']} (+{s['jump']} L), nhưng Clean chỉ tăng {s['clean_gain']} L ({s['clean_pre']} -> {s['clean_post']})")


if __name__ == "__main__":
    main()
