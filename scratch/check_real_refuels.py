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
from scratch.legacy_adaptive_filter import filter_ai_enhanced_adaptive_realtime as legacy_filter


def main():
    model, metadata = load_fuel_state_classifier("models/fuel_state_classifier")
    files = sorted(glob.glob("TienXuLy/*_processed.csv"))
    vehicle_files = [f for f in files if "Bao_cao_lich_su" not in f]

    print("Kiểm tra các ca tăng thật (sustained step >= 25L) trên toàn bộ 28 xe:")
    missed_refuels = []
    confirmed_events = []
    
    for fpath in vehicle_files:
        v_id = Path(fpath).name.replace("_processed.csv", "")
        df = pd.read_csv(fpath)
        if len(df) < 50:
            continue
        df_ai = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")
        clean_new = np.array(new_filter(df_ai))
        raw = df_ai["FuelLevel"].to_numpy()

        i = 10
        while i < len(raw) - 25:
            # Kiểm tra bước nhảy tăng thật: raw tăng >= 25 L và duy trì bền vững trong 15 mẫu tiếp theo
            step = raw[i+8] - raw[i]
            # Mức duy trì: median 10 mẫu sau trừ đi mức trước
            pre_level = float(np.median(raw[max(0, i-5):i+1]))
            post_level = float(np.median(raw[i+8:i+18]))
            sustained_diff = post_level - pre_level
            
            if sustained_diff >= 25.0:
                clean_pre = float(clean_new[i])
                clean_post = float(clean_new[i+15])
                clean_jump = clean_post - clean_pre
                
                event_info = {
                    "vehicle_id": v_id,
                    "time": str(df_ai["FuelTime"].iloc[i]),
                    "pre_raw": round(pre_level, 1),
                    "post_raw": round(post_level, 1),
                    "jump": round(sustained_diff, 1),
                    "clean_pre": round(clean_pre, 1),
                    "clean_post": round(clean_post, 1),
                    "clean_jump": round(clean_jump, 1),
                }
                
                if clean_jump < 0.6 * sustained_diff:
                    missed_refuels.append(event_info)
                else:
                    confirmed_events.append(event_info)
                
                # Nhảy qua sự kiện này
                i += 25
            else:
                i += 1

    print(f"\nTổng số sự kiện tăng bền vững thực tế: {len(confirmed_events) + len(missed_refuels)}")
    print(f"Số sự kiện được bộ lọc mới xác nhận thành công: {len(confirmed_events)}")
    print(f"Số sự kiện bị kẹt thấp hoặc nghi ngờ: {len(missed_refuels)}")

    if missed_refuels:
        print("\nCÁC SỰ KIỆN NGHI NGỜ / BỊ KẸT THẤP:")
        for m in missed_refuels:
            print(f"  Xe {m['vehicle_id']} | Time: {m['time']} | Raw: {m['pre_raw']} -> {m['post_raw']} (+{m['jump']} L) | Clean: {m['clean_pre']} -> {m['clean_post']} (+{m['clean_jump']} L)")
    else:
        print("\n=> 100% CÁC SỰ KIỆN TĂNG BỀN VỮNG ĐỀU ĐƯỢC XÁC NHẬN THÀNH CÔNG, KHÔNG BỊ KẸT THẤP!")


if __name__ == "__main__":
    main()
